#!/usr/bin/env python3
"""Persistent, content-addressed preview queue for CAD Review.

The API process and the preview worker deliberately share only this SQLite/WAL
database and the preview artifact directory.  Model conversion never runs in
the API process.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Mapping


PREVIEW_STATUSES = ("not_applicable", "pending", "processing", "ready", "failed")
JOB_STATUSES = ("pending", "processing", "ready", "failed")
DEFAULT_PIPELINE_VERSION = "three-mf-glb-v3"
DEFAULT_PROFILE = "fast"
DEFAULT_FACE_BUDGET = 100_000


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def preview_revision(
    content_sha256: str,
    pipeline_version: str,
    profile: str,
    face_budget: int,
) -> str:
    identity = "\0".join(
        (content_sha256.lower(), pipeline_version, profile, str(face_budget))
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def storage_dirs_from_output(output_dir: Path) -> dict[str, Path]:
    root = output_dir.resolve()
    return {
        "active": root,
        "archive": root / ".archive",
        "trash": root / ".trash",
    }


class PreviewStore:
    """SQLite-backed source index, conversion queue, and artifact registry."""

    def __init__(
        self,
        preview_dir: Path,
        *,
        pipeline_version: str = DEFAULT_PIPELINE_VERSION,
        profile: str = DEFAULT_PROFILE,
        face_budget: int = DEFAULT_FACE_BUDGET,
    ):
        if face_budget <= 0:
            raise ValueError("face_budget must be greater than zero")
        self.preview_dir = preview_dir.resolve()
        self.objects_dir = self.preview_dir / "objects"
        self.work_dir = self.preview_dir / "work"
        self.database_path = self.preview_dir / "previews.sqlite3"
        self.pipeline_version = pipeline_version
        self.profile = profile
        self.face_budget = face_budget
        self._initialization_lock = threading.Lock()
        self._initialized = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        if self._initialized:
            return
        with self._initialization_lock:
            if self._initialized:
                return
            self.objects_dir.mkdir(parents=True, exist_ok=True)
            self.work_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS preview_jobs (
                        revision TEXT PRIMARY KEY,
                        content_sha256 TEXT NOT NULL,
                        pipeline_version TEXT NOT NULL,
                        profile TEXT NOT NULL,
                        face_budget INTEGER NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'ready', 'failed')),
                        artifact_relpath TEXT,
                        priority_bytes INTEGER NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        started_at REAL,
                        finished_at REAL,
                        next_attempt_at REAL NOT NULL DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS preview_sources (
                        scope TEXT NOT NULL,
                        name TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        mtime_ns INTEGER NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        revision TEXT NOT NULL,
                        seen_at REAL NOT NULL,
                        PRIMARY KEY(scope, name),
                        FOREIGN KEY(revision) REFERENCES preview_jobs(revision)
                    );

                    CREATE INDEX IF NOT EXISTS preview_jobs_queue
                    ON preview_jobs(status, next_attempt_at, priority_bytes, created_at);

                    CREATE INDEX IF NOT EXISTS preview_sources_revision
                    ON preview_sources(revision);
                    """
                )
            self._initialized = True

    def artifact_path(self, revision: str) -> Path:
        return self.objects_dir / f"{revision}.glb"

    def _job_status(self, connection: sqlite3.Connection, revision: str) -> dict:
        row = connection.execute(
            "SELECT * FROM preview_jobs WHERE revision = ?", (revision,)
        ).fetchone()
        if row is None:
            raise KeyError(revision)
        status = str(row["status"])
        artifact = self.artifact_path(revision)
        if status == "ready" and (not artifact.is_file() or artifact.stat().st_size <= 0):
            now = time.time()
            connection.execute(
                """
                UPDATE preview_jobs
                SET status = 'pending', artifact_relpath = NULL,
                    error = 'Preview artifact is missing; regeneration queued',
                    updated_at = ?, next_attempt_at = ?
                WHERE revision = ?
                """,
                (now, now, revision),
            )
            status = "pending"
            row = connection.execute(
                "SELECT * FROM preview_jobs WHERE revision = ?", (revision,)
            ).fetchone()
            assert row is not None
        return {
            "preview_status": status,
            "preview_revision": revision,
            "preview_error": row["error"],
        }

    def ensure_source(
        self,
        scope: str,
        name: str,
        path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> dict:
        """Index a 3MF source and enqueue its content-addressed preview if new."""

        if path.suffix.lower() != ".3mf":
            return {
                "preview_status": "not_applicable",
                "preview_revision": None,
                "preview_error": None,
            }
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(name)
        if expected_sha256 is not None:
            expected_sha256 = expected_sha256.lower()
            if len(expected_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in expected_sha256
            ):
                raise ValueError("sha256 must be a 64-character hexadecimal digest")

        stat = path.stat()
        digest: str | None = None
        with self._connect() as connection:
            source = connection.execute(
                "SELECT * FROM preview_sources WHERE scope = ? AND name = ?",
                (scope, name),
            ).fetchone()
            if (
                source is not None
                and int(source["size"]) == stat.st_size
                and int(source["mtime_ns"]) == stat.st_mtime_ns
            ):
                digest = str(source["content_sha256"])
                if expected_sha256 is not None and digest != expected_sha256:
                    raise ValueError(
                        f"SHA-256 mismatch: expected {expected_sha256}, received {digest}"
                    )
                expected_revision = preview_revision(
                    digest,
                    self.pipeline_version,
                    self.profile,
                    self.face_budget,
                )
                if str(source["revision"]) == expected_revision:
                    connection.execute(
                        "UPDATE preview_sources SET seen_at = ? WHERE scope = ? AND name = ?",
                        (time.time(), scope, name),
                    )
                    return self._job_status(connection, expected_revision)

        if digest is None:
            digest = file_sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch: expected {expected_sha256}, received {digest}"
            )
        revision = preview_revision(
            digest, self.pipeline_version, self.profile, self.face_budget
        )
        artifact = self.artifact_path(revision)
        artifact_ready = artifact.is_file() and artifact.stat().st_size > 0
        now = time.time()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO preview_jobs (
                    revision, content_sha256, pipeline_version, profile,
                    face_budget, status, artifact_relpath, priority_bytes,
                    attempts, error, created_at, updated_at, finished_at,
                    next_attempt_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?)
                ON CONFLICT(revision) DO UPDATE SET
                    priority_bytes = MIN(preview_jobs.priority_bytes, excluded.priority_bytes),
                    updated_at = excluded.updated_at
                """,
                (
                    revision,
                    digest,
                    self.pipeline_version,
                    self.profile,
                    self.face_budget,
                    "ready" if artifact_ready else "pending",
                    f"objects/{revision}.glb" if artifact_ready else None,
                    stat.st_size,
                    now,
                    now,
                    now if artifact_ready else None,
                    now,
                ),
            )
            if artifact_ready:
                connection.execute(
                    """
                    UPDATE preview_jobs
                    SET status = 'ready', artifact_relpath = ?, error = NULL,
                        finished_at = ?, updated_at = ?
                    WHERE revision = ?
                    """,
                    (f"objects/{revision}.glb", now, now, revision),
                )
            connection.execute(
                """
                INSERT INTO preview_sources (
                    scope, name, size, mtime_ns, content_sha256, revision, seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, name) DO UPDATE SET
                    size = excluded.size,
                    mtime_ns = excluded.mtime_ns,
                    content_sha256 = excluded.content_sha256,
                    revision = excluded.revision,
                    seen_at = excluded.seen_at
                """,
                (scope, name, stat.st_size, stat.st_mtime_ns, digest, revision, now),
            )
            return self._job_status(connection, revision)

    def status_for_source(
        self,
        scope: str,
        name: str,
        path: Path | None = None,
        *,
        enqueue_missing: bool = True,
    ) -> dict:
        if path is not None and path.suffix.lower() != ".3mf":
            return {
                "preview_status": "not_applicable",
                "preview_revision": None,
                "preview_error": None,
            }
        with self._connect() as connection:
            source = connection.execute(
                "SELECT * FROM preview_sources WHERE scope = ? AND name = ?",
                (scope, name),
            ).fetchone()
            if source is not None:
                if path is not None and path.is_file() and not path.is_symlink():
                    stat = path.stat()
                    identity = connection.execute(
                        """
                        SELECT size, mtime_ns FROM preview_sources
                        WHERE scope = ? AND name = ?
                        """,
                        (scope, name),
                    ).fetchone()
                    assert identity is not None
                    if (
                        int(identity["size"]) != stat.st_size
                        or int(identity["mtime_ns"]) != stat.st_mtime_ns
                    ):
                        source = None
                if source is not None:
                    current_revision = preview_revision(
                        str(source["content_sha256"]),
                        self.pipeline_version,
                        self.profile,
                        self.face_budget,
                    )
                    if str(source["revision"]) != current_revision:
                        if path is not None and enqueue_missing:
                            source = None
                        else:
                            return {
                                "preview_status": "pending",
                                "preview_revision": current_revision,
                                "preview_error": None,
                            }
                if source is not None:
                    return self._job_status(connection, str(source["revision"]))
        if path is not None and enqueue_missing:
            return self.ensure_source(scope, name, path)
        return {
            "preview_status": "pending",
            "preview_revision": None,
            "preview_error": None,
        }

    def retry_source(self, scope: str, name: str, path: Path) -> dict:
        status = self.ensure_source(scope, name, path)
        revision = status.get("preview_revision")
        if not revision or status["preview_status"] in {"ready", "processing"}:
            return status
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE preview_jobs
                SET status = 'pending', attempts = 0, error = NULL,
                    started_at = NULL, finished_at = NULL,
                    next_attempt_at = ?, updated_at = ?
                WHERE revision = ?
                """,
                (now, now, revision),
            )
            return self._job_status(connection, revision)

    def forget_source(self, scope: str, name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM preview_sources WHERE scope = ? AND name = ?",
                (scope, name),
            )

    def remove_missing_sources(self, scope: str, names: set[str]) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name FROM preview_sources WHERE scope = ?", (scope,)
            ).fetchall()
            missing = [str(row["name"]) for row in rows if str(row["name"]) not in names]
            connection.executemany(
                "DELETE FROM preview_sources WHERE scope = ? AND name = ?",
                ((scope, name) for name in missing),
            )
            return len(missing)

    def reconcile(
        self,
        storage_dirs: Mapping[str, Path],
        *,
        scopes: Iterable[str] = ("active", "archive"),
    ) -> dict:
        """Discover externally written 3MFs and enqueue them smallest-first."""

        indexed = 0
        errors: list[dict[str, str]] = []
        for scope in scopes:
            directory = storage_dirs[scope]
            directory.mkdir(parents=True, exist_ok=True)
            candidates = sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file()
                    and not path.is_symlink()
                    and path.suffix.lower() == ".3mf"
                ),
                key=lambda path: (path.stat().st_size, path.name),
            )
            names = {path.name for path in candidates}
            self.remove_missing_sources(scope, names)
            for path in candidates:
                try:
                    self.ensure_source(scope, path.name, path)
                    indexed += 1
                except (OSError, ValueError) as error:
                    errors.append({"scope": scope, "name": path.name, "error": str(error)})
        return {"indexed": indexed, "errors": errors}

    def recover_stale_processing(self, stale_after_seconds: float) -> int:
        cutoff = time.time() - stale_after_seconds
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE preview_jobs
                SET status = 'pending',
                    error = 'Worker stopped before conversion completed; job requeued',
                    started_at = NULL, next_attempt_at = ?, updated_at = ?
                WHERE status = 'processing' AND started_at < ?
                """,
                (now, now, cutoff),
            )
            return cursor.rowcount

    def claim_next(self, storage_dirs: Mapping[str, Path]) -> dict | None:
        """Atomically claim the smallest pending job with an available source."""

        while True:
            now = time.time()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                job = connection.execute(
                    """
                    SELECT * FROM preview_jobs
                    WHERE status = 'pending' AND next_attempt_at <= ?
                      AND EXISTS (
                          SELECT 1 FROM preview_sources
                          WHERE preview_sources.revision = preview_jobs.revision
                      )
                    ORDER BY priority_bytes ASC, created_at ASC
                    LIMIT 1
                    """,
                    (now,),
                ).fetchone()
                if job is None:
                    return None
                revision = str(job["revision"])
                artifact = self.artifact_path(revision)
                if artifact.is_file() and artifact.stat().st_size > 0:
                    connection.execute(
                        """
                        UPDATE preview_jobs
                        SET status = 'ready', artifact_relpath = ?, error = NULL,
                            finished_at = ?, updated_at = ?
                        WHERE revision = ?
                        """,
                        (f"objects/{revision}.glb", now, now, revision),
                    )
                    continue

                sources = connection.execute(
                    """
                    SELECT * FROM preview_sources
                    WHERE revision = ? ORDER BY size ASC, scope ASC, name ASC
                    """,
                    (revision,),
                ).fetchall()
                chosen = None
                for source in sources:
                    directory = storage_dirs.get(str(source["scope"]))
                    if directory is None:
                        continue
                    candidate = directory / str(source["name"])
                    if candidate.is_file() and not candidate.is_symlink():
                        chosen = (source, candidate)
                        break
                    connection.execute(
                        "DELETE FROM preview_sources WHERE scope = ? AND name = ?",
                        (source["scope"], source["name"]),
                    )
                if chosen is None:
                    connection.execute(
                        """
                        UPDATE preview_jobs
                        SET status = 'failed', error = 'No source file is available',
                            finished_at = ?, updated_at = ?
                        WHERE revision = ?
                        """,
                        (now, now, revision),
                    )
                    continue

                source, source_path = chosen
                connection.execute(
                    """
                    UPDATE preview_jobs
                    SET status = 'processing', attempts = attempts + 1,
                        error = NULL, started_at = ?, updated_at = ?
                    WHERE revision = ?
                    """,
                    (now, now, revision),
                )
                return {
                    "revision": revision,
                    "content_sha256": str(job["content_sha256"]),
                    "pipeline_version": str(job["pipeline_version"]),
                    "profile": str(job["profile"]),
                    "face_budget": int(job["face_budget"]),
                    "attempts": int(job["attempts"]) + 1,
                    "scope": str(source["scope"]),
                    "name": str(source["name"]),
                    "source_path": source_path,
                    "artifact_path": artifact,
                }

    def finish_success(self, revision: str, artifact_path: Path) -> dict:
        if artifact_path != self.artifact_path(revision):
            raise ValueError("Artifact path does not match preview revision")
        if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
            raise ValueError("Preview converter did not create a valid artifact")
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE preview_jobs
                SET status = 'ready', artifact_relpath = ?, error = NULL,
                    finished_at = ?, updated_at = ?
                WHERE revision = ?
                """,
                (f"objects/{revision}.glb", now, now, revision),
            )
            return self._job_status(connection, revision)

    def finish_failure(
        self,
        revision: str,
        error: str,
        *,
        max_attempts: int,
        retry_base_seconds: float = 5.0,
    ) -> dict:
        now = time.time()
        message = error.strip()[:2000] or "Preview conversion failed"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM preview_jobs WHERE revision = ?", (revision,)
            ).fetchone()
            if row is None:
                raise KeyError(revision)
            attempts = int(row["attempts"])
            terminal = attempts >= max_attempts
            next_attempt = now if terminal else now + retry_base_seconds * (2 ** max(0, attempts - 1))
            connection.execute(
                """
                UPDATE preview_jobs
                SET status = ?, error = ?, finished_at = ?,
                    next_attempt_at = ?, updated_at = ?
                WHERE revision = ?
                """,
                (
                    "failed" if terminal else "pending",
                    message,
                    now if terminal else None,
                    next_attempt,
                    now,
                    revision,
                ),
            )
            return self._job_status(connection, revision)

    def queue_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in JOB_STATUSES}
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM preview_jobs GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["count"])
        return counts

    def journal_mode(self) -> str:
        with self._connect() as connection:
            row = connection.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()
