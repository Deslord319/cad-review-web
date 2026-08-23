#!/usr/bin/env python3
"""Serial preview queue worker.

Run directly with ``python server/preview_worker.py``.  Configuration is read
from CAD_OUTPUT_DIR and CAD_VIEWER_* environment variables so the same command
works in a user systemd unit and at the terminal.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from preview_store import (
    DEFAULT_FACE_BUDGET,
    DEFAULT_PIPELINE_VERSION,
    DEFAULT_PROFILE,
    PreviewStore,
    file_sha256,
    storage_dirs_from_output,
)


SERVER_DIR = Path(__file__).resolve().parent
CONVERTER = SERVER_DIR / "preview_convert.py"


def environment_configuration() -> dict:
    output_dir = Path(os.environ.get("CAD_OUTPUT_DIR", "./models")).resolve()
    preview_dir = Path(
        os.environ.get(
            "CAD_VIEWER_PREVIEW_DIR",
            os.environ.get("CAD_VIEWER_PREVIEW_CACHE_DIR", str(output_dir / ".preview-cache")),
        )
    ).resolve()
    return {
        "output_dir": output_dir,
        "preview_dir": preview_dir,
        "pipeline_version": os.environ.get(
            "CAD_VIEWER_PREVIEW_PIPELINE_VERSION", DEFAULT_PIPELINE_VERSION
        ),
        "profile": os.environ.get("CAD_VIEWER_PREVIEW_PROFILE", DEFAULT_PROFILE),
        "face_budget": int(
            os.environ.get("CAD_VIEWER_PREVIEW_FACE_BUDGET", str(DEFAULT_FACE_BUDGET))
        ),
        "timeout": float(os.environ.get("CAD_VIEWER_PREVIEW_TIMEOUT", "300")),
        "max_attempts": int(os.environ.get("CAD_VIEWER_PREVIEW_MAX_ATTEMPTS", "3")),
        "scan_interval": float(os.environ.get("CAD_VIEWER_PREVIEW_SCAN_INTERVAL", "30")),
        "idle_interval": float(os.environ.get("CAD_VIEWER_PREVIEW_IDLE_INTERVAL", "2")),
    }


def run_conversion(job: dict, timeout: float) -> tuple[bool, str]:
    source_path = Path(job["source_path"])
    actual_sha256 = file_sha256(source_path)
    if actual_sha256 != job["content_sha256"]:
        return False, "Source changed after it was queued; waiting for reconciliation"

    command = [
        sys.executable,
        str(CONVERTER),
        "--source",
        str(source_path),
        "--output",
        str(job["artifact_path"]),
        "--face-budget",
        str(job["face_budget"]),
        "--profile",
        str(job["profile"]),
    ]
    environment = os.environ.copy()
    numeric_threads = os.environ.get("CAD_VIEWER_PREVIEW_NUMERIC_THREADS", "1")
    environment["OPENBLAS_NUM_THREADS"] = numeric_threads
    environment["OMP_NUM_THREADS"] = numeric_threads
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return False, f"Preview conversion exceeded the {timeout:g}-second timeout"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return False, detail or f"Preview converter exited with code {completed.returncode}"
    return True, completed.stdout.strip()


def process_one(
    store: PreviewStore,
    storage_dirs: dict[str, Path],
    *,
    timeout: float,
    max_attempts: int,
) -> bool:
    job = store.claim_next(storage_dirs)
    if job is None:
        return False
    print(
        f"Converting {job['scope']}/{job['name']} "
        f"({job['revision'][:12]}, attempt {job['attempts']})",
        flush=True,
    )
    try:
        success, detail = run_conversion(job, timeout)
    except (OSError, ValueError) as error:
        success, detail = False, str(error)
    if success:
        store.finish_success(job["revision"], Path(job["artifact_path"]))
        print(f"Preview ready: {job['revision'][:12]} {detail}", flush=True)
    else:
        status = store.finish_failure(
            job["revision"], detail, max_attempts=max_attempts
        )
        print(
            f"Preview {status['preview_status']}: {job['revision'][:12]} {detail}",
            flush=True,
        )
    return True


def parse_args(configuration: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process CAD Review preview jobs serially")
    parser.add_argument("--output-dir", type=Path, default=configuration["output_dir"])
    parser.add_argument("--preview-dir", type=Path, default=configuration["preview_dir"])
    parser.add_argument("--pipeline-version", default=configuration["pipeline_version"])
    parser.add_argument("--profile", default=configuration["profile"])
    parser.add_argument("--face-budget", type=int, default=configuration["face_budget"])
    parser.add_argument("--timeout", type=float, default=configuration["timeout"])
    parser.add_argument("--max-attempts", type=int, default=configuration["max_attempts"])
    parser.add_argument("--scan-interval", type=float, default=configuration["scan_interval"])
    parser.add_argument("--idle-interval", type=float, default=configuration["idle_interval"])
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    parser.add_argument(
        "--reconcile-only", action="store_true", help="Index external files and exit"
    )
    return parser.parse_args()


def main() -> int:
    configuration = environment_configuration()
    arguments = parse_args(configuration)
    if arguments.timeout <= 0 or arguments.max_attempts <= 0:
        raise SystemExit("timeout and max-attempts must be greater than zero")
    if arguments.scan_interval <= 0 or arguments.idle_interval <= 0:
        raise SystemExit("scan-interval and idle-interval must be greater than zero")

    output_dir = arguments.output_dir.resolve()
    preview_dir = arguments.preview_dir.resolve()
    store = PreviewStore(
        preview_dir,
        pipeline_version=arguments.pipeline_version,
        profile=arguments.profile,
        face_budget=arguments.face_budget,
    )
    storage_dirs = storage_dirs_from_output(output_dir)

    lock_path = preview_dir / "worker.lock"
    lock_handle = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another preview worker owns {lock_path}", file=sys.stderr)
        return 2

    # The non-blocking flock proves no previous worker is alive, so every
    # processing lease left in SQLite is stale and can be resumed immediately.
    store.recover_stale_processing(0)
    result = store.reconcile(storage_dirs)
    print(
        f"Reconciled {result['indexed']} 3MF source(s); {len(result['errors'])} error(s)",
        flush=True,
    )
    if arguments.reconcile_only:
        return 0
    if arguments.once:
        process_one(
            store,
            storage_dirs,
            timeout=arguments.timeout,
            max_attempts=arguments.max_attempts,
        )
        return 0

    next_scan = time.monotonic() + arguments.scan_interval
    while True:
        handled = process_one(
            store,
            storage_dirs,
            timeout=arguments.timeout,
            max_attempts=arguments.max_attempts,
        )
        now = time.monotonic()
        if now >= next_scan:
            result = store.reconcile(storage_dirs)
            if result["errors"]:
                print(f"Reconcile errors: {result['errors']}", flush=True)
            store.recover_stale_processing(max(arguments.timeout * 2, 60))
            next_scan = now + arguments.scan_interval
        if not handled:
            time.sleep(arguments.idle_interval)


if __name__ == "__main__":
    raise SystemExit(main())
