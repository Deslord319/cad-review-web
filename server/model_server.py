#!/usr/bin/env python3
"""Local model index and file server for CAD Review Web."""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from preview_store import (  # noqa: E402
    DEFAULT_FACE_BUDGET,
    DEFAULT_PIPELINE_VERSION,
    DEFAULT_PROFILE,
    PREVIEW_STATUSES,
    PreviewStore,
)


OUTPUT_DIR = Path(os.environ.get("CAD_OUTPUT_DIR", "./models")).resolve()
BIND_HOST = os.environ.get("CAD_VIEWER_API_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("CAD_VIEWER_API_PORT", "8091"))
ALLOWED_ORIGIN = os.environ.get("CAD_VIEWER_ALLOWED_ORIGIN", "http://localhost:5173")
ALLOWED_SUFFIXES = {".stl", ".step", ".stp", ".fcstd", ".png", ".3mf"}
STORAGE_DIRS = {
    "active": OUTPUT_DIR,
    "archive": OUTPUT_DIR / ".archive",
    "trash": OUTPUT_DIR / ".trash",
}
VALID_ACTIONS = {"archive", "trash", "restore"}
PREVIEW_CACHE_DIR = Path(
    os.environ.get(
        "CAD_VIEWER_PREVIEW_DIR",
        os.environ.get("CAD_VIEWER_PREVIEW_CACHE_DIR", OUTPUT_DIR / ".preview-cache"),
    )
).resolve()
PREVIEW_FACE_BUDGET = int(
    os.environ.get("CAD_VIEWER_PREVIEW_FACE_BUDGET", str(DEFAULT_FACE_BUDGET))
)
PREVIEW_PIPELINE_VERSION = os.environ.get(
    "CAD_VIEWER_PREVIEW_PIPELINE_VERSION", DEFAULT_PIPELINE_VERSION
)
PREVIEW_PROFILE = os.environ.get("CAD_VIEWER_PREVIEW_PROFILE", DEFAULT_PROFILE)
_preview_stores: dict[tuple[str, str, str, int], PreviewStore] = {}


def preview_store() -> PreviewStore:
    key = (
        str(PREVIEW_CACHE_DIR.resolve()),
        PREVIEW_PIPELINE_VERSION,
        PREVIEW_PROFILE,
        PREVIEW_FACE_BUDGET,
    )
    store = _preview_stores.get(key)
    if store is None:
        store = PreviewStore(
            PREVIEW_CACHE_DIR,
            pipeline_version=PREVIEW_PIPELINE_VERSION,
            profile=PREVIEW_PROFILE,
            face_budget=PREVIEW_FACE_BUDGET,
        )
        _preview_stores[key] = store
    return store


def _vector_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vector_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def inspect_binary_stl(path: Path) -> dict:
    size = path.stat().st_size
    if size < 84:
        return {}
    with path.open("rb") as handle:
        header = handle.read(84)
        facets = struct.unpack("<I", header[80:84])[0]
        if 84 + facets * 50 != size:
            return {}
        mins = [math.inf, math.inf, math.inf]
        maxs = [-math.inf, -math.inf, -math.inf]
        volume = 0.0
        edges: dict[tuple, int] = {}
        for _ in range(facets):
            record = handle.read(50)
            values = struct.unpack("<12fH", record)
            vertices = [values[3:6], values[6:9], values[9:12]]
            for vertex in vertices:
                for axis in range(3):
                    mins[axis] = min(mins[axis], vertex[axis])
                    maxs[axis] = max(maxs[axis], vertex[axis])
            volume += _vector_dot(vertices[0], _vector_cross(vertices[1], vertices[2])) / 6.0
            points = [tuple(round(value, 5) for value in vertex) for vertex in vertices]
            for start, end in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((points[start], points[end])))
                edges[edge] = edges.get(edge, 0) + 1
        return {
            "facets": facets,
            "dimensions": [round(maxs[i] - mins[i], 4) for i in range(3)],
            "volume": round(abs(volume), 3),
            "watertight": bool(edges) and all(count == 2 for count in edges.values()),
        }


def inspect_ascii_stl(path: Path) -> dict:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    volume = 0.0
    facets = 0
    edges: dict[tuple, int] = {}
    vertices = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertex = tuple(float(value) for value in fields[1:])
                vertices.append(vertex)
                for axis in range(3):
                    mins[axis] = min(mins[axis], vertex[axis])
                    maxs[axis] = max(maxs[axis], vertex[axis])
            elif fields and fields[0].lower() == "endfacet":
                if len(vertices) != 3:
                    raise ValueError("Malformed ASCII STL facet")
                volume += _vector_dot(vertices[0], _vector_cross(vertices[1], vertices[2])) / 6.0
                points = [tuple(round(value, 5) for value in vertex) for vertex in vertices]
                for start, end in ((0, 1), (1, 2), (2, 0)):
                    edge = tuple(sorted((points[start], points[end])))
                    edges[edge] = edges.get(edge, 0) + 1
                facets += 1
                vertices = []

    if facets == 0:
        return {}
    return {
        "facets": facets,
        "dimensions": [round(maxs[i] - mins[i], 4) for i in range(3)],
        "volume": round(abs(volume), 3),
        "watertight": bool(edges) and all(count == 2 for count in edges.values()),
    }


def inspect_stl(path: Path) -> dict:
    binary = inspect_binary_stl(path)
    return binary or inspect_ascii_stl(path)


def storage_dir(scope: str) -> Path:
    if scope not in STORAGE_DIRS:
        raise ValueError(f"Unknown storage scope: {scope}")
    path = STORAGE_DIRS[scope]
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_model_path(scope: str, name: str) -> Path:
    decoded = unquote(name)
    if not decoded or decoded != Path(decoded).name or decoded in {".", ".."}:
        raise ValueError("Invalid model name")
    path = storage_dir(scope) / decoded
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Unsupported model type")
    if path.is_symlink():
        raise ValueError("Symbolic links are not supported")
    return path


def supported_model_paths(scope: str) -> list[Path]:
    return [
        path
        for path in storage_dir(scope).iterdir()
        if not path.is_symlink()
        and path.is_file()
        and path.suffix.lower() in ALLOWED_SUFFIXES
    ]


def storage_snapshot() -> dict[str, list[Path]]:
    return {scope: supported_model_paths(scope) for scope in STORAGE_DIRS}


def model_listing(scope: str = "active", paths: list[Path] | None = None) -> list[dict]:
    entries = []
    for path in paths if paths is not None else supported_model_paths(scope):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        item = {
            "name": path.name,
            "extension": path.suffix.lower().lstrip("."),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "viewable": path.suffix.lower() in {".stl", ".3mf"},
            "scope": scope,
        }
        preview = {
            "preview_status": "not_applicable",
            "preview_revision": None,
            "preview_error": None,
            "preview_url": None,
        }
        if path.suffix.lower() == ".3mf":
            try:
                preview.update(
                    preview_store().status_for_source(
                        scope, path.name, path, enqueue_missing=False
                    )
                )
            except (OSError, ValueError) as error:
                preview.update(
                    {
                        "preview_status": "failed",
                        "preview_revision": None,
                        "preview_error": str(error),
                    }
                )
            preview["preview_url"] = (
                f"/models/{scope}/{quote(path.name, safe='')}?preview=1"
            )
            if preview.get("preview_revision"):
                preview["preview_url"] += (
                    f"&revision={preview['preview_revision']}"
                )
        item.update(preview)
        if path.suffix.lower() == ".stl":
            try:
                item.update(inspect_stl(path))
            except (OSError, ValueError, struct.error):
                pass
        entries.append(item)
    return sorted(entries, key=lambda item: item["modified"], reverse=True)


def preview_counts(models: list[dict]) -> dict[str, int]:
    counts = {status: 0 for status in PREVIEW_STATUSES}
    for model in models:
        status = str(model.get("preview_status", "not_applicable"))
        if status not in counts:
            status = "failed"
        counts[status] += 1
    return counts


def model_count(scope: str) -> int:
    return len(supported_model_paths(scope))


def move_model(action: str, source_scope: str, name: str) -> dict:
    if action not in VALID_ACTIONS:
        raise ValueError("Unsupported action")
    transitions = {
        "archive": ({"active"}, "archive"),
        "trash": ({"active", "archive"}, "trash"),
        "restore": ({"archive", "trash"}, "active"),
    }
    allowed_sources, destination_scope = transitions[action]
    if source_scope not in allowed_sources:
        raise ValueError(f"Cannot {action} a model from {source_scope}")

    source = safe_model_path(source_scope, name)
    destination = safe_model_path(destination_scope, name)
    if not source.is_file():
        raise FileNotFoundError(name)
    if destination.exists():
        raise FileExistsError(name)
    source.replace(destination)
    result = {
        "status": "success",
        "action": action,
        "name": name,
        "from": source_scope,
        "to": destination_scope,
    }
    if destination.suffix.lower() == ".3mf":
        try:
            store = preview_store()
            store.forget_source(source_scope, name)
            preview = store.ensure_source(destination_scope, name, destination)
        except (OSError, ValueError) as error:
            preview = {
                "preview_status": "failed",
                "preview_revision": None,
                "preview_error": str(error),
            }
        result.update(preview)
        result["preview_url"] = (
            f"/models/{destination_scope}/{quote(name, safe='')}?preview=1"
        )
        if preview.get("preview_revision"):
            result["preview_url"] += f"&revision={preview['preview_revision']}"
    return result


def preview_details(scope: str, name: str, path: Path) -> dict:
    details = preview_store().status_for_source(
        scope, name, path, enqueue_missing=False
    )
    details["preview_url"] = f"/models/{scope}/{quote(name, safe='')}?preview=1"
    if details.get("preview_revision"):
        details["preview_url"] += f"&revision={details['preview_revision']}"
    return details


class Handler(BaseHTTPRequestHandler):
    server_version = "CADReviewWeb/1.2"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def send_json(self, status: HTTPStatus, data: dict):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_file(self, path: Path, content_type: str, download_name: str | None = None, immutable: bool = False):
        stat = path.stat()
        etag = f'"{stat.st_size:x}-{stat.st_mtime_ns:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if immutable else "private, max-age=3600")
        if download_name:
            encoded_name = quote(download_name, safe="")
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{encoded_name}")
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 256):
                self.wfile.write(chunk)

    def read_json_body(self) -> dict:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid request length") from error
        if content_length <= 0 or content_length > 16_384:
            raise ValueError("Invalid request body")
        try:
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON") from error
        if not isinstance(body, dict):
            raise ValueError("JSON request body must be an object")
        return body

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/models":
            scope = parse_qs(parsed.query).get("scope", ["active"])[0]
            try:
                snapshot = storage_snapshot()
                if scope not in snapshot:
                    raise ValueError(f"Unknown storage scope: {scope}")
                models = model_listing(scope, snapshot[scope])
            except ValueError as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(error)})
                return
            counts = {key: len(paths) for key, paths in snapshot.items()}
            counts[scope] = len(models)
            self.send_json(
                HTTPStatus.OK,
                {
                    "models": models,
                    "scope": scope,
                    "counts": counts,
                    "preview_counts": preview_counts(models),
                },
            )
            return

        if parsed.path == "/api/previews/status":
            query = parse_qs(parsed.query)
            scope = query.get("scope", ["active"])[0]
            name = query.get("name", [""])[0]
            try:
                path = safe_model_path(scope, name)
                if path.suffix.lower() != ".3mf":
                    raise ValueError("Preview status is only available for 3MF files")
                if not path.is_file():
                    raise FileNotFoundError(name)
                details = preview_details(scope, name, path)
            except ValueError as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(error)})
                return
            except FileNotFoundError:
                self.send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "Model not found"})
                return
            self.send_json(HTTPStatus.OK, {"status": "success", **details})
            return

        if parsed.path.startswith("/models/"):
            query = parse_qs(parsed.query)
            parts = parsed.path.removeprefix("/models/").split("/", 1)
            if len(parts) == 2 and parts[0] in STORAGE_DIRS:
                scope, name = parts
            else:
                scope, name = "active", parsed.path.removeprefix("/models/")
            try:
                path = safe_model_path(scope, name)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if path.suffix.lower() == ".3mf" and query.get("preview") == ["1"]:
                try:
                    details = preview_details(scope, path.name, path)
                except (OSError, ValueError) as error:
                    self.send_json(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        {
                            "status": "error",
                            "preview_status": "failed",
                            "preview_error": str(error),
                            "message": "3MF preview indexing failed",
                        },
                    )
                    return
                preview_status = details["preview_status"]
                if preview_status == "ready":
                    revision = str(details["preview_revision"])
                    self.send_file(
                        preview_store().artifact_path(revision),
                        "model/gltf-binary",
                        immutable=True,
                    )
                    return
                status_code = (
                    HTTPStatus.UNPROCESSABLE_ENTITY
                    if preview_status == "failed"
                    else HTTPStatus.ACCEPTED
                )
                self.send_json(
                    status_code,
                    {
                        "status": "error" if preview_status == "failed" else "pending",
                        **details,
                        "message": (
                            "3MF preview generation failed"
                            if preview_status == "failed"
                            else "3MF preview is queued"
                        ),
                    },
                )
                return
            content_types = {
                ".stl": "model/stl",
                ".step": "application/step",
                ".stp": "application/step",
                ".fcstd": "application/zip",
                ".png": "image/png",
                ".3mf": "model/3mf",
            }
            download_name = path.name if query.get("download") == ["1"] else None
            self.send_file(path, content_types.get(path.suffix.lower(), "application/octet-stream"), download_name)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in {
            "/api/models/action",
            "/api/previews/enqueue",
            "/api/previews/retry",
        }:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            body = self.read_json_body()
            if parsed.path == "/api/models/action":
                result = move_model(
                    str(body.get("action", "")),
                    str(body.get("scope", "")),
                    str(body.get("name", "")),
                )
                response_status = HTTPStatus.OK
            else:
                scope = str(body.get("scope", ""))
                name = str(body.get("name", ""))
                path = safe_model_path(scope, name)
                if scope not in {"active", "archive"}:
                    raise ValueError("Preview jobs may only target active or archive models")
                if path.suffix.lower() != ".3mf":
                    raise ValueError("Preview jobs are only available for 3MF files")
                if not path.is_file():
                    raise FileNotFoundError(name)
                if parsed.path == "/api/previews/retry":
                    details = preview_store().retry_source(scope, name, path)
                else:
                    expected_sha256 = body.get("sha256")
                    if expected_sha256 is not None and not isinstance(expected_sha256, str):
                        raise ValueError("sha256 must be a string")
                    details = preview_store().ensure_source(
                        scope, name, path, expected_sha256=expected_sha256
                    )
                details["preview_url"] = (
                    f"/models/{scope}/{quote(name, safe='')}?preview=1"
                )
                if details.get("preview_revision"):
                    details["preview_url"] += f"&revision={details['preview_revision']}"
                result = {"status": "success", "scope": scope, "name": name, **details}
                if details["preview_status"] == "ready":
                    response_status = HTTPStatus.OK
                elif details["preview_status"] == "failed":
                    response_status = HTTPStatus.UNPROCESSABLE_ENTITY
                else:
                    response_status = HTTPStatus.ACCEPTED
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(error)})
            return
        except FileNotFoundError:
            self.send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "Model not found"})
            return
        except FileExistsError:
            self.send_json(HTTPStatus.CONFLICT, {"status": "error", "message": "A file with this name already exists in the destination"})
            return

        self.send_json(response_status, result)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    print(f"Serving CAD output from {OUTPUT_DIR} at http://{BIND_HOST}:{BIND_PORT}")
    ThreadingHTTPServer((BIND_HOST, BIND_PORT), Handler).serve_forever()
