#!/usr/bin/env python3
"""Local model index and file server for CAD Review Web."""

from __future__ import annotations

import json
import math
import os
import struct
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


OUTPUT_DIR = Path(os.environ.get("CAD_OUTPUT_DIR", "./models")).resolve()
BIND_HOST = os.environ.get("CAD_VIEWER_API_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("CAD_VIEWER_API_PORT", "8091"))
ALLOWED_ORIGIN = os.environ.get("CAD_VIEWER_ALLOWED_ORIGIN", "http://localhost:5173")
ALLOWED_SUFFIXES = {".stl", ".step", ".stp", ".fcstd", ".png"}
STORAGE_DIRS = {
    "active": OUTPUT_DIR,
    "archive": OUTPUT_DIR / ".archive",
    "trash": OUTPUT_DIR / ".trash",
}
VALID_ACTIONS = {"archive", "trash", "restore"}


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


def model_listing(scope: str = "active") -> list[dict]:
    entries = []
    directory = storage_dir(scope)
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        stat = path.stat()
        item = {
            "name": path.name,
            "extension": path.suffix.lower().lstrip("."),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "viewable": path.suffix.lower() == ".stl",
            "scope": scope,
        }
        if item["viewable"]:
            try:
                item.update(inspect_stl(path))
            except (OSError, ValueError, struct.error):
                pass
        entries.append(item)
    return sorted(entries, key=lambda item: item["modified"], reverse=True)


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
    return {
        "status": "success",
        "action": action,
        "name": name,
        "from": source_scope,
        "to": destination_scope,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "CADReviewWeb/1.1"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: HTTPStatus, data: dict):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/models":
            scope = parse_qs(parsed.query).get("scope", ["active"])[0]
            try:
                models = model_listing(scope)
            except ValueError as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(error)})
                return
            counts = {key: len(model_listing(key)) for key in STORAGE_DIRS}
            self.send_json(HTTPStatus.OK, {"models": models, "scope": scope, "counts": counts})
            return

        if parsed.path.startswith("/models/"):
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
            content_types = {
                ".stl": "model/stl",
                ".step": "application/step",
                ".stp": "application/step",
                ".fcstd": "application/zip",
                ".png": "image/png",
            }
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_types.get(path.suffix.lower(), "application/octet-stream"))
            self.send_header("Content-Length", str(path.stat().st_size))
            if parsed.query == "download=1":
                encoded_name = quote(path.name, safe="")
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{encoded_name}")
            self.end_headers()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 256):
                    self.wfile.write(chunk)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/models/action":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "Invalid request length"})
            return
        if content_length <= 0 or content_length > 16_384:
            self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "Invalid request body"})
            return

        try:
            body = json.loads(self.rfile.read(content_length))
            result = move_model(
                str(body.get("action", "")),
                str(body.get("scope", "")),
                str(body.get("name", "")),
            )
        except json.JSONDecodeError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "Invalid JSON"})
            return
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(error)})
            return
        except FileNotFoundError:
            self.send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "Model not found"})
            return
        except FileExistsError:
            self.send_json(HTTPStatus.CONFLICT, {"status": "error", "message": "A file with this name already exists in the destination"})
            return

        self.send_json(HTTPStatus.OK, result)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    print(f"Serving CAD output from {OUTPUT_DIR} at http://{BIND_HOST}:{BIND_PORT}")
    ThreadingHTTPServer((BIND_HOST, BIND_PORT), Handler).serve_forever()
