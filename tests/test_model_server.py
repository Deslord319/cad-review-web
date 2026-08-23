import importlib.util
import json
import struct
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer


SERVER_PATH = Path(__file__).parents[1] / "server" / "model_server.py"
SPEC = importlib.util.spec_from_file_location("model_server", SERVER_PATH)
model_server = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(model_server)


def write_binary_stl(path: Path):
    header = bytearray(80)
    facet = struct.pack(
        "<12fH",
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0,
    )
    path.write_bytes(header + struct.pack("<I", 1) + facet)


def write_ascii_stl(path: Path):
    path.write_text(
        """solid sample
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 2 0 0
    vertex 0 3 0
  endloop
endfacet
endsolid sample
""",
        encoding="utf-8",
    )


class ModelStorageTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        model_server.OUTPUT_DIR = root
        model_server.STORAGE_DIRS = {
            "active": root,
            "archive": root / ".archive",
            "trash": root / ".trash",
        }
        model_server.PREVIEW_CACHE_DIR = root / ".preview-cache"
        model_server.PREVIEW_FACE_BUDGET = 100_000
        model_server._preview_stores.clear()
        write_binary_stl(root / "sample.stl")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_archive_restore_and_trash_are_recoverable_moves(self):
        active = model_server.model_listing("active")
        self.assertEqual([item["name"] for item in active], ["sample.stl"])
        self.assertEqual(active[0]["scope"], "active")

        archived = model_server.move_model("archive", "active", "sample.stl")
        self.assertEqual(archived["to"], "archive")
        self.assertEqual(model_server.model_listing("active"), [])
        self.assertEqual(model_server.model_listing("archive")[0]["name"], "sample.stl")

        restored = model_server.move_model("restore", "archive", "sample.stl")
        self.assertEqual(restored["to"], "active")
        trashed = model_server.move_model("trash", "active", "sample.stl")
        self.assertEqual(trashed["to"], "trash")
        self.assertEqual(model_server.model_listing("trash")[0]["name"], "sample.stl")

        model_server.move_model("restore", "trash", "sample.stl")
        self.assertTrue((Path(self.tempdir.name) / "sample.stl").is_file())

    def test_rejects_traversal_unsupported_transitions_and_overwrites(self):
        with self.assertRaises(ValueError):
            model_server.safe_model_path("active", "../sample.stl")
        with self.assertRaises(ValueError):
            model_server.safe_model_path("active", "sample.exe")
        with self.assertRaises(ValueError):
            model_server.move_model("restore", "active", "sample.stl")

        archive_dir = model_server.storage_dir("archive")
        write_binary_stl(archive_dir / "sample.stl")
        with self.assertRaises(FileExistsError):
            model_server.move_model("archive", "active", "sample.stl")

    def test_rejects_symbolic_links(self):
        root = Path(self.tempdir.name)
        with tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir) / "outside.stl"
            write_binary_stl(outside)
            link = root / "linked.stl"
            link.symlink_to(outside)
            with self.assertRaises(ValueError):
                model_server.safe_model_path("active", "linked.stl")

    def test_ascii_stl_remains_auditable_after_archiving(self):
        ascii_path = Path(self.tempdir.name) / "ascii.stl"
        write_ascii_stl(ascii_path)
        model_server.move_model("archive", "active", "ascii.stl")
        archived = {item["name"]: item for item in model_server.model_listing("archive")}
        self.assertEqual(archived["ascii.stl"]["facets"], 1)
        self.assertEqual(archived["ascii.stl"]["dimensions"], [2.0, 3.0, 0.0])

    def test_3mf_is_listed_as_previewable(self):
        path = Path(self.tempdir.name) / "sample.3mf"
        path.write_bytes(b"PK\x03\x04")
        item = {entry["name"]: entry for entry in model_server.model_listing("active")}["sample.3mf"]
        self.assertTrue(item["viewable"])
        self.assertNotIn("facets", item)
        self.assertEqual(item["preview_status"], "pending")
        self.assertIsNone(item["preview_revision"])
        self.assertIn("preview=1", item["preview_url"])

    def test_listing_never_hashes_or_enqueues_unknown_3mf(self):
        path = Path(self.tempdir.name) / "unknown.3mf"
        path.write_bytes(b"PK\x03\x04unknown")
        fake_store = mock.Mock()
        fake_store.status_for_source.return_value = {
            "preview_status": "pending",
            "preview_revision": None,
            "preview_error": None,
        }
        with mock.patch.object(model_server, "preview_store", return_value=fake_store):
            model_server.model_listing("active")
        fake_store.status_for_source.assert_called_once_with(
            "active", "unknown.3mf", path, enqueue_missing=False
        )
        fake_store.ensure_source.assert_not_called()

    def test_count_matches_unique_supported_files(self):
        root = Path(self.tempdir.name)
        (root / "sample.3mf").write_bytes(b"PK\x03\x04")
        (root / "ignored.txt").write_text("ignored", encoding="utf-8")
        (root / ".preview-cache").mkdir()
        self.assertEqual(model_server.model_count("active"), 2)

    def test_archiving_3mf_enqueues_preview_and_preserves_source_count(self):
        root = Path(self.tempdir.name)
        (root / "queued.3mf").write_bytes(b"PK\x03\x04preview-source")
        result = model_server.move_model("archive", "active", "queued.3mf")
        self.assertEqual(result["preview_status"], "pending")
        self.assertEqual(len(result["preview_revision"]), 64)

        models = model_server.model_listing("archive")
        self.assertEqual([item["name"] for item in models], ["queued.3mf"])
        self.assertEqual(model_server.preview_counts(models)["pending"], 1)
        self.assertEqual(model_server.model_count("archive"), 1)

    def test_preview_counts_count_sources_not_deduplicated_jobs(self):
        root = Path(self.tempdir.name)
        content = b"PK\x03\x04shared-content"
        first = root / "first.3mf"
        second = root / "second.3mf"
        first.write_bytes(content)
        second.write_bytes(content)
        store = model_server.preview_store()
        store.ensure_source("active", first.name, first)
        store.ensure_source("active", second.name, second)

        models = model_server.model_listing("active")
        self.assertEqual(model_server.preview_counts(models)["pending"], 2)
        self.assertEqual(store.queue_counts()["pending"], 1)

    def test_preview_get_is_non_blocking_and_only_serves_ready_artifact(self):
        root = Path(self.tempdir.name)
        source = root / "queued.3mf"
        source.write_bytes(b"PK\x03\x04preview-source")

        server = ThreadingHTTPServer(("127.0.0.1", 0), model_server.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(f"{base}/models/active/queued.3mf?preview=1") as response:
                self.assertEqual(response.status, 202)
                pending = json.load(response)
            self.assertEqual(pending["preview_status"], "pending")
            self.assertFalse(any(model_server.PREVIEW_CACHE_DIR.glob("objects/*.glb")))

            store = model_server.preview_store()
            store.ensure_source("active", source.name, source)
            job = store.claim_next(model_server.STORAGE_DIRS)
            self.assertIsNotNone(job)
            artifact = Path(job["artifact_path"])
            artifact.write_bytes(b"glTF-test-preview")
            store.finish_success(job["revision"], artifact)

            with urlopen(f"{base}/models/active/queued.3mf?preview=1") as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Type"], "model/gltf-binary")
                self.assertEqual(response.read(), b"glTF-test-preview")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_enqueue_status_and_retry_http_contract(self):
        root = Path(self.tempdir.name)
        (root / "queued.3mf").write_bytes(b"PK\x03\x04preview-source")

        server = ThreadingHTTPServer(("127.0.0.1", 0), model_server.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            request = Request(
                f"{base}/api/previews/enqueue",
                data=json.dumps({"scope": "active", "name": "queued.3mf"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                self.assertEqual(response.status, 202)
                enqueued = json.load(response)
            self.assertEqual(enqueued["preview_status"], "pending")

            with urlopen(f"{base}/api/previews/status?scope=active&name=queued.3mf") as response:
                status = json.load(response)
            self.assertEqual(status["preview_revision"], enqueued["preview_revision"])

            store = model_server.preview_store()
            job = store.claim_next(model_server.STORAGE_DIRS)
            store.finish_failure(job["revision"], "test failure", max_attempts=1)
            retry_request = Request(
                f"{base}/api/previews/retry",
                data=json.dumps({"scope": "active", "name": "queued.3mf"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(retry_request) as response:
                self.assertEqual(response.status, 202)
                retried = json.load(response)
            self.assertEqual(retried["preview_status"], "pending")
            self.assertIsNone(retried["preview_error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
