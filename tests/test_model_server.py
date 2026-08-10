import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
