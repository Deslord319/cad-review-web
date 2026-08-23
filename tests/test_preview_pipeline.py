import os
import sys
import tempfile
import types
import unittest
import zipfile
from unittest import mock
from pathlib import Path


SERVER_DIR = Path(__file__).parents[1] / "server"
sys.path.insert(0, str(SERVER_DIR))

from preview_convert import (
    _allocate_face_targets,
    convert_3mf,
    load_3mf_scene,
    simplification_target,
)
from preview_store import PreviewStore, preview_revision, storage_dirs_from_output


class PreviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.storage = storage_dirs_from_output(self.root / "models")
        for directory in self.storage.values():
            directory.mkdir(parents=True, exist_ok=True)
        self.store = PreviewStore(self.root / "preview", face_budget=100_000)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_wal_content_addressing_and_cross_scope_reuse(self):
        content = b"PK\x03\x04same-3mf-content"
        active = self.storage["active"] / "one.3mf"
        archived = self.storage["archive"] / "renamed.3mf"
        active.write_bytes(content)
        archived.write_bytes(content)

        first = self.store.ensure_source("active", active.name, active)
        second = self.store.ensure_source("archive", archived.name, archived)
        self.assertEqual(self.store.journal_mode(), "wal")
        self.assertEqual(first["preview_revision"], second["preview_revision"])
        self.assertEqual(self.store.queue_counts()["pending"], 1)

        job = self.store.claim_next(self.storage)
        self.assertIsNotNone(job)
        artifact = Path(job["artifact_path"])
        artifact.write_bytes(b"glTF-shared")
        self.store.finish_success(job["revision"], artifact)
        self.assertEqual(
            self.store.status_for_source("active", active.name, active)["preview_status"],
            "ready",
        )
        self.assertEqual(
            self.store.status_for_source("archive", archived.name, archived)["preview_status"],
            "ready",
        )

    def test_smallest_source_is_claimed_first(self):
        large = self.storage["archive"] / "large.3mf"
        small = self.storage["archive"] / "small.3mf"
        large.write_bytes(b"PK\x03\x04" + b"L" * 4096)
        small.write_bytes(b"PK\x03\x04" + b"S" * 32)
        self.store.ensure_source("archive", large.name, large)
        self.store.ensure_source("archive", small.name, small)

        job = self.store.claim_next(self.storage)
        self.assertEqual(job["name"], "small.3mf")

    def test_terminal_failure_and_explicit_retry(self):
        source = self.storage["active"] / "broken.3mf"
        source.write_bytes(b"PK\x03\x04broken")
        self.store.ensure_source("active", source.name, source)
        job = self.store.claim_next(self.storage)
        failed = self.store.finish_failure(
            job["revision"], "converter exploded", max_attempts=1
        )
        self.assertEqual(failed["preview_status"], "failed")
        self.assertIn("exploded", failed["preview_error"])

        retried = self.store.retry_source("active", source.name, source)
        self.assertEqual(retried["preview_status"], "pending")
        self.assertIsNone(retried["preview_error"])

    def test_reconcile_discovers_external_files_and_removes_stale_sources(self):
        external = self.storage["archive"] / "external.3mf"
        external.write_bytes(b"PK\x03\x04external")
        result = self.store.reconcile(self.storage)
        self.assertEqual(result, {"indexed": 1, "errors": []})
        self.assertEqual(
            self.store.status_for_source("archive", external.name, external)["preview_status"],
            "pending",
        )

        external.unlink()
        self.store.reconcile(self.storage)
        unknown = self.store.status_for_source(
            "archive", external.name, None, enqueue_missing=False
        )
        self.assertIsNone(unknown["preview_revision"])

    def test_revision_changes_with_pipeline_profile_or_budget(self):
        digest = "a" * 64
        baseline = preview_revision(digest, "v1", "fast", 100_000)
        self.assertNotEqual(baseline, preview_revision(digest, "v2", "fast", 100_000))
        self.assertNotEqual(baseline, preview_revision(digest, "v1", "quality", 100_000))
        self.assertNotEqual(baseline, preview_revision(digest, "v1", "fast", 200_000))

    def test_pipeline_upgrade_reuses_source_digest_but_queues_new_revision(self):
        source = self.storage["archive"] / "upgrade.3mf"
        source.write_bytes(b"PK\x03\x04pipeline-upgrade")
        old_store = PreviewStore(
            self.root / "preview-upgrade", pipeline_version="three-mf-glb-v3"
        )
        old = old_store.ensure_source("archive", source.name, source)
        new_store = PreviewStore(
            self.root / "preview-upgrade", pipeline_version="three-mf-glb-v4"
        )
        before_reconcile = new_store.status_for_source(
            "archive", source.name, source, enqueue_missing=False
        )
        self.assertNotEqual(old["preview_revision"], before_reconcile["preview_revision"])
        self.assertEqual(before_reconcile["preview_status"], "pending")
        with mock.patch("preview_store.file_sha256", side_effect=AssertionError("rehash")):
            new = new_store.ensure_source("archive", source.name, source)
        self.assertNotEqual(old["preview_revision"], new["preview_revision"])
        self.assertEqual(new["preview_status"], "pending")


class PreviewConverterTests(unittest.TestCase):
    def test_face_allocation_is_deterministic_and_strict(self):
        targets = _allocate_face_targets([10, 30, 60], 17)
        self.assertEqual(sum(targets), 17)
        self.assertEqual(targets, _allocate_face_targets([10, 30, 60], 17))
        self.assertTrue(all(0 < target <= count for target, count in zip(targets, [10, 30, 60])))

    def test_large_packaged_part_streams_to_budget_and_cleans_memmap(self):
        core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
        production = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
        vertices = []
        triangles = []
        for index in range(10):
            base = index * 3
            vertices.extend(
                (
                    f'<vertex x="{index}" y="0" z="0"/>',
                    f'<vertex x="{index}" y="1" z="0"/>',
                    f'<vertex x="{index}" y="0" z="1"/>',
                )
            )
            triangles.append(
                f'<triangle v1="{base}" v2="{base + 1}" v3="{base + 2}"/>'
            )
        external = f"""<?xml version="1.0" encoding="UTF-8"?>
        <model unit="millimeter" xmlns="{core}"><resources>
          <object id="1" name="sampled" type="model"><mesh>
            <vertices>{''.join(vertices)}</vertices>
            <triangles>{''.join(triangles)}</triangles>
          </mesh></object>
        </resources></model>"""
        root_model = f"""<?xml version="1.0" encoding="UTF-8"?>
        <model unit="millimeter" xmlns="{core}" xmlns:p="{production}" requiredextensions="p">
          <resources><object id="10" type="model"><components>
            <component p:path="/3D/Objects/large.model" objectid="1"/>
          </components></object></resources>
          <build>
            <item objectid="10"/>
            <item objectid="10" transform="1 0 0 0 1 0 0 0 1 20 0 0"/>
          </build>
        </model>"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "large-split.3mf"
            work = root / "stream-work"
            work.mkdir()
            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("3D/3dmodel.model", root_model)
                archive.writestr("3D/Objects/large.model", external)
            scene, stats = load_3mf_scene(
                source,
                face_budget=4,
                large_part_threshold=1,
                temporary_root=work,
            )
            self.assertEqual(list(work.iterdir()), [])

        self.assertEqual(stats["loader"], "packaged-3mf-stream")
        self.assertTrue(stats["stream_sampled"])
        self.assertEqual(stats["source_unique_faces"], 10)
        self.assertEqual(stats["unique_faces"], 4)
        self.assertEqual(stats["instances"], 2)
        self.assertEqual(stats["instanced_faces"], 8)
        self.assertEqual(len(scene.geometry), 1)
        translations = sorted(
            scene.graph.get(node)[0][0, 3] for node in scene.graph.nodes_geometry
        )
        self.assertEqual(translations, [0.0, 20.0])

    def test_single_model_part_over_threshold_uses_stream_loader(self):
        core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
        model = f"""<?xml version="1.0" encoding="UTF-8"?>
        <model unit="millimeter" xmlns="{core}"><resources>
          <object id="1" type="model"><mesh><vertices>
            <vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/>
            <vertex x="0" y="1" z="0"/><vertex x="0" y="0" z="1"/>
          </vertices><triangles>
            <triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="1" v3="3"/>
            <triangle v1="0" v2="2" v3="3"/><triangle v1="1" v2="2" v3="3"/>
          </triangles></mesh></object>
        </resources><build><item objectid="1"/></build></model>"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "single-large.3mf"
            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("3D/3dmodel.model", model)
            scene, stats = load_3mf_scene(
                source,
                face_budget=2,
                large_part_threshold=1,
            )

        self.assertEqual(stats["loader"], "packaged-3mf-stream")
        self.assertEqual(stats["source_unique_faces"], 4)
        self.assertEqual(stats["unique_faces"], 2)
        self.assertEqual(len(scene.geometry), 1)

    def test_multiple_streamed_parts_obey_global_preview_budget(self):
        core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
        production = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"

        def model_part(offset):
            vertices = []
            triangles = []
            for index in range(10):
                base = index * 3
                vertices.extend(
                    (
                        f'<vertex x="{offset + index}" y="0" z="0"/>',
                        f'<vertex x="{offset + index}" y="1" z="0"/>',
                        f'<vertex x="{offset + index}" y="0" z="1"/>',
                    )
                )
                triangles.append(
                    f'<triangle v1="{base}" v2="{base + 1}" v3="{base + 2}"/>'
                )
            return f"""<model unit="millimeter" xmlns="{core}"><resources>
              <object id="1" type="model"><mesh><vertices>{''.join(vertices)}</vertices>
              <triangles>{''.join(triangles)}</triangles></mesh></object>
              </resources></model>"""

        root_model = f"""<model unit="millimeter" xmlns="{core}" xmlns:p="{production}">
          <resources><object id="10" type="model"><components>
            <component p:path="/3D/Objects/a.model" objectid="1"/>
            <component p:path="/3D/Objects/b.model" objectid="1"/>
          </components></object></resources><build><item objectid="10"/></build></model>"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "two-large.3mf"
            output = root / "two-large.glb"
            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("3D/3dmodel.model", root_model)
                archive.writestr("3D/Objects/a.model", model_part(0))
                archive.writestr("3D/Objects/b.model", model_part(100))
            with mock.patch.dict(
                os.environ,
                {"CAD_VIEWER_3MF_STREAM_THRESHOLD_BYTES": "1"},
            ):
                result = convert_3mf(source, output, 6, "fast")

        self.assertEqual(result["source_faces"], 20)
        self.assertEqual(result["loaded_faces"], 12)
        self.assertEqual(result["preview_faces"], 6)
        self.assertLessEqual(result["preview_faces"], 6)
        self.assertGreater(result["output_bytes"], 20)

    def test_split_3mf_loads_each_external_mesh_once_and_reuses_instances(self):
        core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
        production = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"

        def triangle_mesh(offset):
            return f"""
              <mesh><vertices>
                <vertex x="{offset}" y="0" z="0"/>
                <vertex x="{offset + 1}" y="0" z="0"/>
                <vertex x="{offset}" y="1" z="0"/>
              </vertices><triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh>
            """

        external = f"""<?xml version="1.0" encoding="UTF-8"?>
        <model unit="millimeter" xmlns="{core}"><resources>
          <object id="1" type="model">{triangle_mesh(0)}</object>
          <object id="2" type="model">{triangle_mesh(10)}</object>
          <object id="3" type="model"><components>
            <component objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/>
            <component objectid="2" transform="1 0 0 0 1 0 0 0 1 20 0 0"/>
          </components></object>
        </resources></model>"""
        root_model = f"""<?xml version="1.0" encoding="UTF-8"?>
        <model unit="millimeter" xmlns="{core}" xmlns:p="{production}" requiredextensions="p">
          <resources>
            <object id="10" type="model"><components>
              <component p:path="/3D/Objects/shared.model" objectid="3" transform="1 0 0 0 1 0 0 0 1 1 0 0"/>
              <component p:path="/3D/Objects/shared.model" objectid="1" transform="1 0 0 0 1 0 0 0 1 2 0 0"/>
            </components></object>
            <object id="11" type="model"><components>
              <component p:path="/3D/Objects/shared.model" objectid="3" transform="1 0 0 0 1 0 0 0 1 3 0 0"/>
            </components></object>
          </resources>
          <build>
            <item objectid="10" transform="1 0 0 0 1 0 0 0 1 100 0 0"/>
            <item objectid="11" transform="1 0 0 0 1 0 0 0 1 200 0 0"/>
          </build>
        </model>"""

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "split.3mf"
            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("3D/3dmodel.model", root_model)
                archive.writestr("3D/Objects/shared.model", external)
            scene, stats = load_3mf_scene(source)

        self.assertEqual(stats["loader"], "packaged-3mf")
        self.assertEqual(stats["model_parts_parsed"], 2)
        self.assertEqual(stats["mesh_objects_loaded"], 2)
        self.assertEqual(stats["unique_faces"], 2)
        self.assertEqual(stats["instances"], 5)
        self.assertEqual(stats["instanced_faces"], 5)
        self.assertEqual(len(scene.geometry), 2)

        translations = {}
        for node in scene.graph.nodes_geometry:
            matrix, geometry = scene.graph.get(node)
            translations.setdefault(geometry.rsplit("_", 1)[-1], []).append(matrix[0, 3])
        self.assertEqual(sorted(translations["1"]), [102.0, 111.0, 213.0])
        self.assertEqual(sorted(translations["2"]), [121.0, 223.0])

    def test_simplification_target_never_exceeds_small_geometry(self):
        small_target = simplification_target(712, 400_000, 250_000)
        self.assertIsNotNone(small_target)
        self.assertLess(small_target, 712)
        target = simplification_target(200_000, 400_000, 100_000)
        self.assertEqual(target, 50_000)
        self.assertLess(target, 200_000)

    def test_converter_uses_process_false_and_publishes_atomically(self):
        calls = {"process": None}

        class FakeGeometry:
            def __init__(self, count):
                self.faces = list(range(count))

            def simplify_quadric_decimation(self, *, face_count):
                return FakeGeometry(face_count)

        class FakeScene:
            def __init__(self):
                self.geometry = {"large": FakeGeometry(200), "small": FakeGeometry(3)}

            def export(self, *, file_type):
                self.file_type = file_type
                return b"glTF" + b"x" * 64

        fake_scene = FakeScene()

        def fake_load(_source, *, force, process):
            self.assertEqual(force, "scene")
            calls["process"] = process
            return fake_scene

        fake_trimesh = types.SimpleNamespace(
            load=fake_load,
            Trimesh=type("FakeTrimesh", (), {}),
            Scene=lambda loaded: loaded,
        )
        original = sys.modules.get("trimesh")
        sys.modules["trimesh"] = fake_trimesh
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "sample.3mf"
                output = root / "objects" / "sample.glb"
                source.write_bytes(b"PK\x03\x04fake")
                result = convert_3mf(source, output, 100, "fast")
                self.assertFalse(calls["process"])
                self.assertTrue(output.is_file())
                self.assertEqual(result["source_faces"], 203)
                self.assertLess(result["preview_faces"], result["source_faces"])
                self.assertEqual(list(output.parent.glob("*.tmp")), [])
        finally:
            if original is None:
                sys.modules.pop("trimesh", None)
            else:
                sys.modules["trimesh"] = original


if __name__ == "__main__":
    unittest.main()
