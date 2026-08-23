#!/usr/bin/env python3
"""One-shot 3MF to GLB converter used by the preview worker.

Each invocation converts exactly one model and then exits, so trimesh/numpy
allocations are returned to the operating system instead of accumulating in
the long-running HTTP service.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote


DEFAULT_LARGE_PART_THRESHOLD = 64 * 1024 * 1024


@dataclass(frozen=True)
class ComponentReference:
    object_id: str
    part_path: str
    transform: Any


@dataclass
class ObjectDefinition:
    object_id: str
    name: str
    mesh: Any | None
    components: list[ComponentReference]
    source_face_count: int


@dataclass(frozen=True)
class BuildReference:
    object_id: str
    part_path: str
    transform: Any


@dataclass
class ModelPart:
    path: str
    unit: str
    objects: dict[str, ObjectDefinition]
    build: list[BuildReference]


@dataclass
class StreamObjectStats:
    object_id: str
    name: str
    vertex_count: int = 0
    face_count: int = 0
    components: list[ComponentReference] = field(default_factory=list)
    target_faces: int = 0
    vertex_offset: int = 0


def _path_attribute(attributes: dict) -> str | None:
    return next(
        (
            value
            for key, value in attributes.items()
            if key == "path" or key.endswith("}path") or key.endswith(":path")
        ),
        None,
    )


def _transform_from_attributes(attributes: dict, numpy: Any) -> Any:
    transform = numpy.eye(4, dtype=numpy.float64)
    raw = attributes.get("transform")
    if raw is None:
        return transform
    values = numpy.fromstring(raw, dtype=numpy.float64, sep=" ")
    if values.size != 12:
        raise ValueError("3MF transform must contain exactly 12 numbers")
    transform[:3, :4] = values.reshape((4, 3)).T
    return transform


def _resolve_part_path(
    current_path: str,
    referenced_path: str | None,
    member_lookup: dict[str, str],
) -> str:
    if referenced_path is None:
        return current_path
    decoded = unquote(referenced_path)
    if "\\" in decoded or "\x00" in decoded:
        raise ValueError("Invalid 3MF component path")
    if decoded.startswith("/"):
        candidate = decoded.lstrip("/")
    else:
        candidate = posixpath.join(posixpath.dirname(current_path), decoded)
    candidate = posixpath.normpath(candidate)
    if candidate in {"", ".", ".."} or candidate.startswith("../"):
        raise ValueError("3MF component path escapes the package")
    resolved = member_lookup.get(candidate.lower())
    if resolved is None:
        raise ValueError(f"3MF component references missing model part: {candidate}")
    return resolved


def _read_mesh_element(mesh_element: Any, numpy: Any) -> tuple[Any, Any]:
    vertices_element = mesh_element.find("{*}vertices")
    triangles_element = mesh_element.find("{*}triangles")
    if vertices_element is None or triangles_element is None:
        raise ValueError("3MF mesh is missing vertices or triangles")

    vertex_text = " ".join(
        f"{vertex.attrib['x']} {vertex.attrib['y']} {vertex.attrib['z']}"
        for vertex in vertices_element.iter("{*}vertex")
    )
    triangle_text = " ".join(
        f"{triangle.attrib['v1']} {triangle.attrib['v2']} {triangle.attrib['v3']}"
        for triangle in triangles_element.iter("{*}triangle")
    )
    vertices = numpy.fromstring(vertex_text, dtype=numpy.float64, sep=" ")
    faces = numpy.fromstring(triangle_text, dtype=numpy.int64, sep=" ")
    if vertices.size == 0 or faces.size == 0 or vertices.size % 3 or faces.size % 3:
        raise ValueError("3MF mesh contains malformed vertex or triangle data")
    vertices = vertices.reshape((-1, 3))
    faces = faces.reshape((-1, 3))
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("3MF triangle references an invalid vertex")
    return vertices, faces


def _combined_mesh(object_element: Any, numpy: Any, trimesh: Any) -> Any | None:
    vertex_groups = []
    face_groups = []
    vertex_offset = 0
    for mesh_element in object_element.iter("{*}mesh"):
        vertices, faces = _read_mesh_element(mesh_element, numpy)
        vertex_groups.append(vertices)
        face_groups.append(faces + vertex_offset)
        vertex_offset += len(vertices)
    if not vertex_groups:
        return None
    vertices = vertex_groups[0] if len(vertex_groups) == 1 else numpy.vstack(vertex_groups)
    faces = face_groups[0] if len(face_groups) == 1 else numpy.vstack(face_groups)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _release_xml_element(element: Any) -> None:
    element.clear()
    parent = element.getparent()
    if parent is not None:
        while element.getprevious() is not None:
            del parent[0]


def _allocate_face_targets(face_counts: list[int], face_budget: int) -> list[int]:
    """Allocate a strict deterministic face budget across independent meshes."""

    if face_budget <= 0:
        raise ValueError("face budget must be greater than zero")
    counts = [max(0, int(count)) for count in face_counts]
    total = sum(counts)
    if total <= face_budget:
        return counts

    nonempty = [index for index, count in enumerate(counts) if count > 0]
    targets = [0] * len(counts)
    if len(nonempty) >= face_budget:
        for index in sorted(nonempty, key=lambda item: (-counts[item], item))[
            :face_budget
        ]:
            targets[index] = 1
        return targets

    for index in nonempty:
        targets[index] = 1
    remaining = face_budget - len(nonempty)
    capacities = [max(0, count - 1) for count in counts]
    capacity_total = sum(capacities)
    if remaining <= 0 or capacity_total <= 0:
        return targets

    exact = [remaining * capacity / capacity_total for capacity in capacities]
    additions = [min(capacities[index], int(value)) for index, value in enumerate(exact)]
    for index, addition in enumerate(additions):
        targets[index] += addition
    remainder = remaining - sum(additions)
    order = sorted(
        nonempty,
        key=lambda index: (-(exact[index] - int(exact[index])), index),
    )
    for index in order:
        if remainder <= 0:
            break
        if targets[index] < counts[index]:
            targets[index] += 1
            remainder -= 1
    if remainder:
        for index in nonempty:
            if remainder <= 0:
                break
            available = counts[index] - targets[index]
            addition = min(available, remainder)
            targets[index] += addition
            remainder -= addition
    return targets


def _sample_face_indices(face_count: int, target_faces: int, numpy: Any) -> Any:
    if target_faces <= 0:
        return numpy.empty((0,), dtype=numpy.int64)
    if target_faces >= face_count:
        return numpy.arange(face_count, dtype=numpy.int64)
    return numpy.floor(
        (numpy.arange(target_faces, dtype=numpy.float64) + 0.5)
        * face_count
        / target_faces
    ).astype(numpy.int64)


def _sample_mesh(mesh: Any, target_faces: int, numpy: Any, trimesh: Any) -> Any:
    face_count = len(mesh.faces)
    if target_faces >= face_count:
        return mesh
    indices = _sample_face_indices(face_count, target_faces, numpy)
    selected_faces = numpy.asarray(mesh.faces, dtype=numpy.int64)[indices]
    referenced, inverse = numpy.unique(selected_faces.reshape(-1), return_inverse=True)
    vertices = numpy.asarray(mesh.vertices)[referenced]
    return trimesh.Trimesh(
        vertices=vertices,
        faces=inverse.reshape((-1, 3)),
        process=False,
    )


class Packaged3MFLoader:
    """Load split/production-extension 3MF parts once and instance by object ID."""

    def __init__(
        self,
        archive: zipfile.ZipFile,
        trimesh: Any,
        numpy: Any,
        etree: Any,
        *,
        face_budget: int,
        large_part_threshold: int,
        temporary_root: Path | None,
    ):
        self.archive = archive
        self.trimesh = trimesh
        self.numpy = numpy
        self.etree = etree
        self.face_budget = face_budget
        self.large_part_threshold = large_part_threshold
        self.temporary_root = temporary_root
        self.member_lookup = {
            name.lstrip("/").lower(): name.lstrip("/")
            for name in archive.namelist()
            if not name.endswith("/")
        }
        self.parts: dict[str, ModelPart] = {}
        self.mesh_objects_loaded = 0
        self.stream_sampled_parts = 0

    def root_path(self) -> str:
        root = self.member_lookup.get("3d/3dmodel.model")
        if root is None:
            root = next(
                (
                    path
                    for key, path in self.member_lookup.items()
                    if key.endswith(".model") and "3d/" in key
                ),
                None,
            )
        if root is None:
            raise ValueError("3MF package has no root model part")
        return root

    def _stream_part_index(
        self, path: str
    ) -> tuple[str, dict[str, StreamObjectStats], list[BuildReference]]:
        """First pass: count mesh records and retain only the tiny object graph."""

        objects: dict[str, StreamObjectStats] = {}
        build: list[BuildReference] = []
        unit = "millimeter"
        current_object: StreamObjectStats | None = None
        tags = (
            "{*}model",
            "{*}object",
            "{*}vertex",
            "{*}triangle",
            "{*}component",
            "{*}item",
        )
        with self.archive.open(path) as stream:
            context = self.etree.iterparse(
                stream,
                events=("start", "end"),
                tag=tags,
                huge_tree=True,
                no_network=True,
            )
            for event, element in context:
                local_name = self.etree.QName(element).localname
                if event == "start":
                    if local_name == "model":
                        unit = element.attrib.get("unit", "millimeter")
                    elif local_name == "object":
                        object_id = element.attrib.get("id")
                        if not object_id:
                            raise ValueError(f"3MF object in {path} has no ID")
                        if object_id in objects:
                            raise ValueError(f"Duplicate 3MF object ID in {path}: {object_id}")
                        current_object = StreamObjectStats(
                            object_id=object_id,
                            name=element.attrib.get("name", object_id),
                        )
                        objects[object_id] = current_object
                    continue

                if local_name == "vertex":
                    if current_object is None:
                        raise ValueError(f"3MF vertex outside an object in {path}")
                    current_object.vertex_count += 1
                elif local_name == "triangle":
                    if current_object is None:
                        raise ValueError(f"3MF triangle outside an object in {path}")
                    current_object.face_count += 1
                elif local_name == "component":
                    if current_object is None:
                        raise ValueError(f"3MF component outside an object in {path}")
                    current_object.components.append(
                        ComponentReference(
                            object_id=element.attrib["objectid"],
                            part_path=_resolve_part_path(
                                path,
                                _path_attribute(element.attrib),
                                self.member_lookup,
                            ),
                            transform=_transform_from_attributes(
                                element.attrib, self.numpy
                            ),
                        )
                    )
                elif local_name == "item":
                    build.append(
                        BuildReference(
                            object_id=element.attrib["objectid"],
                            part_path=_resolve_part_path(
                                path,
                                _path_attribute(element.attrib),
                                self.member_lookup,
                            ),
                            transform=_transform_from_attributes(
                                element.attrib, self.numpy
                            ),
                        )
                    )
                elif local_name == "object":
                    current_object = None
                _release_xml_element(element)

        targets = _allocate_face_targets(
            [definition.face_count for definition in objects.values()],
            self.face_budget,
        )
        vertex_offset = 0
        for definition, target in zip(objects.values(), targets):
            definition.target_faces = target
            definition.vertex_offset = vertex_offset
            vertex_offset += definition.vertex_count
        return unit, objects, build

    def _stream_part_meshes(
        self,
        path: str,
        indexed: dict[str, StreamObjectStats],
    ) -> dict[str, ObjectDefinition]:
        """Second pass: memmap vertices and retain only evenly sampled faces."""

        total_vertices = sum(definition.vertex_count for definition in indexed.values())
        if total_vertices <= 0:
            return {
                object_id: ObjectDefinition(
                    object_id=object_id,
                    name=definition.name,
                    mesh=None,
                    components=definition.components,
                    source_face_count=definition.face_count,
                )
                for object_id, definition in indexed.items()
            }

        if self.temporary_root is not None:
            self.temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="cad-3mf-stream-",
            dir=self.temporary_root,
        ) as temporary_directory:
            memmap_path = Path(temporary_directory) / "vertices.float32"
            vertices = self.numpy.memmap(
                memmap_path,
                mode="w+",
                dtype=self.numpy.float32,
                shape=(total_vertices, 3),
            )
            face_buffers = {
                object_id: self.numpy.empty(
                    (definition.target_faces, 3), dtype=self.numpy.int64
                )
                for object_id, definition in indexed.items()
                if definition.target_faces > 0
            }
            sample_indices = {
                object_id: _sample_face_indices(
                    definition.face_count,
                    definition.target_faces,
                    self.numpy,
                )
                for object_id, definition in indexed.items()
                if definition.target_faces > 0
            }
            vertex_seen = {object_id: 0 for object_id in indexed}
            face_seen = {object_id: 0 for object_id in indexed}
            sample_seen = {object_id: 0 for object_id in sample_indices}
            current_object_id: str | None = None
            current_mesh_vertex_base = 0
            tags = (
                "{*}object",
                "{*}mesh",
                "{*}vertex",
                "{*}triangle",
                "{*}component",
            )
            try:
                with self.archive.open(path) as stream:
                    context = self.etree.iterparse(
                        stream,
                        events=("start", "end"),
                        tag=tags,
                        huge_tree=True,
                        no_network=True,
                    )
                    for event, element in context:
                        local_name = self.etree.QName(element).localname
                        if event == "start":
                            if local_name == "object":
                                current_object_id = element.attrib.get("id")
                                if current_object_id not in indexed:
                                    raise ValueError(
                                        f"Unexpected 3MF object in {path}: "
                                        f"{current_object_id}"
                                    )
                            elif local_name == "mesh" and current_object_id is not None:
                                current_mesh_vertex_base = vertex_seen[current_object_id]
                            continue

                        if local_name == "vertex":
                            if current_object_id is None:
                                raise ValueError(f"3MF vertex outside an object in {path}")
                            definition = indexed[current_object_id]
                            local_index = vertex_seen[current_object_id]
                            vertices[definition.vertex_offset + local_index] = (
                                float(element.attrib["x"]),
                                float(element.attrib["y"]),
                                float(element.attrib["z"]),
                            )
                            vertex_seen[current_object_id] = local_index + 1
                        elif local_name == "triangle":
                            if current_object_id is None:
                                raise ValueError(f"3MF triangle outside an object in {path}")
                            local_index = face_seen[current_object_id]
                            selected = sample_indices.get(current_object_id)
                            cursor = sample_seen.get(current_object_id, 0)
                            if selected is not None and cursor < len(selected):
                                if local_index == int(selected[cursor]):
                                    face_buffers[current_object_id][cursor] = (
                                        int(element.attrib["v1"])
                                        + current_mesh_vertex_base,
                                        int(element.attrib["v2"])
                                        + current_mesh_vertex_base,
                                        int(element.attrib["v3"])
                                        + current_mesh_vertex_base,
                                    )
                                    sample_seen[current_object_id] = cursor + 1
                            face_seen[current_object_id] = local_index + 1
                        elif local_name == "object":
                            current_object_id = None
                        _release_xml_element(element)

                objects: dict[str, ObjectDefinition] = {}
                for object_id, definition in indexed.items():
                    if vertex_seen[object_id] != definition.vertex_count:
                        raise ValueError(
                            f"3MF vertex count changed between passes: {path}#{object_id}"
                        )
                    if face_seen[object_id] != definition.face_count:
                        raise ValueError(
                            f"3MF face count changed between passes: {path}#{object_id}"
                        )
                    if sample_seen.get(object_id, 0) != definition.target_faces:
                        raise ValueError(
                            f"3MF face sampling was incomplete: {path}#{object_id}"
                        )

                    mesh = None
                    if definition.target_faces > 0:
                        sampled_faces = face_buffers[object_id]
                        if (
                            sampled_faces.min() < 0
                            or sampled_faces.max() >= definition.vertex_count
                        ):
                            raise ValueError(
                                f"3MF triangle references an invalid vertex: "
                                f"{path}#{object_id}"
                            )
                        referenced, inverse = self.numpy.unique(
                            sampled_faces.reshape(-1), return_inverse=True
                        )
                        selected_vertices = self.numpy.array(
                            vertices[definition.vertex_offset + referenced],
                            dtype=self.numpy.float32,
                            copy=True,
                        )
                        mesh = self.trimesh.Trimesh(
                            vertices=selected_vertices,
                            faces=inverse.reshape((-1, 3)),
                            process=False,
                        )
                        self.mesh_objects_loaded += 1
                    objects[object_id] = ObjectDefinition(
                        object_id=object_id,
                        name=definition.name,
                        mesh=mesh,
                        components=definition.components,
                        source_face_count=definition.face_count,
                    )
                return objects
            finally:
                vertices.flush()
                mmap = getattr(vertices, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                del vertices

    def load_part(self, path: str) -> ModelPart:
        if path in self.parts:
            return self.parts[path]
        archive_info = self.archive.getinfo(path)
        if archive_info.file_size >= self.large_part_threshold:
            unit, indexed, build = self._stream_part_index(path)
            objects = self._stream_part_meshes(path, indexed)
            part = ModelPart(path=path, unit=unit, objects=objects, build=build)
            self.parts[path] = part
            self.stream_sampled_parts += 1
            return part

        objects: dict[str, ObjectDefinition] = {}
        build: list[BuildReference] = []
        unit = "millimeter"
        with self.archive.open(path) as stream:
            context = self.etree.iterparse(
                stream,
                events=("start", "end"),
                tag=("{*}model", "{*}object", "{*}build"),
            )
            for event, element in context:
                local_name = self.etree.QName(element).localname
                if event == "start" and local_name == "model":
                    unit = element.attrib.get("unit", "millimeter")
                    continue
                if event != "end":
                    continue
                if local_name == "object":
                    object_id = element.attrib.get("id")
                    if not object_id:
                        raise ValueError(f"3MF object in {path} has no ID")
                    mesh = _combined_mesh(element, self.numpy, self.trimesh)
                    if mesh is not None:
                        self.mesh_objects_loaded += 1
                    components = [
                        ComponentReference(
                            object_id=component.attrib["objectid"],
                            part_path=_resolve_part_path(
                                path,
                                _path_attribute(component.attrib),
                                self.member_lookup,
                            ),
                            transform=_transform_from_attributes(
                                component.attrib, self.numpy
                            ),
                        )
                        for component in element.iter("{*}component")
                    ]
                    objects[object_id] = ObjectDefinition(
                        object_id=object_id,
                        name=element.attrib.get("name", object_id),
                        mesh=mesh,
                        components=components,
                        source_face_count=0 if mesh is None else len(mesh.faces),
                    )
                    _release_xml_element(element)
                elif local_name == "build":
                    build = [
                        BuildReference(
                            object_id=item.attrib["objectid"],
                            part_path=_resolve_part_path(
                                path,
                                _path_attribute(item.attrib),
                                self.member_lookup,
                            ),
                            transform=_transform_from_attributes(item.attrib, self.numpy),
                        )
                        for item in element.iter("{*}item")
                    ]
                    _release_xml_element(element)

        part = ModelPart(path=path, unit=unit, objects=objects, build=build)
        self.parts[path] = part
        return part

    def scene(self) -> tuple[Any, dict]:
        root_path = self.root_path()
        root = self.load_part(root_path)
        if not root.build:
            raise ValueError("Split 3MF root model has no build items")
        scene = self.trimesh.Scene(base_frame="world", metadata={"units": root.unit})
        geometry_names: dict[tuple[str, str], str] = {}
        source_identities: set[tuple[str, str]] = set()
        instance_count = 0
        instanced_faces = 0
        source_unique_faces = 0
        source_instanced_faces = 0

        def expand(
            part_path: str,
            object_id: str,
            world_transform: Any,
            ancestry: frozenset[tuple[str, str]],
        ) -> None:
            nonlocal instance_count, instanced_faces
            nonlocal source_unique_faces, source_instanced_faces
            identity = (part_path, object_id)
            if identity in ancestry:
                raise ValueError(f"Cyclic 3MF component reference: {part_path}#{object_id}")
            part = self.load_part(part_path)
            definition = part.objects.get(object_id)
            if definition is None:
                raise ValueError(f"3MF object is not defined: {part_path}#{object_id}")
            next_ancestry = ancestry | {identity}

            if identity not in source_identities:
                source_identities.add(identity)
                source_unique_faces += definition.source_face_count
            source_instanced_faces += definition.source_face_count

            if definition.mesh is not None:
                geometry_name = geometry_names.get(identity)
                if geometry_name is None:
                    geometry_name = f"geometry_{len(geometry_names):06d}_{object_id}"
                    geometry_names[identity] = geometry_name
                    scene.geometry[geometry_name] = definition.mesh
                node_name = f"instance_{instance_count:08d}_{object_id}"
                scene.graph.update(
                    frame_to=node_name,
                    frame_from="world",
                    matrix=world_transform,
                    geometry=geometry_name,
                )
                instance_count += 1
                instanced_faces += len(definition.mesh.faces)
                if instance_count > 1_000_000:
                    raise ValueError("3MF expands to more than one million instances")

            for component in definition.components:
                expand(
                    component.part_path,
                    component.object_id,
                    world_transform @ component.transform,
                    next_ancestry,
                )

        for item in root.build:
            expand(item.part_path, item.object_id, item.transform, frozenset())

        unique_faces = sum(len(mesh.faces) for mesh in scene.geometry.values())
        if unique_faces <= 0 or instance_count <= 0:
            raise ValueError("Split 3MF build contains no triangle geometry")
        return scene, {
            "loader": (
                "packaged-3mf-stream"
                if self.stream_sampled_parts
                else "packaged-3mf"
            ),
            "stream_sampled": bool(self.stream_sampled_parts),
            "stream_sampled_parts": self.stream_sampled_parts,
            "model_parts_parsed": len(self.parts),
            "mesh_objects_loaded": self.mesh_objects_loaded,
            "instances": instance_count,
            "unique_faces": unique_faces,
            "source_unique_faces": source_unique_faces,
            "instanced_faces": instanced_faces,
            "source_instanced_faces": source_instanced_faces,
        }


def load_3mf_scene(
    source: Path,
    *,
    face_budget: int = 100_000,
    large_part_threshold: int | None = None,
    temporary_root: Path | None = None,
) -> tuple[Any, dict]:
    """Use the safe packaged loader for split 3MFs, trimesh for ordinary 3MFs."""

    import trimesh

    if face_budget <= 0:
        raise ValueError("face budget must be greater than zero")
    if large_part_threshold is None:
        large_part_threshold = int(
            os.environ.get(
                "CAD_VIEWER_3MF_STREAM_THRESHOLD_BYTES",
                str(DEFAULT_LARGE_PART_THRESHOLD),
            )
        )
    if large_part_threshold <= 0:
        raise ValueError("large 3MF streaming threshold must be greater than zero")
    if temporary_root is None:
        configured_temporary_root = os.environ.get("CAD_VIEWER_PREVIEW_TEMP_DIR")
        if configured_temporary_root:
            temporary_root = Path(configured_temporary_root).resolve()

    try:
        with zipfile.ZipFile(source) as archive:
            model_parts = [
                info
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".model")
            ]
            if len(model_parts) > 1 or any(
                info.file_size >= large_part_threshold for info in model_parts
            ):
                import numpy
                from lxml import etree

                return Packaged3MFLoader(
                    archive,
                    trimesh,
                    numpy,
                    etree,
                    face_budget=face_budget,
                    large_part_threshold=large_part_threshold,
                    temporary_root=temporary_root,
                ).scene()
    except zipfile.BadZipFile:
        pass

    loaded = trimesh.load(source, force="scene", process=False)
    scene = trimesh.Scene(loaded) if isinstance(loaded, trimesh.Trimesh) else loaded
    unique_faces = sum(len(geometry.faces) for geometry in scene.geometry.values())
    nodes_geometry = getattr(getattr(scene, "graph", None), "nodes_geometry", ())
    return scene, {
        "loader": "trimesh",
        "stream_sampled": False,
        "stream_sampled_parts": 0,
        "model_parts_parsed": 1,
        "mesh_objects_loaded": len(scene.geometry),
        "instances": len(nodes_geometry),
        "unique_faces": unique_faces,
        "source_unique_faces": unique_faces,
        "instanced_faces": unique_faces,
        "source_instanced_faces": unique_faces,
    }


def simplification_target(face_count: int, total_faces: int, face_budget: int) -> int | None:
    """Return a valid target strictly below the source count, or None to keep it."""

    if face_count <= 4 or total_faces <= face_budget:
        return None
    proportional = int(face_budget * face_count / total_faces)
    target = max(4, proportional)
    if target >= face_count:
        return None
    return target


def convert_3mf(source: Path, output: Path, face_budget: int, profile: str) -> dict:
    if source.suffix.lower() != ".3mf" or source.is_symlink() or not source.is_file():
        raise ValueError("Source must be a regular 3MF file")
    if face_budget <= 0:
        raise ValueError("face budget must be greater than zero")
    if profile != "fast":
        raise ValueError(f"Unsupported preview profile: {profile}")

    scene, load_stats = load_3mf_scene(source, face_budget=face_budget)
    geometries = list(scene.geometry.items())
    total_faces = sum(len(geometry.faces) for _, geometry in geometries)
    if total_faces <= 0:
        raise ValueError("3MF contains no triangle geometry")

    simplified = 0
    if load_stats.get("stream_sampled"):
        import numpy
        import trimesh

        targets = _allocate_face_targets(
            [len(geometry.faces) for _, geometry in geometries], face_budget
        )
        for (key, geometry), target in zip(geometries, targets):
            if target >= len(geometry.faces):
                continue
            if target <= 0:
                scene.delete_geometry(key)
            else:
                scene.geometry[key] = _sample_mesh(
                    geometry, target, numpy, trimesh
                )
            simplified += 1
    else:
        for key, geometry in geometries:
            original_faces = len(geometry.faces)
            target = simplification_target(original_faces, total_faces, face_budget)
            if target is None:
                continue
            replacement = geometry.simplify_quadric_decimation(face_count=target)
            if len(replacement.faces) <= 0 or len(replacement.faces) >= original_faces:
                continue
            scene.geometry[key] = replacement
            simplified += 1

    payload = scene.export(file_type="glb")
    if not isinstance(payload, (bytes, bytearray)) or len(payload) <= 20:
        raise ValueError("GLB export produced no usable data")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()

    final_faces = sum(len(geometry.faces) for geometry in scene.geometry.values())
    return {
        **load_stats,
        "source_faces": int(load_stats.get("source_unique_faces", total_faces)),
        "loaded_faces": total_faces,
        "preview_faces": final_faces,
        "simplified_geometries": simplified,
        "output_bytes": output.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one 3MF into an atomic GLB preview")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--face-budget", type=int, required=True)
    parser.add_argument("--profile", default="fast")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    result = convert_3mf(
        arguments.source.resolve(),
        arguments.output.resolve(),
        arguments.face_budget,
        arguments.profile,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
