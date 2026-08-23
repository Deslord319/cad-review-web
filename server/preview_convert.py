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
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote


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


class Packaged3MFLoader:
    """Load split/production-extension 3MF parts once and instance by object ID."""

    def __init__(self, archive: zipfile.ZipFile, trimesh: Any, numpy: Any, etree: Any):
        self.archive = archive
        self.trimesh = trimesh
        self.numpy = numpy
        self.etree = etree
        self.member_lookup = {
            name.lstrip("/").lower(): name.lstrip("/")
            for name in archive.namelist()
            if not name.endswith("/")
        }
        self.parts: dict[str, ModelPart] = {}
        self.mesh_objects_loaded = 0

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

    def load_part(self, path: str) -> ModelPart:
        if path in self.parts:
            return self.parts[path]
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
        instance_count = 0
        instanced_faces = 0

        def expand(
            part_path: str,
            object_id: str,
            world_transform: Any,
            ancestry: frozenset[tuple[str, str]],
        ) -> None:
            nonlocal instance_count, instanced_faces
            identity = (part_path, object_id)
            if identity in ancestry:
                raise ValueError(f"Cyclic 3MF component reference: {part_path}#{object_id}")
            part = self.load_part(part_path)
            definition = part.objects.get(object_id)
            if definition is None:
                raise ValueError(f"3MF object is not defined: {part_path}#{object_id}")
            next_ancestry = ancestry | {identity}

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
            "loader": "packaged-3mf",
            "model_parts_parsed": len(self.parts),
            "mesh_objects_loaded": self.mesh_objects_loaded,
            "instances": instance_count,
            "unique_faces": unique_faces,
            "instanced_faces": instanced_faces,
        }


def load_3mf_scene(source: Path) -> tuple[Any, dict]:
    """Use the safe packaged loader for split 3MFs, trimesh for ordinary 3MFs."""

    import trimesh

    try:
        with zipfile.ZipFile(source) as archive:
            model_parts = [
                name
                for name in archive.namelist()
                if not name.endswith("/") and name.lower().endswith(".model")
            ]
            if len(model_parts) > 1:
                import numpy
                from lxml import etree

                return Packaged3MFLoader(archive, trimesh, numpy, etree).scene()
    except zipfile.BadZipFile:
        pass

    loaded = trimesh.load(source, force="scene", process=False)
    scene = trimesh.Scene(loaded) if isinstance(loaded, trimesh.Trimesh) else loaded
    unique_faces = sum(len(geometry.faces) for geometry in scene.geometry.values())
    nodes_geometry = getattr(getattr(scene, "graph", None), "nodes_geometry", ())
    return scene, {
        "loader": "trimesh",
        "model_parts_parsed": 1,
        "mesh_objects_loaded": len(scene.geometry),
        "instances": len(nodes_geometry),
        "unique_faces": unique_faces,
        "instanced_faces": unique_faces,
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

    scene, load_stats = load_3mf_scene(source)
    geometries = list(scene.geometry.items())
    total_faces = sum(len(geometry.faces) for _, geometry in geometries)
    if total_faces <= 0:
        raise ValueError("3MF contains no triangle geometry")

    simplified = 0
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
        "source_faces": total_faces,
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
