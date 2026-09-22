"""Sensor bindings carried entirely by standard MJCF custom elements.

There is no embedded JSON/XML, executable path, or preprocessing requirement.
MuJoCo resolves the tuple's typed site reference; Orca interprets the names.
This layout is independent of both the sensor SDK release and the C ABI.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


CUSTOM_SENSOR_PREFIX = "orca.sensor.v1/"
_RESERVED_PREFIX = "orca.sensor.v"
_DECLARATION_PREFIXES = ("orca.sensor.instances", _RESERVED_PREFIX)


def validate_main_xml_sensor_declarations(path: str | Path, root: ET.Element) -> None:
    """Reject Orca declarations hidden in includes, without expanding the model.

    Geometry and unrelated custom data can still be included. Included XML roots
    are wrappers, so examine their children even when an include sits inside a
    main-file custom block. Match MuJoCo's local-file lookup: main XML directory
    first, then the including file's directory. Resource-provider/VFS includes
    are not supported by this filesystem-based scene entry point.
    """
    main_path = Path(path).absolute()
    active_paths = {main_path.resolve()}

    def inspect(node: ET.Element, source: Path, *, included: bool) -> None:
        name = node.get("name", "")
        if (included and node.tag.lower() in {"text", "tuple", "numeric"}
                and name.startswith(_DECLARATION_PREFIXES)):
            raise ValueError(
                f"Orca sensor declarations must be in the main XML: {name!r} is in "
                f"included file {source}; move its declarations to {main_path}")
        if node.tag.lower() != "include":
            for child in node:
                inspect(child, source, included=included)
            return
        filename = node.get("file")
        if not filename or len(node):
            raise ValueError(f"Invalid XML include in {source}: expected file and no child elements")
        candidates = (main_path.parent / filename, source.parent / filename)
        target = next((candidate for candidate in candidates if candidate.is_file()), None)
        if target is None:
            raise ValueError(f"Cannot inspect XML include {filename!r} from {source}: local file not found")
        resolved = target.resolve()
        if resolved in active_paths:
            raise ValueError(f"Cyclic XML include while checking Orca sensor declarations: {target}")
        active_paths.add(resolved)
        try:
            include_root = ET.parse(target).getroot()
            for child in include_root:
                inspect(child, target, included=True)
        finally:
            active_paths.remove(resolved)

    inspect(root, main_path, included=False)


def _name(value: str | None, label: str, *, instance: bool = False) -> str:
    if (not isinstance(value, str) or not value or "\0" in value or "/" in value
            or len(value.encode("utf-8")) > (198 if instance else 512)):
        raise ValueError(f"Invalid custom sensor {label}")
    if instance and re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None:
        raise ValueError("Invalid custom sensor instance name")
    return value


def read_custom_sensor_instances(root: ET.Element) -> tuple[dict, ...] | None:
    """Return bindings to existing sites/objects, or None when absent.

Each instance has one plugin text and either one site tuple or exact typed
object tuples. Optional geom ownership, seed and scalar config are metadata only. Full names, including instance names
containing dots, are segmented only on '/'. No element ordering is significant.
"""
    instances: dict[str, dict] = {}
    seen: set[str] = set()
    for custom in root.findall("custom"):
        for node in custom:
            full_name = node.get("name", "")
            if not full_name.startswith(_RESERVED_PREFIX):
                continue
            if not full_name.startswith(CUSTOM_SENSOR_PREFIX):
                raise ValueError(f"Unsupported custom sensor declaration version: {full_name}")
            if full_name in seen:
                raise ValueError(f"Duplicate custom sensor declaration: {full_name}")
            seen.add(full_name)
            parts = full_name[len(CUSTOM_SENSOR_PREFIX):].split("/")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid custom sensor declaration name: {full_name}")
            instance_id = _name(parts[0], "instance name", instance=True)
            entry = instances.setdefault(instance_id, {"instance_id": instance_id,
                                                       "global_parameters": {}})
            if len(instances) > 256:
                raise ValueError("At most 256 custom sensor instances are supported")
            field = parts[1]
            if field == "plugin" and len(parts) == 2:
                if node.tag != "text" or set(node.attrib) != {"name", "data"} or len(node):
                    raise ValueError(f"{full_name}: plugin requires a text with name/data")
                type_id = node.get("data")
                if (not isinstance(type_id, str) or len(type_id) > 512
                        or re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+", type_id) is None):
                    raise ValueError(f"{full_name}: invalid plugin type identifier")
                entry["type_id"] = type_id
            elif ((field == "site" and len(parts) == 2)
                  or (field == "object" and len(parts) == 3)):
                reference = _references(node, full_name, single=True)[0]
                if field == "site":
                    if reference["kind"] != "site":
                        raise ValueError(f"{full_name}: expected an exact objtype='site' reference")
                    entry["site"] = reference["name"]
                else:
                    alias = _name(parts[2], "object alias")
                    entry.setdefault("objects", {})[alias] = reference
                    if len(entry["objects"]) > 256:
                        raise ValueError(f"{instance_id}: at most 256 object bindings are supported")
            elif field == "geoms" and len(parts) == 2:
                references = _references(node, full_name, single=False)
                if any(item["kind"] != "geom" for item in references):
                    raise ValueError(f"{full_name}: ownership requires exact geom references")
                names = tuple(item["name"] for item in references)
                if len(set(names)) != len(names):
                    raise ValueError(f"{full_name}: duplicate owned geom")
                entry["instance_geoms"] = names
            elif field == "seed" and len(parts) == 2:
                value = node.get("data", "")
                if (node.tag != "text" or len(node) or set(node.attrib) != {"name", "data"}
                        or re.fullmatch(r"[0-9]{1,20}", value) is None or int(value) >= 2**64):
                    raise ValueError(f"{full_name}: seed requires a uint64 decimal text")
                entry["seed"] = int(value)
            elif field == "config" and len(parts) == 3:
                key = _name(parts[2], "config name")
                if (node.tag != "numeric" or len(node)
                        or not {"name", "data"} <= node.attrib.keys()
                        or node.attrib.keys() - {"name", "data", "size"}
                        or node.get("size", "1") != "1"):
                    raise ValueError(f"{full_name}: config requires a scalar numeric")
                tokens = node.get("data", "").split()
                try:
                    value = float(tokens[0]) if len(tokens) == 1 else float("nan")
                except ValueError as error:
                    raise ValueError(f"{full_name}: config must be a finite scalar") from error
                if not math.isfinite(value):
                    raise ValueError(f"{full_name}: config must be a finite scalar")
                entry["global_parameters"][key] = value
                if len(entry["global_parameters"]) > 256:
                    raise ValueError(f"{instance_id}: at most 256 config parameters are supported")
            else:
                raise ValueError(f"Unknown custom sensor field: {full_name}")
    for name, entry in instances.items():
        if "type_id" not in entry or (("site" in entry) == ("objects" in entry)):
            raise ValueError(f"{name}: custom sensor requires plugin text and either site or object tuples")
        if "instance_geoms" in entry and "objects" not in entry:
            raise ValueError(f"{name}: explicit geom ownership requires object bindings")
    return tuple(instances.values()) if instances else None


def _references(node: ET.Element, label: str, *, single: bool) -> list[dict[str, str]]:
    """Read only exact typed references; no name inference or XML mutation."""
    if (node.tag != "tuple" or set(node.attrib) != {"name"}
            or not 1 <= len(node) <= 4096 or (single and len(node) != 1)):
        raise ValueError(f"{label}: expected a tuple with {'one element' if single else 'geom elements'}")
    result = []
    for target in node:
        if (target.tag != "element" or len(target)
                or not {"objtype", "objname"} <= target.attrib.keys()
                or target.attrib.keys() - {"objtype", "objname", "prm"}
                or target.get("objtype") not in {"body", "geom", "site"}):
            raise ValueError(f"{label}: expected an exact body, geom or site reference")
        try:
            if float(target.get("prm", "0")) != 0:
                raise ValueError()
        except ValueError as error:
            raise ValueError(f"{label}: tuple prm has no sensor meaning and must be zero") from error
        name = target.get("objname")
        if not name or "\0" in name or len(name.encode("utf-8")) > 512:
            raise ValueError(f"{label}: invalid object name")
        result.append({"kind": target.get("objtype"), "name": name})
    return result
