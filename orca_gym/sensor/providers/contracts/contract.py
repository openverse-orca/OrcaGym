"""Public input-layout DTOs and client for the Orca Sensor contract executable.

Contract lowering, validation, fingerprints and C header generation are owned
by the separately built tool. This module only reconstructs an already resolved
C layout from its primitive/array/struct description; it never executes source
code or evaluates type expressions returned by a tool or vendor package.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._rpc import ContractError, SensorToolError, request, source_value
from .version import SDK_VERSION

DTYPES = {"f64": 1, "f32": 2, "u32": 3, "u64": 4, "u8": 5}
CTYPES = {"f64": ctypes.c_double, "f32": ctypes.c_float, "u32": ctypes.c_uint32,
          "u64": ctypes.c_uint64, "u8": ctypes.c_uint8}
FIXED_COUNT_OFFSET = 0xFFFFFFFF
MAX_CAPACITY = 65536
MAX_FIELDS = 64
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_VALUES = 1048576


@dataclass(frozen=True)
class LayoutField:
    field_id: int
    dtype: int
    offset: int
    stride: int
    capacity: int
    components: int
    count_offset: int


@dataclass(frozen=True)
class ValueField:
    field_id: int
    path: str
    dtype: str
    shape: tuple[int, ...]
    capacity: int
    variable: bool
    query_source: str | None


@dataclass(frozen=True)
class CompiledContract:
    document: dict[str, Any]
    fingerprint: str
    ctypes_type: type[ctypes.Structure]
    layout_fields: tuple[LayoutField, ...]
    value_fields: tuple[ValueField, ...]
    resolved_fields: list[dict[str, Any]]
    requirements: list[dict[str, Any]]
    resolved_objects: dict[str, Any]
    sdk_version: str = SDK_VERSION

    @property
    def contract_id(self) -> str:
        return self.document["contract_id"]

    @property
    def c_struct(self) -> str:
        return self.document["c_struct"]

    @property
    def fields(self) -> list[dict[str, Any]]:
        return self.resolved_fields

    @property
    def schema_version(self) -> int:
        return self.document.get("schema_version", 2)

    @property
    def objects(self) -> dict[str, Any]:
        return self.document.get("objects", self.resolved_objects)

    @property
    def size(self) -> int:
        return ctypes.sizeof(self.ctypes_type)

    @property
    def bindings(self) -> dict[str, Any]:
        return self.document.get("bindings", {})

    @property
    def queries(self) -> dict[str, Any]:
        return self.document.get("queries", {})


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isidentifier() or value.startswith("_"):
        raise SensorToolError("Invalid C identifier in Orca Sensor tool layout")
    return value


def _integer(value: Any, maximum: int = MAX_INPUT_BYTES) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise SensorToolError("Invalid size in Orca Sensor tool layout")
    return value


def _restore_ctype(node: dict[str, Any], depth: int = 0):
    if not isinstance(node, dict) or depth > 12:
        raise SensorToolError("Invalid nested C type in Orca Sensor tool layout")
    kind = node.get("kind")
    if kind == "primitive":
        dtype = node.get("dtype")
        if not isinstance(dtype, str) or dtype not in CTYPES:
            raise SensorToolError("Unsupported primitive in Orca Sensor tool layout")
        return CTYPES[dtype]
    if kind == "array":
        item = _restore_ctype(node.get("item"), depth + 1)
        length = _integer(node.get("length"))
        if ctypes.sizeof(item) * length > MAX_INPUT_BYTES:
            raise SensorToolError("Array exceeds the Orca Sensor input byte limit")
        return item * length
    if kind != "struct":
        raise SensorToolError("Unsupported C type in Orca Sensor tool layout")
    name = _identifier(node.get("name"))
    fields = node.get("fields")
    if not isinstance(fields, list) or not 1 <= len(fields) <= MAX_FIELDS * 2:
        raise SensorToolError("Invalid struct fields in Orca Sensor tool layout")
    restored, offsets, names = [], [], set()
    for field in fields:
        if not isinstance(field, dict):
            raise SensorToolError("Invalid struct field in Orca Sensor tool layout")
        field_name = _identifier(field.get("name"))
        if field_name in names:
            raise SensorToolError("Duplicate struct field in Orca Sensor tool layout")
        names.add(field_name)
        offset = field.get("offset")
        if type(offset) is not int or not 0 <= offset < MAX_INPUT_BYTES:
            raise SensorToolError("Invalid field offset in Orca Sensor tool layout")
        restored.append((field_name, _restore_ctype(field.get("type"), depth + 1)))
        offsets.append((field_name, offset))
    result = type(name, (ctypes.Structure,), {"_fields_": restored})
    if (ctypes.sizeof(result) != _integer(node.get("size"))
            or ctypes.alignment(result) != _integer(node.get("alignment"))
            or any(getattr(result, field).offset != offset for field, offset in offsets)):
        raise SensorToolError("Orca Sensor tool C layout does not match this Python platform ABI")
    return result


def _restore_contract(document: dict[str, Any]) -> CompiledContract:
    try:
        ctype = _restore_ctype(document["ctype_schema"])
        if (not issubclass(ctype, ctypes.Structure)
                or ctypes.sizeof(ctype) != document["input_size"]
                or ctypes.alignment(ctype) != document["input_alignment"]):
            raise SensorToolError("Orca Sensor tool input layout has inconsistent size or alignment")
        return CompiledContract(
            document=document["document"], fingerprint=document["fingerprint"], ctypes_type=ctype,
            layout_fields=tuple(LayoutField(**field) for field in document["layout_fields"]),
            value_fields=tuple(ValueField(**{**field, "shape": tuple(field["shape"])})
                               for field in document["value_fields"]),
            resolved_fields=document["resolved_fields"], requirements=document["requirements"],
            resolved_objects=document["resolved_objects"], sdk_version=document["sdk_version"],
        )
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise SensorToolError(f"Malformed compiled layout from Orca Sensor tool: {error}") from error


def load_io_contract(source: str | Path | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = request("load_io_contract", {"source": source_value(source)}, error_type=ContractError)
    return result["input_document"], result["output"]


def load_model_objects(source: str | Path, *, input_document: dict[str, Any] | None = None,
                       output: dict[str, Any] | None = None, mount: str = "sensor_base",
                       sdk_version: str | None = None) -> dict[str, Any]:
    """Read provider-local object types; never install geometry or create physics state."""
    try:
        source = Path(source)
        model = {"kind": "model_text", "text": source.read_text(encoding="utf-8"), "filename": source.name}
    except (OSError, UnicodeError) as error:
        raise ContractError(f"cannot read model metadata {source}: {error}") from error
    return request("load_model_objects", {
        "source": model, "input_document": input_document, "output": output,
        "mount": mount, "allow_native_sensors": False, "sdk_version": sdk_version,
    }, error_type=ContractError)


def compile_contract(source: str | Path | dict[str, Any] | CompiledContract, *,
                     objects: dict[str, Any] | None = None, sdk_version: str | None = None) -> CompiledContract:
    if isinstance(source, CompiledContract):
        if sdk_version is not None and sdk_version != source.sdk_version:
            raise ContractError("A compiled contract cannot be reinterpreted using a different sdk_version")
        if objects is None:
            return source
        sdk_version, source = source.sdk_version, source.document
    return _restore_contract(request("compile_contract", {
        "source": source_value(source), "objects": objects, "sdk_version": sdk_version,
    }, error_type=ContractError))
