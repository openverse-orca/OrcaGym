"""Backend-neutral Python orchestration of the vendor C ABI.

Adapters provide typed fields, NOT backend pointers. NativeHost knows their
layout, not their physical meaning, and assembles preallocated vendor structs.
"""

from __future__ import annotations

import ctypes as C
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import threading
from typing import Mapping

import numpy as np

from .contracts.contract import compile_contract, load_io_contract, load_model_objects
from .contracts.version import load_profile, resolve_sdk_version
from .abi import (FieldView, CreateInfo, InputLayout, NativeAPI, OutputSpec,
                  ProviderDescriptor, SensorError, StepRecord, TypeDescriptor)


DTYPES = {"f64": np.dtype("float64"), "f32": np.dtype("float32"),
          "u32": np.dtype("uint32"), "u64": np.dtype("uint64"), "u8": np.dtype("uint8")}
DTYPE_IDS = {"f64": 1, "f32": 2, "u32": 3, "u64": 4, "u8": 5}
PARAMETERS_UNSET = object()


def resolve_global_parameters(global_parameters=PARAMETERS_UNSET, *, parameters=PARAMETERS_UNSET) -> dict:
    """Copy per-instance create-time configuration, accepting the legacy spelling."""
    if global_parameters is not PARAMETERS_UNSET and parameters is not PARAMETERS_UNSET:
        raise ValueError("Use global_parameters, not both global_parameters and legacy parameters")
    values = parameters if global_parameters is PARAMETERS_UNSET else global_parameters
    if values is PARAMETERS_UNSET or values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("global_parameters must be a name-to-value mapping")
    return dict(values)


@dataclass(frozen=True)
class StepInput:
    """Coherent source-state sample. time is not necessarily integrated sim time."""

    time: float
    dt: float
    step_index: int
    fields: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorInstance:
    """Named instance, not a function. Its opaque native handle belongs to Host."""

    host: SensorHost
    instance_id: str
    generation: int

    def compute(self, sample: StepInput) -> None:
        self.host.compute(self.instance_id, sample, generation=self.generation)

    def read(self) -> np.ndarray:
        return self.host.read(self.instance_id, generation=self.generation)

    def reset(self, seed: int = 0) -> None:
        self.host.reset_sensor(self.instance_id, seed, generation=self.generation)

    def close(self) -> None:
        self.host.close_sensor(self.instance_id, generation=self.generation)


def _integer(value, bits: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**bits:
        raise ValueError(f"{label} must be an unsigned {bits}-bit integer")
    return value


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return float(value)


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value or len(value.encode("utf-8")) > 512:
        raise ValueError(f"Invalid {label}")
    return value


def _identifier(value, label: str) -> str:
    value = _text(value, label)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)+", value):
        raise ValueError(f"{label} must be a namespaced identifier")
    return value


def _keys(value, required: set, optional: set = frozenset()) -> None:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError(f"Expected keys {sorted(required)}, optional {sorted(optional)}")


def _package_file(root: Path, relative: str) -> Path:
    _text(relative, "package path")
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Provider paths must be relative to the package")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Provider path escapes the package or is not a file")
    return resolved


def _validate_channels(output: dict, objects: Mapping) -> None:
    """Output meaning belongs to the contract, not to its optional presentation."""
    if "channels" not in output:
        return
    channels = output["channels"]
    if not isinstance(channels, list) or not 1 <= len(channels) <= 256:
        raise ValueError("Output channels must be a nonempty list of at most 256 entries")
    count = math.prod(output["shape"])
    occupied, names = set(), set()
    for channel in channels:
        _keys(channel, {"name", "offset", "unit", "description"}, {"shape", "objects"})
        name = _text(channel["name"], "channel name")
        if name in names:
            raise ValueError(f"Duplicate output channel: {name}")
        names.add(name)
        _text(channel["unit"], "channel unit")
        _text(channel["description"], "channel description")
        start = _integer(channel["offset"], 32, "channel offset")
        shape = channel.get("shape", [])
        if (not isinstance(shape, list) or len(shape) > 4 or
                any(type(dim) is not int or not 1 <= dim <= 1048576 for dim in shape)):
            raise ValueError(f"Invalid output channel shape: {name}")
        size = math.prod(shape)
        if start + size > count:
            raise ValueError(f"Output channel exceeds output shape: {name}")
        indices = set(range(start, start + size))
        if occupied & indices:
            raise ValueError(f"Overlapping output channels: {name}")
        occupied.update(indices)
        if "objects" in channel:
            references = channel["objects"]
            if (not isinstance(references, list) or len(references) != size
                    or any(not isinstance(name, str) for name in references)
                    or len(set(references)) != len(references)
                    or any(name not in objects for name in references)):
                raise ValueError(f"Output channel objects must match its elements and model metadata: {name}")
    if len(occupied) != count:
        raise ValueError("Output channels must cover the entire output without gaps")


def _load_manifest(path: Path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate manifest key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite manifest constant: {value}")

    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                          parse_constant=invalid_constant)
    _keys(document, {"abi_version", "provider_id", "version", "library", "types"}, {"sdk_version"})
    if "sdk_version" in document:
        _text(document["sdk_version"], "sdk_version")
    sdk_version = resolve_sdk_version(document.get("sdk_version"), legacy="sdk_version" not in document)
    profile = load_profile(sdk_version)
    if type(document["abi_version"]) is not int or document["abi_version"] != profile["abi_version"]:
        raise ValueError(f"SDK {sdk_version} requires provider ABI {profile['abi_version']}; unsupported provider ABI")
    _identifier(document["provider_id"], "provider_id")
    _text(document["version"], "version")
    libraries = document["library"]
    if not isinstance(libraries, dict) or not libraries or libraries.keys() - {"linux", "windows", "darwin"}:
        raise ValueError("library must map linux/windows/darwin to package-relative files")
    for library_path in libraries.values():
        _text(library_path, "library path")
    platform = {"win32": "windows"}.get(sys.platform, sys.platform)
    if platform not in libraries:
        raise ValueError(f"Provider has no binary for {platform}")
    library = _package_file(path.parent, libraries[platform])
    if not isinstance(document["types"], list) or not 1 <= len(document["types"]) <= 256:
        raise ValueError("Provider must declare 1..256 types")
    types = {}
    for entry in document["types"]:
        _keys(entry, {"type_id", "display_name"},
              {"global_parameters", "parameters", "contract_schema", "input_schema", "output",
               "model", "model_schema", "presentation_schema"})
        if ("global_parameters" in entry) == ("parameters" in entry):
            raise ValueError("Declare global_parameters exactly once; do not mix the legacy parameters field")
        entry = dict(entry)
        if "parameters" in entry:
            entry["global_parameters"] = entry.pop("parameters")
        # Model assets are only a source of semantic types for object contracts.
        # Geometry installation and presentation always belong to the application.
        model_config = entry.pop("model", None)
        for metadata in ("model_schema", "presentation_schema"):
            entry.pop(metadata, None)
        if "contract_schema" in entry:
            if {"input_schema", "output"} & entry.keys():
                raise ValueError("contract_schema cannot be mixed with input_schema or inline output")
        elif not {"input_schema", "output"} <= entry.keys():
            raise ValueError("A sensor type requires contract_schema or legacy input_schema/output")
        type_id = _identifier(entry["type_id"], "type_id")
        if type_id in types:
            raise ValueError(f"Duplicate type_id: {type_id}")
        _text(entry["display_name"], "display_name")
        if "contract_schema" in entry:
            input_document, output = load_io_contract(_package_file(path.parent, entry["contract_schema"]))
        else:
            input_document = json.loads(_package_file(path.parent, entry["input_schema"]).read_text(encoding="utf-8"),
                                       object_pairs_hook=unique_object, parse_constant=invalid_constant)
            if isinstance(input_document, dict) and "output" in input_document:
                raise ValueError("An input_schema must not also define output; use contract_schema instead")
            output = entry["output"]
        version = input_document.get("schema_version", 2) if isinstance(input_document, dict) else None
        objects = {}
        if version == 4:
            inputs = input_document.get("inputs")
            needs_objects = isinstance(inputs, list) and any(
                item.get("objects") or item.get("object")
                or item.get("frame", "world") != "world"
                for item in inputs if isinstance(item, dict))
            if needs_objects:
                if (not isinstance(model_config, dict)
                        or model_config.keys() not in ({"asset", "mount_site"}, {"asset", "mount"})):
                    raise ValueError("Object input contracts require provider model metadata with asset and mount_site")
                mount = _text(model_config.get("mount_site", model_config.get("mount")), "model mount_site")
                objects = load_model_objects(_package_file(path.parent, model_config["asset"]),
                                             input_document=input_document, output=output, mount=mount,
                                             sdk_version=sdk_version)
        contract = compile_contract(input_document, objects=objects if version == 4 else None,
                                    sdk_version=sdk_version)
        if version != 4:
            objects = contract.objects
        _keys(output, {"dtype", "shape"}, {"unit", "channels"})
        if "unit" in output:
            _text(output["unit"], "output unit")
        if output["dtype"] != "float64":
            raise ValueError("SDK v2 output must be float64")
        shape = output["shape"]
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 4 or
                any(type(dim) is not int or not 1 <= dim <= 1048576 for dim in shape) or
                math.prod(shape) > 1048576):
            raise ValueError("Invalid output shape (at most 4 dimensions / 1048576 values)")
        if version == 4 and "channels" not in output:
            raise ValueError("A schema 4 type must describe its named output channels")
        _validate_channels(output, objects)
        global_parameters = entry["global_parameters"]
        if not isinstance(global_parameters, list) or len(global_parameters) > 256:
            raise ValueError("global_parameters must contain at most 256 ordered numeric declarations")
        names = set()
        for parameter in global_parameters:
            _keys(parameter, {"name", "default", "min", "max"})
            name = _text(parameter["name"], "parameter name")
            if name in names:
                raise ValueError(f"Duplicate parameter: {name}")
            names.add(name)
            lower, default, upper = (_finite(parameter[k], k) for k in ("min", "default", "max"))
            if not lower <= default <= upper:
                raise ValueError(f"Invalid bounds/default for parameter {name}")
        types[type_id] = {"manifest": entry, "contract": contract,
                          "shape": tuple(shape), "output": output}
    return document, library, types


def _decode(value) -> str:
    if not value:
        raise ValueError("Missing descriptor string")
    return value.decode("utf-8")


def encode_fields(contract, values: Mapping[str, object]):
    """Make borrowed contiguous columns; retain returned arrays through the C call.

    NumPy arrays must have the declared dtype; Python literals are range-checked
    before conversion. No lossy integer casts or silent array truncation.
    """
    if not isinstance(values, Mapping) or values.keys() != {f.path for f in contract.value_fields}:
        raise ValueError("Input fields must exactly match the vendor contract paths")
    arrays = []
    views = (FieldView * len(contract.value_fields))()
    counts = {}
    for index, spec in enumerate(contract.value_fields):
        value = values[spec.path]
        dtype = DTYPES[spec.dtype]
        if isinstance(value, np.ndarray):
            if value.dtype != dtype:
                raise ValueError(f"{spec.path}: expected dtype {dtype}, received {value.dtype}")
            array = value
        else:
            original = np.asarray(value)
            if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
                raise ValueError(f"{spec.path}: numeric values required")
            if dtype.kind == "u" and original.size:
                limit = np.iinfo(dtype).max
                for item in original.flat:
                    number = item.item()
                    if not isinstance(number, int) or not 0 <= number <= limit:
                        raise ValueError(f"{spec.path}: expected unsigned integers in range")
            with np.errstate(over="ignore", invalid="ignore"):
                array = original.astype(dtype)
        tail = tuple(spec.shape)
        if spec.variable:
            if array.ndim != len(tail) + 1 or array.shape[1:] != tail:
                raise ValueError(f"{spec.path}: expected shape (N, {tail})")
            count = array.shape[0]
            if count > spec.capacity:
                raise SensorError(f"{spec.path}: capacity exceeded ({count} > {spec.capacity}); no truncation")
            group = spec.path.split(".", 1)[0]
            if counts.setdefault(group, count) != count:
                raise ValueError(f"{group}: all columns must have the same record count")
        else:
            if array.shape != tail:
                raise ValueError(f"{spec.path}: expected shape {tail}, received {array.shape}")
            count = 1
        if dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError(f"{spec.path}: nonfinite input")
        array = np.ascontiguousarray(array)
        arrays.append(array)
        views[index] = FieldView(spec.field_id, DTYPE_IDS[spec.dtype], count,
                                 math.prod(tail), array.nbytes, array.ctypes.data)
    return views, arrays


class SensorHost:
    """Serialized, explicit provider registry and deterministic instance lifecycle.

    Call compute for each sensor at every physics substep, then publish once per
    successful environment step. Any compute/publish failure requires host.reset.
    Only trusted DLLs: neither ctypes nor C++ catch protects against native crashes.
    """

    def __init__(self, library: str | Path):
        self._api = NativeAPI(library)
        self._lock = threading.RLock()
        self._libraries = {}
        self._types = {}
        self._instances = {}
        self._closed = False
        self._faulted = False

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _check_open(self, *, ready=False):
        if self._closed:
            raise SensorError("SensorHost is closed")
        if ready and self._faulted:
            raise SensorError("SensorHost is faulted; call host.reset() before continuing")

    def _instance(self, instance_id, generation=None):
        state = self._instances[instance_id]
        if generation is not None and state["handle"] != generation:
            raise SensorError(f"SensorInstance {instance_id!r} was closed and replaced")
        return state

    def register_provider(self, manifest: str | Path) -> list[str]:
        """Load an explicitly trusted package and register all types atomically."""
        with self._lock:
            self._check_open(ready=True)
            document, path, types = _load_manifest(Path(manifest).resolve(strict=True))
            provider_id = document["provider_id"]
            if provider_id in self._libraries or types.keys() & self._types.keys():
                raise ValueError("Provider or type_id is already registered")
            library = C.c_uint64()
            try:
                self._api.check(self._api.open(str(path).encode("utf-8"), C.byref(library)))
                pointer = self._api.descriptor(library.value)
                if not pointer:
                    raise SensorError("Host returned no provider descriptor")
                descriptor = pointer.contents
                if (descriptor.struct_size != C.sizeof(ProviderDescriptor) or descriptor.abi_version != 2 or
                        _decode(descriptor.provider_id) != provider_id or
                        _decode(descriptor.provider_version) != document["version"] or
                        descriptor.type_count != len(types)):
                    raise ValueError("Provider descriptor does not match manifest/ABI")
                seen = set()
                for index in range(descriptor.type_count):
                    entry = descriptor.types[index]
                    type_id = _decode(entry.type_id)
                    if type_id not in types or type_id in seen:
                        raise ValueError("DLL type catalog differs from manifest")
                    seen.add(type_id)
                    compiled = types[type_id]["contract"]
                    if (entry.struct_size != C.sizeof(TypeDescriptor) or entry.abi_version != 2 or
                            _decode(entry.input_contract_id) != compiled.contract_id or
                            _decode(entry.input_fingerprint) != compiled.fingerprint):
                        raise ValueError(f"Input contract/ABI mismatch: {type_id}")
                    layout = entry.input_layout.contents
                    actual = [(f.field_id, f.dtype, f.offset, f.stride, f.capacity, f.components, f.count_offset)
                              for f in layout.fields[:layout.field_count]]
                    expected = [(f.field_id, f.dtype, f.offset, f.stride, f.capacity, f.components, f.count_offset)
                                for f in compiled.layout_fields]
                    if (layout.struct_size != C.sizeof(InputLayout) or
                            layout.input_size != compiled.size or actual != expected):
                        raise ValueError(f"Compiled input layout mismatch: {type_id}")
                    if (entry.output.struct_size != C.sizeof(OutputSpec) or
                            tuple(entry.output.shape[:entry.output.ndim]) != types[type_id]["shape"] or
                            entry.global_parameter_count != len(types[type_id]["manifest"]["global_parameters"])):
                        raise ValueError(f"Output/parameter descriptor mismatch: {type_id}")
                    types[type_id]["library"] = library.value
                self._libraries[provider_id] = library.value
                self._types.update(types)
            except BaseException:
                self._libraries.pop(provider_id, None)
                for type_id in types:
                    self._types.pop(type_id, None)
                if library.value:
                    self._api.close(library.value)
                raise
            return list(types)

    def contract_for(self, type_id: str):
        """Return a defensive contract copy for declarative query planning."""
        with self._lock:
            self._check_open()
            contract = self._types[type_id]["contract"]
            return compile_contract(contract.document, objects=contract.objects if contract.schema_version == 4 else None,
                                    sdk_version=contract.sdk_version)

    def output_for(self, type_id: str) -> dict:
        """Copy the named output contract; it remains available without a viewer."""
        with self._lock:
            self._check_open()
            return copy.deepcopy(self._types[type_id]["output"])

    def invalidate(self) -> None:
        """Sampling failure invalidates the logical step until reset."""
        with self._lock:
            self._check_open()
            self._faulted = True

    def create_sensor(self, instance_id: str, type_id: str, *,
                      global_parameters=PARAMETERS_UNSET, seed: int = 0,
                      parameters=PARAMETERS_UNSET) -> SensorInstance:
        with self._lock:
            self._check_open(ready=True)
            _text(instance_id, "instance_id")
            _integer(seed, 64, "seed")
            if instance_id in self._instances:
                raise ValueError(f"Duplicate sensor instance: {instance_id}")
            spec = self._types[type_id]
            declared = spec["manifest"]["global_parameters"]
            supplied = resolve_global_parameters(global_parameters, parameters=parameters)
            if supplied.keys() - {p["name"] for p in declared}:
                raise ValueError("Unknown sensor parameter")
            values = []
            for parameter in declared:
                value = _finite(supplied.get(parameter["name"], parameter["default"]), parameter["name"])
                if not parameter["min"] <= value <= parameter["max"]:
                    raise ValueError(f"Parameter out of range: {parameter['name']}")
                values.append(value)
            data = (C.c_double * len(values))(*values)
            info = CreateInfo(C.sizeof(CreateInfo), seed, len(values), data)
            handle = C.c_uint64()
            try:
                self._api.check(self._api.create(spec["library"], type_id.encode(), C.byref(info), C.byref(handle)))
                self._instances[instance_id] = {"handle": handle.value, "spec": spec, "stamp": None,
                                                "declared_seed": seed}
                return SensorInstance(self, instance_id, handle.value)
            except BaseException:
                self._instances.pop(instance_id, None)
                if handle.value:
                    self._api.destroy(handle.value)
                raise

    def compute(self, instance_id: str, sample: StepInput, *, generation: int | None = None) -> None:
        with self._lock:
            self._check_open(ready=True)
            state = self._instance(instance_id, generation)
            try:
                time = _finite(sample.time, "time")
                dt = _finite(sample.dt, "dt")
                if dt <= 0:
                    raise ValueError("dt must be positive")
                index = _integer(sample.step_index, 64, "step_index")
                views, arrays = encode_fields(state["spec"]["contract"], sample.fields)
                step = StepRecord(C.sizeof(StepRecord), time, dt, index, len(views), views)
                self._api.check(self._api.process(state["handle"], C.byref(step)))
                # Keep the borrowed NumPy buffers alive until the synchronous call returns.
                del arrays
                state["stamp"] = (index, time, dt)
            except BaseException:
                self._faulted = True
                raise

    def publish(self) -> None:
        """Publish all live sensors together, using their last staged substep."""
        with self._lock:
            self._check_open(ready=True)
            handles = (C.c_uint64 * len(self._instances))(*[s["handle"] for s in self._instances.values()])
            try:
                stamps = {state["stamp"] for state in self._instances.values()}
                if self._instances and (None in stamps or len(stamps) != 1):
                    raise SensorError("Every sensor must have a fresh sample from the same source step")
                self._api.check(self._api.publish_batch(handles, len(handles)))
                for state in self._instances.values():
                    state["stamp"] = None
            except BaseException:
                self._faulted = True
                raise

    def read(self, instance_id: str, *, generation: int | None = None) -> np.ndarray:
        """Return a fresh array: no pointer lifetime or next-step overwrite hazard."""
        with self._lock:
            self._check_open(ready=True)
            state = self._instance(instance_id, generation)
            result = np.empty(state["spec"]["shape"], dtype=np.float64)
            self._api.check(self._api.copy_output(state["handle"], result.ctypes.data_as(C.POINTER(C.c_double)), result.size))
            return result

    def process_step(self, samples: Mapping[str, StepInput]) -> dict[str, np.ndarray]:
        """Convenience for a single physics substep, followed by atomic publication."""
        with self._lock:
            self._check_open(ready=True)
            if samples.keys() != self._instances.keys():
                raise ValueError("process_step requires exactly one sample for each live instance")
            try:
                for instance_id, sample in samples.items():
                    self.compute(instance_id, sample)
                self.publish()
                return {instance_id: self.read(instance_id) for instance_id in samples}
            except BaseException:
                self._faulted = True
                raise

    def reset_sensor(self, instance_id: str, seed: int = 0, *, generation: int | None = None) -> None:
        with self._lock:
            self._check_open(ready=True)
            _integer(seed, 64, "seed")
            state = self._instance(instance_id, generation)
            try:
                self._api.check(self._api.reset(state["handle"], seed))
                state["stamp"] = None
            except BaseException:
                self._faulted = True
                raise

    def reset(self, seed: int = 0) -> None:
        """Reset the whole runtime; stable name-derived seeds ignore creation order."""
        with self._lock:
            self._check_open()
            _integer(seed, 64, "seed")
            self._faulted = True
            for name, state in self._instances.items():
                seed_material = seed.to_bytes(8, "little") + name.encode("utf-8")
                if state["declared_seed"]:
                    seed_material += state["declared_seed"].to_bytes(8, "little")
                digest = hashlib.sha256(seed_material).digest()
                instance_seed = int.from_bytes(digest[:8], "little")
                self._api.check(self._api.reset(state["handle"], instance_seed))
                state["stamp"] = None
            self._faulted = False

    def close_sensor(self, instance_id: str, *, generation: int | None = None) -> None:
        with self._lock:
            self._check_open()
            state = self._instances.get(instance_id)
            if state is not None:
                self._instance(instance_id, generation)
                # Native destroy consumes the handle even if a broken callback throws.
                del self._instances[instance_id]
                self._api.check(self._api.destroy(state["handle"]))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            errors = []
            for state in self._instances.values():
                try:
                    self._api.check(self._api.destroy(state["handle"]))
                except SensorError as error:
                    errors.append(error)
            for library in self._libraries.values():
                try:
                    self._api.check(self._api.close(library))
                except SensorError as error:
                    errors.append(error)
            self._instances.clear()
            self._libraries.clear()
            self._types.clear()
            self._closed = True
            if errors:
                raise errors[0]
