"""ctypes mirror of SDK v2. Only Host is loaded here, never vendor callbacks."""

import ctypes as C
from pathlib import Path


class InputField(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("field_id", "dtype", "offset", "stride", "capacity", "components", "count_offset")]


class InputLayout(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("input_size", C.c_uint32),
                ("field_count", C.c_uint32), ("fields", C.POINTER(InputField))]


class OutputSpec(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("ndim", C.c_uint32),
                ("shape", C.c_uint32 * 4)]


class CreateInfo(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("seed", C.c_uint64),
                ("global_parameter_count", C.c_uint32), ("global_parameters", C.POINTER(C.c_double))]


class TypeDescriptor(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
                ("type_id", C.c_char_p), ("input_contract_id", C.c_char_p),
                ("input_fingerprint", C.c_char_p), ("input_layout", C.POINTER(InputLayout)),
                ("output", OutputSpec), ("global_parameter_count", C.c_uint32),
                *[(name, C.c_void_p) for name in ("create", "reset", "compute", "destroy")]]


class ProviderDescriptor(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
                ("provider_id", C.c_char_p), ("provider_version", C.c_char_p),
                ("type_count", C.c_uint32), ("types", C.POINTER(TypeDescriptor))]


class FieldView(C.Structure):
    _fields_ = [("field_id", C.c_uint32), ("dtype", C.c_uint32),
                ("count", C.c_uint32), ("components", C.c_uint32),
                ("byte_size", C.c_uint64), ("data", C.c_void_p)]


class StepRecord(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("time", C.c_double), ("dt", C.c_double),
                ("step_index", C.c_uint64), ("field_count", C.c_uint32),
                ("fields", C.POINTER(FieldView))]


class SensorError(RuntimeError):
    """Host or provider rejected an operation; native crashes are not recoverable."""


class NativeAPI:
    """Explicit signatures avoid pointer truncation and accidental ABI coercion."""

    def __init__(self, path: str | Path):
        self._library = C.CDLL(str(Path(path).resolve(strict=True)))
        signatures = {
            "abi_version": (C.c_uint32, []),
            "last_error": (C.c_char_p, []),
            "open": (C.c_int32, [C.c_char_p, C.POINTER(C.c_uint64)]),
            "descriptor": (C.POINTER(ProviderDescriptor), [C.c_uint64]),
            "close": (C.c_int32, [C.c_uint64]),
            "create": (C.c_int32, [C.c_uint64, C.c_char_p, C.POINTER(CreateInfo), C.POINTER(C.c_uint64)]),
            "destroy": (C.c_int32, [C.c_uint64]),
            "reset": (C.c_int32, [C.c_uint64, C.c_uint64]),
            "process": (C.c_int32, [C.c_uint64, C.POINTER(StepRecord)]),
            "publish_batch": (C.c_int32, [C.POINTER(C.c_uint64), C.c_uint32]),
            "copy_output": (C.c_int32, [C.c_uint64, C.POINTER(C.c_double), C.c_uint64]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self._library, "orca_sensor_host_" + name)
            function.restype = result
            function.argtypes = arguments
            setattr(self, name, function)
        if self.abi_version() != 2:
            raise SensorError("Unsupported Python-to-Host ABI (expected v2); install the matching Orca Host. "
                              "Vendor SDK compatibility is checked separately.")

    def check(self, status: int) -> None:
        if status:
            message = self.last_error()
            raise SensorError(f"NativeHost status {status}: " +
                              (message.decode("utf-8", errors="replace") if message else "unknown error"))
