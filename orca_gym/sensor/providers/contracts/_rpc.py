"""Lazy, value-only client for the sensor tool bundled with OrcaGym.

Only registration and preparation use this interface. Physics steps do not invoke
the tool. The executable is trusted native code, just like the configured Host;
ORCA_SENSOR_TOOL is an explicit developer override, never vendor manifest data.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
from typing import Any


class SensorToolError(RuntimeError):
    """The trusted tool is unavailable or returned an invalid protocol reply."""


class ContractError(ValueError):
    """The contract cannot be represented by the input ABI."""


class XmlModelError(ValueError):
    """The model cannot satisfy the sensor's declared object requirements."""


class SDKCompatibilityError(ValueError):
    """The requested SDK profile is unavailable or inconsistent."""


RUNTIME_ROOT = Path(__file__).resolve().parents[2] / "native"
MAX_MESSAGE_BYTES = 32 * 1024 * 1024
ERROR_TYPES = {error.__name__: error for error in
               (ContractError, XmlModelError, SDKCompatibilityError, ValueError)}


def tool_path() -> Path:
    """Resolve a trusted executable on first use, never during package import."""
    override = os.environ.get("ORCA_SENSOR_TOOL")
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    system = {"linux": "linux", "win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform)
    path = (Path(override).expanduser() if override else
            RUNTIME_ROOT / "bin" / f"{system}-{machine}" /
            ("orca-sensor-tool.exe" if sys.platform == "win32" else "orca-sensor-tool"))
    if not path.is_absolute():
        raise SensorToolError("ORCA_SENSOR_TOOL must be an absolute path to a trusted Orca Sensor tool executable")
    if not path.is_file() or (os.name != "nt" and not os.access(path, os.X_OK)):
        if override:
            raise SensorToolError(f"The developer override ORCA_SENSOR_TOOL is not an executable file: {path}")
        libc, version = platform.libc_ver()
        try:
            glibc = tuple(int(part) for part in version.split(".")[:2])
        except ValueError:
            glibc = ()
        if ((system, machine) != ("linux", "x86_64")
                or libc != "glibc" or glibc < (2, 35)):
            raise SensorToolError(
                f"This OrcaGym release does not bundle sensor provider tooling for {system}-{machine}; "
                "sensor providers require Linux x86_64 with glibc >= 2.35; "
                "ordinary OrcaGym use is unaffected."
            )
        raise SensorToolError(
            f"The installed OrcaGym package is missing its bundled sensor tool for {system}-{machine}. "
            "Upgrade or reinstall the official orca-gym release with sensor support on a supported "
            "platform; no separate Runtime package or SDK is needed. "
            "ordinary OrcaGym use is unaffected."
        )
    if not override:
        manifest = RUNTIME_ROOT / "native-manifest.json"
        try:
            tool_stat, manifest_stat = path.stat(), manifest.stat()
            _verify_bundle(str(manifest), (manifest_stat.st_mtime_ns, manifest_stat.st_size,
                                         tool_stat.st_mtime_ns, tool_stat.st_size), f"{system}-{machine}")
        except OSError as error:
            raise SensorToolError(f"Orca Sensor binary manifest is unavailable: {error}") from error
    return path.resolve()


@lru_cache(maxsize=8)
def _verify_bundle(manifest_path: str, identity: tuple[int, ...], target: str) -> None:
    """Verify the frozen tool and its complete runtime directory before launch."""
    del identity
    root = Path(manifest_path).parent.resolve()
    prefix = f"bin/{target}/"
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or type(manifest.get("format_version")) is not int
                or manifest["format_version"] != 1 or type(manifest.get("abi_version")) is not int
                or manifest["abi_version"] != 2
                or manifest.get("sdk_version") != (root / "VERSION").read_text(encoding="utf-8").strip()
                or not isinstance(manifest.get("files"), dict)):
            raise ValueError("unsupported binary manifest or mismatched SDK release")
        expected = set()
        for name, entry in manifest["files"].items():
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError("binary manifest contains an unsafe path")
            path = root.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(root / "bin" / target):
                raise ValueError("binary manifest file escapes its runtime directory")
            if (not isinstance(entry, dict) or type(entry.get("size")) is not int
                    or entry["size"] < 0 or not isinstance(entry.get("sha256"), str)
                    or len(entry["sha256"]) != 64 or not path.is_file()
                    or path.stat().st_size != entry["size"]):
                raise ValueError(f"invalid or missing binary file: {name}")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"binary checksum mismatch: {name}")
            expected.add(name)
        actual = {path.relative_to(root).as_posix() for path in (root / "bin" / target).rglob("*")
                  if path.is_file()}
        if not expected or actual != expected:
            raise ValueError("runtime directory differs from its pinned binary manifest")
    except (OSError, ValueError, TypeError) as error:
        raise SensorToolError(f"Orca Sensor binary verification failed: {error}") from error


def source_value(source: str | Path | dict[str, Any], *, error_type=ContractError) -> dict[str, Any]:
    """Send unparsed file text so duplicate keys and nonfinite JSON stay visible."""
    if isinstance(source, (str, Path)):
        try:
            return {"kind": "json_text", "text": Path(source).read_text(encoding="utf-8")}
        except (OSError, UnicodeError) as error:
            raise error_type(f"cannot read contract {source}: {error}") from error
    return {"kind": "document", "document": source}


@lru_cache(maxsize=256)
def _exchange(executable: str, identity: tuple[int, int], payload: str) -> str:
    # Cache immutable replies by complete request content, not a pathname that
    # can later contain different data. Decoding below gives callers fresh DTOs.
    del identity
    try:
        completed = subprocess.run(
            [executable, "rpc"], input=payload + "\n", encoding="utf-8",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
        raise SensorToolError(f"Orca Sensor tool could not complete the request: {error}") from error
    if completed.returncode:
        raise SensorToolError(f"Orca Sensor tool exited with status {completed.returncode}")
    if len(completed.stdout.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise SensorToolError("Orca Sensor tool reply exceeds the protocol limit")
    try:
        reply = json.loads(completed.stdout, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    except (ValueError, TypeError) as error:
        raise SensorToolError("Orca Sensor tool returned an invalid JSON reply") from error
    if (not isinstance(reply, dict) or type(reply.get("protocol")) is not int
            or reply["protocol"] != 1 or type(reply.get("id")) is not int or reply["id"] != 1
            or type(reply.get("ok")) is not bool):
        raise SensorToolError("Orca Sensor tool protocol version or request identity does not match")
    if not reply["ok"]:
        error = reply.get("error", {})
        if (not isinstance(error, dict) or not isinstance(error.get("message"), str)
                or not isinstance(error.get("kind"), str)):
            raise SensorToolError("Orca Sensor tool returned a malformed error")
        raise ERROR_TYPES.get(error.get("kind"), SensorToolError)(error["message"])
    if "result" not in reply:
        raise SensorToolError("Orca Sensor tool reply is missing its result")
    return json.dumps(reply["result"], allow_nan=False)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate protocol reply key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite protocol reply value: {value}")


def request(method: str, params: dict[str, Any] | None = None, *, error_type=ValueError) -> Any:
    try:
        _check_json(params)
        payload = json.dumps({"protocol": 1, "id": 1, "method": method, "params": params or {}},
                             sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise error_type(f"Request must contain finite JSON values: {error}") from error
    if len(payload.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise error_type("Orca Sensor tool request exceeds the protocol limit")
    executable = tool_path()
    stat = executable.stat()
    return json.loads(_exchange(str(executable), (stat.st_mtime_ns, stat.st_size), payload))


def _check_json(value: Any) -> None:
    # Transport must not turn tuples into lists or numeric keys into strings;
    # that would hide invalid authoring values from the canonical validator.
    if value is None or type(value) in (bool, int, float, str):
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _check_json(item)
        return
    raise ValueError(f"Non-JSON request value: {type(value).__name__}")
