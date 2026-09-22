"""Locate OrcaGym's bundled sensor runtime without loading it."""

import hashlib
import json
from pathlib import Path
import platform


def runtime_directory() -> Path:
    """Runtime artifacts are independent of the separately distributed vendor SDK."""
    return Path(__file__).resolve().parent.parent / "native"


def default_host_path() -> Path:
    """Return the pinned platform Host, or explain why only plugins are unavailable.

    Explicit trusted paths remain supported by SensorHost and EulerEnv. Normal
    OrcaGym imports and simulations without providers do not call this helper.
    """
    system, machine = platform.system(), platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    system_tag = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}.get(system)
    filename = {"Linux": "liborca_sensor_host.so", "Windows": "orca_sensor_host.dll",
                "Darwin": "liborca_sensor_host.dylib"}.get(system)
    if system_tag is None or machine not in {"x86_64", "aarch64"}:
        raise RuntimeError(f"No bundled sensor Host for {system}/{machine}; core OrcaGym is unaffected")
    root = runtime_directory()
    relative = f"lib/{system_tag}-{machine}/{filename}"
    path, manifest_path = root / relative, root / "native-manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        libc, version = platform.libc_ver()
        try:
            glibc = tuple(int(part) for part in version.split(".")[:2])
        except ValueError:
            glibc = ()
        if ((system_tag, machine) != ("linux", "x86_64")
                or libc != "glibc" or glibc < (2, 35)):
            raise RuntimeError(f"This OrcaGym release does not bundle a sensor Host for "
                               f"{system}/{machine}/{libc} {version}; "
                               "sensor providers require Linux x86_64 with glibc >= 2.35; "
                               "core OrcaGym is unaffected")
        raise RuntimeError("The installed OrcaGym package is missing its bundled sensor Host. "
                           "Upgrade or reinstall the official orca-gym release with sensor support "
                           "on a supported platform; no separate Runtime package or SDK is needed. "
                           "core OrcaGym is unaffected")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("format_version") != 1 or manifest.get("abi_version") != 2
                or manifest.get("sdk_version") != (root / "VERSION").read_text().strip()):
            raise ValueError("Incompatible sensor runtime manifest")
        record = manifest["files"][relative]
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_size != record["size"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError("Sensor Host checksum mismatch")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError("Sensor Host verification failed; reinstall the official orca-gym "
                           "release to restore its bundled runtime") from error
    return path


host_library_path = default_host_path
