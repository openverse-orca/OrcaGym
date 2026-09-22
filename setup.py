"""Bundle the verified sensor Runtime on supported hosts; keep other hosts portable."""

import hashlib
import json
import os
from pathlib import Path
import platform
import sysconfig

from setuptools import Distribution, setup
from setuptools.command.build import build
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.sdist import sdist


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "orca_gym/sensor/native"
MINIMUM_GLIBC = (2, 35)
RUNTIME_LICENSE_FILES = {
    "licenses/ORCA_BINARY_LICENSE.html",
    "licenses/ORCA_BINARY_LICENSE_ADDENDUM.md",
    "licenses/ORCA_BINARY_LICENSE_SOURCE.json",
    "licenses/PUBLIC_SDK_LICENSE.txt",
}


def native_platform_supported():
    libc, version = platform.libc_ver()
    try:
        glibc = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return False
    return (platform.system() == "Linux" and platform.machine() == "x86_64"
            and libc == "glibc" and glibc >= MINIMUM_GLIBC)


NATIVE_MODE = os.environ.get("ORCA_SENSOR_BUNDLE_NATIVE", "auto")
if NATIVE_MODE not in {"auto", "0", "1"}:
    raise ValueError("ORCA_SENSOR_BUNDLE_NATIVE must be auto, 0 (developer-only), or 1")
BUNDLE_NATIVE = NATIVE_MODE == "1" or (NATIVE_MODE == "auto" and native_platform_supported())
SOURCE_SUFFIXES = {".py", ".pyw", ".pyi", ".pyc", ".pyo", ".cpp", ".cc", ".c", ".cxx", ".h", ".hpp", ".pdb"}


def native_files(*, for_sdist=False):
    if not BUNDLE_NATIVE and not for_sdist:
        return []
    if not for_sdist and not native_platform_supported():
        raise RuntimeError("Sensor Runtime requires Linux x86_64 with glibc >= 2.35; "
                           "ORCA_SENSOR_BUNDLE_NATIVE=0 is a developer-only portable build override")
    manifest = json.loads((RUNTIME / "native-manifest.json").read_text())
    if (manifest.get("format_version") != 1 or manifest.get("abi_version") != 2
            or manifest.get("sdk_version") != (RUNTIME / "VERSION").read_text().strip()):
        raise ValueError("Invalid native release manifest")
    required = {"bin/linux-x86_64/orca-sensor-tool", "lib/linux-x86_64/liborca_sensor_host.so"} | RUNTIME_LICENSE_FILES
    if not required.issubset(manifest["files"]):
        raise ValueError("Native release manifest must include the Host, contract tool and license files")
    files = []
    for name, record in manifest["files"].items():
        path = RUNTIME / name
        if (not (name.startswith(("bin/linux-x86_64/", "lib/linux-x86_64/")) or name in RUNTIME_LICENSE_FILES)
                or ".." in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(RUNTIME.resolve())
                or SOURCE_SUFFIXES.intersection(suffix.lower() for suffix in path.suffixes)):
            raise ValueError(f"Invalid native artifact path: {name}")
        if (path.stat().st_size != record["size"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError(f"Native artifact checksum mismatch: {name}")
        files.append(path)
    actual = {path.relative_to(RUNTIME).as_posix()
              for directory in (RUNTIME / "bin", RUNTIME / "lib", RUNTIME / "licenses")
              for path in directory.rglob("*") if path.is_file() or path.is_symlink()}
    if actual != set(manifest["files"]):
        raise ValueError("Native release files do not match the pinned manifest")
    # The complete artifact also pins VERSION and compatibility data. Validate
    # it before building so a matching binary subset cannot hide a mixed release.
    release = json.loads((RUNTIME / "runtime-manifest.json").read_text())
    if (release.get("format_version") != 1 or release.get("kind") != "runtime"
            or release.get("abi_version") != 2 or release.get("sdk_version") != manifest["sdk_version"]):
        raise ValueError("Invalid complete Runtime manifest")
    compatibility = {path.relative_to(RUNTIME).as_posix()
                     for path in (RUNTIME / "compatibility").rglob("*")
                     if path.is_file() or path.is_symlink()}
    expected = set(manifest["files"]) | compatibility | {"VERSION", "native-manifest.json"}
    if set(release["files"]) != expected:
        raise ValueError("Complete Runtime files do not match the pinned manifest")
    for name, record in release["files"].items():
        path = RUNTIME / name
        if (".." in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(RUNTIME.resolve())
                or SOURCE_SUFFIXES.intersection(suffix.lower() for suffix in path.suffixes)):
            raise ValueError(f"Invalid Runtime artifact path: {name}")
        if (path.stat().st_size != record["size"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError(f"Runtime artifact checksum mismatch: {name}")
    return [RUNTIME / name for name in release["files"]] + [RUNTIME / "runtime-manifest.json"]


class SensorSdist(sdist):
    def make_release_tree(self, base_dir, files):
        # A source archive is platform-neutral input. Preserve the verified
        # Runtime even when the publishing machine builds a portable wheel.
        runtime_files = [str(path.relative_to(ROOT)) for path in native_files(for_sdist=True)]
        super().make_release_tree(base_dir, sorted(set(files).union(runtime_files)))


class SensorBuild(build):
    def finalize_options(self):
        super().finalize_options()
        # Separate caches so building a native wheel followed by a portable
        # wheel cannot accidentally retain DLLs in setuptools' build/lib.
        cache = "lib-core"
        if BUNDLE_NATIVE:
            digest = hashlib.sha256((RUNTIME / "runtime-manifest.json").read_bytes()).hexdigest()[:16]
            cache = "lib-sensor-native-" + digest
        self.build_lib = os.path.join(self.build_base, cache)


class SensorBuildPy(build_py):
    def find_data_files(self, package, src_dir):
        files = super().find_data_files(package, src_dir)
        files = [name for name in files if not any(
            part in Path(name).parts for part in ("bin", "lib"))]
        if package == "orca_gym.sensor.native":
            files.extend(str(path) for path in native_files())
        return files


class SensorWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if BUNDLE_NATIVE:
            self.root_is_pure = False

    def get_tag(self):
        if BUNDLE_NATIVE:
            native_files()
            return "py3", "none", sysconfig.get_platform().replace("-", "_").replace(".", "_")
        return super().get_tag()


class SensorDistribution(Distribution):
    def has_ext_modules(self):
        # There is no CPython extension ABI, but package data contains ELF.
        # Tell setuptools to install into platlib, not .data/purelib.
        return BUNDLE_NATIVE or super().has_ext_modules()


setup(cmdclass={"build": SensorBuild, "build_py": SensorBuildPy,
               "bdist_wheel": SensorWheel, "sdist": SensorSdist},
      distclass=SensorDistribution)
