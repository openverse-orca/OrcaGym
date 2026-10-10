"""Exercise the built wheel, including the sdist-to-wheel release path."""

import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import zipfile

import pytest
import setuptools


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = "orca_gym/sensor/native/"
TOOL = RUNTIME + "bin/linux-x86_64/orca-sensor-tool"
HOST = RUNTIME + "lib/linux-x86_64/liborca_sensor_host.so"


def run_build(command, directory, env):
    result = subprocess.run(command, cwd=directory, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def release_wheel(tmp_path_factory):
    project = tmp_path_factory.mktemp("universal-wheel")
    for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, project / name)
    shutil.copytree(ROOT / "orca_gym", project / "orca_gym",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env = dict(os.environ)
    env.pop("ORCA_SENSOR_BUNDLE_NATIVE", None)
    # The default build creates an sdist, then builds the wheel from that sdist.
    run_build([sys.executable, "-m", "build", "--no-isolation"], project, env)
    wheels = list((project / "dist").glob("*.whl"))
    assert len(wheels) == 1
    assert len(list((project / "dist").glob("*.tar.gz"))) == 1
    return wheels[0]


def test_release_wheel_is_universal_and_contains_complete_runtime(release_wheel):
    assert release_wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(release_wheel) as archive:
        names = archive.namelist()
        metadata = archive.read(next(name for name in names if name.endswith(".dist-info/WHEEL"))).decode()
        assert "Root-Is-Purelib: true\n" in metadata
        assert [line for line in metadata.splitlines() if line.startswith("Tag:")] == ["Tag: py3-none-any"]
        manifest = json.loads(archive.read(RUNTIME + "runtime-manifest.json"))
        assert HOST in names
        assert TOOL in names
        assert archive.getinfo(TOOL).external_attr >> 16 & 0o111
        for name, record in manifest["files"].items():
            payload = archive.read(RUNTIME + name)
            assert len(payload) == record["size"], name
            assert hashlib.sha256(payload).hexdigest() == record["sha256"], name
        binary_names = {name[len(RUNTIME):] for name in names
                        if name.startswith((RUNTIME + "bin/", RUNTIME + "lib/"))}
        assert binary_names == {name for name in manifest["files"] if name.startswith(("bin/", "lib/"))}


@pytest.mark.parametrize("mode", [None, "auto", "1"])
@pytest.mark.parametrize("system,machine", [("Windows", "AMD64"), ("Darwin", "arm64")])
def test_build_host_does_not_control_runtime_inclusion(monkeypatch, mode, system, machine):
    # Inspect the packaging hook without creating files in the source checkout.
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(platform, "libc_ver", lambda: ("", ""))
    if mode is None:
        monkeypatch.delenv("ORCA_SENSOR_BUNDLE_NATIVE", raising=False)
    else:
        monkeypatch.setenv("ORCA_SENSOR_BUNDLE_NATIVE", mode)
    config = runpy.run_path(str(ROOT / "setup.py"))
    files = {path.relative_to(ROOT).as_posix() for path in config["native_files"]()}
    assert {TOOL, HOST, RUNTIME + "runtime-manifest.json"} <= files


def test_developer_opt_out_keeps_runtime_in_sdist(monkeypatch):
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    monkeypatch.setenv("ORCA_SENSOR_BUNDLE_NATIVE", "0")
    config = runpy.run_path(str(ROOT / "setup.py"))
    assert config["native_files"]() == []
    files = {path.relative_to(ROOT).as_posix() for path in config["native_files"](for_sdist=True)}
    assert {TOOL, HOST} <= files


def test_installed_wheel_imports_and_uses_bundled_runtime(release_wheel, tmp_path):
    target = tmp_path / "installed"
    env = dict(os.environ)
    env.pop("ORCA_SENSOR_TOOL", None)
    run_build([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
               "--target", str(target), str(release_wheel)], tmp_path, env)
    # -I and an explicit target keep imports away from the source checkout.
    script = '''
import ctypes
import pathlib
import platform
import subprocess
import sys
sys.path.insert(0, sys.argv[1])
load_library, run_process = ctypes.CDLL, subprocess.run
def guarded_library(name, *args, **kwargs):
    assert "orca_sensor_host" not in str(name), "Import loaded the optional Sensor Host"
    return load_library(name, *args, **kwargs)
def guarded_process(command, *args, **kwargs):
    assert "orca-sensor-tool" not in str(command), "Import launched the optional Sensor tool"
    return run_process(command, *args, **kwargs)
ctypes.CDLL, subprocess.run = guarded_library, guarded_process
import orca_gym
from orca_gym.sensor.providers import native
from orca_gym.sensor.providers.abi import NativeAPI
from orca_gym.sensor.providers.contracts import _rpc
ctypes.CDLL, subprocess.run = load_library, run_process
assert pathlib.Path(orca_gym.__file__).is_relative_to(sys.argv[1])
libc, version = platform.libc_ver()
supported = (platform.system() == "Linux" and platform.machine() == "x86_64"
             and libc == "glibc" and tuple(map(int, version.split(".")[:2])) >= (2, 35))
if supported:
    assert NativeAPI(native.default_host_path()).abi_version() == 2
    assert _rpc.request("resolve_sdk_version", {"value": None, "legacy": False}) == "1.0.0"
else:
    for resolve in (native.default_host_path, _rpc.tool_path):
        try:
            resolve()
        except RuntimeError as error:
            assert "unaffected" in str(error)
        else:
            raise AssertionError("Unsupported platform accepted the bundled Sensor runtime")
'''
    run_build([sys.executable, "-I", "-c", script, str(target)], tmp_path, env)
