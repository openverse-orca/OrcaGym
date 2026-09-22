"""Bundled runtime resolution stays lazy and reports installation errors."""

from pathlib import Path
import platform
from types import SimpleNamespace

import pytest


def test_plain_simulation_does_not_launch_tool_or_load_provider_host(tmp_path, monkeypatch):
    # MuJoCo itself has native dependencies. The guard is installed after its
    # ordinary import and only prohibits additional provider native loading.
    from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore
    from orca_gym.sensor.providers import abi
    from orca_gym.sensor.providers.contracts import _rpc

    def forbidden(*args, **kwargs):
        raise AssertionError("Core simulation touched optional sensor binaries")

    monkeypatch.setattr(_rpc.subprocess, "run", forbidden)
    monkeypatch.setattr(abi.C, "CDLL", forbidden)
    monkeypatch.setattr(_rpc, "tool_path", forbidden)
    scene = tmp_path / "plain.xml"
    scene.write_text('<mujoco><worldbody><body><freejoint/><geom size=".01"/></body></worldbody></mujoco>')
    sim = MuJoCoSimCore()
    sim.init_simulation(str(scene))
    sim.step(2)
    sim.reset_data()
    assert sim.query_provider_sensor_data() == {}
    sim.close_provider_sensors()



def test_external_tool_must_be_an_executable_file(tmp_path, monkeypatch):
    from orca_gym.sensor.providers.contracts import _rpc

    monkeypatch.setenv("ORCA_SENSOR_TOOL", str(tmp_path / "missing-tool"))
    with pytest.raises(_rpc.SensorToolError, match="ORCA_SENSOR_TOOL"):
        _rpc.tool_path()


def require_bundled_platform():
    libc, version = platform.libc_ver()
    try:
        glibc = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        glibc = ()
    if (platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}
            or libc != "glibc" or glibc < (2, 35)):
        pytest.skip("Bundled provider runtime currently targets Linux x86_64 / glibc 2.35+")


def test_default_host_and_tool_are_in_the_installed_package(monkeypatch):
    from orca_gym.sensor.providers import native
    from orca_gym.sensor.providers.contracts import _rpc

    require_bundled_platform()
    monkeypatch.delenv("ORCA_SENSOR_TOOL", raising=False)
    root = Path(native.__file__).resolve().parent.parent / "native"
    assert native.default_host_path().resolve() == root / "lib/linux-x86_64/liborca_sensor_host.so"
    assert _rpc.tool_path() == root / "bin/linux-x86_64/orca-sensor-tool"


def test_missing_supported_runtime_reports_an_installation_error(tmp_path, monkeypatch):
    from orca_gym.sensor.providers import native
    from orca_gym.sensor.providers.contracts import _rpc

    monkeypatch.delenv("ORCA_SENSOR_TOOL", raising=False)
    monkeypatch.setattr(native, "runtime_directory", lambda: tmp_path)
    monkeypatch.setattr(_rpc, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(_rpc, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(native.platform, "system", lambda: "Linux")
    monkeypatch.setattr(native.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(native.platform, "libc_ver", lambda: ("glibc", "2.35"))
    with pytest.raises(RuntimeError, match="official orca-gym"):
        native.default_host_path()
    with pytest.raises(_rpc.SensorToolError, match="official orca-gym"):
        _rpc.tool_path()


def test_unsupported_runtime_errors_are_scoped_to_providers(tmp_path, monkeypatch):
    from orca_gym.sensor.providers import native
    from orca_gym.sensor.providers.contracts import _rpc

    monkeypatch.delenv("ORCA_SENSOR_TOOL", raising=False)
    monkeypatch.setattr(native, "runtime_directory", lambda: tmp_path)
    monkeypatch.setattr(_rpc, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(_rpc, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(native.platform, "system", lambda: "Windows")
    monkeypatch.setattr(native.platform, "machine", lambda: "AMD64")
    with pytest.raises(RuntimeError, match="core OrcaGym is unaffected"):
        native.default_host_path()
    with pytest.raises(_rpc.SensorToolError, match="ordinary OrcaGym use is unaffected"):
        _rpc.tool_path()


def test_explicit_tool_override_does_not_require_a_bundled_install(tmp_path, monkeypatch):
    from orca_gym.sensor.providers.contracts import _rpc

    executable = tmp_path / "trusted-tool"
    executable.write_bytes(b"test executable")
    executable.chmod(0o755)
    monkeypatch.setenv("ORCA_SENSOR_TOOL", str(executable))
    monkeypatch.setattr(_rpc, "RUNTIME_ROOT", tmp_path / "missing-bundle")
    assert _rpc.tool_path() == executable
