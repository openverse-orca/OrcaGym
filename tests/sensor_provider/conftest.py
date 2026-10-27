"""Optional real vendor artifacts from the independently released SDK."""

import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def sdk_build(tmp_path_factory):
    value = os.environ.get("ORCA_SENSOR_TEST_SDK")
    if not value:
        pytest.skip("Set ORCA_SENSOR_TEST_SDK to a released SDK to run real provider integration tests")
    sdk = Path(value).expanduser().resolve()
    for relative in ("VERSION", "CMakeLists.txt", "include/orca_sensor_abi.h"):
        if not (sdk / relative).is_file():
            pytest.fail(f"ORCA_SENSOR_TEST_SDK is incomplete: {sdk / relative}")
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(platform.machine().lower(), platform.machine().lower())
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform)
    executable = "orca-sensor-tool.exe" if sys.platform == "win32" else "orca-sensor-tool"
    tool = sdk / "bin" / f"{system}-{machine}" / executable
    if not tool.is_file():
        pytest.skip(f"Released SDK has no native runtime for {system}-{machine}")
    build = tmp_path_factory.mktemp("site-provider-build")
    subprocess.run(["cmake", "-S", str(sdk), "-B", str(build),
                    f"-DPython3_EXECUTABLE={sys.executable}", "-DCMAKE_BUILD_TYPE=Release"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--target",
                    "orca_contact_grid", "orca_rangefinder", "orca_touch_grid", "orca_seven_pad", "-j", "2"],
                   check=True, capture_output=True, text=True)
    host_name = {"linux": "liborca_sensor_host.so", "darwin": "liborca_sensor_host.dylib",
                 "win32": "orca_sensor_host.dll"}[sys.platform]
    return build / host_name, build / "providers/contact_grid/provider.json"
