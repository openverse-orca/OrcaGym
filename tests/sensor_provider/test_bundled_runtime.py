"""Ordinary EulerEnv construction uses OrcaGym's installed runtime."""

from pathlib import Path

import numpy as np
import pytest

from orca_gym.sensor.providers import SensorHost
from orca_gym.sensor.providers import native
from orca_gym.sensor.providers.contracts import _rpc
from test_euler_provider_env import IDS, NOT_READY, ProviderTaskEnv
from test_euler_provider_env import calls as calls
from test_euler_provider_env import press_scene as press_scene
from test_optional_binaries import require_bundled_platform
from test_support import packages


@pytest.mark.parametrize("explicit_host", [False, True])
def test_runtime_steps_replays_and_closes_without_external_tool(
        sdk_build, press_scene, calls, monkeypatch, explicit_host):
    require_bundled_platform()
    monkeypatch.delenv("ORCA_SENSOR_TOOL", raising=False)
    root = Path(native.__file__).resolve().parent.parent / "native"
    bundled_host = root / "lib/linux-x86_64/liborca_sensor_host.so"
    assert native.default_host_path().resolve() == bundled_host
    assert _rpc.tool_path() == root / "bin/linux-x86_64/orca-sensor-tool"
    loaded = []
    initialize = SensorHost.__init__

    def record_initialize(host, library):
        loaded.append(Path(library).resolve())
        initialize(host, library)

    monkeypatch.setattr(SensorHost, "__init__", record_initialize)
    options = {"sensor_provider_manifests": packages(sdk_build)}
    if explicit_host:
        options["sensor_host_path"] = sdk_build[0]
    env = ProviderTaskEnv(press_scene, frame_skip=400, **options)
    try:
        expected_host = sdk_build[0].resolve() if explicit_host else bundled_host
        assert loaded == [expected_host]
        with pytest.raises(NOT_READY):
            env.query_provider_sensor_data()
        env.reset(seed=37)
        assert calls.resets[-1] == 37
        first = env.step([-.05])[0]
        assert first["time"] == pytest.approx(.4)
        assert set(first["sensors"]) == set(IDS)
        assert first["sensors"]["hand.tip"].sum() > 1
        np.testing.assert_allclose(first["sensors"]["second"], first["sensors"]["hand.tip"] * 2)
        assert 0 < first["sensors"]["range"][0] < .1
        env.reset(seed=37)
        assert calls.resets[-1] == 37
        with pytest.raises(NOT_READY):
            env.query_provider_sensor_data()
        replay = env.step([-.05])[0]
        np.testing.assert_array_equal(first["qpos"], replay["qpos"])
        for name in IDS:
            np.testing.assert_array_equal(first["sensors"][name], replay["sensors"][name])
        env.reset()
        assert calls.resets[-1] == 37
    finally:
        env.close()
    assert len(calls.closed) == 1
    for instance in calls.instances:
        with pytest.raises((KeyError, *NOT_READY)):
            instance.read()
    env.close()
    assert len(calls.closed) == 1
