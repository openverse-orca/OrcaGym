"""Replacing a connection must release its provider instances."""

import pytest

from orca_gym.sensor.providers import SensorHost
from test_support import HAND
from test_euler_provider_env import NOT_READY, configured_env
from test_euler_provider_env import calls as calls
from test_euler_provider_env import press_scene as press_scene

ENV_MODULE = "orca_gym.environment.euler.orca_gym_euler_env"


def assert_closed(instances):
    for instance in instances:
        with pytest.raises(NOT_READY):
            instance.read()



def test_reinitialize_connection_closes_all_ten_old_handles_before_rebinding(sdk_build, calls):
    with configured_env(sdk_build, HAND) as env:
        env.mj_step(1)
        original = tuple(calls.instances)
        assert len(original) == 10
        env.initialize_grpc()
        assert_closed(original)
        assert len(calls.instances) == 10
        assert len(calls.closed) == 1

        env.initialize_simulation()
        with pytest.raises(NOT_READY):
            env.query_provider_sensor_data()
        env.mj_step(1)
        assert len(calls.instances) == 20
        assert len(env.query_provider_sensor_data()) == 10
        assert calls.samples[-1].time == 0
        assert calls.samples[-1].index == 0
    assert_closed(calls.instances)
    assert len(calls.closed) == 2



def test_old_provider_cleanup_failure_aborts_replacement_and_can_be_retried(
        sdk_build, press_scene, calls, monkeypatch):
    with configured_env(sdk_build, press_scene) as env:
        env.mj_step(1)
        original = tuple(calls.instances)
        native_close = SensorHost.close

        def fail_after_closing(host):
            native_close(host)
            raise RuntimeError("injected old Host cleanup failure")

        with monkeypatch.context() as patched:
            patched.setattr(SensorHost, "close", fail_after_closing)
            with pytest.raises(RuntimeError, match="old Host cleanup failure"):
                env.initialize_grpc()
        assert len(calls.instances) == len(original)
        assert_closed(original)

        env.initialize_grpc()
        env.initialize_simulation()
        env.mj_step(1)
        assert len(env.query_provider_sensor_data()) == 4
    assert_closed(calls.instances)



def test_new_gym_creation_failure_keeps_old_handles_closed_and_close_is_safe(
        sdk_build, press_scene, calls, monkeypatch):
    with configured_env(sdk_build, press_scene) as env:
        original = tuple(calls.instances)

        def fail_creation(*args, **kwargs):
            raise RuntimeError("injected new Gym creation failure")

        with monkeypatch.context() as patched:
            patched.setattr(ENV_MODULE + ".OrcaGymEuler", fail_creation)
            with pytest.raises(RuntimeError, match="new Gym creation failure"):
                env.initialize_grpc()
        assert_closed(original)
        env.close()
        env.close()

        env.initialize_grpc()
        env.initialize_simulation()
        env.mj_step(1)
        assert len(env.query_provider_sensor_data()) == 4
    assert_closed(calls.instances)
