"""Public EulerEnv integration: one physical state, per-substep DLL observations."""

from contextlib import contextmanager
from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.sensor.providers import SensorError, SensorHost
from test_support import append_custom_sensor_instances
from test_support import HAND, packages, specs, write_scene


IDS = ("hand.tip", "second", "range", "biased")
NOT_READY = (SensorError, RuntimeError)


class ProviderTaskEnv(OrcaGymEulerEnv):
    """A task explicitly chooses when provider results become observations."""

    def __init__(self, xml_path, *, frame_skip=4, time_step=.001, **kwargs):
        self.provider_observations = (kwargs.get("sensor_host_path") is not None
                                      or kwargs.get("sensor_provider_manifests") is not None)
        super().__init__(frame_skip=frame_skip, orcagym_addr="localhost:50051", agent_names=["robot"],
                         time_step=time_step, model_xml_path=str(xml_path), skip_grpc_load=True,
                         render_mode="none", **kwargs)

    def _get_obs(self):
        return {"qpos": self.data.qpos.copy(), "time": float(self.data.time)}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos.copy())
        self.set_joint_qvel(self.init_qvel.copy())
        self.mj_forward()
        self._sync_view()
        # Reset does not invent a tactile observation before the first substep.
        return self._get_obs(), {}

    def step(self, action):
        self.do_simulation(np.asarray(action, dtype=np.float64), self.frame_skip)
        observation = self._get_obs()
        if self.provider_observations:
            observation["sensors"] = self.query_provider_sensor_data()
        return observation, 0.0, False, False, {}


@pytest.fixture
def press_scene(tmp_path):
    root = ET.fromstring('''<mujoco model="provider_controlled_press">
      <option timestep=".001" integrator="implicit" gravity="0 0 0"/>
      <worldbody>
        <body name="pad">
          <geom name="pad_shape" type="box" pos="0 0 .01" size=".03 .03 .01"/>
          <site name="original_tip_site" size=".001"/>
        </body>
        <body name="payload" pos=".004 .003 .08">
          <joint name="lift" type="slide" axis="0 0 1" range="-.07 .05" damping="1"/>
          <geom name="ball" type="sphere" size=".02" mass=".2"/>
        </body>
      </worldbody>
      <actuator><position name="press" joint="lift" kp="1000" kv="20" ctrllimited="true" ctrlrange="-.065 .03"/></actuator>
      <sensor><jointpos name="native_height" joint="lift"/></sensor>
    </mujoco>''')
    append_custom_sensor_instances(root, specs())
    return write_scene(tmp_path, root)


@contextmanager
def configured_env(sdk_build, xml_path, **kwargs):
    env = ProviderTaskEnv(xml_path, sensor_host_path=sdk_build[0],
                          sensor_provider_manifests=packages(sdk_build), **kwargs)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def calls(monkeypatch):
    """Observe public Host calls, never reach through Env/Gym/SimCore internals."""
    result = SimpleNamespace(samples=[], publications=[], resets=[], instances=[], closed=[])
    compute, publish = SensorHost.compute, SensorHost.publish
    reset, create, close = SensorHost.reset, SensorHost.create_sensor, SensorHost.close

    def record_compute(host, instance_id, sample, **kwargs):
        result.samples.append(SimpleNamespace(instance_id=instance_id, time=sample.time, dt=sample.dt,
                                             index=sample.step_index,
                                             fields={name: np.asarray(value).copy()
                                                     for name, value in sample.fields.items()}))
        return compute(host, instance_id, sample, **kwargs)

    def record_publish(host):
        value = publish(host)
        result.publications.append(len(result.samples))
        return value

    def record_reset(host, seed=0):
        result.resets.append(seed)
        return reset(host, seed)

    def record_create(host, *args, **kwargs):
        instance = create(host, *args, **kwargs)
        result.instances.append(instance)
        return instance

    def record_close(host):
        value = close(host)
        result.closed.append(host)
        return value

    monkeypatch.setattr(SensorHost, "compute", record_compute)
    monkeypatch.setattr(SensorHost, "publish", record_publish)
    monkeypatch.setattr(SensorHost, "reset", record_reset)
    monkeypatch.setattr(SensorHost, "create_sensor", record_create)
    monkeypatch.setattr(SensorHost, "close", record_close)
    return result


def test_initialization_and_reset_do_not_compute_or_publish(sdk_build, press_scene, calls):
    with configured_env(sdk_build, press_scene) as env:
        assert len(calls.instances) == len(IDS)
        assert not calls.samples and not calls.publications
        with pytest.raises(NOT_READY):
            env.query_provider_sensor_data()
        observation, info = env.reset(seed=37)
        assert observation["time"] == 0 and info == {}
        assert not calls.samples and not calls.publications
        assert calls.resets[-1] == 37
        with pytest.raises(NOT_READY):
            env.query_provider_sensor_data(["range"])


def test_action_uses_one_native_state_and_publishes_only_last_substep(sdk_build, press_scene, calls):
    model = mujoco.MjModel.from_xml_path(str(press_scene))
    data = mujoco.MjData(model)
    command = np.array([-.05])
    with configured_env(sdk_build, press_scene, frame_skip=8) as env:
        obs, reward, terminated, truncated, _ = env.step(command)
        data.ctrl[:] = command
        mujoco.mj_step(model, data, nstep=8)
        np.testing.assert_allclose(env.data.qpos, data.qpos, rtol=0, atol=1e-13)
        np.testing.assert_allclose(env.data.qvel, data.qvel, rtol=0, atol=1e-13)
        assert env.data.qpos[0] < 0 and env.data.time == pytest.approx(.008)
        assert obs["time"] == pytest.approx(.008)
        assert reward == 0 and not terminated and not truncated
        assert set(obs["sensors"]) == set(IDS)
        assert calls.publications == [8 * len(IDS)]
        for index in range(8):
            batch = calls.samples[index * len(IDS):(index + 1) * len(IDS)]
            assert {value.instance_id for value in batch} == set(IDS)
            for value in batch:
                assert value.index == index and value.time == pytest.approx(index * .001)
                assert value.dt == pytest.approx(.001)
                assert value.fields["sample_time"] == value.time
        native = env.query_sensor_data(["native_height"])
        assert set(native) == {"native_height"}
        np.testing.assert_array_equal(native["native_height"], data.sensordata)


def test_real_control_changes_both_physics_and_tactile_observation(sdk_build, press_scene):
    with configured_env(sdk_build, press_scene, frame_skip=400) as env:
        relaxed = env.step([0.0])[0]
        assert relaxed["sensors"]["hand.tip"].sum() == 0
        assert .05 < relaxed["sensors"]["range"][0] < .1
        env.reset(seed=11)
        pressed = env.step([-.05])[0]
        assert pressed["qpos"][0] < relaxed["qpos"][0] - .02
        tactile = pressed["sensors"]
        assert tactile["hand.tip"].shape == (4, 4) and tactile["range"].shape == (1,)
        assert tactile["hand.tip"].sum() > 1
        assert 0 < tactile["range"][0] < relaxed["sensors"]["range"][0]
        np.testing.assert_allclose(tactile["second"], tactile["hand.tip"] * 2)
        np.testing.assert_allclose(tactile["biased"], tactile["range"] + .01)




def test_mj_step_n_uses_same_substep_pipeline_and_single_publish(sdk_build, press_scene, calls):
    with configured_env(sdk_build, press_scene) as env:
        env.set_ctrl(np.array([-.05]))
        env.mj_step(nstep=5)
        values = env.query_provider_sensor_data()
        assert set(values) == set(IDS)
        assert len(calls.samples) == 5 * len(IDS)
        assert calls.publications == [5 * len(IDS)]
        assert {sample.index for sample in calls.samples[-len(IDS):]} == {4}
        # The next standard step continues the same physical clock and sampler.
        env.step([-.05])
        assert env.data.time == pytest.approx(.009)
        assert calls.publications == [20, 36]
        assert {sample.index for sample in calls.samples[-len(IDS):]} == {8}


def test_result_copies_exact_names_and_read_forward_render_have_no_side_effect(sdk_build, press_scene, calls):
    with configured_env(sdk_build, press_scene) as env:
        env.step([-.05])
        baseline = env.query_provider_sensor_data()
        selected = env.query_provider_sensor_data(["range", "hand.tip"])
        assert set(selected) == {"range", "hand.tip"}
        assert env.query_provider_sensor_data([]) == {}
        for name in selected:
            assert not np.shares_memory(selected[name], baseline[name])
            selected[name][:] = 99
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            env.query_provider_sensor_data(["robot/hand.tip"])
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            env.query_provider_sensor_data(["range", "unknown"])
        before = (len(calls.samples), len(calls.publications), env.data.qpos.copy(), env.data.time)
        env.mj_forward()
        env.render()
        for _ in range(3):
            values = env.query_provider_sensor_data()
            for name in IDS:
                np.testing.assert_array_equal(values[name], baseline[name])
        assert (len(calls.samples), len(calls.publications)) == before[:2]
        np.testing.assert_array_equal(env.data.qpos, before[2])
        assert env.data.time == before[3]


def test_reset_seed_replays_physics_and_dll_outputs_and_keeps_seed_when_omitted(sdk_build, press_scene, calls):
    with configured_env(sdk_build, press_scene, frame_skip=60) as env:
        env.reset(seed=123)
        first = [env.step([-.05])[0] for _ in range(3)]
        env.reset(seed=123)
        assert calls.resets[-1] == 123
        with pytest.raises(NOT_READY):
            env.query_provider_sensor_data()
        second = [env.step([-.05])[0] for _ in range(3)]
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a["qpos"], b["qpos"])
            for name in IDS:
                np.testing.assert_array_equal(a["sensors"][name], b["sensors"][name])
        env.reset()
        assert calls.resets[-1] == 123
        before = len(calls.samples)
        env.step([-.05])
        assert calls.samples[before].index == 0 and calls.samples[before].time == 0


def test_live_timestep_change_reaches_sample_headers_without_a_second_clock(sdk_build, press_scene, calls):
    with configured_env(sdk_build, press_scene) as env:
        env.step([0])
        env.sim_config.timestep = .002
        begin = len(calls.samples)
        env.step([0])
        assert env.data.time == pytest.approx(.012)
        for offset in range(4):
            batch = calls.samples[begin + offset * len(IDS):begin + (offset + 1) * len(IDS)]
            assert all(value.dt == .002 and value.index == offset + 4 for value in batch)
            assert all(value.time == pytest.approx(.004 + offset * .002) for value in batch)


def test_offline_close_releases_handles_and_is_idempotent(sdk_build, press_scene, calls):
    with configured_env(sdk_build, press_scene) as env:
        env.step([0])
        instances = list(calls.instances)
        env.close()
        assert calls.closed
        for instance in instances:
            with pytest.raises((SensorError, KeyError, RuntimeError)):
                instance.read()
        env.close()


def test_reinitialize_simulation_releases_and_rebinds_public_scene(sdk_build, press_scene, calls):
    with configured_env(sdk_build, press_scene) as env:
        env.step([-.05])
        old_instances = list(calls.instances)
        env.initialize_simulation()
        assert len(calls.instances) == len(IDS) * 2
        assert calls.closed
        for instance in old_instances:
            with pytest.raises((SensorError, KeyError, RuntimeError)):
                instance.read()
        with pytest.raises(NOT_READY):
            env.query_provider_sensor_data()
        begin = len(calls.samples)
        env.step([0])
        assert env.data.time == pytest.approx(.004)
        assert calls.samples[begin].index == calls.samples[begin].time == 0


def test_unconfigured_env_keeps_native_behavior_and_ignores_custom_dll_metadata(press_scene, calls):
    env = ProviderTaskEnv(press_scene)
    try:
        assert env.query_provider_sensor_data() == {}
        assert env.query_provider_sensor_data([]) == {}
        env.step([-.05])
        assert not calls.instances and not calls.samples and not calls.publications
        model = mujoco.MjModel.from_xml_path(str(press_scene))
        data = mujoco.MjData(model)
        data.ctrl[:] = -.05
        mujoco.mj_step(model, data, nstep=4)
        np.testing.assert_array_equal(env.data.qpos, data.qpos)
        np.testing.assert_array_equal(env.query_sensor_data(["native_height"])["native_height"], data.sensordata)
    finally:
        env.close()


def test_enabled_config_without_xml_declarations_has_no_algorithms(sdk_build, press_scene, tmp_path, calls):
    root = ET.parse(press_scene).getroot()
    root.remove(root.find("custom"))
    path = write_scene(tmp_path, root)
    with configured_env(sdk_build, path) as env:
        assert env.query_provider_sensor_data() == {}
        env.step([0])
        assert not calls.instances and not calls.samples


@pytest.mark.parametrize("failure", ["host_path", "manifest_path", "missing_manifests"])
def test_invalid_or_incomplete_explicit_provider_configuration_rejected(sdk_build, press_scene, tmp_path, failure):
    options = {"sensor_host_path": sdk_build[0], "sensor_provider_manifests": packages(sdk_build)}
    if failure == "host_path":
        options["sensor_host_path"] = tmp_path / "missing_host.so"
    elif failure == "manifest_path":
        options["sensor_provider_manifests"] = [tmp_path / "missing_provider.json"]
    else:
        options.pop("sensor_provider_manifests")
    with pytest.raises((ValueError, RuntimeError, OSError, SensorError)):
        ProviderTaskEnv(press_scene, **options)


@pytest.mark.parametrize("failure", ["missing_type", "missing_site", "unknown_parameter", "legacy_json"])
def test_invalid_scene_binding_is_not_silently_ignored(sdk_build, press_scene, tmp_path, failure):
    root = ET.parse(press_scene).getroot()
    custom = root.find("custom")
    if failure == "missing_type":
        custom.find("text").set("data", "com.uninstalled.sensor")
    elif failure == "missing_site":
        custom.find("tuple/element").set("objname", "absent_site")
    elif failure == "unknown_parameter":
        ET.SubElement(custom, "numeric", name="orca.sensor.v1/hand.tip/config/not_supported", data="1")
    else:
        custom.clear()
        ET.SubElement(custom, "text", name="orca.sensor.instances.v1", data=json.dumps([
            {"instance_id": "hand.tip", "type_id": "com.orca.examples.contact_grid", "site": "original_tip_site"},
        ]))
    path = write_scene(tmp_path, root)
    with pytest.raises((ValueError, KeyError, RuntimeError, SensorError)):
        with configured_env(sdk_build, path):
            pass


def test_constructor_failure_closes_already_created_algorithm_handles(sdk_build, press_scene, tmp_path, calls):
    root = ET.parse(press_scene).getroot()
    ET.SubElement(root.find("custom"), "numeric", name="orca.sensor.v1/second/config/not_supported", data="1")
    with pytest.raises((ValueError, RuntimeError, SensorError)):
        with configured_env(sdk_build, write_scene(tmp_path, root)):
            pass
    assert calls.instances and calls.closed
    for instance in calls.instances:
        with pytest.raises((SensorError, KeyError, RuntimeError)):
            instance.read()


def test_rk4_at_load_and_runtime_reconfiguration_are_rejected(sdk_build, press_scene, tmp_path, calls):
    with configured_env(sdk_build, press_scene) as env:
        with pytest.raises((ValueError, RuntimeError, SensorError)):
            env.sim_config.integrator = int(mujoco.mjtIntegrator.mjINT_RK4)
            env.step([0])
        assert not calls.samples and not calls.publications
    root = ET.parse(press_scene).getroot()
    root.find("option").set("integrator", "RK4")
    with pytest.raises((ValueError, RuntimeError, SensorError)):
        with configured_env(sdk_build, write_scene(tmp_path, root)):
            pass


def test_dll_failure_invalidates_whole_batch_and_reset_recovers(sdk_build, press_scene, monkeypatch):
    with configured_env(sdk_build, press_scene) as env:
        env.step([0])
        compute = SensorHost.compute
        seen = []

        def fail_second(host, instance_id, sample, **kwargs):
            seen.append(instance_id)
            if len(seen) == 2:
                raise SensorError("test vendor computation failure")
            return compute(host, instance_id, sample, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(SensorHost, "compute", fail_second)
            with pytest.raises((SensorError, RuntimeError), match="failure"):
                env.step([-.05])
            with pytest.raises(NOT_READY):
                env.query_provider_sensor_data()
            failed_time = env.data.time
            failed_qpos = env.data.qpos.copy()
            with pytest.raises(NOT_READY):
                env.step([-.05])
            assert env.data.time == failed_time
            np.testing.assert_array_equal(env.data.qpos, failed_qpos)
        env.reset(seed=5)
        assert set(env.step([0])[0]["sensors"]) == set(IDS)


def test_complete_hand_ten_site_instances_use_the_environment_state(sdk_build, calls):
    original = HAND.read_bytes()
    with configured_env(sdk_build, HAND, frame_skip=3) as env:
        outputs = env.step(np.zeros(env.model.nu))[0]["sensors"]
        assert len(outputs) == 10 and len(calls.instances) == 10
        assert len(calls.samples) == 30 and calls.publications == [30]
        assert env.data.time == pytest.approx(.003)
        for finger in range(1, 6):
            assert outputs[f"touch_f{finger}"].shape == (4, 4)
            assert outputs[f"range_f{finger}"].shape == (1,)
            assert np.isfinite(outputs[f"touch_f{finger}"]).all()
            assert np.isfinite(outputs[f"range_f{finger}"]).all()
    assert HAND.read_bytes() == original
