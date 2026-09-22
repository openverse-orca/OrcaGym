"""Provider sensors share the Euler core's physics state and lifecycle.

Assertions use public core methods and the data view. Native call observation
does not expose any simulator-owned pointer to a provider or a test consumer.
"""

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.sensor.providers import SensorHost
from test_support import append_custom_sensor_instances


GRID_TYPE = "com.orca.examples.contact_grid"
RAY_TYPE = "com.orca.examples.rangefinder"
INSTANCE_IDS = {"grid", "amplified", "ray", "biased"}


def scene_root(*, integrator="implicit", declarations=True):
    root = ET.fromstring('''<mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body name="pad">
          <joint name="pad_joint" type="hinge" axis="0 0 1" damping="1"/>
          <geom name="pad_shape" type="box" pos="0 0 0.01"
                size="0.03 0.03 0.01" mass="1"/>
          <site name="original_tip_site" size="0.001"/>
        </body>
        <body name="payload" pos="0.004 0.003 0.06">
          <freejoint name="payload_joint"/>
          <geom name="ball" type="sphere" size="0.02" mass="0.2"/>
        </body>
      </worldbody>
      <sensor><framepos name="native_tip" objtype="site" objname="original_tip_site"/></sensor>
    </mujoco>''')
    root.find("option").set("integrator", integrator)
    if declarations:
        append_custom_sensor_instances(root, [
            {"instance_id": "grid", "type_id": GRID_TYPE, "site": "original_tip_site"},
            {"instance_id": "amplified", "type_id": GRID_TYPE, "site": "original_tip_site",
             "global_parameters": {"gain": 2}},
            {"instance_id": "ray", "type_id": RAY_TYPE, "site": "original_tip_site"},
            {"instance_id": "biased", "type_id": RAY_TYPE, "site": "original_tip_site",
             "global_parameters": {"bias_m": 0.01}},
        ])
    return root


def save_scene(tmp_path, root, name="scene.xml"):
    path = tmp_path / name
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def manifests(sdk_build):
    return [sdk_build[0].parent / "providers" / name / "provider.json"
            for name in ("contact_grid", "rangefinder")]


def physical_state(core):
    view = OrcaGymDataView()
    core.sync_to_view(view)
    return {"time": view.time, "qpos": view.qpos.copy(),
            "qvel": view.qvel.copy(), "qacc": view.qacc.copy()}


def assert_same_state(actual, expected):
    assert actual["time"] == expected["time"]
    for field in ("qpos", "qvel", "qacc"):
        np.testing.assert_array_equal(actual[field], expected[field])


@pytest.fixture
def provider_core(sdk_build, tmp_path):
    path = save_scene(tmp_path, scene_root())
    core = MuJoCoSimCore()
    core.configure_provider_sensors(sdk_build[0], manifests(sdk_build))
    try:
        core.init_simulation(str(path))
        yield core, path
    finally:
        core.close_provider_sensors()


def test_shared_site_outputs_preserve_native_physics(provider_core, monkeypatch):
    core, path = provider_core
    received = {}
    native_compute = SensorHost.compute

    def record_compute(host, instance_id, sample, **kwargs):
        received[instance_id] = (sample.time, sample.dt, sample.step_index,
                                 {key: np.array(value, copy=True)
                                  for key, value in sample.fields.items()})
        return native_compute(host, instance_id, sample, **kwargs)

    monkeypatch.setattr(SensorHost, "compute", record_compute)
    oracle_model = mujoco.MjModel.from_xml_path(str(path))
    oracle_data = mujoco.MjData(oracle_model)
    core.step(400)
    mujoco.mj_step(oracle_model, oracle_data, 400)
    assert_same_state(physical_state(core), {
        "time": oracle_data.time, "qpos": oracle_data.qpos,
        "qvel": oracle_data.qvel, "qacc": oracle_data.qacc,
    })
    output = core.query_provider_sensor_data()
    assert set(output) == INSTANCE_IDS
    assert output["grid"].shape == (4, 4) and output["ray"].shape == (1,)
    assert all(value.dtype == np.float64 for value in output.values())
    np.testing.assert_allclose(output["grid"].sum(), 0.2 * 9.81, rtol=0.03)
    np.testing.assert_allclose(output["amplified"], 2 * output["grid"])
    assert 0 < output["ray"][0] < 0.1
    np.testing.assert_allclose(output["biased"], output["ray"] + 0.01)
    stamps = {sample[:3] for sample in received.values()}
    assert len(stamps) == 1
    time, dt, index = stamps.pop()
    assert time == pytest.approx(0.399) and dt == 0.001 and index == 399
    grid = received["grid"][3]["force_grid"]
    assert grid.shape == (4, 4, 3) and grid[..., 2].sum() < 0
    np.testing.assert_array_equal(received["amplified"][3]["force_grid"], grid)
    np.testing.assert_allclose(output["grid"], np.linalg.norm(grid, axis=-1))
    native = core.query_sensor_data(["native_tip"], {})
    np.testing.assert_array_equal(native["native_tip"], oracle_data.sensor("native_tip").data)


def test_one_native_model_and_data_substep_compute_and_single_publish(sdk_build, tmp_path, monkeypatch):
    path = save_scene(tmp_path, scene_root())
    allocations = {"model": 0, "data": 0}
    calls = {"steps": [], "compute": [], "publish": 0, "forward": 0}
    native_model = mujoco.MjModel.from_xml_path
    native_data = mujoco.MjData
    native_step = mujoco.mj_step
    native_forward = mujoco.mj_forward
    native_compute = SensorHost.compute
    native_publish = SensorHost.publish

    def make_model(*args, **kwargs):
        allocations["model"] += 1
        return native_model(*args, **kwargs)

    def make_data(*args, **kwargs):
        allocations["data"] += 1
        return native_data(*args, **kwargs)

    def record_step(model, data, nstep=1):
        calls["steps"].append((id(model), id(data), nstep))
        return native_step(model, data, nstep)

    def record_forward(*args, **kwargs):
        calls["forward"] += 1
        return native_forward(*args, **kwargs)

    def record_compute(host, instance_id, sample, **kwargs):
        calls["compute"].append((instance_id, sample.step_index))
        return native_compute(host, instance_id, sample, **kwargs)

    def record_publish(host):
        calls["publish"] += 1
        return native_publish(host)

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", make_model)
    monkeypatch.setattr(mujoco, "MjData", make_data)
    monkeypatch.setattr(mujoco, "mj_step", record_step)
    monkeypatch.setattr(mujoco, "mj_forward", record_forward)
    monkeypatch.setattr(SensorHost, "compute", record_compute)
    monkeypatch.setattr(SensorHost, "publish", record_publish)
    core = MuJoCoSimCore()
    core.configure_provider_sensors(sdk_build[0], manifests(sdk_build))
    try:
        core.init_simulation(str(path))
        assert allocations == {"model": 1, "data": 1}
        forwards_before = calls["forward"]
        core.step(3)
        assert len(calls["steps"]) == 3
        assert {step[2] for step in calls["steps"]} == {1}
        assert len({step[:2] for step in calls["steps"]}) == 1
        assert set(calls["compute"]) == {(name, index) for name in INSTANCE_IDS for index in range(3)}
        assert len(calls["compute"]) == 12 and calls["publish"] == 1
        assert calls["forward"] == forwards_before
        core.forward()
        core.forward()
        assert len(calls["compute"]) == 12 and calls["publish"] == 1
        assert calls["forward"] == forwards_before + 2
        assert allocations == {"model": 1, "data": 1}
    finally:
        core.close_provider_sensors()


def test_query_is_owned_filtered_and_forward_does_not_make_results_ready(provider_core):
    core, _ = provider_core
    with pytest.raises(RuntimeError):
        core.query_provider_sensor_data()
    core.forward()
    with pytest.raises(RuntimeError):
        core.query_provider_sensor_data()
    core.step(10)
    saved = core.query_provider_sensor_data()
    subset = core.query_provider_sensor_data(["ray", "grid"])
    assert set(subset) == {"ray", "grid"}
    subset["ray"][:] = 123
    subset["grid"][:] = 456
    reread = core.query_provider_sensor_data()
    for name in INSTANCE_IDS:
        np.testing.assert_array_equal(reread[name], saved[name])
    with pytest.raises(ValueError, match="Unknown provider sensor instance"):
        core.query_provider_sensor_data(["unknown"])
    core.forward()
    for name, value in core.query_provider_sensor_data().items():
        np.testing.assert_array_equal(value, saved[name])


def test_algorithm_reset_keeps_physics_and_physical_reset_reuses_seed(provider_core, monkeypatch):
    core, _ = provider_core
    resets = []
    native_reset = SensorHost.reset

    def record_reset(host, seed=0):
        resets.append(seed)
        return native_reset(host, seed)

    monkeypatch.setattr(SensorHost, "reset", record_reset)
    core.step(25)
    before = physical_state(core)
    seed = 2**64 - 1
    core.reset_provider_sensors(seed)
    assert resets[-1] == seed
    assert_same_state(physical_state(core), before)
    with pytest.raises(RuntimeError):
        core.query_provider_sensor_data()
    core.forward()
    with pytest.raises(RuntimeError):
        core.query_provider_sensor_data()
    core.step(1)
    core.reset_data()
    assert resets[-1] == seed and physical_state(core)["time"] == 0
    with pytest.raises(RuntimeError):
        core.query_provider_sensor_data()
    core.step(25)
    reference = core.query_provider_sensor_data()
    core.reset_data(sensor_seed=42)
    assert resets[-1] == 42 and physical_state(core)["time"] == 0
    core.step(25)
    assert_same_state(physical_state(core), before)
    for name, value in core.query_provider_sensor_data().items():
        np.testing.assert_array_equal(value, reference[name])
    core.reset_data()
    assert resets[-1] == 42


@pytest.mark.parametrize("seed", [-1, 2**64, True, 1.5, "1"])
@pytest.mark.parametrize("physical", [False, True])
def test_invalid_reset_seed_does_not_mutate_physics_or_output(provider_core, seed, physical):
    core, _ = provider_core
    core.step(5)
    before = physical_state(core)
    saved = core.query_provider_sensor_data()
    with pytest.raises((TypeError, ValueError)):
        if physical:
            core.reset_data(sensor_seed=seed)
        else:
            core.reset_provider_sensors(seed)
    assert_same_state(physical_state(core), before)
    for name, value in core.query_provider_sensor_data().items():
        np.testing.assert_array_equal(value, saved[name])


def test_physics_failure_does_not_roll_back_but_faults_outputs(provider_core, monkeypatch):
    core, _ = provider_core
    core.step(2)
    saved = core.query_provider_sensor_data()
    before = physical_state(core)
    native_step = mujoco.mj_step
    calls = []

    def fail_after_second_advance(model, data, nstep=1):
        native_step(model, data, nstep)
        calls.append(nstep)
        if len(calls) == 2:
            raise RuntimeError("injected physics failure")

    monkeypatch.setattr(mujoco, "mj_step", fail_after_second_advance)
    with pytest.raises(RuntimeError, match="injected physics failure"):
        core.step(3)
    assert calls == [1, 1]
    advanced = physical_state(core)
    assert advanced["time"] == pytest.approx(before["time"] + 0.002)
    assert not np.array_equal(advanced["qpos"], before["qpos"])
    with pytest.raises(RuntimeError):
        core.query_provider_sensor_data()
    with pytest.raises(RuntimeError):
        core.step(1)
    assert_same_state(physical_state(core), advanced)
    assert calls == [1, 1]
    core.reset_provider_sensors(17)
    assert_same_state(physical_state(core), advanced)
    core.step(1)
    assert set(core.query_provider_sensor_data()) == INSTANCE_IDS
    assert all(np.isfinite(value).all() for value in saved.values())


def test_later_substep_compute_failure_never_publishes_and_reset_recovers(provider_core, monkeypatch):
    core, _ = provider_core
    core.step(2)
    before = physical_state(core)
    saved = core.query_provider_sensor_data()
    snapshots = {name: value.copy() for name, value in saved.items()}
    native_compute = SensorHost.compute
    native_publish = SensorHost.publish
    computed = []
    published = []

    def fail_after_staging(host, instance_id, sample, **kwargs):
        result = native_compute(host, instance_id, sample, **kwargs)
        computed.append((instance_id, sample.step_index))
        if sample.step_index == 3 and instance_id == "amplified":
            raise RuntimeError("injected provider compute failure")
        return result

    def record_publish(host):
        published.append(True)
        return native_publish(host)

    monkeypatch.setattr(SensorHost, "compute", fail_after_staging)
    monkeypatch.setattr(SensorHost, "publish", record_publish)
    with pytest.raises(RuntimeError, match="injected provider compute failure"):
        core.step(4)
    assert not published
    assert {name for name, index in computed if index == 2} == INSTANCE_IDS
    assert {index for _, index in computed} == {2, 3}
    advanced = physical_state(core)
    assert advanced["time"] == pytest.approx(before["time"] + 0.002)
    with pytest.raises(RuntimeError):
        core.query_provider_sensor_data()
    calls_before = len(computed)
    with pytest.raises(RuntimeError):
        core.step(1)
    assert len(computed) == calls_before and not published
    assert_same_state(physical_state(core), advanced)
    for name in INSTANCE_IDS:
        np.testing.assert_array_equal(saved[name], snapshots[name])
    core.reset_provider_sensors(17)
    assert_same_state(physical_state(core), advanced)
    core.step(1)
    assert published == [True]
    assert {name for name, index in computed if index == 0} == INSTANCE_IDS
    assert set(core.query_provider_sensor_data()) == INSTANCE_IDS


def test_reload_releases_host_rebinds_and_close_is_idempotent(provider_core, sdk_build, monkeypatch):
    core, path = provider_core
    closed = []
    native_close = SensorHost.close

    def record_close(host):
        closed.append(id(host))
        return native_close(host)

    monkeypatch.setattr(SensorHost, "close", record_close)
    core.step(5)
    reference = core.query_provider_sensor_data()
    with pytest.raises(RuntimeError):
        core.configure_provider_sensors(sdk_build[0], manifests(sdk_build))
    core.init_simulation(str(path))
    assert closed and physical_state(core)["time"] == 0
    with pytest.raises(RuntimeError):
        core.query_provider_sensor_data()
    core.step(5)
    for name, value in core.query_provider_sensor_data().items():
        np.testing.assert_array_equal(value, reference[name])
    core.close_provider_sensors()
    core.close_provider_sensors()
    core.init_simulation(str(path))
    core.step(1)
    assert set(core.query_provider_sensor_data()) == INSTANCE_IDS


def test_rk4_is_rejected_before_any_physics_advance(sdk_build, tmp_path, monkeypatch):
    path = save_scene(tmp_path, scene_root(integrator="RK4"))
    native_step = mujoco.mj_step
    advanced = []

    def record_step(*args, **kwargs):
        advanced.append(True)
        return native_step(*args, **kwargs)

    monkeypatch.setattr(mujoco, "mj_step", record_step)
    core = MuJoCoSimCore()
    core.configure_provider_sensors(sdk_build[0], manifests(sdk_build))
    try:
        with pytest.raises((ValueError, RuntimeError), match="RK4|integrator"):
            core.init_simulation(str(path))
            core.step(1)
        assert not advanced
    finally:
        core.close_provider_sensors()


@pytest.mark.parametrize("legacy", [False, True])
def test_unconfigured_metadata_is_inert_and_does_not_load_providers(tmp_path, monkeypatch, legacy):
    root = scene_root(declarations=not legacy)
    if legacy:
        custom = ET.SubElement(root, "custom")
        ET.SubElement(custom, "text", name="orca.sensor.instances.v1", data=json.dumps([]))
    else:
        root.find("custom/text").set("data", "com.not_installed.sensor")
    path = save_scene(tmp_path, root)

    def unexpected_registration(*args, **kwargs):
        pytest.fail("Unconfigured core must not load any provider")

    monkeypatch.setattr(SensorHost, "register_provider", unexpected_registration)
    core = MuJoCoSimCore()
    core.init_simulation(str(path))
    core.step(3)
    assert physical_state(core)["time"] == pytest.approx(0.003)
    assert core.query_provider_sensor_data() == {}
    core.close_provider_sensors()
    core.close_provider_sensors()


def test_configured_core_rejects_legacy_json_declarations(sdk_build, tmp_path):
    root = scene_root(declarations=False)
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name="orca.sensor.instances.v1", data=json.dumps([]))
    path = save_scene(tmp_path, root)
    core = MuJoCoSimCore()
    core.configure_provider_sensors(sdk_build[0], manifests(sdk_build))
    try:
        with pytest.raises(ValueError):
            core.init_simulation(str(path))
    finally:
        core.close_provider_sensors()
