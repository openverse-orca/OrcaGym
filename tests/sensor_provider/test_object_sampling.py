"""Object inputs sample the existing SimCore state, without another backend.

The recording runtime replaces only provider computation and the declaration
reader. Real MuJoCo state, contact forces, frames and stepping remain in SimCore;
parser, contract routing and real vendor computation have separate tests.
"""

import mujoco
import numpy as np
import pytest

from orca_gym.core.euler import provider_sensor_sampling as sampling
from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.sensor.providers.queries import CompiledQuery, QueryField


class RecordingRuntime:
    def __init__(self, queries):
        self.sampling_queries = tuple(queries)
        self.samples = []
        self.published = 0
        self.invalidated = False
        self.closed = False

    def compute(self, stamp, values):
        self.samples.append((stamp, {query: {key: value.copy() for key, value in columns.items()}
                                     for query, columns in values.items()}))

    def publish(self):
        self.published += 1

    def invalidate(self):
        self.invalidated = True

    def reset(self, seed):
        self.samples.clear()
        self.invalidated = False

    def close(self):
        self.closed = True


@pytest.fixture
def sampling_core(tmp_path, monkeypatch):
    cores = []
    # A trusted dummy path is sufficient because this test never loads a DLL.
    host = tmp_path / "host.so"
    host.write_bytes(b"unused recording runtime")
    manifest = tmp_path / "provider.json"
    manifest.write_text("{}")

    def create(xml, queries):
        path = tmp_path / f"scene_{len(cores)}.xml"
        path.write_text(xml)
        runtime = RecordingRuntime(queries)
        monkeypatch.setattr(sampling, "read_custom_sensor_instances", lambda root: [{"instance_id": "recording"}])
        monkeypatch.setattr(sampling, "SampledSensorRuntime", lambda *args, **kwargs: runtime)
        core = MuJoCoSimCore()
        cores.append(core)
        core.configure_provider_sensors(host, [manifest])
        core.init_simulation(str(path))
        return core, runtime

    yield create
    for core in cores:
        core.close_provider_sensors()


def contact_query(objects, surfaces, *, frame="world", capacity=32, instance_geoms=None):
    fields = {"surface_index": QueryField("u32"), "counterpart_rigid_body_id": QueryField("u64")}
    fields.update({name: QueryField("f64", (3,)) for name in ("position", "normal_force", "tangential_force")})
    return CompiledQuery(fields, "orca.contact.v1", {
        "requirement": {"name": "contacts", "capability": "orca.contact.v1", "objects": surfaces,
                        "frame": frame, "capacity": capacity, "exclude_internal": True},
        "objects": objects, "instance_geoms": instance_geoms,
    })


CONTACT_SCENE = '''<mujoco>
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="left_root" pos="-.04 0 .1">
      <freejoint/>
      <site name="left_frame" pos=".005 .003 0" quat=".7071067811865476 0 .7071067811865476 0"/>
      <body name="left_pad"><geom name="left_geom" type="sphere" size=".05" mass="1"/></body>
      <body name="spare_pad" pos="0 -.3 0"><geom name="spare_geom" type="sphere" size=".01" mass=".1"/></body>
    </body>
    <body name="right_root" pos=".04 0 .1">
      <freejoint/>
      <site name="right_frame"/>
      <body name="right_pad"><geom name="right_geom" type="sphere" size=".05" mass="1"/></body>
    </body>
  </worldbody>
</mujoco>'''


def test_contact_sign_frames_surface_order_welded_counterpart_and_shared_substeps(sampling_core, monkeypatch):
    left_objects = {"spare": {"kind": "body", "name": "spare_pad"},
                    "pad": {"kind": "body", "name": "left_pad"},
                    "frame": {"kind": "site", "name": "left_frame"}}
    right_objects = {"pad": {"kind": "geom", "name": "right_geom"}}
    local = contact_query(left_objects, ["spare", "pad"], frame="frame")
    left_world = contact_query(left_objects, ["spare", "pad"])
    right_world = contact_query(right_objects, ["pad"])
    grid = CompiledQuery({"value": QueryField("f64", (2, 2, 3))}, "orca.contact_grid.v1", {
        "site": "left_frame", "requirement": {"name": "grid", "capability": "orca.contact_grid.v1",
                                               "resolution": [2, 2], "fov_degrees": [180, 180]},
    })
    oracle_model = mujoco.MjModel.from_xml_string(CONTACT_SCENE)
    oracle_data = mujoco.MjData(oracle_model)
    mujoco.mj_step(oracle_model, oracle_data, 3)
    native_step, native_force = mujoco.mj_step, mujoco.mj_contactForce
    native_model, native_data = mujoco.MjModel.from_xml_path, mujoco.MjData
    calls = {"steps": [], "forces": 0, "model": 0, "data": 0}

    def step(model, data, nstep=1):
        calls["steps"].append(nstep)
        return native_step(model, data, nstep)

    def force(*args):
        calls["forces"] += 1
        return native_force(*args)

    def model(*args, **kwargs):
        calls["model"] += 1
        return native_model(*args, **kwargs)

    def data(*args, **kwargs):
        calls["data"] += 1
        return native_data(*args, **kwargs)

    monkeypatch.setattr(mujoco, "mj_step", step)
    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", model)
    monkeypatch.setattr(mujoco, "MjData", data)
    core, runtime = sampling_core(CONTACT_SCENE, [local, left_world, right_world, grid])
    core.step(3)
    assert calls == {"steps": [1, 1, 1], "forces": 3, "model": 1, "data": 1}
    assert runtime.published == 1
    assert [sample[0].index for sample in runtime.samples] == [0, 1, 2]
    assert [sample[0].time for sample in runtime.samples] == pytest.approx([0, .001, .002])
    values = runtime.samples[-1][1]
    left, right, rotated = (values[query] for query in (left_world, right_world, local))
    assert left["surface_index"].tolist() == [1]
    assert right["surface_index"].tolist() == [0]
    assert left["surface_index"].dtype == np.uint32
    assert left["counterpart_rigid_body_id"].dtype == np.uint64
    np.testing.assert_allclose(left["normal_force"], -right["normal_force"])
    np.testing.assert_allclose(left["tangential_force"], -right["tangential_force"])
    assert np.linalg.norm(left["normal_force"]) > 0
    left_geom = mujoco.mj_name2id(oracle_model, mujoco.mjtObj.mjOBJ_GEOM, "left_geom")
    right_pad = mujoco.mj_name2id(oracle_model, mujoco.mjtObj.mjOBJ_BODY, "right_pad")
    right_root = mujoco.mj_name2id(oracle_model, mujoco.mjtObj.mjOBJ_BODY, "right_root")
    assert right_pad != right_root
    assert left["counterpart_rigid_body_id"].tolist() == [right_root]
    contact = oracle_data.contact[0]
    expected_force = np.empty(6)
    native_force(oracle_model, oracle_data, 0, expected_force)
    sign = -1 if contact.geom1 == left_geom else 1
    expected_normal = sign * contact.frame.reshape(3, 3)[0] * expected_force[0]
    np.testing.assert_allclose(left["normal_force"][0], expected_normal)
    site = mujoco.mj_name2id(oracle_model, mujoco.mjtObj.mjOBJ_SITE, "left_frame")
    rotation = oracle_data.site_xmat[site].reshape(3, 3).T
    np.testing.assert_allclose(rotated["normal_force"][0], rotation @ expected_normal)
    np.testing.assert_allclose(rotated["position"][0], rotation @ (contact.pos - oracle_data.site_xpos[site]))
    np.testing.assert_allclose(values[grid]["value"].sum(axis=(0, 1)),
                               (rotated["normal_force"] + rotated["tangential_force"]).sum(axis=0))
    view = OrcaGymDataView()
    core.sync_to_view(view)
    assert view.time == oracle_data.time
    np.testing.assert_array_equal(view.qpos, oracle_data.qpos)
    np.testing.assert_array_equal(view.qvel, oracle_data.qvel)


RAY_SCENE = '''<mujoco>
  <option timestep=".001" gravity="0 0 0"/>
  <worldbody><body name="hand">
    <geom name="other_hand" type="sphere" pos=".12 0 0" size=".01"/>
    <body name="sensor" quat=".7071067811865476 0 .7071067811865476 0">
      <site name="frame"/>
      <geom name="frame_geom" size=".001" contype="0" conaffinity="0"/>
      <geom name="own_geom" type="sphere" pos="0 0 .03" size=".01"/>
      <body name="nested"><geom name="nested_geom" type="sphere" pos="0 0 .07" size=".01"/></body>
    </body>
  </body></worldbody>
</mujoco>'''


@pytest.mark.parametrize("kind,name", [("body", "sensor"), ("geom", "frame_geom"), ("site", "frame")])
@pytest.mark.parametrize("owned,exclude,expected", [(None, True, .06),
                                                     (["own_geom", "frame_geom", "nested_geom"], True, .11),
                                                     (None, False, .02)])
def test_object_ray_uses_frame_and_only_direct_owned_geometry(sampling_core, kind, name, owned, exclude, expected):
    query = CompiledQuery({"value": QueryField("f64")}, "orca.raycast.v1", {
        "requirement": {"name": "ray", "capability": "orca.raycast.v1", "frame": "frame",
                        "direction": [0, 0, 1], "max_distance": 1, "exclude_self": exclude},
        "objects": {"frame": {"kind": kind, "name": name}, "shell": {"kind": "body", "name": "sensor"}},
        "instance_geoms": owned,
    })
    core, runtime = sampling_core(RAY_SCENE, [query])
    core.step(1)
    assert runtime.samples[0][1][query]["value"] == pytest.approx(expected)


def test_site_only_object_ray_requires_explicit_ownership(sampling_core):
    query = CompiledQuery({"value": QueryField("f64")}, "orca.raycast.v1", {
        "requirement": {"name": "ray", "capability": "orca.raycast.v1", "frame": "frame",
                        "direction": [0, 0, 1], "max_distance": 1},
        "objects": {"frame": {"kind": "site", "name": "frame"}}, "instance_geoms": None,
    })
    with pytest.raises(ValueError, match="instance_geoms"):
        sampling_core(RAY_SCENE, [query])


def test_contact_capacity_overflow_faults_whole_batch_without_truncation(sampling_core):
    xml = '''<mujoco><option timestep=".001"/><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="pad" pos="0 0 .09"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body>
    </worldbody></mujoco>'''
    query = contact_query({"pad": {"kind": "body", "name": "pad"}}, ["pad"], capacity=1)
    core, runtime = sampling_core(xml, [query])
    with pytest.raises(ValueError, match="capacity exceeded; no truncation"):
        core.step(1)
    assert runtime.invalidated and runtime.samples == [] and runtime.published == 0
    with pytest.raises(RuntimeError, match="faulted"):
        core.query_provider_sensor_data()
    view = OrcaGymDataView()
    core.sync_to_view(view)
    assert view.time == .001


def test_contact_surface_frame_across_moving_joint_is_rejected(sampling_core):
    xml = '''<mujoco><worldbody><body name="root">
      <joint type="hinge"/><geom size=".01"/><site name="fixed_frame"/>
      <body name="moving" pos="0 0 .1"><joint type="hinge"/><geom size=".02"/></body>
    </body></worldbody></mujoco>'''
    query = contact_query({"pad": {"kind": "body", "name": "moving"},
                           "frame": {"kind": "site", "name": "fixed_frame"}}, ["pad"], frame="frame")
    with pytest.raises(ValueError, match="same rigid weld group"):
        sampling_core(xml, [query])


def test_explicit_ownership_cannot_omit_a_contact_surface(sampling_core):
    query = contact_query({"pad": {"kind": "body", "name": "left_pad"}}, ["pad"],
                           instance_geoms=["spare_geom"])
    with pytest.raises(ValueError, match="include every bound surface"):
        sampling_core(CONTACT_SCENE, [query])
