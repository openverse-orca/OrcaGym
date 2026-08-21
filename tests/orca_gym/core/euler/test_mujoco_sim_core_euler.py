"""MuJoCoSimCoreEuler — lazy 同步原语 + 维度 property 单元测试（Phase B）。

用 FakeSolver（无 GPU、无 orca.euler）验收 lazy 同步状态机（L1/L5）：
_mark_dirty/_commit_if_dirty（H2D 合并）、_mark_stale/_ensure_host_fresh（D2H 幂等）、
维度 property（无同步）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import os
import types
import unittest

import mujoco
import numpy as np

from orca_gym.core.euler.mujoco_sim_core_euler import MuJoCoSimCoreEuler


# G1 模型 XML（Phase D 查询功能测试用，含 free joint + 29 hinge joint +
# sensors/actuators/sites/geoms）
_G1_XML = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "environment", "euler", "fixtures", "g1_29dof_camera_simplified.xml",
))


def _make_fake_mj_model() -> types.SimpleNamespace:
    """提供 nq/nv/nu/site_bodyid 的最小 mj_model stub。"""
    return types.SimpleNamespace(nq=3, nv=3, nu=1, site_bodyid=np.array([0], dtype=np.int32))


def _make_fake_host() -> types.SimpleNamespace:
    """提供常用 host 字段的 numpy 数组 stub。"""
    return types.SimpleNamespace(
        qpos=np.zeros(3),
        qvel=np.zeros(3),
        ctrl=np.zeros(1),
        xfrc_applied=np.zeros((2, 6)),   # 2 bodies
        mocap_pos=np.zeros((0, 3)),
        mocap_quat=np.zeros((0, 4)),
    )


class FakeSolver:
    """记录 flush/sync_to_host 调用次数 + 提供 host/mj_model。"""

    def __init__(self) -> None:
        self.flush_count = 0
        self.sync_count = 0
        self.mj_model = _make_fake_mj_model()   # 提供 nq=3, nv=3, nu=1
        self.host = _make_fake_host()           # 提供 qpos/qvel/ctrl/xfrc_applied 等

    def flush(self) -> None:
        self.flush_count += 1

    def sync_to_host(self) -> None:
        self.sync_count += 1

    def step(self, nstep: int) -> None:
        pass

    def forward(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def apply_body_force(self, body_id, force, torque) -> None:
        pass


class TestLazySyncPrimitives(unittest.TestCase):
    """lazy 同步状态机验收（L5 状态转换表）。"""

    def test_init_flags(self):
        """构造后 _host_dirty=False、_host_stale=True、_solver=None。"""
        core = MuJoCoSimCoreEuler()
        self.assertIs(core._solver, None)
        self.assertIs(core._host_dirty, False)
        self.assertIs(core._host_stale, True)

    def test_mark_dirty_then_commit(self):
        """_mark_dirty 后 _commit_if_dirty 只 flush 1 次且 dirty 归 False。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._mark_dirty()
        core._commit_if_dirty()
        self.assertEqual(core._solver.flush_count, 1)
        self.assertIs(core._host_dirty, False)

    def test_commit_merges_multiple_dirty(self):
        """连续多次 _mark_dirty 后 _commit_if_dirty 仍只 flush 1 次（合并）。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._mark_dirty()
        core._mark_dirty()
        core._commit_if_dirty()
        self.assertEqual(core._solver.flush_count, 1)
        self.assertIs(core._host_dirty, False)

    def test_commit_noop_when_clean(self):
        """无 dirty 时 _commit_if_dirty 不调 flush。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._commit_if_dirty()
        self.assertEqual(core._solver.flush_count, 0)

    def test_mark_stale_then_ensure(self):
        """_mark_stale 后 _ensure_host_fresh 只 sync 1 次且 stale 归 False。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._host_stale = False
        core._mark_stale()
        core._ensure_host_fresh()
        self.assertEqual(core._solver.sync_count, 1)
        self.assertIs(core._host_stale, False)

    def test_ensure_idempotent_when_fresh(self):
        """无 stale 时 _ensure_host_fresh 不调 sync_to_host（幂等）。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._host_stale = False
        core._ensure_host_fresh()
        self.assertEqual(core._solver.sync_count, 0)


class TestDimensionProperties(unittest.TestCase):
    """维度 property（G 类）验收。"""

    def test_dimension_properties(self):
        """nq/nv/nu 返回正确值，且不触发 flush/sync（sync 计数为 0）。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()  # nq=3, nv=3, nu=1
        self.assertEqual(core.nq, 3)
        self.assertEqual(core.nv, 3)
        self.assertEqual(core.nu, 1)
        self.assertEqual(core._solver.flush_count, 0)
        self.assertEqual(core._solver.sync_count, 0)

    def test_dimension_uninit_raises(self):
        """_solver 为 None 时 nq 抛 RuntimeError('Simulation not initialized')。"""
        core = MuJoCoSimCoreEuler()
        with self.assertRaises(RuntimeError) as ctx:
            _ = core.nq
        self.assertEqual(str(ctx.exception), "Simulation not initialized")


class FakeDataView:
    """记录 _sync_from_mjdata 调用（供 sync_to_view 验收）。"""

    def __init__(self) -> None:
        self.sync_calls = 0
        self.last_host = None
        self.last_model = None

    def _sync_from_mjdata(self, host, model) -> None:
        self.sync_calls += 1
        self.last_host = host
        self.last_model = model


class TestLifecycleMethods(unittest.TestCase):
    """A 类生命周期方法验收（step/forward/reset/sync_to_view 状态转换）。"""

    def test_step_flushes_dirty_then_marks_stale(self):
        """step 前置 flush（H2D）后置 stale：dirty→False、stale→True、flush=1。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._mark_dirty()
        core.step(1)
        self.assertEqual(core._solver.flush_count, 1)
        self.assertIs(core._host_dirty, False)
        self.assertIs(core._host_stale, True)

    def test_forward_flushes_dirty_then_marks_stale(self):
        """forward 前置 flush 后置 stale。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._mark_dirty()
        core.forward()
        self.assertEqual(core._solver.flush_count, 1)
        self.assertIs(core._host_dirty, False)
        self.assertIs(core._host_stale, True)

    def test_sync_to_view_ensures_fresh(self):
        """sync_to_view 前置 D2H（sync=1）且清 stale，并回调 DataView。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._host_stale = True
        view = FakeDataView()
        core.sync_to_view(view)
        self.assertEqual(core._solver.sync_count, 1)
        self.assertIs(core._host_stale, False)
        self.assertEqual(view.sync_calls, 1)
        self.assertIs(view.last_host, core._solver.host)
        self.assertIs(view.last_model, core._solver.mj_model)

    def test_sync_to_view_idempotent_when_fresh(self):
        """无 stale 时 sync_to_view 不重复 sync（sync=0）。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._host_stale = False
        view = FakeDataView()
        core.sync_to_view(view)
        self.assertEqual(core._solver.sync_count, 0)
        self.assertEqual(view.sync_calls, 1)

    def test_reset_data_clears_flags(self):
        """reset_data 清 dirty/stale 两标志。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._host_dirty = True
        core._host_stale = True
        core.reset_data()
        self.assertIs(core._host_dirty, False)
        self.assertIs(core._host_stale, False)

    def test_step_uninitialized_raises(self):
        """_solver 为 None 时 step 抛 RuntimeError。"""
        core = MuJoCoSimCoreEuler()
        with self.assertRaises(RuntimeError):
            core.step(1)

    def test_init_simulation_nworld_not_one_raises(self):
        """nworld != 1 抛 NotImplementedError（且不 import orca.euler）。"""
        core = MuJoCoSimCoreEuler()
        with self.assertRaises(NotImplementedError):
            core.init_simulation("model.xml", device="cuda", nworld=2)


class TestWriteMethods(unittest.TestCase):
    """B 类写入方法验收：写后 _mark_dirty、step 时合并 flush。"""

    def test_set_ctrl_marks_dirty(self):
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core.set_ctrl(np.zeros(1))
        self.assertIs(core._host_dirty, True)
        self.assertEqual(core._solver.flush_count, 0)

    def test_set_qpos_qvel_marks_dirty(self):
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core.set_qpos_qvel(np.zeros(3), np.zeros(3))
        self.assertIs(core._host_dirty, True)

    def test_set_mocap_pos_and_quat_empty_marks_dirty(self):
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core.set_mocap_pos_and_quat({})
        self.assertIs(core._host_dirty, True)

    def test_write_then_step_flushes_once(self):
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core.set_ctrl(np.zeros(1))
        core.step(1)
        self.assertEqual(core._solver.flush_count, 1)

    def test_multiple_writes_merged_single_flush(self):
        """连续 set_ctrl + apply_body_force 后 step 只 flush 1 次（合并）。"""
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core.set_ctrl(np.zeros(1))
        core.apply_body_force(0, np.zeros(3), np.zeros(3))
        core.step(1)
        self.assertEqual(core._solver.flush_count, 1)


class TestForceMethods(unittest.TestCase):
    """C 类力应用方法验收：写 host.xfrc_applied + _mark_dirty。"""

    def test_apply_body_force_marks_dirty(self):
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core.apply_body_force(0, np.zeros(3), np.zeros(3))
        self.assertIs(core._host_dirty, True)

    def test_clear_body_force_marks_dirty_and_zeroes(self):
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._solver.host.xfrc_applied[0, :] = 1.0
        core.clear_body_force(0)
        self.assertIs(core._host_dirty, True)
        self.assertEqual(float(core._solver.host.xfrc_applied[0].sum()), 0.0)

    def test_clear_all_forces_marks_dirty_and_zeroes(self):
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._solver.host.xfrc_applied[:] = 1.0
        core.clear_all_forces()
        self.assertIs(core._host_dirty, True)
        self.assertEqual(float(core._solver.host.xfrc_applied.sum()), 0.0)

    def test_mj_apply_force_at_site_marks_dirty(self):
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core.mj_apply_force_at_site(0, np.zeros(3), np.zeros(3))
        self.assertIs(core._host_dirty, True)

    def test_mj_clear_xfrc_applied_for_site_marks_dirty(self):
        core = MuJoCoSimCoreEuler()
        core._solver = FakeSolver()
        core._solver.host.xfrc_applied[0, :] = 1.0
        core.mj_clear_xfrc_applied_for_site(0)
        self.assertIs(core._host_dirty, True)
        self.assertEqual(float(core._solver.host.xfrc_applied[0].sum()), 0.0)


class RealDataSolver:
    """用真实 mujoco.MjModel/MjData 包装，统计 flush/sync 次数（Phase D）。"""

    def __init__(self, model, data) -> None:
        self.flush_count = 0
        self.sync_count = 0
        self.mj_model = model
        self.host = data

    def flush(self) -> None:
        self.flush_count += 1

    def sync_to_host(self) -> None:
        self.sync_count += 1

    def step(self, nstep: int) -> None:
        pass

    def forward(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def apply_body_force(self, body_id, force, torque) -> None:
        pass


def _make_real_core():
    """构造带真实 G1 模型 + 统计 solver 的 MuJoCoSimCoreEuler（未 forward）。"""
    model = mujoco.MjModel.from_xml_path(_G1_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    core = MuJoCoSimCoreEuler()
    core._solver = RealDataSolver(model, data)
    return core


class TestQueryD2HBehavior(unittest.TestCase):
    """D 类查询方法 lazy-D2H 触发验收（每个查询 sync 1 次、读不 flush）。"""

    def test_query_triggers_sync_once_then_fresh(self):
        core = _make_real_core()
        core._host_stale = True
        core.query_joint_qpos(["left_hip_pitch_joint"])
        self.assertEqual(core._solver.sync_count, 1)
        self.assertIs(core._host_stale, False)
        self.assertEqual(core._solver.flush_count, 0)

    def test_consecutive_queries_sync_once(self):
        core = _make_real_core()
        core._host_stale = True
        core.query_joint_qpos(["left_hip_pitch_joint"])
        core.query_body_xpos_xmat_xquat(["pelvis"])
        core.query_site_pos_and_mat(["imu"])
        self.assertEqual(core._solver.sync_count, 1)

    def test_read_queries_never_flush(self):
        core = _make_real_core()
        core._host_stale = True
        core.query_joint_qvel(["floating_base_joint"])
        core.get_cfrc_ext()
        core.query_sensor_data(["left_hip_pitch_pos"], {})
        core.query_actuator_torques(["left_hip_pitch"])
        self.assertEqual(core._solver.flush_count, 0)
        self.assertEqual(core._solver.sync_count, 1)

    def test_query_uninitialized_raises(self):
        core = MuJoCoSimCoreEuler()
        with self.assertRaises(RuntimeError):
            core.query_joint_qpos(["left_hip_pitch_joint"])


class TestQueryReturnTypes(unittest.TestCase):
    """D 类返回类型验收：offsets/lengths 元组、adr 返回 Python int。"""

    def setUp(self):
        self.core = _make_real_core()

    def test_offsets_lengths_return_tuples_of_arrays(self):
        names = ["floating_base_joint", "left_hip_pitch_joint"]
        off = self.core.query_joint_offsets(names)
        lens = self.core.query_joint_lengths(names)
        self.assertIsInstance(off, tuple)
        self.assertIsInstance(lens, tuple)
        self.assertEqual(len(off), 3)
        self.assertEqual(len(lens), 3)
        for arr in off + lens:
            self.assertIsInstance(arr, np.ndarray)

    def test_free_joint_lengths(self):
        names = ["floating_base_joint", "left_hip_pitch_joint"]
        qpos_len, qvel_len, qacc_len = self.core.query_joint_lengths(names)
        self.assertEqual(int(qpos_len[0]), 7)
        self.assertEqual(int(qvel_len[0]), 6)
        self.assertEqual(int(qpos_len[1]), 1)
        self.assertEqual(int(qvel_len[1]), 1)

    def test_jnt_qposadr_dofadr_return_python_int(self):
        self.assertIsInstance(self.core.jnt_qposadr("floating_base_joint"), int)
        self.assertIsInstance(self.core.jnt_dofadr("floating_base_joint"), int)
        self.assertNotIsInstance(
            self.core.jnt_qposadr("left_hip_pitch_joint"), np.integer
        )

    def test_query_joint_dofadrs_returns_dict_of_int(self):
        result = self.core.query_joint_dofadrs(
            ["floating_base_joint", "left_hip_pitch_joint"]
        )
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertIsInstance(v, int)


class TestQueryNumericalConsistency(unittest.TestCase):
    """D 类数值一致性验收：Euler 查询结果与宿主 MjModel/MjData 直接读取一致。"""

    def setUp(self):
        self.core = _make_real_core()
        self.model = self.core._solver.mj_model
        self.data = self.core._solver.host

    def test_query_joint_qpos_matches_host(self):
        name = "left_hip_pitch_joint"
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = int(self.model.jnt_qposadr[jid])
        np.testing.assert_array_equal(
            self.core.query_joint_qpos([name])[name], self.data.qpos[adr:adr + 1]
        )

    def test_query_joint_qvel_qacc(self):
        name = "floating_base_joint"
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = int(self.model.jnt_dofadr[jid])
        np.testing.assert_array_equal(
            self.core.query_joint_qvel([name])[name], self.data.qvel[adr:adr + 6]
        )
        np.testing.assert_array_equal(
            self.core.query_joint_qacc([name])[name], self.data.qacc[adr:adr + 6]
        )

    def test_query_body_matches_host(self):
        name = "pelvis"
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        result = self.core.query_body_xpos_xmat_xquat([name])[name]
        np.testing.assert_array_equal(result["xpos"], self.data.xpos[bid])
        np.testing.assert_array_equal(result["xmat"], self.data.xmat[bid].reshape(3, 3))
        np.testing.assert_array_equal(result["xquat"], self.data.xquat[bid])

    def test_query_site_matches_host(self):
        pos_mat = self.core.query_site_pos_and_mat(["imu"])["imu"]
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        np.testing.assert_array_equal(pos_mat["xpos"], self.data.site_xpos[sid])
        np.testing.assert_array_equal(
            pos_mat["xmat"], self.data.site_xmat[sid].reshape(3, 3)
        )
        np.testing.assert_array_equal(
            self.core.query_site_size(["imu"])["imu"], self.model.site_size[sid]
        )

    def test_query_sensor_matches_host(self):
        name = "left_hip_pitch_pos"
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        expected = self.data.sensordata[adr:adr + dim]
        np.testing.assert_array_equal(
            self.core.query_sensor_data([name], {})[name], expected
        )
        np.testing.assert_array_equal(
            self.core.query_sensor_data([name], {name: {"adr": adr, "dim": dim}})[name],
            expected,
        )

    def test_query_actuator_torques_matches_host(self):
        name = "left_hip_pitch"
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        np.testing.assert_array_equal(
            self.core.query_actuator_torques([name])[name],
            self.data.actuator_force[aid:aid + 1],
        )

    def test_query_contact_simple_returns_list(self):
        contacts = self.core.query_contact_simple()
        self.assertIsInstance(contacts, list)
        self.assertEqual(len(contacts), self.data.ncon)

    def test_query_contact_force(self):
        if self.data.ncon == 0:
            self.skipTest("G1 初始姿态无接触")
        cid = 0
        expected = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, cid, expected)
        np.testing.assert_array_equal(
            self.core.query_contact_force([cid])[cid], expected
        )

    def test_get_cfrc_ext_shape(self):
        cfrc = self.core.get_cfrc_ext()
        self.assertEqual(cfrc.shape, (self.model.nbody, 6))

    def test_get_goal_bounding_box_matches_host(self):
        name = "manipulation_box_geom"
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        np.testing.assert_array_equal(
            self.core.get_goal_bounding_box(name), self.model.geom_size[gid]
        )


class TestJacMethods(unittest.TestCase):
    """E 类雅可比方法验收：原地写、返回 None、数值与 mujoco 一致、D2H 触发。"""

    def setUp(self):
        self.core = _make_real_core()
        self.model = self.core._solver.mj_model
        self.data = self.core._solver.host

    def test_mj_jacBody_writes_inplace_and_matches(self):
        self.core._host_stale = True
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        ret = self.core.mj_jacBody(jacp, jacr, bid)
        self.assertIsNone(ret)
        ref_p = np.zeros((3, self.model.nv))
        ref_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, ref_p, ref_r, bid)
        np.testing.assert_allclose(jacp, ref_p)
        np.testing.assert_allclose(jacr, ref_r)
        self.assertEqual(self.core._solver.sync_count, 1)
        self.assertIs(self.core._host_stale, False)
        self.assertEqual(self.core._solver.flush_count, 0)

    def test_mj_jacSite_writes_inplace_and_matches(self):
        self.core._host_stale = True
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        ret = self.core.mj_jacSite(jacp, jacr, sid)
        self.assertIsNone(ret)
        ref_p = np.zeros((3, self.model.nv))
        ref_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, ref_p, ref_r, sid)
        np.testing.assert_allclose(jacp, ref_p)
        np.testing.assert_allclose(jacr, ref_r)

    def test_mj_jac_site_returns_dict(self):
        result = self.core.mj_jac_site(["imu", "camera_head_site"])
        self.assertIsInstance(result, dict)
        for name in ("imu", "camera_head_site"):
            self.assertIn(name, result)
            self.assertEqual(result[name]["jacp"].shape, (3, self.model.nv))
            self.assertEqual(result[name]["jacr"].shape, (3, self.model.nv))

    def test_query_body_xvel_matches_jac(self):
        """D 类 xvel 方法现在可用（依赖 E 类 mj_jacBody）。"""
        name = "torso_link"
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        result = self.core.query_body_xpos_xmat_xquat_xvel([name])[name]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bid)
        np.testing.assert_allclose(result["xvel"], jacp @ self.data.qvel)
        self.assertEqual(result["xvel"].shape, (3,))


class TestModelWriteMethods(unittest.TestCase):
    """F 类模型写入方法验收：统一抛 NotImplementedError 且信息含方法名 + P2。"""

    CASES = [
        ("set_geom_friction", ({},)),
        ("add_extra_weight", ({},)),
        ("update_equality_constraints", ([],)),
        ("modify_equality_objects", ([],)),
        ("set_equality_active", (0, True)),
        ("set_equality_solref", (0, np.zeros(2))),
        ("set_equality_solimp", (0, np.zeros(1))),
    ]

    def test_all_f_methods_raise_not_implemented_with_method_name_and_p2(self):
        core = MuJoCoSimCoreEuler()
        for method, args in self.CASES:
            with self.subTest(method=method):
                with self.assertRaises(NotImplementedError) as ctx:
                    getattr(core, method)(*args)
                msg = str(ctx.exception)
                self.assertIn(method, msg)
                self.assertIn("P2", msg)


if __name__ == "__main__":
    unittest.main()