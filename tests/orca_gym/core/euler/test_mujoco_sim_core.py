"""阶段二 Step 1: MuJoCoSimCore 功能验收测试。

验证 MuJoCoSimCore 的真实 MuJoCo 操作（init/step/forward/set_ctrl/
reset_data/set_qpos_qvel/sync_to_view）和维度 property 功能正确
（架构 §5.3, §12.2）。力应用方法仍 raise NotImplementedError（留待完整 P4）。

阶段三 3.1.1 扩展：关节查询方法（query_joint_qpos/qvel/qacc/offsets/
lengths/dofadrs, jnt_qposadr/dofadr）的架构遵从性测试 + 功能单元测试。

运行方式:
    <conda-base>/envs/OrcaFlow_Flow/bin/python tests/run_tests.py --component core/euler
"""

import inspect
import os
import unittest

import mujoco
import numpy as np

from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


# 测试用 XML 模型：单铰链倒立摆（nq=1, nv=1, nu=1）
_PENDULUM_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "OrcaPlayground", "envs", "euler", "scenes", "simple_pendulum.xml",
)
_PENDULUM_XML = os.path.abspath(_PENDULUM_XML)

# G1 模型 XML（阶段三 3.1.1 功能测试用，含 free joint + 29 hinge joint）
_G1_XML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "OrcaPlayground", "envs", "euler", "robots", "g1_29dof_camera.xml",
)
_G1_XML = os.path.abspath(_G1_XML)


class TestMuJoCoSimCoreStructure(unittest.TestCase):
    """MuJoCoSimCore 结构验收：私有属性、方法签名、property 存在。"""

    def test_sim_core_constructable(self):
        """MuJoCoSimCore() 可无参构造。"""
        sim = MuJoCoSimCore()
        self.assertIsInstance(sim, MuJoCoSimCore)

    def test_sim_core_has_mj_model_data_private(self):
        """实例有 _mjModel/_mjData 私有属性（带下划线），初始为 None。"""
        sim = MuJoCoSimCore()
        self.assertTrue(hasattr(sim, "_mjModel"))
        self.assertTrue(hasattr(sim, "_mjData"))
        self.assertIsNone(sim._mjModel)
        self.assertIsNone(sim._mjData)

    def test_sim_core_has_lifecycle_methods(self):
        """init_simulation/step/forward/set_ctrl/sync_to_view 方法存在。"""
        for name in ["init_simulation", "step", "forward", "set_ctrl",
                     "sync_to_view", "reset_data", "set_qpos_qvel"]:
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(MuJoCoSimCore, name, None)))

    def test_sim_core_has_force_methods(self):
        """apply_body_force/clear_body_force/clear_all_forces + site 力方法存在。"""
        for name in [
            "apply_body_force", "clear_body_force", "clear_all_forces",
            "mj_apply_force_at_site", "mj_clear_xfrc_applied_for_site",
        ]:
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(MuJoCoSimCore, name, None)))

    def test_sim_core_has_nq_nv_nu_properties(self):
        """nq/nv/nu property 存在。"""
        self.assertIsInstance(MuJoCoSimCore.nq, property)
        self.assertIsInstance(MuJoCoSimCore.nv, property)
        self.assertIsInstance(MuJoCoSimCore.nu, property)

    def test_sim_core_docstring_forbids_external_access(self):
        """docstring 含「禁止」和 _mjModel/_mjData 关键词。"""
        doc = MuJoCoSimCore.__doc__ or ""
        self.assertIn("禁止", doc)
        self.assertIn("_mjModel", doc)
        self.assertIn("_mjData", doc)


class TestSimCoreForceMethodsArchCompliance(unittest.TestCase):
    """子步骤 3.2.1 架构遵从性测试（P2/K11）。

    对应文档 §6.2 架构遵从性测试表。
    """

    def test_simcore_force_methods_write_xfrc_only(self):
        """P2/K11: grep 断言力应用方法只写 xfrc_applied，不返回 MjData/MjModel。"""
        source = inspect.getsource(MuJoCoSimCore)
        # 找到力应用区块
        start = source.find("def apply_body_force")
        self.assertGreater(start, 0)
        block = source[start:]
        end = block.find("# --- 关节查询")
        if end > 0:
            block = block[:end]
        # 写 xfrc_applied（合法）
        self.assertIn("xfrc_applied", block)
        # 不返回 _mjData/_mjModel
        self.assertNotIn("return self._mjData", block)
        self.assertNotIn("return self._mjModel", block)

    def test_simcore_force_methods_return_none(self):
        """K11: apply_body_force/clear_* 返回 None（写操作无返回值）。"""
        sim = MuJoCoSimCore()
        sim.init_simulation(_PENDULUM_XML)
        ret = sim.apply_body_force(0, np.zeros(3), np.zeros(3))
        self.assertIsNone(ret)
        ret = sim.clear_body_force(0)
        self.assertIsNone(ret)
        ret = sim.clear_all_forces()
        self.assertIsNone(ret)
        # 新增 site 力方法也返回 None
        ret = sim.mj_apply_force_at_site(0, np.zeros(3), np.zeros(3))
        self.assertIsNone(ret)
        ret = sim.mj_clear_xfrc_applied_for_site(0)
        self.assertIsNone(ret)

    def test_simcore_force_no_mjdata_leak(self):
        """P2/K11: grep 断言不 return self._mjData。"""
        source = inspect.getsource(MuJoCoSimCore)
        start = source.find("def apply_body_force")
        block = source[start:]
        end = block.find("# --- 关节查询")
        if end > 0:
            block = block[:end]
        self.assertNotIn("return self._mjData", block)


class TestSimCoreForceMethodsFunctional(unittest.TestCase):
    """子步骤 3.2.1 功能单元测试（G1 XML 真实数据）。

    对应文档 §6.2 功能单元测试表。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()
        # pelvis body_id（用于 body 力测试）
        self.pelvis_id = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.imu_site_id = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        # imu site 关联的 body_id（site_bodyid[imu_site_id]）
        self.imu_body_id = int(self.sim._mjModel.site_bodyid[self.imu_site_id])

    def test_apply_body_force_writes_xfrc(self):
        """施力后 _mjData.xfrc_applied[body_id, :3] 等于 force。"""
        force = np.array([1.0, 2.0, 3.0])
        torque = np.array([0.1, 0.2, 0.3])
        self.sim.apply_body_force(self.pelvis_id, force, torque)
        np.testing.assert_allclose(
            self.sim._mjData.xfrc_applied[self.pelvis_id, :3], force
        )
        np.testing.assert_allclose(
            self.sim._mjData.xfrc_applied[self.pelvis_id, 3:6], torque
        )

    def test_clear_body_force_zeroes_xfrc(self):
        """清力后 xfrc_applied[body_id, :6] 为 0。"""
        self.sim.apply_body_force(self.pelvis_id, np.ones(3), np.ones(3))
        self.sim.clear_body_force(self.pelvis_id)
        np.testing.assert_allclose(
            self.sim._mjData.xfrc_applied[self.pelvis_id, :6], 0.0
        )

    def test_clear_all_forces_zeroes_all(self):
        """清全部后 xfrc_applied[:] 为 0。"""
        self.sim.apply_body_force(self.pelvis_id, np.ones(3), np.ones(3))
        self.sim.clear_all_forces()
        np.testing.assert_allclose(self.sim._mjData.xfrc_applied[:], 0.0)

    def test_mj_apply_force_at_site_writes_body_xfrc(self):
        """site 施力后关联 body 的 xfrc 写入（累加）。"""
        force = np.array([0.5, 0.0, 0.0])
        torque = np.array([0.0, 0.5, 0.0])
        # 先清零
        self.sim.clear_all_forces()
        self.sim.mj_apply_force_at_site(self.imu_site_id, force, torque)
        np.testing.assert_allclose(
            self.sim._mjData.xfrc_applied[self.imu_body_id, :3], force
        )
        np.testing.assert_allclose(
            self.sim._mjData.xfrc_applied[self.imu_body_id, 3:6], torque
        )

    def test_mj_clear_xfrc_for_site_clears_body(self):
        """清 site xfrc 后关联 body 的 xfrc 清零。"""
        self.sim.mj_apply_force_at_site(self.imu_site_id, np.ones(3), np.ones(3))
        self.sim.mj_clear_xfrc_applied_for_site(self.imu_site_id)
        np.testing.assert_allclose(
            self.sim._mjData.xfrc_applied[self.imu_body_id, :6], 0.0
        )


# =============================================================================
# 阶段三 3.2.2：MuJoCoSimCore 状态设置方法
# =============================================================================


class TestSimCoreSetMethodsArchCompliance(unittest.TestCase):
    """子步骤 3.2.2 架构遵从性测试（P2/K11）。

    对应文档 §6.3 架构遵从性测试表。
    """

    def test_simcore_set_methods_return_none(self):
        """K11: 3 个设置方法返回 None（写操作无返回值）。"""
        sim = MuJoCoSimCore()
        sim.init_simulation(_G1_XML)
        ret = sim.set_mocap_pos_and_quat(
            {"ActorManipulator_Anchor": {"pos": np.zeros(3), "quat": np.array([1, 0, 0, 0])}}
        )
        self.assertIsNone(ret)
        ret = sim.set_geom_friction({"manipulation_box_geom": np.array([1.0, 0.005, 0.0001])})
        self.assertIsNone(ret)
        ret = sim.add_extra_weight({"pelvis": 1.0})
        self.assertIsNone(ret)

    def test_simcore_set_methods_no_mjdata_leak(self):
        """P2/K11: grep 断言不 return self._mjData/self._mjModel。"""
        source = inspect.getsource(MuJoCoSimCore)
        start = source.find("# --- 状态设置方法（阶段三 3.2.2）")
        self.assertGreater(start, 0)
        block = source[start:]
        end = block.find("# --- 关节查询")
        if end > 0:
            block = block[:end]
        self.assertNotIn("return self._mjData", block)
        self.assertNotIn("return self._mjModel", block)

    def test_simcore_set_geom_friction_writes_model(self):
        """P2: grep 断言 set_geom_friction 写 _mjModel.geom_friction（模型字段，非 data）。"""
        source = inspect.getsource(MuJoCoSimCore)
        start = source.find("def set_geom_friction")
        end = source.find("def add_extra_weight")
        block = source[start:end]
        self.assertIn("_mjModel.geom_friction", block)


class TestSimCoreSetMethodsFunctional(unittest.TestCase):
    """子步骤 3.2.2 功能单元测试（G1 XML 真实数据）。

    对应文档 §6.3 功能单元测试表。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()
        # mocap body id + mocap_id
        self.mocap_body_id = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_BODY, "ActorManipulator_Anchor"
        )
        self.mocap_id = int(self.sim._mjModel.body_mocapid[self.mocap_body_id])
        # geom id
        self.box_geom_id = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_GEOM, "manipulation_box_geom"
        )
        # pelvis body id
        self.pelvis_id = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

    def test_set_mocap_pos_and_quat_writes_mocap(self):
        """mocap_pos/quat 正确写入。"""
        pos = np.array([0.5, 0.3, 0.8])
        quat = np.array([0.7071, 0.0, 0.7071, 0.0])  # w,x,y,z
        self.sim.set_mocap_pos_and_quat(
            {"ActorManipulator_Anchor": {"pos": pos, "quat": quat}}
        )
        np.testing.assert_allclose(
            self.sim._mjData.mocap_pos[self.mocap_id], pos, atol=1e-6
        )
        np.testing.assert_allclose(
            self.sim._mjData.mocap_quat[self.mocap_id], quat, atol=1e-6
        )

    def test_set_geom_friction_persists(self):
        """geom_friction 修改持久化。"""
        new_friction = np.array([2.5, 0.01, 0.002])
        self.sim.set_geom_friction({"manipulation_box_geom": new_friction})
        np.testing.assert_allclose(
            self.sim._mjModel.geom_friction[self.box_geom_id], new_friction, atol=1e-6
        )

    def test_add_extra_weight_increases_mass(self):
        """添加重量后 body_mass 增加。"""
        old_mass = float(self.sim._mjModel.body_mass[self.pelvis_id])
        self.sim.add_extra_weight({"pelvis": 2.0})
        new_mass = float(self.sim._mjModel.body_mass[self.pelvis_id])
        self.assertAlmostEqual(new_mass - old_mass, 2.0, places=6)


class TestMuJoCoSimCoreFunctional(unittest.TestCase):
    """MuJoCoSimCore 真实 MuJoCo 功能测试（对应阶段二 Step 1 验收标准）。"""

    def setUp(self):
        """每个测试前创建 SimCore 并加载 pendulum 模型。"""
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_PENDULUM_XML)

    def test_init_simulation_loads_model(self):
        """init_simulation 后 _mjModel/_mjData 非 None，维度正确。"""
        self.assertIsNotNone(self.sim._mjModel)
        self.assertIsNotNone(self.sim._mjData)
        self.assertEqual(self.sim.nq, 1)
        self.assertEqual(self.sim.nv, 1)
        self.assertEqual(self.sim.nu, 1)

    def test_step_advances_time(self):
        """step(1) 后 _mjData.time > 0。"""
        self.assertEqual(self.sim._mjData.time, 0.0)
        self.sim.step(1)
        self.assertGreater(self.sim._mjData.time, 0.0)

    def test_forward_updates_kinematics(self):
        """forward() 后 body_xpos 可读（派生量已更新）。"""
        self.sim.forward()
        # pendulum body 的 xpos 应为 (3,) 数组
        xpos = self.sim._mjData.body(1).xpos  # body 0 = world, body 1 = pendulum
        self.assertEqual(xpos.shape, (3,))

    def test_set_ctrl_writes_ctrl_array(self):
        """set_ctrl([0.5]) 后 _mjData.ctrl[0] == 0.5。"""
        self.sim.set_ctrl(np.array([0.5]))
        self.assertAlmostEqual(self.sim._mjData.ctrl[0], 0.5)

    def test_set_qpos_qvel_writes_state(self):
        """set_qpos_qvel([0.3], [0.1]) 后 qpos[0]==0.3, qvel[0]==0.1。"""
        self.sim.set_qpos_qvel(np.array([0.3]), np.array([0.1]))
        self.assertAlmostEqual(self.sim._mjData.qpos[0], 0.3)
        self.assertAlmostEqual(self.sim._mjData.qvel[0], 0.1)

    def test_reset_data_zeroes_state(self):
        """reset_data() 后 qpos/qvel 恢复默认（全零）。"""
        # 先设置非零状态
        self.sim.set_qpos_qvel(np.array([0.5]), np.array([0.3]))
        self.sim.forward()
        # reset
        self.sim.reset_data()
        self.assertEqual(self.sim._mjData.qpos[0], 0.0)
        self.assertEqual(self.sim._mjData.qvel[0], 0.0)

    def test_reset_data_raises_before_init(self):
        """reset_data() 在未初始化时抛 RuntimeError。"""
        sim = MuJoCoSimCore()
        with self.assertRaises(RuntimeError):
            sim.reset_data()

    def test_sync_to_view_populates_view(self):
        """sync_to_view(view) 后 view.qpos 与 _mjData.qpos 一致（零拷贝视图）。"""
        self.sim.set_qpos_qvel(np.array([0.7]), np.array([0.2]))
        self.sim.forward()

        view = OrcaGymDataView()
        self.sim.sync_to_view(view)

        np.testing.assert_array_equal(view.qpos, self.sim._mjData.qpos)
        np.testing.assert_array_equal(view.qvel, self.sim._mjData.qvel)
        self.assertAlmostEqual(view.time, self.sim._mjData.time)

    def test_sync_to_view_is_zero_copy(self):
        """sync_to_view 后 view.qpos 是 _mjData.qpos 的视图（修改同步）。"""
        view = OrcaGymDataView()
        self.sim.sync_to_view(view)

        # 修改 _mjData.qpos，view.qpos 应同步变化（零拷贝视图）
        self.sim._mjData.qpos[0] = 0.42
        self.assertAlmostEqual(view.qpos[0], 0.42)

    def test_nq_nv_nu_properties_return_int(self):
        """nq/nv/nu property 返回 int 类型。"""
        self.assertIsInstance(self.sim.nq, int)
        self.assertIsInstance(self.sim.nv, int)
        self.assertIsInstance(self.sim.nu, int)


# =============================================================================
# 阶段三 3.1.1：MuJoCoSimCore 关节查询方法
# =============================================================================


class TestSimCoreJointQueryArchCompliance(unittest.TestCase):
    """子步骤 3.1.1 架构遵从性测试（K11 typed 返回 + P2 不泄漏 MjData/MjModel）。

    对应文档 §5.2 架构遵从性测试表。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_simcore_joint_query_returns_ndarray(self):
        """K11: query_joint_qpos/qvel/qacc 返回 dict[str, np.ndarray]，不返回 MjData/MjModel。"""
        names = ["left_hip_pitch_joint", "left_hip_roll_joint"]
        for method_name, expected_value_type in [
            ("query_joint_qpos", np.ndarray),
            ("query_joint_qvel", np.ndarray),
            ("query_joint_qacc", np.ndarray),
        ]:
            with self.subTest(method=method_name):
                result = getattr(self.sim, method_name)(names)
                self.assertIsInstance(result, dict)
                self.assertNotIsInstance(result, (mujoco.MjData, mujoco.MjModel))
                for jname in names:
                    self.assertIn(jname, result)
                    self.assertIsInstance(result[jname], expected_value_type)
                    self.assertNotIsInstance(
                        result[jname], (mujoco.MjData, mujoco.MjModel)
                    )

    def test_simcore_joint_query_no_mjdata_leak(self):
        """P2/K11: grep 断言关节查询方法源码不 return self._mjData / self._mjModel。"""
        source = inspect.getsource(MuJoCoSimCore)
        # 定位关节查询区块（从 _joint_id 到 jnt_dofadr）
        start = source.find("    # --- 关节查询（阶段三 3.1.1）---")
        end = source.find("    # --- 维度 property ---")
        self.assertGreater(start, 0, "关节查询区块标记未找到")
        self.assertGreater(end, start, "维度 property 标记未找到")
        joint_query_source = source[start:end]
        self.assertNotIn(
            "return self._mjData", joint_query_source,
            "关节查询方法不得 return self._mjData（P2 泄漏）",
        )
        self.assertNotIn(
            "return self._mjModel", joint_query_source,
            "关节查询方法不得 return self._mjModel（P2 泄漏）",
        )

    def test_simcore_jnt_qposadr_returns_int(self):
        """K11: jnt_qposadr/jnt_dofadr 返回 int（非 numpy 标量泄漏）。"""
        for jname in ["left_hip_pitch_joint", "floating_base_joint"]:
            with self.subTest(joint=jname):
                qpos_adr = self.sim.jnt_qposadr(jname)
                dof_adr = self.sim.jnt_dofadr(jname)
                self.assertIsInstance(qpos_adr, int, "jnt_qposadr 必须返回 int")
                self.assertIsInstance(dof_adr, int, "jnt_dofadr 必须返回 int")
                # 排除 numpy 标量（np.int64 等）
                self.assertNotIsInstance(qpos_adr, np.integer)
                self.assertNotIsInstance(dof_adr, np.integer)


class TestSimCoreJointQueryFunctional(unittest.TestCase):
    """子步骤 3.1.1 功能单元测试（G1 XML 真实数据）。

    对应文档 §5.2 功能单元测试表。加载 G1 模型验证切片数值正确。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_query_joint_qpos_returns_correct_slice(self):
        """query_joint_qpos(["left_hip_pitch_joint"]) 与 _mjData.qpos[adr:adr+len] 一致。"""
        jname = "left_hip_pitch_joint"
        result = self.sim.query_joint_qpos([jname])
        jid = mujoco.mj_name2id(self.sim._mjModel, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr = int(self.sim._mjModel.jnt_qposadr[jid])
        expected = self.sim._mjData.qpos[adr:adr + 1]
        np.testing.assert_array_equal(result[jname], expected)
        self.assertEqual(result[jname].shape, (1,))

    def test_query_joint_qvel_matches_dof_slice(self):
        """qvel 切片正确（含 free joint 的 6 维 qvel 切片验证）。"""
        # free joint: qvel 长度 6
        free_result = self.sim.query_joint_qvel(["floating_base_joint"])
        free_jid = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        free_adr = int(self.sim._mjModel.jnt_dofadr[free_jid])
        free_expected = self.sim._mjData.qvel[free_adr:free_adr + 6]
        np.testing.assert_array_equal(free_result["floating_base_joint"], free_expected)
        self.assertEqual(free_result["floating_base_joint"].shape, (6,))

        # hinge joint: qvel 长度 1
        hinge_result = self.sim.query_joint_qvel(["left_hip_pitch_joint"])
        hinge_jid = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint"
        )
        hinge_adr = int(self.sim._mjModel.jnt_dofadr[hinge_jid])
        hinge_expected = self.sim._mjData.qvel[hinge_adr:hinge_adr + 1]
        np.testing.assert_array_equal(hinge_result["left_hip_pitch_joint"], hinge_expected)

    def test_query_joint_offsets_lengths_consistent(self):
        """offsets + lengths 与 jnt_qposadr/dofadr 一致。"""
        names = ["floating_base_joint", "left_hip_pitch_joint", "left_hip_roll_joint"]
        qpos_off, qvel_off, qacc_off = self.sim.query_joint_offsets(names)
        qpos_len, qvel_len, qacc_len = self.sim.query_joint_lengths(names)

        # 长度等于输入
        self.assertEqual(len(qpos_off), 3)
        self.assertEqual(len(qvel_len), 3)

        # free joint: qpos_len=7, qvel_len=6
        self.assertEqual(qpos_len[0], 7)
        self.assertEqual(qvel_len[0], 6)
        # hinge joint: qpos_len=1, qvel_len=1
        self.assertEqual(qpos_len[1], 1)
        self.assertEqual(qvel_len[1], 1)

        # offsets 与 jnt_qposadr/jnt_dofadr 一致
        for i, name in enumerate(names):
            self.assertEqual(int(qpos_off[i]), self.sim.jnt_qposadr(name))
            self.assertEqual(int(qvel_off[i]), self.sim.jnt_dofadr(name))
            # qacc 偏移与 qvel 相同（共享 dof 地址）
            self.assertEqual(int(qacc_off[i]), int(qvel_off[i]))
            # qacc 长度与 qvel 相同
            self.assertEqual(int(qacc_len[i]), int(qvel_len[i]))

    def test_jnt_qposadr_returns_correct_adr(self):
        """jnt_qposadr("left_hip_pitch_joint") 与 _mjModel.jnt_qposadr 一致。"""
        jname = "left_hip_pitch_joint"
        jid = mujoco.mj_name2id(self.sim._mjModel, mujoco.mjtObj.mjOBJ_JOINT, jname)
        expected_adr = int(self.sim._mjModel.jnt_qposadr[jid])
        self.assertEqual(self.sim.jnt_qposadr(jname), expected_adr)
        # floating_base_joint 的 qpos adr 应为 0（首个关节）
        self.assertEqual(self.sim.jnt_qposadr("floating_base_joint"), 0)

    def test_query_joint_qpos_is_zero_copy_view(self):
        """qpos 切片为 _mjData.qpos 的视图（修改同步，零拷贝）。"""
        jname = "left_hip_pitch_joint"
        result = self.sim.query_joint_qpos([jname])
        # 修改 _mjData.qpos 对应位置，切片应同步
        adr = self.sim.jnt_qposadr(jname)
        self.sim._mjData.qpos[adr] = 0.42
        self.assertAlmostEqual(result[jname][0], 0.42)

    def test_query_joint_dofadrs_returns_dict_of_int(self):
        """query_joint_dofadrs 返回 dict[str, int]，值为 int 类型。"""
        names = ["floating_base_joint", "left_hip_pitch_joint"]
        result = self.sim.query_joint_dofadrs(names)
        self.assertIsInstance(result, dict)
        for name in names:
            self.assertIn(name, result)
            self.assertIsInstance(result[name], int)
            self.assertNotIsInstance(result[name], np.integer)
        # floating_base_joint 的 dof adr 应为 0
        self.assertEqual(result["floating_base_joint"], 0)


# =============================================================================
# 阶段三 3.1.2：MuJoCoSimCore Body/Site 查询方法
# =============================================================================


class TestSimCoreBodySiteQueryArchCompliance(unittest.TestCase):
    """子步骤 3.1.2 架构遵从性测试（K11 typed 返回 + P2 不泄漏 MjData/MjModel）。

    对应文档 §5.3 架构遵从性测试表。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_simcore_body_query_returns_dict_of_dict(self):
        """K11: query_body_* 返回 dict[str, dict]，内层含 xpos/xmat/xquat 键，值为 np.ndarray。"""
        body_names = ["pelvis", "torso_link"]
        # query_body_xpos_xmat_xquat
        result = self.sim.query_body_xpos_xmat_xquat(body_names)
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, (mujoco.MjData, mujoco.MjModel))
        for bname in body_names:
            self.assertIn(bname, result)
            inner = result[bname]
            self.assertIsInstance(inner, dict)
            for key in ("xpos", "xmat", "xquat"):
                self.assertIn(key, inner)
                self.assertIsInstance(inner[key], np.ndarray)
                self.assertNotIsInstance(inner[key], (mujoco.MjData, mujoco.MjModel))

        # query_body_xpos_xmat_xquat_xvel
        result_v = self.sim.query_body_xpos_xmat_xquat_xvel(body_names)
        for bname in body_names:
            inner = result_v[bname]
            for key in ("xpos", "xmat", "xquat", "xvel"):
                self.assertIn(key, inner)
                self.assertIsInstance(inner[key], np.ndarray)

    def test_simcore_site_query_returns_dict(self):
        """K11: query_site_pos_and_mat/query_site_size 返回 typed dict。"""
        site_names = ["camera_head_site", "imu"]
        # query_site_pos_and_mat
        pos_mat = self.sim.query_site_pos_and_mat(site_names)
        self.assertIsInstance(pos_mat, dict)
        self.assertNotIsInstance(pos_mat, (mujoco.MjData, mujoco.MjModel))
        for sname in site_names:
            self.assertIn(sname, pos_mat)
            inner = pos_mat[sname]
            self.assertIsInstance(inner, dict)
            for key in ("xpos", "xmat"):
                self.assertIn(key, inner)
                self.assertIsInstance(inner[key], np.ndarray)

        # query_site_size
        sizes = self.sim.query_site_size(site_names)
        self.assertIsInstance(sizes, dict)
        for sname in site_names:
            self.assertIn(sname, sizes)
            self.assertIsInstance(sizes[sname], np.ndarray)

    def test_simcore_body_query_no_mjdata_leak(self):
        """P2/K11: grep 断言 Body/Site 查询方法源码不 return self._mjData / self._mjModel。"""
        source = inspect.getsource(MuJoCoSimCore)
        start = source.find("    # --- Body/Site 查询（阶段三 3.1.2）---")
        end = source.find("    # --- 维度 property ---")
        self.assertGreater(start, 0, "Body/Site 查询区块标记未找到")
        self.assertGreater(end, start, "维度 property 标记未找到")
        body_site_source = source[start:end]
        self.assertNotIn(
            "return self._mjData", body_site_source,
            "Body/Site 查询方法不得 return self._mjData（P2 泄漏）",
        )
        self.assertNotIn(
            "return self._mjModel", body_site_source,
            "Body/Site 查询方法不得 return self._mjModel（P2 泄漏）",
        )


class TestSimCoreBodySiteQueryFunctional(unittest.TestCase):
    """子步骤 3.1.2 功能单元测试（G1 XML 真实数据）。

    对应文档 §5.3 功能单元测试表。验证 body/site 数值与 _mjData/_mjModel 一致。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_query_body_xpos_shape(self):
        """xpos 形状 (3,)，xmat 形状 (3,3)，xquat 形状 (4,)。"""
        result = self.sim.query_body_xpos_xmat_xquat(["pelvis", "torso_link"])
        for bname in ("pelvis", "torso_link"):
            with self.subTest(body=bname):
                self.assertEqual(result[bname]["xpos"].shape, (3,))
                self.assertEqual(result[bname]["xmat"].shape, (3, 3))
                self.assertEqual(result[bname]["xquat"].shape, (4,))

    def test_query_body_xpos_matches_mjdata(self):
        """返回值与 _mjData.xpos/xmat/xquat[body_id] 一致。"""
        bname = "pelvis"
        bid = mujoco.mj_name2id(self.sim._mjModel, mujoco.mjtObj.mjOBJ_BODY, bname)
        result = self.sim.query_body_xpos_xmat_xquat([bname])
        np.testing.assert_array_equal(result[bname]["xpos"], self.sim._mjData.xpos[bid])
        np.testing.assert_array_equal(
            result[bname]["xmat"], self.sim._mjData.xmat[bid].reshape(3, 3)
        )
        np.testing.assert_array_equal(result[bname]["xquat"], self.sim._mjData.xquat[bid])

    def test_query_body_xpos_xmat_xquat_xvel_matches_jac(self):
        """xvel 与 jacp @ qvel 一致；xpos/xmat/xquat 与 _mjData 一致。"""
        bname = "torso_link"
        bid = mujoco.mj_name2id(self.sim._mjModel, mujoco.mjtObj.mjOBJ_BODY, bname)
        result = self.sim.query_body_xpos_xmat_xquat_xvel([bname])
        # xpos/xmat/xquat 一致
        np.testing.assert_array_equal(result[bname]["xpos"], self.sim._mjData.xpos[bid])
        np.testing.assert_array_equal(
            result[bname]["xmat"], self.sim._mjData.xmat[bid].reshape(3, 3)
        )
        np.testing.assert_array_equal(result[bname]["xquat"], self.sim._mjData.xquat[bid])
        # xvel 与手动 jacp @ qvel 一致
        jacp = np.zeros((3, self.sim._mjModel.nv))
        jacr = np.zeros((3, self.sim._mjModel.nv))
        mujoco.mj_jacBody(self.sim._mjModel, self.sim._mjData, jacp, jacr, bid)
        expected_xvel = jacp @ self.sim._mjData.qvel
        np.testing.assert_allclose(result[bname]["xvel"], expected_xvel)
        self.assertEqual(result[bname]["xvel"].shape, (3,))

    def test_query_site_pos_and_mat_matches_mjdata(self):
        """与 _mjData.site_xpos/site_xmat 一致。"""
        sname = "camera_head_site"
        sid = mujoco.mj_name2id(self.sim._mjModel, mujoco.mjtObj.mjOBJ_SITE, sname)
        result = self.sim.query_site_pos_and_mat([sname])
        np.testing.assert_array_equal(result[sname]["xpos"], self.sim._mjData.site_xpos[sid])
        np.testing.assert_array_equal(
            result[sname]["xmat"], self.sim._mjData.site_xmat[sid].reshape(3, 3)
        )
        self.assertEqual(result[sname]["xpos"].shape, (3,))
        self.assertEqual(result[sname]["xmat"].shape, (3, 3))

    def test_query_site_size_matches_model(self):
        """与 _mjModel.site_size 一致。"""
        sname = "imu"
        sid = mujoco.mj_name2id(self.sim._mjModel, mujoco.mjtObj.mjOBJ_SITE, sname)
        result = self.sim.query_site_size([sname])
        np.testing.assert_array_equal(result[sname], self.sim._mjModel.site_size[sid])
        self.assertEqual(result[sname].shape, (3,))


# =============================================================================
# 阶段三 3.1.3：MuJoCoSimCore 传感器/执行器/接触/Geom 查询方法
# =============================================================================


class TestSimCoreSensorActuatorContactGeomArchCompliance(unittest.TestCase):
    """子步骤 3.1.3 架构遵从性测试（K11 typed 返回 + P2 不泄漏 MjData/MjModel）。

    对应文档 §5.4 架构遵从性测试表。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_simcore_sensor_query_returns_dict_ndarray(self):
        """K11: query_sensor_data 返回 dict[str, np.ndarray]，不返回 MjData/MjModel。"""
        names = ["left_hip_pitch_pos", "left_hip_pitch_vel"]
        # sensor_info 为空 dict，强制走 _mjModel 回退路径
        result = self.sim.query_sensor_data(names, {})
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, (mujoco.MjData, mujoco.MjModel))
        for sname in names:
            self.assertIn(sname, result)
            self.assertIsInstance(result[sname], np.ndarray)
            self.assertNotIsInstance(result[sname], (mujoco.MjData, mujoco.MjModel))

    def test_simcore_actuator_query_returns_dict_ndarray(self):
        """K11: query_actuator_torques 返回 dict[str, np.ndarray]。"""
        names = ["left_hip_pitch", "right_shoulder_pitch"]
        result = self.sim.query_actuator_torques(names)
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, (mujoco.MjData, mujoco.MjModel))
        for aname in names:
            self.assertIn(aname, result)
            self.assertIsInstance(result[aname], np.ndarray)

    def test_simcore_contact_simple_returns_list_of_dict(self):
        """K11: query_contact_simple 返回 list[dict]，dict 含 geom1/geom2/dist 等键。"""
        contacts = self.sim.query_contact_simple()
        self.assertIsInstance(contacts, list)
        # G1 初始姿态可能无接触，但若有则验证结构
        for c in contacts:
            self.assertIsInstance(c, dict)
            for key in ("geom1", "geom2", "dist", "pos", "frame"):
                self.assertIn(key, c)
            self.assertIsInstance(c["geom1"], int)
            self.assertIsInstance(c["geom2"], int)
            self.assertIsInstance(c["dist"], float)
            self.assertIsInstance(c["pos"], np.ndarray)
            self.assertIsInstance(c["frame"], np.ndarray)

    def test_simcore_contact_force_returns_dict_int_ndarray(self):
        """K11: query_contact_force 返回 dict[int, np.ndarray]。"""
        # 即使无接触，空列表也应返回空 dict
        result = self.sim.query_contact_force([])
        self.assertIsInstance(result, dict)
        # 若有接触，验证结构
        contacts = self.sim.query_contact_simple()
        if contacts:
            cid = 0
            result = self.sim.query_contact_force([cid])
            self.assertIn(cid, result)
            self.assertIsInstance(result[cid], np.ndarray)
            self.assertEqual(result[cid].shape, (6,))

    def test_simcore_get_cfrc_ext_returns_ndarray(self):
        """K11/P2: get_cfrc_ext 返回 np.ndarray，非 MjData。"""
        cfrc = self.sim.get_cfrc_ext()
        self.assertIsInstance(cfrc, np.ndarray)
        self.assertNotIsInstance(cfrc, (mujoco.MjData, mujoco.MjModel))

    def test_simcore_geom_query_no_mjmodel_leak(self):
        """P2/K11: grep 断言传感器/执行器/接触/Geom 查询区块不 return self._mjModel。"""
        source = inspect.getsource(MuJoCoSimCore)
        start = source.find("    # --- 传感器/执行器/接触/Geom 查询（阶段三 3.1.3）---")
        end = source.find("    # --- 维度 property ---")
        self.assertGreater(start, 0, "传感器/执行器/接触/Geom 查询区块标记未找到")
        self.assertGreater(end, start, "维度 property 标记未找到")
        block_source = source[start:end]
        self.assertNotIn(
            "return self._mjModel", block_source,
            "传感器/执行器/接触/Geom 查询方法不得 return self._mjModel（P2 泄漏）",
        )
        self.assertNotIn(
            "return self._mjData", block_source,
            "传感器/执行器/接触/Geom 查询方法不得 return self._mjData（P2 泄漏）",
        )


class TestSimCoreSensorActuatorContactGeomFunctional(unittest.TestCase):
    """子步骤 3.1.3 功能单元测试（G1 XML 真实数据）。

    对应文档 §5.4 功能单元测试表。验证数值与 _mjData/_mjModel 一致。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_query_sensor_data_matches_sensordata(self):
        """传感器数据与 _mjData.sensordata 切片一致（含 sensor_info 与回退两条路径）。"""
        sname = "left_hip_pitch_pos"
        sid = mujoco.mj_name2id(self.sim._mjModel, mujoco.mjtObj.mjOBJ_SENSOR, sname)
        adr = int(self.sim._mjModel.sensor_adr[sid])
        dim = int(self.sim._mjModel.sensor_dim[sid])
        expected = self.sim._mjData.sensordata[adr:adr + dim]

        # 路径 1: sensor_info 提供 adr/dim
        sensor_info = {sname: {"adr": adr, "dim": dim}}
        result1 = self.sim.query_sensor_data([sname], sensor_info)
        np.testing.assert_array_equal(result1[sname], expected)

        # 路径 2: sensor_info 为空，回退到 _mjModel
        result2 = self.sim.query_sensor_data([sname], {})
        np.testing.assert_array_equal(result2[sname], expected)

    def test_query_actuator_torques_matches_force(self):
        """与 _mjData.actuator_force 切片一致。"""
        aname = "left_hip_pitch"
        aid = mujoco.mj_name2id(self.sim._mjModel, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        result = self.sim.query_actuator_torques([aname])
        # actuator_force 按 actuator_id 索引，每执行器 1 维
        np.testing.assert_array_equal(
            result[aname], self.sim._mjData.actuator_force[aid:aid + 1]
        )
        self.assertEqual(result[aname].shape, (1,))

    def test_query_contact_simple_returns_list(self):
        """接触列表结构正确（长度 == _mjData.ncon）。"""
        contacts = self.sim.query_contact_simple()
        self.assertEqual(len(contacts), self.sim._mjData.ncon)
        # 若有接触，验证数值与 _mjData.contact 一致
        for i, c in enumerate(contacts):
            self.assertEqual(c["geom1"], int(self.sim._mjData.contact[i].geom1))
            self.assertEqual(c["geom2"], int(self.sim._mjData.contact[i].geom2))
            self.assertAlmostEqual(c["dist"], float(self.sim._mjData.contact[i].dist))
            np.testing.assert_array_equal(c["pos"], self.sim._mjData.contact[i].pos)
            np.testing.assert_array_equal(c["frame"], self.sim._mjData.contact[i].frame)

    def test_query_contact_force_via_mj_contactForce(self):
        """接触力与 mujoco.mj_contactForce 一致。"""
        contacts = self.sim.query_contact_simple()
        if not contacts:
            self.skipTest("G1 初始姿态无接触，跳过接触力数值验证")
        cid = 0
        result = self.sim.query_contact_force([cid])
        expected = np.zeros(6)
        mujoco.mj_contactForce(self.sim._mjModel, self.sim._mjData, cid, expected)
        np.testing.assert_array_equal(result[cid], expected)
        self.assertEqual(result[cid].shape, (6,))

    def test_get_cfrc_ext_shape(self):
        """形状 (nbody, 6)。"""
        cfrc = self.sim.get_cfrc_ext()
        self.assertEqual(cfrc.shape, (self.sim._mjModel.nbody, 6))

    def test_get_goal_bounding_box_matches_geom_size(self):
        """与 _mjModel.geom_size 一致。"""
        gname = "manipulation_box_geom"
        gid = mujoco.mj_name2id(self.sim._mjModel, mujoco.mjtObj.mjOBJ_GEOM, gname)
        result = self.sim.get_goal_bounding_box(gname)
        np.testing.assert_array_equal(result, self.sim._mjModel.geom_size[gid])
        self.assertEqual(result.shape, (3,))


# =============================================================================
# 阶段三 3.3.1：MuJoCoSimCore mj_jacBody / mj_jacSite
# =============================================================================


class TestSimCoreJacArchCompliance(unittest.TestCase):
    """子步骤 3.3.1 架构遵从性测试（K11 返回 None + P2 不泄漏）。

    对应文档 §7.2 架构遵从性测试表。
    """

    def test_simcore_jac_methods_write_inplace(self):
        """K11: grep 断言 mj_jacBody/mj_jacSite 原地写 jacp/jacr，不返回新数组。"""
        source = inspect.getsource(MuJoCoSimCore)
        start = source.find("# --- 雅可比计算（阶段三 3.3.1）---")
        self.assertGreater(start, 0, "未找到 3.3.1 雅可比计算区块")
        block = source[start:]
        end = block.find("# --- 维度 property ---")
        self.assertGreater(end, 0, "未找到维度 property 标记")
        block = block[:end]
        # 不返回新数组（无 return 语句返回 jacp/jacr）
        self.assertNotIn("return jacp", block)
        self.assertNotIn("return jacr", block)
        self.assertNotIn("return self._mjData", block)
        self.assertNotIn("return self._mjModel", block)

    def test_simcore_jac_methods_return_none(self):
        """K11: mj_jacBody/mj_jacSite 返回 None（原地写操作）。"""
        sim = MuJoCoSimCore()
        sim.init_simulation(_G1_XML)
        sim.forward()
        nv = sim._mjModel.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        pelvis_id = mujoco.mj_name2id(
            sim._mjModel, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        ret = sim.mj_jacBody(jacp, jacr, pelvis_id)
        self.assertIsNone(ret)
        imu_id = mujoco.mj_name2id(
            sim._mjModel, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        ret = sim.mj_jacSite(jacp, jacr, imu_id)
        self.assertIsNone(ret)

    def test_simcore_jac_no_mjdata_leak(self):
        """P2/K11: grep 断言不 return self._mjData/_mjModel。"""
        source = inspect.getsource(MuJoCoSimCore)
        start = source.find("# --- 雅可比计算（阶段三 3.3.1）---")
        self.assertGreater(start, 0)
        block = source[start:]
        end = block.find("# --- 维度 property ---")
        if end > 0:
            block = block[:end]
        self.assertNotIn("return self._mjData", block)
        self.assertNotIn("return self._mjModel", block)


class TestSimCoreJacFunctional(unittest.TestCase):
    """子步骤 3.3.1 功能单元测试（G1 XML 真实数据）。

    对应文档 §7.2 功能单元测试表。验证雅可比形状与数值与原生 mujoco 一致。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_mj_jacBody_writes_correct_shape(self):
        """jacp 形状 (3, nv)，jacr 形状 (3, nv)。"""
        nv = self.sim._mjModel.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        pelvis_id = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.sim.mj_jacBody(jacp, jacr, pelvis_id)
        self.assertEqual(jacp.shape, (3, nv))
        self.assertEqual(jacr.shape, (3, nv))
        # 写入后应有非零元素（pelvis 为 floating base，雅可比非零）
        self.assertTrue(np.any(jacp != 0.0))

    def test_mj_jacBody_matches_mujoco(self):
        """与直接调 mujoco.mj_jacBody 结果一致。"""
        nv = self.sim._mjModel.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        pelvis_id = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.sim.mj_jacBody(jacp, jacr, pelvis_id)
        # 直接调原生 mujoco.mj_jacBody 对照
        expected_jacp = np.zeros((3, nv))
        expected_jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(
            self.sim._mjModel, self.sim._mjData, expected_jacp, expected_jacr, pelvis_id
        )
        np.testing.assert_array_equal(jacp, expected_jacp)
        np.testing.assert_array_equal(jacr, expected_jacr)

    def test_mj_jacSite_writes_correct_shape(self):
        """site 雅可比形状正确 (3, nv)。"""
        nv = self.sim._mjModel.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        imu_id = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        self.sim.mj_jacSite(jacp, jacr, imu_id)
        self.assertEqual(jacp.shape, (3, nv))
        self.assertEqual(jacr.shape, (3, nv))
        self.assertTrue(np.any(jacp != 0.0))

    def test_mj_jacSite_matches_mujoco(self):
        """与直接调 mujoco.mj_jacSite 结果一致。"""
        nv = self.sim._mjModel.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        imu_id = mujoco.mj_name2id(
            self.sim._mjModel, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        self.sim.mj_jacSite(jacp, jacr, imu_id)
        expected_jacp = np.zeros((3, nv))
        expected_jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(
            self.sim._mjModel, self.sim._mjData, expected_jacp, expected_jacr, imu_id
        )
        np.testing.assert_array_equal(jacp, expected_jacp)
        np.testing.assert_array_equal(jacr, expected_jacr)


# =============================================================================
# 阶段三 3.3.2：MuJoCoSimCore mj_jac_site 批量
# =============================================================================


class TestSimCoreJacSiteBatchArchCompliance(unittest.TestCase):
    """子步骤 3.3.2 架构遵从性测试（K11 typed 返回 + P2 不泄漏）。

    对应文档 §7.3 架构遵从性测试表。
    """

    def test_simcore_jac_site_batch_returns_dict(self):
        """K11: 返回 dict[str, dict]，内层含 jacp/jacr 键，值为 np.ndarray。"""
        sim = MuJoCoSimCore()
        sim.init_simulation(_G1_XML)
        sim.forward()
        result = sim.mj_jac_site(["imu"])
        self.assertIsInstance(result, dict)
        self.assertIn("imu", result)
        entry = result["imu"]
        self.assertIsInstance(entry, dict)
        self.assertIn("jacp", entry)
        self.assertIn("jacr", entry)
        self.assertIsInstance(entry["jacp"], np.ndarray)
        self.assertIsInstance(entry["jacr"], np.ndarray)

    def test_simcore_jac_site_batch_no_mjdata_leak(self):
        """P2/K11: grep 断言不 return self._mjData/_mjModel。"""
        source = inspect.getsource(MuJoCoSimCore.mj_jac_site)
        self.assertNotIn("return self._mjData", source)
        self.assertNotIn("return self._mjModel", source)


class TestSimCoreJacSiteBatchFunctional(unittest.TestCase):
    """子步骤 3.3.2 功能单元测试（G1 XML 真实数据）。

    对应文档 §7.3 功能单元测试表。验证批量与单点一致。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_mj_jac_site_batch_returns_all_sites(self):
        """每个 site_name 都有对应 entry。"""
        names = ["imu"]
        result = self.sim.mj_jac_site(names)
        for name in names:
            self.assertIn(name, result)

    def test_mj_jac_site_batch_matches_single(self):
        """批量结果与单点 mj_jacSite 逐 site 一致。"""
        names = ["imu"]
        result = self.sim.mj_jac_site(names)
        nv = self.sim._mjModel.nv
        for name in names:
            sid = mujoco.mj_name2id(
                self.sim._mjModel, mujoco.mjtObj.mjOBJ_SITE, name
            )
            expected_jacp = np.zeros((3, nv))
            expected_jacr = np.zeros((3, nv))
            self.sim.mj_jacSite(expected_jacp, expected_jacr, sid)
            np.testing.assert_array_equal(result[name]["jacp"], expected_jacp)
            np.testing.assert_array_equal(result[name]["jacr"], expected_jacr)


# =============================================================================
# 阶段三 3.5.1：MuJoCoSimCore 等式约束方法
# =============================================================================


class TestSimCoreEqualityArchCompliance(unittest.TestCase):
    """子步骤 3.5.1 架构遵从性测试（K11 返回 None + P2 只写 model）。

    对应文档 §9.2 架构遵从性测试表。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_simcore_eq_methods_return_none(self):
        """K11: 2 个约束方法返回 None（写操作无返回值）。"""
        # 用模型实际的 obj id 匹配槽位（按 obj_id 匹配写入语义）
        model = self.sim._mjModel
        orig_obj1 = int(model.eq_obj1id[0])
        orig_obj2 = int(model.eq_obj2id[0])
        data = np.zeros(mujoco.mjNEQDATA)
        eq_list = [
            {
                "type": mujoco.mjtEq.mjEQ_CONNECT,
                "obj1_id": orig_obj1,
                "obj2_id": orig_obj2,
                "data": data,
            }
        ]
        ret = self.sim.update_equality_constraints(eq_list)
        self.assertIsNone(ret)
        ret = self.sim.modify_equality_objects([0], obj1_ids=[3])
        self.assertIsNone(ret)

    def test_simcore_eq_methods_write_model_only(self):
        """P2/K11: grep 断言只写 _mjModel.eq_*，不返回 MjModel/MjData。"""
        source = inspect.getsource(MuJoCoSimCore)
        start = source.find("# --- 等式约束（阶段三 3.5.1）---")
        self.assertGreater(start, 0, "未找到 3.5.1 等式约束区块")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        # 只写 _mjModel.eq_*，不触 _mjData
        self.assertNotIn("_mjData", block)
        self.assertIn("model.eq_type", block)
        self.assertIn("model.eq_obj1id", block)
        self.assertIn("model.eq_obj2id", block)
        self.assertIn("model.eq_data", block)

    def test_simcore_eq_no_mjmodel_leak(self):
        """P2/K11: grep 断言不 return self._mjModel / self._mjData。"""
        source = inspect.getsource(MuJoCoSimCore)
        start = source.find("# --- 等式约束（阶段三 3.5.1）---")
        block_source = source[start:]
        end = block_source.find("\n    # ---", 1)
        if end < 0:
            end = len(block_source)
        block = block_source[:end]
        self.assertNotIn("return self._mjModel", block)
        self.assertNotIn("return self._mjData", block)


class TestSimCoreEqualityFunctional(unittest.TestCase):
    """子步骤 3.5.1 功能单元测试（G1 XML 真实数据）。

    对应文档 §9.2 功能单元测试表。验证 eq_* 字段正确写入。
    """

    def setUp(self):
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_G1_XML)
        self.sim.forward()

    def test_update_equality_constraints_writes_eq_fields(self):
        """调用后 eq_type/eq_data 正确写入（按 obj_id 匹配槽位）。"""
        model = self.sim._mjModel
        orig_obj1 = int(model.eq_obj1id[0])
        orig_obj2 = int(model.eq_obj2id[0])
        data = np.zeros(mujoco.mjNEQDATA)
        data[0:3] = [0.1, 0.2, 0.3]
        eq_list = [
            {
                "type": mujoco.mjtEq.mjEQ_WELD,
                "obj1_id": orig_obj1,
                "obj2_id": orig_obj2,
                "data": data,
            }
        ]
        self.sim.update_equality_constraints(eq_list)
        self.assertEqual(model.eq_type[0], mujoco.mjtEq.mjEQ_WELD)
        np.testing.assert_array_equal(model.eq_data[0], data)

    def test_modify_equality_objects_updates_obj_ids(self):
        """eq_obj1id/eq_obj2id 更新正确。"""
        # modify_equality_objects 按索引直接修改，不受按 obj_id 匹配影响
        self.sim.modify_equality_objects([0], obj1_ids=[10], obj2_ids=[20])
        model = self.sim._mjModel
        self.assertEqual(model.eq_obj1id[0], 10)
        self.assertEqual(model.eq_obj2id[0], 20)

    def test_update_equality_constraints_idempotent(self):
        """重复调用结果一致（按 obj_id 匹配槽位）。"""
        model = self.sim._mjModel
        orig_obj1 = int(model.eq_obj1id[0])
        orig_obj2 = int(model.eq_obj2id[0])
        data = np.zeros(mujoco.mjNEQDATA)
        data[0] = 0.5
        eq_list = [
            {
                "type": mujoco.mjtEq.mjEQ_CONNECT,
                "obj1_id": orig_obj1,
                "obj2_id": orig_obj2,
                "data": data,
            }
        ]
        self.sim.update_equality_constraints(eq_list)
        first_type = int(self.sim._mjModel.eq_type[0])
        first_data = self.sim._mjModel.eq_data[0].copy()

        self.sim.update_equality_constraints(eq_list)
        self.assertEqual(int(self.sim._mjModel.eq_type[0]), first_type)
        np.testing.assert_array_equal(self.sim._mjModel.eq_data[0], first_data)


if __name__ == "__main__":
    unittest.main()
