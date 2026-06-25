"""阶段二 Step 1: MuJoCoSimCore 功能验收测试。

验证 MuJoCoSimCore 的真实 MuJoCo 操作（init/step/forward/set_ctrl/
reset_data/set_qpos_qvel/sync_to_view）和维度 property 功能正确
（架构 §5.3, §12.2）。力应用方法仍 raise NotImplementedError（留待完整 P4）。

运行方式:
    <conda-base>/envs/OrcaFlow_Flow/bin/python tests/run_tests.py --component core/euler
"""

import os
import unittest

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
        """apply_body_force/clear_body_force/clear_all_forces 方法存在。"""
        for name in ["apply_body_force", "clear_body_force", "clear_all_forces"]:
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


class TestMuJoCoSimCoreForceStubs(unittest.TestCase):
    """力应用方法仍 raise NotImplementedError（留待完整 P4）。"""

    def test_force_methods_raise_not_implemented(self):
        sim = MuJoCoSimCore()
        with self.assertRaises(NotImplementedError):
            sim.apply_body_force(0, None, None)
        with self.assertRaises(NotImplementedError):
            sim.clear_body_force(0)
        with self.assertRaises(NotImplementedError):
            sim.clear_all_forces()


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


if __name__ == "__main__":
    unittest.main()
