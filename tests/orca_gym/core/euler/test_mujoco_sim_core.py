"""P2-Step2: MuJoCoSimCore 骨架验收测试。

验证 MuJoCoSimCore 的私有属性、生命周期方法、力应用方法、维度 property
签名完整（架构 §5.3, §12.2），不验证 MuJoCo 功能正确性（骨架阶段
不执行真实仿真，方法体 raise NotImplementedError）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import unittest

from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore


class TestMuJoCoSimCoreSkeleton(unittest.TestCase):
    """MuJoCoSimCore 骨架验收测试（对应 P2-Step2 验收标准）。"""

    def test_sim_core_constructable(self):
        """MuJoCoSimCore() 可无参构造。"""
        sim = MuJoCoSimCore()
        self.assertIsInstance(sim, MuJoCoSimCore)

    def test_sim_core_has_mj_model_data_private(self):
        """实例有 _mjModel/_mjData 私有属性（带下划线）。"""
        sim = MuJoCoSimCore()
        self.assertTrue(hasattr(sim, "_mjModel"))
        self.assertTrue(hasattr(sim, "_mjData"))
        # 初始化为 None（待 init_simulation 填充）
        self.assertIsNone(sim._mjModel)
        self.assertIsNone(sim._mjData)

    def test_sim_core_has_lifecycle_methods(self):
        """init_simulation/step/forward/set_ctrl/sync_to_view 方法存在。"""
        self.assertTrue(callable(getattr(MuJoCoSimCore, "init_simulation", None)))
        self.assertTrue(callable(getattr(MuJoCoSimCore, "step", None)))
        self.assertTrue(callable(getattr(MuJoCoSimCore, "forward", None)))
        self.assertTrue(callable(getattr(MuJoCoSimCore, "set_ctrl", None)))
        self.assertTrue(callable(getattr(MuJoCoSimCore, "sync_to_view", None)))

    def test_sim_core_has_force_methods(self):
        """apply_body_force/clear_body_force/clear_all_forces 方法存在。"""
        self.assertTrue(callable(getattr(MuJoCoSimCore, "apply_body_force", None)))
        self.assertTrue(callable(getattr(MuJoCoSimCore, "clear_body_force", None)))
        self.assertTrue(callable(getattr(MuJoCoSimCore, "clear_all_forces", None)))

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


class TestMuJoCoSimCoreMethodStubs(unittest.TestCase):
    """补充：验证方法在骨架阶段按约定 raise NotImplementedError。

    骨架阶段不执行真实仿真（架构 §1.3），方法体应为占位。
    """

    def test_lifecycle_methods_raise_not_implemented(self):
        sim = MuJoCoSimCore()
        with self.assertRaises(NotImplementedError):
            sim.init_simulation("dummy.xml")
        with self.assertRaises(NotImplementedError):
            sim.step(1)
        with self.assertRaises(NotImplementedError):
            sim.forward()
        with self.assertRaises(NotImplementedError):
            sim.set_ctrl(None)
        with self.assertRaises(NotImplementedError):
            sim.sync_to_view(None)

    def test_force_methods_raise_not_implemented(self):
        sim = MuJoCoSimCore()
        with self.assertRaises(NotImplementedError):
            sim.apply_body_force(0, None, None)
        with self.assertRaises(NotImplementedError):
            sim.clear_body_force(0)
        with self.assertRaises(NotImplementedError):
            sim.clear_all_forces()

    def test_dimension_properties_raise_not_implemented(self):
        """nq/nv/nu property 在骨架阶段 raise NotImplementedError。"""
        sim = MuJoCoSimCore()
        with self.assertRaises(NotImplementedError):
            _ = sim.nq
        with self.assertRaises(NotImplementedError):
            _ = sim.nv
        with self.assertRaises(NotImplementedError):
            _ = sim.nu


if __name__ == "__main__":
    unittest.main()
