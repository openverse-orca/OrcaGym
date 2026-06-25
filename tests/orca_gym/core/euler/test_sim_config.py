"""P1-Step1: SimConfig 骨架验收测试。

验证 SimConfig 的签名和 typed 接口完整（架构 §5.6, §12.2），
不验证 MuJoCo 功能正确性（骨架阶段不持有真实 mjModel）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import unittest

import numpy as np

from orca_gym.core.euler.sim_config import SimConfig


class TestSimConfigSkeleton(unittest.TestCase):
    """SimConfig 骨架验收测试（对应 P1-Step1 验收标准）。"""

    def test_sim_config_constructable(self):
        """SimConfig() 可无参构造（骨架不依赖真实 mjModel）。"""
        config = SimConfig()
        self.assertIsInstance(config, SimConfig)

    def test_sim_config_has_timestep_property(self):
        """timestep 是 property，有 getter 和 setter。"""
        self.assertIsInstance(SimConfig.timestep, property)
        self.assertIsNotNone(SimConfig.timestep.fget)
        self.assertIsNotNone(SimConfig.timestep.fset)

    def test_sim_config_has_integrator_property(self):
        """integrator 是 property，有 getter 和 setter。"""
        self.assertIsInstance(SimConfig.integrator, property)
        self.assertIsNotNone(SimConfig.integrator.fget)
        self.assertIsNotNone(SimConfig.integrator.fset)

    def test_sim_config_has_iterations_property(self):
        """iterations 是 property，有 getter 和 setter。"""
        self.assertIsInstance(SimConfig.iterations, property)
        self.assertIsNotNone(SimConfig.iterations.fget)
        self.assertIsNotNone(SimConfig.iterations.fset)

    def test_sim_config_has_gravity_property(self):
        """gravity 是 property，有 getter 和 setter。"""
        self.assertIsInstance(SimConfig.gravity, property)
        self.assertIsNotNone(SimConfig.gravity.fget)
        self.assertIsNotNone(SimConfig.gravity.fset)

    def test_sim_config_has_load_from_dict(self):
        """load_from_dict 方法存在且可调用。"""
        self.assertTrue(callable(getattr(SimConfig, "load_from_dict", None)))

    def test_sim_config_has_to_dict(self):
        """to_dict 方法存在且可调用。"""
        self.assertTrue(callable(getattr(SimConfig, "to_dict", None)))

    def test_sim_config_docstring_has_contract(self):
        """docstring 含「使用契约」和「禁止」关键词（K12）。"""
        doc = SimConfig.__doc__ or ""
        self.assertIn("使用契约", doc)
        self.assertIn("禁止", doc)


class TestSimConfigPropertyMechanism(unittest.TestCase):
    """补充：验证 property getter/setter 机制在占位字段上可用。

    骨架阶段 property 操作内部占位字段（架构 §12.2 允许），
    此测试验证接口机制本身可用，不验证 MuJoCo 功能。
    """

    def test_timestep_round_trip(self):
        config = SimConfig()
        config.timestep = 0.005
        self.assertAlmostEqual(config.timestep, 0.005)

    def test_integrator_round_trip(self):
        config = SimConfig()
        config.integrator = 1
        self.assertEqual(config.integrator, 1)

    def test_iterations_round_trip(self):
        config = SimConfig()
        config.iterations = 200
        self.assertEqual(config.iterations, 200)

    def test_gravity_round_trip(self):
        config = SimConfig()
        new_g = np.array([0.0, 0.0, -10.0])
        config.gravity = new_g
        np.testing.assert_array_equal(config.gravity, new_g)

    def test_load_from_dict_sets_values(self):
        config = SimConfig()
        config.load_from_dict({"timestep": 0.001, "iterations": 50})
        self.assertAlmostEqual(config.timestep, 0.001)
        self.assertEqual(config.iterations, 50)

    def test_to_dict_returns_all_keys(self):
        config = SimConfig()
        d = config.to_dict()
        self.assertIn("timestep", d)
        self.assertIn("integrator", d)
        self.assertIn("iterations", d)
        self.assertIn("gravity", d)


if __name__ == "__main__":
    unittest.main()
