"""SimConfig P2 单元测试。

验证点参见 docs/design/development/orca_gym_euler_development.md 第 3.3 节。
"""

from __future__ import annotations

import os
import unittest

import mujoco
import numpy as np

from orca_gym.core.euler.sim_config import SimConfig

_SCENE_XML = os.path.join(os.path.dirname(__file__), "fixtures", "test_scene.xml")


class TestSimConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(_SCENE_XML)
        self.config = SimConfig(self.model)

    def test_timestep_get_set(self) -> None:
        # 读写 timestep 反映到 _mjModel.opt.timestep
        self.config.timestep = 0.005
        self.assertAlmostEqual(self.model.opt.timestep, 0.005)
        self.assertAlmostEqual(self.config.timestep, 0.005)

    def test_integrator_get_set(self) -> None:
        # 读写 integrator 反映到 _mjModel.opt.integrator
        self.config.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.assertEqual(self.model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
        self.assertEqual(self.config.integrator, mujoco.mjtIntegrator.mjINT_EULER)

    def test_iterations_get_set(self) -> None:
        # 读写 iterations 反映到 _mjModel.opt.iterations
        self.config.iterations = 50
        self.assertEqual(self.model.opt.iterations, 50)
        self.assertEqual(self.config.iterations, 50)

    def test_gravity_get_set(self) -> None:
        # 读写 gravity 反映到 _mjModel.opt.gravity
        new_gravity = np.array([0.0, 0.0, -5.0])
        self.config.gravity = new_gravity
        np.testing.assert_allclose(self.model.opt.gravity, new_gravity)
        np.testing.assert_allclose(self.config.gravity, new_gravity)

    def test_load_from_dict(self) -> None:
        # load_from_dict({...}) 批量设置多个参数
        self.config.load_from_dict({
            "timestep": 0.001,
            "iterations": 200,
            "integrator": mujoco.mjtIntegrator.mjINT_EULER,
        })
        self.assertAlmostEqual(self.model.opt.timestep, 0.001)
        self.assertEqual(self.model.opt.iterations, 200)
        self.assertEqual(self.model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)

    def test_to_dict(self) -> None:
        # to_dict() 返回所有参数的字典
        d = self.config.to_dict()
        self.assertIn("timestep", d)
        self.assertIn("integrator", d)
        self.assertIn("iterations", d)
        self.assertIn("gravity", d)
        self.assertIn("filterparent", d)
        # 值与 opt 一致
        self.assertAlmostEqual(d["timestep"], self.model.opt.timestep)
        np.testing.assert_allclose(d["gravity"], self.model.opt.gravity)

    def test_all_opt_fields_covered(self) -> None:
        # 遍历 _mjModel.opt 所有字段，确认 SimConfig 都有对应属性
        opt = self.model.opt
        opt_fields = [
            n for n in dir(opt)
            if not n.startswith("_") and not callable(getattr(opt, n, None))
        ]
        for field in opt_fields:
            with self.subTest(field=field):
                self.assertTrue(
                    hasattr(self.config, field),
                    f"SimConfig 缺少 opt 字段 '{field}'",
                )


if __name__ == "__main__":
    unittest.main()
