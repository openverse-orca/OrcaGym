"""OrcaGymEuler P1 单元测试。

验证点参见 docs/design/development/orca_gym_euler_development.md 第 2.3 节。
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler

_SCENE_XML = os.path.join(os.path.dirname(__file__), "fixtures", "test_scene.xml")


class TestOrcaGymEuler(unittest.TestCase):
    def setUp(self) -> None:
        self.gym = OrcaGymEuler(stub=None)

    def test_blocked_attrs_raise_guidance_error(self) -> None:
        # 访问 _mjData/_mjModel 抛出 AttributeError 且消息含引导文本
        for attr in ("_mjData", "_mjModel", "mj_data", "mjModel"):
            with self.subTest(attr=attr):
                with self.assertRaises(AttributeError) as ctx:
                    getattr(self.gym, attr)
                msg = str(ctx.exception)
                self.assertIn("env.data", msg)
                self.assertIn("apply_body_force", msg)

    def test_dir_only_exposes_public_api(self) -> None:
        # dir(gym) 不含 _mjData/_mjModel/_sim
        visible = set(dir(self.gym))
        self.assertNotIn("_mjData", visible)
        self.assertNotIn("_mjModel", visible)
        self.assertNotIn("_sim", visible)
        # 公共 API 可见
        self.assertIn("init_simulation", visible)
        self.assertIn("mj_step", visible)

    def test_init_simulation_delegates_to_sim_core(self) -> None:
        # gym.init_simulation(path) 后 sim_core 已加载
        self.gym.init_simulation(_SCENE_XML)
        self.assertIsNotNone(self.gym._sim._mjModel)
        self.assertIsNotNone(self.gym._sim._mjData)

    def test_mj_step_delegates(self) -> None:
        # gym.mj_step(1) 后 time 推进
        self.gym.init_simulation(_SCENE_XML)
        t0 = float(self.gym._sim._mjData.time)
        self.gym.mj_step(1)
        t1 = float(self.gym._sim._mjData.time)
        self.assertAlmostEqual(t1 - t0, self.gym._sim._mjModel.opt.timestep)

    def test_mj_forward_delegates(self) -> None:
        # gym.mj_forward() 不报错
        self.gym.init_simulation(_SCENE_XML)
        self.gym.mj_forward()

    def test_set_ctrl_delegates(self) -> None:
        # gym.set_ctrl(ctrl) 后 sim_core 的 ctrl 一致
        self.gym.init_simulation(_SCENE_XML)
        ctrl = np.array([0.7], dtype=np.float64)
        self.gym.set_ctrl(ctrl)
        np.testing.assert_allclose(np.asarray(self.gym._sim._mjData.ctrl), ctrl)


if __name__ == "__main__":
    unittest.main()
