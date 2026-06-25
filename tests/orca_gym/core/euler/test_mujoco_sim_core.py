"""MuJoCoSimCore P1 单元测试。

验证点参见 docs/design/development/orca_gym_euler_development.md 第 2.3 节。
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore

_SCENE_XML = os.path.join(os.path.dirname(__file__), "fixtures", "test_scene.xml")


class TestMuJoCoSimCore(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = MuJoCoSimCore()
        self.sim.init_simulation(_SCENE_XML)
        # pendulum body id（world=0, pendulum=1）
        self.pendulum_id = 1

    def test_init_simulation_loads_model(self) -> None:
        # 加载简单 MJCF，_mjModel/_mjData 非 None
        self.assertIsNotNone(self.sim._mjModel)
        self.assertIsNotNone(self.sim._mjData)

    def test_step_advances_time(self) -> None:
        # step(1) 后 _mjData.time 增加 timestep
        t0 = float(self.sim._mjData.time)
        self.sim.step(1)
        t1 = float(self.sim._mjData.time)
        self.assertAlmostEqual(t1 - t0, self.sim._mjModel.opt.timestep)

    def test_forward_updates_derived(self) -> None:
        # forward() 后 qacc 非零（重力作用于偏置摆）
        self.sim.forward()
        self.assertNotEqual(float(self.sim._mjData.qacc[0]), 0.0)

    def test_set_ctrl_sets_actuator(self) -> None:
        # set_ctrl 后 _mjData.ctrl 与输入一致
        ctrl = np.array([0.42], dtype=np.float64)
        self.sim.set_ctrl(ctrl)
        np.testing.assert_allclose(np.asarray(self.sim._mjData.ctrl), ctrl)

    def test_apply_body_force_writes_xfrc(self) -> None:
        # apply_body_force 后 xfrc_applied 对应位置非零
        force = np.array([1.0, 2.0, 3.0])
        torque = np.array([4.0, 5.0, 6.0])
        self.sim.apply_body_force(self.pendulum_id, force, torque)
        np.testing.assert_allclose(
            np.asarray(self.sim._mjData.xfrc_applied[self.pendulum_id, :3]), force
        )
        np.testing.assert_allclose(
            np.asarray(self.sim._mjData.xfrc_applied[self.pendulum_id, 3:]), torque
        )

    def test_clear_body_force_zeros_xfrc(self) -> None:
        # clear_body_force 后对应位置为零
        force = np.array([1.0, 2.0, 3.0])
        torque = np.array([4.0, 5.0, 6.0])
        self.sim.apply_body_force(self.pendulum_id, force, torque)
        self.sim.clear_body_force(self.pendulum_id)
        np.testing.assert_allclose(
            np.asarray(self.sim._mjData.xfrc_applied[self.pendulum_id]), 0.0
        )

    def test_clear_all_forces_zeros_all(self) -> None:
        # clear_all_forces 后 xfrc_applied 全零
        force = np.array([1.0, 2.0, 3.0])
        torque = np.array([4.0, 5.0, 6.0])
        self.sim.apply_body_force(self.pendulum_id, force, torque)
        self.sim.clear_all_forces()
        np.testing.assert_allclose(np.asarray(self.sim._mjData.xfrc_applied), 0.0)


if __name__ == "__main__":
    unittest.main()
