"""OrcaGymEulerEnv 骨架 P3 单元测试。

验证点参见 docs/design/development/orca_gym_euler_development.md 第 4.4 节。
使用离线模式（skip_grpc_load=True）验证骨架 API 契约，不依赖 gRPC 服务器。
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from orca_gym.environment.orca_gym_euler_env import OrcaGymEulerEnv

_SCENE_XML = os.path.join(
    os.path.dirname(__file__), "..", "..", "core", "euler", "fixtures", "test_scene.xml"
)
_SCENE_XML = os.path.abspath(_SCENE_XML)


class _SimpleEulerEnv(OrcaGymEulerEnv):
    """最小可实例化的 Euler Env 子类，用于骨架测试。"""

    def __init__(self, **kwargs):
        super().__init__(
            frame_skip=5,
            orcagym_addr="localhost:50051",
            agent_names=["agent0"],
            time_step=0.002,
            model_xml_path=_SCENE_XML,
            skip_grpc_load=True,
            **kwargs,
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        return obs, 0.0, False, False, {}

    def reset_model(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        return self._get_obs(), {}


class TestOrcaGymEulerEnvSkeleton(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _SimpleEulerEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_env_initializes_without_grpc(self) -> None:
        # 离线模式可成功初始化
        self.assertIsNotNone(self.env.gym)
        self.assertTrue(self.env.gym.studio.is_offline)

    def test_data_returns_orca_gym_data_view(self) -> None:
        # env.data 返回 OrcaGymDataView 实例
        from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView

        self.assertIsInstance(self.env.data, OrcaGymDataView)

    def test_model_returns_orca_gym_model(self) -> None:
        # env.model 返回 OrcaGymModel 实例
        from orca_gym import OrcaGymModel

        self.assertIsInstance(self.env.model, OrcaGymModel)

    def test_sim_config_returns_sim_config(self) -> None:
        # env.sim_config 返回 SimConfig 实例
        from orca_gym.core.euler.sim_config import SimConfig

        self.assertIsInstance(self.env.sim_config, SimConfig)

    def test_init_qpos_qvel_populated(self) -> None:
        # init_qpos / init_qvel 已初始化
        self.assertEqual(self.env.init_qpos.shape, (self.env.model.nq,))
        self.assertEqual(self.env.init_qvel.shape, (self.env.model.nv,))

    def test_step_advances_time(self) -> None:
        # step 后仿真时间推进 dt = time_step * frame_skip
        t0 = float(self.env.data.time)
        ctrl = np.zeros(self.env.model.nu, dtype=np.float32)
        self.env.step(ctrl)
        t1 = float(self.env.data.time)
        expected_dt = self.env.time_step * self.env.frame_skip
        self.assertAlmostEqual(t1 - t0, expected_dt, places=6)

    def test_reset_returns_obs_and_info(self) -> None:
        # reset 返回 (obs, info) 元组
        obs, info = self.env.reset()
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)

    def test_dt_property(self) -> None:
        # dt = time_step * frame_skip
        expected = self.env.time_step * self.env.frame_skip
        self.assertAlmostEqual(self.env.dt, expected, places=6)

    def test_do_simulation_validates_action_dim(self) -> None:
        # 错误维度的 ctrl 抛出 ValueError
        bad_ctrl = np.zeros(self.env.model.nu + 1, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.env.do_simulation(bad_ctrl, self.env.frame_skip)


class TestOrcaGymEulerEnvP3A(unittest.TestCase):
    """P3A：在线模式渲染循环相关测试。"""

    def setUp(self) -> None:
        self.env = _SimpleEulerEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_render_mode_default_human(self) -> None:
        # 默认 render_mode 为 "human"
        self.assertEqual(self.env.render_mode, "human")

    def test_render_mode_none(self) -> None:
        # render_mode="none" 时不渲染
        env = _SimpleEulerEnv(render_mode="none")
        try:
            self.assertEqual(env.render_mode, "none")
            # render() 应直接返回，不调用 gRPC
            env.render()  # 不应抛出异常
        finally:
            env.close()

    def test_sync_render_default_false(self) -> None:
        # 默认 sync_render 为 False
        self.assertFalse(self.env.sync_render)

    def test_sync_render_true(self) -> None:
        # sync_render=True 时启用同步渲染
        env = _SimpleEulerEnv(sync_render=True)
        try:
            self.assertTrue(env.sync_render)
        finally:
            env.close()

    def test_render_skips_in_offline_mode(self) -> None:
        # 离线模式（skip_grpc_load=True）render() 直接返回
        # 不应抛出异常
        self.env.render()

    def test_set_ctrl_applies_override_ctrls(self) -> None:
        # OrcaGymEuler.set_ctrl 应用 override_ctrls
        # 模拟 Studio UI 返回的 override_ctrls
        nu = self.env.model.nu
        if nu == 0:
            self.skipTest("模型无执行器，跳过 override_ctrls 测试")
        # 设置 override_ctrls
        self.env.gym.studio._override_ctrls = {0: 0.5}
        ctrl = np.zeros(nu, dtype=np.float32)
        self.env.gym.set_ctrl(ctrl)
        # 验证 override 被应用
        applied_ctrl = self.env.gym._sim._mjData.ctrl
        self.assertAlmostEqual(float(applied_ctrl[0]), 0.5, places=6)

    def test_set_ctrl_override_does_not_mutate_input(self) -> None:
        # set_ctrl 不应修改输入 ctrl 数组
        nu = self.env.model.nu
        if nu == 0:
            self.skipTest("模型无执行器，跳过测试")
        self.env.gym.studio._override_ctrls = {0: 0.7}
        ctrl = np.zeros(nu, dtype=np.float32)
        ctrl_copy = ctrl.copy()
        self.env.gym.set_ctrl(ctrl)
        np.testing.assert_array_equal(ctrl, ctrl_copy)


if __name__ == "__main__":
    unittest.main()
