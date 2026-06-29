"""阶段二变更修订 — Step 4: SimpleEulerEnv 违规修正验证。

验证 M0 下违规模式不可达、合规 API 使用、功能正确。

对齐文档: docs/design/development/orca_gym_euler_phase2_revision_development.md Step 4
"""

import pathlib
import re
import sys
import unittest

import numpy as np

_SIMPLE_ENV_PATH = (
    pathlib.Path(__file__).resolve().parents[4].parent
    / "OrcaPlayground" / "envs" / "euler" / "simple_env.py"
)


def _read_simple_env_source() -> str:
    return _SIMPLE_ENV_PATH.read_text(encoding="utf-8")


class TestPhase2SimpleEnvNoTunnelAccess(unittest.TestCase):
    """M0/K3/K5: 无穿墙访问。"""

    def test_no_gym_sim_tunnel(self):
        """源码不含 .gym._sim / _gym._sim 穿墙。"""
        source = _read_simple_env_source()
        for pattern in [".gym._sim", "_gym._sim"]:
            self.assertNotIn(pattern, source)

    def test_no_mjdata_mjmodel_tunnel(self):
        """源码不含 .gym._mjData / .gym._mjModel 穿墙。"""
        source = _read_simple_env_source()
        for pattern in [".gym._mjData", ".gym._mjModel", "_gym._mjData", "_gym._mjModel"]:
            self.assertNotIn(pattern, source)

    def test_no_self_gym_access(self):
        """源码不含 self.gym 访问（M0: env.gym 不存在）。

        旧文档违规点 self.gym._sim._mjData 在 M0 下不可达,
        但仍需确认源码无 self.gym 残留（会在运行时抛 AttributeError）。
        """
        source = _read_simple_env_source()
        # 排除注释和 docstring
        exec_source = re.sub(r'"""[\s\S]*?"""', '', source)
        exec_source = re.sub(r'#.*', '', exec_source)
        self.assertNotIn("self.gym", exec_source,
                         "M0 违规: simple_env.py 含 self.gym 访问（env.gym 不存在）")

    def test_reset_model_uses_compliant_api(self):
        """reset_model 使用 set_joint_qpos/set_joint_qvel/mj_forward/_sync_view。"""
        source = _read_simple_env_source()
        match = re.search(
            r"def reset_model\(self\):(.*?)(?=\n    def |\nclass |\Z)",
            source, re.DOTALL,
        )
        self.assertIsNotNone(match, "reset_model 方法未找到")
        body = match.group(1)
        self.assertIn("set_joint_qpos", body)
        self.assertIn("set_joint_qvel", body)
        self.assertIn("mj_forward", body)
        self.assertIn("_sync_view", body)
        self.assertNotIn("_sim._mjData", body)

    def test_no_xfrc_applied_direct_write(self):
        """源码不含直接写 _mjData.xfrc_applied（W2 应走 apply_body_force）。"""
        source = _read_simple_env_source()
        for pattern in ["xfrc_applied", "_mjData.xfrc", "_mjData.qpos"]:
            self.assertNotIn(pattern, source)


class TestPhase2SimpleEnvFunctional(unittest.TestCase):
    """功能验证: reset_model 与 step 在新架构下正常工作。"""

    @classmethod
    def setUpClass(cls):
        _orca_playground = str(
            pathlib.Path(__file__).resolve().parents[4].parent / "OrcaPlayground"
        )
        if _orca_playground not in sys.path:
            sys.path.insert(0, _orca_playground)
        from envs.euler.simple_env import SimpleEulerEnv
        cls.env = SimpleEulerEnv()

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def test_reset_model_writes_perturbed_state(self):
        """reset_model 后 qpos 反映随机扰动。"""
        env = self.env
        env.reset_simulation()
        env.init_qpos_qvel()
        env.np_random = np.random.RandomState(42)
        env.reset_model()
        qpos = float(env.data.qpos[0])
        self.assertNotAlmostEqual(qpos, 0.0, places=3)
        self.assertLessEqual(abs(qpos), 0.1 + 1e-6)

    def test_step_works_after_reset_model(self):
        """reset_model 后 step 正常工作（time 累计正确）。"""
        env = self.env
        env.reset_simulation()
        env.init_qpos_qvel()
        env.np_random = np.random.RandomState(42)
        env.reset_model()
        time_before = float(env.data.time)
        env.step(np.array([0.0], dtype=np.float32))
        time_after = float(env.data.time)
        expected_dt = env.frame_skip * env.sim_config.timestep
        self.assertAlmostEqual(time_after - time_before, expected_dt, places=5)

    def test_env_gym_raises_attribute_error(self):
        """M0: SimpleEulerEnv 继承 Env，env.gym 抛原生 AttributeError。"""
        env = self.env
        with self.assertRaises(AttributeError):
            _ = env.gym

    def test_step_returns_gymnasium_tuple(self):
        """step 返回 Gymnasium 5-tuple (obs, reward, terminated, truncated, info)。"""
        env = self.env
        env.reset_simulation()
        env.init_qpos_qvel()
        env.np_random = np.random.RandomState(42)
        env.reset_model()
        result = env.step(np.array([0.0], dtype=np.float32))
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(obs.shape, (3,))   # [cos, sin, theta_dot]
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_observation_space_box(self):
        """observation_space 是 Box,形状 (3,)。"""
        from gymnasium import spaces
        env = self.env
        self.assertIsInstance(env.observation_space, spaces.Box)
        self.assertEqual(env.observation_space.shape, (3,))


if __name__ == "__main__":
    unittest.main()
