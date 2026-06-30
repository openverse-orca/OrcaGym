"""阶段二-Step 7: SimpleEulerEnv 架构违规修正回归测试。

验证 SimpleEulerEnv.reset_model 不再穿墙访问 _sim._mjData（K3/K5），
改用 Env 公共方法 set_joint_qpos/set_joint_qvel/mj_forward/_sync_view。

验收标准（对应 docs/design/development/orca_gym_euler_phase2_filling_development.md）:
- reset_model 不再出现 self.gym._sim / self._gym._sim 穿墙访问
- 源码 grep `.gym._sim` / `.gym._mjData` / `.gym._mjModel` 无结果
- env.reset() 后 env.data.qpos 反映随机扰动后的初始状态

运行方式:
    <conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.test_simple_euler_env_compliance
"""

import pathlib
import re
import unittest

import numpy as np

# simple_env.py 位于 OrcaPlayground（OrcaGym 同级目录）
# __file__ = OrcaGym/tests/orca_gym/environment/euler/test_*.py
# parents[4] = OrcaGym，再向上一级到 repo 根，再到 OrcaPlayground
_SIMPLE_ENV_PATH = (
    pathlib.Path(__file__).resolve().parents[4].parent
    / "OrcaPlayground" / "envs" / "euler" / "simple_env.py"
)


def _read_simple_env_source() -> str:
    """读取 simple_env.py 源码（用于 K3/K5 源码审查）。"""
    return _SIMPLE_ENV_PATH.read_text(encoding="utf-8")


class TestSimpleEnvK3K5NoTunnelAccess(unittest.TestCase):
    """K3/K5: reset_model 不穿墙访问 _sim._mjData。"""

    def test_no_gym_sim_tunnel(self):
        """源码不含 .gym._sim / _gym._sim 穿墙访问。"""
        source = _read_simple_env_source()
        forbidden = [".gym._sim", "_gym._sim"]
        for pattern in forbidden:
            self.assertNotIn(
                pattern, source,
                f"K3/K5 违规: simple_env.py 包含穿墙访问 '{pattern}'",
            )

    def test_no_mjdata_mjmodel_tunnel(self):
        """源码不含 .gym._mjData / .gym._mjModel 穿墙访问。"""
        source = _read_simple_env_source()
        forbidden = [".gym._mjData", ".gym._mjModel", "_gym._mjData", "_gym._mjModel"]
        for pattern in forbidden:
            self.assertNotIn(
                pattern, source,
                f"K3/K5 违规: simple_env.py 包含穿墙访问 '{pattern}'",
            )

    def test_reset_model_uses_compliant_api(self):
        """reset_model 可执行代码使用 set_joint_qpos/set_joint_qvel/mj_forward/_sync_view。"""
        source = _read_simple_env_source()
        # 提取 reset_model 方法体（从 def reset_model 到下一个 def）
        match = re.search(
            r"def reset_model\(self\):(.*?)(?=\n    def |\nclass |\Z)",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "reset_model 方法未找到")
        body = match.group(1)
        # 合规 API 应出现
        self.assertIn("set_joint_qpos", body)
        self.assertIn("set_joint_qvel", body)
        self.assertIn("mj_forward", body)
        self.assertIn("_sync_view", body)
        # 违规 API 不应出现
        self.assertNotIn("_sim._mjData", body)
        self.assertNotIn("self.gym.mj_forward", body)
        self.assertNotIn("self.gym.sync_to_view", body)

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

    def test_no_xfrc_applied_direct_write(self):
        """源码不含直接写 _mjData.xfrc_applied（W2 应走 apply_body_force）。"""
        source = _read_simple_env_source()
        for pattern in ["xfrc_applied", "_mjData.xfrc", "_mjData.qpos"]:
            self.assertNotIn(pattern, source)


class TestSimpleEnvResetModelFunctional(unittest.TestCase):
    """功能验证: reset_model 正确写入随机扰动后的状态。"""

    @classmethod
    def setUpClass(cls):
        """构造 SimpleEulerEnv（离线模式）。"""
        import sys
        _orca_playground = str(
            pathlib.Path(__file__).resolve().parents[4].parent / "OrcaPlayground"
        )
        if _orca_playground not in sys.path:
            sys.path.insert(0, _orca_playground)
        from envs.euler.simple_env import SimpleEulerEnv
        cls.SimpleEulerEnv = SimpleEulerEnv
        cls.env = SimpleEulerEnv()

    @classmethod
    def tearDownClass(cls):
        """清理环境。"""
        del cls.env

    def test_reset_model_writes_perturbed_qpos(self):
        """reset_model 后 qpos 反映随机扰动（与 init_qpos 不同，但在 ±0.1 范围内）。"""
        env = self.env
        env.reset_simulation()
        env.init_qpos_qvel()
        # 固定随机种子以保证可重复
        env.np_random = np.random.RandomState(42)
        obs, _ = env.reset_model()
        # pendulum init_qpos=[0.]，扰动后 qpos 应非 0 且在 ±0.1 范围
        qpos = float(env.data.qpos[0])
        self.assertNotAlmostEqual(qpos, 0.0, places=3,
                                  msg="reset_model 未写入扰动后的 qpos")
        self.assertLessEqual(abs(qpos), 0.1 + 1e-6,
                             msg="qpos 扰动超出 ±0.1 范围")

    def test_reset_model_writes_perturbed_qvel(self):
        """reset_model 后 qvel 反映随机扰动。"""
        env = self.env
        env.reset_simulation()
        env.init_qpos_qvel()
        env.np_random = np.random.RandomState(42)
        env.reset_model()
        qvel = float(env.data.qvel[0])
        self.assertLessEqual(abs(qvel), 0.1 + 1e-6,
                             msg="qvel 扰动超出 ±0.1 范围")

    def test_reset_model_returns_correct_obs(self):
        """reset_model 返回 obs.shape=(3,) = [cos, sin, theta_dot]。"""
        env = self.env
        env.reset_simulation()
        env.init_qpos_qvel()
        env.np_random = np.random.RandomState(42)
        obs, info = env.reset_model()
        self.assertEqual(obs.shape, (3,))
        # obs = [cos(theta), sin(theta), theta_dot]
        theta = float(env.data.qpos[0])
        theta_dot = float(env.data.qvel[0])
        np.testing.assert_array_almost_equal(
            obs, [np.cos(theta), np.sin(theta), theta_dot], decimal=5
        )

    def test_reset_model_repeatable(self):
        """reset_model 可多次调用，每次都写入新的随机扰动。"""
        env = self.env
        env.reset_simulation()
        env.init_qpos_qvel()
        env.np_random = np.random.RandomState(1)
        env.reset_model()
        qpos1 = float(env.data.qpos[0])
        env.np_random = np.random.RandomState(2)
        env.reset_model()
        qpos2 = float(env.data.qpos[0])
        # 不同种子应产生不同扰动
        self.assertNotAlmostEqual(qpos1, qpos2, places=3)

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
