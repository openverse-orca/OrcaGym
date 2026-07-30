"""OrcaGymEnvMixin 单元测试。

验证 Mixin 方法存在性、可独立调用、不依赖引擎特定字段。
"""

import unittest
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace

from orca_gym.environment.orca_gym_env_mixin import OrcaGymEnvMixin


class _DummyEnv(OrcaGymEnvMixin, gym.Env):
    """最小化 Env 桩，仅提供 Mixin 依赖的字段/方法。

    继承 gym.Env 使 Mixin.reset() 中 super().reset(seed=seed) 能走 MRO 找到 gym.Env.reset，
    与实际 OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env) 一致。
    """

    def __init__(self, agent_names: list[str]):
        self._agent_names = agent_names
        self._reset_called = False
        self._reset_model_called = False
        self._render_called = False

    def reset_simulation(self):
        self._reset_called = True

    def reset_model(self):
        self._reset_model_called = True
        return {"obs": np.zeros(3)}, {}

    def render(self):
        self._render_called = True
        return None


class TestMixinStructure(unittest.TestCase):
    """Mixin 结构约束。"""

    def test_mixin_has_no_init(self):
        """Mixin 不定义 __init__。"""
        self.assertNotIn("__init__", OrcaGymEnvMixin.__dict__)

    def test_mixin_methods_exist(self):
        """Mixin 包含全部 10 个公共方法。"""
        expected = [
            "body", "joint", "actuator", "site", "mocap", "sensor",
            "_name_with_agent0", "_name_with_agent",
            "generate_action_space", "generate_observation_space",
            "reset", "set_seed_value", "_get_reset_info",
            "agent_num",
        ]
        for name in expected:
            with self.subTest(method=name):
                self.assertTrue(hasattr(OrcaGymEnvMixin, name),
                                f"Mixin 缺少方法 '{name}'")


class TestMixinNamespace(unittest.TestCase):
    """名称空间解析。"""

    def test_body_with_agent0_prefix(self):
        env = _DummyEnv(["agent0", "agent1"])
        self.assertEqual(env.body("torso"), "agent0_torso")

    def test_body_with_agent_id(self):
        env = _DummyEnv(["agent0", "agent1"])
        self.assertEqual(env.body("torso", agent_id=1), "agent1_torso")

    def test_body_no_agent_names(self):
        env = _DummyEnv([])
        self.assertEqual(env.body("torso"), "torso")

    def test_all_namespace_methods_work(self):
        env = _DummyEnv(["agent0"])
        for method in ["body", "joint", "actuator", "site", "mocap", "sensor"]:
            with self.subTest(method=method):
                result = getattr(env, method)("test_name")
                self.assertEqual(result, "agent0_test_name")


class TestMixinSpaceGeneration(unittest.TestCase):
    """动作/观测空间生成。"""

    def test_generate_action_space(self):
        env = _DummyEnv(["agent0"])
        bounds = np.array([[0.0, 1.0], [-1.0, 1.0]])
        space = env.generate_action_space(bounds)
        self.assertIsInstance(space, Box)
        self.assertEqual(space.shape, (2,))

    def test_generate_observation_space_array(self):
        env = _DummyEnv(["agent0"])
        obs = np.zeros(5)
        space = env.generate_observation_space(obs)
        self.assertIsInstance(space, Box)
        self.assertEqual(space.shape, (5,))

    def test_generate_observation_space_dict(self):
        env = _DummyEnv(["agent0"])
        obs = {"a": np.zeros(3), "b": np.zeros(2)}
        space = env.generate_observation_space(obs)
        self.assertIsInstance(space, DictSpace)
        self.assertIn("a", space.spaces)
        self.assertIn("b", space.spaces)


class TestMixinReset(unittest.TestCase):
    """reset 编排。"""

    def test_reset_calls_lifecycle(self):
        env = _DummyEnv(["agent0"])
        env.reset()
        self.assertTrue(env._reset_called)
        self.assertTrue(env._reset_model_called)
        self.assertTrue(env._render_called)

    def test_reset_returns_obs_info(self):
        env = _DummyEnv(["agent0"])
        obs, info = env.reset()
        self.assertIn("obs", obs)
        self.assertIsInstance(info, dict)

    def test_reset_with_seed(self):
        env = _DummyEnv(["agent0"])
        env.reset(seed=42)
        self.assertEqual(env.seed_value, 42)


class TestMixinAgentNum(unittest.TestCase):
    """agent_num property。"""

    def test_agent_num(self):
        env = _DummyEnv(["agent0", "agent1", "agent2"])
        self.assertEqual(env.agent_num, 3)


if __name__ == "__main__":
    unittest.main()
