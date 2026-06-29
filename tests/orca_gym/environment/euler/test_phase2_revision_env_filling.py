"""阶段二变更修订 — Step 2: Env 层填充与新骨架兼容性验证。

验证 K14 继承链、M0 原生 AttributeError、自主编排生命周期、
SimConfig _bind 缓存路径、步进/状态设置委托链路。

对齐文档: docs/design/development/orca_gym_euler_phase2_revision_development.md Step 2
"""

import pathlib
import re
import unittest

import numpy as np

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


_PENDULUM_XML = (
    pathlib.Path(__file__).resolve().parents[4].parent
    / "OrcaPlayground" / "envs" / "euler" / "scenes" / "simple_pendulum.xml"
)


def _make_env():
    """构造离线模式 Env。"""
    return OrcaGymEulerEnv(
        frame_skip=4,
        orcagym_addr="localhost:50051",
        agent_names=["agent0"],
        time_step=0.002,
        model_xml_path=str(_PENDULUM_XML),
        skip_grpc_load=True,
    )


class TestPhase2EnvK14Inheritance(unittest.TestCase):
    """K14: 继承链约束。"""

    def test_inheritance_chain(self):
        """OrcaGymEulerEnv.__bases__ 含 gym.Env + OrcaGymEnvMixin，不含 OrcaGymBaseEnv。"""
        from orca_gym.environment.orca_gym_env_mixin import OrcaGymEnvMixin
        from orca_gym.environment.orca_gym_env import OrcaGymBaseEnv
        import gymnasium as gym

        bases = OrcaGymEulerEnv.__bases__
        self.assertIn(OrcaGymEnvMixin, bases)
        self.assertIn(gym.Env, bases)
        self.assertNotIn(OrcaGymBaseEnv, bases)


class TestPhase2EnvM0NativeAttributeError(unittest.TestCase):
    """M0: env.gym/stub/channel 抛 Python 原生 AttributeError（无自定义消息）。"""

    def test_env_gym_raises_native_attribute_error(self):
        """env.gym 抛 AttributeError，消息不含自定义引导文本。"""
        env = _make_env()
        with self.assertRaises(AttributeError) as ctx:
            _ = env.gym
        # M0: 原生 AttributeError，消息为 "'OrcaGymEulerEnv' object has no attribute 'gym'"
        # 不含"通过公共 API 访问"等自定义引导（那是旧 __getattr__ 的消息）
        self.assertNotIn("公共", str(ctx.exception))
        self.assertNotIn("API 契约", str(ctx.exception))

    def test_env_stub_raises_native_attribute_error(self):
        env = _make_env()
        with self.assertRaises(AttributeError):
            _ = env.stub

    def test_env_channel_raises_native_attribute_error(self):
        env = _make_env()
        with self.assertRaises(AttributeError):
            _ = env.channel

    def test_env_no_getattr_method(self):
        """Env 类不定义 __getattr__（M0 替代）。"""
        self.assertNotIn("__getattr__", vars(OrcaGymEulerEnv))

    def test_env_no_setattr_method(self):
        """Env 类不定义 __setattr__（K10 删除）。"""
        self.assertNotIn("__setattr__", vars(OrcaGymEulerEnv))


class TestPhase2EnvSelfOrchestratedLifecycle(unittest.TestCase):
    """Env __init__ 自主编排生命周期（不调 super().__init__）。"""

    def test_init_completes_without_error(self):
        """__init__ 完整执行不报错。"""
        env = _make_env()
        self.assertIsNotNone(env._gym)

    def test_init_orchestrates_lifecycle_in_order(self):
        """__init__ 按序调用生命周期方法（initialize_grpc → ... → init_qpos_qvel）。"""
        env = _make_env()
        # 验证生命周期已执行：_gym 已创建、init_qpos/init_qvel 已保存
        self.assertIsNotNone(env._gym)
        self.assertTrue(hasattr(env, "init_qpos"))
        self.assertTrue(hasattr(env, "init_qvel"))
        self.assertEqual(env.init_qpos.shape, (1,))   # pendulum nq=1

    def test_init_does_not_call_super_init(self):
        """Env 不调 super().__init__()（不触发 OrcaGymBaseEnv 编排）。"""
        # 直接读源码文件
        env_file = (
            pathlib.Path(__file__).resolve().parents[4]
            / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
        )
        exec_source = re.sub(r'"""[\s\S]*?"""', '', env_file.read_text(encoding="utf-8"))
        # 去除注释行
        exec_source = "\n".join(
            line for line in exec_source.splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertNotIn("super().__init__()", exec_source)


class TestPhase2EnvSimConfigBindCachePath(unittest.TestCase):
    """SimConfig _bind 缓存路径：set_time_step 在 init_simulation 前调用。"""

    def test_time_step_cached_before_init(self):
        """__init__ 中 set_time_step 在 init_simulation 前调用，_time_step 缓存生效。"""
        env = _make_env()
        # 验证 _time_step 已缓存
        self.assertEqual(env._time_step, 0.002)
        # 验证 init_simulation 后 sim_config.timestep 已应用缓存值
        self.assertAlmostEqual(env.sim_config.timestep, 0.002)

    def test_sim_config_bound_after_init(self):
        """init_simulation 后 SimConfig 已绑定 mjModel。"""
        env = _make_env()
        # 绑定后 setter 应写入 mjModel.opt（非缓存）
        env.sim_config.timestep = 0.005
        self.assertAlmostEqual(env.sim_config.timestep, 0.005)

    def test_dt_uses_sim_config(self):
        """env.dt = sim_config.timestep * frame_skip（K7）。"""
        env = _make_env()
        expected = env.sim_config.timestep * env.frame_skip
        self.assertAlmostEqual(env.dt, expected)


class TestPhase2EnvSteppingAndStateSetting(unittest.TestCase):
    """步进与状态设置委托链路（K4/K8）。"""

    def test_do_simulation_delegates_to_step_with_coupling(self):
        """do_simulation 委托 _gym.step_with_coupling（K8: 不读 _euler）。"""
        env = _make_env()
        time_before = float(env.data.time)
        env.do_simulation(np.array([0.0]), 5)
        time_after = float(env.data.time)
        expected_dt = 5 * env.sim_config.timestep
        self.assertAlmostEqual(time_after - time_before, expected_dt, places=5)

    def test_set_joint_qpos_delegates_to_gym(self):
        """set_joint_qpos 委托 _gym.set_qpos_qvel（K4: 走公共方法）。"""
        env = _make_env()
        env.set_joint_qpos(np.array([0.5]))
        env.mj_forward()
        env._gym.sync_to_view()
        self.assertAlmostEqual(float(env.data.qpos[0]), 0.5)

    def test_set_joint_qvel_delegates_to_gym(self):
        """set_joint_qvel 委托 _gym.set_qpos_qvel（K4: 走公共方法）。"""
        env = _make_env()
        env.set_joint_qvel(np.array([0.3]))
        env.mj_forward()
        env._gym.sync_to_view()
        self.assertAlmostEqual(float(env.data.qvel[0]), 0.3)

    def test_data_returns_dataview(self):
        """env.data 返回 OrcaGymDataView（K6）。"""
        env = _make_env()
        self.assertIsInstance(env.data, OrcaGymDataView)

    def test_do_simulation_validates_action_dim(self):
        """do_simulation 对错误维度抛 ValueError。"""
        env = _make_env()
        with self.assertRaises(ValueError):
            env.do_simulation(np.zeros(0), 1)


if __name__ == "__main__":
    unittest.main()
