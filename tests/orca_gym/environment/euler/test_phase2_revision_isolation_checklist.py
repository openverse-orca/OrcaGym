"""阶段二变更修订 — Step 5: 隔离机制验收清单（K1-K14 + M0-M7）。

修订老文档 §7:
- K2: __getattr__ 拦截 → M0 原生 AttributeError
- K10: __setattr__ 屏蔽 → 删除
- K14: 新增继承链约束
- §7.2: grep → ruff SLF001 静态扫描
- §7.3: match 模式修订

对齐文档: docs/design/development/orca_gym_euler_phase2_revision_development.md Step 5
"""

import pathlib
import re
import subprocess
import sys
import unittest

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimConfig


def _make_env():
    _pendulum_xml = (
        pathlib.Path(__file__).resolve().parents[4].parent
        / "OrcaPlayground" / "envs" / "euler" / "scenes" / "simple_pendulum.xml"
    )
    return OrcaGymEulerEnv(
        frame_skip=4, orcagym_addr="localhost:50051",
        agent_names=["agent0"], time_step=0.002,
        model_xml_path=str(_pendulum_xml), skip_grpc_load=True,
    )


class TestPhase2K1NamingConstraint(unittest.TestCase):
    """K1: 命名约束 — 内部组件带下划线。"""

    def test_env_no_public_internal_attrs(self):
        env = _make_env()
        self.assertNotIn("gym", env.__dict__)
        self.assertNotIn("stub", env.__dict__)
        self.assertNotIn("channel", env.__dict__)
        self.assertIn("_gym", env.__dict__)
        self.assertIn("_stub", env.__dict__)
        self.assertIn("_channel", env.__dict__)


class TestPhase2K2M0NativeAttributeError(unittest.TestCase):
    """K2 + M0: env.gym/stub/channel 抛原生 AttributeError（无 __getattr__）。"""

    def test_env_gym_native_attribute_error(self):
        """env.gym 抛 AttributeError,消息为 Python 原生格式。"""
        env = _make_env()
        with self.assertRaises(AttributeError) as ctx:
            _ = env.gym
        # M0: 原生 AttributeError,不含旧 __getattr__ 的自定义引导
        self.assertNotIn("公共", str(ctx.exception))
        self.assertNotIn("API 契约", str(ctx.exception))

    def test_env_stub_native_attribute_error(self):
        env = _make_env()
        with self.assertRaises(AttributeError):
            _ = env.stub

    def test_env_channel_native_attribute_error(self):
        env = _make_env()
        with self.assertRaises(AttributeError):
            _ = env.channel

    def test_env_no_getattr_classvar(self):
        """Env 类不定义 __getattr__(M0 替代)。"""
        self.assertNotIn("__getattr__", vars(OrcaGymEulerEnv))


class TestPhase2K10Deleted(unittest.TestCase):
    """K10: __setattr__ 屏蔽机制已删除。"""

    def test_env_no_setattr_classvar(self):
        """Env 类不定义 __setattr__。"""
        self.assertNotIn("__setattr__", vars(OrcaGymEulerEnv))

    def test_env_no_shielded_attrs_classvar(self):
        """Env 类不定义 _SHIELDED_ATTRS。"""
        self.assertNotIn("_SHIELDED_ATTRS", vars(OrcaGymEulerEnv))

    def test_env_attribute_assignment_works(self):
        """Env 实例属性赋值正常工作(无 __setattr__ 屏蔽)。"""
        env = _make_env()
        env._test_field = "test_value"
        self.assertEqual(env._test_field, "test_value")


class TestPhase2K14Inheritance(unittest.TestCase):
    """K14: 继承链约束。"""

    def test_inheritance_chain(self):
        from orca_gym.environment.orca_gym_env_mixin import OrcaGymEnvMixin
        from orca_gym.environment.orca_gym_env import OrcaGymBaseEnv
        import gymnasium as gym
        bases = OrcaGymEulerEnv.__bases__
        self.assertIn(OrcaGymEnvMixin, bases)
        self.assertIn(gym.Env, bases)
        self.assertNotIn(OrcaGymBaseEnv, bases)


class TestPhase2K4K8K9SourceAudit(unittest.TestCase):
    """K4/K8/K9: 源码审查(Env 不穿墙访问 Gym 私有)。"""

    @classmethod
    def setUpClass(cls):
        env_file = (
            pathlib.Path(__file__).resolve().parents[4]
            / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
        )
        source = env_file.read_text(encoding="utf-8")
        # 去除 docstring 和注释
        source = re.sub(r'"""[\s\S]*?"""', '', source)
        cls.exec_source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )

    def test_k4_no_gym_private_access(self):
        """K4: Env 可执行代码不含 _gym._sim/_studio/_registry/_opt/_view/_euler。"""
        for pattern in ["_gym._sim", "_gym._studio", "_gym._registry",
                        "_gym._opt", "_gym._view", "_gym._euler"]:
            with self.subTest(pattern=pattern):
                self.assertNotIn(pattern, self.exec_source)

    def test_k8_no_euler_private_access(self):
        """K8: Env 可执行代码不含 _euler 属性访问。"""
        match = re.search(r'(?<![\w])_euler(?![\w])', self.exec_source)
        self.assertIsNone(match)

    def test_k9_no_studio_property_access(self):
        """K9: Env 可执行代码不含 gym.studio 穿墙(允许 _gym.studio_bridge())。"""
        cleaned = self.exec_source.replace("_gym.studio_bridge", "")
        self.assertNotIn("gym.studio", cleaned)


class TestPhase2K6K7K11TypedReturn(unittest.TestCase):
    """K6/K7/K11: 类型化返回。"""

    def test_k6_data_returns_dataview(self):
        env = _make_env()
        self.assertIsInstance(env.data, OrcaGymDataView)

    def test_k7_sim_config_returns_config(self):
        env = _make_env()
        self.assertIsInstance(env.sim_config, SimConfig)

    def test_k11_data_not_mjdata(self):
        env = _make_env()
        self.assertNotEqual(type(env.data).__name__, "MjData")


class TestPhase2K12Docstring(unittest.TestCase):
    """K12: docstring 含使用契约。"""

    def test_env_docstring_has_contract(self):
        doc = OrcaGymEulerEnv.__doc__ or ""
        self.assertIn("使用契约", doc)
        self.assertIn("禁止", doc)


class TestPhase2M1RuffSLF001StaticCheck(unittest.TestCase):
    """M1: ruff SLF001 静态检查(Euler 代码零报警)。"""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = pathlib.Path(__file__).resolve().parents[4]

    def test_ruff_slf001_euler_env_clean(self):
        """Euler Env 源码 ruff SLF001 零报警。"""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "SLF001",
             str(self.repo_root / "orca_gym" / "environment" / "euler")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
                         f"ruff SLF001 报警:\n{result.stdout}")

    def test_ruff_slf001_euler_core_clean(self):
        """Euler Core 源码 ruff SLF001 零报警。"""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "SLF001",
             str(self.repo_root / "orca_gym" / "core" / "euler")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
                         f"ruff SLF001 报警:\n{result.stdout}")


class TestPhase2M3DirControl(unittest.TestCase):
    """M3: __dir__ 控制(只暴露公共 API)。"""

    def test_env_dir_no_internal(self):
        """dir(env) 不含 gym/stub/channel/_gym/_studio_bridge/_mjData/_mjModel。"""
        env = _make_env()
        d = dir(env)
        for name in ["gym", "stub", "channel", "_gym", "_stub", "_channel",
                     "_studio_bridge", "_mjData", "_mjModel"]:
            with self.subTest(attr=name):
                self.assertNotIn(name, d)

    def test_env_dir_contains_public_api(self):
        """dir(env) 含公共 API。"""
        env = _make_env()
        d = dir(env)
        for name in ["data", "model", "sim_config", "dt", "ctrl",
                     "do_simulation", "mj_step", "mj_forward", "render"]:
            with self.subTest(attr=name):
                self.assertIn(name, d)


if __name__ == "__main__":
    unittest.main()
