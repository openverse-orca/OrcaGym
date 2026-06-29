"""阶段二变更修订 — Step 1: 基线建立与影响面扫描。

验证 Euler 代码已对齐新骨架架构（K14 继承链、无补丁机制、无旧式赋值）。

对齐文档: docs/design/development/orca_gym_euler_phase2_revision_development.md Step 1
"""

import pathlib
import re
import unittest

_ENV_SOURCE = (
    pathlib.Path(__file__).resolve().parents[4]
    / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
).read_text(encoding="utf-8")


def _strip_docstrings(source: str) -> str:
    """去除源码中的 docstring, 仅保留可执行代码用于模式扫描。"""
    return re.sub(r'"""[\s\S]*?"""', '', source)


def _strip_comments(source: str) -> str:
    """去除源码中的注释行。"""
    return "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )


class TestPhase2BaselineNoLegacySkeleton(unittest.TestCase):
    """Env 源码不含旧骨架残留。"""

    def test_no_old_inheritance_chain(self):
        """Env 类不继承 OrcaGymBaseEnv（K14）。"""
        self.assertNotIn(
            "class OrcaGymEulerEnv(OrcaGymBaseEnv)", _ENV_SOURCE,
            "K14 违规: Env 仍继承 OrcaGymBaseEnv",
        )

    def test_no_blocked_attrs_in_env(self):
        """Env 类不定义 _BLOCKED_ATTRS（补丁机制已删除）。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        self.assertNotIn("_BLOCKED_ATTRS", exec_source)

    def test_no_getattr_in_env(self):
        """Env 类不定义 __getattr__（M0 替代）。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        self.assertNotIn("def __getattr__", exec_source)

    def test_no_setattr_in_env(self):
        """Env 类不定义 __setattr__（K10 删除）。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        self.assertNotIn("def __setattr__", exec_source)

    def test_no_super_init_in_env(self):
        """Env __init__ 不调用 super().__init__()（自主编排）。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        # 去除注释后再检查, 避免 docstring 中的描述性文字误判
        exec_source = _strip_comments(exec_source)
        self.assertNotIn("super().__init__()", exec_source)

    def test_no_public_gym_assignment(self):
        """Env 源码不出现 self.gym = 赋值（应为 self._gym = ）。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        self.assertNotIn("self.gym =", exec_source)

    def test_no_object_setattr_bypass(self):
        """Env 源码不出现 object.__setattr__(self, 绕过屏蔽。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        self.assertNotIn("object.__setattr__(self,", exec_source)

    def test_no_shielded_attrs_in_env(self):
        """Env 类不定义 _SHIELDED_ATTRS（K10 屏蔽机制已删除）。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        self.assertNotIn("_SHIELDED_ATTRS", exec_source)


class TestPhase2BaselineNewArchitecturePresent(unittest.TestCase):
    """Env 源码含新架构要素。"""

    def test_new_inheritance_chain(self):
        """Env 类继承 OrcaGymEnvMixin, gym.Env（K14）。"""
        self.assertIn(
            "class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env)", _ENV_SOURCE,
        )

    def test_mixin_imported(self):
        """Env 源码 import OrcaGymEnvMixin。"""
        self.assertIn("from ..orca_gym_env_mixin import OrcaGymEnvMixin", _ENV_SOURCE)

    def test_self_orchestrated_lifecycle(self):
        """Env __init__ 自主编排生命周期（含 initialize_grpc/set_time_step/initialize_simulation）。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        self.assertIn("self.initialize_grpc()", exec_source)
        self.assertIn("self.set_time_step(time_step)", exec_source)
        self.assertIn("self.initialize_simulation()", exec_source)
        self.assertIn("self.reset_simulation()", exec_source)
        self.assertIn("self.init_qpos_qvel()", exec_source)

    def test_private_gym_field_used(self):
        """Env 源码使用 self._gym（K1 命名约束, 带下划线）。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        self.assertIn("self._gym", exec_source)

    def test_dir_method_present(self):
        """Env 类定义 __dir__ 方法（K2: 只列公共 API）。"""
        exec_source = _strip_docstrings(_ENV_SOURCE)
        self.assertIn("def __dir__", exec_source)


if __name__ == "__main__":
    unittest.main()
