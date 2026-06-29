"""阶段二变更修订 — Step 6: ruff SLF001 与 AGENTS.md 合规(M1/M2)。

对齐文档: docs/design/development/orca_gym_euler_phase2_revision_development.md Step 6
"""

import pathlib
import re
import subprocess
import sys
import unittest


class TestPhase2M1RuffSLF001(unittest.TestCase):
    """M1: ruff SLF001 静态检查。"""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = pathlib.Path(__file__).resolve().parents[4]

    def test_ruff_installed(self):
        """ruff 已安装且可执行。"""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, "ruff 未安装")

    def test_ruff_slf001_euler_clean(self):
        """Euler 代码 ruff SLF001 零报警。"""
        for sub in ["environment/euler", "core/euler"]:
            with self.subTest(path=sub):
                result = subprocess.run(
                    [sys.executable, "-m", "ruff", "check", "--select", "SLF001",
                     str(self.repo_root / "orca_gym" / sub)],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0,
                                 f"ruff SLF001 报警 [{sub}]:\n{result.stdout}")


class TestPhase2M2AgentsMdApiIsolation(unittest.TestCase):
    """M2: AGENTS.md 含 API 隔离强制章节。"""

    @classmethod
    def setUpClass(cls):
        cls.agents_md = (
            pathlib.Path(__file__).resolve().parents[4] / "AGENTS.md"
        ).read_text(encoding="utf-8")

    def test_agents_md_has_api_isolation_rule(self):
        """AGENTS.md 含'API 隔离强制'章节。"""
        self.assertIn("规则 4", self.agents_md)
        self.assertIn("API 隔离强制", self.agents_md)

    def test_agents_md_lists_blocked_attrs(self):
        """AGENTS.md 列出禁止穿墙的内部属性。"""
        self.assertIn("env._gym", self.agents_md)
        self.assertIn("env._stub", self.agents_md)
        self.assertIn("env._channel", self.agents_md)
        self.assertIn("_mjModel", self.agents_md)
        self.assertIn("_mjData", self.agents_md)

    def test_agents_md_has_correct_usage_table(self):
        """AGENTS.md 含正确/禁止 API 使用对照表。"""
        self.assertIn("env.data.qpos", self.agents_md)
        self.assertIn("env.set_joint_qpos", self.agents_md)
        self.assertIn("env.do_simulation", self.agents_md)
        self.assertIn("env.sim_config.timestep", self.agents_md)

    def test_agents_md_has_ruff_command(self):
        """AGENTS.md 含 ruff SLF001 检查命令。"""
        self.assertIn("ruff check --select SLF001", self.agents_md)


class TestPhase2NoqaExemptionDiscipline(unittest.TestCase):
    """ruff SLF001 noqa 豁免规范: 仅 core 层组件编排允许。"""

    @classmethod
    def setUpClass(cls):
        env_file = (
            pathlib.Path(__file__).resolve().parents[4]
            / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
        )
        cls.env_source = env_file.read_text(encoding="utf-8")
        gym_file = (
            pathlib.Path(__file__).resolve().parents[4]
            / "orca_gym" / "core" / "euler" / "orca_gym_euler.py"
        )
        cls.gym_source = gym_file.read_text(encoding="utf-8")

    def test_env_no_noqa_slf001(self):
        """Env 源码不使用 noqa: SLF001 豁免(Env 不应穿墙)。"""
        self.assertNotIn("noqa: SLF001", self.env_source)
        self.assertNotIn("noqa:SLF001", self.env_source)

    def test_gym_noqa_only_for_bind_orchestration(self):
        """Gym 源码 noqa: SLF001 仅用于 _bind 编排(非穿墙访问)。"""
        # 提取 noqa 行
        noqa_lines = re.findall(
            r".*# noqa: ?SLF001.*", self.gym_source
        )
        self.assertGreaterEqual(len(noqa_lines), 1, "Gym 应有 _bind 编排的 noqa 豁免")
        # 每条 noqa 应伴随 _bind 或 sync_to_view 等组件编排模式
        for line in noqa_lines:
            self.assertTrue(
                "_bind(" in line or "object.__getattribute__" in line,
                f"noqa 行非组件编排豁免: {line.strip()}",
            )


if __name__ == "__main__":
    unittest.main()
