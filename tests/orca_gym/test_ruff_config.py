"""ruff SLF001 配置与可执行性测试。"""

import re
import subprocess
import sys
import unittest
from pathlib import Path


class TestRuffConfig(unittest.TestCase):
    """ruff 配置与可执行性。"""

    @classmethod
    def setUpClass(cls):
        cls.pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        cls.content = cls.pyproject.read_text()

    def test_ruff_installed(self):
        """ruff 已安装且可执行。"""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, "ruff 未安装")
        self.assertIn("ruff", result.stdout.lower())

    def test_ruff_config_has_slf001(self):
        """配置文件已配置 SLF001。"""
        self.assertIn("[tool.ruff.lint]", self.content)
        self.assertIn("SLF001", self.content)

    def test_ruff_tests_ignored(self):
        """测试目录已配置 SLF001 忽略。"""
        self.assertIn("tests/**", self.content)

    def test_ruff_init_ignored(self):
        """__init__.py 已配置忽略。"""
        self.assertIn("__init__.py", self.content)

    def test_ruff_exclude_section_exists(self):
        """配置文件含 ruff exclude 配置（第三方 fork 排除）。"""
        self.assertIn("exclude", self.content)


class TestNoqaExemptionDiscipline(unittest.TestCase):
    """ruff SLF001 noqa 豁免规范：仅 core 层组件编排允许。"""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.env_source = (
            repo_root / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
        ).read_text(encoding="utf-8")
        cls.gym_source = (
            repo_root / "orca_gym" / "core" / "euler" / "orca_gym_euler.py"
        ).read_text(encoding="utf-8")

    def test_env_no_noqa_slf001(self):
        """Env 源码不使用 noqa: SLF001 豁免（Env 不应穿墙）。"""
        self.assertNotIn("noqa: SLF001", self.env_source)
        self.assertNotIn("noqa:SLF001", self.env_source)

    def test_gym_noqa_only_for_bind_orchestration(self):
        """Gym 源码 noqa: SLF001 仅用于 _bind 编排（非穿墙访问）。"""
        noqa_lines = re.findall(r".*# noqa: ?SLF001.*", self.gym_source)
        self.assertGreaterEqual(len(noqa_lines), 1, "Gym 应有 _bind 编排的 noqa 豁免")
        for line in noqa_lines:
            self.assertTrue(
                "_bind(" in line or "object.__getattribute__" in line,
                f"noqa 行非组件编排豁免: {line.strip()}",
            )


if __name__ == "__main__":
    unittest.main()
