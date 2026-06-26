"""ruff SLF001 配置与可执行性测试。"""

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


if __name__ == "__main__":
    unittest.main()
