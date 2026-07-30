"""AGENTS.md API 隔离章节内容校验测试。"""

import unittest
from pathlib import Path


class TestAgentsMd(unittest.TestCase):
    """AGENTS.md 内容约束。"""

    @classmethod
    def setUpClass(cls):
        cls.content = (Path(__file__).resolve().parents[2] / "AGENTS.md").read_text()

    def test_has_api_isolation_section(self):
        """AGENTS.md 包含 API 隔离强制章节。"""
        self.assertTrue(
            "API 隔离强制" in self.content
            or "API Isolation Enforcement" in self.content
        )

    def test_has_ruff_requirement(self):
        """AGENTS.md 要求执行 ruff SLF001。"""
        self.assertIn("ruff check", self.content)
        self.assertIn("SLF001", self.content)

    def test_has_public_api_table(self):
        """AGENTS.md 含"正确 vs 禁止"公共 API 对照表。"""
        has_cn = "正确" in self.content and "禁止" in self.content
        has_en = "Correct" in self.content and "Forbidden" in self.content
        self.assertTrue(has_cn or has_en, "缺少公共 API 对照表")

    def test_no_legacy_getattr_description(self):
        """AGENTS.md 不再描述 __getattr__ 拦截机制。"""
        self.assertNotIn("__getattr__ 拦截", self.content)
        self.assertNotIn("_BLOCKED_ATTRS", self.content)

    def test_mechanism_version_updated(self):
        """机制描述更新为 M0-M7。"""
        self.assertNotIn("M1-M6 六层机制", self.content)


if __name__ == "__main__":
    unittest.main()
