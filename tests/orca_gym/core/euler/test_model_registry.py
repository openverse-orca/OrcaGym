"""P2-Step1: ModelRegistry 骨架验收测试。

验证 ModelRegistry 的类结构、构建方法、查询方法签名完整
（架构 §5.5, §12.2），不验证 MuJoCo 功能正确性（骨架阶段
不执行真实模型构建，方法体 raise NotImplementedError）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import unittest

from orca_gym.core.euler.model_registry import ModelRegistry


class TestModelRegistrySkeleton(unittest.TestCase):
    """ModelRegistry 骨架验收测试（对应 P2-Step1 验收标准）。"""

    def test_registry_constructable(self):
        """ModelRegistry() 可无参构造。"""
        registry = ModelRegistry()
        self.assertIsInstance(registry, ModelRegistry)

    def test_registry_has_build_methods(self):
        """build_orca_gym_model / build_orca_gym_data 方法存在。"""
        self.assertTrue(callable(getattr(ModelRegistry, "build_orca_gym_model", None)))
        self.assertTrue(callable(getattr(ModelRegistry, "build_orca_gym_data", None)))

    def test_registry_has_query_methods(self):
        """body_subtree_mass / equality_data_width / equality_object_ids 方法存在。"""
        self.assertTrue(callable(getattr(ModelRegistry, "body_subtree_mass", None)))
        self.assertTrue(callable(getattr(ModelRegistry, "equality_data_width", None)))
        self.assertTrue(callable(getattr(ModelRegistry, "equality_object_ids", None)))


class TestModelRegistryMethodStubs(unittest.TestCase):
    """补充：验证方法在骨架阶段按约定 raise NotImplementedError。

    骨架阶段不执行真实构建/查询（架构 §1.3），方法体应为占位。
    """

    def test_build_methods_raise_not_implemented(self):
        registry = ModelRegistry()
        with self.assertRaises(NotImplementedError):
            registry.build_orca_gym_model()
        with self.assertRaises(NotImplementedError):
            registry.build_orca_gym_data()

    def test_query_methods_raise_not_implemented(self):
        registry = ModelRegistry()
        with self.assertRaises(NotImplementedError):
            registry.body_subtree_mass("dummy_body")
        with self.assertRaises(NotImplementedError):
            registry.equality_data_width()
        with self.assertRaises(NotImplementedError):
            registry.equality_object_ids(0)


class TestModelRegistryDocstring(unittest.TestCase):
    """补充：验证 docstring 契约（K12）。"""

    def test_registry_docstring_has_contract(self):
        """docstring 含「使用契约」和「禁止」关键词（K12）。"""
        doc = ModelRegistry.__doc__ or ""
        self.assertIn("使用契约", doc)
        self.assertIn("禁止", doc)


if __name__ == "__main__":
    unittest.main()
