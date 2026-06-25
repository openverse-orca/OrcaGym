"""P1-Step2: OrcaGymDataView 骨架验收测试。

验证 OrcaGymDataView 的字段定义、方法签名、__getattr__ 兜底机制完整
（架构 §5.7, §7.4, §12.2），不验证 MuJoCo 功能正确性（骨架阶段
不持有真实数据，查询方法 raise NotImplementedError）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import unittest

import numpy as np

from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


class TestOrcaGymDataViewSkeleton(unittest.TestCase):
    """OrcaGymDataView 骨架验收测试（对应 P1-Step2 验收标准）。"""

    def test_data_view_constructable(self):
        """OrcaGymDataView() 可无参构造。"""
        view = OrcaGymDataView()
        self.assertIsInstance(view, OrcaGymDataView)

    def test_data_view_has_basic_fields(self):
        """实例有 qpos/qvel/qacc/qfrc_bias/time 五个基本字段。"""
        view = OrcaGymDataView()
        self.assertTrue(hasattr(view, "qpos"))
        self.assertTrue(hasattr(view, "qvel"))
        self.assertTrue(hasattr(view, "qacc"))
        self.assertTrue(hasattr(view, "qfrc_bias"))
        self.assertTrue(hasattr(view, "time"))

    def test_data_view_has_extended_fields(self):
        """实例有 xfrc_applied/actuator_force/contact 扩展字段。"""
        view = OrcaGymDataView()
        self.assertTrue(hasattr(view, "xfrc_applied"))
        self.assertTrue(hasattr(view, "actuator_force"))
        self.assertTrue(hasattr(view, "contact"))

    def test_data_view_has_body_query_methods(self):
        """body_xpos/body_xquat/body_xmat/body_cvel/body_subtree_mass 方法存在。"""
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_xpos", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_xquat", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_xmat", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_cvel", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "body_subtree_mass", None)))

    def test_data_view_has_site_query_methods(self):
        """site_xpos/site_xmat 方法存在。"""
        self.assertTrue(callable(getattr(OrcaGymDataView, "site_xpos", None)))
        self.assertTrue(callable(getattr(OrcaGymDataView, "site_xmat", None)))

    def test_data_view_getattr_guidance(self):
        """访问不存在的字段（如 cvel）抛 AttributeError，消息含引导文本。

        验证 M3 兜底机制（架构 §7.4）：缺字段时引导扩展而非绕道。
        引导文本应列出可用字段/方法。
        """
        view = OrcaGymDataView()
        with self.assertRaises(AttributeError) as ctx:
            _ = view.cvel  # cvel 不是 DataView 的字段（需通过 body_cvel(name) 查询）

        msg = str(ctx.exception)
        # 引导文本应包含：被访问的字段名、可用字段/方法提示、扩展引导
        self.assertIn("cvel", msg)
        self.assertIn("可用字段", msg)
        self.assertIn("可用方法", msg)
        self.assertIn("OrcaGymDataView", msg)

    def test_data_view_docstring_has_contract(self):
        """docstring 含「使用契约」和「禁止」关键词（K12）。"""
        doc = OrcaGymDataView.__doc__ or ""
        self.assertIn("使用契约", doc)
        self.assertIn("禁止", doc)


class TestOrcaGymDataViewFieldTypes(unittest.TestCase):
    """补充：验证基本字段和扩展字段的类型符合架构 §5.7 定义。

    骨架阶段字段为空数组/默认值，但类型应与文档一致。
    """

    def test_basic_fields_types(self):
        view = OrcaGymDataView()
        self.assertIsInstance(view.qpos, np.ndarray)
        self.assertIsInstance(view.qvel, np.ndarray)
        self.assertIsInstance(view.qacc, np.ndarray)
        self.assertIsInstance(view.qfrc_bias, np.ndarray)
        self.assertIsInstance(view.time, float)

    def test_extended_fields_types(self):
        view = OrcaGymDataView()
        self.assertIsInstance(view.xfrc_applied, np.ndarray)
        self.assertIsInstance(view.actuator_force, np.ndarray)
        self.assertIsInstance(view.contact, list)

    def test_basic_fields_not_none(self):
        """基本字段不应为 None（初始化为空数组/默认值，便于用户直接读取）。"""
        view = OrcaGymDataView()
        self.assertIsNotNone(view.qpos)
        self.assertIsNotNone(view.qvel)
        self.assertIsNotNone(view.qacc)
        self.assertIsNotNone(view.qfrc_bias)
        self.assertIsNotNone(view.time)

    def test_extended_fields_not_none(self):
        """扩展字段不应为 None。"""
        view = OrcaGymDataView()
        self.assertIsNotNone(view.xfrc_applied)
        self.assertIsNotNone(view.actuator_force)
        self.assertIsNotNone(view.contact)


class TestOrcaGymDataViewMethodStubs(unittest.TestCase):
    """补充：验证查询方法在骨架阶段按约定 raise NotImplementedError。

    骨架阶段不实现真实查询逻辑（架构 §1.3），方法体应为占位。
    """

    def test_body_query_methods_raise_not_implemented(self):
        view = OrcaGymDataView()
        for method_name in [
            "body_xpos", "body_xquat", "body_xmat",
            "body_cvel", "body_subtree_mass",
        ]:
            with self.subTest(method=method_name):
                with self.assertRaises(NotImplementedError):
                    getattr(view, method_name)("dummy_body")

    def test_site_query_methods_raise_not_implemented(self):
        view = OrcaGymDataView()
        for method_name in ["site_xpos", "site_xmat"]:
            with self.subTest(method=method_name):
                with self.assertRaises(NotImplementedError):
                    getattr(view, method_name)("dummy_site")


if __name__ == "__main__":
    unittest.main()
