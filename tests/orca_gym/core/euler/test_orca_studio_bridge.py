"""P2-Step3: OrcaStudioBridge 骨架验收测试。

验证 OrcaStudioBridge 的类结构、7 个骨架方法签名完整，以及依赖反转
设计（不持有 _mjData/_mjModel）（架构 §5.4, §12.2），不验证 gRPC
功能正确性（骨架阶段不执行真实通信，方法体 raise NotImplementedError）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import unittest

from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge


class TestOrcaStudioBridgeSkeleton(unittest.TestCase):
    """OrcaStudioBridge 骨架验收测试（对应 P2-Step3 验收标准）。"""

    def test_bridge_constructable(self):
        """OrcaStudioBridge() 可无参构造（stub=None）。"""
        bridge = OrcaStudioBridge()
        self.assertIsInstance(bridge, OrcaStudioBridge)

    def test_bridge_has_render(self):
        """render 方法存在。"""
        self.assertTrue(callable(getattr(OrcaStudioBridge, "render", None)))

    def test_bridge_has_load_model_xml(self):
        """load_model_xml 方法存在。"""
        self.assertTrue(callable(getattr(OrcaStudioBridge, "load_model_xml", None)))

    def test_bridge_has_pause_simulation(self):
        """pause_simulation 方法存在。"""
        self.assertTrue(callable(getattr(OrcaStudioBridge, "pause_simulation", None)))

    def test_bridge_has_configure_offline(self):
        """configure_offline 方法存在。"""
        self.assertTrue(callable(getattr(OrcaStudioBridge, "configure_offline", None)))

    def test_bridge_has_set_timestep_remote(self):
        """set_timestep_remote 方法存在。"""
        self.assertTrue(callable(getattr(OrcaStudioBridge, "set_timestep_remote", None)))

    def test_bridge_has_body_manipulation_methods(self):
        """get_body_manipulation_anchored / get_body_manipulation_movement 方法存在。"""
        self.assertTrue(callable(getattr(OrcaStudioBridge, "get_body_manipulation_anchored", None)))
        self.assertTrue(callable(getattr(OrcaStudioBridge, "get_body_manipulation_movement", None)))

    def test_bridge_no_mjdata_attribute(self):
        """实例 __dict__ 不含 _mjData/_mjModel/mjData/mjModel（依赖反转）。

        验证架构 §5.4 依赖反转设计：Bridge 不持有 MuJoCo 内部数据结构。
        """
        bridge = OrcaStudioBridge()
        forbidden_attrs = ["_mjData", "_mjModel", "mjData", "mjModel"]
        for attr in forbidden_attrs:
            with self.subTest(attr=attr):
                self.assertNotIn(attr, bridge.__dict__,
                                 f"Bridge 不应持有 {attr}（依赖反转设计）")

    def test_bridge_docstring_mentions_decoupling(self):
        """docstring 含「依赖反转」或「解耦」关键词。"""
        doc = OrcaStudioBridge.__doc__ or ""
        self.assertTrue("依赖反转" in doc or "解耦" in doc,
                        "docstring 应说明依赖反转/解耦设计")


class TestOrcaStudioBridgeMethodStubs(unittest.TestCase):
    """补充：验证方法在骨架阶段按约定 raise NotImplementedError。

    骨架阶段不执行真实 gRPC 通信（架构 §1.3），方法体应为占位。
    """

    def test_sync_methods_raise_not_implemented(self):
        """同步方法 configure_offline / set_timestep_remote raise NotImplementedError。"""
        bridge = OrcaStudioBridge()
        with self.assertRaises(NotImplementedError):
            bridge.configure_offline("dummy.xml")
        with self.assertRaises(NotImplementedError):
            bridge.set_timestep_remote(0.002)


if __name__ == "__main__":
    unittest.main()
