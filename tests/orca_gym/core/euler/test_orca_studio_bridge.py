"""OrcaStudioBridge 验收测试（阶段二 Step 1）。

验证 OrcaStudioBridge 的类结构、方法签名完整，依赖反转设计
（不持有 _mjData/_mjModel）（架构 §5.4, §12.2），以及离线模式
所有方法 no-op 不抛异常（阶段二 Step 1 验收标准）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import unittest

import numpy as np

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


class TestOrcaStudioBridgeOfflineNoOp(unittest.TestCase):
    """阶段二-Step 1: 离线模式（stub=None）所有方法 no-op 不抛异常。"""

    def test_sync_methods_offline_noop(self):
        """同步方法 configure_offline / set_timestep_remote 离线模式 no-op。"""
        bridge = OrcaStudioBridge()
        # 不应抛异常
        bridge.configure_offline("dummy.xml")
        bridge.set_timestep_remote(0.002)

    def test_configure_offline_stores_local_path(self):
        """configure_offline 正确存储本地路径（验收标准）。"""
        bridge = OrcaStudioBridge()
        bridge.configure_offline("/tmp/test_model.xml")
        self.assertIsNotNone(bridge._local_xml_path)
        self.assertTrue(bridge._local_xml_path.endswith("test_model.xml"))
        # 默认 assets_dir 为 XML 所在目录
        self.assertEqual(bridge._xml_assets_dir, "/tmp")

    def test_configure_offline_custom_assets_dir(self):
        """configure_offline 支持自定义 assets_dir。"""
        bridge = OrcaStudioBridge()
        bridge.configure_offline("/tmp/test_model.xml", assets_dir="/tmp/assets")
        self.assertEqual(bridge._xml_assets_dir, "/tmp/assets")

    def test_render_signature_is_qpos_sim_time(self):
        """render 签名为 (qpos, sim_time)（依赖反转，验收标准）。"""
        import inspect
        sig = inspect.signature(OrcaStudioBridge.render)
        params = list(sig.parameters.keys())
        # ['self', 'qpos', 'sim_time']
        self.assertIn("qpos", params)
        self.assertIn("sim_time", params)

    def test_get_override_ctrls_returns_dict(self):
        """get_override_ctrls 返回 dict[int, float]（验收标准）。"""
        bridge = OrcaStudioBridge()
        result = bridge.get_override_ctrls()
        self.assertIsInstance(result, dict)
        # 离线模式空 dict
        self.assertEqual(len(result), 0)

    def test_async_methods_offline_noop(self):
        """异步方法离线模式 no-op 不抛异常。"""
        import asyncio
        bridge = OrcaStudioBridge()
        # render 离线 no-op
        asyncio.run(bridge.render(np.zeros(1), 0.0))
        # pause_simulation 离线 no-op
        asyncio.run(bridge.pause_simulation())
        # load_model_xml 离线模式未配置应抛 RuntimeError（非 NotImplementedError）
        bridge2 = OrcaStudioBridge()
        with self.assertRaises(RuntimeError):
            asyncio.run(bridge2.load_model_xml())
        # 配置后返回本地路径
        bridge2.configure_offline("/tmp/dummy_nonexistent.xml")
        with self.assertRaises(FileNotFoundError):
            asyncio.run(bridge2.load_model_xml())

    def test_get_body_manipulation_offline_defaults(self):
        """体操作方法离线模式返回默认值。"""
        import asyncio
        bridge = OrcaStudioBridge()
        body_name, anchor_type = asyncio.run(bridge.get_body_manipulation_anchored())
        self.assertIsNone(body_name)
        self.assertEqual(anchor_type, 0)  # AnchorType.NONE
        movement = asyncio.run(bridge.get_body_manipulation_movement())
        self.assertIn("delta_pos", movement)
        self.assertIn("delta_quat", movement)
        np.testing.assert_array_almost_equal(movement["delta_pos"], np.zeros(3))
        np.testing.assert_array_almost_equal(
            movement["delta_quat"], np.array([1.0, 0.0, 0.0, 0.0])
        )

    def test_render_updates_override_ctrls_cache(self):
        """render 在线模式更新 override_ctrls 缓存（用 mock stub 验证）。"""
        import asyncio
        # 构造 mock stub 模拟在线模式
        class MockResponse:
            def __init__(self, ctrls):
                self.override_ctrls = ctrls
        class MockCtrl:
            def __init__(self, index, value):
                self.index = index
                self.value = value
        class MockStub:
            async def UpdateLocalEnv(self, request):
                return MockResponse([MockCtrl(0, 0.5), MockCtrl(1, -0.3)])
        bridge = OrcaStudioBridge(stub=MockStub())
        asyncio.run(bridge.render(np.array([0.1]), 1.0))
        ctrls = bridge.get_override_ctrls()
        self.assertEqual(ctrls, {0: 0.5, 1: -0.3})


# =============================================================================
# 阶段三 3.2.3：OrcaStudioBridge mocap 远端同步
# =============================================================================


class TestBridgeMocapArchCompliance(unittest.TestCase):
    """子步骤 3.2.3 架构遵从性测试（K9/P2）。

    对应文档 §6.4 架构遵从性测试表。
    """

    def test_bridge_mocap_offline_noop(self):
        """K9: 离线模式（_stub is None）不抛错，直接 return。"""
        import asyncio
        bridge = OrcaStudioBridge()  # stub=None
        # send_remote=True 但 stub=None，应 no-op 不抛错
        asyncio.run(
            bridge.set_mocap_pos_and_quat(
                {"anchor": {"pos": np.zeros(3), "quat": np.array([1, 0, 0, 0])}},
                send_remote=True,
            )
        )

    def test_bridge_mocap_no_mjdata_dependency(self):
        """K9/P2: grep 断言 Bridge 不 import MjData/MjModel，仅操作 gRPC stub。"""
        import inspect
        import re as _re
        source = inspect.getsource(OrcaStudioBridge)
        # set_mocap_pos_and_quat 区块
        start = source.find("async def set_mocap_pos_and_quat")
        self.assertGreater(start, 0)
        block = source[start:]
        # 去除 docstring（其中可能引用 _mjData 作为禁止说明）
        block = _re.sub(r'"""[\s\S]*?"""', '', block, count=1)
        # 不应访问 _mjData/_mjModel（代码体）
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)
        # 应通过 self._stub.SetMocapPosAndQuat 走 gRPC
        self.assertIn("self._stub.SetMocapPosAndQuat", block)

    def test_bridge_mocap_async_signature(self):
        """K9: 方法为 async def，返回 None。"""
        import asyncio
        import inspect
        # async def
        self.assertTrue(inspect.iscoroutinefunction(OrcaStudioBridge.set_mocap_pos_and_quat))
        # 返回 None
        bridge = OrcaStudioBridge()
        ret = asyncio.run(
            bridge.set_mocap_pos_and_quat({}, send_remote=True)
        )
        self.assertIsNone(ret)


class TestBridgeMocapFunctional(unittest.TestCase):
    """子步骤 3.2.3 功能单元测试。

    对应文档 §6.4 功能单元测试表。
    """

    def test_set_mocap_remote_offline_returns_none(self):
        """离线模式返回 None 不抛错。"""
        import asyncio
        bridge = OrcaStudioBridge()
        ret = asyncio.run(
            bridge.set_mocap_pos_and_quat(
                {"anchor": {"pos": np.array([1, 2, 3]), "quat": np.array([1, 0, 0, 0])}},
                send_remote=True,
            )
        )
        self.assertIsNone(ret)

    def test_set_mocap_remote_online_calls_stub(self):
        """在线模式（mock stub）调用 SetMocapPosAndQuat。"""
        import asyncio

        captured = {}

        class MockStub:
            async def SetMocapPosAndQuat(self, request):
                captured["request"] = request
                # 返回一个简单 response（success 字段）
                from orca_gym.protos import mjc_message_pb2
                return mjc_message_pb2.SetMocapPosAndQuatResponse(success=[True])

        bridge = OrcaStudioBridge(stub=MockStub())
        asyncio.run(
            bridge.set_mocap_pos_and_quat(
                {
                    "anchor1": {
                        "pos": np.array([0.5, 0.3, 0.8]),
                        "quat": np.array([0.7071, 0.0, 0.7071, 0.0]),
                    }
                },
                send_remote=True,
            )
        )
        # 断言 stub 被调用
        self.assertIn("request", captured)
        req = captured["request"]
        self.assertEqual(len(req.mocap_body_info), 1)
        info = req.mocap_body_info[0]
        self.assertEqual(info.mocap_body_name, "anchor1")
        self.assertEqual(list(info.pos), [0.5, 0.3, 0.8])
        self.assertEqual(list(info.quat), [0.7071, 0.0, 0.7071, 0.0])

    def test_set_mocap_send_remote_false_noop(self):
        """send_remote=False 时即使有 stub 也不调用。"""
        import asyncio

        called = {"count": 0}

        class MockStub:
            async def SetMocapPosAndQuat(self, request):
                called["count"] += 1
                from orca_gym.protos import mjc_message_pb2
                return mjc_message_pb2.SetMocapPosAndQuatResponse(success=[True])

        bridge = OrcaStudioBridge(stub=MockStub())
        asyncio.run(
            bridge.set_mocap_pos_and_quat(
                {"anchor": {"pos": np.zeros(3), "quat": np.array([1, 0, 0, 0])}},
                send_remote=False,
            )
        )
        self.assertEqual(called["count"], 0)


if __name__ == "__main__":
    unittest.main()
