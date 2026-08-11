"""OrcaStudioBridge 验收测试（阶段二 Step 1）。

验证 OrcaStudioBridge 的类结构、方法签名完整，依赖反转设计
（不持有 _mjData/_mjModel）（架构 §5.4, §12.2），以及离线模式
所有方法 no-op 不抛异常（阶段二 Step 1 验收标准）。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler
"""

import os
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


# =============================================================================
# 阶段三 3.4.1 / 3.4.2：OrcaStudioBridge 视频录制 / 帧捕获方法（已废弃）
# 引擎侧 BeginSaveMp4File / StopSaveMp4File / GetCurrentFrameIndex / GetTimeStamp
# 四个 RPC 已从 proto 中删除。bridge 层降级为 no-op + DeprecationWarning。
# 录制能力已迁移到客户端 PyAV remux（orca_gym/recorder/）。
# =============================================================================


class TestBridgeVideoArchCompliance(unittest.TestCase):
    """子步骤 3.4.1 架构遵从性测试（K9 走 bridge + 离线 no-op）。

    对应文档 §8.2 架构遵从性测试表。

    .. note::
        引擎侧 MP4 录制 RPC 已删除，bridge 层降级为 no-op + DeprecationWarning。
        在线/离线模式行为一致：不调用 stub，直接返回 None 并发出警告。
    """

    def test_bridge_video_offline_noop(self):
        """K9: 离线模式（_stub is None）不抛错，no-op + DeprecationWarning。"""
        import asyncio
        import warnings

        bridge = OrcaStudioBridge()
        # 离线模式调用不抛错（但发出 DeprecationWarning）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.run(bridge.begin_save_video("/tmp/test.mp4", capture_mode=0))
            asyncio.run(bridge.stop_save_video())

    def test_bridge_video_no_mjdata_dependency(self):
        """K9/P2: grep 断言视频方法不触 MjData/MjModel/_mjData/_mjModel。"""
        import inspect

        source = inspect.getsource(OrcaStudioBridge)
        start = source.find("# --- 视频录制 / 帧捕获（已废弃）---")
        self.assertGreater(start, 0, "未找到视频录制/帧捕获（已废弃）区块")
        block = source[start:]
        # 视频方法块内不应出现 MjData/MjModel 的访问
        self.assertNotIn("MjData", block)
        self.assertNotIn("MjModel", block)
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)

    def test_bridge_video_async_signature(self):
        """K9: 方法为 async def，返回 None。"""
        import asyncio
        import inspect
        import warnings

        self.assertTrue(inspect.iscoroutinefunction(OrcaStudioBridge.begin_save_video))
        self.assertTrue(inspect.iscoroutinefunction(OrcaStudioBridge.stop_save_video))
        # 离线模式返回 None
        bridge = OrcaStudioBridge()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ret = asyncio.run(bridge.begin_save_video("/tmp/test.mp4", capture_mode=0))
            self.assertIsNone(ret)
            ret = asyncio.run(bridge.stop_save_video())
            self.assertIsNone(ret)

    def test_bridge_video_emits_deprecation_warning(self):
        """新增：所有视频/帧捕获方法应发出 DeprecationWarning。"""
        import asyncio
        import warnings

        bridge = OrcaStudioBridge()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            asyncio.run(bridge.begin_save_video("/tmp/test.mp4", capture_mode=0))
            asyncio.run(bridge.stop_save_video())
            asyncio.run(bridge.get_current_frame())
            asyncio.run(bridge.get_camera_time_stamp(0))
            self.assertEqual(len(w), 4)
            for warning in w:
                self.assertIs(warning.category, DeprecationWarning)


class TestBridgeVideoFunctional(unittest.TestCase):
    """子步骤 3.4.1 功能单元测试。

    对应文档 §8.2 功能单元测试表。

    .. note::
        引擎侧 MP4 录制 RPC 已删除，在线/离线模式均为 no-op + DeprecationWarning。
        原 online_calls_stub 测试已废弃（不再调用 stub）。
    """

    def test_begin_save_video_offline_noop(self):
        """离线模式返回 None 不抛错。"""
        import asyncio
        import warnings

        bridge = OrcaStudioBridge()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ret = asyncio.run(
                bridge.begin_save_video("/tmp/test.mp4", capture_mode=0)
            )
            self.assertIsNone(ret)

    def test_stop_save_video_offline_noop(self):
        """离线模式返回 None 不抛错。"""
        import asyncio
        import warnings

        bridge = OrcaStudioBridge()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ret = asyncio.run(bridge.stop_save_video())
            self.assertIsNone(ret)

    def test_begin_save_video_online_noop(self):
        """在线模式（mock stub）不再调用 stub，no-op + DeprecationWarning。

        引擎侧 BeginSaveMp4File RPC 已删除，bridge 层降级为 no-op。
        """
        import asyncio
        import warnings

        called = {"count": 0}

        class MockStub:
            async def BeginSaveMp4File(self, request):
                called["count"] += 1

        bridge = OrcaStudioBridge(stub=MockStub())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.run(
                bridge.begin_save_video("/tmp/test.mp4", capture_mode=1)
            )
        # stub 不应被调用
        self.assertEqual(called["count"], 0)

    def test_stop_save_video_online_noop(self):
        """在线模式不再调用 stub，no-op + DeprecationWarning。

        引擎侧 StopSaveMp4File RPC 已删除，bridge 层降级为 no-op。
        """
        import asyncio
        import warnings

        called = {"count": 0}

        class MockStub:
            async def StopSaveMp4File(self, request):
                called["count"] += 1

        bridge = OrcaStudioBridge(stub=MockStub())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.run(bridge.stop_save_video())
        self.assertEqual(called["count"], 0)


# =============================================================================
# 阶段三 3.4.2：OrcaStudioBridge 帧捕获方法（已废弃）
# =============================================================================


class TestBridgeFrameArchCompliance(unittest.TestCase):
    """子步骤 3.4.2 架构遵从性测试（K9 走 bridge + K11 typed 返回）。

    对应文档 §8.3 架构遵从性测试表。

    .. note::
        引擎侧帧捕获 RPC 已删除，bridge 层降级为 no-op + DeprecationWarning。
        返回值仍保持 typed（int / dict）。
    """

    def test_bridge_frame_offline_returns_default(self):
        """K9: 离线模式返回默认值（-1/空 dict）不抛错。"""
        import asyncio
        import warnings

        bridge = OrcaStudioBridge()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(asyncio.run(bridge.get_current_frame()), -1)
            self.assertEqual(asyncio.run(bridge.get_camera_time_stamp(0)), {})
            # get_frame_png 离线 no-op（未废弃）
            asyncio.run(bridge.get_frame_png("/tmp/test.png"))

    def test_bridge_frame_no_mjdata_dependency(self):
        """K9/P2: grep 断言帧方法不触 MjData/MjModel/_mjData/_mjModel。"""
        import inspect

        source = inspect.getsource(OrcaStudioBridge)
        start = source.find("# --- 视频录制 / 帧捕获（已废弃）---")
        self.assertGreater(start, 0, "未找到视频录制/帧捕获（已废弃）区块")
        block = source[start:]
        self.assertNotIn("MjData", block)
        self.assertNotIn("MjModel", block)
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)

    def test_bridge_frame_returns_typed(self):
        """K11: get_current_frame 返回 int，get_camera_time_stamp 返回 dict。"""
        import asyncio
        import warnings

        bridge = OrcaStudioBridge()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ret = asyncio.run(bridge.get_current_frame())
            self.assertIsInstance(ret, int)
            ret = asyncio.run(bridge.get_camera_time_stamp(0))
            self.assertIsInstance(ret, dict)


class TestBridgeFrameFunctional(unittest.TestCase):
    """子步骤 3.4.2 功能单元测试。

    对应文档 §8.3 功能单元测试表。

    .. note::
        引擎侧帧捕获 RPC 已删除，在线/离线模式均为 no-op + DeprecationWarning。
        原 online_calls_stub 测试已废弃（不再调用 stub）。
    """

    def test_get_current_frame_offline_returns_neg1(self):
        """离线模式返回 -1。"""
        import asyncio
        import warnings

        bridge = OrcaStudioBridge()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(asyncio.run(bridge.get_current_frame()), -1)

    def test_get_camera_time_stamp_offline_returns_empty(self):
        """离线模式返回空 dict。"""
        import asyncio
        import warnings

        bridge = OrcaStudioBridge()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(asyncio.run(bridge.get_camera_time_stamp(0)), {})

    def test_get_frame_png_offline_noop(self):
        """离线模式 no-op（get_frame_png 未废弃）。"""
        import asyncio

        bridge = OrcaStudioBridge()
        ret = asyncio.run(bridge.get_frame_png("/tmp/test.png"))
        self.assertIsNone(ret)

    def test_get_current_frame_online_noop(self):
        """在线模式不再调用 stub，no-op + DeprecationWarning，返回 -1。

        引擎侧 GetCurrentFrameIndex RPC 已删除，bridge 层降级为 no-op。
        """
        import asyncio
        import warnings

        called = {"count": 0}

        class MockStub:
            async def GetCurrentFrameIndex(self, request):
                called["count"] += 1
                from orca_gym.protos import mjc_message_pb2
                resp = mjc_message_pb2.GetCurrentFrameIndexResponse()
                resp.current_frame = 42
                return resp

        bridge = OrcaStudioBridge(stub=MockStub())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ret = asyncio.run(bridge.get_current_frame())
        # stub 不应被调用，返回 -1
        self.assertEqual(called["count"], 0)
        self.assertEqual(ret, -1)


# =============================================================================
# 阶段三 3.4.3：OrcaStudioBridge 内容文件方法
# =============================================================================


class TestBridgeContentFileArchCompliance(unittest.TestCase):
    """子步骤 3.4.3 架构遵从性测试（K9 走 bridge + 离线 no-op）。

    对应文档 §8.4 架构遵从性测试表。
    """

    def test_bridge_content_file_offline_noop(self):
        """K9: 离线模式 no-op 不抛错。"""
        import asyncio

        bridge = OrcaStudioBridge()
        asyncio.run(bridge.load_content_file("mesh.obj"))

    def test_bridge_content_file_no_mjdata_dependency(self):
        """K9/P2: grep 断言内容文件方法不触 MjData/MjModel/_mjData/_mjModel。"""
        import inspect

        source = inspect.getsource(OrcaStudioBridge)
        start = source.find("# --- 内容文件（阶段三 3.4.3）---")
        self.assertGreater(start, 0, "未找到 3.4.3 内容文件区块")
        block = source[start:]
        self.assertNotIn("MjData", block)
        self.assertNotIn("MjModel", block)
        self.assertNotIn("_mjData", block)
        self.assertNotIn("_mjModel", block)

    def test_bridge_content_file_async_signature(self):
        """K9: 方法为 async def。"""
        import inspect

        self.assertTrue(
            inspect.iscoroutinefunction(OrcaStudioBridge.load_content_file)
        )


class TestBridgeContentFileFunctional(unittest.TestCase):
    """子步骤 3.4.3 功能单元测试。

    对应文档 §8.4 功能单元测试表。
    """

    def test_load_content_file_offline_noop(self):
        """离线模式 no-op。"""
        import asyncio

        bridge = OrcaStudioBridge()
        ret = asyncio.run(bridge.load_content_file("mesh.obj"))
        self.assertIsNone(ret)

    def test_load_content_file_online_calls_stub(self):
        """在线模式委托 stub。"""
        import asyncio

        captured = {}

        class MockStub:
            async def LoadContentFile(self, request):
                captured["request"] = request
                from orca_gym.protos import mjc_message_pb2
                resp = mjc_message_pb2.LoadContentFileResponse()
                resp.status = mjc_message_pb2.LoadContentFileResponse.SUCCESS
                return resp

        bridge = OrcaStudioBridge(stub=MockStub())
        asyncio.run(
            bridge.load_content_file("mesh.obj", remote_file_dir="/remote")
        )
        self.assertIn("request", captured)
        req = captured["request"]
        self.assertEqual(req.file_name, "mesh.obj")
        self.assertEqual(req.file_dir, "/remote")


# =============================================================================
# 在线场景 mesh 资源自动下载（process_xml_file / process_xml_node / _download_asset_to_cache）
# =============================================================================


class TestBridgeMeshDownloadProcessNode(unittest.TestCase):
    """process_xml_node 递归解析与缺失检测测试。"""

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp(prefix="orcagym_test_mesh_")
        self.bridge = OrcaStudioBridge()
        # 通过 configure_offline 设置 xml_file_dir 为临时目录（仅用于定位资源目录，
        # process_xml_node 本身不依赖 stub）
        self.bridge.configure_offline(
            os.path.join(self.tmp_dir, "dummy.xml"), assets_dir=self.tmp_dir
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_process_xml_node_downloads_missing_mesh(self):
        """mesh 节点 file 不存在时调用 _download_asset_to_cache。"""
        import asyncio
        import xml.etree.ElementTree as ET
        called = {"args": []}

        async def fake_download(name):
            called["args"].append(name)

        self.bridge._download_asset_to_cache = fake_download  # noqa: SLF001 测试白盒
        node = ET.Element("mesh", {"file": "foot.stl"})
        asyncio.run(self.bridge.process_xml_node(node))
        self.assertEqual(called["args"], ["foot.stl"])

    def test_process_xml_node_skips_existing_mesh(self):
        """mesh 文件已存在时不调用 _download_asset_to_cache。"""
        import asyncio
        import xml.etree.ElementTree as ET
        called = {"count": 0}

        async def fake_download(name):
            called["count"] += 1

        # 预创建文件
        with open(os.path.join(self.tmp_dir, "exist.stl"), "wb") as f:
            f.write(b"data")
        self.bridge._download_asset_to_cache = fake_download  # noqa: SLF001 测试白盒
        node = ET.Element("mesh", {"file": "exist.stl"})
        asyncio.run(self.bridge.process_xml_node(node))
        self.assertEqual(called["count"], 0)

    def test_process_xml_node_recurses_children(self):
        """非 mesh/hfield 节点递归处理子节点。"""
        import asyncio
        import xml.etree.ElementTree as ET
        called = {"args": []}

        async def fake_download(name):
            called["args"].append(name)

        self.bridge._download_asset_to_cache = fake_download  # noqa: SLF001 测试白盒
        # asset 节点下嵌套两个 mesh
        root = ET.Element("asset")
        ET.SubElement(root, "mesh", {"file": "a.stl"})
        ET.SubElement(root, "mesh", {"file": "b.stl"})
        asyncio.run(self.bridge.process_xml_node(root))
        self.assertEqual(called["args"], ["a.stl", "b.stl"])

    def test_process_xml_node_ignores_node_without_file_attr(self):
        """mesh 节点无 file 属性时不调用下载。"""
        import asyncio
        import xml.etree.ElementTree as ET
        called = {"count": 0}

        async def fake_download(name):
            called["count"] += 1

        self.bridge._download_asset_to_cache = fake_download  # noqa: SLF001 测试白盒
        node = ET.Element("mesh")
        asyncio.run(self.bridge.process_xml_node(node))
        self.assertEqual(called["count"], 0)

    def test_process_xml_node_handles_hfield(self):
        """hfield 节点同样触发下载。"""
        import asyncio
        import xml.etree.ElementTree as ET
        called = {"args": []}

        async def fake_download(name):
            called["args"].append(name)

        self.bridge._download_asset_to_cache = fake_download  # noqa: SLF001 测试白盒
        node = ET.Element("hfield", {"file": "terrain.png"})
        asyncio.run(self.bridge.process_xml_node(node))
        self.assertEqual(called["args"], ["terrain.png"])


class TestBridgeMeshDownloadProcessFile(unittest.TestCase):
    """process_xml_file 入口测试。"""

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp(prefix="orcagym_test_xml_")
        self.bridge = OrcaStudioBridge()
        self.bridge.configure_offline(
            os.path.join(self.tmp_dir, "dummy.xml"), assets_dir=self.tmp_dir
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_process_xml_file_parses_and_dispatches(self):
        """process_xml_file 解析 XML 文件并调用 process_xml_node。"""
        import asyncio
        called = {"files": []}

        async def fake_process_node(node):
            # 收集所有 mesh/hfield 的 file 属性
            if node.tag in ("mesh", "hfield"):
                f = node.get("file")
                if f:
                    called["files"].append(f)
            else:
                for child in node:
                    await fake_process_node(child)

        self.bridge.process_xml_node = fake_process_node
        xml_path = os.path.join(self.tmp_dir, "scene.xml")
        with open(xml_path, "w") as f:
            f.write(
                '<mujoco><asset>'
                '<mesh file="foot.stl"/>'
                '<hfield file="terrain.png"/>'
                '</asset></mujoco>'
            )
        asyncio.run(self.bridge.process_xml_file(xml_path))
        self.assertEqual(called["files"], ["foot.stl", "terrain.png"])


class TestBridgeDownloadAssetOffline(unittest.TestCase):
    """_download_asset_to_cache 离线模式行为测试。"""

    def test_download_asset_offline_raises_filenotfound(self):
        """离线模式（_stub is None）抛 FileNotFoundError。"""
        import asyncio
        bridge = OrcaStudioBridge()
        with self.assertRaises(FileNotFoundError) as ctx:
            asyncio.run(bridge._download_asset_to_cache("foot.stl"))  # noqa: SLF001 测试白盒
        # 错误消息应含 xml_file_dir 引导
        self.assertIn("Offline mode", str(ctx.exception))


class TestBridgeDownloadAssetOnline(unittest.TestCase):
    """_download_asset_to_cache 在线模式原子落盘测试。"""

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp(prefix="orcagym_test_dl_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_download_asset_writes_atomically(self):
        """在线模式下载资源并原子落盘到 xml_file_dir。"""
        import asyncio
        from orca_gym.protos import mjc_message_pb2

        asset_content = b"STL_BINARY_DATA"

        class MockStub:
            async def LoadContentFile(self, request):
                resp = mjc_message_pb2.LoadContentFileResponse()
                resp.status = mjc_message_pb2.LoadContentFileResponse.SUCCESS
                resp.content = asset_content
                return resp

        bridge = OrcaStudioBridge(stub=MockStub())
        bridge.configure_offline(
            os.path.join(self.tmp_dir, "dummy.xml"), assets_dir=self.tmp_dir
        )
        result = asyncio.run(bridge._download_asset_to_cache("foot.stl"))  # noqa: SLF001 测试白盒
        # 返回路径正确
        self.assertEqual(result, os.path.join(self.tmp_dir, "foot.stl"))
        # 文件内容正确
        with open(result, "rb") as f:
            self.assertEqual(f.read(), asset_content)
        # 无残留临时文件
        leftovers = [
            f for f in os.listdir(self.tmp_dir)
            if f.endswith(".tmp") and f != "dummy.xml"
        ]
        self.assertEqual(leftovers, [])

    def test_download_asset_skips_when_exists(self):
        """文件已存在时不发起 gRPC 请求。"""
        import asyncio
        from orca_gym.protos import mjc_message_pb2

        # 预创建文件
        existing_path = os.path.join(self.tmp_dir, "exist.stl")
        with open(existing_path, "wb") as f:
            f.write(b"EXISTING")

        grpc_called = {"count": 0}

        class MockStub:
            async def LoadContentFile(self, request):
                grpc_called["count"] += 1
                resp = mjc_message_pb2.LoadContentFileResponse()
                resp.status = mjc_message_pb2.LoadContentFileResponse.SUCCESS
                resp.content = b"SHOULD_NOT_BE_USED"
                return resp

        bridge = OrcaStudioBridge(stub=MockStub())
        bridge.configure_offline(
            os.path.join(self.tmp_dir, "dummy.xml"), assets_dir=self.tmp_dir
        )
        result = asyncio.run(bridge._download_asset_to_cache("exist.stl"))  # noqa: SLF001 测试白盒
        self.assertEqual(result, existing_path)
        self.assertEqual(grpc_called["count"], 0)
        # 内容未被覆盖
        with open(result, "rb") as f:
            self.assertEqual(f.read(), b"EXISTING")

    def test_download_asset_creates_subdir(self):
        """file 含子目录时自动创建子目录。"""
        import asyncio
        from orca_gym.protos import mjc_message_pb2

        class MockStub:
            async def LoadContentFile(self, request):
                resp = mjc_message_pb2.LoadContentFileResponse()
                resp.status = mjc_message_pb2.LoadContentFileResponse.SUCCESS
                resp.content = b"DATA"
                return resp

        bridge = OrcaStudioBridge(stub=MockStub())
        bridge.configure_offline(
            os.path.join(self.tmp_dir, "dummy.xml"), assets_dir=self.tmp_dir
        )
        result = asyncio.run(bridge._download_asset_to_cache("g1/foot.stl"))  # noqa: SLF001 测试白盒
        self.assertTrue(os.path.isfile(result))
        self.assertIn("g1", result)

    def test_download_asset_raises_on_grpc_failure(self):
        """gRPC 返回失败状态时抛异常。"""
        import asyncio
        from orca_gym.protos import mjc_message_pb2

        class MockStub:
            async def LoadContentFile(self, request):
                resp = mjc_message_pb2.LoadContentFileResponse()
                resp.status = mjc_message_pb2.LoadContentFileResponse.ERROR
                resp.error_message = "remote not found"
                return resp

        bridge = OrcaStudioBridge(stub=MockStub())
        bridge.configure_offline(
            os.path.join(self.tmp_dir, "dummy.xml"), assets_dir=self.tmp_dir
        )
        with self.assertRaises(Exception):
            asyncio.run(bridge._download_asset_to_cache("bad.stl"))  # noqa: SLF001 测试白盒

    def test_download_asset_raises_on_empty_content(self):
        """gRPC 返回空内容时抛异常。"""
        import asyncio
        from orca_gym.protos import mjc_message_pb2

        class MockStub:
            async def LoadContentFile(self, request):
                resp = mjc_message_pb2.LoadContentFileResponse()
                resp.status = mjc_message_pb2.LoadContentFileResponse.SUCCESS
                resp.content = b""
                return resp

        bridge = OrcaStudioBridge(stub=MockStub())
        bridge.configure_offline(
            os.path.join(self.tmp_dir, "dummy.xml"), assets_dir=self.tmp_dir
        )
        with self.assertRaises(Exception):
            asyncio.run(bridge._download_asset_to_cache("empty.stl"))  # noqa: SLF001 测试白盒


class TestBridgeLoadModelXmlIntegration(unittest.TestCase):
    """load_model_xml 两分支后统一调用 process_xml_file 集成测试。"""

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp(prefix="orcagym_test_load_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_model_xml_offline_calls_process_xml_file(self):
        """离线模式 load_model_xml 返回前调用 process_xml_file。"""
        import asyncio
        called = {"path": None}

        async def fake_process_xml_file(file_path):
            called["path"] = file_path

        bridge = OrcaStudioBridge()
        bridge.process_xml_file = fake_process_xml_file
        # 创建本地 XML 文件
        xml_path = os.path.join(self.tmp_dir, "scene.xml")
        with open(xml_path, "w") as f:
            f.write("<mujoco/>")
        bridge.configure_offline(xml_path, assets_dir=self.tmp_dir)
        result = asyncio.run(bridge.load_model_xml())
        self.assertEqual(result, xml_path)
        self.assertEqual(called["path"], xml_path)

    def test_load_model_xml_online_calls_process_xml_file(self):
        """在线模式 load_model_xml 返回前调用 process_xml_file。"""
        import asyncio
        from orca_gym.protos import mjc_message_pb2

        process_called = {"path": None}

        async def fake_process_xml_file(file_path):
            process_called["path"] = file_path

        class MockStub:
            async def LoadLocalEnv(self, request):
                resp = mjc_message_pb2.LoadLocalEnvResponse()
                if request.req_type == mjc_message_pb2.LoadLocalEnvRequest.XML_FILE_NAME:
                    resp.status = mjc_message_pb2.LoadLocalEnvResponse.SUCCESS
                    resp.file_name = "scene.xml"
                else:
                    resp.status = mjc_message_pb2.LoadLocalEnvResponse.SUCCESS
                    resp.xml_content = b"<mujoco/>"
                return resp

        bridge = OrcaStudioBridge(stub=MockStub())
        bridge.process_xml_file = fake_process_xml_file
        bridge.configure_offline(
            os.path.join(self.tmp_dir, "dummy.xml"), assets_dir=self.tmp_dir
        )
        result = asyncio.run(bridge.load_model_xml())
        # process_xml_file 被以返回路径调用
        self.assertIsNotNone(process_called["path"])
        self.assertEqual(result, process_called["path"])


if __name__ == "__main__":
    unittest.main()
