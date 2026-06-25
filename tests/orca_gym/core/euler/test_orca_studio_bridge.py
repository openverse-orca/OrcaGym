"""OrcaStudioBridge P3 单元测试。

验证点参见 docs/design/development/orca_gym_euler_development.md 第 4.3 节。
"""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
from orca_gym.core.orca_gym_local import AnchorType, CaptureMode

_SCENE_XML = os.path.join(
    os.path.dirname(__file__), "fixtures", "test_scene.xml"
)


class TestOrcaStudioBridge(unittest.TestCase):
    def test_init_with_none_stub(self) -> None:
        # stub=None 时不报错，is_offline 为 True
        bridge = OrcaStudioBridge(stub=None)
        self.assertIsNone(bridge.stub)
        self.assertTrue(bridge.is_offline)

    def test_init_with_stub(self) -> None:
        # stub 非 None 时 is_offline 为 False
        stub = MagicMock()
        bridge = OrcaStudioBridge(stub=stub)
        self.assertFalse(bridge.is_offline)

    def test_offline_render_skips_grpc(self) -> None:
        # 离线模式下 render 安全跳过，不调用 gRPC
        bridge = OrcaStudioBridge(stub=None)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(bridge.render(np.zeros(3), 0.0))
        finally:
            loop.close()

    def test_offline_pause_skips_grpc(self) -> None:
        bridge = OrcaStudioBridge(stub=None)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(bridge.pause_simulation())
        finally:
            loop.close()

    def test_offline_video_methods_skip_grpc(self) -> None:
        bridge = OrcaStudioBridge(stub=None)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                bridge.begin_save_video("/tmp/test.mp4", CaptureMode.ASYNC)
            )
            loop.run_until_complete(bridge.stop_save_video())
            frame = loop.run_until_complete(bridge.get_current_frame())
            self.assertEqual(frame, -1)
            ts = loop.run_until_complete(bridge.get_camera_time_stamp(0))
            self.assertEqual(ts, {})
            png = loop.run_until_complete(bridge.get_frame_png("/tmp/test.png"))
            self.assertEqual(png, {})
        finally:
            loop.close()

    def test_offline_manipulation_returns_defaults(self) -> None:
        bridge = OrcaStudioBridge(stub=None)
        loop = asyncio.new_event_loop()
        try:
            anchored, anchor_type = loop.run_until_complete(
                bridge.get_body_manipulation_anchored()
            )
            self.assertIsNone(anchored)
            self.assertEqual(anchor_type, AnchorType.NONE)
            movement = loop.run_until_complete(
                bridge.get_body_manipulation_movement()
            )
            self.assertIn("delta_pos", movement)
            self.assertIn("delta_quat", movement)
        finally:
            loop.close()

    def test_offline_load_model_xml_with_local_path(self) -> None:
        # 离线模式配置 local_xml_path 后，load_model_xml 返回该路径
        bridge = OrcaStudioBridge(stub=None)
        bridge.configure_offline(_SCENE_XML)
        loop = asyncio.new_event_loop()
        try:
            xml_path = loop.run_until_complete(bridge.load_model_xml())
            self.assertEqual(xml_path, os.path.abspath(_SCENE_XML))
        finally:
            loop.close()

    def test_offline_load_model_xml_without_path_raises(self) -> None:
        # 离线模式未配置 local_xml_path 时抛出 RuntimeError
        bridge = OrcaStudioBridge(stub=None)
        loop = asyncio.new_event_loop()
        try:
            with self.assertRaises(RuntimeError):
                loop.run_until_complete(bridge.load_model_xml())
        finally:
            loop.close()

    def test_render_passes_qpos_and_time(self) -> None:
        # 在线模式：render(qpos, time) 调用 stub.UpdateLocalEnv
        stub = MagicMock()
        response = MagicMock()
        response.override_ctrls = []
        stub.UpdateLocalEnv = AsyncMock(return_value=response)
        bridge = OrcaStudioBridge(stub=stub)
        qpos = np.array([1.0, 2.0, 3.0])
        sim_time = 0.5
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(bridge.render(qpos, sim_time))
            stub.UpdateLocalEnv.assert_called_once()
            call_args = stub.UpdateLocalEnv.call_args[0][0]
            np.testing.assert_allclose(call_args.qpos, qpos)
            self.assertEqual(call_args.time, sim_time)
        finally:
            loop.close()

    def test_pause_simulation_calls_stub(self) -> None:
        # 在线模式：pause_simulation 调用 stub.SetSimulationState
        stub = MagicMock()
        stub.SetSimulationState = AsyncMock()
        bridge = OrcaStudioBridge(stub=stub)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(bridge.pause_simulation())
            stub.SetSimulationState.assert_called_once()
        finally:
            loop.close()

    def test_configure_offline_sets_paths(self) -> None:
        # configure_offline 设置本地 XML 路径和资源目录
        bridge = OrcaStudioBridge(stub=None)
        bridge.configure_offline(_SCENE_XML)
        self.assertEqual(bridge._local_xml_path, os.path.abspath(_SCENE_XML))
        self.assertEqual(
            bridge._xml_assets_dir, os.path.dirname(os.path.abspath(_SCENE_XML))
        )

    def test_override_ctrls_property(self) -> None:
        # override_ctrls 属性返回字典
        bridge = OrcaStudioBridge(stub=None)
        self.assertIsInstance(bridge.override_ctrls, dict)


if __name__ == "__main__":
    unittest.main()
