"""``orca_gym.recorder.video_recorder`` 单元测试。

覆盖：
- ``RollingFrameBuffer`` 区间提取
- ``VideoRecorder.remux_range`` 返回 ``RemuxResult``（帧号 ↔ 物理 index 映射 + 路径）
- ``VideoRecorder.remux_range`` PTS 使用 ``timestamp_ns``（不使用固定 FPS 时间基）
- ``VideoRecorderManager`` 统一接口（``save_streaming``）
- ``VideoRecorderManager.stop_all_and_save`` 自动保存
- ``VideoRecorder`` 等待触发任务队列（多个区间任务独立，互不干扰）
- ``VideoRecorder.start`` 连接超时机制（不要求首帧 IDR）

运行方式:
    <conda-base>/envs/orca_gym/bin/python -m pytest tests/orca_gym/recorder/test_video_recorder.py -v
"""

from __future__ import annotations

import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import av

from orca_gym.recorder.camera_recorder import RemuxResult, _nal_data_is_keyframe
from orca_gym.recorder.rolling_frame_buffer import RollingFrameBuffer
from orca_gym.recorder.video_recorder import (
    VideoRecorder,
    VideoRecorderManager,
    _STREAM_COLOR,
)


# =============================================================================
# RollingFrameBuffer
# =============================================================================


class TestRollingFrameBuffer(unittest.TestCase):
    """``RollingFrameBuffer`` 区间提取。"""

    def test_extract_range_inclusive(self):
        buf = RollingFrameBuffer(max_frames=100)
        for i in range(10):
            buf.append(i, timestamp_ns=i * 1_000_000, nal_data=b"\x65\x00", is_keyframe=(i == 0))
        frames = buf.extract_range(2, 5)
        self.assertEqual(len(frames), 4)
        self.assertEqual([f.simulate_index for f in frames], [2, 3, 4, 5])

    def test_extract_range_empty(self):
        buf = RollingFrameBuffer(max_frames=100)
        buf.append(0, 0, b"\x65\x00", True)
        self.assertEqual(buf.extract_range(10, 20), [])

    def test_eviction(self):
        buf = RollingFrameBuffer(max_frames=3)
        for i in range(5):
            buf.append(i, i * 1000, b"\x65\x00", True)
        # 只保留最新 3 帧：2, 3, 4
        self.assertEqual(len(buf), 3)
        self.assertEqual(buf.get_oldest_simulate_index(), 2)
        self.assertEqual(buf.get_latest_simulate_index(), 4)


# =============================================================================
# VideoRecorder.remux_range
# =============================================================================


def _make_idr_nal() -> bytes:
    """构造一个 IDR NAL 单元（type 5）。"""
    # 0x65 = 0x60 (NAL header) | 0x05 (type 5 IDR)
    return b"\x65\x00\x00\x00\x01\x00"


def _make_p_nal() -> bytes:
    """构造一个 P 帧的 NAL 单元（type 1）。"""
    return b"\x41\x00\x00\x00\x01\x00"


class TestDoRemux(unittest.TestCase):
    """``VideoRecorder.remux_range`` 返回 ``RemuxResult`` 且 PTS 使用 timestamp_ns。"""

    def _build_recorder_with_frames(
        self, n_frames: int, start_idx: int = 0
    ) -> tuple[VideoRecorder, list[int], list[int]]:
        """构造一个已缓存 n_frames 帧的 VideoRecorder（不启动线程）。

        Returns:
            (recorder, frame_indices, timestamps_ns)
        """
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        frame_indices: list[int] = []
        timestamps_ns: list[int] = []
        for i in range(n_frames):
            sim_idx = start_idx + i
            # 时间戳间隔不均匀：1ms, 3ms, 6ms, ...（i*(i+1)/2 ms）
            # 验证不使用固定 FPS 时间基
            ts_ns = (i * (i + 1) // 2) * 1_000_000
            nal = _make_idr_nal() if i == 0 else _make_p_nal()
            rec._buffer.append(
                simulate_index=sim_idx,
                timestamp_ns=ts_ns,
                nal_data=nal,
                is_keyframe=(i == 0),
            )
            frame_indices.append(sim_idx)
            timestamps_ns.append(ts_ns)
        return rec, frame_indices, timestamps_ns

    def test_remux_returns_remux_result_with_mapping(self):
        """``remux_range`` 返回 ``RemuxResult``，包含帧号 ↔ 物理 index 映射和路径。"""
        rec, expected_indices, expected_ts = self._build_recorder_with_frames(
            n_frames=5, start_idx=10
        )
        with TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "out.mp4")
            result = rec.remux_range(
                file_path=out_path,
                start_simulate_index=10,
                end_simulate_index=14,
            )
            self.assertIsInstance(result, RemuxResult)
            self.assertEqual(result.frame_count, 5)
            # 帧号 ↔ 物理 index 映射
            self.assertEqual(result.frame_indices, expected_indices)
            # 时间戳列表
            self.assertEqual(result.timestamps_ns, expected_ts)
            # 文件路径
            self.assertTrue(result.file_path.endswith("out.mp4"))
            self.assertTrue(Path(result.file_path).exists())

    def test_remux_pts_uses_timestamp_ns(self):
        """PTS 使用 ``timestamp_ns`` 而非固定 FPS 时间基。

        验证方式：读取生成的 MP4，检查帧的 PTS 间隔与 timestamp_ns 间隔一致
        （而非固定 1/fps）。
        """
        rec, expected_indices, expected_ts = self._build_recorder_with_frames(
            n_frames=3, start_idx=0
        )
        # timestamp_ns: [0, 1ms, 3ms]（间隔 1ms, 2ms）
        self.assertEqual(expected_ts, [0, 1_000_000, 3_000_000])

        with TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "out.mp4")
            result = rec.remux_range(
                file_path=out_path,
                start_simulate_index=0,
                end_simulate_index=2,
            )

            # 读取 MP4 验证 PTS
            container = av.open(result.file_path, mode="r")
            try:
                stream = container.streams.video[0]
                # time_base 应为 1/1_000_000（微秒）
                self.assertEqual(stream.time_base, Fraction(1, 1_000_000))

                packets = list(container.demux(stream))
                # 过滤空 packet（flush packet）
                pkts = [p for p in packets if p.dts is not None]
                self.assertEqual(len(pkts), 3)

                # PTS 应为 timestamp_ns 转微秒（首帧偏移到 0）
                first_ts_ns = expected_ts[0]
                expected_pts_us = [
                    (ts - first_ts_ns) // 1000 for ts in expected_ts
                ]
                actual_pts = [p.pts for p in pkts]
                self.assertEqual(actual_pts, expected_pts_us)
            finally:
                container.close()

    def test_remux_empty_range_raises(self):
        """区间无帧时 ``remux_range`` 抛 ValueError。"""
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                rec.remux_range(
                    file_path=str(Path(tmpdir) / "out.mp4"),
                    start_simulate_index=0,
                    end_simulate_index=10,
                )

    def test_save_streaming_returns_remux_result(self):
        """``save_streaming`` 返回 ``Future``，保存完成后解析出 ``RemuxResult``。"""
        rec, _, _ = self._build_recorder_with_frames(n_frames=3, start_idx=0)
        with TemporaryDirectory() as tmpdir:
            future = rec.save_streaming(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=0,
                end_simulate_index=2,
            )
            result = future.result(timeout=10)
        self.assertIsInstance(result, RemuxResult)
        self.assertEqual(result.frame_count, 3)


# =============================================================================
# VideoRecorderManager 统一接口
# =============================================================================


class TestVideoRecorderManagerUnifiedInterface(unittest.TestCase):
    """``VideoRecorderManager`` 统一录制接口。"""

    def test_save_streaming_via_manager(self):
        """通过 manager 的 ``save_streaming`` 完成区间保存。"""
        manager = VideoRecorderManager()
        recorder = VideoRecorder("cam1", color_port=7071)
        manager._recorders[("cam1", _STREAM_COLOR)] = recorder

        # 缓存帧
        for i in range(5):
            nal = _make_idr_nal() if i == 0 else _make_p_nal()
            recorder._buffer.append(
                simulate_index=i,
                timestamp_ns=i * 1_000_000,
                nal_data=nal,
                is_keyframe=(i == 0),
            )

        with TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "out.mp4")
            # 端帧已缓存，save_streaming 立即触发由 worker 异步 remux
            future = manager.save_streaming(
                "cam1",
                out_path,
                start_simulate_index=0,
                end_simulate_index=4,
            )
            result = future.result(timeout=10)

        self.assertIsInstance(result, RemuxResult)
        self.assertEqual(result.frame_count, 5)
        self.assertEqual(result.frame_indices, [0, 1, 2, 3, 4])
        # get_last_result 应返回相同 Future
        self.assertIs(manager.get_last_result("cam1"), future)

    def test_save_streaming_multiple_independent_tasks(self):
        """同时注册多个区间任务，各任务独立携带 start/end，互不覆盖。

        这是本次重构的核心：早期实现用单一 ``_task`` 变量，第二个任务会覆盖
        前一个任务的 start/end，导致保存区间错乱。现在每个任务独立入队。
        """
        manager = VideoRecorderManager()
        recorder = VideoRecorder("cam1", color_port=7071)
        manager._recorders[("cam1", _STREAM_COLOR)] = recorder

        # 缓存 0..9 帧，IDR 在 0
        for i in range(10):
            nal = _make_idr_nal() if i == 0 else _make_p_nal()
            recorder._buffer.append(
                simulate_index=i,
                timestamp_ns=i * 1_000_000,
                nal_data=nal,
                is_keyframe=(i == 0),
            )

        with TemporaryDirectory() as tmpdir:
            # 注册两个区间任务（端帧均已缓存，立即触发）
            future_a = manager.save_streaming(
                "cam1",
                str(Path(tmpdir) / "a.mp4"),
                start_simulate_index=1,
                end_simulate_index=3,
            )
            future_b = manager.save_streaming(
                "cam1",
                str(Path(tmpdir) / "b.mp4"),
                start_simulate_index=4,
                end_simulate_index=6,
            )
            result_a = future_a.result(timeout=10)
            result_b = future_b.result(timeout=10)

        # 两个任务各自独立，区间互不干扰
        self.assertEqual(result_a.frame_indices, [1, 2, 3])
        self.assertEqual(result_b.frame_indices, [4, 5, 6])

    def test_save_streaming_no_recorder_raises(self):
        """对不存在的相机调用 ``save_streaming`` 抛 ValueError。"""
        manager = VideoRecorderManager()
        with self.assertRaises(ValueError):
            manager.save_streaming(
                "nonexistent", "/tmp/x.mp4", start_simulate_index=0, end_simulate_index=10
            )

    def test_stop_all_and_save(self):
        """``stop_all_and_save`` 自动保存未完成任务。"""
        manager = VideoRecorderManager()
        rec1 = VideoRecorder("cam1", color_port=7071)
        rec2 = VideoRecorder("cam2", color_port=7072)
        manager._recorders[("cam1", _STREAM_COLOR)] = rec1
        manager._recorders[("cam2", _STREAM_COLOR)] = rec2

        for rec in [rec1, rec2]:
            for i in range(3):
                nal = _make_idr_nal() if i == 0 else _make_p_nal()
                rec._buffer.append(
                    simulate_index=i,
                    timestamp_ns=i * 1_000_000,
                    nal_data=nal,
                    is_keyframe=(i == 0),
                )

        with TemporaryDirectory() as tmpdir:
            # cam1 注册未触发任务（end=100 未到），cam2 不注册
            manager.save_streaming(
                "cam1",
                str(Path(tmpdir) / "cam1.mp4"),
                start_simulate_index=0,
                end_simulate_index=100,
            )
            saved = manager.stop_all_and_save()

        # cam1 保存成功（复合键 cam1__color__0），cam2 未注册任务不保存
        # 键名后缀 __0 是 pending future 的索引（一个 recorder 可能有多个
        # pending save，按 __0 / __1 / ... 编号）
        self.assertIn("cam1__color__0", saved)
        self.assertNotIn("cam2__color__0", saved)
        self.assertEqual(saved["cam1__color__0"].frame_count, 3)

    def test_is_saving_any(self):
        """``is_saving_any`` 正确反映是否有等待触发的保存任务。"""
        manager = VideoRecorderManager()
        rec = VideoRecorder("cam1", color_port=7071)
        manager._recorders[("cam1", _STREAM_COLOR)] = rec
        self.assertFalse(manager.is_saving_any())

        # 注册未触发任务（end 未到），is_saving 应为 True
        manager.save_streaming(
            "cam1", "/tmp/x.mp4", start_simulate_index=0, end_simulate_index=100
        )
        self.assertTrue(manager.is_saving_any())

    def test_get_stats(self):
        """``get_stats`` 返回所有录制器状态（复合键 ``cam__kind``）。"""
        manager = VideoRecorderManager()
        manager._recorders[("cam1", _STREAM_COLOR)] = VideoRecorder("cam1", color_port=7071)
        manager._recorders[("cam2", _STREAM_COLOR)] = VideoRecorder("cam2", color_port=7072)
        stats = manager.get_stats()
        self.assertEqual(len(stats), 2)
        self.assertIn("cam1__color", stats)
        self.assertIn("cam2__color", stats)
        self.assertEqual(stats["cam1__color"]["camera_name"], "cam1")
        # stream_kind 过滤：只返回 color，键为 camera_name
        stats_color = manager.get_stats("color")
        self.assertEqual(len(stats_color), 2)
        self.assertIn("cam1", stats_color)
        self.assertEqual(stats_color["cam1"]["camera_name"], "cam1")


# =============================================================================
# VideoRecorder.start 超时机制
# =============================================================================


class TestStartTimeoutMechanism(unittest.TestCase):
    """``VideoRecorder.start`` 连接超时机制（不要求首帧 IDR）。

    通过连接一个不存在的端口，验证：
    - 连接失败后 ``start`` 抛 ``ConnectionError``
    - 超时后 ``start`` 抛 ``ConnectionError``（不要求首帧 IDR）
    """

    def test_start_raises_on_connect_failure(self):
        """WebSocket 连接失败时 ``start`` 抛 ``ConnectionError``。

        连接一个不存在的端口，重试耗尽后应抛 ``ConnectionError``。
        """
        rec = VideoRecorder(camera_name="test_cam", color_port=9999)

        # 加速：1 次重试，0.01s 间隔
        with patch("orca_gym.recorder.camera_recorder._WS_CONNECT_RETRY_INTERVAL", 0.01):
            with patch("orca_gym.recorder.camera_recorder._WS_CONNECT_RETRIES", 1):
                with patch(
                    "orca_gym.recorder.camera_recorder._WS_CONNECT_TIMEOUT", 5.0
                ):
                    with self.assertRaises(ConnectionError) as ctx:
                        rec.start()
        self.assertIn("Failed to connect", str(ctx.exception))
        self.assertFalse(rec.is_running)

    def test_start_does_not_wait_for_first_frame(self):
        """``start`` 只等待 WebSocket 连接建立，不等待首帧 IDR。

        验证方式：检查 ``_ws_connected_event`` 机制存在且 ``is_connected``
        属性独立于 ``received_first_idr``。本测试通过属性检查验证设计，
        不实际建立连接（实际连接测试需真实 OrcaStudio 环境）。
        """
        rec = VideoRecorder(camera_name="test_cam", color_port=9999)
        # 初始状态
        self.assertFalse(rec.is_connected)
        self.assertFalse(rec.received_first_idr)
        # is_connected 独立于 received_first_idr
        rec._ws_connected_event.set()
        self.assertTrue(rec.is_connected)
        self.assertFalse(rec.received_first_idr)


# =============================================================================
# VideoRecorderManager.start_recorder
# =============================================================================


class TestManagerStartRecorder(unittest.TestCase):
    """``VideoRecorderManager.start_recorder`` 行为。"""

    def test_start_recorder_creates_and_fails_on_bad_port(self):
        """``start_recorder`` 连接失败时 recorder 仍保留但未运行。"""
        manager = VideoRecorderManager()

        # 加速重试
        with patch("orca_gym.recorder.camera_recorder._WS_CONNECT_RETRY_INTERVAL", 0.01):
            with patch("orca_gym.recorder.camera_recorder._WS_CONNECT_RETRIES", 1):
                with patch(
                    "orca_gym.recorder.camera_recorder._WS_CONNECT_TIMEOUT", 5.0
                ):
                    # 连接不存在的端口会失败
                    with self.assertRaises(ConnectionError):
                        manager.start_recorder("cam1", color_port=9999)

        # 失败后 recorder 仍存在但未运行
        rec = manager.get_recorder("cam1")
        self.assertIsNotNone(rec)
        self.assertFalse(rec.is_running)

    def test_start_recorder_idempotent_when_already_running(self):
        """``start_recorder`` 幂等：已运行则不重新连接。

        通过手动设置 ``_running=True`` 模拟已运行的录制器，
        验证 ``start_recorder`` 不会再次调用 ``start()``。
        """
        manager = VideoRecorderManager()
        rec = VideoRecorder("cam1", color_port=9999)
        manager._recorders[("cam1", _STREAM_COLOR)] = rec
        # 手动标记为已运行（模拟已连接状态）
        rec._running = True
        rec._ws_connected_event.set()

        # start_recorder 应直接返回，不调用 start()
        # （若调用 start() 会抛 RuntimeError "already running"）
        manager.start_recorder("cam1", color_port=9999)
        self.assertTrue(rec.is_running)


# =============================================================================
# IDR primer 回溯
# =============================================================================


class TestIDRPrimerBacktrack(unittest.TestCase):
    """``remux_range`` 首帧非 IDR 时的 primer 回溯逻辑。"""

    def _build_recorder_with_periodic_idr(
        self,
        n_frames: int,
        idr_interval: int,
        start_idx: int = 0,
    ) -> VideoRecorder:
        """构造录制器，每隔 ``idr_interval`` 帧一个 IDR。"""
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        for i in range(n_frames):
            sim_idx = start_idx + i
            ts_ns = i * 1_000_000
            is_idr = (i % idr_interval == 0)
            nal = _make_idr_nal() if is_idr else _make_p_nal()
            rec._buffer.append(
                simulate_index=sim_idx,
                timestamp_ns=ts_ns,
                nal_data=nal,
                is_keyframe=is_idr,
            )
        return rec

    def test_remux_first_frame_is_idr_no_primer(self):
        """区间首帧是 IDR 时不回溯 primer。"""
        # sim_idx 0=IDR, 1=P, 2=P, 3=P, 4=IDR, 5=P, ...
        rec = self._build_recorder_with_periodic_idr(
            n_frames=10, idr_interval=5, start_idx=0
        )
        with TemporaryDirectory() as tmpdir:
            result = rec.remux_range(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=0,  # 首帧是 IDR
                end_simulate_index=4,
            )
        # 无 primer，frame_count == 区间帧数
        self.assertEqual(result.frame_count, 5)
        self.assertEqual(result.frame_indices, [0, 1, 2, 3, 4])

    def test_remux_first_frame_not_idr_backtracks_primer(self):
        """区间首帧非 IDR 时回溯到最近的 primer。

        缓存布局（idr_interval=4）：sim_idx 0=IDR, 1=P, 2=P, 3=P, 4=IDR, 5=P, 6=P, 7=P, 8=IDR, ...
        提取区间 [5, 7]：首帧 sim_idx=5 是 P 帧，
        回溯到 sim_idx=4 的 IDR 作为 primer。
        """
        rec = self._build_recorder_with_periodic_idr(
            n_frames=10, idr_interval=4, start_idx=0
        )
        with TemporaryDirectory() as tmpdir:
            result = rec.remux_range(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=5,  # 首帧是 P 帧
                end_simulate_index=7,
            )
            # frame_count 包含 primer（1 + 区间 3 帧 = 4）
            self.assertEqual(result.frame_count, 4)
            # frame_indices 不含 primer（primer sim_idx=4 属于上一段）
            self.assertEqual(result.frame_indices, [5, 6, 7])

            # 验证 MP4 首帧是 IDR（primer）
            container = av.open(result.file_path, mode="r")
            try:
                stream = container.streams.video[0]
                pkts = [p for p in container.demux(stream) if p.dts is not None]
                self.assertEqual(len(pkts), 4)
                # 首帧 PTS 应为 0（primer 是基准）
                self.assertEqual(pkts[0].pts, 0)
            finally:
                container.close()

    def test_remux_no_primer_available_when_first_not_idr(self):
        """首帧非 IDR 且无可用 primer 时仍保存（带警告）。"""
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        # 只放 P 帧，无任何 IDR
        for i in range(5):
            rec._buffer.append(
                simulate_index=i,
                timestamp_ns=i * 1_000_000,
                nal_data=_make_p_nal(),
                is_keyframe=False,
            )
        with TemporaryDirectory() as tmpdir:
            result = rec.remux_range(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=0,
                end_simulate_index=4,
            )
        # 无 primer，frame_count == 区间帧数
        self.assertEqual(result.frame_count, 5)
        self.assertEqual(result.frame_indices, [0, 1, 2, 3, 4])

    def test_primer_pts_is_zero(self):
        """primer 帧的 PTS 应为 0（基准时间戳）。"""
        rec = self._build_recorder_with_periodic_idr(
            n_frames=10, idr_interval=4, start_idx=0
        )
        with TemporaryDirectory() as tmpdir:
            result = rec.remux_range(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=5,
                end_simulate_index=7,
            )
            container = av.open(result.file_path, mode="r")
            try:
                stream = container.streams.video[0]
                pkts = [p for p in container.demux(stream) if p.dts is not None]
                # primer 是第 0 帧，PTS 应为 0
                self.assertEqual(pkts[0].pts, 0)
                # 后续帧 PTS > 0
                self.assertGreater(pkts[1].pts, 0)
            finally:
                container.close()

    def test_primer_fallback_when_idr_has_negative_sim_idx(self):
        """I 帧 sim_idx=-1 时通过 timestamp fallback 回溯 primer。

        场景：引擎自动产生的 I 帧携带 sim_idx=-1（uint32 溢出），
        sim_idx 回溯无法找到，改用 timestamp 回溯。

        缓存布局（按插入顺序）：
            sim_idx=-1, ts=0    : IDR（引擎自动产生）
            sim_idx=0,  ts=1000 : P
            sim_idx=1,  ts=2000 : P
            sim_idx=2,  ts=3000 : P
            sim_idx=3,  ts=4000 : P
            sim_idx=4,  ts=5000 : P
        提取区间 [1, 4]：首帧 sim_idx=1 是 P 帧。
        sim_idx 回溯查找 < 1 的 IDR：sim_idx=-1 的 IDR 满足 -1 < 1，
        但因为 -1 < 0 触发 early return None。
        fallback 用 timestamp 回溯：找到 ts=0 < 2000 的 IDR（sim_idx=-1）。
        """
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        # sim_idx=-1 的 IDR
        rec._buffer.append(-1, 0, _make_idr_nal(), True)
        # sim_idx 0-4 的 P 帧
        for i in range(5):
            rec._buffer.append(
                simulate_index=i,
                timestamp_ns=(i + 1) * 1000,
                nal_data=_make_p_nal(),
                is_keyframe=False,
            )

        with TemporaryDirectory() as tmpdir:
            result = rec.remux_range(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=1,
                end_simulate_index=4,
            )
            # frame_count 包含 primer（1 + 区间 4 帧 = 5）
            self.assertEqual(result.frame_count, 5)
            # frame_indices 不含 primer
            self.assertEqual(result.frame_indices, [1, 2, 3, 4])

    def test_primer_fallback_picks_nearest_idr_by_timestamp(self):
        """timestamp fallback 选择最近的 IDR（而非最旧的）。

        缓存布局：
            sim_idx=-1, ts=0    : IDR（旧）
            sim_idx=0,  ts=1000 : P
            sim_idx=-1, ts=2000 : IDR（新）
            sim_idx=1,  ts=3000 : P
            sim_idx=2,  ts=4000 : P

        提取区间 [1, 2]：首帧 sim_idx=1 是 P 帧。
        sim_idx 回溯失败（-1 < 0）。
        timestamp fallback 应选 ts=2000 的 IDR（最近），而非 ts=0 的。
        """
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        rec._buffer.append(-1, 0, _make_idr_nal(), True)
        rec._buffer.append(0, 1000, _make_p_nal(), False)
        rec._buffer.append(-1, 2000, _make_idr_nal(), True)
        rec._buffer.append(1, 3000, _make_p_nal(), False)
        rec._buffer.append(2, 4000, _make_p_nal(), False)

        with TemporaryDirectory() as tmpdir:
            result = rec.remux_range(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=1,
                end_simulate_index=2,
            )
            # frame_count = 1 primer + 2 区间帧 = 3
            self.assertEqual(result.frame_count, 3)
            self.assertEqual(result.frame_indices, [1, 2])

            # 验证 primer 的 PTS=0（基准时间戳应为 ts=2000）
            container = av.open(result.file_path, mode="r")
            try:
                stream = container.streams.video[0]
                pkts = [p for p in container.demux(stream) if p.dts is not None]
                # primer PTS=0
                self.assertEqual(pkts[0].pts, 0)
                # 第二帧（sim_idx=1, ts=3000）PTS = (3000-2000)//1000 = 1
                self.assertEqual(pkts[1].pts, 1)
            finally:
                container.close()


# =============================================================================
# RollingFrameBuffer.get_latest_keyframe_strictly_before
# =============================================================================


class TestGetLatestKeyframeStrictlyBefore(unittest.TestCase):
    """``get_latest_keyframe_strictly_before`` 方法。"""

    def test_find_strictly_before(self):
        buf = RollingFrameBuffer(max_frames=100)
        # sim_idx 0=IDR, 1=P, 2=P, 3=P, 4=IDR, 5=P
        for i in range(6):
            buf.append(
                simulate_index=i,
                timestamp_ns=i * 1000,
                nal_data=b"\x65" if i % 4 == 0 else b"\x41",
                is_keyframe=(i % 4 == 0),
            )
        # 查找 < 5 的最近 IDR：应为 sim_idx=4
        kf = buf.get_latest_keyframe_strictly_before(5)
        self.assertIsNotNone(kf)
        self.assertEqual(kf.simulate_index, 4)

    def test_find_strictly_before_excludes_start(self):
        """``get_latest_keyframe_strictly_before`` 排除 ``sim_idx`` 自身。"""
        buf = RollingFrameBuffer(max_frames=100)
        # sim_idx 0=IDR
        buf.append(0, 0, b"\x65", True)
        # sim_idx 5=IDR
        buf.append(5, 5000, b"\x65", True)

        # 查找 < 5 的最近 IDR：应为 sim_idx=0（不返回 5 自身）
        kf = buf.get_latest_keyframe_strictly_before(5)
        self.assertIsNotNone(kf)
        self.assertEqual(kf.simulate_index, 0)

    def test_find_strictly_before_none_when_no_idr(self):
        buf = RollingFrameBuffer(max_frames=100)
        buf.append(0, 0, b"\x41", False)
        buf.append(1, 1000, b"\x41", False)
        kf = buf.get_latest_keyframe_strictly_before(2)
        self.assertIsNone(kf)

    def test_find_strictly_before_none_when_empty(self):
        buf = RollingFrameBuffer(max_frames=100)
        kf = buf.get_latest_keyframe_strictly_before(0)
        self.assertIsNone(kf)

    def test_strictly_before_returns_none_for_negative_sim_idx(self):
        """``get_latest_keyframe_strictly_before`` 对负数 sim_idx 返回 None。"""
        buf = RollingFrameBuffer(max_frames=100)
        buf.append(-1, 0, b"\x65", True)
        buf.append(0, 1000, b"\x41", False)
        # sim_idx=-1 是无效值，应返回 None
        kf = buf.get_latest_keyframe_strictly_before(-1)
        self.assertIsNone(kf)


class TestGetLatestKeyframeBeforeTimestamp(unittest.TestCase):
    """``get_latest_keyframe_before_timestamp`` 方法（timestamp fallback）。"""

    def test_find_by_timestamp(self):
        """按时间戳查找之前的最近 IDR。"""
        buf = RollingFrameBuffer(max_frames=100)
        buf.append(0, 1000, b"\x65", True)   # IDR, ts=1000
        buf.append(1, 2000, b"\x41", False)  # P,   ts=2000
        buf.append(2, 3000, b"\x65", True)   # IDR, ts=3000
        buf.append(3, 4000, b"\x41", False)  # P,   ts=4000

        # 查找 ts < 4000 的最近 IDR：应为 ts=3000
        kf = buf.get_latest_keyframe_before_timestamp(4000)
        self.assertIsNotNone(kf)
        self.assertEqual(kf.timestamp_ns, 3000)
        self.assertEqual(kf.simulate_index, 2)

    def test_find_by_timestamp_with_negative_sim_idx(self):
        """I 帧 sim_idx=-1 时按时间戳仍能找到。"""
        buf = RollingFrameBuffer(max_frames=100)
        buf.append(-1, 1000, b"\x65", True)   # IDR, sim_idx=-1
        buf.append(0, 2000, b"\x41", False)   # P
        buf.append(1, 3000, b"\x41", False)   # P

        # 查找 ts < 3000 的最近 IDR：应为 ts=1000（sim_idx=-1）
        kf = buf.get_latest_keyframe_before_timestamp(3000)
        self.assertIsNotNone(kf)
        self.assertEqual(kf.timestamp_ns, 1000)
        self.assertEqual(kf.simulate_index, -1)

    def test_find_by_timestamp_excludes_target(self):
        """排除 ts == target 的帧。"""
        buf = RollingFrameBuffer(max_frames=100)
        buf.append(0, 1000, b"\x65", True)
        buf.append(1, 2000, b"\x65", True)

        # 查找 ts < 2000 的最近 IDR：应为 ts=1000（不返回 ts=2000）
        kf = buf.get_latest_keyframe_before_timestamp(2000)
        self.assertIsNotNone(kf)
        self.assertEqual(kf.timestamp_ns, 1000)

    def test_find_by_timestamp_none_when_empty(self):
        buf = RollingFrameBuffer(max_frames=100)
        kf = buf.get_latest_keyframe_before_timestamp(1000)
        self.assertIsNone(kf)


class TestGetFirstKeyframeAfter(unittest.TestCase):
    """``get_first_keyframe_after`` 方法（前向查找第一个 IDR 帧）。"""

    def test_find_first_keyframe_after(self):
        """返回 >= sim_idx 的第一个 IDR 帧。

        缓存布局（idr_interval=4）：sim_idx 0=IDR, 1=P, 2=P, 3=P, 4=IDR, 5=P, 6=P, 7=P, 8=IDR
        """
        buf = RollingFrameBuffer(max_frames=100)
        for i in range(9):
            buf.append(
                simulate_index=i,
                timestamp_ns=i * 1000,
                nal_data=b"\x65" if i % 4 == 0 else b"\x41",
                is_keyframe=(i % 4 == 0),
            )
        # 从 1 开始找：应返回 4（第一个 >= 1 的 IDR）
        kf = buf.get_first_keyframe_after(1)
        self.assertIsNotNone(kf)
        self.assertEqual(kf.simulate_index, 4)
        # 从 4 开始找：包含 4 自身
        kf = buf.get_first_keyframe_after(4)
        self.assertIsNotNone(kf)
        self.assertEqual(kf.simulate_index, 4)

    def test_find_inclusive_when_target_is_keyframe(self):
        """>= 语义：目标 sim_idx 自身若是关键帧则返回它。"""
        buf = RollingFrameBuffer(max_frames=100)
        buf.append(0, 0, b"\x65", True)
        buf.append(1, 1000, b"\x41", False)
        kf = buf.get_first_keyframe_after(0)
        self.assertIsNotNone(kf)
        self.assertEqual(kf.simulate_index, 0)

    def test_find_none_when_no_keyframe_after(self):
        buf = RollingFrameBuffer(max_frames=100)
        buf.append(0, 0, b"\x65", True)
        buf.append(1, 1000, b"\x41", False)
        buf.append(2, 2000, b"\x41", False)
        # 目标 3 之后无关键帧
        kf = buf.get_first_keyframe_after(3)
        self.assertIsNone(kf)

    def test_find_none_when_empty(self):
        buf = RollingFrameBuffer(max_frames=100)
        kf = buf.get_first_keyframe_after(0)
        self.assertIsNone(kf)


# =============================================================================
# 前向截断（truncate_to_keyframe）
# =============================================================================


class TestForwardTruncation(unittest.TestCase):
    """``remux_range(truncate_to_keyframe=True)`` 前向截断到首个关键帧。"""

    def _build_recorder_with_periodic_idr(
        self, n_frames: int, idr_interval: int, start_idx: int = 0
    ) -> VideoRecorder:
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        for i in range(n_frames):
            sim_idx = start_idx + i
            is_idr = (i % idr_interval == 0)
            rec._buffer.append(
                simulate_index=sim_idx,
                timestamp_ns=i * 1_000_000,
                nal_data=_make_idr_nal() if is_idr else _make_p_nal(),
                is_keyframe=is_idr,
            )
        return rec

    def test_truncate_drops_leading_non_keyframes(self):
        """区间首帧非关键帧时，截断到首个关键帧并丢弃其之前帧。

        缓存布局（idr_interval=4）：sim_idx 0=IDR, 1=P, 2=P, 3=P, 4=IDR, 5=P, ...
        提取区间 [1, 5]：首帧 sim_idx=1 是 P 帧，前向找到首个关键帧 sim_idx=4，
        丢弃 1..3，保存 4..5。
        """
        rec = self._build_recorder_with_periodic_idr(
            n_frames=10, idr_interval=4, start_idx=0
        )
        with TemporaryDirectory() as tmpdir:
            result = rec.remux_range(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=1,
                end_simulate_index=5,
                truncate_to_keyframe=True,
            )
        # 截断后仅保留 4, 5
        self.assertEqual(result.frame_indices, [4, 5])
        self.assertEqual(result.frame_count, 2)

    def test_truncate_no_drop_when_first_is_keyframe(self):
        """区间首帧即关键帧时不截断。"""
        rec = self._build_recorder_with_periodic_idr(
            n_frames=10, idr_interval=4, start_idx=0
        )
        with TemporaryDirectory() as tmpdir:
            result = rec.remux_range(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=4,
                end_simulate_index=7,
                truncate_to_keyframe=True,
            )
        # 首帧 4 是关键帧，无截断
        self.assertEqual(result.frame_indices, [4, 5, 6, 7])
        self.assertEqual(result.frame_count, 4)

    def test_non_truncation_default_keeps_primer(self):
        """默认（truncate_to_keyframe=False）保持原 primer 语义，不截断。"""
        rec = self._build_recorder_with_periodic_idr(
            n_frames=10, idr_interval=4, start_idx=0
        )
        with TemporaryDirectory() as tmpdir:
            result = rec.remux_range(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=5,
                end_simulate_index=7,
            )
        # primer 回溯：frame_count 含 primer，frame_indices 不含 primer
        self.assertEqual(result.frame_count, 4)
        self.assertEqual(result.frame_indices, [5, 6, 7])


# =============================================================================
# 关键帧检测（Annex-B 起始码扫描）
# =============================================================================


class TestKeyframeDetection(unittest.TestCase):
    """``_nal_data_is_keyframe`` 扫描 Annex-B 起始码内的 NAL type==5。"""

    def test_annexb_idr_with_start_code(self):
        """"00 00 01" 起始码后 NAL type==5 判定为关键帧。"""
        # 00 00 01 | 0x65 (type 5 IDR)
        self.assertTrue(_nal_data_is_keyframe(b"\x00\x00\x01\x65\x00\x00"))

    def test_annexb_4byte_start_code_idr(self):
        """"00 00 00 01" 起始码后 NAL type==5 判定为关键帧。"""
        self.assertTrue(_nal_data_is_keyframe(b"\x00\x00\x00\x01\x65\x00"))

    def test_annexb_sps_pps_then_idr(self):
        """SPS(type7)/PPS(type8) 前置的 IDR 仍判定为关键帧。"""
        # SPS(0x67) + PPS(0x68) + IDR(0x65)
        data = b"\x00\x00\x01\x67\x64\x00\x00\x00\x01\x68\xee\x3c\x80\x00\x00\x01\x65\x88\x84"
        self.assertTrue(_nal_data_is_keyframe(data))

    def test_annexb_leading_start_code_idr(self):
        """"00 00 00 01" 起始码开头 + IDR 判定为关键帧（旧实现会漏判）。"""
        self.assertTrue(_nal_data_is_keyframe(b"\x00\x00\x00\x01\x65\x00"))

    def test_non_idr_not_keyframe(self):
        """只有 P 帧（type 1）时判定为非关键帧。"""
        self.assertFalse(_nal_data_is_keyframe(b"\x00\x00\x01\x41\x00"))

    def test_empty_not_keyframe(self):
        self.assertFalse(_nal_data_is_keyframe(b""))

    def test_no_start_code_single_nal(self):
        """无起始码时按单 NAL 处理（首字节即 NAL header）。"""
        self.assertTrue(_nal_data_is_keyframe(b"\x65\x00\x00"))
        self.assertFalse(_nal_data_is_keyframe(b"\x41\x00\x00"))


# =============================================================================
# 异步保存回调任务队列（延迟帧触发）
# =============================================================================


class TestAsyncSaveTrigger(unittest.TestCase):
    """注册保存任务后，端帧延迟到达时触发（非阻塞 Future）。"""

    def test_save_future_triggered_when_end_frame_arrives(self):
        """端帧尚未到达时 ``save_streaming`` 不阻塞，目标帧到达后触发保存。"""
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        # 先缓存 0..4（latest=4）
        for i in range(5):
            rec._buffer.append(
                simulate_index=i,
                timestamp_ns=i * 1000,
                nal_data=_make_idr_nal() if i == 0 else _make_p_nal(),
                is_keyframe=(i == 0),
            )
        with TemporaryDirectory() as tmpdir:
            # 请求保存 0..490，但端帧 490 尚未到达
            future = rec.save_streaming(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=0,
                end_simulate_index=490,
            )
            # 注册后不应立即完成（端帧未到）
            self.assertFalse(future.done())
            # 模拟后续帧到达，触发保存任务
            for i in range(5, 491):
                rec._buffer.append(
                    simulate_index=i,
                    timestamp_ns=i * 1000,
                    nal_data=_make_p_nal(),
                    is_keyframe=False,
                )
                rec._dispatch_triggered_tasks(i)
            result = future.result(timeout=10)
        self.assertEqual(result.frame_count, 491)
        self.assertEqual(result.frame_indices[0], 0)
        self.assertEqual(result.frame_indices[-1], 490)

    def test_save_streaming_multiple_tasks_trigger_independently(self):
        """多个未触发任务在各自 end 到达时独立触发，互不影响。

        模拟早期 bug 场景：任务 A 的 end=5 与任务 B 的 start=3 紧邻，
        后注册的任务 B 不得覆盖任务 A 的 start/end。
        """
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        # 先缓存 0..2（latest=2）
        for i in range(3):
            rec._buffer.append(
                simulate_index=i,
                timestamp_ns=i * 1000,
                nal_data=_make_idr_nal() if i == 0 else _make_p_nal(),
                is_keyframe=(i == 0),
            )
        with TemporaryDirectory() as tmpdir:
            # 任务 A: [0, 5]，任务 B: [3, 7]（B 紧跟在 A 之后注册）
            future_a = rec.save_streaming(
                file_path=str(Path(tmpdir) / "a.mp4"),
                start_simulate_index=0,
                end_simulate_index=5,
            )
            future_b = rec.save_streaming(
                file_path=str(Path(tmpdir) / "b.mp4"),
                start_simulate_index=3,
                end_simulate_index=7,
            )
            # 两个任务均未触发（latest=2 < 5）
            self.assertFalse(future_a.done())
            self.assertFalse(future_b.done())

            # 模拟后续帧 3..7 到达
            for i in range(3, 8):
                rec._buffer.append(
                    simulate_index=i,
                    timestamp_ns=i * 1000,
                    nal_data=_make_p_nal(),
                    is_keyframe=False,
                )
                rec._dispatch_triggered_tasks(i)

            result_a = future_a.result(timeout=10)
            result_b = future_b.result(timeout=10)

        # 各自独立区间，互不覆盖
        self.assertEqual(result_a.frame_indices, [0, 1, 2, 3, 4, 5])
        self.assertEqual(result_b.frame_indices, [3, 4, 5, 6, 7])

    def test_save_future_completes_immediately_when_end_already_received(self):
        """端帧已到达时 ``save_streaming`` 立即触发。"""
        rec = VideoRecorder(camera_name="test_cam", color_port=7070)
        for i in range(3):
            rec._buffer.append(
                simulate_index=i,
                timestamp_ns=i * 1000,
                nal_data=_make_idr_nal() if i == 0 else _make_p_nal(),
                is_keyframe=(i == 0),
            )
        with TemporaryDirectory() as tmpdir:
            future = rec.save_streaming(
                file_path=str(Path(tmpdir) / "out.mp4"),
                start_simulate_index=0,
                end_simulate_index=2,
            )
            result = future.result(timeout=10)
        self.assertEqual(result.frame_count, 3)


if __name__ == "__main__":
    unittest.main()
