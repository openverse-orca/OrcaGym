"""彩色相机录制器 + 多相机管理器（color/depth 统一管理）。

``VideoRecorder`` 是彩色（RGB）录制器，继承 [[CameraRecorder]] 复用通用
录制逻辑；深度相机由独立的 [[DepthRecorder]] 实现（见 ``depth_recorder.py``）。
两者接口一致，由 ``VideoRecorderManager`` 统一管理 color 与 depth 推流、
录制、保存与实时可视化。
"""

from __future__ import annotations

import asyncio
import traceback
from concurrent.futures import Future

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.protos import mjc_message_pb2
from orca_gym.recorder.camera_recorder import CameraRecorder, RemuxResult
from orca_gym.recorder.depth_recorder import DepthRecorder
from orca_gym.recorder.recording_task import RangeSaveTask, RecordingTask
from orca_gym.recorder.video_stream_viewer import VideoStreamViewer

_logger = get_orca_logger()

# stop_all_and_save 等待保存 Future 完成的总超时（秒）
_SAVE_WAIT_TIMEOUT = 60.0

#: 可通过 ``start_recorder(..., **kwargs)`` 直接写入相机属性（映射到
#: ``CameraProperty`` 字段）的键。记录器专属参数（如 ``max_buffer_frames``）
#: 不在此集合中，由 ``start_recorder`` 单独提取。
_CAMERA_PROP_KWARGS = frozenset(
    {
        "capture_rgb",
        "capture_depth",
        "capture_normal",
        "capture_object_color",
        "random_object_color",
        "use_nvenc",
        "nvenc_gpu_index",
        "width",
        "height",
        "vertical_fov",
        "near_clip",
        "far_clip",
        "gamma",
        "color_port",
        "depth_port",
        "use_dds",
        "dds_topic",
        "dds_stream_id",
    }
)

#: 流类型标识
_STREAM_COLOR = "color"
_STREAM_DEPTH = "depth"

#: 流类型 → 录制器子类的注册表（工厂注册模式，便于扩展新流类型）
_RECORDER_REGISTRY: dict[str, type[CameraRecorder]] = {
    _STREAM_COLOR: None,  # 延迟填充，见下方
    _STREAM_DEPTH: None,
}


class VideoRecorder(CameraRecorder):
    """彩色（RGB）H.264 流录制器。

    继承 [[CameraRecorder]] 的全部通用录制逻辑（WebSocket 接收、滚动缓存、
    区间保存、PyAV remux）。``stream_kind == "color"``。
    """

    stream_kind = _STREAM_COLOR

    def __init__(
        self,
        camera_name: str,
        color_port: int,
        max_buffer_frames: int = 36000,
    ) -> None:
        """初始化彩色录制器。

        Args:
            camera_name: 相机名称（用于日志和标识）
            color_port: RGB 流 WebSocket 推流端口
            max_buffer_frames: 滚动缓存最大帧数
        """
        super().__init__(
            camera_name=camera_name,
            port=color_port,
            max_buffer_frames=max_buffer_frames,
        )

    @property
    def color_port(self) -> int:
        """RGB 流 WebSocket 推流端口。"""
        return self._port


# 填充注册表（类定义完成后）
_RECORDER_REGISTRY[_STREAM_COLOR] = VideoRecorder
_RECORDER_REGISTRY[_STREAM_DEPTH] = DepthRecorder


def create_camera_recorder(
    camera_name: str,
    stream_kind: str,
    port: int,
    max_buffer_frames: int = 36000,
    *,
    near_clip: float | None = None,
    far_clip: float | None = None,
    gamma: float | None = None,
) -> CameraRecorder:
    """统一的录制器工厂函数（color/depth 共用）。

    按 ``stream_kind`` 从注册表中查找对应的录制器子类并创建。
    depth 流必须提供 ``near_clip`` / ``far_clip`` / ``gamma``。

    Args:
        camera_name: 相机名称。
        stream_kind: ``"color"`` 或 ``"depth"``。
        port: WebSocket 推流端口。
        max_buffer_frames: 滚动缓存最大帧数。
        near_clip: 近裁剪面距离（米），仅 depth 需要。
        far_clip: 远裁剪面距离（米），仅 depth 需要。
        gamma: gamma 校正值，仅 depth 需要。

    Returns:
        新创建的 ``CameraRecorder`` 子类实例（未启动）。

    Raises:
        ValueError: ``stream_kind`` 不合法，或 depth 缺少必要参数。
    """
    cls = _RECORDER_REGISTRY.get(stream_kind)
    if cls is None:
        raise ValueError(
            f"Unknown stream_kind '{stream_kind}', expected 'color' or 'depth'"
        )
    if stream_kind == _STREAM_DEPTH:
        if near_clip is None or far_clip is None or gamma is None:
            raise ValueError(
                f"DepthRecorder requires near_clip, far_clip, and gamma, "
                f"got near_clip={near_clip}, far_clip={far_clip}, gamma={gamma}"
            )
        return DepthRecorder(
            camera_name=camera_name,
            depth_port=port,
            near_clip=near_clip,
            far_clip=far_clip,
            gamma=gamma,
            max_buffer_frames=max_buffer_frames,
        )
    return VideoRecorder(
        camera_name=camera_name,
        color_port=port,
        max_buffer_frames=max_buffer_frames,
    )


class VideoRecorderManager:
    """多相机录制管理器（color / depth 统一管理）。

    管理多个录制器（``VideoRecorder`` / ``DepthRecorder``），以一致的接口
    提供录制、保存与实时可视化。内部使用 ``(camera_name, stream_kind)``
    复合键统一存储，所有方法通过 ``stream_kind`` 参数区分 color/depth，
    不再有对称重复方法。

    子类差异通过多态实现（``update_params``），Manager
    不包含任何 ``isinstance`` 判断。

    公共接口：
        - ``start_recorder``：按相机配置自动启动 color/depth 录制器
        - ``save_streaming``：非阻塞保存（通过 ``stream_kind`` 区分）
        - ``start_viewer``：非阻塞实时可视化
        - ``stop_all_and_save``：env.close() 时自动保存所有未完成任务
        - ``start_all`` / ``stop_all``：批量启停接收线程
    """

    def __init__(self, stub=None, loop=None) -> None:
        # 复合键 (camera_name, stream_kind) -> CameraRecorder
        self._recorders: dict[tuple[str, str], CameraRecorder] = {}
        # 复合键 (camera_name, stream_kind) -> VideoStreamViewer
        self._viewers: dict[tuple[str, str], VideoStreamViewer] = {}
        # 复合键 -> 最近一次 save_streaming 的 Future
        self._last_results: dict[tuple[str, str], Future[RemuxResult]] = {}
        self._stub = stub
        self._loop = loop

    # ------------------------------------------------------------------
    # 录制器获取
    # ------------------------------------------------------------------

    def get_recorder(
        self, camera_name: str, stream_kind: str = _STREAM_COLOR
    ) -> CameraRecorder | None:
        """获取指定相机的录制器。不存在返回 None。"""
        return self._recorders.get((camera_name, stream_kind))

    def remove_recorder(self, camera_name: str) -> None:
        """移除并停止指定相机的全部录制器及可视化窗口。"""
        for kind in (_STREAM_COLOR, _STREAM_DEPTH):
            recorder = self._recorders.pop((camera_name, kind), None)
            if recorder is not None:
                recorder.stop()
            viewer = self._viewers.pop((camera_name, kind), None)
            if viewer is not None:
                viewer.stop()
            self._last_results.pop((camera_name, kind), None)

    # ------------------------------------------------------------------
    # 启动录制
    # ------------------------------------------------------------------

    def start_recorder(
        self, camera_name: str, **kwargs
    ) -> list[CameraRecorder]:
        """按相机实际配置启动录制（color/depth 依据传感器开关自动创建）。

        流程（后端 gRPC 可用时）：
            1. 通过 ``get_camera_properties`` 查询相机属性，确认相机存在；
               相机不存在则抛出 ``ValueError``。
            2. 校验 ``kwargs`` 与当前相机属性是否一致。若不一致，先暂停串流
               （属性仅在 Idle 态可改），按 ``kwargs`` 重新设置。
            3. 开启串流（``set_streaming_enabled(True)``），确保 WebSocket
               端口已就绪。
            4. 依据 ``capture_rgb`` / ``capture_depth`` 创建并启动对应的
               录制器。若两者均未启用，输出警告并返回空列表。

        Args:
            camera_name: 相机名称。
            **kwargs: 相机属性键（见 ``_CAMERA_PROP_KWARGS``）+ recorder
                专属键 ``max_buffer_frames``。

        Returns:
            已启动的录制器列表（color 和/或 depth）；均未启用返回空列表。

        Raises:
            ValueError: 相机不存在，或启用了传感器但缺少有效端口。
            ConnectionError: WebSocket 连接失败。
        """
        max_buffer_frames = kwargs.pop("max_buffer_frames", 36000)
        prop_kwargs = {k: v for k, v in kwargs.items() if k in _CAMERA_PROP_KWARGS}

        props = None
        if self._stub is not None:
            try:
                props = self.get_camera_properties(camera_name)
            except RuntimeError as e:
                raise ValueError(
                    f"Camera '{camera_name}' not found or not registered on the "
                    f"backend: {e}"
                ) from e

        if props is not None:
            self._ensure_camera_config(camera_name, props, prop_kwargs)
            # 确保串流已开启（WebSocket 端口就绪后才能连接 Recorder）
            props = self.get_camera_properties(camera_name)
            if not props.streaming_enabled:
                self.set_streaming_enabled(camera_name, True)

        # 确定要启动的流（无后端时从 kwargs 推断）
        if props is not None:
            want_color = props.capture_rgb
            want_depth = props.capture_depth
        else:
            want_color = "color_port" in kwargs
            want_depth = "depth_port" in kwargs

        if not want_color and not want_depth:
            _logger.warning(
                f"Camera '{camera_name}': no stream sensors enabled "
                f"(capture_rgb=False, capture_depth=False). "
                f"No recorder started."
            )
            return []

        recorders: list[CameraRecorder] = []
        if want_color:
            port = kwargs.get("color_port") or (props.color_port if props else None)
            recorders.append(
                self._ensure_recorder(
                    camera_name, _STREAM_COLOR, port, max_buffer_frames
                )
            )
        if want_depth:
            port = kwargs.get("depth_port") or (props.depth_port if props else None)
            recorders.append(
                self._ensure_recorder(
                    camera_name, _STREAM_DEPTH, port, max_buffer_frames, kwargs=kwargs
                )
            )
        return recorders

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    def save_streaming(
        self,
        camera_name: str,
        file_path: str,
        start_simulate_index: int,
        end_simulate_index: int,
        stream_kind: str = _STREAM_COLOR,
        truncate_to_keyframe: bool = False,
    ) -> Future[RemuxResult]:
        """保存某相机指定流 ``[start, end]`` 区间为 MP4。**非阻塞**。

        内部构造 ``RangeSaveTask`` 并委托 ``submit_task``，保证任务派发
        路径与通用任务一致（立即触发判断、worker 移交等）。

        Args:
            camera_name: 相机名称
            file_path: MP4 输出文件路径
            start_simulate_index: 区间起始（含）
            end_simulate_index: 区间结束（含）
            stream_kind: ``"color"``（默认）或 ``"depth"``。
            truncate_to_keyframe: 是否前向截断到区间内第一个关键帧。

        Returns:
            ``Future[RemuxResult]``。

        Raises:
            ValueError: 录制器不存在。
        """
        recorder = self._require_recorder(camera_name, stream_kind)
        task = RangeSaveTask(
            file_path=file_path,
            start_simulate_index=start_simulate_index,
            end_simulate_index=end_simulate_index,
            truncate_to_keyframe=truncate_to_keyframe,
        )
        future = recorder.submit_task(task)
        self._last_results[(camera_name, stream_kind)] = future
        return future

    def submit_task(
        self,
        camera_name: str,
        task: RecordingTask,
        stream_kind: str = _STREAM_COLOR,
    ) -> Future:
        """提交任意录制任务到指定相机的等待队列。**非阻塞**。

        解耦任务构造与 recorder 工作线程：调用方自行构造 ``task``（如
        ``SingleFrameTask`` 用于逐帧回调），Manager 只负责路由到对应
        recorder。recorder 在帧到达时触发任务，移交 save_worker 执行。

        与 ``save_streaming`` 的区别：``save_streaming`` 内部创建
        ``RangeSaveTask`` 并固定返回 ``Future[RemuxResult]``；本方法接受
        任意 ``RecordingTask`` 子类，返回 ``task.future``（结果类型由
        具体任务决定）。

        Args:
            camera_name: 相机名称。
            task: 录制任务实例。
            stream_kind: ``"color"``（默认）或 ``"depth"``。

        Returns:
            ``task.future``：任务完成后可获取结果或异常。

        Raises:
            ValueError: 录制器不存在。
        """
        recorder = self._require_recorder(camera_name, stream_kind)
        return recorder.submit_task(task)

    def get_latest_frame_simulate_index(
        self,
        camera_name: str,
        stream_kind: str = _STREAM_COLOR,
    ) -> int | None:
        """获取指定相机最新已到达帧的 ``simulate_index``。

        用于降频门控：上层（如 OrcaManipulation 的 LeRobotDataStorage）
        判断是否有新视频帧到达，决定是否提交 ``SingleFrameTask``。

        Args:
            camera_name: 相机名称。
            stream_kind: ``"color"``（默认）或 ``"depth"``。

        Returns:
            最新帧的 ``simulate_index``，或 ``None``（无帧到达/录制器不存在）。
        """
        recorder = self._recorders.get((camera_name, stream_kind))
        if recorder is None:
            return None
        return recorder.get_latest_simulate_index()

    def get_last_decoded_frame(
        self,
        camera_name: str,
        stream_kind: str = _STREAM_COLOR,
    ):
        """获取指定相机最近一次解码的 RGB numpy 数组。

        .. deprecated::
            多相机同步采集应改为对每个相机提交 ``SingleFrameTask``（目标
            sim_idx 相同），通过 ``future.result()`` 收集各相机解码帧。
            直接读本属性存在跨 recorder 无锁访问 + sim_idx 不可控问题
            （读到"过新"或"过旧"帧）。保留仅供调试或单相机场景使用。

        Args:
            camera_name: 相机名称。
            stream_kind: ``"color"``（默认）或 ``"depth"``。

        Returns:
            RGB numpy 数组 ``(H, W, 3)`` dtype=uint8，或 ``None``
            （录制器不存在或尚未解码任何帧）。
        """
        recorder = self._recorders.get((camera_name, stream_kind))
        if recorder is None:
            return None
        return recorder.last_decoded_frame

    def decode_frame_at(
        self,
        camera_name: str,
        simulate_index: int,
        stream_kind: str = _STREAM_COLOR,
    ):
        """解码指定相机 ``simulate_index`` 的视频帧为 RGB numpy 数组。

        .. deprecated::
            新方案中 save_worker 维护持久 CodecContext 实时解码，
            ``SingleFrameTask.on_frame`` 回调直接接收解码帧，无需调用本方法。
            保留仅供向后兼容或调试使用。

        Args:
            camera_name: 相机名称。
            simulate_index: 目标帧的 simulate_index。
            stream_kind: ``"color"``（默认）或 ``"depth"``。

        Returns:
            RGB numpy 数组 ``(H, W, 3)`` dtype=uint8，或 ``None``。
        """
        recorder = self._recorders.get((camera_name, stream_kind))
        if recorder is None:
            return None
        return recorder.decode_frame_at(simulate_index)

    def get_last_result(
        self, camera_name: str, stream_kind: str = _STREAM_COLOR
    ) -> Future[RemuxResult] | None:
        """获取某相机指定流最近一次保存的 ``Future``。"""
        return self._last_results.get((camera_name, stream_kind))

    # ------------------------------------------------------------------
    # 实时可视化
    # ------------------------------------------------------------------

    def start_viewer(
        self,
        camera_name: str,
        window_name: str | None = None,
        stream_kind: str = _STREAM_COLOR,
    ) -> VideoStreamViewer:
        """启动某相机指定流的实时可视化窗口。**非阻塞**。

        深度流直接显示原始灰度帧，不做伪彩色转换
        （``DepthRecorder.grayscale_to_depth`` / ``depth_to_visualization``
        保留供程序化使用，如离线分析）。

        Args:
            camera_name: 相机名称（须已存在对应录制器）。
            window_name: 窗口标题；默认 ``Camera: <相机名>`` / ``Depth: <相机名>``。
            stream_kind: ``"color"``（默认）或 ``"depth"``。

        Raises:
            ValueError: 录制器不存在。
        """
        recorder = self._require_recorder(camera_name, stream_kind)
        key = (camera_name, stream_kind)
        viewer = self._viewers.get(key)
        if viewer is None or not viewer.is_running:
            default_name = (
                f"Depth: {camera_name}" if stream_kind == _STREAM_DEPTH
                else f"Camera: {camera_name}"
            )
            viewer = VideoStreamViewer(
                recorder,
                window_name=window_name or default_name,
            )
            viewer.start()
            self._viewers[key] = viewer
        return viewer

    def get_viewer(
        self, camera_name: str, stream_kind: str = _STREAM_COLOR
    ) -> VideoStreamViewer | None:
        """获取某相机指定流的查看器。未启动返回 None。"""
        return self._viewers.get((camera_name, stream_kind))

    def stop_viewer(
        self, camera_name: str, stream_kind: str = _STREAM_COLOR
    ) -> None:
        """停止某相机指定流的可视化窗口。不存在时静默。"""
        viewer = self._viewers.pop((camera_name, stream_kind), None)
        if viewer is not None:
            viewer.stop()

    def stop_all_viewers(self) -> None:
        """停止所有可视化窗口。"""
        for viewer in self._viewers.values():
            viewer.stop()
        self._viewers.clear()

    def get_viewer_stats(
        self, stream_kind: str | None = None
    ) -> dict[str, dict]:
        """返回可视化窗口的状态统计。

        Args:
            stream_kind: ``None`` 返回全部（键 ``"<cam>__<kind>"``）；
                指定类型只返回该类型（键为 ``camera_name``）。
        """
        return self._filter_by_kind(self._viewers, stream_kind)

    # ------------------------------------------------------------------
    # 相机属性查询/设置 + 推流状态机
    # ------------------------------------------------------------------

    def get_camera_names(self) -> list[str]:
        """获取所有已注册相机名称列表。无后端返回空列表。"""
        if self._stub is None:
            return []
        request = mjc_message_pb2.GetCameraNamesRequest()
        resp = self._run_async(self._stub.GetCameraNames(request))
        if resp.status != mjc_message_pb2.GetCameraNamesResponse.SUCCESS:
            raise RuntimeError(f"GetCameraNames failed: {resp.error_message}")
        return list(resp.camera_names)

    def get_camera_properties(self, camera_name: str):
        """查询相机属性 + 推流状态。无后端返回默认实例。"""
        if self._stub is None:
            return mjc_message_pb2.GetCameraPropertiesResponse()
        request = mjc_message_pb2.GetCameraPropertiesRequest(camera_name=camera_name)
        resp = self._run_async(self._stub.GetCameraProperties(request))
        if resp.status != mjc_message_pb2.GetCameraPropertiesResponse.SUCCESS:
            raise RuntimeError(f"GetCameraProperties failed: {resp.error_message}")
        return resp

    def set_camera_properties(self, camera_name: str, **kwargs) -> None:
        """批量设置相机属性。仅在 Idle 状态允许。"""
        if self._stub is None:
            return
        prop = mjc_message_pb2.CameraProperty()
        for key, val in kwargs.items():
            if val is None:
                continue
            if hasattr(prop, key):
                setattr(prop, key, val)
        request = mjc_message_pb2.SetCameraPropertiesRequest(
            camera_name=camera_name, property=prop
        )
        resp = self._run_async(self._stub.SetCameraProperties(request))
        if resp.status != mjc_message_pb2.SetCameraPropertiesResponse.SUCCESS:
            raise RuntimeError(f"SetCameraProperties failed: {resp.error_message}")

    def set_streaming_enabled(self, camera_name: str, enabled: bool) -> None:
        """切换推流状态（Idle ↔ Streaming）。"""
        if self._stub is None:
            return
        request = mjc_message_pb2.SetStreamingEnabledRequest(
            camera_name=camera_name, enabled=enabled
        )
        resp = self._run_async(self._stub.SetStreamingEnabled(request))
        if resp.status != mjc_message_pb2.SetStreamingEnabledResponse.SUCCESS:
            raise RuntimeError(f"SetStreamingEnabled failed: {resp.error_message}")

    # ------------------------------------------------------------------
    # 批量操作
    # ------------------------------------------------------------------

    def stop_recorder(self, camera_name: str) -> None:
        """停止某相机的全部接收线程（不保存未完成任务）。"""
        for kind in (_STREAM_COLOR, _STREAM_DEPTH):
            recorder = self._recorders.get((camera_name, kind))
            if recorder is not None:
                recorder.stop()

    def start_all(self) -> None:
        """启动所有已创建的录制器。"""
        for recorder in self._recorders.values():
            if not recorder.is_running:
                recorder.start()

    def stop_all(self) -> None:
        """停止所有录制器与可视化窗口（不保存未完成任务）。"""
        self.stop_all_viewers()
        for recorder in self._recorders.values():
            recorder.stop()

    def stop_all_and_save(self) -> dict[str, RemuxResult]:
        """停止所有录制器并保存未完成任务。**阻塞等待**保存完成。

        Returns:
            ``{"<camera_name>__<stream_kind>__<n>": RemuxResult}`` 保存成功的映射。
            一个 recorder 若有多个 pending save，键名后缀 ``__0``、``__1``。
        """
        # 收集所有 pending futures（一个 recorder 可能有多个未触发任务）
        pending_futures: list[tuple[str, Future[RemuxResult]]] = []
        for (camera_name, kind), recorder in self._recorders.items():
            futures = recorder.flush_pending_saves()
            for idx, fut in enumerate(futures):
                dkey = f"{camera_name}__{kind}__{idx}"
                pending_futures.append((dkey, fut))
                if idx == len(futures) - 1:
                    self._last_results[(camera_name, kind)] = fut

        self.stop_all_viewers()
        for recorder in self._recorders.values():
            recorder.stop()

        saved: dict[str, RemuxResult] = {}
        for key, future in pending_futures:
            try:
                saved[key] = future.result(timeout=_SAVE_WAIT_TIMEOUT)
            except Exception as e:
                _logger.error(
                    f"[RecorderManager] failed to resolve save for '{key}': {e}\n"
                    f"{traceback.format_exc()}"
                )
        return saved

    # ------------------------------------------------------------------
    # 批量查询
    # ------------------------------------------------------------------

    def get_stats(self, stream_kind: str | None = None) -> dict[str, dict]:
        """返回录制器状态统计。

        Args:
            stream_kind: ``None`` 返回全部（键 ``"<cam>__<kind>"``）；
                指定类型只返回该类型（键为 ``camera_name``）。
        """
        return self._filter_by_kind(self._recorders, stream_kind)

    def is_saving_any(self) -> bool:
        """是否有任何录制器正在进行保存任务。"""
        return any(r.is_saving for r in self._recorders.values())

    @property
    def recorder_count(self) -> int:
        """已注册的录制器总数。"""
        return len(self._recorders)

    @property
    def camera_names(self) -> list[str]:
        """已注册的相机名列表（去重）。"""
        return list({cam for cam, _ in self._recorders})

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _require_recorder(
        self, camera_name: str, stream_kind: str = _STREAM_COLOR
    ) -> CameraRecorder:
        """获取指定类型的录制器，不存在则抛 ValueError。"""
        recorder = self._recorders.get((camera_name, stream_kind))
        if recorder is None:
            raise ValueError(
                f"No {stream_kind} recorder for camera '{camera_name}'. "
                "Call start_recorder / start_streaming first."
            )
        return recorder

    def _ensure_camera_config(
        self, camera_name: str, props, prop_kwargs: dict
    ) -> None:
        """校验 kwargs 与相机属性一致性，必要时停流重配。

        属性变更需在 Idle 状态进行。若当前正在串流且属性需要变更，
        先停流再设置属性。**不负责重新开启串流**——由调用方
        （``start_recorder``）在配置完成后统一开启。
        """
        if not prop_kwargs:
            return
        changed = any(
            getattr(props, key, None) != val
            for key, val in prop_kwargs.items()
        )
        if changed:
            if props.streaming_enabled:
                self.set_streaming_enabled(camera_name, False)
            self.set_camera_properties(camera_name, **prop_kwargs)

    def _ensure_recorder(
        self,
        camera_name: str,
        stream_kind: str,
        port: int | None,
        max_buffer_frames: int,
        *,
        kwargs: dict | None = None,
    ) -> CameraRecorder:
        """统一的录制器获取/创建/启动逻辑（color/depth 共用，多态）。

        若已有录制器且端口变化则停止重建；已有实例时通过多态
        ``update_params`` 更新参数（depth 更新 near/far/gamma，color 空操作）。
        最后确保录制器已启动。

        Args:
            camera_name: 相机名称。
            stream_kind: ``"color"`` 或 ``"depth"``。
            port: WebSocket 推流端口。
            max_buffer_frames: 滚动缓存最大帧数。
            kwargs: 原始 kwargs（depth 从中提取 near_clip / far_clip / gamma，
                或从后端查询）。color 不使用。
        """
        if not port or port <= 0:
            raise ValueError(
                f"Cannot determine a valid {stream_kind}_port for camera "
                f"'{camera_name}'. Provide {stream_kind}_port in kwargs or "
                f"ensure the camera has one set."
            )

        # depth 流需要额外的深度参数（从 kwargs 或后端获取）
        depth_params: dict = {}
        if stream_kind == _STREAM_DEPTH and kwargs is not None:
            depth_params = self._resolve_depth_params(camera_name, kwargs)

        key = (camera_name, stream_kind)
        existing = self._recorders.get(key)

        # 端口变化则停止旧实例并重建
        if existing is not None and existing.port != port:
            existing.stop()
            del self._recorders[key]
            existing = None

        if existing is None:
            existing = create_camera_recorder(
                camera_name=camera_name,
                stream_kind=stream_kind,
                port=port,
                max_buffer_frames=max_buffer_frames,
                **depth_params,
            )
            self._recorders[key] = existing
        elif depth_params:
            # 已有实例，通过多态更新参数
            existing.update_params(**depth_params)

        if not existing.is_running:
            existing.start()
        return existing

    def _resolve_depth_params(
        self, camera_name: str, kwargs: dict
    ) -> dict:
        """从 kwargs 或后端属性解析 depth 参数（near_clip / far_clip / gamma）。"""
        near_clip = kwargs.get("near_clip")
        far_clip = kwargs.get("far_clip")
        gamma = kwargs.get("gamma")
        if (near_clip is None or far_clip is None or gamma is None) and self._stub is not None:
            props = self.get_camera_properties(camera_name)
            if near_clip is None:
                near_clip = props.near_clip
            if far_clip is None:
                far_clip = props.far_clip
            if gamma is None:
                gamma = props.gamma

        if near_clip is None or far_clip is None or gamma is None:
            raise ValueError(
                f"Cannot determine depth params (near_clip/far_clip/gamma) "
                f"for camera '{camera_name}'. Provide them in kwargs or "
                "ensure the camera is registered with a gRPC stub."
            )
        return {
            "near_clip": near_clip,
            "far_clip": far_clip,
            "gamma": gamma,
        }

    def _filter_by_kind(
        self, source: dict[tuple[str, str], object], stream_kind: str | None
    ) -> dict[str, dict]:
        """从复合键 dict 中提取统计，按 stream_kind 过滤。"""
        if stream_kind is None:
            return {
                f"{cam}__{kind}": obj.get_stats()  # type: ignore[attr-defined]
                for (cam, kind), obj in source.items()
            }
        return {
            cam: obj.get_stats()  # type: ignore[attr-defined]
            for (cam, kind), obj in source.items()
            if kind == stream_kind
        }

    def _run_async(self, coro):
        """同步桥接后端的异步接口。"""
        loop = self._loop
        if loop is None:
            return asyncio.run(coro)
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        return loop.run_until_complete(coro)


def CreateVideoRecorderManager(stub=None, loop=None) -> VideoRecorderManager:
    """创建 ``VideoRecorderManager`` 实例。"""
    return VideoRecorderManager(stub=stub, loop=loop)
