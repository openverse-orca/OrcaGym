"""相机录制器抽象基类。

``CameraRecorder`` 封装单路相机 H.264 流的通用录制逻辑：在守护线程中接收
WebSocket 码流，缓存到 ``RollingFrameBuffer``，通过 ``save_streaming`` 注册
区间保存任务，由保存 worker 线程使用 PyAV remux 保存为 MP4。每个区间任务
独立携带自己的 start/end，互不干扰。

color（``VideoRecorder``）与 depth（``DepthRecorder``）相机各自继承本基类，
仅通过 ``stream_kind`` 与端口属性区分，其余录制/缓存/保存逻辑完全复用，
保证 [[VideoRecorderManager]] 能以一致的接口统一管理。

Note:
    PyAV (``av``) 必须已安装，否则录制功能不可用（仅缓存可用）。
"""

from __future__ import annotations

import asyncio
import queue
import struct
import threading
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import websockets

if TYPE_CHECKING:
    import numpy as np

try:
    import av
except ImportError:
    av = None  # type: ignore[assignment]

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.recorder.recording_task import (
    DecodeTask,
    FrameCallback,
    RangeSaveTask,
    RecordingTask,
    RecordingTaskQueue,
    TriggerFn,
)
from orca_gym.recorder.rolling_frame_buffer import FrameEntry, RollingFrameBuffer

_logger = get_orca_logger()


# WebSocket 帧格式：[8 字节时间戳 uint64 LE][4 字节 simulate_index int32 LE][H.264 NAL 数据]
_WS_HEADER_SIZE = 12
_TS_OFFSET = 0
_SIM_IDX_OFFSET = 8
_NAL_OFFSET = 12

# H.264 NAL type 掩码：第 1 字节低 5 位
_NAL_TYPE_MASK = 0x1F
_NAL_TYPE_IDR = 5  # IDR 帧

# Annex-B 起始码（0x01 前面的连续 0 字节）
_START_CODE_3 = b"\x00\x00\x01"
_START_CODE_4 = b"\x00\x00\x00\x01"

# 保存 worker 线程的退出哨兵
_SAVE_WORKER_STOP = object()

# WebSocket 连接重试
_WS_CONNECT_RETRIES = 3
_WS_CONNECT_RETRY_INTERVAL = 1.0  # 秒

# start() 等待 WebSocket 连接建立的总超时（秒）。
# 引擎需要先开启物理仿真才能输出图像，这里给一个较宽松的超时。
_WS_CONNECT_TIMEOUT = 30.0
# 等待轮询间隔
_WS_CONNECT_POLL_INTERVAL = 0.1

# stop_all_and_save 等待保存 Future 完成的总超时（秒）
_SAVE_WAIT_TIMEOUT = 60.0


def _nal_data_is_keyframe(nal_data: bytes) -> bool:
    """判断 H.264 数据是否包含 IDR 关键帧（NAL type == 5）。

    引擎推送的编码数据为 **Annex-B 基本流**：以起始码 ``00 00 01``（或
    ``00 00 00 01``）分隔多个 NAL 单元。一帧编码数据常包含
    SPS/PPS + IDR slice 等多个 NAL，因此**必须扫描所有 NAL 单元的 type**，
    若任一 NAL type == 5（IDR 切片）即判定为关键帧。

    不能只检查首字节：Annex-B 数据以 ``0x00``（起始码）开头，
    且 SPS/PPS（type 7/8）可能排在 IDR 之前，首字节判断会漏判。

    Args:
        nal_data: 引擎推送的 H.264 NAL 数据（Annex-B 基本流）。

    Returns:
        是否包含 IDR 切片。
    """
    if not nal_data:
        return False

    pos = 0
    n = len(nal_data)
    while True:
        idx = nal_data.find(_START_CODE_3, pos)
        if idx == -1:
            break
        # 起始码后第一个字节即 NAL header（''00 00 01'' 的 01 之后）
        na_start = idx + 3
        if na_start < n and (nal_data[na_start] & _NAL_TYPE_MASK) == _NAL_TYPE_IDR:
            return True
        pos = na_start

    # 兜底：若无起始码，按"单 NAL 单元"处理（首字节即 NAL header）
    return (nal_data[0] & _NAL_TYPE_MASK) == _NAL_TYPE_IDR


@dataclass
class RemuxResult:
    """``remux_range`` 返回值，携带帧号 ↔ 物理 index 映射与输出路径。

    Attributes:
        file_path: 生成的 MP4 文件绝对路径。
        frame_count: 写入 MP4 的总帧数。
        frame_indices: 物理仿真步索引列表（按写入顺序），长度 == frame_count。
            每一项对应 MP4 中第 i 帧的 ``simulate_index``。
        timestamps_ns: 每帧的纳秒时间戳列表（按写入顺序），长度 == frame_count。
            用于校验和调试，也作为 PTS 的原始时间基。
    """

    file_path: str
    frame_count: int
    frame_indices: list[int] = field(default_factory=list)
    timestamps_ns: list[int] = field(default_factory=list)


class CameraRecorder(ABC):
    """单相机 H.264 流录制器抽象基类。

    在守护线程中接收 WebSocket H.264 码流，缓存到 RollingFrameBuffer。
    通过 ``save_streaming(path, start, end)`` 注册区间保存任务，端帧到达后
    由保存 worker 线程异步执行 PyAV remux 写入 MP4。

    子类只需提供：
        - ``stream_kind``：流类型标识（如 ``"color"`` / ``"depth"``）
        - 构造函数传入对应的推流端口

    生命周期:
        1. ``__init__``: 配置端口、缓存容量
        2. ``start()``: 连接 WebSocket，开始接收缓存（守护线程）
        3. ``save_streaming(path, start, end) -> Future[RemuxResult]``:
           注册区间保存任务并立即返回 ``Future``，目标帧到达后由保存 worker
           线程异步写入 MP4（不阻塞接收/上层线程）
        4. ``stop()``: 断开 WebSocket，清理资源
    """

    #: 流类型标识（color / depth 等），子类覆盖。
    stream_kind: str = "unknown"

    def __init__(
        self,
        camera_name: str,
        port: int,
        max_buffer_frames: int = 36000,
    ) -> None:
        """初始化录制器。

        Args:
            camera_name: 相机名称（用于日志和标识）
            port: WebSocket 推流端口
            max_buffer_frames: 滚动缓存最大帧数
        """
        self._camera_name = camera_name
        self._port = port
        self._buffer = RollingFrameBuffer(max_frames=max_buffer_frames)

        self._running = False
        self._thread: threading.Thread | None = None
        self._ws: websockets.WebSocketClientProtocol | None = None

        # 等待触发任务队列（保存流任务，线程安全，内部持锁）
        self._task_queue = RecordingTaskQueue()

        # 异步保存：待 remux 队列 + worker 线程
        # 接收线程轮询任务队列，触发后将任务放入 _save_queue（自带线程安全）
        self._save_queue: "queue.Queue[RecordingTask | object]" = queue.Queue()
        self._save_worker: threading.Thread | None = None

        # 解码状态（save_worker 线程独占写入；``last_decoded_frame`` property
        # 通过 ``SingleFrameTask.execute`` 在同一线程读取，FIFO 保证一致）
        # 持久 CodecContext 保持 DPB 参考帧状态，避免 P 帧回溯解码
        self._codec_ctx = None  # av.CodecContext，lazy init
        self._last_decoded_frame: "np.ndarray | None" = None  # 最近一次解码的 RGB 帧

        # 线程异常通知主线程
        self._thread_error: BaseException | None = None
        self._error_event = threading.Event()

        # WebSocket 连接状态（线程间共享，用 _ws_connected_lock 保护）
        self._ws_connected = False
        self._ws_connected_event = threading.Event()

        # 已收到首个 IDR 帧
        self._received_first_idr = False

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动 WebSocket 接收线程。

        前置条件：引擎侧已 ``SetStreamingEnabled(true)``。

        本方法会等待 WebSocket **连接建立**（不要求收到首帧），超时后抛出
        ``ConnectionError``。由于引擎需要先开启物理仿真才能输出图像，
        这里不要求首帧到达，给上层一个宽松的启动窗口。

        PyAV 缺失时仍允许启动接收线程（仅缓存可用），解码/remux 路径在
        实际调用时才 raise。符合模块顶部 Note "PyAV 未安装时仅录制不可用
        （缓存可用）"的降级契约。

        Raises:
            RuntimeError: 录制器已在运行
            ConnectionError: WebSocket 连接超时或失败
        """
        if self._running:
            raise RuntimeError(f"Recorder for camera '{self._camera_name}' is already running")

        if av is None:
            _logger.warning(
                f"[Recorder:{self._camera_name}] PyAV (av) is not installed; "
                f"receiver/buffer will work but decode/remux are disabled. "
                f"Install with: pip install av"
            )

        self._running = True
        self._received_first_idr = False
        self._thread_error = None
        self._error_event.clear()
        self._ws_connected = False
        self._ws_connected_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"CameraRecorder-{self._camera_name}",
            daemon=True,
        )
        self._thread.start()

        # 启动保存 worker 线程（负责 remux，不阻塞接收/上层线程）
        if self._save_worker is None or not self._save_worker.is_alive():
            self._save_worker = threading.Thread(
                target=self._save_worker_loop,
                name=f"CameraRecorderSaver-{self._camera_name}",
                daemon=True,
            )
            self._save_worker.start()

        # 等待 WebSocket 连接建立（或失败），不等待首帧
        deadline = time.monotonic() + _WS_CONNECT_TIMEOUT
        while time.monotonic() < deadline:
            # 连接已建立 → 成功
            if self._ws_connected_event.is_set():
                return
            # 线程已失败（重试耗尽或异常）
            if self._error_event.is_set():
                self._running = False
                raise ConnectionError(
                    f"Failed to connect WebSocket for camera '{self._camera_name}' "
                    f"on port {self._port}: {self._thread_error}"
                )
            time.sleep(_WS_CONNECT_POLL_INTERVAL)

        # 超时
        self._running = False
        raise ConnectionError(
            f"WebSocket connection to camera '{self._camera_name}' on port "
            f"{self._port} did not establish within "
            f"{_WS_CONNECT_TIMEOUT}s. Ensure engine simulation is running and "
            f"streaming is enabled."
        )

    def stop(self, save_pending: bool = False) -> None:
        """停止接收线程，断开 WebSocket。

        Args:
            save_pending: 是否将等待队列中未触发的任务"尽力保存"后退出。
                ``True``：将未触发任务全部移交 worker 保存（用于
                ``stop_all_and_save`` 场景）。
                ``False``：取消所有未触发的任务（用于不保存的停止）。

        缓存保留，可继续调用 ``extract_range``。
        """
        self._running = False

        if save_pending:
            self.flush_pending_saves()
        else:
            # 取消未触发的待保存任务
            for task in self._task_queue.drain():
                task.future.cancel()

        # 接收线程在 _receive_loop 中用 asyncio.wait_for(ws.recv(), timeout=1.0)
        # 轮询 _running 标志，设置 _running=False 后最多 1 秒内退出。
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                _logger.warning(
                    f"[Recorder:{self._camera_name}] "
                    f"receive thread did not exit within 5s"
                )
        self._thread = None
        self._ws = None
        self._ws_connected = False
        self._ws_connected_event.clear()

        # 通知保存 worker 退出（FIFO 队列保证已入队任务先处理完）
        if self._save_worker is not None and self._save_worker.is_alive():
            self._save_queue.put(_SAVE_WORKER_STOP)
            self._save_worker.join(timeout=10.0)
            if self._save_worker.is_alive():
                # worker 仍在执行长耗时 remux。**保留引用**避免
                # ``_ensure_save_worker`` 重复创建 worker 从同一队列取任务
                # （会导致两个 worker 竞争同一队列，任务被任意一方取走，
                # 且 ``_SAVE_WORKER_STOP`` 哨兵只被一方收到，另一方永不退出）。
                # daemon 线程会在进程退出时被回收。后续若再次 ``start()``，
                # ``_ensure_save_worker`` 检测到旧 worker 仍存活将复用它
                # （但注意：``_SAVE_WORKER_STOP`` 已入队，旧 worker 处理完
                # 当前任务后会取到哨兵并退出，之后新任务将无人处理——
                # 这是超时场景的已知限制，正常路径不受影响）。
                _logger.error(
                    f"[Recorder:{self._camera_name}] "
                    f"save worker did not exit within 10s, "
                    f"pending saves may be lost. Keeping worker reference "
                    f"to prevent duplicate creation."
                )
            else:
                self._save_worker = None
        else:
            self._save_worker = None

        # 清理解码状态（CodecContext 不可跨 start/stop 复用，DPB 状态可能已失效）
        self._codec_ctx = None
        self._last_decoded_frame = None

    # ------------------------------------------------------------------
    # 录制任务管理
    # ------------------------------------------------------------------

    def save_streaming(
        self,
        file_path: str,
        start_simulate_index: int,
        end_simulate_index: int,
        trigger_fn: TriggerFn | None = None,
        truncate_to_keyframe: bool = False,
    ) -> Future[RemuxResult]:
        """保存区间 ``[start, end]`` 为 MP4。**非阻塞**，返回 ``Future``。

        在等待任务队列中注册一个区间保存任务：当接收线程收到
        ``simulate_index >= end`` 的帧（通过触发回调判断）时，将任务移交
        保存 worker 线程执行 PyAV remux。每个任务独立携带自己的 start/end，
        可同时注册多个互不干扰的区间任务。

        内部构造 ``RangeSaveTask`` 并委托 ``submit_task``，保证立即触发
        判断与通用任务路径一致。

        Args:
            file_path: MP4 输出文件路径
            start_simulate_index: 区间起始（含）
            end_simulate_index: 区间结束（含）
            trigger_fn: 可选触发回调 ``(task, current_index) -> bool``。
                默认当 ``current_index >= end_simulate_index`` 时触发。
            truncate_to_keyframe: 保存时是否前向截断到区间内第一个关键帧，
                默认 ``False``（配合录制起点 ``request_idr=True`` 使用）。

        Returns:
            ``Future[RemuxResult]``：保存完成后 ``future.result()`` 返回
            ``RemuxResult``（``file_path`` / ``frame_count`` /
            ``frame_indices`` / ``timestamps_ns``）。

        Raises:
            ValueError: 区间内无帧（保存完成后通过 ``future.exception()`` 获取）
            av.FFmpegError: PyAV mux 失败（同上；PyAV 10+ 已用
                ``av.FFmpegError`` 替代旧版 ``av.AVError``）
        """
        task = RangeSaveTask(
            file_path=file_path,
            start_simulate_index=start_simulate_index,
            end_simulate_index=end_simulate_index,
            trigger_fn=trigger_fn,
            truncate_to_keyframe=truncate_to_keyframe,
        )
        self.submit_task(task)

        _logger.info(
            f"[Recorder:{self._camera_name}] save_streaming: "
            f"path={file_path}, range=[{start_simulate_index}, "
            f"{end_simulate_index}]"
        )
        return task.future

    def submit_task(self, task: RecordingTask) -> Future:
        """提交任意录制任务到等待队列。**非阻塞**。

        通用任务提交接口，与 ``save_streaming``（内部创建 ``RangeSaveTask``）
        解耦：调用方可构造任意 ``RecordingTask`` 子类（如 ``SingleFrameTask``）
        提交，recorder 只负责轮询触发和移交 save_worker 执行。

        若触发条件已满足（缓存最新 ``simulate_index`` 已达任务目标），
        立即派发到 save_worker，避免等待下一帧到达。

        Args:
            task: 录制任务实例（``RangeSaveTask`` / ``SingleFrameTask`` / 自定义）。

        Returns:
            ``task.future``：任务完成后可获取结果或异常。
        """
        self._ensure_save_worker()
        self._task_queue.add(task)
        # 若触发条件已满足（缓存最新 index 已达目标），立即派发
        latest = self._buffer.get_latest_simulate_index()
        if latest >= 0:
            self._dispatch_triggered_tasks(latest)
        return task.future

    def get_frame(self, simulate_index: int) -> FrameEntry | None:
        """从缓存获取指定 ``simulate_index`` 的单帧。

        供 ``SingleFrameTask.execute`` 调用。若该帧已被滚动淘汰或从未存在
        （渲染跳号），返回 None。

        Args:
            simulate_index: 目标帧的 simulate_index。

        Returns:
            ``FrameEntry`` 或 None。
        """
        return self._buffer.get_frame(simulate_index)

    def get_latest_simulate_index(self) -> int | None:
        """获取缓存中最新已到达帧的 ``simulate_index``。

        用于降频门控：上层判断是否有新视频帧到达，决定是否提交
        ``SingleFrameTask``。无帧到达时返回 None。

        Returns:
            最新帧的 ``simulate_index``，或 None。
        """
        idx = self._buffer.get_latest_simulate_index()
        return idx if idx >= 0 else None

    def decode_frame_at(self, simulate_index: int) -> "np.ndarray | None":
        """解码指定 ``simulate_index`` 的视频帧为 RGB numpy 数组。

        .. deprecated::
            新方案中 save_worker 维护持久 CodecContext 实时解码，
            ``SingleFrameTask.on_frame`` 回调直接接收解码帧，无需调用本方法。
            保留仅供向后兼容或调试使用。

        Args:
            simulate_index: 目标帧的 simulate_index。

        Returns:
            RGB numpy 数组 ``(H, W, 3)`` dtype=uint8，或 ``None``。
        """
        if av is None:
            return None
        target = self._buffer.get_frame(simulate_index)
        if target is None:
            return None
        # 兼容接口：单次解码（无 DPB 状态，P 帧可能解码失败）
        try:
            codec = av.CodecContext.create("h264", "r")
            packet = av.Packet(target.nal_data)
            for frame in codec.decode(packet):
                return frame.to_ndarray(format="rgb24")
        except av.FFmpegError:
            pass
        return None

    def flush_pending_saves(self) -> list[Future[RemuxResult]]:
        """取出所有未触发任务并加入保存队列，尽力保存。

        用于 ``stop_all_and_save`` / ``stop(save_pending=True)`` 场景：
        停流后不再有新帧，将等待队列中尚未触发的任务全部移交 worker 执行
        （用当前缓存已有帧保存）。

        Returns:
            所有已移交任务的 ``Future`` 列表。
        """
        tasks = self._task_queue.drain()
        futures = [t.future for t in tasks]
        for t in tasks:
            self._save_queue.put(t)
        return futures

    # ------------------------------------------------------------------
    # 异步保存任务队列（任务触发 → 移交 worker 执行）
    # ------------------------------------------------------------------

    def _dispatch_triggered_tasks(self, current_simulate_index: int) -> None:
        """将满足触发条件的任务移入保存队列（由 worker 执行）。

        在接收线程每收到一帧后调用：轮询任务队列，取出所有
        ``should_trigger(current_simulate_index)`` 为真的任务，放入保存队列。
        """
        for task in self._task_queue.poll_triggered(current_simulate_index):
            self._save_queue.put(task)

    def _ensure_save_worker(self) -> None:
        """确保保存 worker 线程已启动。"""
        if self._save_worker is None or not self._save_worker.is_alive():
            self._save_worker = threading.Thread(
                target=self._save_worker_loop,
                name=f"CameraRecorderSaver-{self._camera_name}",
                daemon=True,
            )
            self._save_worker.start()

    def _save_worker_loop(self) -> None:
        """保存 worker 线程主循环：从队列取任务并执行。

        队列内容统一为 ``RecordingTask`` 子类（``DecodeTask`` /
        ``RangeSaveTask`` / ``SingleFrameTask``），FIFO 保证执行顺序：
        接收线程先提交 ``DecodeTask``（解码 NAL + 更新 DPB 状态），
        再触发 ``SingleFrameTask``（回调内从 ``last_decoded_frame`` 取解码结果）。

        ``DecodeTask`` 轻量（~0.1-0.5ms），不阻塞后续任务；``RangeSaveTask``
        的 remux 较重，但通常在 episode 结束时才触发，与实时解码不冲突。
        """
        while True:
            item = self._save_queue.get()
            if item is _SAVE_WORKER_STOP:
                return
            task: RecordingTask = item
            try:
                task.execute(self)
            finally:
                self._save_queue.task_done()

    def _decode_nal(self, nal_data: bytes, simulate_index: int) -> None:
        """在 save_worker 线程解码一帧 NAL，更新 ``_last_decoded_frame``。

        使用持久 ``CodecContext`` 保持 DPB 参考帧状态，P 帧可直接解码
        无需回溯到 IDR。解码失败时保留上一帧（``_last_decoded_frame`` 不变）。

        Args:
            nal_data: H.264 NAL 单元数据（Annex-B 基本流）。
            simulate_index: 帧的 simulate_index（用于日志）。
        """
        if av is None:
            return
        try:
            if self._codec_ctx is None:
                self._codec_ctx = av.CodecContext.create("h264", "r")
            packet = av.Packet(nal_data)
            for frame in self._codec_ctx.decode(packet):
                self._last_decoded_frame = frame.to_ndarray(format="rgb24")
        except av.FFmpegError as e:
            _logger.debug(
                f"[Recorder:{self._camera_name}] decode error at "
                f"sim_idx={simulate_index}: {e}"
            )

    def remux_range(
        self,
        file_path: str,
        start_simulate_index: int,
        end_simulate_index: int,
        truncate_to_keyframe: bool = False,
    ) -> RemuxResult:
        """从缓存提取区间并 PyAV remux 写入 MP4。

        区间保存的公共实现，供 ``RangeSaveTask`` 在执行阶段调用（worker 线程）。
        使用 ``timestamp_ns`` 作为 PTS 时间基（不使用固定 FPS 时间基）：
        - ``out_stream.time_base`` 设为微秒（``Fraction(1, 1_000_000)``）
        - 每帧 ``pts = dts = timestamp_ns // 1000``
        - 首帧时间戳偏移到 0（``pts -= first_ts_us``）

        **IDR primer 机制**：若区间首帧不是 IDR（I 帧），回溯到
        ``start_simulate_index`` 之前最近的 IDR 帧作为解码器初始化帧
        （primer）。primer 帧写入文件最前面，使后续 P 帧可正确解码。
        primer 帧的 ``simulate_index`` 不包含在返回的 ``frame_indices`` 中
        （它属于上一段区间），但 ``frame_count`` 包含 primer。

        **前向截断（``truncate_to_keyframe=True``）**：丢弃区间首个关键帧
        之前的非关键帧，使视频从关键帧开始（避免以 P 帧开头导致解码花屏）。
        配合录制起点 ``request_idr=True`` 使用：起点附近输出的关键帧成为
        视频首帧。截断后首帧即关键帧，primer 机制自动跳过。

        Args:
            file_path: MP4 输出文件路径。
            start_simulate_index: 区间起始（含）。
            end_simulate_index: 区间结束（含）。
            truncate_to_keyframe: 是否前向截断到区间内第一个关键帧，默认
                ``False``（保持原有 primer 语义）。

        Returns:
            ``RemuxResult``：包含文件路径、帧数（含 primer）、
            每帧物理 index（不含 primer）和时间戳（不含 primer）。
        """
        frames = self._buffer.extract_range(start_simulate_index, end_simulate_index)
        if not frames:
            raise ValueError(
                f"No frames in buffer for range "
                f"[{start_simulate_index}, {end_simulate_index}]. "
                f"Buffer range: "
                f"[{self._buffer.get_oldest_simulate_index()}, "
                f"{self._buffer.get_latest_simulate_index()}]"
            )

        # 前向截断：丢弃首个关键帧之前的非关键帧，使视频从关键帧开始（截断后首帧即关键帧，primer 分支自动跳过）。
        # 复用 get_first_keyframe_after 前向查找首个关键帧。
        if truncate_to_keyframe:
            first_kf = self._buffer.get_first_keyframe_after(start_simulate_index)
            if first_kf is not None:
                # first_kf 必在 frames 内（>= start 且 extract_range 已含 [start, end]），按对象身份定位切片。
                first_kf_rel = next(
                    (i for i, f in enumerate(frames) if f is first_kf), None
                )
                if first_kf_rel is not None and first_kf_rel > 0:
                    _logger.info(
                        f"[Recorder:{self._camera_name}] truncate_to_keyframe: "
                        f"dropping {first_kf_rel} leading non-keyframe frame(s), "
                        f"first keyframe at sim_idx={first_kf.simulate_index}"
                    )
                    frames = frames[first_kf_rel:]

        if av is None:
            raise RuntimeError(
                "PyAV (av) is not installed. Install with: pip install av"
            )

        # 确保输出目录存在
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 收集每帧的物理 index 和时间戳（按写入顺序）
        frame_indices: list[int] = [f.simulate_index for f in frames]
        timestamps_ns: list[int] = [f.timestamp_ns for f in frames]

        # --- IDR primer 回溯 ---
        # 若区间首帧不是 IDR，回溯到 start 之前最近的 IDR 帧作为 primer。
        # primer 写入文件最前面，使解码器能正确初始化后续 P 帧。
        #
        # 两级回溯策略：
        #   1. 优先用 simulate_index 回溯（精确，但当 I 帧 sim_idx=-1 时可能失败）
        #   2. fallback 用 timestamp_ns 回溯（兼容 sim_idx=-1 的 I 帧）
        primer: FrameEntry | None = None
        if not frames[0].is_keyframe:
            # 1. 尝试 sim_idx 回溯
            primer = self._buffer.get_latest_keyframe_strictly_before(
                start_simulate_index
            )
            if primer is not None:
                _logger.info(
                    f"[Recorder:{self._camera_name}] primer (by sim_idx): "
                    f"first frame sim_idx={frames[0].simulate_index} is not IDR, "
                    f"using primer sim_idx={primer.simulate_index} "
                    f"(< {start_simulate_index})"
                )
            else:
                # 2. fallback: 用 timestamp 回溯
                # 当 I 帧 sim_idx=-1（引擎自动产生，uint32 溢出）时，
                # sim_idx 回溯无法找到，改用时间戳回溯。
                first_ts = frames[0].timestamp_ns
                primer = self._buffer.get_latest_keyframe_before_timestamp(first_ts)
                if primer is not None:
                    _logger.info(
                        f"[Recorder:{self._camera_name}] primer (by timestamp "
                        f"fallback): first frame sim_idx={frames[0].simulate_index} "
                        f"ts_ns={first_ts} is not IDR, using primer "
                        f"sim_idx={primer.simulate_index} ts_ns={primer.timestamp_ns}"
                    )
                else:
                    _logger.warning(
                        f"[Recorder:{self._camera_name}] no IDR primer found "
                        f"before start_simulate_index={start_simulate_index} "
                        f"ts_ns={first_ts}, video may have decode errors "
                        f"at the beginning"
                    )

        # 使用 timestamp_ns 作为 PTS 时间基（微秒）
        # 基准时间戳：primer 存在时用 primer 的，否则用首帧的
        if primer is not None:
            base_ts_ns = primer.timestamp_ns
        else:
            base_ts_ns = timestamps_ns[0] if timestamps_ns else 0

        output_container = av.open(str(output_path), mode="w")
        try:
            # 不指定 rate；实际帧率由 PTS（timestamp_ns）间隔决定
            out_stream = output_container.add_stream("h264")
            # time_base 使用微秒，避免纳秒溢出 Fraction 精度
            out_stream.time_base = Fraction(1, 1_000_000)

            # 写入 primer 帧（若存在）
            if primer is not None:
                packet = av.Packet(primer.nal_data)
                packet.is_keyframe = True
                pts_us = (primer.timestamp_ns - base_ts_ns) // 1000
                packet.pts = pts_us
                packet.dts = pts_us
                packet.stream = out_stream
                output_container.mux(packet)

            # 写入区间内的所有帧
            for entry in frames:
                packet = av.Packet(entry.nal_data)
                # 标记 IDR 帧
                if entry.is_keyframe:
                    packet.is_keyframe = True
                # PTS/DTS 使用微秒，相对基准偏移到 0
                pts_us = (entry.timestamp_ns - base_ts_ns) // 1000
                packet.pts = pts_us
                packet.dts = pts_us
                packet.stream = out_stream
                output_container.mux(packet)
        except Exception:
            # mux 失败，删除半成品文件
            output_container.close()
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            raise
        finally:
            output_container.close()

        # frame_count 包含 primer，frame_indices/timestamps_ns 不含 primer
        # （primer 属于上一段区间，不应出现在本段的物理 index 映射中）
        frame_count = len(frames) + (1 if primer is not None else 0)
        _logger.info(
            f"[Recorder:{self._camera_name}] saved {frame_count} frames "
            f"({len(frames)} + {'1 primer' if primer else 'no primer'}) to "
            f"{file_path}, range=[{start_simulate_index}, {end_simulate_index}], "
            f"base_ts_ns={base_ts_ns}"
        )
        return RemuxResult(
            file_path=str(output_path),
            frame_count=frame_count,
            frame_indices=frame_indices,
            timestamps_ns=timestamps_ns,
        )

    # ------------------------------------------------------------------
    # WebSocket 接收线程
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """线程入口：运行 asyncio 事件循环。"""
        try:
            asyncio.run(self._receive_loop())
        except Exception as e:
            self._thread_error = e
            self._error_event.set()
            _logger.error(
                f"[Recorder:{self._camera_name}] thread error: {e}\n"
                f"{traceback.format_exc()}"
            )
        finally:
            self._running = False

    async def _receive_loop(self) -> None:
        """WebSocket 接收 + 缓存主循环。"""
        uri = f"ws://localhost:{self._port}"
        last_error: BaseException | None = None

        for attempt in range(1, _WS_CONNECT_RETRIES + 1):
            if not self._running:
                return
            try:
                async with websockets.connect(uri) as ws:
                    self._ws = ws
                    # 连接已建立，通知主线程
                    self._ws_connected = True
                    self._ws_connected_event.set()
                    _logger.info(
                        f"[Recorder:{self._camera_name}] "
                        f"WebSocket connected to {uri}"
                    )
                    while self._running:
                        try:
                            data = await asyncio.wait_for(
                                ws.recv(), timeout=1.0
                            )
                        except asyncio.TimeoutError:
                            continue  # 正常超时，检查 _running 后继续 recv
                        self._process_frame(data)
                return
            except asyncio.TimeoutError:
                continue  # wait_for 超时已在内层处理，不会到达这里
            except Exception as e:
                last_error = e
                if attempt < _WS_CONNECT_RETRIES:
                    _logger.warning(
                        f"[Recorder:{self._camera_name}] "
                        f"WebSocket connect attempt {attempt}/{_WS_CONNECT_RETRIES} "
                        f"failed: {e}, retrying in {_WS_CONNECT_RETRY_INTERVAL}s..."
                    )
                    await asyncio.sleep(_WS_CONNECT_RETRY_INTERVAL)
                else:
                    _logger.error(
                        f"[Recorder:{self._camera_name}] "
                        f"WebSocket connect failed after {_WS_CONNECT_RETRIES} "
                        f"attempts: {e}"
                    )

        # 所有重试失败
        self._thread_error = ConnectionError(
            f"WebSocket connection to {uri} failed after "
            f"{_WS_CONNECT_RETRIES} attempts: {last_error}"
        )
        self._error_event.set()

    def _process_frame(self, data: bytes) -> None:
        """解析 WebSocket 帧并写入缓存。

        帧格式: [8B timestamp uint64 LE][4B simulate_index int32 LE][NAL data]
        """
        if len(data) < _WS_HEADER_SIZE:
            _logger.warning(
                f"[Recorder:{self._camera_name}] "
                f"frame too short ({len(data)} bytes), skipping"
            )
            return

        timestamp_ns = struct.unpack_from("<Q", data, _TS_OFFSET)[0]
        simulate_index = struct.unpack_from("<i", data, _SIM_IDX_OFFSET)[0]
        nal_data = data[_NAL_OFFSET:]

        # 判断 IDR 帧（扫描 Annex-B 起始码内的 NAL type==5）
        # 引擎推送的编码数据是 Annex-B 基本流（00 00 01 起始码分隔多个 NAL），
        # 一帧常含 SPS/PPS + IDR slice。不能只检查首字节，否则会漏判
        # 起始码开头或 SPS/PPS 前置的 IDR 帧。
        is_keyframe = _nal_data_is_keyframe(nal_data)
        if is_keyframe and not self._received_first_idr:
            self._received_first_idr = True
            _logger.info(
                f"[Recorder:{self._camera_name}] "
                f"first IDR received, sim_idx={simulate_index}"
            )

        self._buffer.append(
            simulate_index=simulate_index,
            timestamp_ns=timestamp_ns,
            nal_data=nal_data,
            is_keyframe=is_keyframe,
        )

        # 提交解码任务到 save_worker（维持持久 CodecContext DPB 状态）
        # DecodeTask 立即触发（提交时帧已到达），放入 _save_queue 后由
        # worker 解码并更新 _last_decoded_frame。FIFO 保证解码先于后续
        # SingleFrameTask 回调（同帧的回调任务在下方 _dispatch_triggered_tasks
        # 提交，排在解码任务之后）。
        self._save_queue.put(DecodeTask(nal_data, simulate_index))

        # 轮询任务队列，触发已满足条件的保存任务（当前 simulate_index 比较）
        self._dispatch_triggered_tasks(simulate_index)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def camera_name(self) -> str:
        """相机名称。"""
        return self._camera_name

    @property
    def port(self) -> int:
        """WebSocket 推流端口。"""
        return self._port

    @property
    def is_running(self) -> bool:
        """接收线程是否在运行。"""
        return self._running

    @property
    def is_connected(self) -> bool:
        """WebSocket 是否已建立连接（不代表已收到帧）。"""
        return self._ws_connected_event.is_set()

    @property
    def is_saving(self) -> bool:
        """是否有未完成触发的保存任务（等待队列中仍有待触发任务）。"""
        return self._task_queue.pending_count > 0

    @property
    def latest_simulate_index(self) -> int:
        """缓存中最新的 simulate_index。"""
        return self._buffer.get_latest_simulate_index()

    def read_frames(
        self, start_simulate_index: int, end_simulate_index: int
    ) -> list[FrameEntry]:
        """读取缓存中 ``[start, end]`` 闭区间内的帧（只读、不解码）。

        供可视化查看器等只读消费方使用。线程安全，不阻塞接收/保存线程，
        也不会影响 ``save_streaming`` 的区间提取。

        Args:
            start_simulate_index: 区间起始（含）。
            end_simulate_index: 区间结束（含）。

        Returns:
            按 ``simulate_index`` 升序排列的 ``FrameEntry`` 列表；
            区间内无帧返回空列表。
        """
        return self._buffer.extract_range(start_simulate_index, end_simulate_index)

    @property
    def buffered_frame_count(self) -> int:
        """当前缓存帧数。"""
        return len(self._buffer)

    @property
    def received_first_idr(self) -> bool:
        """是否已收到首个 IDR 帧。

        Note:
            ``_received_first_idr`` 在接收线程写入，本属性无锁。跨线程读取
            （如从主线程查 ``get_stats``）为最终一致，非强一致——可能短暂
            返回旧值 False，但不影响功能正确性（状态单调 False→True）。
        """
        return self._received_first_idr

    @property
    def last_decoded_frame(self) -> "np.ndarray | None":
        """save_worker 最近一次解码的 RGB numpy 数组 ``(H, W, 3)``。

        由持久 ``CodecContext`` 解码（保持 DPB 参考帧状态），FIFO 保证
        解码任务先于 ``SingleFrameTask`` 回调执行，回调可直接读取本属性
        获取解码帧，无需回溯到 IDR。

        Note:
            ``_last_decoded_frame`` 在 save_worker 线程写入，本属性无锁。
            仅限在 save_worker 线程内访问（如 ``SingleFrameTask.execute``
            回调），同线程 FIFO 保证读取一致。外部跨线程调用（如已废弃的
            ``VideoRecorderManager.get_last_decoded_frame``）存在无锁访问
            风险，不应在生产路径使用。
        """
        return self._last_decoded_frame

    def get_stats(self) -> dict:
        """返回录制器状态统计。"""
        return {
            "camera_name": self._camera_name,
            "port": self._port,
            "stream_kind": self.stream_kind,
            "is_running": self._running,
            "is_connected": self._ws_connected_event.is_set(),
            "is_saving": self.is_saving,
            "buffered_frames": len(self._buffer),
            "latest_simulate_index": self._buffer.get_latest_simulate_index(),
            "oldest_simulate_index": self._buffer.get_oldest_simulate_index(),
            "received_first_idr": self._received_first_idr,
            "has_thread_error": self._error_event.is_set(),
        }

    # ------------------------------------------------------------------
    # 子类可覆盖的扩展点（多态，避免 manager 层 isinstance 判断）
    # ------------------------------------------------------------------

    def update_params(self, **kwargs) -> None:
        """更新录制器专属参数（子类覆盖）。

        基类为空操作。``DepthRecorder`` 覆盖为更新 near_clip / far_clip / gamma。
        Manager 在复用已有录制器实例时调用，无需判断子类类型。
        """