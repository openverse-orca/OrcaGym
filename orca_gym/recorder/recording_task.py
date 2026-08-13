"""录制任务抽象与等待触发队列。

``VideoRecorder`` 使用等待队列缓存待触发的录制任务。接收线程每收到一帧，
用当前 ``simulate_index`` 调用任务的触发回调判断是否满足触发条件；满足时
将任务移交保存 worker 线程执行。

将「任务」与「触发条件」抽象，便于后续扩展新的任务类型（如按时间戳、
按帧数触发的任务），同时保证每个区间任务独立携带自己的 start/end，
互不干扰（解决早期实现中单一 ``_task`` 被后续任务覆盖导致区间错乱的问题）。
"""

from __future__ import annotations

import threading
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from orca_gym.log.orca_log import get_orca_logger

if TYPE_CHECKING:
    import numpy as np

    from orca_gym.recorder.camera_recorder import CameraRecorder
    from orca_gym.recorder.rolling_frame_buffer import FrameEntry

_logger = get_orca_logger()

#: 触发回调类型：``(task, current_simulate_index) -> bool``。
#: ``current_simulate_index`` 为接收线程当前收到的 simulate_index，
#: 触发判断必须与之比较。
TriggerFn = Callable[["RecordingTask", int], bool]

#: 单帧回调类型：``(frame_entry, decoded_frame) -> Any``。在 save_worker 线程调用，
#: 接收目标帧的 ``FrameEntry``（含 timestamp_ns / simulate_index / nal_data）
#: 和解码后的 RGB numpy 数组 ``(H, W, 3)`` dtype=uint8（由 save_worker 的持久
#: CodecContext 解码，无需回溯）。``decoded_frame`` 为 None 表示解码失败。
#: 回调返回值会作为 ``SingleFrameTask.future`` 的结果（``set_result``），
#: 供多相机同步场景通过 future 收集回调产出（如返回 decoded_frame 供主相机合并）。
FrameCallback = Callable[["FrameEntry", "np.ndarray | None"], Any]

_TTask = TypeVar("_TTask", bound="RecordingTask")


class RecordingTask(ABC, Generic[_TTask]):
    """录制任务抽象基类。

    每个任务独立携带触发回调（``trigger_fn``）与执行逻辑（``execute``）。
    放入等待队列后，由接收线程逐帧轮询触发；触发后移交 worker 线程执行。

    Attributes:
        future: 任务完成结果（``Future``）。执行完成后填充结果或异常。
    """

    def __init__(self, trigger_fn: TriggerFn) -> None:
        """初始化任务。

        Args:
            trigger_fn: 触发回调 ``(task, current_simulate_index) -> bool``。
                接收线程每收到一帧即调用，判断当前帧是否满足触发条件。
        """
        if not callable(trigger_fn):
            raise TypeError("trigger_fn must be callable")
        self._trigger_fn: TriggerFn = trigger_fn
        self._future: Future = Future()

    @property
    def future(self) -> Future:
        """任务完成结果。"""
        return self._future

    def should_trigger(self, current_simulate_index: int) -> bool:
        """当前帧是否满足触发条件。

        Args:
            current_simulate_index: 接收线程当前收到的 simulate_index。
        """
        return self._trigger_fn(self, current_simulate_index)

    @abstractmethod
    def execute(self, recorder: CameraRecorder) -> None:
        """触发后由 worker 线程执行；结果写入 ``self.future``。

        Args:
            recorder: 所属 ``CameraRecorder``，用于访问帧缓存等资源。
        """
        raise NotImplementedError


def _default_range_trigger(
    task: "RecordingTask", current_simulate_index: int
) -> bool:
    """``RangeSaveTask`` 默认触发条件：当前 index 达到 end。

    触发条件宽松（``>=`` 而非 ``==``），以容忍「物理仿真步 → 引擎渲染 →
    取帧」延迟导致的跳号/重复。
    """
    return current_simulate_index >= task.end_simulate_index  # type: ignore[attr-defined]


class RangeSaveTask(RecordingTask["RangeSaveTask"]):
    """保存 ``[start, end]`` 区间为 MP4 的录制任务。"""

    def __init__(
        self,
        file_path: str,
        start_simulate_index: int,
        end_simulate_index: int,
        trigger_fn: TriggerFn | None = None,
        truncate_to_keyframe: bool = False,
    ) -> None:
        """初始化区间保存任务。

        Args:
            file_path: MP4 输出文件路径。
            start_simulate_index: 区间起始（含）。
            end_simulate_index: 区间结束（含）。
            trigger_fn: 可选触发回调，默认当 ``current_index >= end`` 时触发。
            truncate_to_keyframe: 保存时是否前向截断到第一个关键帧。
                为 ``True`` 时丢弃区间首个关键帧之前的非关键帧，使视频从
                关键帧开始（配合录制起点 ``request_idr=True`` 使用）。
        """
        super().__init__(trigger_fn or _default_range_trigger)
        self.file_path = file_path
        self.start_simulate_index = start_simulate_index
        self.end_simulate_index = end_simulate_index
        self.truncate_to_keyframe = truncate_to_keyframe

    def execute(self, recorder: CameraRecorder) -> None:
        """从缓存提取区间并执行 PyAV remux，结果写入 ``self.future``。"""
        try:
            result = recorder.remux_range(
                file_path=self.file_path,
                start_simulate_index=self.start_simulate_index,
                end_simulate_index=self.end_simulate_index,
                truncate_to_keyframe=self.truncate_to_keyframe,
            )
            self.future.set_result(result)
        except Exception as e:  # noqa: BLE001 - 汇报给 Future，不中断 worker
            self.future.set_exception(e)
            _logger.error(
                f"save task failed for {self.file_path}: {e}\n"
                f"{traceback.format_exc()}"
            )


def _default_single_frame_trigger(
    task: "RecordingTask", current_simulate_index: int
) -> bool:
    """``SingleFrameTask`` 默认触发条件：当前 index 达到目标。

    与 ``_default_range_trigger`` 一致，使用 ``>=`` 容忍渲染延迟导致的跳号。
    """
    return current_simulate_index >= task.simulate_index  # type: ignore[attr-defined]


class DecodeTask(RecordingTask["DecodeTask"]):
    """解码一帧 NAL 数据，更新 recorder 的持久 CodecContext DPB 状态。

    由接收线程在每帧 NAL 到达时提交到 ``_save_queue``，save_worker FIFO
    保证解码任务先于同帧的 ``SingleFrameTask`` 回调执行，使回调可直接
    从 ``recorder.last_decoded_frame`` 取得解码结果，无需回溯到 IDR。

    触发条件：立即触发（``should_trigger`` 恒返回 ``True``），因为解码
    任务提交时目标帧已到达（接收线程刚收到该帧）。
    """

    def __init__(
        self,
        nal_data: bytes,
        simulate_index: int,
    ) -> None:
        """初始化解码任务。

        Args:
            nal_data: H.264 NAL 单元数据（Annex-B 基本流）。
            simulate_index: 帧的 simulate_index（用于日志）。
        """
        super().__init__(_always_trigger)
        self.nal_data = nal_data
        self.simulate_index = simulate_index

    def execute(self, recorder: CameraRecorder) -> None:
        """调用 ``recorder._decode_nal`` 解码 NAL 并更新 DPB 状态。"""
        try:
            # _decode_nal 是 CameraRecorder 的内部方法，但 DecodeTask 在
            # save_worker 线程内执行（recorder 持有 CodecContext），属于
            # 同模块内的受控内部访问。
            recorder._decode_nal(self.nal_data, self.simulate_index)  # type: ignore[attr-defined]  # noqa: SLF001
            self.future.set_result(None)
        except Exception as e:  # noqa: BLE001 - 汇报给 Future，不中断 worker
            self.future.set_exception(e)
            _logger.error(
                f"decode task failed for sim_idx={self.simulate_index}: {e}\n"
                f"{traceback.format_exc()}"
            )


def _always_trigger(task: "RecordingTask", current_simulate_index: int) -> bool:
    """``DecodeTask`` 触发条件：恒返回 ``True``（立即触发）。"""
    return True


class SingleFrameTask(RecordingTask["SingleFrameTask"]):
    """在指定 ``simulate_index`` 的帧到达时执行回调，并返回回调的返回值。

    用于逐帧回调场景（如 LeRobot 流式写帧）：目标帧到达后由 save_worker
    线程执行回调，回调内可访问该帧的 ``FrameEntry``（含 timestamp_ns /
    simulate_index）和解码后的 RGB numpy 数组。主线程提交后不阻塞。

    **future 返回值**：``future.result()`` 返回 ``on_frame`` 回调的返回值
    （``Any`` 类型），由回调自行决定返回什么。例如多相机同步场景中，副相机
    回调可 ``return decoded_frame`` 供主相机回调通过 future 收集；主相机
    回调可 ``return None``（已完成 add_frame，无需返回值）。
    跳过时返回 ``None``（见下）。

    **降频采样跳过语义**：当控制频率 > 渲染频率时（如 50Hz 控制 vs 30Hz
    渲染），目标 ``simulate_index`` 可能没有对应的视频帧（渲染跳号，
    例如目标 sim_idx=41，但帧到达序列为 40, 42，41 被跳过）。此时
    ``execute`` 不执行回调，``future.set_result(None)`` 表示跳过该帧
    （非异常，调用方可通过 ``future.result() is None`` 判断是否跳过）。

    触发条件：``current_simulate_index >= simulate_index``。与
    ``RangeSaveTask`` 一致，容忍渲染延迟导致的跳号/重复。
    """

    def __init__(
        self,
        simulate_index: int,
        on_frame: FrameCallback,
        trigger_fn: TriggerFn | None = None,
    ) -> None:
        """初始化单帧回调任务。

        Args:
            simulate_index: 目标帧的 simulate_index。
            on_frame: 帧到达后执行的回调。在 save_worker 线程调用，接收
                ``FrameEntry`` 和解码后的 ``np.ndarray | None``。回调返回值
                会作为 ``future`` 的结果（``set_result``）。回调内异常会被
                捕获并写入 future。
            trigger_fn: 可选触发回调，默认
                ``current_index >= simulate_index`` 时触发。

        Raises:
            TypeError: ``on_frame`` 不是 callable。
        """
        if not callable(on_frame):
            raise TypeError("on_frame must be callable")
        super().__init__(trigger_fn or _default_single_frame_trigger)
        self.simulate_index = simulate_index
        self.on_frame = on_frame

    def execute(self, recorder: CameraRecorder) -> None:
        """从缓存提取目标帧并执行回调，回调返回值作为 future 结果。"""
        try:
            frame = recorder.get_frame(self.simulate_index)
            if frame is None:
                # 降频采样跳过：目标 sim_idx 无对应视频帧（渲染跳号）
                self.future.set_result(None)
                return
            # save_worker FIFO 保证 DecodeTask 先于本任务执行，此处可直接取解码帧
            decoded = recorder.last_decoded_frame
            result = self.on_frame(frame, decoded)
            self.future.set_result(result)
        except Exception as e:  # noqa: BLE001 - 汇报给 Future，不中断 worker
            self.future.set_exception(e)
            _logger.error(
                f"single frame task failed for sim_idx={self.simulate_index}: {e}\n"
                f"{traceback.format_exc()}"
            )


class RecordingTaskQueue:
    """等待触发任务队列。线程安全。

    主线程提交任务（``add``），接收线程逐帧轮询（``poll_triggered``），
    stop/close 时取出全部未触发任务（``drain``）。
    """

    def __init__(self) -> None:
        self._tasks: list[RecordingTask] = []
        self._lock = threading.RLock()

    def add(self, task: RecordingTask) -> None:
        """向队列追加一个等待触发的任务。"""
        with self._lock:
            self._tasks.append(task)

    def poll_triggered(
        self, current_simulate_index: int
    ) -> list[RecordingTask]:
        """取出并返回所有满足触发条件的任务。

        Args:
            current_simulate_index: 接收线程当前收到的 simulate_index，
                用于与各任务的触发回调比较。
        """
        with self._lock:
            triggered = [
                t
                for t in self._tasks
                if t.should_trigger(current_simulate_index)
            ]
            for t in triggered:
                self._tasks.remove(t)
            return triggered

    def drain(self) -> list[RecordingTask]:
        """取出并清空所有未触发任务（用于 stop/close 时尽力保存）。"""
        with self._lock:
            tasks = list(self._tasks)
            self._tasks.clear()
            return tasks

    @property
    def pending_count(self) -> int:
        """队列中等待触发的任务数。"""
        with self._lock:
            return len(self._tasks)