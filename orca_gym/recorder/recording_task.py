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
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from orca_gym.log.orca_log import get_orca_logger

if TYPE_CHECKING:
    from orca_gym.recorder.camera_recorder import CameraRecorder

_logger = get_orca_logger()

#: 触发回调类型：``(task, current_simulate_index) -> bool``。
#: ``current_simulate_index`` 为接收线程当前收到的 simulate_index，
#: 触发判断必须与之比较。
TriggerFn = Callable[["RecordingTask", int], bool]

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
                f"save task failed for {self.file_path}: {e}",
                exc_info=True,
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