"""按 simulate_index 索引的滚动帧缓存。

用于 VideoRecorder 缓存 WebSocket 接收到的 H.264 NAL 帧，
支持按 [start, end] 区间提取帧列表。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from collections import OrderedDict

from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


@dataclass
class FrameEntry:
    """单个 H.264 帧缓存条目。

    Attributes:
        simulate_index: 物理仿真步索引（来自 WebSocket 帧头）
        timestamp_ns: 纳秒时间戳（来自 WebSocket 帧头）
        nal_data: H.264 NAL 单元数据
        is_keyframe: 是否 IDR 帧（NAL type == 5）
    """

    simulate_index: int
    timestamp_ns: int
    nal_data: bytes
    is_keyframe: bool


class RollingFrameBuffer:
    """按 simulate_index 索引的滚动帧缓存。

    保留最近 max_frames 帧，超出时自动淘汰最旧帧。
    支持按 [start, end] 闭区间提取帧列表。

    使用 OrderedDict 维护插入顺序和 O(1) 淘汰，
    同时维护有序的 simulate_index 列表支持 O(log n) 区间查询。

    线程安全：接收线程（写）与保存/查询线程（读）并发访问，
    所有公共方法内部持有 ``_lock``（可重入锁）保证互斥。
    """

    def __init__(self, max_frames: int = 36000) -> None:
        """初始化滚动缓存。

        Args:
            max_frames: 最大缓存帧数。默认 36000（约 20 分钟 @ 30fps）。
        """
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        self._max_frames = max_frames
        # OrderedDict[simulate_index -> FrameEntry]，按插入顺序
        self._frames: OrderedDict[int, FrameEntry] = OrderedDict()
        # 保护 _frames 的读写互斥（可重入，避免嵌套调用死锁）
        self._lock = threading.RLock()

    def append(
        self,
        simulate_index: int,
        timestamp_ns: int,
        nal_data: bytes,
        is_keyframe: bool,
    ) -> None:
        """追加一帧到缓存。

        Args:
            simulate_index: 物理仿真步索引
            timestamp_ns: 纳秒时间戳
            nal_data: H.264 NAL 单元数据
            is_keyframe: 是否 IDR 帧
        """
        with self._lock:
            # 若该 simulate_index 已存在，先删除旧条目再插入（更新）
            if simulate_index in self._frames:
                del self._frames[simulate_index]

            self._frames[simulate_index] = FrameEntry(
                simulate_index=simulate_index,
                timestamp_ns=timestamp_ns,
                nal_data=nal_data,
                is_keyframe=is_keyframe,
            )

            # 淘汰最旧帧
            while len(self._frames) > self._max_frames:
                self._frames.popitem(last=False)

    def extract_range(
        self,
        start_simulate_index: int,
        end_simulate_index: int,
    ) -> list[FrameEntry]:
        """提取 [start, end] 闭区间内的所有帧。

        Args:
            start_simulate_index: 区间起始（含）
            end_simulate_index: 区间结束（含）

        Returns:
            按 simulate_index 升序排列的 FrameEntry 列表。
            若区间内无帧，返回空列表。
        """
        if start_simulate_index > end_simulate_index:
            raise ValueError(
                f"start_simulate_index ({start_simulate_index}) > "
                f"end_simulate_index ({end_simulate_index})"
            )

        with self._lock:
            # OrderedDict 按插入顺序遍历，但插入顺序不保证数值升序
            # （sim_idx=-1 的帧或重复 append reinsert 会破坏升序），
            # 返回前显式按 simulate_index 排序，保证 docstring 声明的升序契约。
            result = [
                entry
                for sim_idx, entry in self._frames.items()
                if start_simulate_index <= sim_idx <= end_simulate_index
            ]
            result.sort(key=lambda e: e.simulate_index)
            return result

    def get_latest_simulate_index(self) -> int:
        """返回缓存中最新的 simulate_index。空缓存返回 -1。"""
        with self._lock:
            if not self._frames:
                return -1
            # next(reversed(...)) 取最后一个 key（最新插入的）
            return next(reversed(self._frames))

    def get_oldest_simulate_index(self) -> int:
        """返回缓存中最旧的 simulate_index。空缓存返回 -1。"""
        with self._lock:
            if not self._frames:
                return -1
            return next(iter(self._frames))

    def get_first_keyframe_after(self, sim_idx: int) -> FrameEntry | None:
        """返回 >= sim_idx 的第一个 IDR 帧（前向查找）。

        用于录制段保存的**前向截断**：录制起点调用 ``request_idr=True`` 后，
        段内首帧通常是 IDR；但若关键帧未精确落在起点（如引擎在起点附近
        才输出 IDR），则前向找到第一个关键帧，丢弃其之前的非关键帧，使
        视频从关键帧开始（避免以 P 帧开头导致解码花屏）。

        Args:
            sim_idx: 目标 simulate_index（含，返回 >= 该值的关键帧）

        Returns:
            大于等于 ``sim_idx`` 的第一个 IDR FrameEntry，若不存在返回 None。
        """
        with self._lock:
            for entry in self._frames.values():
                if entry.simulate_index < sim_idx:
                    continue
                if entry.is_keyframe:
                    return entry
            return None

    def get_latest_keyframe_strictly_before(self, sim_idx: int) -> FrameEntry | None:
        """返回 < sim_idx 的最近 IDR 帧。

        用于 ``remux_range`` 首帧非 IDR 时的 primer 帧回溯：
        当用户请求的区间 ``[start, end]`` 首帧不是 IDR 时，
        需要回溯到 ``start`` 之前最近的 IDR 帧作为解码器初始化帧。

        Args:
            sim_idx: 区间起始 simulate_index（不含）

        Returns:
            严格小于 ``sim_idx`` 的最近 IDR FrameEntry，若不存在返回 None。

        Note:
            当 ``sim_idx`` 为负数（如 -1，表示引擎侧未设置 simulate_index，
            uint32 转换后溢出）时，本方法无法正确比较，返回 None。
            此时应使用 ``get_latest_keyframe_before_timestamp`` 作为 fallback。
        """
        if sim_idx < 0:
            return None
        with self._lock:
            latest_keyframe: FrameEntry | None = None
            for entry in self._frames.values():
                if entry.simulate_index >= sim_idx:
                    break
                if entry.is_keyframe:
                    latest_keyframe = entry
            return latest_keyframe

    def get_latest_keyframe_before_timestamp(
        self, timestamp_ns: int
    ) -> FrameEntry | None:
        """返回 ``timestamp_ns`` 之前最近的 IDR 帧（不限 simulate_index）。

        作为 ``get_latest_keyframe_strictly_before`` 的 fallback：
        当 sim_idx 回溯失败（I 帧的 sim_idx 为 -1 等无效值，
        或所有帧 sim_idx 都为 -1 时）时，用时间戳回溯最近的 IDR 帧。

        不限制 ``simulate_index``，因为引擎自动产生的 I 帧可能携带
        ``sim_idx=-1``（uint32 溢出），这些帧仍然有效。

        Args:
            timestamp_ns: 区间首帧的时间戳（不含，严格小于）

        Returns:
            ``timestamp_ns < target`` 的最近 IDR FrameEntry。
            若不存在返回 None。
        """
        with self._lock:
            latest_keyframe: FrameEntry | None = None
            for entry in self._frames.values():
                if entry.timestamp_ns >= timestamp_ns:
                    break
                if entry.is_keyframe:
                    latest_keyframe = entry
            return latest_keyframe

    def get_frame(self, simulate_index: int) -> FrameEntry | None:
        """获取指定 simulate_index 的单帧。

        供 ``SingleFrameTask.execute`` 调用：当降频采样（控制频率 > 渲染频率）
        时，目标 ``simulate_index`` 可能没有对应的视频帧（渲染跳号），此时
        返回 None，由调用方决定是否跳过该帧。

        Args:
            simulate_index: 目标帧的 simulate_index。

        Returns:
            ``FrameEntry`` 或 ``None``（不存在或已被滚动淘汰）。
        """
        with self._lock:
            return self._frames.get(simulate_index)

    def clear(self) -> None:
        """清空缓存。"""
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        """当前缓存帧数。"""
        with self._lock:
            return len(self._frames)

    def __contains__(self, simulate_index: int) -> bool:
        """判断指定 simulate_index 是否在缓存中。"""
        with self._lock:
            return simulate_index in self._frames
