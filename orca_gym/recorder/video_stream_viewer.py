"""视频流实时可视化查看器（子进程 + 独立 WebSocket + matplotlib）。

每个 viewer 启动一个独立子进程，子进程自己建立 WebSocket 连接接收 H.264
码流、解码、用 matplotlib 在主线程中渲染显示。

设计要点:
    - **完全进程隔离**：每个 viewer 一个子进程，独立 WebSocket 连接，
      独立解码，独立 matplotlib 窗口。互不干扰，无线程安全问题。
    - **不依赖主进程的滚动缓存**：子进程直接从 WebSocket 读取码流，
      即使主进程的录制器停止也不影响已启动的 viewer。
    - **手动 ``new_timer`` 驱动刷新**：在子进程主线程中运行，符合
      matplotlib GUI 要求。单个定时器同时承担"帧刷新"（~30fps）和
      ``stop_event`` 轮询（~500ms）职责，**不使用 ``FuncAnimation``**
      （``plt.close(fig)`` 后内部 ``event_source`` 被置 None，已入队
      定时器回调仍会访问 ``None.interval`` 抛 ``AttributeError``）。
    - 关闭窗口时子进程通过 ``Event`` 通知主进程。
    - 深度流直接显示原始灰度帧，不做伪彩色转换。

跨平台支持:
    - matplotlib 在 Linux (QtAgg/TkAgg) 和 Windows 上均可工作。
    - 子进程模式避免 matplotlib "GUI outside main thread" 限制。

依赖: ``matplotlib``、``numpy``、``av``（PyAV）、``websockets``、``opencv-python``。
"""

from __future__ import annotations

import multiprocessing
import sys
from typing import TYPE_CHECKING

from orca_gym.log.orca_log import get_orca_logger

if TYPE_CHECKING:
    from orca_gym.recorder.camera_recorder import CameraRecorder

_logger = get_orca_logger()

#: WebSocket 连接重试次数
_WS_RETRIES = 3

#: WebSocket 连接重试间隔（秒）
_WS_RETRY_INTERVAL = 1.0

#: WebSocket 连接超时（秒）
_WS_CONNECT_TIMEOUT = 5.0


def _viewer_subprocess(
    port: int,
    window_name: str,
    stop_event: multiprocessing.synchronize.Event,
) -> None:
    """子进程入口：独立 WebSocket 连接 + 解码 + matplotlib 显示。

    本函数在子进程的**主线程**中运行，matplotlib GUI 可以正常工作。

    Args:
        port: WebSocket 推流端口。
        window_name: 窗口标题。
        stop_event: 停止事件（主进程设置时子进程退出）。
    """
    import asyncio
    import io
    import threading

    import av
    import cv2
    import matplotlib

    # 自动选择可用的 GUI 后端
    _backend_found = False
    for _backend in ("QtAgg", "TkAgg", "GTK4Agg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(_backend)
            import matplotlib.pyplot as plt
            _backend_found = True
            break
        except Exception:
            continue
    if not _backend_found:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        print(
            f"[Viewer:{window_name}] No interactive matplotlib backend available, "
            "window will not display. Install tkinter or PyQt5."
        )

    import numpy as np
    import websockets

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    latest_frame: list[np.ndarray | None] = [None]  # 用 list 做可变容器
    running = [True]  # 子进程运行标志

    # ------------------------------------------------------------------
    # WebSocket 接收 + 解码线程
    # ------------------------------------------------------------------
    async def _receive_loop() -> None:
        uri = f"ws://localhost:{port}"
        for attempt in range(1, _WS_RETRIES + 1):
            try:
                async with websockets.connect(
                    uri, open_timeout=_WS_CONNECT_TIMEOUT
                ) as websocket:
                    _logger.info(
                        f"[Viewer:{window_name}] WebSocket connected to {uri}"
                    )
                    raw_buf = io.BytesIO()
                    container = None
                    cur_pos = 0

                    while running[0] and not stop_event.is_set():
                        try:
                            data = await asyncio.wait_for(
                                websocket.recv(), timeout=1.0
                            )
                        except asyncio.TimeoutError:
                            continue

                        # 跳过 8 字节时间戳头
                        payload = data[8:]
                        raw_buf.write(payload)
                        raw_buf.seek(cur_pos)

                        if cur_pos == 0:
                            container = av.open(raw_buf, mode="r")

                        try:
                            for packet in container.demux():
                                if packet.size == 0:
                                    continue
                                for frame in packet.decode():
                                    latest_frame[0] = frame.to_ndarray(
                                        format="bgr24"
                                    )
                        except av.FFmpegError:
                            # 码流不完整或解码瞬时错误（PyAV 10+ 已移除
                            # ``av.AVError``，统一用 ``av.FFmpegError`` 基类
                            # 捕获 ffmpeg 相关错误）。等待更多数据后重试。
                            pass

                        cur_pos += len(payload)
                    return  # 正常退出
            except Exception as e:
                _logger.warning(
                    f"[Viewer:{window_name}] WebSocket connect attempt "
                    f"{attempt}/{_WS_RETRIES} failed: {e}, "
                    f"retrying in {_WS_RETRY_INTERVAL}s..."
                )
                if attempt < _WS_RETRIES:
                    await asyncio.sleep(_WS_RETRY_INTERVAL)

        _logger.error(
            f"[Viewer:{window_name}] WebSocket connect failed after "
            f"{_WS_RETRIES} attempts"
        )

    def _ws_thread_target() -> None:
        try:
            asyncio.run(_receive_loop())
        except Exception as e:  # noqa: BLE001
            _logger.error(
                f"[Viewer:{window_name}] receive thread error: {e}",
                exc_info=True,
            )

    # 启动接收线程
    ws_thread = threading.Thread(
        target=_ws_thread_target,
        name=f"ViewerWS-{window_name}",
        daemon=True,
    )
    ws_thread.start()

    # ------------------------------------------------------------------
    # matplotlib 窗口（主线程）
    # ------------------------------------------------------------------
    # **不使用 FuncAnimation**：FuncAnimation 内部维护 event_source，
    # plt.close(fig) 后 event_source 被置 None，但定时器队列里可能还有
    # pending tick，下次 _step 访问 None.interval 抛 AttributeError。
    # 改用单个 new_timer 同时承担"帧刷新"和"stop_event 轮询"职责，
    # 回调内自行判断状态，完全避开 FuncAnimation 内部状态机。
    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title(window_name)
    ax.set_title(window_name)
    ax.axis("off")
    img_obj = ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))

    def _on_close(_event):  # type: ignore[no-untyped-def]
        running[0] = False
        stop_event.set()

    fig.canvas.mpl_connect("close_event", _on_close)

    # 单定时器：每 33ms 刷新帧 + 每 N 次 tick 检查 stop_event
    _TIMER_INTERVAL_MS = 33  # ~30fps
    _STOP_POLL_EVERY_N_TICKS = 15  # 33ms * 15 ≈ 500ms 检查一次 stop_event
    _tick_count = [0]

    def _on_tick():
        # 1) 检查停止事件（每 ~500ms 一次）
        _tick_count[0] += 1
        if _tick_count[0] % _STOP_POLL_EVERY_N_TICKS == 0 and stop_event.is_set():
            _logger.info(
                f"[Viewer:{window_name}] stop_event detected, closing window"
            )
            try:
                render_timer.stop()
            except Exception:  # noqa: BLE001
                pass
            plt.close(fig)
            return
        # 2) 刷新帧
        frame = latest_frame[0]
        if frame is not None:
            # BGR → RGB for matplotlib
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_obj.set_data(rgb)
            try:
                fig.canvas.draw_idle()
            except Exception:  # noqa: BLE001
                # figure 已关闭等场景，忽略
                pass

    render_timer = None
    try:
        render_timer = fig.canvas.new_timer(interval=_TIMER_INTERVAL_MS)
        render_timer.add_callback(_on_tick)
        render_timer.start()
    except (AttributeError, NotImplementedError):
        # 非交互后端（如 Agg）无可用定时器，靠 close_event 与
        # KeyboardInterrupt 兜底（但 stop_event.set() 不可被监听）。
        render_timer = None

    _logger.info(f"[Viewer:{window_name}] subprocess display started (port={port})")

    try:
        plt.show()
    except KeyboardInterrupt:
        # Ctrl+C 时 SIGINT 广播到所有子进程，matplotlib Qt 后端会抛出
        # KeyboardInterrupt。正常退出，不需要报错。
        pass
    except Exception as e:  # noqa: BLE001
        _logger.error(f"[Viewer:{window_name}] display error: {e}", exc_info=True)
    finally:
        running[0] = False
        if render_timer is not None:
            try:
                render_timer.stop()
            except Exception:  # noqa: BLE001
                pass
        ws_thread.join(timeout=2.0)
        try:
            plt.close(fig)
        except Exception:  # noqa: BLE001
            pass
        _logger.info(f"[Viewer:{window_name}] subprocess display stopped")


class VideoStreamViewer:
    """单相机的实时视频可视化查看器（子进程，独立 WebSocket）。

    启动一个独立子进程，子进程自己建立 WebSocket 连接接收 H.264 码流、
    解码、用 matplotlib 渲染显示。与主进程的录制器完全解耦——子进程
    直接连 WebSocket 端口，不读取主进程的滚动缓存。

    深度流直接显示原始灰度帧，不做伪彩色转换。
    ``DepthRecorder.grayscale_to_depth`` / ``depth_to_visualization``
    保留供程序化使用（如离线分析）。

    Args:
        recorder: 关联的 ``CameraRecorder``（用于获取端口）。
        window_name: 窗口标题；默认 ``Camera: <相机名>``。
    """

    def __init__(
        self,
        recorder: CameraRecorder,
        window_name: str | None = None,
    ) -> None:
        self._recorder = recorder
        self._window_name = window_name or f"Camera: {recorder.camera_name}"
        self._running = False
        self._process: multiprocessing.Process | None = None
        self._stop_event: multiprocessing.synchronize.Event | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动渲染子进程（非阻塞）。

        子进程独立建立 WebSocket 连接、解码、渲染。

        Raises:
            RuntimeError: 已在运行。
            ValueError: 无法确定端口。
        """
        if self._running:
            raise RuntimeError(f"Viewer '{self._window_name}' is already running")

        port = self._recorder.port
        if not port or port <= 0:
            raise ValueError(
                f"Cannot determine port for camera "
                f"'{self._recorder.camera_name}'"
            )

        # fork 在 Linux 上启动快且共享内存页；Windows 不支持 fork，用 spawn。
        ctx = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
        self._stop_event = ctx.Event()

        self._process = ctx.Process(
            target=_viewer_subprocess,
            args=(
                port,
                self._window_name,
                self._stop_event,
            ),
            daemon=True,
            name=f"ViewerProc-{self._window_name}",
        )
        self._process.start()
        self._running = True
        _logger.info(
            f"[Viewer:{self._window_name}] started (camera "
            f"'{self._recorder.camera_name}', port={port}, "
            f"pid={self._process.pid})"
        )

    def stop(self) -> None:
        """停止渲染子进程（幂等）。"""
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
        self._process = None
        _logger.info(f"[Viewer:{self._window_name}] stopped")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """查看器子进程是否在运行。"""
        return (
            self._running
            and self._process is not None
            and self._process.is_alive()
        )

    def get_stats(self) -> dict:
        """返回查看器状态统计。"""
        return {
            "window_name": self._window_name,
            "camera_name": self._recorder.camera_name,
            "is_running": self.is_running,
            "process_pid": self._process.pid if self._process else None,
        }
