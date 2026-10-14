"""OrcaGym 相机录制模块。

提供 VideoRecorder 和 VideoRecorderManager，用于从 WebSocket 接收 H.264 码流
并使用 PyAV remux 保存为 MP4 文件。录制任务抽象（RecordingTask / RangeSaveTask）
与等待队列（RecordingTaskQueue）支持按触发回调扩展新的任务类型。

典型用法（推荐通过 VideoRecorderManager 统一接口操作）::

    from orca_gym.recorder import CreateVideoRecorderManager

    # stub 为 gRPC 能力后端对象（OrcaGymLocal / OrcaGymEuler），
    # 提供相机属性查询/设置接口；可为 None（仅录制）。
    manager = CreateVideoRecorderManager(stub=None)
    manager.start_recorder("Camera", color_port=7070)

    # 仿真循环中...
    current_simulate_index = 100  # 由调用方维护当前物理仿真步索引
    env.render(simulate_index=current_simulate_index)

    # 保存 [100, 500] 区间（非阻塞：注册区间任务，端帧到达后由 worker 线程异步 remux）
    future = manager.save_streaming("Camera", "/tmp/output.mp4",
                                    start_simulate_index=100,
                                    end_simulate_index=500)
    result = future.result()  # 等待保存完成，返回 RemuxResult
    print(result.file_path, result.frame_count, result.frame_indices)

    manager.stop_recorder("Camera")
"""

from orca_gym.recorder.camera_recorder import CameraRecorder, RemuxResult
from orca_gym.recorder.depth_recorder import DepthRecorder
from orca_gym.recorder.recording_task import (
    FrameCallback,
    RangeSaveTask,
    RecordingTask,
    RecordingTaskQueue,
    SingleFrameTask,
    TriggerFn,
)
from orca_gym.recorder.rolling_frame_buffer import FrameEntry, RollingFrameBuffer
from orca_gym.recorder.video_recorder import (
    CreateVideoRecorderManager,
    VideoRecorder,
    VideoRecorderManager,
    create_camera_recorder,
)
from orca_gym.recorder.video_stream_viewer import VideoStreamViewer

__all__ = [
    "CameraRecorder",
    "CreateVideoRecorderManager",
    "DepthRecorder",
    "FrameCallback",
    "FrameEntry",
    "RangeSaveTask",
    "RecordingTask",
    "RecordingTaskQueue",
    "RemuxResult",
    "RollingFrameBuffer",
    "SingleFrameTask",
    "TriggerFn",
    "VideoRecorder",
    "VideoRecorderManager",
    "VideoStreamViewer",
    "create_camera_recorder",
]
