# 🔄 相机接口迁移指南（rgbd_camera → recorder）

本版本对相机串流、录制与可视化接口做了**破坏性变更**：

- **录制从引擎侧移到 Python 侧**：旧的引擎侧 MP4 录制 gRPC（`StartCameraRecording` / `StopCameraRecording` 等）已删除，录制与保存全部由 Python 客户端完成（`orca_gym.recorder` 模块，基于 PyAV）。
- **旧串流模块 `orca_gym.sensor.rgbd_camera` 废弃**：`CameraWrapper` / `Monitor` / `CameraCacher` / `CameraDataParser` / `VideoPlayer` 不再推荐使用，请改用环境层新接口。

## 旧 → 新接口对照

| 旧接口（rgbd_camera / 引擎侧录制） | 新接口（orca_gym.recorder） |
|---|---|
| `CameraWrapper(name, port)` 手动连 WebSocket 收帧 | `env.start_streaming(camera_name, ...)` 一键配置 + 推流 + 启动录制器 |
| `Monitor(name, fps, port)` 阻塞式监控窗口 | `env.show_camera(camera_name)` 非阻塞子进程可视化 |
| `CameraCacher` 落盘裸 `.h264` + 时间戳 `.bin` | `env.save_streaming(...)` 直接保存标准 MP4 |
| `CameraDataParser` / `VideoPlayer` 离线解析回放 | MP4 可用任意播放器 / PyAV 直接读取，结果含帧号↔物理步映射 |
| gRPC `StartCameraRecording` / `StopCameraRecording`（引擎侧录制） | **已删除**，由 `env.save_streaming` / `VideoRecorderManager` 替代 |
| gRPC `QueryCameras` / `ConfigureCamera` / `StartCameraStream` / `StopCameraStream` | `env.get_camera_names()` / `manager.set_camera_properties(...)` / `manager.set_streaming_enabled(...)` |
| **env 层 `begin_save_video` / `stop_save_video`（旧常用录制入口）** | **已废弃**（调用时发出 DeprecationWarning 并 no-op），由 `env.save_streaming` 替代 |
| env 层 `get_current_frame` / `get_next_frame` / `get_camera_time_stamp` | **已废弃**（no-op），帧号与时间戳改由 `RemuxResult.frame_indices` / `timestamps_ns` 提供 |

### `begin_save_video` 用户迁移示例

`begin_save_video` / `stop_save_video` 是旧版最常用的"全程录制"入口，迁移方式如下：

**旧写法（废弃）：**

```python
env.begin_save_video("/tmp/episode.mp4")   # 已废弃：no-op + DeprecationWarning
# ... 仿真循环 ...
env.stop_save_video()
```

**新写法：**

```python
env.start_streaming(camera, capture_rgb=True, color_port=7070)

# 录制区间第一帧请求 IDR 关键帧（保证视频可从起点播放）
for i in range(total_steps):
    env.step(action)
    if i == 0:
        env.render(simulate_index=0, request_idr=True)   # 区间首帧
    elif i == total_steps - 1:
        env.render(simulate_index=i, request_idr=True)   # 区间末帧
    else:
        env.render(simulate_index=i)

future = env.save_streaming(camera, "color", "/tmp/episode.mp4", 0, total_steps - 1)
result = future.result()
```

## 新接口快速上手

```python
env = MyEnv(...)  # OrcaGymEulerEnv / OrcaGymLocalEnv 子类
env.reset()

# 1. 枚举相机
camera_names = env.get_camera_names()
camera = camera_names[0]

# 2. 一键启动串流（自动配置相机属性 + 开启推流 + 启动客户端录制器）
env.start_streaming(
    camera,
    capture_rgb=True,        # 需要深度时加 capture_depth=True
    color_port=7070,
    width=640, height=480,
)

# 3. 仿真循环：render 必须传入递增的 simulate_index（用于帧对齐与区间提取）
for i in range(100):
    env.step(action)
    if i == 0 or i == 99:
        env.render(simulate_index=i, request_idr=True)  # 录制区间首帧/末帧请求 IDR
    else:
        env.render(simulate_index=i)

# 4. 实时预览（非阻塞，独立子进程窗口；深度流显示原始灰度帧）
env.show_camera(camera, camera_type="color")

# 5. 保存区间视频为 MP4（非阻塞，返回 Future）
future = env.save_streaming(
    camera_name=camera,
    camera_type="color",                 # "color" 或 "depth"
    file_path="/tmp/episode.mp4",
    start_simulate_index=0,              # 含端点
    end_simulate_index=99,
)
result = future.result()                 # RemuxResult
print(result.file_path, result.frame_count, result.frame_indices)

env.close()  # 自动保存所有未完成的录制任务
```

## 关键变化说明

1. **帧对齐**：`render(simulate_index=...)` 的索引必须递增，保存区间按此索引提取帧。
2. **IDR 关键帧请求（重要）**：需要保存视频时，**录制区间第一帧**与**最后一帧**渲染时都必须
   传 `request_idr=True`（即 `render(simulate_index=i, request_idr=True)`）。
   首帧 IDR 保证输出视频可从区间起点正常解码播放；末帧 IDR 保证区间末尾的帧
   在 H.264 码流中能被完整解码写入 MP4。中间帧无需额外请求。
3. **非阻塞保存**：`save_streaming` 立即返回 `Future[RemuxResult]`，由后台 worker 线程在
   结束帧到达后执行 remux；`env.close()` 会阻塞等待未完成任务保存完成。
4. **多相机**：对每个相机分别 `start_streaming`（不同端口），一次 `render` 触发所有相机渲染；
   color / depth 是独立码流，用 `camera_type` 区分操作。
5. **进阶用法**：`manager = env.get_recorder_manager()` 获取 `VideoRecorderManager`，
   可使用 `submit_task`（逐帧回调 `SingleFrameTask`）、`get_latest_frame_simulate_index` 等高级接口，
   详见 [Environment API](../api_reference/environment.md)。

## 获取单帧图像

新接口下有三种获取单帧的方式，按场景选用：

### 方式一：`SingleFrameTask` 逐帧回调（推荐，支持多相机同步）

目标帧到达后在后台线程执行回调，回调直接接收解码后的 RGB 数组，主线程不阻塞：

```python
from orca_gym.recorder import SingleFrameTask

manager = env.get_recorder_manager()

def on_frame(frame_entry, decoded_frame):
    # frame_entry.timestamp_ns / simulate_index / is_keyframe
    # decoded_frame: np.ndarray (H, W, 3) uint8 RGB，解码失败为 None
    return decoded_frame   # 返回值作为 future.result()

task = SingleFrameTask(simulate_index=100, on_frame=on_frame)
future = manager.submit_task(camera, task)   # 非阻塞
frame = future.result()   # None 表示该 sim_idx 无对应视频帧（渲染跳号被跳过）
```

> 控制频率 > 渲染频率时，目标 `simulate_index` 可能无对应视频帧（渲染跳号），
> 此时 `future.result()` 返回 `None`，属正常跳过而非异常。

### 方式二：读取最新已解码帧（调试 / 单相机）

```python
frame = manager.get_last_decoded_frame(camera, "color")
# np.ndarray (H, W, 3) uint8，或 None（尚未解码任何帧）
```

> 注意：该方式跨 recorder 无锁、sim_idx 不可控（可能读到"过新"或"过旧"帧），
> 多相机同步采集请使用方式一。

## 依赖

录制与可视化需要：`av`（PyAV）、`websockets`、`numpy`；可视化另需 `matplotlib`、`opencv-python`。

## 迁移示例

**旧写法（废弃）：**

```python
from orca_gym.sensor.rgbd_camera import CameraWrapper

camera = CameraWrapper("front_camera", port=7070)
camera.start()
frame, idx = camera.get_frame(format="rgb24")
camera.stop()
```

**新写法：**

```python
env.start_streaming("front_camera", capture_rgb=True, color_port=7070)
env.render(simulate_index=step_idx)
frame = env.get_recorder_manager().get_last_decoded_frame("front_camera", "color")
```

> 完整教程见 [📷 相机与视觉](../tutorial/beginners/camera-and-vision.md)。
