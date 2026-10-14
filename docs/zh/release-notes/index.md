# 📝 Release Notes

## 26.9.1

### ⚠️ 破坏性变更：相机接口重构（rgbd_camera → recorder）

本版本对相机串流、录制与可视化接口做了破坏性变更，**录制从引擎侧移到 Python 客户端侧**：

**移除**

- 引擎侧 MP4 录制 gRPC：`StartCameraRecording` / `StopCameraRecording` 等
- gRPC 相机接口：`QueryCameras` / `ConfigureCamera` / `StartCameraStream` / `StopCameraStream`

**废弃（保留兼容，调用时发出 DeprecationWarning 并 no-op）**

- `orca_gym.sensor.rgbd_camera` 模块：`CameraWrapper` / `Monitor` / `CameraCacher` / `CameraDataParser` / `VideoPlayer`
- 环境层旧录制接口：`begin_save_video` / `stop_save_video` / `get_current_frame` / `get_next_frame` / `get_camera_time_stamp`

**新增**

- `orca_gym.recorder` 模块（基于 PyAV），统一管理串流、录制、保存与实时可视化：
  - `env.start_streaming(camera_name, ...)` — 一键配置 + 推流 + 启动录制器
  - `env.save_streaming(camera_name, camera_type, file_path, start_simulate_index, end_simulate_index)` — 保存区间视频为标准 MP4（非阻塞，返回 `Future[RemuxResult]`）
  - `env.show_camera(camera_name)` — 非阻塞实时预览窗口
  - `env.get_recorder_manager()` — `VideoRecorderManager`，支持 `SingleFrameTask` 逐帧回调等多相机同步采集
  - `env.get_camera_names()` / `manager.set_camera_properties(...)` / `manager.set_streaming_enabled(...)` 替代旧 gRPC 相机配置
- 录制输出为标准 MP4（替代旧 `.h264` + 时间戳 `.bin`），结果自带帧号 ↔ 物理步映射（`frame_indices` / `timestamps_ns`）

**使用注意**

1. `render(simulate_index=...)` 的索引必须递增，用于帧对齐与保存区间提取
2. **保存视频时，录制区间的第一帧与最后一帧渲染时必须传 `request_idr=True`**（首帧保证视频可从起点播放，末帧保证区间末尾帧完整写入 MP4）
3. 新增依赖：`av`（PyAV）、`websockets`；实时预览另需 `matplotlib`、`opencv-python`

📋 详细迁移步骤见 [相机接口迁移指南](../migration/camera-recorder.md)，新接口完整用法见
[📷 相机与视觉教程](../tutorial/beginners/camera-and-vision.md)。
