# 📷 Sensor API

传感器数据查询接口。相机串流、录制与可视化已迁移至 `orca_gym.recorder` 模块
（见下方"相机接口"章节）。

## 目录

- [相机接口（新）](#相机接口orca_gymrecorder)：`start_streaming` / `save_streaming` / `show_camera` / `VideoRecorderManager`
- [rgbd_camera（已废弃）](#rgbd_camera已废弃)：`CameraWrapper` / `Monitor` / `CameraCacher` 等旧类
- [传感器数据查询（Environment 层）](#传感器数据查询environment-layer)
- [LiDAR 点云查询](#lidar-点云查询)

---

## 相机接口（orca_gym.recorder）

录制已从引擎侧移到 Python 客户端侧，由 `orca_gym.recorder` 模块
（`VideoRecorderManager`，基于 PyAV）统一管理串流、录制、保存与实时可视化。
环境层提供以下入口（`OrcaGymEulerEnv` / `OrcaGymLocalEnv`）：

```python
def get_camera_names() -> list[str]          # 枚举所有已注册相机
def start_streaming(camera_name, **kwargs)   # 一键配置 + 推流 + 启动录制器
def show_camera(camera_name, camera_type="color", window_name=None)  # 非阻塞实时预览
def save_streaming(camera_name, camera_type, file_path,
                   start_simulate_index, end_simulate_index) -> Future[RemuxResult]
                                             # 保存区间视频为 MP4（非阻塞）
def get_recorder_manager() -> VideoRecorderManager  # 底层管理器（高级用法）
```

典型用法：

```python
camera = env.get_camera_names()[0]
env.start_streaming(camera, capture_rgb=True, color_port=7070)

for i in range(100):
    env.step(action)
    env.render(simulate_index=i)          # 索引必须递增，用于帧对齐

env.show_camera(camera)                    # 实时预览（非阻塞）
future = env.save_streaming(camera, "color", "/tmp/out.mp4", 0, 99)
result = future.result()                   # RemuxResult: file_path / frame_count /
                                           #             frame_indices / timestamps_ns
env.close()                                # 自动保存未完成任务
```

完整 API 说明（含 `SingleFrameTask` 逐帧回调、深度流、多相机）见
[Environment API](environment.md) 与教程 [📷 相机与视觉](../tutorial/beginners/camera-and-vision.md)。
从旧版本迁移见 [迁移指南](../migration/camera-recorder.md)。

---

## rgbd_camera（已废弃）

> ⚠️ `orca_gym.sensor.rgbd_camera` 模块（`CameraWrapper` / `Monitor` /
> `CameraCacher` / `CameraDataParser` / `VideoPlayer`）**已废弃**：
> 录制移到 Python 侧后由 `orca_gym.recorder` 取代。旧类保留仅为兼容，
> 请按 [迁移指南](../migration/camera-recorder.md) 迁移到新接口。

| 旧类 | 替代接口 |
|------|----------|
| `CameraWrapper` | `env.start_streaming()` + `manager.get_last_decoded_frame()` |
| `Monitor` | `env.show_camera()` |
| `CameraCacher` | `env.save_streaming()`（直接输出 MP4） |
| `CameraDataParser` / `VideoPlayer` | 标准 MP4，任意播放器 / PyAV 读取 |

---

## 传感器数据查询（Environment 层）

Environment 层提供了直接查询传感器数据（加速度计、陀螺仪、触觉等）的接口：

```python
sensor_data = env.query_sensor_data(["imu_accelerometer", "imu_gyro", "touch_left_finger"])
# 返回: {"imu_accelerometer": array(3,), "imu_gyro": array(3,), "touch_left_finger": array(1,)}
```

传感器类型包括：
- `accelerometer`: 加速度计（线性加速度）
- `gyro`: 陀螺仪（角速度）
- `touch`: 触觉传感器（接触力）
- `velocimeter`: 速度计（线性速度）
- `framequat`: 框架姿态（四元数）
