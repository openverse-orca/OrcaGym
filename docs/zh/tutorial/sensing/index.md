# 📷 传感与感知

OrcaGym 提供多种传感器接口和 RGB-D 相机支持。

> 完整可运行代码见 [OrcaPlayground examples/euler/04_query_api/](https://github.com/OrcaGym/OrcaPlayground) 和 [08_video_capture/](https://github.com/OrcaGym/OrcaPlayground)。

## 传感器类型

| 传感器 | MuJoCo 类型 | 输出 |
|--------|-------------|------|
| 加速度计 | `mjSENS_ACCELEROMETER` | (3,) 加速度 |
| 陀螺仪 | `mjSENS_GYRO` | (3,) 角速度 |
| 力/扭矩 | `mjSENS_FORCE` / `mjSENS_TORQUE` | (3,) / (3,) |
| 触觉 | `mjSENS_TOUCH` | (n,) 触觉阵列 |
| 关节位置 | `mjSENS_JOINTPOS` | (1,) 关节角度 |
| 关节速度 | `mjSENS_JOINTVEL` | (1,) 关节速度 |
| RGB-D 相机 | WebSocket 流式传输 | (H, W, 3) 图像 + 深度 |

## 查询方式

所有传感器数据通过统一的 `env.query_sensor_data(names)` API 查询：

```python
sensor_data = env.query_sensor_data(["g1_imu_quat", "g1_imu_gyro"])
imu_quat = sensor_data["g1_imu_quat"]  # (4,) 姿态四元数
imu_gyro = sensor_data["g1_imu_gyro"]  # (3,) 角速度
```

## 章节导航

- [🖲️ 传感器系统](sensors.md) — MuJoCo 原生传感器查询与数据布局
- [📹 RGB-D 相机](rgbd-camera.md) — WebSocket 相机流式传输
- [🤝 接触感知](contact-sensing.md) — 接触力作为触觉感知
