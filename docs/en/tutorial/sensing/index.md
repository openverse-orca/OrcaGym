# 📷 Sensing & Perception

OrcaGym provides various sensor interfaces and RGB-D camera support.

> For complete runnable code, see [OrcaPlayground examples/euler/05_query_api/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/05_query_api) and [08_video_capture/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/08_video_capture).

## Sensor Types

| Sensor | MuJoCo Type | Output |
|--------|-------------|--------|
| Accelerometer | `mjSENS_ACCELEROMETER` | (3,) acceleration |
| Gyroscope | `mjSENS_GYRO` | (3,) angular velocity |
| Force/Torque | `mjSENS_FORCE` / `mjSENS_TORQUE` | (3,) / (3,) |
| Touch | `mjSENS_TOUCH` | (n,) touch array |
| Joint Position | `mjSENS_JOINTPOS` | (1,) joint angle |
| Joint Velocity | `mjSENS_JOINTVEL` | (1,) joint velocity |
| RGB-D Camera | WebSocket streaming | (H, W, 3) image + depth |

## Query Method

All sensor data is queried through the unified `env.query_sensor_data(names)` API:

```python
sensor_data = env.query_sensor_data(["g1_imu_quat", "g1_imu_gyro"])
imu_quat = sensor_data["g1_imu_quat"]  # (4,) orientation quaternion
imu_gyro = sensor_data["g1_imu_gyro"]  # (3,) angular velocity
```

## Chapter Navigation

- [🖲️ Sensor System](sensors.md) — MuJoCo native sensor querying and data layout

- [🤝 Contact Sensing](contact-sensing.md) — Contact forces as tactile perception
