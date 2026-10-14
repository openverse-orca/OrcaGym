# 📷 Sensor API

Sensor data query interface. Camera streaming, recording, and visualization have
migrated to the `orca_gym.recorder` module (see the "Camera Interface" section below).

## Contents

- [Camera Interface (new)](#camera-interface-orca_gymrecorder): `start_streaming` / `save_streaming` / `show_camera` / `VideoRecorderManager`
- [rgbd_camera (deprecated)](#rgbd_camera-deprecated): `CameraWrapper` / `Monitor` / `CameraCacher`, etc.
- [Sensor Data Queries (Environment Layer)](#sensor-data-queries-environment-layer)
- [LiDAR Point Cloud Query](#lidar-point-cloud-query)

---

## Camera Interface (orca_gym.recorder)

Recording has moved from the engine side to the Python client side. The
`orca_gym.recorder` module (`VideoRecorderManager`, based on PyAV) uniformly
manages streaming, recording, saving, and live visualization. The environment
layer (`OrcaGymEulerEnv` / `OrcaGymLocalEnv`) provides these entry points:

```python
def get_camera_names() -> list[str]          # Enumerate all registered cameras
def start_streaming(camera_name, **kwargs)   # One-call config + streaming + recorder startup
def show_camera(camera_name, camera_type="color", window_name=None)  # Non-blocking live preview
def save_streaming(camera_name, camera_type, file_path,
                   start_simulate_index, end_simulate_index) -> Future[RemuxResult]
                                             # Save a range to MP4 (non-blocking)
def get_recorder_manager() -> VideoRecorderManager  # Low-level manager (advanced usage)
```

Typical usage:

```python
camera = env.get_camera_names()[0]
env.start_streaming(camera, capture_rgb=True, color_port=7070)

for i in range(100):
    env.step(action)
    env.render(simulate_index=i)          # Index must increase; used for frame alignment

env.show_camera(camera)                    # Live preview (non-blocking)
future = env.save_streaming(camera, "color", "/tmp/out.mp4", 0, 99)
result = future.result()                   # RemuxResult: file_path / frame_count /
                                           #             frame_indices / timestamps_ns
env.close()                                # Automatically saves pending tasks
```

For the full API (including `SingleFrameTask` per-frame callbacks, depth streams,
multi-camera setups), see the [Environment API](environment.md) and the tutorial
[📷 Camera & Vision](../tutorial/beginners/camera-and-vision.md).
For upgrading from the old version, see the [Migration Guide](../migration/camera-recorder.md).

---

## rgbd_camera (Deprecated)

> ⚠️ The `orca_gym.sensor.rgbd_camera` module (`CameraWrapper` / `Monitor` /
> `CameraCacher` / `CameraDataParser` / `VideoPlayer`) is **deprecated**:
> after recording moved to the Python side, it has been replaced by
> `orca_gym.recorder`. The old classes are kept only for compatibility —
> please migrate via the [Migration Guide](../migration/camera-recorder.md).

| Old class | Replacement |
|-----------|-------------|
| `CameraWrapper` | `env.start_streaming()` + `manager.get_last_decoded_frame()` |
| `Monitor` | `env.show_camera()` |
| `CameraCacher` | `env.save_streaming()` (outputs standard MP4 directly) |
| `CameraDataParser` / `VideoPlayer` | Standard MP4, readable by any player / PyAV |

---

## Sensor Data Queries (Environment Layer)

The Environment layer provides interfaces to directly query sensor data (accelerometer, gyroscope, touch, etc.):

```python
sensor_data = env.query_sensor_data(["imu_accelerometer", "imu_gyro", "touch_left_finger"])
# Returns: {"imu_accelerometer": array(3,), "imu_gyro": array(3,), "touch_left_finger": array(1,)}
```

Sensor types include:
- `accelerometer`: Accelerometer (linear acceleration)
- `gyro`: Gyroscope (angular velocity)
- `touch`: Touch sensor (contact force)
- `velocimeter`: Velocimeter (linear velocity)
- `framequat`: Frame orientation (quaternion)

### LiDAR Point Cloud Query

`OrcaGymEulerEnv` / `OrcaGymLocalEnv` provide a LiDAR point cloud query interface:

```python
lidar_data = env.query_lidar_point_cloud("LiDAREntity")
# Returns a dict containing:
# - bin_count, vertical_layers, angular_resolution, max_h_angle,
#   min_v_angle, v_step, min_range, max_range
# - sensor_position: world-space sensor origin, shape (3,)
# - sensor_rotation: world-space sensor quaternion (x, y, z, w), shape (4,)
# - ranges: range image, shape (bin_count, vertical_layers), -1.0 = unscanned
# - points: world-space 3D points, shape (bin_count, vertical_layers, 3)
# - intensities: intensity image, shape (bin_count, vertical_layers)
# Returns None if the entity is not found, the ring buffer is empty, or in offline mode
```

> Note: The point cloud (`points`) is in **world coordinates**. Use `sensor_position` / `sensor_rotation` to transform between world and sensor frames.
