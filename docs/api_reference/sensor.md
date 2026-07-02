# 📷 Sensor API

传感器接口，提供相机图像接收、缓存、解析、回放等功能。

## 类概览

| 类 | 说明 |
|----|------|
| `CameraWrapper` | 实时接收仿真渲染的相机图像 |
| `CameraCacher` | 将相机流保存到本地文件 |
| `CameraDataParser` | 解析本地缓存的相机数据 |
| `VideoPlayer` | 逐帧回放本地缓存的视频 |
| `Monitor` | 实时相机监控窗口 |

---

## CameraWrapper

通过 WebSocket 连接仿真服务接收实时相机图像。

### 构造

```python
class CameraWrapper:
    def __init__(self, name: str, port: int)
```
- `name`: 相机名称
- `port`: WebSocket 端口号

### 属性

```python
name: str                       # 相机名称
port: int                       # WebSocket 端口
image: np.ndarray               # 当前图像 (H, W, 3)，BGR 格式
image_index: int                # 当前帧索引
enabled: bool                   # 是否启用（默认 True）
received_first_frame: bool      # 是否已收到第一帧
```

### 生命周期

```python
def start()                     # 启动后台接收线程
def stop()                      # 停止接收
```

### 帧获取

```python
def get_frame(format: str = 'bgr24', size: tuple | None = None) -> tuple[np.ndarray, int]
```
- `format`: `'bgr24'`（默认）或 `'rgb24'`
- `size`: 可选的目标尺寸 `(width, height)`
- 返回: `(frame, image_index)`

```python
def is_first_frame_received() -> bool  # 是否已收到第一帧
```

### 使用示例

```python
from orca_gym.sensor.rgbd_camera import CameraWrapper
import time

camera = CameraWrapper("front_camera", port=8765)
camera.start()

# 等待第一帧
while not camera.is_first_frame_received():
    time.sleep(0.1)

# 循环获取帧
for i in range(100):
    frame, idx = camera.get_frame(format='rgb24', size=(224, 224))
    print(f"帧 #{idx}, 形状: {frame.shape}")

camera.stop()
```

---

## CameraCacher

将 WebSocket 相机流保存为本地文件，包含 H.264 视频和二进制时间戳。

### 构造

```python
class CameraCacher:
    def __init__(self, name: str, port: int)
```
- `name`: 相机名称（生成 `{name}_video.h264` 和 `{name}_ts.bin`）
- `port`: WebSocket 端口

### 生命周期

```python
def start()
def stop()
def is_first_frame_received() -> bool
```

---

## CameraDataParser

解析 CameraCacher 保存的离线数据。

### 构造

```python
class CameraDataParser:
    def __init__(self, name: str)
```

### 帧查找

```python
def get_closed_frame(ts: int) -> tuple[int, np.ndarray]  # 按时间戳查找最近帧
def get_frame(index: int) -> np.ndarray                   # 按索引获取指定帧
```

---

## VideoPlayer

用 OpenCV 逐帧播放本地缓存的 H.264 视频。

```python
class VideoPlayer:
    def __init__(self, name: str)
    def play()                     # 播放，按 q 退出
```

---

## Monitor

基于 Matplotlib 的实时相机监控窗口。

```python
class Monitor:
    def __init__(self, name: str, fps: int = 30, port: int = 7070)
    def start()                    # 启动监控窗口（会阻塞当前线程）
    def stop()                     # 停止
```

---

## 完整使用示例

### 实时监控

```python
from orca_gym.sensor.rgbd_camera import Monitor

monitor = Monitor("my_camera", fps=30, port=8765)
monitor.start()  # 阻塞直到关闭窗口
```

### 离线缓存 + 按需读取

```python
from orca_gym.sensor.rgbd_camera import CameraCacher, CameraDataParser, VideoPlayer

# === 采集阶段 ===
cacher = CameraCacher("my_dataset", port=8765)
cacher.start()
# ... 运行仿真 ...
cacher.stop()

# === 解析阶段 ===
parser = CameraDataParser("my_dataset")
ts = 1234567890
index, frame = parser.get_closed_frame(ts)

# === 回放阶段 ===
player = VideoPlayer("my_dataset")
player.play()
```

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
