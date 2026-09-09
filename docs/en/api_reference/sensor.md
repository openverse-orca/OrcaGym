# 📷 Sensor API

Sensor interface, providing camera image receiving, caching, parsing, and playback capabilities.

## Class Overview

| Class | Description |
|----|------|
| `CameraWrapper` | Receive rendered camera images from the simulation in real time |
| `CameraCacher` | Save camera streams to local files |
| `CameraDataParser` | Parse locally cached camera data |
| `VideoPlayer` | Play back locally cached video frame by frame |
| `Monitor` | Real-time camera monitoring window |

---

## CameraWrapper

Receives real-time camera images from the simulation service via WebSocket.

### Constructor

```python
class CameraWrapper:
    def __init__(self, name: str, port: int)
```
- `name`: Camera name
- `port`: WebSocket port number

### Properties

```python
name: str                       # Camera name (@property, stored internally as _name)
port: int                       # WebSocket port
image: np.ndarray               # Current image (H, W, 3), BGR format; random noise (480, 640, 3) before the first frame arrives
image_index: int                # Current frame index
enabled: bool                   # Whether enabled (default True)
received_first_frame: bool      # Whether the first frame has been received
```

### Lifecycle

```python
def start()                     # Start the background receiving thread
def stop()                      # Stop receiving
```

### Frame Retrieval

```python
def get_frame(self, format='bgr24', size: tuple = None) -> tuple[np.ndarray, int]
```
- `format`: `'bgr24'` (default) or `'rgb24'`
- `size`: Optional target size `(width, height)`
- Returns: `(frame, image_index)`

```python
def is_first_frame_received() -> bool  # Whether the first frame has been received
```

### Usage Example

```python
from orca_gym.sensor.rgbd_camera import CameraWrapper
import time

camera = CameraWrapper("front_camera", port=8765)
camera.start()

# Wait for the first frame
while not camera.is_first_frame_received():
    time.sleep(0.1)

# Retrieve frames in a loop
for i in range(100):
    frame, idx = camera.get_frame(format='rgb24', size=(224, 224))
    print(f"Frame #{idx}, shape: {frame.shape}")

camera.stop()
```

---

## CameraCacher

Saves WebSocket camera streams as local files, including H.264 video and binary timestamps.

### Constructor

```python
class CameraCacher:
    def __init__(self, name: str, port: int)
```
- `name`: Camera name (generates `{name}_video.h264` and `{name}_ts.bin`)
- `port`: WebSocket port number

### Lifecycle

```python
def start()
def stop()
def is_first_frame_received() -> bool
```

---

## CameraDataParser

Parses offline data saved by CameraCacher.

### Constructor

```python
class CameraDataParser:
    def __init__(self, name: str)
```

### Frame Lookup

```python
def get_closed_frame(self, ts) -> tuple[int, np.ndarray]  # Find the nearest frame by timestamp
def get_frame(self, index) -> np.ndarray                  # Get a specific frame by index (linear forward iteration, not random access)
```

> Note: `get_frame` only supports linear forward access. When `index == current_index` it returns the cached frame,
> otherwise it decodes forward to the target frame and caches it. Random backward access is not possible.

### Module-Level Helper Functions

```python
def find_closest_index(a: np.ndarray, target: int) -> int
    # Binary search: returns the index in the ascending timestamp array `a` closest to `target`
```

---

## VideoPlayer

Plays locally cached H.264 video frame by frame using OpenCV.

```python
class VideoPlayer:
    def __init__(self, name: str)
    def play()                     # Play; press q to quit
```

---

## Monitor

Matplotlib-based real-time camera monitoring window.

```python
class Monitor:
    def __init__(self, name: str, fps: int = 30, port: int = 7070)
    def start()                    # Start the monitoring window (blocks the current thread)
    def stop()                     # Stop: release camera resources (camera.stop()) and close the figure window (plt.close)
    def update(frame)              # matplotlib animation callback, used internally
```

---

## Complete Usage Examples

### Real-Time Monitoring

```python
from orca_gym.sensor.rgbd_camera import Monitor

monitor = Monitor("my_camera", fps=30, port=8765)
monitor.start()  # Blocks until the window is closed
```

### Offline Caching + On-Demand Reading

```python
from orca_gym.sensor.rgbd_camera import CameraCacher, CameraDataParser, VideoPlayer

# === Collection phase ===
cacher = CameraCacher("my_dataset", port=8765)
cacher.start()
# ... run the simulation ...
cacher.stop()

# === Parsing phase ===
parser = CameraDataParser("my_dataset")
ts = 1234567890
index, frame = parser.get_closed_frame(ts)

# === Playback phase ===
player = VideoPlayer("my_dataset")
player.play()
```

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
