# 📹 RGB-D Camera

OrcaGym provides real-time RGB-D camera streaming via WebSocket.

## CameraWrapper

```python
from orca_sensor.rgbd_camera import CameraWrapper

# Create a camera wrapper
camera = CameraWrapper(
    name="front_camera",
    port=8765  # camera WebSocket port
)

# Start the camera stream (in a background thread)
camera.start()

# Wait for the first frame
while not camera.is_first_frame_received():
    time.sleep(0.1)

# Get the current image
image = camera.image  # (H, W, 3) BGR numpy array
print(f"Image shape: {image.shape}")
print(f"Frame index: {camera.image_index}")
```

## Multi-Camera Setup

```python
# Create multiple cameras
cameras = {
    "front": CameraWrapper("front", port=8765),
    "side": CameraWrapper("side", port=8766),
    "top": CameraWrapper("top", port=8767),
}

# Start all
for cam in cameras.values():
    cam.start()

# Wait for all cameras to be ready
for cam in cameras.values():
    while not cam.is_first_frame_received():
        time.sleep(0.1)

# Synchronously capture all camera frames
def get_all_frames(cameras):
    return {name: cam.image.copy() for name, cam in cameras.items()}
```

## Camera Data Stream Architecture

```
OrcaStudio/OrcaLab
 ├── Rendering Engine
 │   └── Camera Frame → H.264 Encode → WebSocket Stream
 │
Python Client
 └── CameraWrapper
     ├── WebSocket Connection (ws://localhost:<port>)
     ├── H.264 Decode (PyAV)
     └── NumPy BGR Array
```

## Camera Pose

```python
# Get camera frames and pose information
camera_transforms = env.get_frame_png("path/to/save")
# → {"front_camera": {"pos": [x,y,z], "quat": [w,x,y,z]}, ...}

for camera_name, transform in camera_transforms.items():
    print(f"{camera_name}:")
    print(f"  Position: {transform['pos']}")
    print(f"  Orientation: {transform['quat']}")
```

## Camera Monitor

OrcaGym includes a camera monitor script:

```bash
# Launch the camera monitor
python -m orca_scripts.camera_monitor
```

## Camera Timestamps

```python
# Get camera timestamp information
last_frame = 0
timestamps = env.get_camera_time_stamp(last_frame)
# → {"camera_name": [ts1, ts2, ts3, ...], ...}

# Get the current frame index
current_frame = env.get_current_frame()

# Wait for a new frame
next_frame = env.get_next_frame()
```

## Using Cameras in RL Training

```python
class VisionEnv(OrcaGymEulerEnv):
    def __init__(self, ...):
        super().__init__(...)

        # Set up camera
        self.camera = CameraWrapper("agent_view", port=8765)
        self.camera.start()

        # Observation space includes images
        self.observation_space = spaces.Dict({
            "proprio": spaces.Box(-np.inf, np.inf, shape=(proprio_dim,)),
            "image": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
        })

    def _get_obs(self):
        return {
            "proprio": np.concatenate([
                self.data.qpos.copy(),
                self.data.qvel.copy(),
            ]).astype(np.float32),
            "image": self.camera.image.copy(),
        }
```

## Performance Tips

1. **Async mode** (CaptureMode.ASYNC) is more friendly for visual RL
2. **Reduce image resolution** to increase frame rate
3. **Lower the frame rate appropriately** — 30 FPS is usually sufficient
4. **Decode in multiple threads** — avoid blocking the main simulation thread
