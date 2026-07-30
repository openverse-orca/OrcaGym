# 📷 Camera & Vision — Getting RGB-D Images

Beyond reading joint angles and body poses, you can also obtain **images** from the simulation — like giving the robot eyes.

OrcaGym receives real-time rendered camera frames from OrcaStudio via WebSocket.

---

## Quick Start: Getting Your First Image

```python
"""
first_camera.py — Get the first image from a simulation camera
"""

import time
import numpy as np
from orca_gym.sensor.rgbd_camera import CameraWrapper

# 1. Create camera wrapper
# name: camera name (arbitrary)
# port: WebSocket port (configured in OrcaStudio)
camera = CameraWrapper(name="front_camera", port=8765)

# 2. Start camera stream (a background thread auto-receives and decodes frames)
camera.start()
print("Camera started, waiting for first frame...")

# 3. Wait for the first frame to arrive
while not camera.is_first_frame_received():
    time.sleep(0.1)

print(f"✅ First frame received!")

# 4. Get the image
image = camera.image  # NumPy array, shape (H, W, 3), format BGR
print(f" Resolution: {image.shape[1]}x{image.shape[0]}")
print(f" Data type: {image.dtype}")
print(f" Pixel range: [{image.min()}, {image.max()}]")
print(f" Frame index: {camera.image_index}")

# 5. Save the image (with OpenCV)
import cv2
cv2.imwrite("first_frame.png", image)
print("✅ Image saved to first_frame.png")

# 6. Stop the camera
camera.stop()
```

---

## Integrating the Camera into an Environment Class

Make the camera part of the environment, capturing an image at each step:

```python
"""
vision_env.py — Environment with camera observations
"""

import time
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.sensor.rgbd_camera import CameraWrapper


class VisionEnv(OrcaGymEulerEnv):
    """Environment that includes camera images in observations."""

    def __init__(self, frame_skip, orcagym_addr, agent_names, time_step,
                 camera_port: int = 8765, **kwargs):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        # -- Set up camera --
        self._camera = CameraWrapper(name="agent_view", port=camera_port)
        self._camera.start()

        # Wait for first frame
        print("Waiting for camera to be ready...")
        while not self._camera.is_first_frame_received():
            time.sleep(0.1)
        print(f"✅ Camera ready: {self._camera.image.shape}")

        # Action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        obs_sample = self._get_obs()
        self.observation_space = spaces.Dict({
            key: spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype)
            if isinstance(v, np.ndarray) and v.dtype != np.uint8
            else spaces.Box(low=0, high=255, shape=v.shape, dtype=np.uint8)
            for key, v in obs_sample.items()
        })

    def _get_obs(self):
        """
        Observation = proprioception (joint angles) + vision (camera image)

        Proprioception: the state the robot "feels" about itself
        Vision: what the camera sees
        """
        return {
            # Proprioception
            "joint_pos": self.data.qpos.copy().astype(np.float32),
            "joint_vel": self.data.qvel.copy().astype(np.float32),

            # Vision
            "image": self._camera.image.copy(),  # (H, W, 3) uint8
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}

    def close(self):
        self._camera.stop()
        super().close()
```

---

## Displaying the Camera Feed

Live display with Matplotlib:

```python
import matplotlib.pyplot as plt

def show_camera_live(camera: CameraWrapper, duration: float = 10.0):
    """
    Live display of camera feed (for duration seconds).

    Note: This is only a simple display example.
    Real RL training does not require frame-by-frame display — just use the image array directly.
    """
    plt.ion()  # interactive mode
    fig, ax = plt.subplots()
    img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    ax.set_title("Camera Feed")
    ax.axis('off')

    start = time.time()
    while time.time() - start < duration:
        frame = camera.image.copy()
        # OpenCV BGR -> Matplotlib RGB
        frame_rgb = frame[..., ::-1]
        img_display.set_data(frame_rgb)
        fig.canvas.flush_events()
        plt.pause(0.03)  # ~30 FPS

    plt.ioff()
    plt.close()
```

---

## Multi-Camera Setup

Need multiple viewpoints? Create multiple `CameraWrapper` instances:

```python
def setup_multi_camera():
    """Start multiple cameras simultaneously."""

    cameras = {
        "front": CameraWrapper("front", port=8765),
        "side": CameraWrapper("side", port=8766),
        "top": CameraWrapper("top", port=8767),
    }

    # Start all
    for cam in cameras.values():
        cam.start()

    # Wait for all cameras to be ready
    for name, cam in cameras.items():
        while not cam.is_first_frame_received():
            time.sleep(0.1)
        print(f"✅ {name}: {cam.image.shape}")

    # Synchronized retrieval of all views
    def get_all_views():
        return {name: cam.image.copy() for name, cam in cameras.items()}

    return cameras, get_all_views


# Usage
cameras, get_views = setup_multi_camera()
views = get_views()
print(f"Available views: {list(views.keys())}")
```

---

## Camera Parameter Configuration

Each camera's parameters can be configured in OrcaStudio:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| Resolution | Image width x height | 640x480, 1280x720 |
| Frame rate | Frames per second | 15, 30, 60 |
| RGB | Whether to output color image | `True` |
| Depth | Whether to output depth image | `True` (when needed) |

Configure on the Python side via `CameraSensorInfo`:

```python
from orca_gym.scene.orca_gym_scene import CameraSensorInfo

# Configure sensor parameters for a specific camera
camera_config = CameraSensorInfo(
    capture_rgb=True,      # output RGB image
    capture_depth=True,    # output depth map
    save_mp4_file=False,   # do not save to file
    use_dds=False,         # do not use DDS compression
)
scene.set_camera_sensor_info("camera_actor_name", camera_config)
```

---

## Performance Tips

1. **Do not `imshow` in the main loop** — image display is very CPU-intensive. During training, just process the array directly.
2. **Downscale images** — if full resolution is not needed, resize in `_get_obs()`.
3. **Lower the frame rate** — 30 FPS is usually sufficient; higher frame rates waste bandwidth.
4. **Asynchronous rendering** — the camera decodes frames in a background thread without blocking the main simulation thread.

```python
def _get_obs(self):
    image = self._camera.image.copy()
    # Downscale to 128x128 to reduce computation
    small_image = cv2.resize(image, (128, 128))
    return {..., "image": small_image}
```

---

## Next Step

Now you can both "feel" (read state) and "see" (camera). Next, learn how to **precisely control the robot**: [🎮 Simple Controller](simple-controller.md).
