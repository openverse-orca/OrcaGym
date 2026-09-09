# 📷 Camera & Vision — Getting RGB-D Images

Beyond reading joint angles and body poses, you can also obtain **images** from the simulation — like giving the robot eyes.

OrcaGym receives real-time rendered camera frames from OrcaStudio via WebSocket, supporting frame alignment by physics-step index, saving intervals as MP4, and live preview.

---

## Quick Start: Getting Your First Image

```python
"""
first_camera.py — Get camera frames via the environment layer
"""

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class CameraEnv(OrcaGymEulerEnv):
    """Minimal camera environment."""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            sync_render=kwargs.pop("sync_render", True),
            render_fps=kwargs.pop("render_fps", 30),
            **kwargs,
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        return self._get_obs(), 0.0, False, False, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.data.qpos.copy()


if __name__ == "__main__":
    env = CameraEnv(
        model_xml_path="tests/orca_gym/environment/euler/fixtures/simple_pendulum.xml",
        orcagym_addr="localhost:50051",
        skip_grpc_load=False,  # online mode, requires connecting to OrcaStudio
        render_mode="human",
    )
    env.reset()

    # 1. Enumerate all registered camera names
    camera_names = env.get_camera_names()
    print(f"Available cameras: {camera_names}")
    assert camera_names, "No cameras found. Add a Camera Entity in the OrcaStudio scene first"
    camera_name = camera_names[0]

    # 2. Start streaming (one-shot configuration + push stream)
    env.start_streaming(
        camera_name,
        capture_rgb=True,       # enable RGB stream
        color_port=7070,        # RGB stream WebSocket port
        width=640,
        height=480,
    )
    print(f"✅ Camera '{camera_name}' streaming started")

    # 3. Main loop: render(simulate_index=...) drives rendering
    #    simulate_index is used to extract frames by interval; it must be monotonically increasing
    for step_idx in range(100):
        action = np.zeros(env.model.nu, dtype=np.float32)
        env.step(action)
        env.render(simulate_index=step_idx)

    # 4. Live preview (non-blocking)
    env.show_camera(camera_name, camera_type="color")

    # 5. Save an interval video as MP4 (non-blocking, returns a Future)
    future = env.save_streaming(
        camera_name=camera_name,
        camera_type="color",
        file_path="/tmp/pendulum.mp4",
        start_simulate_index=0,
        end_simulate_index=99,
    )
    result = future.result()  # wait for saving to finish
    print(f"✅ Saved: {result.file_path} ({result.frame_count} frames)")

    env.close()
```

> ⚠️ **The examples in this section require online mode** (`skip_grpc_load=False`) and a connection to OrcaStudio.
> In offline mode, the camera interface returns an empty list / no-op.

---

## Core API Reference

### 1. Enumerate Camera Names

```python
camera_names = env.get_camera_names()
# → ["Camera_Entity_[uuid1]", "Camera_Entity_[uuid2]", ...]
```

Returns all camera names registered in the OrcaStudio scene. In offline mode it returns an empty list.

### 2. Start Streaming (`start_streaming`) ⭐ Recommended

```python
env.start_streaming(
    camera_name,
    capture_rgb=True,           # enable RGB stream
    capture_depth=True,         # enable depth stream
    color_port=7070,            # RGB stream WebSocket port
    depth_port=7071,            # depth stream WebSocket port
    width=1280,                 # image width
    height=720,                 # image height
    vertical_fov=60.0,          # vertical field of view (degrees)
    near_clip=0.01,             # near clip plane
    far_clip=100.0,             # far clip plane
    gamma=1.0,                  # depth camera gamma correction
    use_nvenc=True,             # use NvEnc hardware encoding
    nvenc_gpu_index=0,          # NvEnc GPU index
)
```

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `capture_rgb` | whether to output RGB color image | `True` |
| `capture_depth` | whether to output depth image | `True` (when needed) |
| `capture_normal` | whether to output normal map | `False` |
| `capture_object_color` | whether to output instance-segmentation color-coded image | `False` |
| `width` / `height` | image resolution | 640×480, 1280×720 |
| `vertical_fov` | vertical field of view (degrees) | 60.0 |
| `near_clip` / `far_clip` | clip plane distances | 0.01 / 100.0 |
| `color_port` / `depth_port` | WebSocket ports | 7070 / 7071 |
| `use_nvenc` | NvEnc hardware encoding | `True` (when an NVIDIA GPU is available) |

> 💡 This method internally handles camera-property synchronization and streaming-state transitions, so the upper layer does not need to care about the underlying state machine.

### 3. Drive Rendering (`render`)

```python
env.render(simulate_index=step_idx, request_idr=False)
```

- **`simulate_index`**: physics simulation step index, passed through to the engine camera pipeline for frame alignment.
  When saving interval video, frames are extracted by this index, which **must be monotonically increasing**.
- **`request_idr`**: whether to request the engine to output an IDR keyframe.
  Set `True` at the start of a saved segment to ensure the output video starts playing correctly.

Rendering frequency is controlled by `set_render_fps(fps)`:
- `sync_render=True`: throttled by physics step (render one frame every N physics steps)
- `sync_render=False`: throttled by wall-clock time (render one frame every `1/fps` seconds)

### 4. Save Interval Video (`save_streaming`)

```python
future = env.save_streaming(
    camera_name="Camera_Entity_[uuid]",
    camera_type="color",          # "color" or "depth"
    file_path="/tmp/output.mp4",
    start_simulate_index=50,       # interval start (inclusive)
    end_simulate_index=200,       # interval end (inclusive)
)
result = future.result()          # non-blocking, wait for result
print(result.file_path, result.frame_count)
```

- **Non-blocking**: returns a `Future` immediately; call `future.result()` to get the result after saving completes.
- `start_streaming` must be called first to start the stream.

### 5. Live Preview (`show_camera`)

```python
env.show_camera(camera_name, camera_type="color", window_name="Front View")
```

- **Non-blocking**: shows frames in a separate window without affecting the main simulation thread.
- Depth streams display raw grayscale frames directly, without pseudo-color conversion.
- `start_streaming` must be called first to start the stream.

---

## Integrating the Camera into an Environment Class

```python
"""
vision_env.py — Environment with camera observations
"""

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class VisionEnv(OrcaGymEulerEnv):
    """Environment that includes camera images in observations."""

    def __init__(self, model_xml_path, camera_name, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            sync_render=True,
            render_fps=30,
            render_mode="human",
            skip_grpc_load=kwargs.pop("skip_grpc_load", False),
            **kwargs,
        )

        # ── Camera configuration ──
        self._camera_name = camera_name
        self.start_streaming(
            camera_name,
            capture_rgb=True,
            color_port=7070,
            width=640,
            height=480,
        )
        self._step_idx = 0

        # ── Action space ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # ── Observation space ──
        obs_sample = self._get_obs()
        self.observation_space = spaces.Dict({
            key: spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype)
            if isinstance(v, np.ndarray) and v.dtype != np.uint8
            else spaces.Box(low=0, high=255, shape=v.shape, dtype=np.uint8)
            for key, v in obs_sample.items()
        })

    def _get_obs(self):
        """Observation = proprioception + vision"""
        # Get the latest decoded frame (np.ndarray (H, W, 3) uint8 BGR, or None)
        frame = self.get_recorder_manager().get_last_decoded_frame(
            self._camera_name, "color"
        )
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return {
            "joint_pos": self.data.qpos.copy().astype(np.float32),
            "joint_vel": self.data.qvel.copy().astype(np.float32),
            "image": frame.copy(),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        self.do_simulation(action, self.frame_skip)

        # Drive rendering + frame alignment
        self._step_idx += 1
        self.render(simulate_index=self._step_idx)

        obs = self._get_obs()
        return obs, 0.0, False, False, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        self._step_idx = 0
        return self._get_obs(), {}
```

---

## Multi-Camera Setup

```python
# 1. Enumerate all cameras
camera_names = env.get_camera_names()

# 2. Start streaming for each (use different ports to avoid conflicts)
env.start_streaming(camera_names[0], capture_rgb=True, color_port=7070)
env.start_streaming(camera_names[1], capture_rgb=True, color_port=7071)
env.start_streaming(camera_names[2], capture_rgb=True, color_port=7072)

# 3. Render once in the main loop (one render triggers rendering for all cameras)
for step_idx in range(100):
    env.step(action)
    env.render(simulate_index=step_idx)

# 4. Preview individually
env.show_camera(camera_names[0], "color", window_name="Front")
env.show_camera(camera_names[2], "color", window_name="Top")
```

> 💡 A single `render(simulate_index=...)` triggers the engine to render all enabled cameras,
> so there is no need to call render for each camera individually.

---

## Depth Camera

```python
# Start the depth stream
env.start_streaming(
    camera_name,
    capture_rgb=False,
    capture_depth=True,
    depth_port=7071,
    near_clip=0.01,    # depth camera near clip plane
    far_clip=100.0,    # depth camera far clip plane
    gamma=1.0,         # depth camera gamma correction
)

# Preview (grayscale display, no pseudo-color conversion)
env.show_camera(camera_name, camera_type="depth")

# Save depth video
future = env.save_streaming(
    camera_name, "depth", "/tmp/depth.mp4",
    start_simulate_index=0, end_simulate_index=99,
)
```

> ⚠️ Depth and color streams are independent streams; operate on them separately with `camera_type="color"` / `"depth"`.
> The same camera can enable RGB + Depth simultaneously (`capture_rgb=True, capture_depth=True`).

---

## Performance Tips

1. **Don't `imshow` in the main loop** — image display is very CPU-intensive.
   Use `show_camera()` to display it in a separate window, or skip display entirely during training.
2. **Lower the frame rate** — 15–30 FPS is usually sufficient during training; higher frame rates waste bandwidth.
3. **NvEnc hardware encoding** — set `use_nvenc=True` when an NVIDIA GPU is available to greatly reduce encoding CPU overhead.
4. **Downscale images** — if your RL policy does not need full resolution, resize in `_get_obs()`:
   ```python
   import cv2
   small = cv2.resize(frame, (128, 128))
   ```
5. **Record intervals rather than the full run** — use `save_streaming(start, end)` to save only key intervals,
   with constant memory usage.

---

## Next Step

Now you can both "feel" (read state) and "see" (camera). Next, learn how to **precisely control the robot**: [🎮 Simple Controller](simple-controller.md).
