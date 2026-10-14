# 🔄 Camera Interface Migration Guide (rgbd_camera → recorder)

This release introduces **breaking changes** to the camera streaming, recording, and visualization interfaces:

- **Recording moved from the engine side to the Python side**: the old engine-side MP4 recording gRPC calls (`StartCameraRecording` / `StopCameraRecording`, etc.) have been removed. Recording and saving are now handled entirely by the Python client (the `orca_gym.recorder` module, based on PyAV).
- **The legacy streaming module `orca_gym.sensor.rgbd_camera` is deprecated**: `CameraWrapper` / `Monitor` / `CameraCacher` / `CameraDataParser` / `VideoPlayer` are no longer recommended. Use the new environment-level APIs instead.

## Old → New Interface Mapping

| Old (rgbd_camera / engine-side recording) | New (orca_gym.recorder) |
|---|---|
| `CameraWrapper(name, port)` — manual WebSocket frame receiving | `env.start_streaming(camera_name, ...)` — one-call config + streaming + recorder startup |
| `Monitor(name, fps, port)` — blocking monitor window | `env.show_camera(camera_name)` — non-blocking subprocess viewer |
| `CameraCacher` — raw `.h264` + timestamp `.bin` files | `env.save_streaming(...)` — standard MP4 output |
| `CameraDataParser` / `VideoPlayer` — offline parsing & playback | MP4 playable by any player / readable via PyAV, with frame-index ↔ physics-step mapping |
| gRPC `StartCameraRecording` / `StopCameraRecording` (engine-side) | **Removed**; replaced by `env.save_streaming` / `VideoRecorderManager` |
| gRPC `QueryCameras` / `ConfigureCamera` / `StartCameraStream` / `StopCameraStream` | `env.get_camera_names()` / `manager.set_camera_properties(...)` / `manager.set_streaming_enabled(...)` |
| **Env-level `begin_save_video` / `stop_save_video` (old everyday recording entry point)** | **Deprecated** (emits DeprecationWarning and no-ops); replaced by `env.save_streaming` |
| Env-level `get_current_frame` / `get_next_frame` / `get_camera_time_stamp` | **Deprecated** (no-op); frame indices and timestamps are now provided by `RemuxResult.frame_indices` / `timestamps_ns` |

### `begin_save_video` Migration Example

`begin_save_video` / `stop_save_video` were the most commonly used "record everything" entry points in the old version. Migrate as follows:

**Old style (deprecated):**

```python
env.begin_save_video("/tmp/episode.mp4")   # Deprecated: no-op + DeprecationWarning
# ... simulation loop ...
env.stop_save_video()
```

**New style:**

```python
env.start_streaming(camera, capture_rgb=True, color_port=7070)

# Request an IDR keyframe on the FIRST frame of the recording range
for i in range(total_steps):
    env.step(action)
    if i == 0:
        env.render(simulate_index=0, request_idr=True)   # range first frame
    elif i == total_steps - 1:
        env.render(simulate_index=i, request_idr=True)   # range last frame
    else:
        env.render(simulate_index=i)

future = env.save_streaming(camera, "color", "/tmp/episode.mp4", 0, total_steps - 1)
result = future.result()
```

## New Interface Quick Start

```python
env = MyEnv(...)  # OrcaGymEulerEnv / OrcaGymLocalEnv subclass
env.reset()

# 1. Enumerate cameras
camera_names = env.get_camera_names()
camera = camera_names[0]

# 2. Start streaming in one call (auto-configures camera properties,
#    enables streaming, and starts client-side recorders)
env.start_streaming(
    camera,
    capture_rgb=True,        # add capture_depth=True for depth
    color_port=7070,
    width=640, height=480,
)

# 3. Simulation loop: render must receive an increasing simulate_index
#    (used for frame alignment and range extraction)
for i in range(100):
    env.step(action)
    if i == 0 or i == 99:
        env.render(simulate_index=i, request_idr=True)  # first/last frame of range: request IDR
    else:
        env.render(simulate_index=i)

# 4. Live preview (non-blocking, subprocess window; depth shown as raw grayscale)
env.show_camera(camera, camera_type="color")

# 5. Save a range to MP4 (non-blocking, returns a Future)
future = env.save_streaming(
    camera_name=camera,
    camera_type="color",                 # "color" or "depth"
    file_path="/tmp/episode.mp4",
    start_simulate_index=0,              # inclusive
    end_simulate_index=99,
)
result = future.result()                 # RemuxResult
print(result.file_path, result.frame_count, result.frame_indices)

env.close()  # automatically saves all pending recording tasks
```

## Key Behavior Notes

1. **Frame alignment**: the `simulate_index` passed to `render(simulate_index=...)` must increase monotonically; saved ranges are extracted by this index.
2. **IDR keyframe requests (important)**: when saving video, you must pass `request_idr=True` when rendering **both the first frame and the last frame of the recording range** (i.e., `render(simulate_index=i, request_idr=True)`). The first-frame IDR guarantees the output video decodes and plays correctly from the range start; the last-frame IDR guarantees the final frames of the range are fully decoded into the MP4 from the H.264 stream. Intermediate frames need no extra requests.
3. **Non-blocking saves**: `save_streaming` returns a `Future[RemuxResult]` immediately; a background worker thread performs the remux once the end frame arrives. `env.close()` blocks until pending tasks are saved.
4. **Multi-camera**: call `start_streaming` per camera (distinct ports); a single `render` triggers rendering for all cameras. Color and depth are independent streams, selected via `camera_type`.
5. **Advanced usage**: `manager = env.get_recorder_manager()` returns the `VideoRecorderManager`, which offers `submit_task` (per-frame callbacks via `SingleFrameTask`), `get_latest_frame_simulate_index`, and more — see the [Environment API](../api_reference/environment.md).

## Capturing a Single Frame

There are three ways to capture a single frame with the new interface; choose by use case:

### Option 1: `SingleFrameTask` per-frame callback (recommended; multi-camera sync)

The callback runs on a background thread once the target frame arrives and directly receives the decoded RGB array; the main thread does not block:

```python
from orca_gym.recorder import SingleFrameTask

manager = env.get_recorder_manager()

def on_frame(frame_entry, decoded_frame):
    # frame_entry.timestamp_ns / simulate_index / is_keyframe
    # decoded_frame: np.ndarray (H, W, 3) uint8 RGB, or None on decode failure
    return decoded_frame   # the return value becomes future.result()

task = SingleFrameTask(simulate_index=100, on_frame=on_frame)
future = manager.submit_task(camera, task)   # non-blocking
frame = future.result()   # None means no video frame for this sim_idx (render skipped it)
```

> When the control rate > render rate, the target `simulate_index` may have no
> corresponding video frame (rendered indices skip numbers). `future.result()`
> then returns `None` — a normal skip, not an error.

### Option 2: Read the latest decoded frame (debugging / single camera)

```python
frame = manager.get_last_decoded_frame(camera, "color")
# np.ndarray (H, W, 3) uint8, or None (no frame decoded yet)
```

> Note: this path is unlocked across recorders and the sim_idx is uncontrolled
> (you may read a frame that is too new or too old). For synchronized multi-camera
> capture, use Option 1.

## Dependencies

Recording and visualization require: `av` (PyAV), `websockets`, `numpy`; visualization additionally requires `matplotlib` and `opencv-python`.

## Migration Example

**Old style (deprecated):**

```python
from orca_gym.sensor.rgbd_camera import CameraWrapper

camera = CameraWrapper("front_camera", port=7070)
camera.start()
frame, idx = camera.get_frame(format="rgb24")
camera.stop()
```

**New style:**

```python
env.start_streaming("front_camera", capture_rgb=True, color_port=7070)
env.render(simulate_index=step_idx)
frame = env.get_recorder_manager().get_last_decoded_frame("front_camera", "color")
```

> For the full tutorial see [📷 Camera & Vision](../tutorial/beginners/camera-and-vision.md).
