# 📝 Release Notes

## 26.9.1

### ⚠️ Breaking Change: Camera Interface Refactor (rgbd_camera → recorder)

This release introduces breaking changes to the camera streaming, recording, and visualization interfaces. **Recording has moved from the engine side to the Python client side:**

**Removed**

- Engine-side MP4 recording gRPC calls: `StartCameraRecording` / `StopCameraRecording`, etc.
- gRPC camera calls: `QueryCameras` / `ConfigureCamera` / `StartCameraStream` / `StopCameraStream`

**Deprecated (kept for compatibility; emits DeprecationWarning and no-ops)**

- The `orca_gym.sensor.rgbd_camera` module: `CameraWrapper` / `Monitor` / `CameraCacher` / `CameraDataParser` / `VideoPlayer`
- Environment-level legacy recording APIs: `begin_save_video` / `stop_save_video` / `get_current_frame` / `get_next_frame` / `get_camera_time_stamp`

**Added**

- The `orca_gym.recorder` module (based on PyAV), uniformly managing streaming, recording, saving, and live visualization:
  - `env.start_streaming(camera_name, ...)` — one-call config + streaming + recorder startup
  - `env.save_streaming(camera_name, camera_type, file_path, start_simulate_index, end_simulate_index)` — save a range to standard MP4 (non-blocking, returns `Future[RemuxResult]`)
  - `env.show_camera(camera_name)` — non-blocking live preview window
  - `env.get_recorder_manager()` — the `VideoRecorderManager`, supporting multi-camera synchronized capture via `SingleFrameTask` per-frame callbacks
  - `env.get_camera_names()` / `manager.set_camera_properties(...)` / `manager.set_streaming_enabled(...)` replace the old gRPC camera configuration
- Recording outputs standard MP4 (replacing the legacy `.h264` + timestamp `.bin`), with a built-in frame-index ↔ physics-step mapping (`frame_indices` / `timestamps_ns`)

**Usage Notes**

1. The `simulate_index` passed to `render(simulate_index=...)` must increase monotonically; it is used for frame alignment and save-range extraction
2. **When saving video, you must pass `request_idr=True` when rendering both the first and the last frame of the recording range** (the first-frame IDR ensures the video plays from the range start; the last-frame IDR ensures the final frames are fully written to the MP4)
3. New dependencies: `av` (PyAV), `websockets`; live preview additionally requires `matplotlib` and `opencv-python`

📋 For detailed migration steps see the [Camera Interface Migration Guide](../migration/camera-recorder.md); for full usage of the new interface see the
[📷 Camera & Vision tutorial](../tutorial/beginners/camera-and-vision.md).
