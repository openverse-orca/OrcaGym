# Fake RGBD Camera

This module provides a fake camera implementation that follows the same API as the original RGBD camera but reads from video files instead of websockets. This is useful for testing and development purposes when you don't have access to the actual camera hardware.

## Features

- **API Compatibility**: Implements the same interface as `CameraWrapper` from `rgbd_camera.py`
- **Video Playback**: Reads from video files (MP4, AVI, etc.) using OpenCV
- **Frame Rate Control**: Configurable playback speed
- **Looping Support**: Option to loop videos continuously
- **Multiple Formats**: Support for both BGR and RGB color formats
- **Resizing**: Optional frame resizing
- **Threading**: Thread-safe implementation with proper cleanup

## Classes

### FakeCameraWrapper

The main fake camera class that mimics the behavior of `CameraWrapper`.

```python
from fake_rgbd_camera import FakeCameraWrapper

# Create a fake camera
camera = FakeCameraWrapper(
    name="test_camera",
    video_path="path/to/video.mp4",
    loop_video=True,
    fps=30.0
)

# Start the camera
camera.start()

# Get frames
frame, index = camera.get_frame(format='rgb24')
frame_bgr, index = camera.get_frame(format='bgr24', size=(320, 240))

# Stop the camera
camera.stop()
```

#### Constructor Parameters

- `name` (str): Camera name identifier
- `video_path` (str): Path to the video file
- `loop_video` (bool): Whether to loop the video when it ends (default: True)
- `fps` (float): Target playback FPS (default: 30.0, use None for original video FPS)

#### Methods

- `start()`: Start video playback thread
- `stop()`: Stop video playback and cleanup
- `get_frame(format='bgr24', size=None)`: Get current frame
- `is_first_frame_received()`: Check if first frame has been loaded
- `reset()`: Reset video to beginning
- `get_video_info()`: Get video metadata

### FakeCameraCacher

A mock implementation of `CameraCacher` for testing code that expects caching behavior.

```python
from fake_rgbd_camera import FakeCameraCacher

cacher = FakeCameraCacher("test_cache", "path/to/video.mp4")
cacher.start()
# Simulates caching behavior without actually caching
cacher.stop()
```

### FakeMonitor

A matplotlib-based monitor for displaying fake camera feeds.

```python
from fake_rgbd_camera import FakeMonitor

monitor = FakeMonitor("test_monitor", "path/to/video.mp4", fps=30, loop_video=True)
monitor.start()  # Opens matplotlib window with video display
```

## Usage Examples

### Basic Usage

```python
import time
from fake_rgbd_camera import FakeCameraWrapper

# Create fake camera
camera = FakeCameraWrapper("my_camera", "video.mp4", loop_video=True, fps=30)

# Start camera
camera.start()

# Wait for first frame
while not camera.is_first_frame_received():
    time.sleep(0.1)

# Get frames
for i in range(100):
    frame, index = camera.get_frame(format='rgb24')
    print(f"Frame {index}: shape={frame.shape}")
    time.sleep(0.1)

# Cleanup
camera.stop()
```

### With Existing Code

You can easily replace the real camera with the fake camera in existing code:

```python
# Instead of:
# from rgbd_camera import CameraWrapper
# camera = CameraWrapper("camera", 7070)

# Use:
from fake_rgbd_camera import FakeCameraWrapper
camera = FakeCameraWrapper("camera", "test_video.mp4")

# The rest of your code remains the same
camera.start()
frame, index = camera.get_frame()
camera.stop()
```

### Creating Synthetic Videos

The example script includes a function to create synthetic test videos:

```python
from fake_camera_example import create_synthetic_video

# Create a 10-second synthetic video at 30 FPS
create_synthetic_video("test_video.mp4", duration=10, fps=30)
```

## Running the Example

```bash
# Run the example script
cd OrcaGym/orca_gym/sensor
python fake_camera_example.py

# Test with an existing video file
python fake_camera_example.py path/to/your/video.mp4
```

## Dependencies

- OpenCV (`cv2`)
- NumPy
- Matplotlib (for FakeMonitor)
- Threading (built-in)

## Video Format Support

The fake camera supports all video formats that OpenCV can read, including:
- MP4
- AVI
- MOV
- MKV
- And many others

## Performance Considerations

- The fake camera uses threading to simulate real-time video playback
- Frame rate control is implemented using timing, not actual video FPS
- For high-performance applications, consider using the original video FPS (`fps=None`)
- Memory usage depends on video resolution and frame rate

## Error Handling

The fake camera includes proper error handling:
- Invalid video file paths
- Unsupported video formats
- Thread cleanup on errors
- Graceful shutdown

## Testing

The module includes comprehensive tests in `fake_camera_example.py`:
- Basic functionality test
- Format conversion test
- Resizing test
- Monitor display test
- Error handling test

Run the tests to verify everything works correctly in your environment. 