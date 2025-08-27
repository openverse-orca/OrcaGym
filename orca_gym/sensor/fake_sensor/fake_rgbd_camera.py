import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple


class FakeCameraWrapper:
    """
    A fake camera wrapper that reads from video files instead of websockets.
    Implements the same API as CameraWrapper for compatibility.
    """
    
    def __init__(self, name: str, video_path: str, loop_video: bool = True, fps: float = 30.0):
        """
        Initialize the fake camera.
        
        Args:
            name: Camera name
            video_path: Path to the video file
            loop_video: Whether to loop the video when it ends
            fps: Target FPS for playback (if None, uses original video FPS)
        """
        self._name = name
        self.video_path = video_path
        self.loop_video = loop_video
        self.target_fps = fps
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use target FPS if specified, otherwise use original FPS
        self.current_fps = self.target_fps if self.target_fps else self.original_fps
        self.frame_interval = 1.0 / self.current_fps
        
        # Initialize state
        self.image = np.random.randint(0, 255, size=(self.height, self.width, 3), dtype=np.uint8)
        self.enabled = True
        self.received_first_frame = False
        self.image_index = 0
        self.running = False
        self.thread = None
        
        # Timing control
        self.last_frame_time = 0
        
    def __del__(self):
        """Cleanup when object is destroyed."""
        if not self.enabled:
            return
        self.stop()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    @property
    def name(self):
        return self._name
    
    def is_first_frame_received(self):
        return self.received_first_frame
    
    def start(self):
        """Start the video playback thread."""
        if not self.enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()
    
    def loop(self):
        """Main loop for video playback."""
        while self.running:
            current_time = time.time()
            
            # Control frame rate
            if current_time - self.last_frame_time >= self.frame_interval:
                ret, frame = self.cap.read()
                
                if ret:
                    # Convert BGR to RGB for consistency
                    self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.image_index += 1
                    
                    if not self.received_first_frame:
                        self.received_first_frame = True
                    
                    self.last_frame_time = current_time
                else:
                    # End of video reached
                    if self.loop_video:
                        # Reset to beginning
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        # Stop playback
                        self.running = False
                        break
            else:
                # Sleep to maintain frame rate
                time.sleep(0.001)  # 1ms sleep
    
    def stop(self):
        """Stop the video playback."""
        if not self.enabled:
            return
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
    
    def get_frame(self, format='bgr24', size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, int]:
        """
        Get the current frame.
        
        Args:
            format: Output format ('bgr24' or 'rgb24')
            size: Optional resize dimensions (width, height)
            
        Returns:
            Tuple of (frame, frame_index)
        """
        if format == 'bgr24':
            frame = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        elif format == 'rgb24':
            frame = self.image.copy()
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        if size is not None:
            frame = cv2.resize(frame, size)
            
        return frame, self.image_index
    
    def reset(self):
        """Reset video to beginning."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.image_index = 0
    
    def get_video_info(self):
        """Get video information."""
        return {
            'path': self.video_path,
            'width': self.width,
            'height': self.height,
            'original_fps': self.original_fps,
            'current_fps': self.current_fps,
            'frame_count': self.frame_count,
            'loop_video': self.loop_video
        }


class FakeCameraCacher:
    """
    A fake camera cacher that simulates the caching behavior but doesn't actually cache.
    Useful for testing code that expects a CameraCacher interface.
    """
    
    def __init__(self, name: str, video_path: str):
        self.name = name
        self.video_path = video_path
        self.received_first_frame = False
        self.running = False
        self.thread = None
    
    def __del__(self):
        self.stop()
    
    def start(self):
        """Simulate starting the cache process."""
        self.running = True
        self.received_first_frame = True
        # In a real implementation, this would start caching
        # For fake implementation, we just set the flag
    
    def stop(self):
        """Stop the cache process."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
    
    def is_first_frame_received(self):
        return self.received_first_frame


class FakeMonitor:
    """
    A fake monitor that displays video from a file using matplotlib.
    """
    
    def __init__(self, name: str, video_path: str, fps=30, loop_video=True):
        self.camera = FakeCameraWrapper(name, video_path, loop_video, fps)
        self.camera.start()
        
        self.fps = fps
        self.interval = 1000 / self.fps  # Update interval in milliseconds
        
        # Create matplotlib figure and axis
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')  # Turn off axes
        
        frame, _ = self.camera.get_frame(format='rgb24')
        
        # Display initial image
        self.im = self.ax.imshow(frame)
        self.ax.set_title(f"Fake Camera Feed: {name}")
    
    def update(self, frame_num):
        """Update function for matplotlib animation."""
        frame, _ = self.camera.get_frame(format='rgb24')
        
        # Update image data
        self.im.set_data(frame)
        return self.im,
    
    def start(self):
        """Start the video display."""
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=self.interval,
            blit=True
        )
        plt.show()
    
    def stop(self):
        """Stop the video display."""
        self.camera.stop()
        import matplotlib.pyplot as plt
        plt.close(self.fig)
    
    def __del__(self):
        self.stop()


# Example usage and test function
def test_fake_camera():
    """Test function to demonstrate fake camera usage."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    # You would replace this with an actual video file path
    video_path = "path/to/your/video.mp4"  # Replace with actual video path
    
    try:
        # Create fake camera
        camera = FakeCameraWrapper("test_camera", video_path, loop_video=True, fps=30)
        camera.start()
        
        # Wait for first frame
        while not camera.is_first_frame_received():
            time.sleep(0.1)
        
        print("Fake camera started successfully!")
        print(f"Video info: {camera.get_video_info()}")
        
        # Get a few frames
        for i in range(10):
            frame, index = camera.get_frame(format='rgb24')
            print(f"Frame {index}: shape={frame.shape}")
            time.sleep(0.1)
        
        camera.stop()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_fake_camera() 