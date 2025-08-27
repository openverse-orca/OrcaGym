#!/usr/bin/env python3
"""
Example script demonstrating how to use the fake camera.
This script either uses an existing video file or creates a synthetic video for testing.
"""

import cv2
import numpy as np
import time
import os
from fake_rgbd_camera import FakeCameraWrapper, FakeMonitor


def create_synthetic_video(output_path: str, duration: int = 10, fps: int = 30):
    """
    Create a synthetic video for testing purposes.
    
    Args:
        output_path: Path to save the synthetic video
        duration: Duration in seconds
        fps: Frames per second
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating synthetic video: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Resolution: {width}x{height}")
    
    for frame_num in range(duration * fps):
        # Create a moving pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a moving circle
        center_x = int(width/2 + 100 * np.sin(frame_num * 0.1))
        center_y = int(height/2 + 50 * np.cos(frame_num * 0.15))
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = frame_num / fps
        cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add a moving rectangle
        rect_x = int(50 + 200 * np.sin(frame_num * 0.05))
        rect_y = int(100 + 100 * np.cos(frame_num * 0.08))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 80, rect_y + 60), (255, 0, 0), -1)
        
        out.write(frame)
    
    out.release()
    print(f"Synthetic video created: {output_path}")


def test_fake_camera_basic():
    """Basic test of the fake camera functionality."""
    print("=== Basic Fake Camera Test ===")
    
    # Create synthetic video if it doesn't exist
    video_path = "synthetic_test_video.mp4"
    if not os.path.exists(video_path):
        create_synthetic_video(video_path, duration=5, fps=30)
    
    try:
        # Create fake camera
        camera = FakeCameraWrapper("test_camera", video_path, loop_video=True, fps=30)
        
        # Get video info
        info = camera.get_video_info()
        print(f"Video info: {info}")
        
        # Start camera
        camera.start()
        
        # Wait for first frame
        print("Waiting for first frame...")
        while not camera.is_first_frame_received():
            time.sleep(0.1)
        print("First frame received!")
        
        # Get a few frames
        print("Getting frames...")
        for i in range(10):
            frame, index = camera.get_frame(format='rgb24')
            print(f"Frame {index}: shape={frame.shape}, dtype={frame.dtype}")
            time.sleep(0.1)
        
        # Test different formats
        print("Testing different formats...")
        frame_bgr, _ = camera.get_frame(format='bgr24')
        frame_rgb, _ = camera.get_frame(format='rgb24')
        print(f"BGR frame shape: {frame_bgr.shape}")
        print(f"RGB frame shape: {frame_rgb.shape}")
        
        # Test resizing
        frame_resized, _ = camera.get_frame(format='rgb24', size=(320, 240))
        print(f"Resized frame shape: {frame_resized.shape}")
        
        camera.stop()
        print("Basic test completed successfully!")
        
    except Exception as e:
        print(f"Basic test failed: {e}")


def test_fake_camera_monitor():
    """Test the fake camera with matplotlib monitor."""
    print("\n=== Fake Camera Monitor Test ===")
    
    # Create synthetic video if it doesn't exist
    video_path = "synthetic_test_video.mp4"
    if not os.path.exists(video_path):
        create_synthetic_video(video_path, duration=10, fps=30)
    
    try:
        # Create and start monitor
        monitor = FakeMonitor("test_monitor", video_path, fps=30, loop_video=True)
        
        print("Starting monitor... Press 'q' in the matplotlib window to close.")
        print("The monitor will display the synthetic video in a matplotlib window.")
        
        # Start the monitor (this will block until window is closed)
        monitor.start()
        
        print("Monitor test completed!")
        
    except Exception as e:
        print(f"Monitor test failed: {e}")


def test_with_existing_video(video_path: str):
    """Test with an existing video file."""
    print(f"\n=== Testing with existing video: {video_path} ===")
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    try:
        # Create fake camera
        camera = FakeCameraWrapper("existing_video", video_path, loop_video=False, fps=30)
        
        # Get video info
        info = camera.get_video_info()
        print(f"Video info: {info}")
        
        # Start camera
        camera.start()
        
        # Wait for first frame
        while not camera.is_first_frame_received():
            time.sleep(0.1)
        
        print("Playing video (non-looping)...")
        frame_count = 0
        while camera.running:
            frame, index = camera.get_frame(format='rgb24')
            print(f"Frame {index}: shape={frame.shape}")
            frame_count += 1
            
            # Stop after 50 frames for demo
            if frame_count >= 50:
                break
            
            time.sleep(0.1)
        
        camera.stop()
        print("Existing video test completed!")
        
    except Exception as e:
        print(f"Existing video test failed: {e}")


def main():
    """Main function to run all tests."""
    print("Fake Camera Example Script")
    print("=" * 50)
    
    # Run basic test
    test_fake_camera_basic()
    
    # Test with existing video if provided
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        test_with_existing_video(video_path)
    
    # Ask user if they want to run monitor test
    print("\n" + "=" * 50)
    response = input("Do you want to run the monitor test? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        test_fake_camera_monitor()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main() 