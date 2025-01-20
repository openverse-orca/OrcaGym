import os
import sys
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

if project_root not in sys.path:
    sys.path.append(project_root)


from orca_gym.sensor.rgbd_camera import Monitor


if __name__ == "__main__":
    monitor = Monitor("camera")
    try:
        print("Start monitoring...")
        monitor.start()
    except KeyboardInterrupt:
        monitor.stop()
        print("Program terminated by user.")
