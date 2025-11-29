import os
import sys
import time
import argparse
import subprocess
import signal
from pathlib import Path

from orca_gym.sensor.rgbd_camera import Monitor

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



def start_monitor(port=7070):
    """
    启动 monitor.py 作为子进程。
    """
    monitor_script = Path(__file__).resolve()
    _logger.info(f"monitor_script:  {monitor_script}")

    # 启动 monitor.py
    # 使用 sys.executable 确保使用相同的 Python 解释器
    process = subprocess.Popen(
        [sys.executable, monitor_script, "--port", f"{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return process

def terminate_monitor(process):
    """
    终止子进程。
    """
    try:
        if os.name != 'nt':
            # Unix/Linux: 发送 SIGTERM 给整个进程组
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            # Windows: 使用 terminate 方法
            process.terminate()
    except Exception as e:
        print(f"终止子进程时发生错误: {e}")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7070, help="Port number for the camera monitor.")
    
    args = parser.parse_args()
    port = args.port
    
    
    monitor = Monitor(name="camera", port=port)
    try:
        _logger.info("Start monitoring...")
        monitor.start()
    except KeyboardInterrupt:
        monitor.stop()
        _logger.info("Program terminated by user.")
