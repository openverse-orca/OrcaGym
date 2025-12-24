#!/usr/bin/env python3
"""
生成gRPC Python代码的脚本
"""

import os
import subprocess
import sys
from pathlib import Path

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



def generate_grpc_code():
    """生成gRPC Python代码"""
    current_dir = Path(__file__).parent
    proto_file = current_dir / "inference.proto"
    
    if not proto_file.exists():
        _logger.error(f"Error: {proto_file} not found!")
        return False
    
    try:
        # 生成Python代码
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--python_out={current_dir}",
            f"--grpc_python_out={current_dir}",
            f"--proto_path={current_dir}",
            str(proto_file)
        ]
        
        _logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
        
        if result.returncode == 0:
            _logger.info("Successfully generated gRPC Python code!")
            _logger.info("Generated files:")
            for file in current_dir.glob("inference_pb2*.py"):
                _logger.info(f"  - {file}")
            return True
        else:
            _logger.error(f"Error generating gRPC code:")
            _logger.info(f"stdout: {result.stdout}")
            _logger.info(f"stderr: {result.stderr}")
            return False
            
    except FileNotFoundError:
        _logger.error("Error: grpc_tools.protoc not found!")
        _logger.info("Please install grpcio-tools:")
        _logger.info("  pip install grpcio-tools")
        return False
    except Exception as e:
        _logger.error(f"Unexpected error: {e}")
        return False


def check_dependencies():
    """检查依赖是否安装"""
    try:
        import grpc
        import grpc_tools
        return True
    except ImportError as e:
        _logger.info(f"Missing dependency: {e}")
        _logger.info("Please install required packages:")
        _logger.info("  pip install grpcio grpcio-tools")
        return False


if __name__ == "__main__":
    _logger.info("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    _logger.info("Generating gRPC Python code...")
    if generate_grpc_code():
        _logger.info("Done!")
    else:
        _logger.error("Failed to generate gRPC code!")
        sys.exit(1) 