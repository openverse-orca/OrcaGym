#!/usr/bin/env python3
"""
生成gRPC Python代码的脚本
"""

import os
import subprocess
import sys
from pathlib import Path


def generate_grpc_code():
    """生成gRPC Python代码"""
    current_dir = Path(__file__).parent
    proto_file = current_dir / "inference.proto"
    
    if not proto_file.exists():
        print(f"Error: {proto_file} not found!")
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
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
        
        if result.returncode == 0:
            print("Successfully generated gRPC Python code!")
            print("Generated files:")
            for file in current_dir.glob("inference_pb2*.py"):
                print(f"  - {file}")
            return True
        else:
            print(f"Error generating gRPC code:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Error: grpc_tools.protoc not found!")
        print("Please install grpcio-tools:")
        print("  pip install grpcio-tools")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def check_dependencies():
    """检查依赖是否安装"""
    try:
        import grpc
        import grpc_tools
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("  pip install grpcio grpcio-tools")
        return False


if __name__ == "__main__":
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("Generating gRPC Python code...")
    if generate_grpc_code():
        print("Done!")
    else:
        print("Failed to generate gRPC code!")
        sys.exit(1) 