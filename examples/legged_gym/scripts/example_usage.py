#!/usr/bin/env python3
"""
gRPC服务器使用示例
"""

import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def example_usage():
    """使用示例"""
    print("=" * 50)
    print("gRPC服务器使用示例")
    print("=" * 50)
    
    # 配置文件路径
    config_path = "../configs/lite3_sim_config.yaml"
    
    print(f"\n1. 配置文件: {config_path}")
    
    # 读取配置文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✓ 配置文件加载成功")
        
        # 显示ONNX模型配置
        if 'model_file' in config and 'onnx' in config['model_file']:
            print("\n2. ONNX模型配置:")
            for model_name, model_path in config['model_file']['onnx'].items():
                print(f"   {model_name}: {model_path}")
        else:
            print("✗ 未找到ONNX模型配置")
            return
        
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return
    
    print("\n3. 启动服务器命令:")
    print("   cd examples/legged_gym/scripts")
    print(f"   python grpc_server.py --config {config_path} --port 50051 --verbose")
    
    print("\n4. 测试服务器命令:")
    print("   python test_grpc_server.py --config ../configs/lite3_sim_config.yaml --port 50051 --no-models")
    
    print("\n5. 运行仿真命令:")
    print("   cd examples/legged_gym")
    print("   python run_legged_sim.py --config configs/lite3_sim_config.yaml")
    
    print("\n6. 键盘控制:")
    print("   M - 切换模型类型 (sb3 ↔ onnx ↔ grpc)")
    print("   Space - 切换地形")
    print("   R - 重置机器人")
    print("   WASD - 控制移动")
    
    print("\n" + "=" * 50)
    print("使用步骤:")
    print("1. 确保ONNX模型文件存在")
    print("2. 启动gRPC服务器")
    print("3. 测试服务器连接")
    print("4. 运行仿真")
    print("=" * 50)

def check_dependencies():
    """检查依赖"""
    print("检查依赖...")
    
    dependencies = {
        "grpcio": "gRPC通信",
        "grpcio-tools": "gRPC代码生成",
        "onnxruntime": "ONNX模型推理",
        "pyyaml": "YAML配置文件解析"
    }
    
    missing = []
    for dep, desc in dependencies.items():
        try:
            __import__(dep.replace("-", "_"))
            print(f"✓ {dep}: {desc}")
        except ImportError:
            print(f"✗ {dep}: {desc}")
            missing.append(dep)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    return True

def check_files():
    """检查必要文件"""
    print("\n检查文件...")
    
    files = [
        "grpc_server.py",
        "grpc_client.py", 
        "test_grpc_server.py",
        "start_grpc_server.sh",
        "proto/inference.proto",
        "proto/inference_pb2.py",
        "proto/inference_pb2_grpc.py"
    ]
    
    missing = []
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")
            missing.append(file)
    
    if missing:
        print(f"\n缺少文件: {', '.join(missing)}")
        return False
    
    return True

def main():
    """主函数"""
    print("gRPC服务器环境检查")
    print("=" * 50)
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    # 检查文件
    files_ok = check_files()
    
    if deps_ok and files_ok:
        print("\n✓ 环境检查通过")
        example_usage()
    else:
        print("\n✗ 环境检查失败，请解决上述问题后重试")

if __name__ == "__main__":
    main() 