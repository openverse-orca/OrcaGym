#!/usr/bin/env python3
"""
测试CUDA和torch是否正确安装和启用的脚本
用于验证环境配置是否满足运行legged_gym的要求
"""

import sys
import subprocess
import platform

def check_python_version():
    """检查Python版本"""
    print("=" * 50)
    print("Python版本检查")
    print("=" * 50)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    if sys.version_info < (3, 7):
        print("❌ Python版本过低，建议使用Python 3.7或更高版本")
        return False
    else:
        print("✅ Python版本符合要求")
        return True

def check_cuda_installation():
    """检查CUDA安装"""
    print("\n" + "=" * 50)
    print("CUDA安装检查")
    print("=" * 50)
    
    # 检查nvidia-smi命令
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi命令可用")
            print("GPU信息:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi命令不可用")
            return False
    except FileNotFoundError:
        print("❌ 未找到nvidia-smi命令，可能CUDA未安装或PATH中未包含")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi命令执行超时")
        return False
    
    # 检查CUDA版本
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvcc命令可用")
            # 提取CUDA版本
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"CUDA版本: {line.strip()}")
                    break
        else:
            print("❌ nvcc命令不可用")
            return False
    except FileNotFoundError:
        print("❌ 未找到nvcc命令，可能CUDA toolkit未安装或PATH中未包含")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvcc命令执行超时")
        return False
    
    return True

def check_torch_installation():
    """检查torch安装"""
    print("\n" + "=" * 50)
    print("PyTorch安装检查")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA支持已启用")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"可用GPU数量: {torch.cuda.device_count()}")
            
            # 显示GPU信息
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # 测试CUDA张量操作
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.mm(x, y)
                print("✅ CUDA张量运算测试通过")
            except Exception as e:
                print(f"❌ CUDA张量运算测试失败: {e}")
                return False
                
        else:
            print("❌ PyTorch CUDA支持未启用")
            print("可能的原因:")
            print("1. 安装的是CPU版本的PyTorch")
            print("2. CUDA版本与PyTorch不匹配")
            print("3. 环境变量配置问题")
            return False
            
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    except Exception as e:
        print(f"❌ PyTorch检查过程中出现错误: {e}")
        return False
    
    return True

def check_system_info():
    """检查系统信息"""
    print("\n" + "=" * 50)
    print("系统信息")
    print("=" * 50)
    print(f"操作系统: {platform.system()}")
    print(f"系统版本: {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"处理器: {platform.processor()}")

def main():
    """主函数"""
    print("CUDA和PyTorch环境检查工具")
    print("用于验证legged_gym运行环境")
    
    # 检查系统信息
    check_system_info()
    
    # 检查Python版本
    python_ok = check_python_version()
    
    # 检查CUDA安装
    cuda_ok = check_cuda_installation()
    
    # 检查PyTorch安装
    torch_ok = check_torch_installation()
    
    # 总结
    print("\n" + "=" * 50)
    print("检查结果总结")
    print("=" * 50)
    
    if python_ok and cuda_ok and torch_ok:
        print("🎉 所有检查都通过了！环境配置正确，可以继续后续步骤。")
        return True
    else:
        print("⚠️  存在一些问题，请根据上述提示进行修复：")
        if not python_ok:
            print("- 升级Python版本")
        if not cuda_ok:
            print("- 安装或配置CUDA")
        if not torch_ok:
            print("- 重新安装匹配的PyTorch版本")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 