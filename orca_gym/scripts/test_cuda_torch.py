#!/usr/bin/env python3
"""
测试CUDA和torch是否正确安装和启用的脚本
用于验证环境配置是否满足运行legged_gym的要求
"""

import sys
import subprocess
import platform

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def check_python_version():
    """检查Python版本"""
    _logger.info("=" * 50)
    _logger.info("Python版本检查")
    _logger.info("=" * 50)
    _logger.info(f"Python版本: {sys.version}")
    _logger.info(f"Python路径: {sys.executable}")
    
    if sys.version_info < (3, 7):
        _logger.info("❌ Python版本过低，建议使用Python 3.7或更高版本")
        return False
    else:
        _logger.info("✅ Python版本符合要求")
        return True

def check_cuda_installation():
    """检查CUDA安装"""
    _logger.info("\n" + "=" * 50)
    _logger.info("CUDA安装检查")
    _logger.info("=" * 50)
    
    # 检查nvidia-smi命令
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            _logger.info("✅ nvidia-smi命令可用")
            _logger.info("GPU信息:")
            _logger.info(result.stdout)
        else:
            _logger.info("❌ nvidia-smi命令不可用")
            return False
    except FileNotFoundError:
        _logger.info("❌ 未找到nvidia-smi命令，可能CUDA未安装或PATH中未包含")
        return False
    except subprocess.TimeoutExpired:
        _logger.info("❌ nvidia-smi命令执行超时")
        return False
    
    # 检查CUDA版本
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            _logger.info("✅ nvcc命令可用")
            # 提取CUDA版本
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    _logger.info(f"CUDA版本: {line.strip()}")
                    break
        else:
            _logger.info("❌ nvcc命令不可用")
            return False
    except FileNotFoundError:
        _logger.info("❌ 未找到nvcc命令，可能CUDA toolkit未安装或PATH中未包含")
        return False
    except subprocess.TimeoutExpired:
        _logger.info("❌ nvcc命令执行超时")
        return False
    
    return True

def check_torch_installation():
    """检查torch安装"""
    _logger.info("\n" + "=" * 50)
    _logger.info("PyTorch安装检查")
    _logger.info("=" * 50)
    
    try:
        import torch
        _logger.info(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            _logger.info("✅ PyTorch CUDA支持已启用")
            _logger.info(f"CUDA版本: {torch.version.cuda}")
            _logger.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
            _logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
            
            # 显示GPU信息
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                _logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # 测试CUDA张量操作
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.mm(x, y)
                _logger.info("✅ CUDA张量运算测试通过")
            except Exception as e:
                _logger.info(f"❌ CUDA张量运算测试失败: {e}")
                return False
                
        else:
            _logger.info("❌ PyTorch CUDA支持未启用")
            _logger.info("可能的原因:")
            _logger.info("1. 安装的是CPU版本的PyTorch")
            _logger.info("2. CUDA版本与PyTorch不匹配")
            _logger.info("3. 环境变量配置问题")
            return False
            
    except ImportError:
        _logger.info("❌ PyTorch未安装")
        return False
    except Exception as e:
        _logger.info(f"❌ PyTorch检查过程中出现错误: {e}")
        return False
    
    return True

def check_system_info():
    """检查系统信息"""
    _logger.info("\n" + "=" * 50)
    _logger.info("系统信息")
    _logger.info("=" * 50)
    _logger.info(f"操作系统: {platform.system()}")
    _logger.info(f"系统版本: {platform.release()}")
    _logger.info(f"架构: {platform.machine()}")
    _logger.info(f"处理器: {platform.processor()}")

def main():
    """主函数"""
    _logger.info("CUDA和PyTorch环境检查工具")
    _logger.info("用于验证legged_gym运行环境")
    
    # 检查系统信息
    check_system_info()
    
    # 检查Python版本
    python_ok = check_python_version()
    
    # 检查CUDA安装
    cuda_ok = check_cuda_installation()
    
    # 检查PyTorch安装
    torch_ok = check_torch_installation()
    
    # 总结
    _logger.info("\n" + "=" * 50)
    _logger.info("检查结果总结")
    _logger.info("=" * 50)
    
    if python_ok and cuda_ok and torch_ok:
        _logger.info("🎉 所有检查都通过了！环境配置正确，可以继续后续步骤。")
        return True
    else:
        _logger.info("⚠️  存在一些问题，请根据上述提示进行修复：")
        if not python_ok:
            _logger.info("- 升级Python版本")
        if not cuda_ok:
            _logger.info("- 安装或配置CUDA")
        if not torch_ok:
            _logger.info("- 重新安装匹配的PyTorch版本")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 