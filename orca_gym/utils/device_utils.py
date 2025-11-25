"""
GPU 设备工具函数
支持 CUDA、MUSA 等多种 GPU 后端
"""
import torch
import os


def get_torch_device(try_to_use_gpu=True):
    """
    获取 PyTorch 设备，自动检测可用的 GPU 后端
    
    优先级：MUSA > CUDA > CPU
    
    Args:
        try_to_use_gpu (bool): 是否尝试使用 GPU
        
    Returns:
        device (torch.device): 可用的设备
    """
    if not try_to_use_gpu:
        return torch.device("cpu")
    
    # 优先检测 MUSA GPU
    try:
        import torch_musa
        if torch.musa.is_available():
            return torch.device("musa:0")
    except (ImportError, AttributeError):
        pass
    
    # 其次检测 CUDA GPU
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    
    # 最后使用 CPU
    return torch.device("cpu")


def get_gpu_info():
    """
    获取 GPU 信息
    
    Returns:
        dict: GPU 信息字典，包含：
            - device_type: "musa", "cuda", 或 "cpu"
            - device_name: GPU 名称
            - device_count: GPU 数量
            - available: 是否可用
            - debug_info: 调试信息（可选）
    """
    info = {
        "device_type": "cpu",
        "device_name": "CPU",
        "device_count": 0,
        "available": False,
    }
    
    # 检测 MUSA
    try:
        import torch_musa
        info["debug_info"] = {"torch_musa_imported": True, "torch_version": torch.__version__}
        
        # 检查 torch.musa 模块是否存在
        if hasattr(torch, 'musa'):
            musa_available = torch.musa.is_available()
            info["debug_info"]["torch_musa_module_exists"] = True
            info["debug_info"]["torch_musa_is_available"] = musa_available
            
            if musa_available:
                info["device_type"] = "musa"
                info["available"] = True
                info["device_count"] = torch.musa.device_count()
                if info["device_count"] > 0:
                    try:
                        info["device_name"] = torch.musa.get_device_name(0)
                    except Exception as e:
                        info["device_name"] = f"MUSA GPU (name unavailable: {e})"
                return info
            else:
                info["debug_info"]["reason"] = "torch.musa.is_available() returned False"
        else:
            info["debug_info"]["torch_musa_module_exists"] = False
            info["debug_info"]["reason"] = "torch.musa module not found after importing torch_musa"
    except ImportError as e:
        error_msg = str(e)
        info["debug_info"] = {
            "torch_musa_imported": False, 
            "error": error_msg,
            "torch_version": torch.__version__
        }
        # 检查是否是符号未定义错误（通常是版本不匹配或安装问题）
        if "undefined symbol" in error_msg or "symbol" in error_msg.lower():
            info["debug_info"]["likely_cause"] = "Symbol mismatch (version/installation issue)"
            info["debug_info"]["suggestion"] = (
                f"torch_musa 2.0.1 requires PyTorch 2.2.0. "
                f"Current PyTorch version: {torch.__version__}. "
                f"Please install from local wheel files: "
                f"torch-2.2.0, torch_musa-2.0.1, torchvision-0.17.2, torchaudio-2.2.2"
            )
        elif "Please try running Python from a different directory" in error_msg:
            info["debug_info"]["likely_cause"] = "Import path issue"
            info["debug_info"]["suggestion"] = "Try running from a different directory or check installation"
    except AttributeError as e:
        info["debug_info"] = {"torch_musa_imported": True, "attribute_error": str(e)}
    except Exception as e:
        info["debug_info"] = {"torch_musa_imported": True, "unexpected_error": str(e)}
    
    # 检测 CUDA
    if torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["available"] = True
        info["device_count"] = torch.cuda.device_count()
        if info["device_count"] > 0:
            info["device_name"] = torch.cuda.get_device_name(0)
        return info
    
    return info


def print_gpu_info():
    """打印 GPU 信息"""
    info = get_gpu_info()
    print(f"GPU 类型: {info['device_type']}")
    print(f"GPU 可用: {info['available']}")
    if info['available']:
        print(f"GPU 数量: {info['device_count']}")
        print(f"GPU 名称: {info['device_name']}")
    else:
        print("使用 CPU")
        # 打印调试信息
        if 'debug_info' in info:
            debug = info['debug_info']
            print(f"[DEBUG] torch_musa 导入状态: {debug.get('torch_musa_imported', 'unknown')}")
            if 'torch_version' in debug:
                print(f"[DEBUG] PyTorch 版本: {debug['torch_version']}")
            if 'likely_cause' in debug:
                print(f"[DEBUG] 可能原因: {debug['likely_cause']}")
            if 'suggestion' in debug:
                print(f"[DEBUG] 建议: {debug['suggestion']}")
            if 'torch_musa_module_exists' in debug:
                print(f"[DEBUG] torch.musa 模块存在: {debug['torch_musa_module_exists']}")
            if 'torch_musa_is_available' in debug:
                print(f"[DEBUG] torch.musa.is_available(): {debug['torch_musa_is_available']}")
            if 'reason' in debug:
                print(f"[DEBUG] 原因: {debug['reason']}")
            if 'error' in debug:
                error_msg = debug['error']
                # 截断过长的错误信息
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                print(f"[DEBUG] 导入错误: {error_msg}")


def clear_gpu_cache():
    """清理 GPU 缓存（支持 MUSA 和 CUDA）"""
    try:
        import torch_musa
        if torch.musa.is_available():
            torch.musa.empty_cache()
            return
    except (ImportError, AttributeError):
        pass
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device_string(device=None):
    """
    获取设备字符串表示
    
    Args:
        device (torch.device, optional): 设备对象，如果为 None 则自动检测
        
    Returns:
        str: 设备字符串，如 "musa:0", "cuda:0", "cpu"
    """
    if device is None:
        device = get_torch_device(try_to_use_gpu=True)
    return str(device)

