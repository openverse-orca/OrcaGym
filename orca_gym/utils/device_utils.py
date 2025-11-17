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
        if torch.musa.is_available():
            info["device_type"] = "musa"
            info["available"] = True
            info["device_count"] = torch.musa.device_count()
            if info["device_count"] > 0:
                info["device_name"] = torch.musa.get_device_name(0)
            return info
    except (ImportError, AttributeError):
        pass
    
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

