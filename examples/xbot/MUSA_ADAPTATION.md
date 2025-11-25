# XBot MUSA 显卡适配说明

## 概述

XBot 项目已适配支持 MUSA（摩尔线程）显卡。代码会自动检测可用的 GPU 类型（优先级：MUSA > CUDA > CPU）。

## 设备检测优先级

1. **MUSA GPU**（摩尔线程显卡）
2. **CUDA GPU**（NVIDIA 显卡）
3. **CPU**（默认回退）

## 使用方法

### 方法 1: 自动检测（推荐）

```bash
# 自动检测可用GPU（优先MUSA，其次CUDA，最后CPU）
python run_xbot_orca.py --device auto
python run_xbot_keyboard.py --device auto
```

### 方法 2: 手动指定 MUSA

```bash
# 强制使用 MUSA GPU
python run_xbot_orca.py --device musa
python run_xbot_keyboard.py --device musa
```

### 方法 3: 使用 CUDA 或 CPU

```bash
# 使用 CUDA GPU
python run_xbot_orca.py --device cuda

# 使用 CPU
python run_xbot_orca.py --device cpu
```

## 安装 MUSA 支持

### 1. 安装 torch_musa

```bash
# 根据摩尔线程官方文档安装 torch_musa
# 通常需要从摩尔线程官方源安装
pip install torch_musa
```

### 2. 验证安装

```python
import torch
try:
    import torch_musa
    if torch.musa.is_available():
        print(f"MUSA GPU available: {torch.musa.get_device_name(0)}")
        print(f"MUSA device count: {torch.musa.device_count()}")
    else:
        print("MUSA GPU not available")
except ImportError:
    print("torch_musa not installed")
```

## 代码修改说明

### 主要修改

1. **`load_xbot_policy()` 函数** (`run_xbot_orca.py`)：
   - 添加 `device` 参数，支持 `"auto"`, `"musa"`, `"cuda"`, `"cpu"`
   - 使用 `get_torch_device()` 自动检测 GPU 类型
   - 支持 `torch_musa` 库

2. **`run_xbot_keyboard.py`**：
   - 同样支持 MUSA GPU 自动检测
   - 添加设备检测逻辑

3. **命令行参数**：
   - 添加 `--device` 参数，支持 `musa` 选项

## 设备检测逻辑

使用 `orca_gym.utils.device_utils` 模块：

```python
from orca_gym.utils.device_utils import get_torch_device, get_gpu_info, print_gpu_info

# 自动检测（优先级：MUSA > CUDA > CPU）
device = get_torch_device(try_to_use_gpu=True)
# 返回: torch.device("musa:0") 或 torch.device("cuda:0") 或 torch.device("cpu")

# 获取GPU信息
gpu_info = get_gpu_info()
# 返回: {"device_type": "musa", "device_name": "...", "available": True, ...}

# 打印GPU信息
print_gpu_info()
```

## 验证 MUSA 使用

运行时会显示：

```
GPU 类型: musa
GPU 可用: True
GPU 数量: 1
GPU 名称: <MUSA GPU名称>
[INFO] Auto-detected device: musa:0
Device: MUSA
[INFO] Using MUSA GPU: <GPU名称>
[INFO] Policy loaded on device: musa:0
```

## 故障排除

### 问题 1: torch_musa 未安装

**错误信息**：
```
[WARNING] torch_musa not installed. Falling back to CPU.
```

**解决方案**：
```bash
# 安装 torch_musa（根据摩尔线程官方文档）
pip install torch_musa
```

### 问题 2: MUSA GPU 不可用

**错误信息**：
```
[WARNING] MUSA GPU not available. Falling back to CPU.
```

**解决方案**：
1. 检查 MUSA 驱动是否安装
2. 检查 `torch.musa.is_available()` 返回值
3. 参考摩尔线程官方文档

### 问题 3: 自动回退到 CPU

如果 MUSA GPU 不可用，程序会自动回退到 CPU，并显示警告信息。

## 性能说明

| 设备 | 推理延迟 | 适用场景 |
|------|---------|---------|
| **MUSA GPU** | ~0.5-2ms + 传输开销 | 批量推理、训练 |
| **CUDA GPU** | ~0.5-2ms + 传输开销 | 批量推理、训练 |
| **CPU** | ~1-5ms | 实时控制（推荐） |

**注意**：MuJoCo 物理仿真本身不支持 GPU，只有策略推理可以使用 GPU。

---

**文档更新时间**: 2025-01-XX
**适配分支**: moer

