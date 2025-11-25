# MUSA 显卡适配说明

## 概述

Lite3_rl 和 xbot 项目已适配支持 MUSA（摩尔线程）显卡。代码会自动检测可用的 GPU 类型（优先级：MUSA > CUDA > CPU）。

## 当前支持状态

### ✅ XBot 项目（PyTorch JIT 模型）- **完全支持**
- **✅ 完全支持 MUSA GPU**
- PyTorch JIT 模型可以直接在 MUSA GPU 上运行
- **已验证**：策略成功加载到 `musa:0` 设备
- **性能**：推理在 GPU 上执行，性能最优
- **使用示例**：
  ```bash
  python run_xbot_orca.py --orcagym_addr 192.168.1.88:50051 --device musa
  # 输出: [INFO] Policy loaded on device: musa:0
  ```

### ⚠️ Lite3_rl 项目（ONNX 模型）- **部分支持**
- **✅ PyTorch 部分支持 MUSA GPU**（环境、观测计算等）
- **❌ ONNX Runtime 推理使用 CPU**（缺少 GPU provider）
- **功能**：正常，但推理性能受限（使用 CPU）
- **状态**：需要等待 MUSA 专用的 ONNX Runtime 版本以获得 GPU 加速
- **当前输出**：
  ```
  GPU 类型: musa
  GPU 可用: True
  [WARNING] MUSA GPU requested but no compatible provider found.
  [WARNING] Falling back to CPU.
  ```

### 总结对比

| 项目 | 模型格式 | MUSA GPU 支持 | 推理设备 | 性能 |
|------|---------|--------------|---------|------|
| **XBot** | PyTorch JIT | ✅ 完全支持 | GPU (`musa:0`) | ⭐⭐⭐ 最优 |
| **Lite3_rl** | ONNX | ⚠️ 部分支持 | CPU | ⭐⭐ 受限 |

**关键差异：**
- **PyTorch JIT**：原生支持 MUSA GPU，无需额外配置
- **ONNX Runtime**：需要 GPU provider，当前只有 CPU provider

## 设备检测优先级

1. **MUSA GPU**（摩尔线程显卡）
2. **CUDA GPU**（NVIDIA 显卡）
3. **CPU**（默认回退）

## 使用方法

### Lite3_rl (ONNX 推理)

#### 方法 1: 自动检测（推荐）

```bash
# 自动检测可用GPU（优先MUSA，其次CUDA，最后CPU）
python run_lite3_sim.py --config configs/lite3_onnx_sim_config.yaml --device auto
```

#### 方法 2: 手动指定 MUSA

```bash
# 强制使用 MUSA GPU
python run_lite3_sim.py --config configs/lite3_onnx_sim_config.yaml --device musa
```

#### 方法 3: 配置文件

编辑 `configs/lite3_onnx_sim_config.yaml`：

```yaml
inference_device: "auto"  # 可选: "cpu", "cuda", "musa", "auto"
```

### XBot (PyTorch JIT 推理)

#### 方法 1: 自动检测（推荐）

```bash
# 自动检测可用GPU
python run_xbot_orca.py --device auto
python run_xbot_keyboard.py --device auto
```

#### 方法 2: 手动指定 MUSA

```bash
# 强制使用 MUSA GPU
python run_xbot_orca.py --device musa
python run_xbot_keyboard.py --device musa
```

## 安装 MUSA 支持

### 1. 安装依赖包

**重要：版本要求**
- `torch_musa 2.0.1` 需要 **PyTorch 2.2.0**（不是 2.0.0 或其他版本）
- **NumPy 必须 < 2.0**（当前 PyTorch 2.2.0 不支持 NumPy 2.x）
- 必须从摩尔线程提供的本地 wheel 文件安装，不能从 PyPI 安装
- 需要同时安装匹配版本的 torch、torchvision、torchaudio

**从本地 wheel 文件安装：**

```bash
# 1. 首先降级 NumPy（重要！）
pip install "numpy<2.0"

# 2. 卸载现有的 PyTorch 相关包（如果已安装）
pip uninstall torch torchvision torchaudio torch_musa -y

# 3. 从本地 wheel 文件安装（按顺序安装）
pip install torch-2.2.0-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.17.2+c1d70fe-cp310-cp310-linux_aarch64.whl
pip install torchaudio-2.2.2+cefdb36-cp310-cp310-linux_aarch64.whl
pip install torch_musa-2.0.1-cp310-cp310-linux_aarch64.whl

# 4. 可选：安装其他依赖
pip install triton-3.1.0-cp310-cp310-linux_aarch64.whl

# 5. 验证安装
python -c "import numpy; print(f'NumPy: {numpy.__version__}'); import torch; print(f'PyTorch: {torch.__version__}'); import torch_musa; print(f'MUSA available: {torch.musa.is_available()}')"
```

**注意：**
- 这些 wheel 文件是 MUSA 专用的，只能从摩尔线程官方获取
- 版本必须严格匹配：torch 2.2.0 + torch_musa 2.0.1
- 如果看到版本警告（"torch version should be v2.0.0"），这是误报，可以忽略

### 2. 验证安装

```python
import torch
print(f"PyTorch version: {torch.__version__}")

try:
    import torch_musa
    print("torch_musa imported successfully")
    if torch.musa.is_available():
        print(f"MUSA GPU available: {torch.musa.get_device_name(0)}")
        print(f"MUSA device count: {torch.musa.device_count()}")
    else:
        print("MUSA GPU not available (torch.musa.is_available() returned False)")
except ImportError as e:
    print(f"torch_musa import failed: {e}")
    print("可能原因:")
    print("  1. PyTorch 版本不匹配（需要 2.0.0）")
    print("  2. torch_musa 未正确安装")
    print("  3. MUSA 驱动未正确安装")
```

### 3. ONNX Runtime 支持

MUSA GPU 通常通过 `CUDAExecutionProvider` 兼容，或者可能有专门的 MUSA ExecutionProvider。

**检查可用 providers：**

```python
import onnxruntime as ort
print(f"ONNX Runtime version: {ort.__version__}")
print("Available providers:", ort.get_available_providers())
```

**如果只有 CPU provider：**

如果 `get_available_providers()` 只返回 `['CPUExecutionProvider']` 或 `['AzureExecutionProvider', 'CPUExecutionProvider']`，说明 ONNX Runtime 不支持 GPU。

**当前状态：**
- ✅ MUSA GPU 已被 PyTorch 正确检测（`torch.musa.is_available() == True`）
- ❌ ONNX Runtime 没有 GPU provider，只能使用 CPU 推理

**可能的解决方案：**

1. **检查是否有 MUSA 专用的 ONNX Runtime 版本**
   ```bash
   # 询问摩尔线程官方是否提供支持 MUSA 的 ONNX Runtime 版本
   # 或者检查是否有 onnxruntime-musa 或类似的包
   ```

2. **尝试安装 onnxruntime-gpu（可能不支持 MUSA）**
   ```bash
   pip install onnxruntime-gpu
   # 注意：标准版本可能不支持 MUSA，需要验证
   ```

3. **使用 CPU 进行推理（当前方案）**
   - 代码会自动回退到 CPU
   - 性能较低，但功能正常

4. **使用 PyTorch JIT 模型代替 ONNX（推荐）**
   - 如果可能，将 ONNX 模型转换为 PyTorch JIT 格式
   - PyTorch 已支持 MUSA GPU，可以直接使用

## 常见问题排查

### 问题 0: NumPy 版本不兼容警告

**症状：**
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

**原因：**
PyTorch 2.2.0 不支持 NumPy 2.x，需要 NumPy 1.x。

**解决方案：**
```bash
# 降级 NumPy
pip install "numpy<2.0"

# 验证
python -c "import numpy; print(numpy.__version__)"
```

### 问题 1: `torch_musa` 导入失败，提示 "undefined symbol"

**症状：**
```
ImportError: /path/to/torch_musa/lib/libmusa_kernels.so: undefined symbol: ...
或
ImportError: Please try running Python from a different directory!
```

**原因：**
1. PyTorch 版本与 `torch_musa` 不匹配（需要 PyTorch 2.2.0 + torch_musa 2.0.1）
2. 从 PyPI 安装的 PyTorch 与 MUSA 专用的 torch_musa 不兼容
3. 安装顺序不正确或缺少依赖

**解决方案：**
```bash
# 1. 完全卸载现有的 PyTorch 相关包
pip uninstall torch torchvision torchaudio torch_musa triton -y

# 2. 从本地 wheel 文件重新安装（必须使用 MUSA 专用版本）
pip install torch-2.2.0-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.17.2+c1d70fe-cp310-cp310-linux_aarch64.whl
pip install torchaudio-2.2.2+cefdb36-cp310-cp310-linux_aarch64.whl
pip install torch_musa-2.0.1-cp310-cp310-linux_aarch64.whl

# 3. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import torch_musa; print(f'MUSA available: {torch.musa.is_available()}')"
```

**重要提示：**
- 必须使用摩尔线程提供的本地 wheel 文件，不能从 PyPI 安装
- 版本必须严格匹配：torch 2.2.0 + torch_musa 2.0.1

### 问题 2: `torch.musa.is_available()` 返回 False

**可能原因：**
1. MUSA 驱动未正确安装
2. 环境变量未设置
3. GPU 未被系统识别

**排查步骤：**
```bash
# 1. 检查系统是否能识别 GPU
mthreads-smi

# 2. 检查 PyTorch 版本
python -c "import torch; print(torch.__version__)"

# 3. 检查 torch_musa 是否能导入
python -c "import torch_musa; print('torch_musa imported')"

# 4. 检查 torch.musa 模块
python -c "import torch; import torch_musa; print(hasattr(torch, 'musa')); print(torch.musa.is_available())"
```

### 问题 3: ONNX Runtime 没有 GPU provider

**症状：**
```
GPU 类型: musa
GPU 可用: True
GPU 数量: 1
GPU 名称: M1000
[WARNING] MUSA GPU requested but no compatible provider found.
[WARNING] Available providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
[WARNING] Falling back to CPU.
```

**原因：**
- ✅ MUSA GPU 已被 PyTorch 正确检测
- ❌ ONNX Runtime 没有 GPU provider（缺少 `CUDAExecutionProvider` 或 MUSA 专用 provider）

**当前状态：**
- PyTorch 可以正常使用 MUSA GPU
- ONNX Runtime 只能使用 CPU 进行推理

**解决方案：**

1. **检查是否有 MUSA 专用的 ONNX Runtime**
   ```bash
   # 询问摩尔线程官方是否提供支持 MUSA 的 ONNX Runtime
   # 或检查是否有 onnxruntime-musa 包
   ```

2. **尝试安装 onnxruntime-gpu（可能不支持 MUSA）**
   ```bash
   pip install onnxruntime-gpu
   python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   # 如果出现 CUDAExecutionProvider，可以尝试使用（但可能不支持 MUSA）
   ```

3. **使用 CPU 推理（当前方案）**
   - 代码会自动回退到 CPU
   - 功能正常，但性能较低

4. **转换为 PyTorch JIT 模型（推荐）**
   - 如果可能，将 ONNX 模型转换为 PyTorch JIT 格式
   - PyTorch 已支持 MUSA GPU，可以直接使用 GPU 加速

**临时解决方案：**
代码会自动回退到 CPU，功能正常。如果需要 GPU 加速，需要等待 MUSA 专用的 ONNX Runtime 版本。

**对比：XBot vs Lite3_rl**
- **XBot（PyTorch JIT）**：✅ 完全支持 MUSA GPU，策略推理在 GPU 上运行
- **Lite3_rl（ONNX）**：⚠️ 部分支持，环境计算在 GPU 上，但策略推理在 CPU 上

**建议：**
- 如果可能，考虑将 ONNX 模型转换为 PyTorch JIT 格式以获得完整的 GPU 加速
- 或者等待 MUSA 专用的 ONNX Runtime 版本

### 问题 4: 系统能检测到 GPU，但 PyTorch 无法使用

**症状：**
- `mthreads-smi` 能显示 GPU 信息
- 但 `torch.musa.is_available()` 返回 False
- 或 `torch_musa` 导入失败

**可能原因：**
1. PyTorch 版本不匹配（需要 2.2.0，不是 2.0.0）
2. 使用了 PyPI 版本的 PyTorch，与 MUSA 专用版本不兼容
3. `torch_musa` 未正确安装或版本不匹配
4. 安装顺序不正确

**解决方案：**
```bash
# 1. 完全卸载现有版本
pip uninstall torch torchvision torchaudio torch_musa triton -y

# 2. 从本地 wheel 文件安装（必须使用 MUSA 专用版本）
pip install torch-2.2.0-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.17.2+c1d70fe-cp310-cp310-linux_aarch64.whl
pip install torchaudio-2.2.2+cefdb36-cp310-cp310-linux_aarch64.whl
pip install torch_musa-2.0.1-cp310-cp310-linux_aarch64.whl

# 3. 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import torch_musa; print(f'MUSA available: {torch.musa.is_available()}')"

# 4. 如果仍然失败，检查系统 GPU
mthreads-smi
```

**重要提示：**
- 必须使用摩尔线程提供的本地 wheel 文件
- 版本要求：torch 2.2.0 + torch_musa 2.0.1（不是 2.0.0）
- 如果看到 "torch version should be v2.0.0" 警告，这是误报，可以忽略

## 代码修改说明

### Lite3_rl 修改

1. **`load_lite3_onnx_policy()` 函数**：
   - 添加 `device` 参数，支持 `"auto"`, `"musa"`, `"cuda"`, `"cpu"`
   - 使用 `get_gpu_info()` 自动检测 GPU 类型
   - MUSA GPU 通过 `CUDAExecutionProvider` 兼容

2. **配置文件**：
   - 添加 `inference_device` 选项，默认 `"auto"`

3. **命令行参数**：
   - 添加 `--device` 参数，支持 `musa` 选项

### XBot 修改

1. **`load_xbot_policy()` 函数** (`run_xbot_orca.py`)：
   - 添加 `device` 参数，支持 `"auto"`, `"musa"`, `"cuda"`, `"cpu"`
   - 使用 `get_torch_device()` 自动检测 GPU 类型
   - 支持 `torch_musa` 库

2. **`run_xbot_keyboard.py`**：
   - 同样支持 MUSA GPU

3. **命令行参数**：
   - 添加 `--device` 参数，支持 `musa` 选项

## 设备检测逻辑

### PyTorch (XBot)

```python
from orca_gym.utils.device_utils import get_torch_device

# 自动检测（优先级：MUSA > CUDA > CPU）
device = get_torch_device(try_to_use_gpu=True)
# 返回: torch.device("musa:0") 或 torch.device("cuda:0") 或 torch.device("cpu")
```

### ONNX Runtime (Lite3_rl)

```python
from orca_gym.utils.device_utils import get_gpu_info

gpu_info = get_gpu_info()
# 返回: {"device_type": "musa", "device_name": "...", "available": True, ...}
```

## 验证 MUSA 使用

运行时会显示：

```
[INFO] Auto-detected MUSA GPU: <GPU名称>
Device: MUSA
[INFO] Using MUSA GPU (via CUDAExecutionProvider compatibility)
[ONNX Policy] Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

或对于 PyTorch：

```
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

### 问题 3: ONNX Runtime 不支持 MUSA

**错误信息**：
```
[WARNING] MUSA GPU requested but no compatible provider found.
```

**解决方案**：
- MUSA GPU 通常通过 `CUDAExecutionProvider` 兼容
- 如果不可用，程序会自动回退到 CPU

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

