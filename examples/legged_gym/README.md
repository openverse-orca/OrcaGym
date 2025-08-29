

# Sim2Real 需要的配置：

## 导出ONNX模型

1. 安装依赖
- 安装onnx包
```bash
pip install onnx onnxruntime-gpu onnxscript
```
- 安装cudnn（如果还没有装过的话）
```bash
conda install -y -c conda-forge cudnn=9.*
```


2. 导出ONNX模型
```bash
python scripts/convert_sb3_to_onnx.py --model_path models/ppo_model.zip --output_path models/ppo_model.onnx
```

# 使用 Ray RLLib 框架分布式训练需要的配置

## 安装Ray RLlib
要安装Ray RLlib，请使用以下命令：

```bash
pip install ray[rllib]
```

## 安装与你的CUDA版本匹配的torch
如果你使用的是conda环境，并且你的CUDA版本是12.8，请使用以下命令安装torch：
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 安装与你的cuda版本匹配的cuda-toolkit
如果你使用的是conda环境，并且你的CUDA版本是12.8，请使用以下命令安装cuda-toolkit：

```bash
conda install -c conda-forge -c nvidia cuda-toolkit=12.8
```