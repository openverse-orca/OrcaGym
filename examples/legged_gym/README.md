



## 导出ONNX模型

1. 安装依赖
- 安装onnx包
```bash
pip install onnx onnxruntime-gpu
```
- 安装cudnn（如果还没有装过的话）
```bash
conda install -y -c conda-forge cudnn=9.*
```


2. 导出ONNX模型
```bash
python scripts/convert_sb3_to_onnx.py --model_path models/ppo_model.zip --output_path models/ppo_model.onnx
```