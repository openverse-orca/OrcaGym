



## 导出ONNX模型

1. 需要安装onnx包
```bash
pip install onnx onnxruntime
```

2. 导出ONNX模型
```bash
python scripts/convert_sb3_to_onnx.py --model_path models/ppo_model.zip --output_path models/ppo_model.onnx
```