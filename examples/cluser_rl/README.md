
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

## 运行本地示例
### 训练
1. 启动OrcaLab，加载Ant关卡
2. 运行关卡
3. 在OrcaGym/examples/cluser_rl目录下，运行以下命令：
```bash
python run_ant_local.py --run_mode training
```
默认情况下，会启动16个环境运行器，每个运行器启动64个环境。
在配备RTX3070 显卡以及 i7-14700H CPU的电脑上，训练50个迭代大约需要10分钟。
训练完成后将输出“Best checkpoint directory: xxxx”

### 测试
运行以下命令：
```bash
python run_ant_local.py --run_mode testing --checkpoint xxxx
```
可以看到Ant机器人向远方(x轴正方向)移动。

## 运行集群示例

```bash
python run_ant_local.py --run_mode training --iter 50
```