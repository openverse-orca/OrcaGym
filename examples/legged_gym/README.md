

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
# head和worker节点都需要
pip install ray[rllib]==2.49.0 

# 仅head节点需要
pip install ray[default]==2.49.0
```

## 配置集群其他节点

由于Ray要求集群节点的python版本必须与head节点一致。因此在完成head节点配置后，在head查询python具体版本号：

```bash
python --version
```

如果与worker上已有的orca环境的python版本号不一致，就需要使用这个版本号在其他节点上安装python：
（注意，python版本号精确到第三位，如3.12.11）

```bash
conda create -n orca_ray python=xxx
```

然后按照orca环境的安装方式从新安装一次，直到完成所有worker的配置

## 启动Ray集群

### 启动Head节点

首先安装nfs服务端，并启动nfs服务

```bash
sudo apt-get install nfs-kernel-server
sudo systemctl start nfs-kernel-server
```


在head节点机器上运行：

```bash
bash ./scripts/run_ray_node.sh head 192.168.xxx.xxx
```

这将：
- 从你的小网ip启动head节点。为了连接稳定性，推荐使用有线网口，尽量不要用无线网口
- 启动Ray head节点
- 显示Ray集群地址

### 启动Worker节点

首先安装nfs客户端，支持mount.nfs命令。

```bash
sudo apt-get install nfs-common
```

在worker节点机器上运行：

```bash
bash ./scripts/run_ray_node.sh worker 192.168.xxx.xxx
```

###  管理集群

#### 查看集群状态

```bash
ray status
```

#### 停止集群（head节点运行，则停止整个集群，worker节点运行则当前节点退出集群）

```bash
ray stop
```

### 配置文件

脚本会自动读取 `configs/rllib_appo_config.yaml` 文件中的配置：

```yaml
orcagym_addresses: ["192.168.1.100:50051"]    # 配置成你的头结点ip地址
```

**重要**：请根据你的实际网络环境修改这个IP地址。

### 网络配置

#### 端口说明

- **Ray服务端口**：6379
- **Dashboard端口**：8265（如果安装了完整版Ray）
- **OrcaGym端口**：50051

#### 防火墙设置

确保以下端口在head节点上开放：

```bash
# Ubuntu/Debian
sudo ufw allow 6379
sudo ufw allow 8265
sudo ufw allow 50051

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=6379/tcp
sudo firewall-cmd --permanent --add-port=8265/tcp
sudo firewall-cmd --permanent --add-port=50051/tcp
sudo firewall-cmd --reload
```

# 模型提取和查看

## 功能特性

- 支持从SB3 PPO模型提取PyTorch模型
- 支持从RLLib APPO checkpoint提取PyTorch模型
- 详细的模型结构分析
- 参数统计和可视化
- 模型推理测试
- 保存为独立的PyTorch模型

## 使用方法

### 1. 基本用法（自动检测模型类型）

```bash
# 激活conda环境
conda activate orca

# 自动检测并分析最新的模型
python scripts/extract_pytorch_model.py
```

### 2. 指定模型类型和路径

```bash
# 分析RLLib APPO模型
python scripts/extract_pytorch_model.py \
    --type rllib \
    --checkpoint path/to/checkpoint_xxxxxx \
    --analyze-only

# 分析SB3模型
python scripts/extract_pytorch_model.py \
    --type sb3 \
    --checkpoint path/to/model.zip \
    --analyze-only
```

### 3. 保存提取的模型

```bash
# 提取并保存RLLib模型
python scripts/extract_pytorch_model.py \
    --type rllib \
    --checkpoint path/to/checkpoint_000000 \
    --output my_rllib_model.pth

# 提取并保存SB3模型
python scripts/extract_pytorch_model.py \
    --type sb3 \
    --checkpoint path/to/model.zip \
    --output my_sb3_model.pth
```

## 命令行参数

- `--checkpoint`: 模型checkpoint路径
- `--type`: 模型类型 (`sb3` 或 `rllib`)
- `--output`: 输出PyTorch模型路径
- `--analyze-only`: 只分析模型结构，不保存模型

## 输出信息

脚本会输出以下信息：

1. **模型组件结构**：显示编码器、策略网络、价值网络的结构
2. **参数统计**：每个组件的参数数量和可训练参数数量
3. **参数详情**：权重和偏置的统计信息（最小值、最大值、均值、标准差）
4. **推理测试**：使用示例输入测试模型推理能力