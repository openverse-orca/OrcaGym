

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
pip install ray[rllib]==2.49.0
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

在head节点机器上运行：

```bash
bash ./scripts/run_ray_node.sh head
```

这将：
- 自动激活orca conda环境
- 从配置文件读取IP地址（当前为192.168.1.100）
- 启动Ray head节点
- 显示Ray集群地址

### 启动Worker节点

在worker节点机器上运行：

```bash
bash ./scripts/run_ray_node.sh worker
```

或者指定head节点IP：

```bash
bash ./scripts/run_ray_node.sh worker 192.168.xxx.xxx
```

###  管理集群

#### 查看集群状态

```bash
bash ./scripts/run_ray_node.sh status
```

#### 停止集群

```bash
bash ./scripts/run_ray_node.sh stop
```

#### 查看帮助

```bash
bash ./scripts/run_ray_node.sh help
```

### 配置文件

脚本会自动读取 `examples/legged_gym/configs/rllib_appo_config.yaml` 文件中的配置：

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