# Ray集群管理脚本使用说明

## 概述

`run_ray_node.sh` 是一个用于管理Ray分布式计算集群的脚本，支持启动head节点和worker节点，适用于OrcaGym项目的分布式训练。

## 功能特性

- 自动从配置文件读取head节点IP地址
- 支持conda环境自动激活
- 彩色输出信息，便于调试
- 端口冲突检测和处理
- 完整的错误处理机制

## 使用方法

### 1. 启动Head节点

在head节点机器上运行：

```bash
cd examples/legged_gym/scripts
./run_ray_node.sh head
```

这将：
- 自动激活orca conda环境
- 从配置文件读取IP地址（当前为192.168.1.100）
- 启动Ray head节点
- 启动Ray Dashboard（端口8265）

### 2. 启动Worker节点

在worker节点机器上运行：

```bash
cd examples/legged_gym/scripts
./run_ray_node.sh worker
```

或者指定head节点IP：

```bash
./run_ray_node.sh worker 192.168.1.100
```

### 3. 查看集群状态

```bash
./run_ray_node.sh status
```

### 4. 停止Ray节点

```bash
./run_ray_node.sh stop
```

### 5. 查看帮助

```bash
./run_ray_node.sh help
```

## 配置文件

脚本会自动读取 `examples/legged_gym/configs/rllib_appo_config.yaml` 文件中的 `orcagym_addresses` 配置项来获取head节点IP地址。

当前配置：
```yaml
orcagym_addresses: ["192.168.1.100:50051"]    # 配置成你的头结点ip地址
```

## 端口配置

- Ray服务端口：6379
- Ray Dashboard端口：8265
- Dashboard访问地址：http://192.168.1.100:8265

## 环境要求

- Linux系统
- Conda环境（orca）
- Ray已安装在orca环境中

## 故障排除

### 1. Conda环境问题

如果遇到conda激活问题，请确保：
- conda已正确安装
- orca环境存在
- 脚本有执行权限

### 2. 端口冲突

如果端口被占用，脚本会尝试停止现有进程。如果仍有问题，可以手动停止：

```bash
pkill -f "ray start"
```

### 3. 网络连接

确保worker节点能够访问head节点的6379端口：

```bash
telnet 192.168.1.100 6379
```

## 示例工作流程

1. **在head节点上启动集群**：
   ```bash
   ./run_ray_node.sh head
   ```

2. **在worker节点上加入集群**：
   ```bash
   ./run_ray_node.sh worker
   ```

3. **验证集群状态**：
   ```bash
   ./run_ray_node.sh status
   ```

4. **访问Dashboard**：
   在浏览器中打开 http://192.168.1.100:8265

5. **运行训练**：
   ```bash
   python run_legged_rl.py --config configs/rllib_appo_config.yaml --train
   ```

## 注意事项

- 确保所有节点都在同一网络环境中
- 防火墙需要开放6379和8265端口
- 建议在启动训练前先验证集群状态
- 如果修改了配置文件中的IP地址，需要重启相应的节点 