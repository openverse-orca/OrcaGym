# OrcaGym

[![PyPI version](https://img.shields.io/pypi/v/orca-gym)](https://pypi.org/project/orca-gym/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

欢迎来到 OrcaGym！这是 OrcaGym 的核心库，提供与 OpenAI Gym/Gymnasium 接口兼容的机器人仿真环境。与松应科技的 OrcaStudio、OrcaLab 平台联合开发，OrcaGym 为多物理引擎和分布式仿真提供强大支持，同时保持与流行 RL 框架的编程接口兼容性。

> **注意**: `orca-gym` PyPI 包仅包含核心功能模块。如需使用强化学习训练、模仿学习、输入设备等功能，请参考各 `examples` 目录下的 README，按需安装额外依赖。

## 背景
机器人仿真作为具身智能训练的关键平台，需要物理准确性和可扩展的基础设施。传统解决方案往往在保真度和计算效率之间面临权衡，特别是在扩展到分布式系统时。OrcaGym 通过将实时物理仿真与云原生架构相结合来弥合这一差距，使研究人员能够在高保真环境中原型化算法并在大规模部署。

## 目的
OrcaGym 旨在：

1. 提供与 OpenAI Gym/Gymnasium API 兼容的 GPU 加速仿真环境
2. 通过 OrcaStudio、OrcaLab 集成支持多种物理后端（Mujoco、PhysX、ODE）
3. 实现跨异构计算节点的分布式训练场景
4. 通过光线追踪为基于视觉的 RL 任务提供逼真的渲染

## 主要特性
- 🎮 **Gym/Gymnasium API 兼容性** - 与现有 RL 算法无缝集成
- ⚡ **多物理后端** - 同时进行 Mujoco/PhysX/ODE 仿真
- 🌐 **分布式部署** - 通过 gRPC 实现混合本地/远程操作
- 🔍 **光线追踪渲染** - 逼真的视觉观察
- 🤖 **多智能体支持** - 原生异构智能体管理

## 安装

### 从 PyPI 安装（推荐）

```bash
# 安装核心包
pip install orca-gym

# 或者安装带可选依赖的版本
pip install orca-gym[rl]          # 强化学习训练
pip install orca-gym[imitation]   # 模仿学习
pip install orca-gym[devices]     # 输入设备支持
pip install orca-gym[sensors]     # 相机和传感器
pip install orca-gym[all]         # 所有可选依赖
```

### 从源码安装（开发者）

```bash
# 克隆仓库
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym

# 初始化资源和子模块（如果需要运行示例）
git lfs install
git lfs pull
git submodule update --init --recursive

# 创建 Python 环境
conda create -n orca python=3.12
conda activate orca

# 安装核心包
pip install -e .

# 全量安装
pip install -e ".[all]"

# 或者安装开发依赖
pip install -e ".[dev]"

# 安装可选依赖
pip install -e ".[rl]"
pip install -e ".[imitation]"
pip install -e ".[devices]"
pip install -e ".[sensors]"
```

### 特定示例的额外依赖

如果你想运行 `examples` 目录中的示例代码，请参考各示例目录下的 README 文件，了解需要安装的额外依赖。例如：

- **强化学习训练** (`examples/legged_gym`, `examples/cluser_rl`): 需要 `pip install orca-gym[rl]` 和可能的 PyTorch/CUDA 支持
- **模仿学习** (`examples/imitation`, `examples/openpi`): 需要 `pip install orca-gym[imitation]` 和相关依赖
- **输入设备** (`examples/realman`): 需要 `pip install orca-gym[devices]`


## OrcaStudio、OrcaLab 配置

从[官方门户](http://orca3d.cn/)下载并安装 OrcaStudio、OrcaLab

### 使用 orcagym-loop 命令启动仿真循环

`orcagym-loop` 是一个用于测试的常用脚本，用于启动基本的仿真循环。安装 `orca-gym` 后，可以直接使用该命令。

**使用 OrcaStudio 的情况**：
1. 在 OrcaStudio 中，点击"运行"按钮（或按快捷键 `Ctrl+G`）启动仿真服务器
2. 在控制台执行 `orcagym-loop` 命令，启动仿真循环
3. 仿真将在本地 `localhost:50051` 地址上运行

**使用 OrcaLab 的情况**：
1. 在 OrcaLab 中点击"运行"按钮启动仿真服务器
2. 选择"无仿真程序"选项
3. 在控制台执行 `orcagym-loop` 命令，启动仿真循环
4. 仿真将在本地 `localhost:50051` 地址上运行

**命令示例**：
```bash
# 安装 orca-gym 后，直接运行
orcagym-loop
```

该命令会启动一个基本的仿真循环，按 `Ctrl+C` 可以停止仿真。

## 核心包说明

`orca-gym` 包含以下核心模块：

- **core**: 核心仿真接口，支持本地 (Mujoco) 和远程 (gRPC) 模式
- **environment**: Gymnasium 兼容的环境基类
- **protos**: gRPC 协议定义
- **scene**: 场景管理和运行时
- **utils**: 实用工具函数（旋转、控制器等）

**不包含**：
- 训练示例代码 (`examples/`)
- 特定环境实现 (`envs/`)
- 适配器 (`orca_gym/adapters/`)
- 输入设备 (`orca_gym/devices/`)
- 传感器 (`orca_gym/sensor/`)
- 工具脚本 (`orca_gym/tools/`, `orca_gym/scripts/`)

这些功能可以通过克隆仓库获取，并按需安装对应的可选依赖。

## 使用示例

### 四足机器人仿真 (Legged Gym)

OrcaGym 提供了完整的四足机器人仿真环境，支持强化学习训练和实时控制。

**快速开始**：
```bash
# 克隆项目并安装依赖
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym
pip install -e .

# 运行四足机器人仿真
cd examples/legged_gym
python run_legged_sim.py --config configs/lite3_sim_config.yaml
```

**主要功能**：
- **实时仿真**：支持键盘控制 (WASD 移动，Space 重置，M 切换模式)
- **强化学习训练**：集成 Stable-Baselines3 和 Ray RLLib
- **gRPC 服务**：支持远程推理和分布式训练
- **地形生成**：动态生成各种复杂地形

**训练示例**：
```bash
# 使用 Stable-Baselines3 训练
python run_legged_rl.py --config configs/lite3_sim_config.yaml --mode training

# 使用 Ray RLLib 训练
python run_legged_rl.py --config configs/lite3_sim_config.yaml --mode rllib_training
```

更多示例请参考 `examples/` 目录下的各个子项目。

## 性能与配置考虑

* **远程模式**: 需要配置并运行 OrcaStudio 或 OrcaLab。请参考[官方门户](http://orca3d.cn/)获取安装和配置指南。
* **本地模式**: 使用 Mujoco 进行本地仿真，适合快速原型开发和测试。
* **兼容性**: OrcaGym 完全兼容 Gymnasium API，可以与现有的 RL 框架无缝集成。

## 开发与维护

### 开发模式安装

```bash
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym
pip install -e ".[dev]"
```

### 发布到 PyPI

项目包含完整的发布自动化脚本。查看详细文档：

```bash
# 查看快速参考
cat scripts/release/QUICK_REFERENCE.md

# 查看完整文档
cat scripts/release/README.md
```

**快速发布流程**：

```bash
# 使用 Makefile（推荐）
make bump-version VERSION=25.10.1  # 更新版本号
make release-test                   # 发布到 TestPyPI
make release-prod                   # 发布到 PyPI

# 或使用脚本
./scripts/release/bump_version.sh 25.10.1
./scripts/release/release.sh test
./scripts/release/release.sh prod
```

详细信息请参阅：
- 📚 [发布脚本文档](scripts/release/README.md)
- ⚡ [快速参考](scripts/release/QUICK_REFERENCE.md)
- 📝 [PyPI 发布指南](PYPI_RELEASE.md)

## 贡献

我们欢迎对 OrcaGym 项目的贡献。如果您有建议、错误报告或功能请求，请在我们的 GitHub 仓库上开一个 issue 或提交 pull request。

### 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交变更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 引用
```bibtex
@software{OrcaGym2024,  
  author = {松应科技},  
  title = {OrcaGym: 云原生机器人仿真平台},  
  year = {2024},  
  publisher = {GitHub},  
  journal = {GitHub 仓库},  
  howpublished = {\url{https://github.com/openverse-orca/OrcaGym}}  
}  
```

## 许可证
基于 MIT 许可证分发。详情请参见 **LICENSE**。

## 联系方式
如有任何疑问或需要支持，请联系我们：huangwei@orca3d.cn

---

我们希望您发现 OrcaGym 是您机器人和强化学习研究的宝贵工具。祝仿真愉快！
