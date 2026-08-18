# OrcaGym

[![PyPI version](https://img.shields.io/pypi/v/orca-gym)](https://pypi.org/project/orca-gym/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

欢迎来到 OrcaGym！这是 OrcaGym 的核心库，提供与 OpenAI Gym/Gymnasium 接口兼容的机器人仿真环境。与松应科技的 OrcaStudio、OrcaLab 平台联合开发，OrcaGym 为多物理引擎和分布式仿真提供强大支持，同时保持与流行 RL 框架的编程接口兼容性。

> **配套项目**
> - **初学者示例**：推荐先体验 **[OrcaPlayground](https://github.com/openverse-orca/OrcaPlayground)**，包含上手示例与环境模板。
> - **完整数据流水线**：**[OrcaManipulation](https://github.com/openverse-orca/OrcaManipulation)** 提供数据采集 → 数据合成 → 训练 → 推理的端到端系统。

> **注意**: `orca-gym` PyPI 包仅包含核心功能模块。强化学习、模仿学习、输入设备、数据采集等**应用层示例**请前往上述配套项目仓库，并按其文档安装依赖；本仓库仍可通过可选依赖 `pip install orca-gym[rl]` 等安装算法侧常用库。

## 背景
机器人仿真作为具身智能训练的关键平台，需要物理准确性和可扩展的基础设施。传统解决方案往往在保真度和计算效率之间面临权衡，特别是在扩展到分布式系统时。OrcaGym 通过将实时物理仿真与云原生架构相结合来弥合这一差距，使研究人员能够在高保真环境中原型化算法并在大规模部署。

## 主要特性
- 🎮 **Gym/Gymnasium API 兼容性** - 提供与 OpenAI Gym/Gymnasium API 兼容的仿真环境，与现有 RL 算法无缝集成
- ⚡ **双物理后端** - MuJoCo（CPU，开源标准）与 Euler（GPU，Orca 团队自研），两条路径互不隶属
- 🌐 **分布式部署** - 支持跨异构计算节点的分布式训练场景
- 🔍 **可扩展渲染** - 可接入外部可视化工具（如 OrcaStudio/OrcaLab）进行光线追踪等高质量渲染
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

## 外部可视化工具配置（可选）

OrcaGym 的物理仿真不依赖外部工具，可直接在本地运行。如需可视化场景或远程渲染，可从[官方门户](http://orca3d.cn/)下载并安装 OrcaStudio 或 OrcaLab，并参照其各自文档完成配置。

### 使用 orcagym-loop 命令启动仿真循环

`orcagym-loop` 是一个用于测试的常用脚本，用于启动基本的仿真循环。安装 `orca-gym` 后，可以直接使用该命令。

```bash
# 安装 orca-gym 后，直接运行
orcagym-loop
```

该命令会启动一个基本的仿真循环，按 `Ctrl+C` 可以停止仿真。

## 教程与端到端项目

### OrcaPlayground — 初学者示例

**[OrcaPlayground](https://github.com/openverse-orca/OrcaPlayground)** 是 OrcaGym 官方示例仓库，集成 OrcaLab 一键启动，涵盖足式机器人 RL 训练、机械臂操作、轮式底盘、人形机器人、角色动画、流体仿真等场景，便于快速上手。

### OrcaManipulation — 端到端数据流水线

**[OrcaManipulation](https://github.com/openverse-orca/OrcaManipulation)** 提供数据采集 → 数据合成 → 训练 → 推理的端到端系统，覆盖 VR 遥操作采集、数据增强、场景随机化、HDF5 存储等完整工具链。


## 贡献

我们欢迎对 OrcaGym 项目的贡献。如果您有建议、错误报告或功能请求，请在我们的 GitHub 仓库上开一个 issue 或提交 pull request。

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
如有任何疑问或需要支持，请访问 [www.orca3d.cn](https://www.orca3d.cn) 与我们建立联系。

---

我们希望您发现 OrcaGym 是您机器人和强化学习研究的宝贵工具。祝仿真愉快！
