# OrcaGym 项目

[![PyPI version](https://img.shields.io/pypi/v/orca-gym)](https://pypi.org/project/orca-gym/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

欢迎来到 OrcaGym 项目！OrcaGym 是一个高性能仿真环境，设计为与 OpenAI Gym/Gymnasium 接口兼容，能够与现有的强化学习算法无缝集成。与松应科技的 OrcaStudio、OrcaLab 平台联合开发，OrcaGym 为多物理引擎和光线追踪渲染提供强大支持，同时保持与流行 RL 框架的编程接口兼容性。

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
```bash
# 克隆仓库
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym

# 初始化资源
git lfs install
git lfs pull

# 初始化子模块
git submodule update --init --recursive

# 创建 conda 环境
conda create -n orca python=3.12
conda activate orca
```

### 安装与你的CUDA版本匹配的torch，通过 nvidia-smi 命令查看你的CUDA版本
假设你的CUDA版本是12.8，请使用以下命令安装torch：
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 安装与你的cuda版本匹配的cuda-toolkit
假设你的CUDA版本是12.8，请使用以下命令安装cuda-toolkit：

```bash
conda install -c conda-forge -c nvidia cuda-toolkit=12.8
```

### 验证环境配置

安装完成后，建议运行测试脚本验证CUDA和PyTorch是否正确配置：

```bash
python orca_gym/scripts/test_cuda_torch.py
```

### 安装核心包

```bash
pip install -e .
```


## OrcaStudio、OrcaLab 配置

从[官方门户](http://orca3d.cn/)下载并安装 OrcaStudio、OrcaLab

## 性能考虑
* 性能：高保真渲染和复杂物理仿真可能在计算上很密集。确保您的硬件满足有效运行 OrcaStudio、OrcaLab 的要求。
* 配置：正确配置 OrcaStudio、OrcaLab 以匹配您的仿真需求。请参考 OrcaStudio、OrcaLab 文档了解详细的配置选项。
* 兼容性：虽然 OrcaGym 旨在与 OpenAI Gym 兼容，但一些高级功能可能需要额外配置或修改现有的 Gym 环境。

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
如有任何疑问或需要支持，请联系我们：huangwei@openverse.com.cn

---

我们希望您发现 OrcaGym 是您机器人和强化学习研究的宝贵工具。祝仿真愉快！
