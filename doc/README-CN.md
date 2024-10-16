# OrcaGym 项目

欢迎来到 OrcaGym 项目！OrcaGym 是基于 OpenAI Gymnasium 框架的增强型仿真环境，旨在与现有的 OpenAI Gym 仿真环境和算法无缝集成。OrcaGym 由松影科技开发，OrcaStudio 提供了对多种物理引擎和光线追踪渲染的强力支持，实现了物理和视觉上的高精度仿真。本文件旨在介绍 OrcaGym 的背景、目的、使用方法和重要注意事项。
# 背景

在机器人仿真领域，一个多功能且精确的仿真环境对算法的开发和测试至关重要。OpenAI Gym 一直是该领域的重要基石，提供了标准化的强化学习（RL）任务接口。然而，随着对更高级功能的需求（如多物理引擎支持和高精度渲染），OrcaStudio 应运而生。OrcaGym 连接了 OpenAI Gym 和 OrcaStudio，使研究人员和开发者能够利用 OrcaStudio 的高级功能，同时保持对 OpenAI Gym 环境和算法的兼容性。
# 目的

OrcaGym 的主要目标是通过与 OrcaStudio 仿真平台的集成，增强 OpenAI Gym 的功能。该集成允许用户：

    利用多种物理引擎： OrcaStudio 支持 Mujoco、PhysX、ODE 等多种物理引擎，用户可以根据任务需求灵活选择最合适的物理引擎。
    实现高精度渲染： 通过光线追踪技术，OrcaStudio 提供了视觉精确的仿真环境，特别适用于需要精准视觉反馈的任务。
    支持分布式部署： OrcaGym 和 OrcaStudio 可以在同一节点或不同节点上运行，便于分布式部署和大规模 AI 集群训练。

# 特性

    兼容 OpenAI Gym： 与现有的 OpenAI Gym 环境和算法无缝集成。
    多物理引擎支持： 支持 Mujoco、PhysX、ODE 等物理引擎的选择。
    高精度渲染： 光线追踪支持，实现精准的视觉仿真。
    分布式部署： 支持在同一或不同节点上运行仿真，支持大规模 AI 训练。
    易于使用： 提供简单接口，便于从 OpenAI Gym 过渡到 OrcaGym。

# 安装

要安装 OrcaGym，请按照以下步骤操作：

    克隆代码库：

```bash

git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym
```
    安装依赖：

为便于快速安装，可以创建一个新的 Conda 环境：（如果您尚未安装 anaconda 或 miniconda，请访问 官网 进行安装）

```bash

conda create --name orca_gym_test python=3.11
conda activate orca_gym_test
```
然后在新创建的环境中安装依赖：

```bash

pip install -r requirements.txt
```
    安装 PyTorch：

使用 PyTorch 与 CUDA 结合可以有效加速强化学习的训练。请根据您的 GPU 设备安装对应的 CUDA 软件包。以下是一个示例：

```bash

pip install torch --extra-index-url https://download.pytorch.org/whl/cu12x
```
设置 OrcaStudio：

请按照 OrcaStudio 文档 中的说明，在您的系统上安装并配置 OrcaStudio。

请注意，OrcaStudio 是一款可以在 Windows 和 Linux 上运行的程序。因此，在使用 WSL 运行 OrcaGym 与 OrcaStudio（运行于 Windows）结合时，我们实际上是在进行跨虚拟机通信。这需要进行一些额外的 配置。
使用方法

使用 OrcaGym 非常简单。以下是一些入门示例：
    下载项目文件:
    在终端中运行以下脚本，下载最新的项目文件：
    cd orca_gym
    python down_projects.py
    移动下载的文件
    下载并解压文件后，将项目文件夹移动到你的 OrcaStudio 安装目录,将 orca-studio-projects/{your-orca-studio-version} 目录中的文件（关卡、资源）移动到您的 OrcaStudio 安装目录。假设安装目录为 $MyWorkSpace/OrcaStudio，则将文件移动到 $MyWorkSpace/OrcaStudio/Projects/OrcaProject 目录中。

    验证 OrcaGym 环境：
        启动 OrcaStudio： 启动 OrcaStudio 并加载相应的关卡，例如 Ant_Multiagent。点击“启动”按钮（Ctrl-G）进入游戏模式。
        按照指南操作： 请参阅 tutorial/GymEnvTest.ipynb 文档中的说明，以验证基本功能。
    OrcaGym 的 Mujoco 接口：
        启动 OrcaStudio： 启动 OrcaStudio 并加载 Humanoid_LQR 关卡。点击“启动”按钮（Ctrl-G）进入游戏模式。
        按照指南操作： 请参阅 tutorial/Humanoid-LQR.ipynb 以了解如何将 Mujoco 项目中包含的 LQR 示例移植到 OrcaGym 中。

    使用 Xbox 控制器控制 Franka Panda 机械臂：
        启动 OrcaStudio： 启动 OrcaStudio 并加载 Franka_Joystick 关卡。点击“启动”按钮（Ctrl-G）进入游戏模式。
        按照指南操作： 请参阅 tutorial/Xbox-Joystick-Control.ipynb 学习如何使用控制器控制 Franka Panda 机械臂，并实现操作记录和回放。

    强化学习训练示例：
        启动 OrcaStudio： 启动 OrcaStudio 并加载 FrankaPanda_RL 关卡。点击“启动”按钮（Ctrl-G）进入游戏模式。
        按照指南操作： 请参阅 tutorial/FrankaPanda-RL/FrankaPanda-RL.md 学习如何使用多智能体强化学习进行训练。

# 重要注意事项

    性能： 高精度渲染和复杂物理仿真可能需要较高的计算资源。请确保您的硬件能够有效运行 OrcaStudio。
    配置： 根据您的仿真需求正确配置 OrcaStudio。有关详细的配置选项，请参考 OrcaStudio 文档。
    兼容性： 虽然 OrcaGym 旨在与 OpenAI Gym 保持兼容，但某些高级功能可能需要对现有 Gym 环境进行额外配置或修改。

贡献

我们欢迎对 OrcaGym 项目的贡献。如果您有任何建议、错误报告或功能需求，请在我们的 GitHub 仓库中提交 Issue 或 Pull Request。