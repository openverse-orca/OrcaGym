# OrcaGym 项目
欢迎来到 OrcaGym 项目！OrcaGym 是一个基于 OpenAI Gymnasium 框架的增强型仿真环境，旨在与现有的 OpenAI Gym 仿真环境和算法无缝集成。由松颖科技开发的 OrcaStudio 提供了对多种物理引擎和光线追踪渲染的强大支持，提供物理和视觉的精确性。本文件将介绍 OrcaGym 项目的背景、目的、使用方法和重要事项。

## 背景
在机器人仿真领域，拥有一个多功能且精确的环境对算法的开发和测试至关重要。OpenAI Gym 一直是这一领域的基石，提供了一个标准化的强化学习（RL）任务接口。然而，随着对更多高级功能的需求，如支持多个物理引擎和高保真渲染，OrcaStudio 的开发应运而生。OrcaGym 连接了 OpenAI Gym 和 OrcaStudio，使得研究人员和开发者可以利用 OrcaStudio 的先进功能，同时保持与 OpenAI Gym 环境和算法的兼容性。

## 目的
OrcaGym 的主要目标是通过将其与 OrcaStudio 仿真平台结合，增强 OpenAI Gym 的功能。此集成使用户能够：

1. **使用多个物理引擎**：OrcaStudio 支持 Mujoco、PhysX、ODE 等，为用户提供了选择最适合任务的物理引擎的灵活性。
2. **实现高保真渲染**：通过光线追踪支持，OrcaStudio 提供精确的视觉仿真，这对于需要准确视觉反馈的任务至关重要。
3. **启用分布式部署**：OrcaGym 和 OrcaStudio 可以在同一节点或不同节点上运行，支持分布式部署和大规模 AI 集群训练。

## 特性
* **与 OpenAI Gym 兼容**：无缝集成现有的 OpenAI Gym 环境和算法。
* **支持多个物理引擎**：可选择 Mujoco、PhysX、ODE 等。
* **高保真渲染**：支持光线追踪，提供精确的视觉仿真。
* **分布式部署**：支持在同一节点或不同节点上运行仿真，支持大规模 AI 训练。
* **易于使用**：提供简单的接口，轻松过渡至 OrcaGym 环境。

## 安装
按照以下步骤安装 OrcaGym：

1. **克隆代码库：**
如果您是 **安装版** 用户，请跳过以下步骤。安装版将自动为您配置环境，无需手动下载和配置依赖。
```bash
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym
git lfs install
git lfs pull
git submodule update
```
2. **安装依赖：**
为了方便快速安装，我们可以创建一个新的 Conda 环境：（如果您没有安装 Anaconda 或 Miniconda，请访问官方网页(https://www.anaconda.com/)进行安装）
```bash
conda create --name orca_gym_test python=3.11
conda activate orca_gym_test
```
然后在新创建的环境中安装依赖：
```bash
pip install -r requirements.txt
cd 3rd_party/robomimic
pip install -e .
```
3. **安装 PyTorch：**
结合使用 PyTorch 和 CUDA 可以有效加速强化学习训练。根据您的 GPU 设备安装相应的 CUDA 包。以下是一个安装示例：
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu12x
```
## 设置 OrcaStudio：
请参阅 OrcaStudio 文档 中的说明，按照系统要求安装并配置 OrcaStudio。

请注意，OrcaStudio 是一个运行在 Windows 和 Linux 系统上的程序。因此，当使用 WSL 在 Windows 上运行 OrcaGym 时，实际上是在进行跨虚拟机通信，这需要进行一些额外的 配置。

## 使用
使用 OrcaGym 非常简单。以下是一些入门示例：

* **下载项目文件** 
运行以下脚本以下载最新的项目文件：
```bash
cd orca_gym
python down_projects.py
```
移动下载的文件 下载并解压项目文件后，将文件夹移动到您的 OrcaStudio 安装目录。将 orca-studio-projects/{your-orca-studio-version} 目录中的文件（如级别、资产）复制到您的 OrcaStudio 安装目录中。假设您的安装目录为 $MyWorkSpace/OrcaStudio，则将文件复制到 $MyWorkSpace/OrcaStudio/Projects/OrcaProject 目录。

* **验证 OrcaGym 环境：**

* **启动 OrcaStudio**：启动 OrcaStudio 并加载相应的关卡，例如 Ant_Multiagent。点击 "Start" 按钮（Ctrl-G）进入游戏模式。
跟随指南操作：根据 tutorial/GymEnvTest.ipynb 中的指示操作，验证基本功能。

* **OrcaGym 的 Mujoco 接口**

* **启动 OrcaStudio**：启动 OrcaStudio 并加载 Humanoid_LQR 关卡。点击 "Start" 按钮（Ctrl-G）进入游戏模式。
* **跟随指南操作**：根据 tutorial/Humanoid-LQR.ipynb 中的指示，学习如何将 Mujoco 项目中的 LQR 示例移植到 OrcaGym。

* **使用 Xbox 控制器控制 Franka Panda 机器人臂**

* **启动 OrcaStudio**：启动 OrcaStudio 并加载 Franka_Joystick 级别。点击 "Start" 按钮（Ctrl-G）进入游戏模式。
* **跟随指南操作**：根据 tutorial/Xbox-Joystick-Control.ipynb 中的指示，学习如何使用 Xbox 控制器控制 Franka Panda 机器人臂，并实现操作记录和回放。

* **强化学习训练示例**

* **启动 OrcaStudio**：启动 OrcaStudio 并加载 FrankaPanda_RL 级别。点击 "Start" 按钮（Ctrl-G）进入游戏模式。
* **跟随指南操作**：根据 tutorial/FrankaPanda-RL/FrankaPanda-RL.md 中的指示，学习如何进行多智能体强化学习训练。

## 重要事项
* 性能：高保真渲染和复杂的物理仿真可能会占用较多计算资源，请确保您的硬件满足运行 OrcaStudio 的要求。
* 配置：请根据您的仿真需求正确配置 OrcaStudio。详细配置选项请参考 OrcaStudio 文档。
* 兼容性：虽然 OrcaGym 旨在与 OpenAI Gym 保持兼容，但某些高级功能可能需要额外的配置或修改现有的 Gym 环境。

## 贡献
我们欢迎对 OrcaGym 项目的贡献。如果您有任何建议、bug 报告或功能请求，请在 GitHub 仓库中提交 issue 或 pull request。

## 许可
OrcaGym 使用 MIT 许可证。详情请参见 LICENSE 文件。

## 联系
如有任何问题或需要支持，请通过 huangwei@openverse.com.cn 与我们联系。