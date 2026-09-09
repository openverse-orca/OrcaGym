# 🛠️ 安装指南

## 环境要求

| 依赖 | 版本要求 |
|------|----------|
| Python | ≥ 3.10 (推荐 3.12) |
| pip | ≥ 21.0 |
| 操作系统 | Ubuntu 20.04+ / Windows 10+ / macOS 12+ |

## 从 PyPI 安装（推荐）

```bash
# 安装核心包
pip install orca-gym

# 或者安装带可选依赖的版本
pip install orca-gym[rl]          # 强化学习训练 (Stable-Baselines3 等)
pip install orca-gym[imitation]   # 模仿学习
pip install orca-gym[devices]     # 输入设备支持
pip install orca-gym[sensors]     # 相机和传感器
pip install orca-gym[all]         # 所有可选依赖
```

## 可选依赖说明

| 组名 | 包含内容 | 适用场景 |
|------|----------|----------|
| `[rl]` | stable-baselines3, sb3_contrib, tensorboard | RL 训练 |
| `[imitation]` | h5py, opencv-python, tqdm | 模仿学习 |
| `[robomimic]` | h5py, termcolor, opencv-python | robomimic 工具链 |
| `[devices]` | pygame | 手柄/键盘控制 |
| `[sensors]` | opencv-python, av, websockets, matplotlib, pillow | 相机视觉 |

## 配置外部可视化工具（可选）

OrcaGym 的物理仿真不依赖外部工具，可直接在本地运行。如需可视化场景或远程渲染，可下载并安装 [OrcaStudio](https://www.orca3d.cn/) 或 OrcaLab，具体功能与配置请参照其各自文档。

## 验证安装

```bash
# 检查导入
python -c "import orca_gym; print('OrcaGym 安装成功!')"

# 检查版本
pip show orca-gym
```

## 常见安装问题

### 问题：MuJoCo 导入失败

```bash
# 确保已安装 mujoco（版本与 orca-gym 钉死的版本一致）
pip install mujoco==3.7.0

# Linux 用户可能需要安装额外的系统依赖
# 注意：libglew 的包名随 Ubuntu 版本不同
#   - Ubuntu 22.04+：libglew2.2
#   - Ubuntu 20.04：libglew2.1
sudo apt-get install libglfw3 libglew2.2 libosmesa6
```

### 问题：gRPC 版本冲突

```bash
# 重新安装
pip install grpcio grpcio-tools --force-reinstall
```

### 问题：mesh/hfield 资源下载失败

这是正常现象——mesh 和纹理文件在首次仿真启动时按需下载，确保网络连接正常即可。
