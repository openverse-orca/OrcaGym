# OrcaGym Examples

本目录包含 OrcaGym 的示例代码，演示如何使用核心库进行各种任务。

## 📦 与 PyPI 包的关系

⚠️ **重要提示**: `examples/` 和 `envs/` 目录**不包含**在 `orca-gym` PyPI 包中。

这些示例代码需要从 GitHub 源码仓库获取：

```bash
# 克隆完整仓库
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym

# 安装核心包
pip install orca-gym

# 或者以开发模式安装（推荐用于运行示例）
pip install -e .
```

## 🚀 运行示例的两种方式

### 方式 1：开发模式安装（推荐）

这种方式会将项目根目录添加到 Python 路径：

```bash
cd /path/to/OrcaGym
pip install -e .

# 现在可以直接运行示例
python examples/legged_gym/run_legged_sim.py
```

### 方式 2：设置 PYTHONPATH

如果不想安装，可以临时设置环境变量：

```bash
cd /path/to/OrcaGym
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行示例
python examples/legged_gym/run_legged_sim.py
```

或者在脚本开头添加：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 现在可以导入 envs 和 examples
from envs.legged_gym import LeggedSimEnv
```

## 📁 示例目录结构

```
examples/
├── README.md                    # 本文件
├── INSTALLATION_GUIDE.md        # 依赖安装指南
├── character/                   # 角色仿真
├── cluser_rl/                   # 集群 RL 训练
├── hand_detection/              # 手部检测
├── imitation/                   # 模仿学习
├── legged_gym/                  # 足式机器人
├── openpi/                      # OpenPI 集成
├── realman/                     # Realman 机器人
├── replicator/                  # 场景复制
├── vln/                         # 视觉导航
└── wheeled_chassis/             # 轮式底盘
```

## 🎯 按类别浏览

### 🦿 强化学习训练

需要额外安装：`pip install orca-gym[rl]` + PyTorch/CUDA

- **legged_gym/** - 四足机器人 RL 训练
  ```bash
  pip install orca-gym[rl]
  pip install torch  # 根据你的 CUDA 版本
  python examples/legged_gym/run_legged_rl.py --config configs/go2.yaml
  ```

- **cluser_rl/** - 分布式 RL 训练
  ```bash
  python examples/cluser_rl/run_ant_local.py
  ```

### 🤖 模仿学习

需要额外安装：`pip install orca-gym[imitation]`

- **imitation/** - 基础模仿学习
  ```bash
  pip install orca-gym[imitation]
  python examples/imitation/run_franka_single_arm.py
  ```

- **openpi/** - OpenPI 策略学习
  ```bash
  pip install orca-gym[imitation]
  python examples/openpi/run_dual_arm_sim.py
  ```

### 🎮 输入设备控制

需要额外安装：`pip install orca-gym[devices]`

- **realman/** - Realman 机器人控制
  ```bash
  pip install orca-gym[devices]
  python examples/realman/rm75bv_xbox_osc_ctrl.py
  ```

### 🎬 场景编辑

- **replicator/** - NVIDIA Replicator 集成
  ```bash
  python examples/replicator/run_actors.py
  ```

### 🗺️ 视觉导航

需要额外安装：`pip install orca-gym[sensors]`

- **vln/** - 视觉语言导航
  ```bash
  pip install orca-gym[sensors]
  # 参考 examples/vln/README.md
  ```

## 📚 每个示例的详细说明

每个示例子目录通常包含：

- `README.md` - 详细说明和使用方法
- `requirements.txt` - 额外依赖（如果有）
- Python 脚本 - 可执行示例
- `configs/` - 配置文件（如果需要）

请查看具体目录的 README 了解详情。

## 🔧 故障排查

### 问题：`ModuleNotFoundError: No module named 'envs'`

**原因**: 没有将项目根目录添加到 Python 路径。

**解决方案**:
```bash
# 方案 1：开发模式安装
cd /path/to/OrcaGym
pip install -e .

# 方案 2：设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/OrcaGym"
```

### 问题：`ModuleNotFoundError: No module named 'orca_gym'`

**原因**: 没有安装核心包。

**解决方案**:
```bash
pip install orca-gym
# 或者开发模式
pip install -e .
```

### 问题：缺少特定依赖

**解决方案**: 查看 [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) 并安装对应的可选依赖：

```bash
# 强化学习
pip install orca-gym[rl]

# 模仿学习
pip install orca-gym[imitation]

# 所有功能
pip install orca-gym[all]
```

## 💡 开发自己的环境

如果你想基于示例创建自己的环境：

1. **复制相关代码**
   ```bash
   cp -r envs/manipulation my_project/
   ```

2. **修改导入路径**
   ```python
   # 从
   from envs.manipulation import SingleArmEnv
   
   # 改为
   from my_project.manipulation import SingleArmEnv
   ```

3. **独立开发**
   ```bash
   cd my_project
   pip install -e .
   ```

## 🌟 贡献示例

欢迎贡献新的示例！请确保：

1. 添加 README.md 说明
2. 列出额外依赖
3. 提供配置文件示例
4. 代码注释清晰

## 📖 相关文档

- [核心库文档](../README.md)
- [依赖安装指南](INSTALLATION_GUIDE.md)
- [PyPI 包说明](../PACKAGE_CHANGES.md)
- [发布流程](../scripts/release/README.md)

## 🆘 获取帮助

- 查看具体示例的 README
- 提交 GitHub Issue
- 联系：huangwei@orca3d.cn

---

**注意**: 示例代码持续更新中，某些示例可能需要特定版本的依赖或配置。请参考各示例目录下的 README 获取最新信息。

