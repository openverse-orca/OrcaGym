# Lite3_rl_deploy 迁移到 OrcaGym-dev

本目录包含从Lite3_rl_deploy迁移到OrcaGym-dev的Demo和工具。

## 文件说明

### `run_lite3_onnx_demo.py`
Lite3 ONNX策略运行Demo，展示如何：
- 加载Lite3配置
- 加载ONNX策略模型
- 计算Lite3格式的45维观测
- 运行策略推理

## 快速开始

### 1. 安装依赖

```bash
pip install onnxruntime numpy
```

### 2. 运行Demo

```bash
cd /home/guojiatao/OrcaWorkStation/OrcaGym-dev/examples/Lite3_rl
python run_lite3_onnx_demo.py --onnx_model_path /path/to/policy.onnx --test_obs
```

### 3. 在仿真环境中运行

使用提供的仿真运行脚本：

```bash
cd /home/guojiatao/OrcaWorkStation/OrcaGym-dev/examples/Lite3_rl
python run_lite3_sim.py --config configs/lite3_onnx_sim_config.yaml --remote localhost:50051
```

或者使用默认配置：

```bash
python run_lite3_sim.py --onnx_model_path policy.onnx --remote localhost:50051
```

**控制说明：**
- `Z`: 进入默认状态（站立）
- `C`: 进入RL控制状态
- `W/S`: 前进/后退
- `A/D`: 左移/右移
- `Q/E`: 顺时针/逆时针旋转
- `LShift`: 加速模式（Turbo）
- `R`: 重置环境

### 4. 在代码中使用

参考 `run_lite3_sim.py`，在环境循环中：

```python
from envs.legged_gym.utils.onnx_policy import load_onnx_policy
from envs.legged_gym.utils.lite3_obs_helper import compute_lite3_obs, get_dof_pos_default_policy
from envs.legged_gym.robot_config.Lite3_config import Lite3Config

# 加载策略
policy = load_onnx_policy("path/to/policy.onnx")

# 在环境循环中
obs = compute_lite3_obs(...)  # 计算45维观测
actions = policy(obs)          # 运行策略
env.step(actions)              # 应用动作
```

## 迁移内容

### 1. 配置文件更新
- 文件: `envs/legged_gym/robot_config/Lite3_config.py`
- 添加了迁移参数: `omega_scale`, `dof_vel_scale`, `max_cmd_vel`, `dof_pos_default_policy`等

### 2. 工具文件
- `envs/legged_gym/utils/onnx_policy.py` - ONNX策略加载器（支持单样本和批量推理）
- `envs/legged_gym/utils/lite3_obs_helper.py` - Lite3观测计算辅助函数

### 3. 批量推理支持
- `ONNXPolicy` 类自动检测模型是否支持动态batch
- 如果模型固定batch_size=1，会自动逐个处理批量输入
- 支持单样本 `[45]` 和批量 `[N, 45]` 两种输入格式

## 关键参数

### 观测空间 (45维)
- `base_ang_vel * 0.25` (3维)
- `projected_gravity` (3维)
- `commands * max_cmd_vel` (3维)
- `dof_pos - default_pos` (12维)
- `dof_vel * 0.05` (12维)
- `last_actions` (12维)

### 动作空间 (12维)
- 策略输出 → `actions * action_scale + default_dof_pos`
- PD控制: `τ = kp*(q_d - q) + kd*(dq_d - dq)`

### PD控制器参数
- `kp = 30.0`
- `kd = 0.7` (OrcaGym默认) 或 `1.0` (原始实现)

## 参考文档

- 详细迁移分析: `Lite3_rl_deploy/MIGRATION_ANALYSIS.md`
- 代码示例: `Lite3_rl_deploy/MIGRATION_CODE_EXAMPLES.md`
- 快速参考: `Lite3_rl_deploy/MIGRATION_QUICK_REFERENCE.md`

