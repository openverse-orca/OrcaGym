# Isaac Gym 到 OrcaGym 四足强化学习迁移指南

## 目录
1. [框架对比分析](#1-框架对比分析)
2. [核心架构差异](#2-核心架构差异)
3. [移植核心思路](#3-移植核心思路)
4. [详细移植步骤](#4-详细移植步骤)
5. [API 映射表](#5-api-映射表)
6. [常见问题与解决方案](#6-常见问题与解决方案)

---

## 1. 框架对比分析

### 1.1 代码设计对比

| 维度 | Isaac Gym (legged_gym) | OrcaGym |
|------|----------------------|---------|
| **编程范式** | 面向对象，基于类的配置 | 面向对象 + 字典配置 |
| **环境接口** | VecEnv (自定义向量化接口) | Gymnasium (标准化接口) |
| **数据结构** | PyTorch Tensor (GPU) | NumPy Array (CPU/GPU混合) |
| **并行模式** | 同步向量化 (所有环境同步步进) | 异步分布式 (支持跨节点) |
| **配置系统** | 嵌套类结构 (LeggedRobotCfg) | 字典嵌套结构 (Python dict) |

### 1.2 架构对比

#### Isaac Gym 架构
```
训练入口 (train.py)
    ↓
Task Registry (注册环境和配置)
    ↓
LeggedRobot Environment (VecEnv)
    ├── Isaac Gym 物理引擎 (GPU)
    ├── 观测计算 (Tensor 批量操作)
    ├── 奖励计算 (Tensor 批量操作)
    └── 重置逻辑 (并行批量重置)
    ↓
RSL-RL PPO Runner (自定义 RL 算法)
```

#### OrcaGym 架构
```
训练入口 (run_legged_rl.py)
    ↓
Gymnasium 环境注册
    ↓
LeggedGymEnv (OrcaGymAsyncEnv)
    ├── Mujoco / 其他物理引擎 (本地或远程)
    ├── LeggedRobot Agent (单个 agent 逻辑)
    │   ├── 观测计算 (NumPy 操作)
    │   ├── 奖励计算 (NumPy 操作)
    │   └── 控制计算
    └── 异步环境管理
    ↓
Stable-Baselines3 / RLlib (标准 RL 库)
```

### 1.3 核心依赖对比

| 依赖类型 | Isaac Gym | OrcaGym |
|---------|-----------|---------|
| **物理引擎** | Isaac Gym (NVIDIA) | Mujoco / 其他物理引擎 |
| **深度学习** | PyTorch (必需) | PyTorch / TensorFlow (可选) |
| **RL 算法库** | RSL-RL (内置) | Stable-Baselines3 / RLlib |
| **环境接口** | VecEnv (自定义) | Gymnasium |
| **通信层** | 无 (本地 GPU) | 本地或远程通信 |
| **并行计算** | CUDA (GPU 张量并行) | CPU 多进程 / GPU (可选) |

---

## 2. 核心架构差异

### 2.1 环境执行模式

#### Isaac Gym: 同步向量化
```python
# Isaac Gym - 所有环境同步执行
env, env_cfg = task_registry.make_env(name="anymal_c_flat")
# env.num_envs = 4096，所有环境同时存在于 GPU 内存

obs = env.reset()  # shape: [4096, obs_dim]
for _ in range(1000):
    actions = policy(obs)  # shape: [4096, action_dim]
    obs, rew, done, info = env.step(actions)
    # 自动批量重置已完成的环境
```

#### OrcaGym: 异步分布式
```python
# OrcaGym - 支持分布式异步执行
env = gym.make("LeggedGym-v0", 
               orcagym_addr="localhost:50051",
               agent_num=32)  # 单个仿真实例支持 32 个 agent

obs = env.reset()  # 返回单个 agent 的观测
for _ in range(1000):
    action = policy(obs)
    obs, rew, done, truncated, info = env.step(action)
    if done:
        obs = env.reset()
```

### 2.2 数据流对比

#### Isaac Gym 数据流
```
GPU Tensor Operations:
Robot States → Observations (Tensor) → Policy (GPU) → Actions (Tensor) 
                                                              ↓
                                                    Torques (Tensor) → Isaac Gym Physics
```

#### OrcaGym 数据流
```
NumPy Operations:
Robot States → Observations (NumPy) → Policy → Actions (NumPy)
                                                        ↓
                                              Mujoco Physics (本地或远程)
```

### 2.3 配置系统差异

#### Isaac Gym: 嵌套类配置
```python
class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 235
        
    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            orientation = -0.2
            
    class control:
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
```

#### OrcaGym: 字典配置
```python
LeggedRobotConfig = {
    "go2": {
        "reward_coeff": {
            "rough_terrain": {
                "follow_command_linvel": 10.0,
                "body_orientation": -1.0,
            }
        },
        "kps": [20.0] * 12,
        "kds": [0.5] * 12,
    }
}
```

---

## 3. 移植核心思路

### 3.1 总体策略

移植一个 Isaac Gym 程序到 OrcaGym 需要**三层转换**：

1. **物理引擎层转换**: Isaac Gym Physics → Mujoco / 其他物理引擎
2. **环境接口层转换**: VecEnv → Gymnasium AsyncEnv
3. **RL 算法层转换**: RSL-RL → Stable-Baselines3/RLlib

### 3.2 迁移路线图

```
原始 Isaac Gym 项目
    ↓
第一步: 理解项目结构
    ├── 识别环境类 (继承自 VecEnv)
    ├── 识别配置文件 (LeggedRobotCfg)
    ├── 识别奖励函数
    └── 识别训练入口
    ↓
第二步: 准备物理模型
    ├── 准备机器人 URDF 文件
    ├── 转换 URDF 为 MJCF 格式
    ├── 在 OrcaStudio 中加载 MJCF 模型
    ├── 配置场景和地形
    └── 生成高度图
    ↓
第三步: 重写环境类
    ├── 创建 LeggedRobot Agent 类
    ├── 创建 LeggedGymEnv 环境类
    ├── 转换配置为字典格式
    └── 实现 Gymnasium 接口
    ↓
第四步: 转换核心逻辑
    ├── 观测函数: Tensor → NumPy
    ├── 奖励函数: 批量 → 单个 agent
    ├── 重置逻辑: 批量 → 单个 agent
    └── 动作执行: Tensor → NumPy
    ↓
第五步: 适配 RL 算法
    ├── 使用 Stable-Baselines3/RLlib
    ├── 配置向量化环境 (VecEnv wrapper)
    └── 调整超参数
    ↓
第六步: 测试与调优
    ├── 单步测试
    ├── Episode 测试
    └── 训练测试
```

### 3.3 关键转换点

| Isaac Gym 组件 | 转换目标 | 难度 | 说明 |
|---------------|---------|------|------|
| `LeggedRobot(VecEnv)` | `LeggedGymEnv(OrcaGymAsyncEnv)` | ⭐⭐⭐ | 核心环境类重写 |
| `compute_observations()` | `LeggedRobot.get_obs()` | ⭐⭐ | Tensor → NumPy |
| `compute_reward()` | `LeggedRobot.compute_reward()` | ⭐⭐ | 批量 → 单个 |
| `reset_idx()` | `reset_agents()` | ⭐⭐ | 批量 → 单个 |
| `step()` | `step()` (Gymnasium) | ⭐⭐⭐ | 接口适配 |
| `_init_buffers()` | 移除 (无需预分配) | ⭐ | - |
| `gym.refresh_*_tensor()` | 本地查询方法 | ⭐⭐ | API 转换 |
| `LeggedRobotCfg` | 字典配置 | ⭐ | 格式转换 |
| `OnPolicyRunner` | PPO (SB3/RLlib) | ⭐⭐ | 算法替换 |

---

## 4. 详细移植步骤

### 步骤 1: 环境分析与准备

#### 1.1 分析原有 Isaac Gym 代码结构

```python
# 原始 Isaac Gym 项目结构
legged_gym/
├── envs/
│   ├── base/
│   │   ├── legged_robot.py       # 核心环境类
│   │   └── legged_robot_config.py # 配置基类
│   └── anymal_c/
│       └── anymal_c_config.py    # 具体机器人配置
├── scripts/
│   └── train.py                   # 训练入口
└── utils/
    ├── task_registry.py           # 环境注册
    └── terrain.py                 # 地形生成
```

**关键代码识别清单**:
- [ ] 环境类的 `__init__()` 方法
- [ ] `compute_observations()` 方法
- [ ] `compute_reward()` 方法及所有 `_reward_*()` 方法
- [ ] `reset_idx()` 方法
- [ ] `step()` 方法
- [ ] 配置类的所有参数
- [ ] 机器人 URDF/asset 文件路径

#### 1.2 准备 OrcaGym 项目结构

```bash
# 创建 OrcaGym 项目目录
mkdir -p your_project/
cd your_project/

# 创建标准结构
mkdir -p envs/legged_gym/{robot_config,scripts}
mkdir -p assets/{robots,terrains}
mkdir -p configs
```

### 步骤 2: 模型与场景准备

#### 2.1 准备机器人模型

**步骤1**: 复制 URDF 和 mesh 文件
```bash
# 从 Isaac Gym 项目复制 URDF 和 mesh 文件
cp -r legged_gym/resources/robots/anymal_c/ your_project/assets/robots/
```

**步骤2**: 转换 URDF 为 MJCF
```bash
# OrcaStudio 只支持加载 MJCF 格式
# 使用 Mujoco 官方提供的 URDF → MJCF 转换工具
# 详见: https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler
# 或参考 Mujoco 文档中的模型转换章节
```

**注意**: OrcaStudio 不支持直接加载 URDF，需要先转换为 MJCF 格式。

#### 2.2 在 OrcaStudio 中配置场景

1. 启动 OrcaStudio
2. 加载机器人 MJCF 模型（确保已从 URDF 转换）
3. 创建地形场景
4. 设置初始位置
5. 导出场景配置

#### 2.3 生成高度图

```python
# 使用 OrcaGym 工具生成高度图
from scripts.scene_util import generate_height_map_file

height_map_file = generate_height_map_file(
    orcagym_addresses=["localhost:50051"],
)
print(f"Height map saved to: {height_map_file}")
```

### 步骤 3: 创建配置文件

#### 3.1 转换机器人配置

**Isaac Gym 配置** (`anymal_c_config.py`):
```python
class AnymalCRoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.62]
        default_joint_angles = {
            'LF_HAA': 0.0,
            'LH_HAA': 0.0,
            # ...
        }
    
    class control(LeggedRobotCfg.control):
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}
        action_scale = 0.25
        
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            orientation = -1.0
```

**OrcaGym 配置** (`anymal_c_config.py`):
```python
AnymalCConfig = {
    "base_joint_name": "base",
    "leg_joint_names": [
        "LF_HAA", "LF_HFE", "LF_KFE",
        "RF_HAA", "RF_HFE", "RF_KFE",
        "LH_HAA", "LH_HFE", "LH_KFE",
        "RH_HAA", "RH_HFE", "RH_KFE",
    ],
    
    "neutral_joint_angles": {
        "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
        "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
        "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
        "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
    },
    
    "kps": [80.] * 12,  # Stiffness for all joints
    "kds": [2.] * 12,   # Damping for all joints
    "action_scale": [0.25] * 12,
    
    "base_neutral_height_offset": 0.62,
    "base_born_height_offset": 0.1,
    
    "foot_body_names": ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
    "base_contact_body_names": ["base"],
    "leg_contact_body_names": [
        "LF_THIGH", "LF_SHANK", "RF_THIGH", "RF_SHANK",
        "LH_THIGH", "LH_SHANK", "RH_THIGH", "RH_SHANK"
    ],
    
    "reward_coeff": {
        "rough_terrain": {
            "follow_command_linvel": 10.0,
            "follow_command_angvel": 5.0,
            "body_orientation": -1.0,
            "body_lin_vel": -2.0,
            "body_ang_vel": -0.05,
            "action_rate": -0.01,
            "torques": -0.00001,
            # ... 其他奖励系数
        }
    },
    
    # Domain randomization
    "randomize_friction": True,
    "friction_range": [0.5, 1.25],
    "randomize_base_mass": True,
    "added_mass_range": [-1., 3.],
    "push_robots": True,
    "push_interval_s": 15,
    "max_push_vel_xy": 1.0,
}
```

#### 3.2 创建环境配置

```python
# envs/legged_gym/legged_config.py
LeggedEnvConfig = {
    "TIME_STEP": 0.005,      # 仿真步长 200Hz
    "FRAME_SKIP": 1,          # PD控制频率 200Hz
    "ACTION_SKIP": 4,         # 策略频率 50Hz
    "EPISODE_TIME_LONG": 20,  # Episode 时长
}

LeggedObsConfig = {
    "scale": {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "qpos": 1.0,
        "qvel": 0.05,
    },
    "noise": {
        "noise_level": 1.0,
        "qpos": 0.01,
        "qvel": 1.5,
        "lin_vel": 0.1,
        "ang_vel": 0.2,
        "orientation": 0.05,
    }
}
```

### 步骤 4: 实现 Agent 类

创建 `envs/legged_gym/legged_robot.py`:

```python
import numpy as np
from orca_gym.environment.async_env import OrcaGymAsyncAgent
from gymnasium import spaces

class LeggedRobot(OrcaGymAsyncAgent):
    def __init__(self, 
                 env_id: str,
                 agent_name: str,
                 task: str,
                 max_episode_steps: int,
                 dt: float,
                 robot_config: dict,
                 legged_obs_config: dict,
                 **kwargs):
        
        super().__init__(env_id, agent_name, task, max_episode_steps, dt, **kwargs)
        
        self._robot_config = robot_config
        self._legged_obs_config = legged_obs_config
        
        # 关节配置
        self._base_joint_name = self.name_space(robot_config["base_joint_name"])
        self._leg_joint_names = self.name_space_list(robot_config["leg_joint_names"])
        self._neutral_joint_values = np.array([
            robot_config["neutral_joint_angles"][key] 
            for key in robot_config["leg_joint_names"]
        ])
        
        # PD 控制参数
        self._kps = np.array(robot_config["kps"])
        self._kds = np.array(robot_config["kds"])
        self._action_scale = np.array(robot_config["action_scale"])
        
        # 奖励函数配置
        self._reward_functions = self._setup_reward_functions()
        
        # 状态缓存
        self._leg_joint_qpos = np.zeros(len(self._leg_joint_names))
        self._leg_joint_qvel = np.zeros(len(self._leg_joint_names))
        self._body_lin_vel = np.zeros(3)
        self._body_ang_vel = np.zeros(3)
        self._body_orientation = np.zeros(3)
        
        # 动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self._leg_joint_names),),
            dtype=np.float32
        )
        
        # 观测空间 (需要根据实际情况调整)
        obs_dim = (
            3 +  # body linear velocity
            3 +  # body angular velocity  
            3 +  # body orientation
            4 +  # command (lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_z)
            len(self._leg_joint_names) +  # joint positions
            len(self._leg_joint_names) +  # joint velocities
            len(self._leg_joint_names)    # last actions
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def get_obs(self, sensor_data: dict, qpos_buffer: np.ndarray, 
                qvel_buffer: np.ndarray, qacc_buffer: np.ndarray,
                contact_dict: dict, site_pos_quat: dict, 
                height_map: np.ndarray) -> dict:
        """
        计算观测值
        
        从 Isaac Gym 的 compute_observations() 转换而来
        """
        # 更新关节状态
        self._leg_joint_qpos = qpos_buffer[
            self._qpos_index["leg_start"]: 
            self._qpos_index["leg_start"] + self._qpos_index["leg_length"]
        ]
        self._leg_joint_qvel = qvel_buffer[
            self._qvel_index["leg_start"]: 
            self._qvel_index["leg_start"] + self._qvel_index["leg_length"]
        ]
        
        # 计算身体局部坐标系下的速度和方向
        self._body_lin_vel, self._body_ang_vel, self._body_orientation = \
            self._get_body_local(qpos_buffer, qvel_buffer)
        
        # 构建观测向量
        obs = np.concatenate([
            self._body_lin_vel,
            self._body_ang_vel,
            self._body_orientation,
            self._command_values,  # 目标速度指令
            (self._leg_joint_qpos - self._neutral_joint_values),
            self._leg_joint_qvel,
            self._action,  # 上一步动作
        ]).astype(np.float32)
        
        # 缩放和噪声
        obs *= self._obs_scale_vec
        noise = (self._np_random.random(len(obs)) * 2 - 1) * self._noise_scale_vec
        obs += noise
        
        return {
            "observation": obs,
            "achieved_goal": self._achieved_goal,
            "desired_goal": self._desired_goal,
        }
    
    def compute_reward(self, achieved_goal, desired_goal) -> float:
        """
        计算奖励值
        
        从 Isaac Gym 的 compute_reward() 转换而来
        """
        total_reward = 0.0
        
        for reward_fn in self._reward_functions:
            if reward_fn["coeff"] == 0:
                continue
            reward = reward_fn["function"](reward_fn["coeff"])
            total_reward += reward
        
        return total_reward
    
    def _compute_reward_follow_command_linvel(self, coeff: float) -> float:
        """
        跟踪线速度指令的奖励
        
        从 Isaac Gym 的 _reward_tracking_lin_vel() 转换而来
        """
        tracking_sigma = 0.25
        lin_vel_error = np.sum(
            np.square(self._command["lin_vel"][:2] - self._body_lin_vel[:2])
        )
        reward = np.exp(-lin_vel_error / tracking_sigma) * coeff * self.dt
        return reward
    
    def _compute_reward_body_orientation(self, coeff: float) -> float:
        """
        惩罚身体倾斜
        
        从 Isaac Gym 的 _reward_orientation() 转换而来
        """
        reward = -np.sum(np.square(self._body_orientation[:2])) * coeff * self.dt
        return reward
    
    def step(self, action: np.ndarray, **kwargs):
        """
        执行一步动作
        """
        self._current_episode_step += 1
        self._action = action
        
        # 动作映射到控制指令
        scaled_action = action * self._action_scale
        position_ctrl = np.clip(
            self._neutral_joint_values + scaled_action,
            self._ctrl_range[:, 0],
            self._ctrl_range[:, 1]
        )
        
        self._position_ctrl = position_ctrl
        return self._ctrl, {}
    
    def reset(self, np_random, height_map: np.ndarray):
        """
        重置 agent 状态
        
        从 Isaac Gym 的 reset_idx() 转换而来
        """
        # 重置控制为中性位置
        self._ctrl = self._neutral_joint_values.copy()
        
        # 随机化初始位置
        pos_noise = np_random.uniform(-0.5, 0.5, 3)
        pos_noise[2] = 0.0
        
        # 随机化初始朝向
        z_rotation_angle = np_random.uniform(-np.pi, np.pi)
        z_rotation_quat = euler2quat([0, 0, z_rotation_angle])
        
        # 设置关节位置
        joint_qpos = {
            joint_name: np.array([self._neutral_joint_values[i]])
            for i, joint_name in enumerate(self._leg_joint_names)
        }
        
        # 设置基座位置
        base_qpos = self._init_base_joint_qpos.copy()
        base_qpos[:3] += pos_noise
        base_qpos[3:7] = z_rotation_quat
        joint_qpos[self._base_joint_name] = base_qpos
        
        # 重置速度为零
        joint_qvel = {
            joint_name: np.zeros(self._qvel_index[joint_name]["len"])
            for joint_name in self._joint_names
        }
        
        # 生成新的指令
        self._command = self._generate_command(z_rotation_angle)
        self._command_values = np.concatenate([
            self._command["lin_vel"], 
            [self._command["ang_vel"]]
        ])
        
        return joint_qpos, joint_qvel
    
    def is_terminated(self, achieved_goal, desired_goal) -> bool:
        """
        判断是否提前终止
        
        从 Isaac Gym 的 check_termination() 转换而来
        """
        # 检查身体是否与地面接触
        if np.any(self._body_contact):
            return True
        return False
    
    def _setup_reward_functions(self):
        """设置奖励函数列表"""
        reward_coeff = self._robot_config["reward_coeff"][self._task]
        return [
            {"function": self._compute_reward_follow_command_linvel, 
             "coeff": reward_coeff.get("follow_command_linvel", 0)},
            {"function": self._compute_reward_body_orientation, 
             "coeff": reward_coeff.get("body_orientation", 0)},
            # ... 添加其他奖励函数
        ]
```

### 步骤 5: 实现环境类

创建 `envs/legged_gym/legged_gym_env.py`:

```python
import numpy as np
from orca_gym.environment.async_env import OrcaGymAsyncEnv
from .legged_robot import LeggedRobot

class LeggedGymEnv(OrcaGymAsyncEnv):
    metadata = {
        'render_modes': ['human', 'none'],
        'version': '0.0.1',
        'render_fps': 30
    }

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        max_episode_steps: int,
        render_mode: str,
        is_subenv: bool,
        height_map_file: str,
        run_mode: str,
        env_id: str,
        task: str,
        robot_config: dict,
        legged_obs_config: dict,
        **kwargs,
    ):
        self._run_mode = run_mode
        self._height_map = np.zeros((2000, 2000))
        
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            agent_entry="envs.legged_gym.legged_robot:LeggedRobot",
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            is_subenv=is_subenv,
            env_id=env_id,
            task=task,
            robot_config=robot_config,
            legged_obs_config=legged_obs_config,
            **kwargs,
        )
        
        self._init_height_map(height_map_file)
        self._randomize_agent_foot_friction()
        self._add_randomized_weight()
    
    @property
    def agents(self) -> list[LeggedRobot]:
        return self._agents
    
    def step_agents(self, action: np.ndarray) -> None:
        """
        执行所有 agents 的动作
        
        从 Isaac Gym 的 step() 转换而来
        """
        action = action.reshape(len(self.agents), -1)
        
        joint_qvels = {}
        for i, agent in enumerate(self.agents):
            act = action[i]
            
            # 更新 agent 的指令
            agent.update_command(self.data.qpos)
            
            # 执行 agent 的 step
            agent_ctrl, _ = agent.step(act)
            
            # 可选: 随机推动机器人
            joint_qvel_dict = agent.push_robot(self.data.qvel)
            joint_qvels.update(joint_qvel_dict)
        
        # 应用速度扰动
        self.set_joint_qvel(joint_qvels)
    
    def get_obs(self) -> tuple:
        """
        获取观测值
        
        从 Isaac Gym 的 post_physics_step() 转换而来
        """
        # 查询传感器数据
        sensor_data = self._query_sensor_data()
        
        # 查询接触信息
        contact_dict = self._generate_contact_dict()
        
        # 查询 site 位置
        site_pos_quat = self._query_site_pos_and_quat()
        
        # 为每个 agent 计算观测
        env_obs_list = []
        achieved_goals = []
        desired_goals = []
        
        for agent in self.agents:
            obs = agent.get_obs(
                sensor_data,
                self.data.qpos,
                self.data.qvel,
                self.data.qacc,
                contact_dict,
                site_pos_quat,
                self._height_map
            )
            env_obs_list.append(obs["observation"])
            achieved_goals.append(obs["achieved_goal"])
            desired_goals.append(obs["desired_goal"])
        
        env_obs = {
            "observation": np.array(env_obs_list),
            "achieved_goal": np.array(achieved_goals),
            "desired_goal": np.array(desired_goals),
        }
        
        return env_obs, agent_obs, achieved_goals, desired_goals
    
    def reset_agents(self, agents: list[LeggedRobot]) -> None:
        """
        重置指定的 agents
        
        从 Isaac Gym 的 reset_idx() 转换而来
        """
        if len(agents) == 0:
            return
        
        joint_qpos = {}
        joint_qvel = {}
        
        for agent in agents:
            agent_qpos, agent_qvel = agent.reset(
                self.np_random,
                height_map=self._height_map
            )
            joint_qpos.update(agent_qpos)
            joint_qvel.update(agent_qvel)
        
        # 应用新的位置和速度
        self.set_joint_qpos(joint_qpos)
        self.set_joint_qvel(joint_qvel)
        self.mj_forward()
        self.update_data()
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        计算奖励
        
        从 Isaac Gym 的 compute_reward() 转换而来
        """
        if achieved_goal.ndim == 1:
            return self.agents[0].compute_reward(achieved_goal, desired_goal)
        elif achieved_goal.ndim == 2:
            rewards = np.zeros(len(achieved_goal))
            for i in range(len(achieved_goal)):
                rewards[i] = self.agents[i].compute_reward(
                    achieved_goal[i],
                    desired_goal[i]
                )
            return rewards
```

### 步骤 6: 创建训练脚本

创建 `scripts/train.py`:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import yaml

def make_env(env_id, rank, config):
    """创建单个环境实例"""
    def _init():
        env = gym.make(
            env_id,
            orcagym_addr=config['orcagym_addr'],
            agent_names=[f"anymal_{rank:03d}"],
            task=config['task'],
            frame_skip=config['frame_skip'],
            time_step=config['time_step'],
            max_episode_steps=config['max_episode_steps'],
            render_mode=config['render_mode'],
            is_subenv=(rank > 0),  # 第一个环境不是 subenv
            height_map_file=config['height_map_file'],
            run_mode='training',
            env_id=f"env_{rank:03d}",
            robot_config=config['robot_config'],
            legged_obs_config=config['legged_obs_config'],
        )
        return env
    return _init

def train(config):
    # 创建向量化环境
    num_envs = config['num_envs']
    env = SubprocVecEnv([
        make_env("LeggedGym-v0", i, config) 
        for i in range(num_envs)
    ])
    
    # 创建 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        tensorboard_log=config['log_dir'],
        verbose=1,
    )
    
    # 创建检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=config['save_freq'],
        save_path=config['checkpoint_dir'],
        name_prefix='anymal_c',
    )
    
    # 开始训练
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=checkpoint_callback,
    )
    
    # 保存最终模型
    model.save(f"{config['checkpoint_dir']}/final_model")
    
    env.close()

if __name__ == '__main__':
    # 加载配置
    with open('configs/anymal_c_rough.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 开始训练
    train(config)
```

创建配置文件 `configs/anymal_c_rough.yaml`:

```yaml
# OrcaGym 地址
orcagym_addr: "localhost:50051"

# 任务配置
task: "rough_terrain"
num_envs: 32
agent_num: 32

# 仿真参数
time_step: 0.005      # 仿真频率 200Hz，建议 ≤0.005 以保证接触检测准确
frame_skip: 1
action_skip: 4
max_episode_steps: 1000

# 渲染
render_mode: "none"

# 高度图
height_map_file: "/path/to/height_map.npy"

# PPO 超参数
learning_rate: 3.0e-4
n_steps: 24
batch_size: 96
n_epochs: 5
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.0
vf_coef: 0.5
max_grad_norm: 1.0

# 训练配置
total_timesteps: 10000000
save_freq: 100000
log_dir: "./logs"
checkpoint_dir: "./checkpoints"

# 机器人配置 (引用 robot_config/anymal_c_config.py)
robot_config: !include robot_config/anymal_c_config.py
legged_obs_config: !include legged_config.py:LeggedObsConfig
```

### 步骤 7: 注册环境

创建 `envs/__init__.py`:

```python
from gymnasium.envs.registration import register

register(
    id='LeggedGym-v0',
    entry_point='envs.legged_gym.legged_gym_env:LeggedGymEnv',
)
```

### 步骤 8: 测试与调试

#### 8.1 单步测试

```python
# test_env.py
import gymnasium as gym
import numpy as np

# 创建环境
env = gym.make(
    'LeggedGym-v0',
    orcagym_addr="localhost:50051",
    agent_names=["anymal_000"],
    task="rough_terrain",
    # ... 其他参数
)

# 测试重置
obs, info = env.reset()
print(f"Observation shape: {obs['observation'].shape}")
print(f"Observation: {obs['observation']}")

# 测试单步
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")

env.close()
```

#### 8.2 Episode 测试

```python
# test_episode.py
import gymnasium as gym

env = gym.make('LeggedGym-v0', ...)

for episode in range(5):
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(1000):
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: Total reward = {total_reward}, Steps = {step+1}")

env.close()
```

---

## 5. API 映射表

### 5.1 环境类方法映射

| Isaac Gym | OrcaGym | 说明 |
|-----------|---------|------|
| `__init__(cfg, sim_params, ...)` | `__init__(frame_skip, orcagym_addr, ...)` | 初始化参数不同 |
| `step(actions)` | `step(action)` | 批量 → 单个 |
| `reset()` | `reset()` | 返回值格式不同 |
| `compute_observations()` | `get_obs()` | Tensor → NumPy |
| `compute_reward()` | `compute_reward()` | 批量 → 单个 |
| `reset_idx(env_ids)` | `reset_agents(agents)` | 批量 → 列表 |
| `check_termination()` | `is_terminated()` | 批量 → 单个 |
| `create_sim()` | 不需要 | 物理引擎由 gRPC 管理 |
| `_init_buffers()` | 不需要 | 无需预分配缓冲区 |

### 5.2 物理引擎 API 映射

| Isaac Gym | OrcaGym | 说明 |
|-----------|---------|------|
| `self.gym.refresh_dof_state_tensor()` | `self.update_data()` | 更新状态 |
| `self.gym.refresh_actor_root_state_tensor()` | `self.update_data()` | 更新状态 |
| `self.gym.refresh_net_contact_force_tensor()` | `self.query_contact_simple()` | 查询接触 |
| `self.gym.set_dof_actuation_force_tensor()` | `self.ctrl = ...` | 设置控制 |
| `self.gym.set_dof_state_tensor_indexed()` | `self.set_joint_qpos()` | 设置关节状态 |
| `self.gym.set_actor_root_state_tensor_indexed()` | `self.set_joint_qpos()` | 设置位置 |
| `self.gym.simulate()` | `self.do_simulation()` | 执行仿真 |

### 5.3 数据结构映射

| Isaac Gym | OrcaGym | 说明 |
|-----------|---------|------|
| `torch.Tensor` (GPU) | `numpy.ndarray` (CPU) | 数据类型 |
| `self.dof_pos` | `qpos_buffer[...]` | 关节位置 |
| `self.dof_vel` | `qvel_buffer[...]` | 关节速度 |
| `self.root_states` | `qpos_buffer[base_index]` | 基座状态 |
| `self.contact_forces` | `query_contact_simple()` | 接触力 |
| `self.obs_buf` | `obs["observation"]` | 观测缓冲区 |
| `self.rew_buf` | `reward` (scalar/array) | 奖励缓冲区 |
| `self.reset_buf` | `terminated/truncated` | 重置标志 |

### 5.4 配置映射

| Isaac Gym Config | OrcaGym Config | 说明 |
|-----------------|---------------|------|
| `cfg.env.num_envs` | `num_envs` (训练脚本) | 环境数量 |
| `cfg.control.decimation` | `action_skip` | 控制频率 |
| `cfg.sim.dt` | `time_step` | 仿真步长 |
| `cfg.control.stiffness` | `kps` | PD 刚度 |
| `cfg.control.damping` | `kds` | PD 阻尼 |
| `cfg.rewards.scales.*` | `reward_coeff.*` | 奖励系数 |
| `cfg.normalization.obs_scales` | `LeggedObsConfig["scale"]` | 观测缩放 |
| `cfg.noise.noise_scales` | `LeggedObsConfig["noise"]` | 噪声缩放 |

---

## 6. 常见问题与解决方案

### 6.1 性能问题

**问题**: OrcaGym 训练速度比 Isaac Gym 慢

**原因**:
- Isaac Gym 使用 GPU 并行计算所有环境，批量 Tensor 操作效率极高
- OrcaGym 使用 CPU 物理引擎（Mujoco），单环境性能相对较低
- OrcaGym 采用异步架构，有一定的调度开销

**解决方案**:
1. **增加 `frame_skip` 和 `action_skip`**: 减少控制频率，提高每秒样本数
2. **使用多个物理引擎实例**: 通过 Ray 分布式运行，提高并行度
3. **优化环境代码**: 减少不必要的计算和数据传输
4. **使用向量化环境**: 使用 SubprocVecEnv 提高并行效率
5. **分布式部署**: 如需超大规模训练，可部署到多机

```python
# 示例: 使用 RLlib 多实例并行
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("LeggedGym-v0", env_config={...})
    .env_runners(
        num_env_runners=16,        # 16 个并行运行器
        num_envs_per_env_runner=4, # 每个运行器 4 个环境
    )
)
```

### 6.2 数值精度问题

**问题**: 奖励或观测值与 Isaac Gym 不一致

**原因**:
- 浮点精度差异 (GPU vs CPU)
- 物理引擎实现差异 (Isaac Gym vs Mujoco)

**解决方案**:
1. **调整奖励系数**: 根据实际训练效果调整
2. **验证物理模型**: 确保 URDF 参数一致
3. **对齐观测缩放**: 检查 `obs_scale_vec` 和 `noise_scale_vec`

```python
# 打印调试信息
print(f"Body lin vel: {self._body_lin_vel}")
print(f"Leg joint qpos: {self._leg_joint_qpos}")
print(f"Reward: {total_reward}")
```

### 6.3 接触检测问题

**问题**: 接触检测不准确或缺失

**原因**:
- Mujoco 和 Isaac Gym 的接触检测算法不同
- **仿真时间步长 `time_step` 过大**：如果 `time_step > 0.01`（低于100Hz）可能导致穿模等问题

**解决方案**:
1. **使用合适的仿真时间步长**: 建议 `time_step <= 0.005`（即 ≥200Hz 仿真频率）
2. **调整接触阈值**: `foot_touch_force_threshold`
3. **使用触觉传感器**: 配置 Mujoco touch sensor

```python
# 在 MJCF 中添加触觉传感器
<sensor>
    <touch name="LF_foot_touch" site="LF_foot_site"/>
    <touch name="RF_foot_touch" site="RF_foot_site"/>
    ...
</sensor>
```

### 6.4 地形高度图问题

**问题**: 机器人陷入地面或悬空

**原因**:
- 高度图分辨率不够
- 高度图坐标系转换错误

**解决方案**:
1. **提高高度图分辨率**: 增加采样点
2. **验证坐标转换**: 检查 `_get_body_height()` 逻辑
3. **可视化高度图**: 确保与实际地形一致

```python
# 可视化高度图
import matplotlib.pyplot as plt

plt.imshow(height_map, cmap='terrain')
plt.colorbar()
plt.title('Height Map')
plt.show()
```

### 6.5 Domain Randomization 问题

**问题**: 随机化参数不生效或效果异常

**原因**:
- **Mujoco 参数配置理解不正确**: 不同物理引擎的参数含义和取值范围不同
- 某些参数可能不支持动态修改
- 参数设置的时机不对

**解决方案**:
1. **理解 Mujoco 参数含义**: 参考 [Mujoco 文档](https://mujoco.readthedocs.io/) 了解正确的参数配置
2. **验证参数范围**: 确保随机化范围在 Mujoco 支持的范围内
3. **在重置时应用**: 在 `reset_agents()` 中设置物理参数

```python
def reset_agents(self, agents):
    # 随机化摩擦力
    geom_friction_dict = {}
    for agent in agents:
        friction_dict = agent.randomize_foot_friction(self.model.get_geom_dict())
        geom_friction_dict.update(friction_dict)
    self.set_geom_friction(geom_friction_dict)
    
    # 随机化质量
    weight_load_dict = {}
    for agent in agents:
        weight = self.np_random.uniform(agent.added_mass_range[0], agent.added_mass_range[1])
        # ...
    self.add_extra_weight(weight_load_dict)
```

### 6.6 奖励设计问题

**问题**: 训练不收敛或行为异常

**原因**:
- 奖励系数设置不当
- 奖励冲突或稀疏

**解决方案**:
1. **使用奖励打印工具**: 监控各项奖励
2. **逐步调整系数**: 从简单奖励开始
3. **归一化奖励**: 确保各项奖励量级相近

```python
# 使用 RewardPrinter
from orca_gym.utils.reward_printer import RewardPrinter

self._reward_printer = RewardPrinter()

def _compute_reward_xxx(self, coeff):
    reward = ...
    self._reward_printer.print_reward("XXX reward: ", reward, coeff * self.dt)
    return reward
```

### 6.7 训练稳定性问题

**问题**: 训练过程中出现 NaN 或 Inf

**原因**:
- 奖励爆炸
- 观测值未归一化
- 学习率过大

**解决方案**:
1. **Clip 观测和奖励**: 限制数值范围
2. **检查奖励计算**: 确保无除零错误
3. **降低学习率**: 从 1e-4 开始

```python
# Clip 奖励
def compute_reward(self, achieved_goal, desired_goal):
    total_reward = 0.0
    for reward_fn in self._reward_functions:
        reward = reward_fn["function"](reward_fn["coeff"])
        reward = np.clip(reward, -10.0, 10.0)  # Clip
        total_reward += reward
    return total_reward
```

---

## 7. 完整示例对比

### 7.1 Isaac Gym 完整示例

```python
# train.py (Isaac Gym)
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
```

运行:
```bash
python legged_gym/scripts/train.py --task=anymal_c_rough
```

### 7.2 OrcaGym 完整示例

```python
# train.py (OrcaGym)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import yaml

def make_env(config, rank):
    def _init():
        return gym.make('LeggedGym-v0', **config)
    return _init

def train(config):
    env = SubprocVecEnv([make_env(config, i) for i in range(config['num_envs'])])
    model = PPO("MlpPolicy", env, **config['ppo_params'])
    model.learn(total_timesteps=config['total_timesteps'])
    model.save("final_model")
    env.close()

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train(config)
```

运行:
```bash
python scripts/train.py --config configs/anymal_c_rough.yaml
```

---

## 8. 总结与建议

### 8.1 移植建议

1. **循序渐进**: 从简单环境开始，逐步增加复杂度
2. **保持对照**: 保留 Isaac Gym 代码作为参考
3. **充分测试**: 每个模块单独测试后再整合
4. **记录差异**: 文档化每个关键转换点
5. **性能优化**: 最后再优化，不要过早优化

### 8.2 迁移检查清单

- [ ] 机器人 URDF 文件
- [ ] URDF 转换为 MJCF
- [ ] 场景配置 (地形、高度图)
- [ ] Agent 类实现
- [ ] 环境类实现
- [ ] 配置文件转换
- [ ] 观测计算
- [ ] 奖励函数
- [ ] 重置逻辑
- [ ] Domain Randomization
- [ ] 训练脚本
- [ ] 单步测试通过
- [ ] Episode 测试通过
- [ ] 训练收敛测试

### 8.3 资源链接

- **OrcaGym 文档**: [https://docs.orcagym.ai](https://docs.orcagym.ai)
- **OrcaGym GitHub**: [https://github.com/OrcaGym/OrcaGym](https://github.com/OrcaGym/OrcaGym)
- **Stable-Baselines3 文档**: [https://stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)
- **RLlib 文档**: [https://docs.ray.io/en/latest/rllib](https://docs.ray.io/en/latest/rllib)
- **Gymnasium 文档**: [https://gymnasium.farama.org](https://gymnasium.farama.org)

### 8.4 社区支持

如果遇到问题，可以通过以下渠道获取帮助:
- OrcaGym GitHub Issues
- OrcaGym 社区论坛
- Email: support@orcagym.ai

---

**祝移植顺利！**

