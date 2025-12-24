# OrcaGym Environments

本目录包含各种机器人环境的实现，这些是**参考实现**，供用户学习和定制。

## 📦 重要说明

⚠️ **这些环境不包含在 `orca-gym` PyPI 包中**

原因：
- 这些是**示例环境**，不是通用库
- 用户通常需要根据自己的任务**定制环境**
- 环境与特定的机器人模型和任务绑定

## 🎯 如何使用这些环境

### 方式 1：克隆仓库 + 开发模式安装（推荐）

```bash
# 1. 克隆完整仓库
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym

# 2. 以开发模式安装
pip install -e .

# 3. 直接使用
python examples/legged_gym/run_legged_sim.py
```

### 方式 2：复制到自己的项目

```bash
# 复制需要的环境
cp -r envs/manipulation my_project/envs/

# 修改导入路径
# 从: from envs.manipulation import SingleArmEnv
# 到: from my_project.envs.manipulation import SingleArmEnv
```

## 📁 环境目录结构

```
envs/
├── README.md                    # 本文件
├── __init__.py
├── aloha/                       # ALOHA 双臂机器人
├── character/                   # 人形角色
├── hand_detection/              # 手部检测环境
├── legged_gym/                  # 足式机器人
│   ├── legged_config.py         # 配置
│   ├── legged_gym_env.py        # Gym 环境
│   ├── legged_robot.py          # 机器人定义
│   ├── legged_sim_env.py        # 仿真环境
│   └── robot_config/            # 机器人配置
├── manipulation/                # 机械臂操作
│   ├── single_arm_env.py        # 单臂环境
│   ├── dual_arm_env.py          # 双臂环境
│   ├── dual_arm_robot.py        # 双臂机器人
│   └── robots/                  # 机器人模型
├── mujoco/                      # Mujoco 示例
├── realman/                     # Realman 机器人
└── wheeled_chassis/             # 轮式底盘
```

## 🦿 足式机器人 (legged_gym)

用于四足/双足机器人的 RL 训练环境。

**包含**:
- Go2, Unitree, ANYmal 等机器人配置
- 地形生成
- Curriculum learning 支持

**使用示例**:
```python
from envs.legged_gym.legged_sim_env import LeggedSimEnv
from envs.legged_gym.legged_config import LeggedEnvConfig

env = LeggedSimEnv(
    orcagym_addr="localhost:50051",
    config=LeggedEnvConfig()
)
```

**相关示例**: `examples/legged_gym/`

## 🦾 机械臂操作 (manipulation)

单臂和双臂机械臂操作环境。

**包含**:
- 单臂环境 (Franka, UR5, etc.)
- 双臂环境 (OpenLoong, ALOHA)
- 多种控制模式 (关节控制, OSC, IK)

**使用示例**:
```python
from envs.manipulation.single_arm_env import SingleArmEnv, RunMode

env = SingleArmEnv(
    orcagym_addr="localhost:50051",
    robot_name="franka",
    run_mode=RunMode.SIM
)
```

**相关示例**: `examples/imitation/`, `examples/openpi/`

## 🤖 ALOHA 机器人 (aloha)

ALOHA 双臂移动操作平台。

**使用示例**:
```python
from envs.aloha.aloha_env import AlohaEnv

env = AlohaEnv(orcagym_addr="localhost:50051")
```

**相关示例**: `examples/openpi/`

## 🚗 轮式底盘 (wheeled_chassis)

差速驱动和阿克曼转向底盘。

**使用示例**:
```python
from envs.wheeled_chassis.wheeled_chassis_env import WheeledChassisEnv

env = WheeledChassisEnv(orcagym_addr="localhost:50051")
```

**相关示例**: `examples/wheeled_chassis/`

## 👤 人形角色 (character)

人形角色控制和动画。

**相关示例**: `examples/character/`

## 🎮 Realman 机器人 (realman)

Realman RM65B/RM75BV 机器人接口。

**相关示例**: `examples/realman/`

## 🔧 定制自己的环境

### 1. 继承基类

所有环境都继承自 `orca_gym.environment.OrcaGymBaseEnv`:

```python
from orca_gym.environment import OrcaGymRemoteEnv
import gymnasium as gym

class MyCustomEnv(OrcaGymRemoteEnv):
    def __init__(self, **kwargs):
        super().__init__(
            frame_skip=5,
            orcagym_addr="localhost:50051",
            agent_names=["my_robot"],
            time_step=0.002,
            **kwargs
        )
        
    def _get_obs(self):
        # 自定义观察
        pass
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        # 自定义奖励
        pass
```

### 2. 定义观察空间

```python
def _get_obs(self):
    obs = {
        'observation': np.concatenate([
            self.data.qpos,  # 关节位置
            self.data.qvel,  # 关节速度
        ]),
        'achieved_goal': self.get_end_effector_pos(),
        'desired_goal': self.goal_pos,
    }
    return obs
```

### 3. 定义动作空间

```python
self.action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(7,),  # 7 DOF
    dtype=np.float32
)
```

### 4. 实现奖励函数

```python
def compute_reward(self, achieved_goal, desired_goal, info):
    distance = np.linalg.norm(achieved_goal - desired_goal)
    return -distance
```

## 📚 环境开发指南

### 必需方法

```python
class MyEnv(OrcaGymBaseEnv):
    def reset(self, *, seed=None, options=None):
        """重置环境"""
        pass
        
    def step(self, action):
        """执行动作"""
        pass
        
    def _get_obs(self):
        """获取观察"""
        pass
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        """计算奖励"""
        pass
```

### 可选方法

```python
def render(self):
    """渲染（通常由 OrcaStudio 处理）"""
    pass
    
def close(self):
    """清理资源"""
    super().close()
```

## 🧪 测试环境

```python
import gymnasium as gym
from envs.manipulation import SingleArmEnv

# 创建环境
env = SingleArmEnv(orcagym_addr="localhost:50051")

# 测试 reset
obs, info = env.reset()
print(f"Observation shape: {obs['observation'].shape}")

# 测试 step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}")

env.close()
```

## 📖 相关文档

- [核心库 API](../orca_gym/README.md)
- [示例代码](../examples/README.md)
- [Gymnasium 文档](https://gymnasium.farama.org/)

## 💡 最佳实践

1. **继承而不是修改** - 继承现有环境来定制，不要直接修改
2. **配置化** - 使用配置文件而不是硬编码参数
3. **文档化** - 添加清晰的文档说明环境
4. **测试** - 编写单元测试验证环境
5. **版本控制** - 记录环境的变更

## 🆘 常见问题

### Q: 为什么这些环境不在 PyPI 包中？

A: 因为这些是**示例和参考实现**，用户通常需要根据自己的任务定制。将其作为独立文件更灵活。

### Q: 如何在我的项目中使用这些环境？

A: 有两种方式：
1. 克隆仓库，以开发模式安装
2. 复制需要的环境到你的项目，修改导入路径

### Q: 可以直接修改这些环境吗？

A: 建议**不要直接修改**。创建新的类继承这些环境，然后覆盖需要的方法。

### Q: 如何贡献新环境？

A: 
1. Fork 仓库
2. 在 `envs/` 下添加你的环境
3. 在 `examples/` 下添加使用示例
4. 提交 Pull Request

## 📞 获取帮助

- 查看示例代码: `examples/`
- 查看核心库文档: `orca_gym/`
- 提交 Issue: https://github.com/openverse-orca/OrcaGym/issues
- 联系: huangwei@orca3d.cn

---

**记住**: 这些环境是起点，不是终点。根据你的需求自由定制！🚀

