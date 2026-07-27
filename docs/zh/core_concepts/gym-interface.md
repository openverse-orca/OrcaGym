# 🏋️ Gymnasium 接口

OrcaGym 严格遵循 Gymnasium 规范，提供标准的 RL 环境接口。

## 环境基类

```
gymnasium.Env
  └── OrcaGymEulerEnv      # 👈 推荐：当前主路径
        └── 你的环境类
```

## OrcaGymEulerEnv

`OrcaGymEulerEnv` 是你编写自定义环境的推荐基类。它封装了仿真核心功能，让你专注于任务逻辑。

### 构造参数

```python
class MyEnv(OrcaGymEulerEnv):
    def __init__(self, ...):
        super().__init__(
            frame_skip: int,          # 每次 step() 对应的物理步数
            orcagym_addr: str,        # gRPC 服务地址
            agent_names: list[str],   # 智能体名称列表
            time_step: float,         # 物理时间步长 (秒)
            *,
            model_xml_path: str | None = None,   # 本地模型路径（离线模式）
            skip_grpc_load: bool = False,        # True → 离线模式
            render_mode: str = "human",          # "human" / "none"
            **kwargs,
        )
```

### 必须实现的抽象方法

每个子类必须重写以下三个方法：

```python
def step(self, action) -> tuple:
    """执行一步仿真，返回 obs, reward, terminated, truncated, info"""
    ...

def reset_model(self) -> tuple:
    """重置机器人状态到初始位姿"""
    ...

def _get_obs(self):
    """构建观测"""
    ...
```

### 关键属性

```python
env.data          # 完整状态只读视图
env.model         # 模型结构信息
env.sim_config    # 求解器配置
env.ctrl          # 当前控制输入

@property
env.dt            # 环境时间步长 = timestep × frame_skip
```

### 关键方法

```python
# 仿真步进 —— 核心方法
env.do_simulation(ctrl, n_frames)

# 状态设置
env.set_joint_qpos(qpos)
env.set_joint_qvel(qvel)

# 前向计算（更新派生量）
env.mj_forward()

# 施加外力
env.apply_body_force(body_name, force, torque)

# Mocap 控制
env.set_mocap_pos_and_quat(mocap_dict)
```

## 观测与动作空间

### 动作空间

OrcaGym 自动从执行器控制范围生成 `action_space`：

```python
# internal: action_space = spaces.Box(low=ctrl_range[:,0], high=ctrl_range[:,1])
print(env.action_space)  # Box(low=-1.0, high=1.0, shape=(nu,), float32)
```

### 观测空间

观测空间由你在 `_get_obs()` 中定义：

```python
def _get_obs(self):
    """构建观测，返回 np.ndarray 或 dict"""
    obs = np.concatenate([self.data.qpos, self.data.qvel])
    return obs.astype(np.float32)

# 首次 reset 时自动推断 observation_space
```

### 支持的空间类型

| 类型 | Python 类 | 说明 |
|------|-----------|------|
| Box 连续空间 | `spaces.Box` | 观测/动作是 numpy 数组 |
| Dict 字典空间 | `spaces.Dict` | 观测是字典（多模态） |
| Discrete 离散空间 | `spaces.Discrete` | 离散动作 |

## 创建环境

### 直接实例化（推荐）

```python
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

class MyTaskEnv(OrcaGymEulerEnv):
    ...

env = MyTaskEnv(
    frame_skip=20,
    orcagym_addr="localhost:50051",
    agent_names=["agent0"],
    time_step=0.001,
)
```

### 通过 gym.make

```python
import gymnasium as gym

env = gym.make(
    "YourTaskEnv-v0",
    frame_skip=20,
    orcagym_addr="localhost:50051",
    agent_names=["agent0"],
    time_step=0.001,
)
```

### 通过注册

```python
gym.register(
    id="MyTask-v0",
    entry_point="my_package:MyTaskEnv",
    max_episode_steps=1000,
)
env = gym.make("MyTask-v0", **specific_kwargs)
```

## 关键约定

1. **`env.dt`** 是策略控制周期，不是 MuJoCo 的 `timestep`
2. **`frame_skip`** 决定一个 step 内执行多少次物理步
3. **观测应在 `_get_obs()` 中构建**，`do_simulation()` 后数据已更新
4. **多智能体时**，body/joint/actuator 名称自动加 agent 前缀（通过 `self.body()` 等方法）
5. **`do_simulation()`** 返回后 `env.data` 已自动同步
6. **状态配置**通过 `env.sim_config` 访问
