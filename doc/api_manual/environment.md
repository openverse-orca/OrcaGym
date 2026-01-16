# OrcaGym API Manual: `orca_gym/environment`

> **📖 这是什么文档？**  
> 这是 `orca_gym/environment` 模块的完整 API 参考手册，采用“索引 + 详情”的版式设计，便于快速查找和深入学习。

## 📚 文档说明

### 文档特点

- **索引优先**：每个模块和类都提供索引表格，方便快速浏览和定位
- **详情展开**：点击或展开详情部分，查看完整的方法签名、参数说明和使用示例
- **面向本地环境**：本手册主要覆盖本地环境实现（`OrcaGymLocalEnv`）
- **仅公开接口**：只列出 public 符号（不以下划线开头），聚焦实际可用的 API

### 如何使用本手册

1. **快速查找**：使用下方的模块索引表格，找到你需要的模块
2. **浏览类列表**：进入模块后，先看“Classes（索引）”表格，了解有哪些类
3. **查看方法**：每个类都有“方法索引”表格，快速了解可用方法
4. **深入阅读**：展开“方法详情”部分，查看完整的签名、参数说明和使用示例

### 相关文档

- **快速概览**：查看 [`API_REFERENCE.md`](../API_REFERENCE.md) 了解整体架构和典型调用链
- **详细参考**：查看 [`api_detail/environment.md`](../api_detail/environment.md) 获取自动生成的完整 API 签名列表
- **Core 模块**：查看 [`api_manual/core.md`](core.md) 了解底层核心接口

---

## 📦 Modules（索引）

快速浏览所有模块，点击模块名跳转到详细内容：

| Module | 说明 |
| --- | --- |
| [`orca_gym/environment/orca_gym_env.py`](#orca_gymenvironmentorca_gym_envpy) | **基础环境类**：所有 OrcaGym 环境的基类，提供 Gymnasium 标准接口 |

---

## `orca_gym/environment/orca_gym_env.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `RewardType` | _No docstring._ |
| `OrcaGymBaseEnv` | Superclass for all OrcaSim environments. |

### Classes（详情）

#### `class RewardType`

#### `class OrcaGymBaseEnv`

> Superclass for all OrcaSim environments.

##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymBaseEnv.step` | `def step(self, action) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]` | _No docstring._ |  |
| `OrcaGymBaseEnv.reset_model` | `def reset_model(self) -> tuple[dict, dict]` | Reset the robot degrees of freedom (qpos and qvel). |  |
| `OrcaGymBaseEnv.initialize_simulation` | `def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]` | Initialize MuJoCo simulation data structures mjModel and mjData. |  |
| `OrcaGymBaseEnv.render` | `def render(self) -> Union[NDArray[np.float64], None]` | Render a frame from the MuJoCo simulation as specified by the render_mode. |  |
| `OrcaGymBaseEnv.generate_action_space` | `def generate_action_space(self, bounds) -> Space` | 生成环境的动作空间 |  |
| `OrcaGymBaseEnv.generate_observation_space` | `def generate_observation_space(self, obs) -> spaces.Space` | 生成环境的观测空间 |  |
| `OrcaGymBaseEnv.reset` | `def reset(self, *, seed=None, options=None)` | _No docstring._ |  |
| `OrcaGymBaseEnv.set_seed_value` | `def set_seed_value(self, seed=None)` | 设置随机数种子 |  |
| `OrcaGymBaseEnv.body` | `def body(self, name, agent_id=None) -> str` | 获取带智能体前缀的 body 名称 |  |
| `OrcaGymBaseEnv.joint` | `def joint(self, name, agent_id=None) -> str` | 获取带智能体前缀的关节名称 |  |
| `OrcaGymBaseEnv.actuator` | `def actuator(self, name, agent_id=None) -> str` | 获取带智能体前缀的执行器名称 |  |
| `OrcaGymBaseEnv.site` | `def site(self, name, agent_id=None) -> str` | 获取带智能体前缀的 site 名称 |  |
| `OrcaGymBaseEnv.mocap` | `def mocap(self, name, agent_id=None) -> str` | 获取带智能体前缀的 mocap 名称 |  |
| `OrcaGymBaseEnv.sensor` | `def sensor(self, name, agent_id=None) -> str` | 获取带智能体前缀的传感器名称 |  |
| `OrcaGymBaseEnv.dt` | `def dt(self) -> float` | 获取环境时间步长（物理时间步长 × frame_skip） | `@property` |
| `OrcaGymBaseEnv.agent_num` | `def agent_num(self) -> int` | 获取智能体数量 | `@property` |
| `OrcaGymBaseEnv.do_simulation` | `def do_simulation(self, ctrl, n_frames) -> None` | Step the simulation n number of frames and applying a control action. |  |
| `OrcaGymBaseEnv.close` | `def close(self)` | Close all processes like rendering contexts |  |
| `OrcaGymBaseEnv.initialize_grpc` | `def initialize_grpc(self)` | Initialize the GRPC communication channel. |  |
| `OrcaGymBaseEnv.pause_simulation` | `def pause_simulation(self)` | Pause the simulation. |  |
| `OrcaGymBaseEnv.init_qpos_qvel` | `def init_qpos_qvel(self)` | Init qpos and qvel of the model. |  |
| `OrcaGymBaseEnv.reset_simulation` | `def reset_simulation(self)` | Reset the simulation. |  |
| `OrcaGymBaseEnv.set_time_step` | `def set_time_step(self, time_step)` | Set the time step of the simulation. |  |

##### 方法详情

#### `OrcaGymBaseEnv.step`

Signature:
```python
def step(self, action) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]
```

Description:
_No docstring._

#### `OrcaGymBaseEnv.reset_model`

Signature:
```python
def reset_model(self) -> tuple[dict, dict]
```

Description:
Reset the robot degrees of freedom (qpos and qvel).
Implement this in each subclass.

#### `OrcaGymBaseEnv.initialize_simulation`

Signature:
```python
def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]
```

Description:
Initialize MuJoCo simulation data structures mjModel and mjData.

#### `OrcaGymBaseEnv.render`

Signature:
```python
def render(self) -> Union[NDArray[np.float64], None]
```

Description:
Render a frame from the MuJoCo simulation as specified by the render_mode.

#### `OrcaGymBaseEnv.generate_action_space`

Signature:
```python
def generate_action_space(self, bounds) -> Space
```

Description:
生成环境的动作空间

术语说明:
    - 动作空间 (Action Space): 强化学习中智能体可以执行的所有动作的集合
    - Box Space: 连续动作空间，每个维度有上下界限制
    - 动作: 发送给执行器的控制命令，通常是扭矩、位置或速度

使用示例:
    ```python
    # 根据执行器控制范围生成动作空间
    ctrlrange = self.model.get_actuator_ctrlrange()
    self.action_space = self.generate_action_space(ctrlrange)
    # 动作空间形状: (nu,)，每个值在 [min, max] 范围内
    ```

#### `OrcaGymBaseEnv.generate_observation_space`

Signature:
```python
def generate_observation_space(self, obs) -> spaces.Space
```

Description:
生成环境的观测空间

术语说明:
    - 观测空间 (Observation Space): 强化学习中智能体能够观察到的状态信息集合
    - 观测 (Observation): 智能体在每个时间步接收到的状态信息
    - Dict Space: 字典类型的观测空间，包含多个子空间
    - Box Space: 连续观测空间，每个维度有上下界限制

使用示例:
    ```python
    # 根据观测数据生成观测空间
    obs = self._get_obs()  # 获取示例观测
    self.observation_space = self.generate_observation_space(obs)
    # 观测空间可能是 Dict 或 Box，取决于 obs 的类型
    ```

#### `OrcaGymBaseEnv.reset`

Signature:
```python
def reset(self, *, seed=None, options=None)
```

Description:
_No docstring._

#### `OrcaGymBaseEnv.set_seed_value`

Signature:
```python
def set_seed_value(self, seed=None)
```

Description:
设置随机数种子

#### `OrcaGymBaseEnv.body`

Signature:
```python
def body(self, name, agent_id=None) -> str
```

Description:
获取带智能体前缀的 body 名称

#### `OrcaGymBaseEnv.joint`

Signature:
```python
def joint(self, name, agent_id=None) -> str
```

Description:
获取带智能体前缀的关节名称

#### `OrcaGymBaseEnv.actuator`

Signature:
```python
def actuator(self, name, agent_id=None) -> str
```

Description:
获取带智能体前缀的执行器名称

#### `OrcaGymBaseEnv.site`

Signature:
```python
def site(self, name, agent_id=None) -> str
```

Description:
获取带智能体前缀的 site 名称

#### `OrcaGymBaseEnv.mocap`

Signature:
```python
def mocap(self, name, agent_id=None) -> str
```

Description:
获取带智能体前缀的 mocap 名称

#### `OrcaGymBaseEnv.sensor`

Signature:
```python
def sensor(self, name, agent_id=None) -> str
```

Description:
获取带智能体前缀的传感器名称

#### `OrcaGymBaseEnv.dt` (property)

Signature:
```python
def dt(self) -> float
```

Description:
获取环境时间步长（物理时间步长 × frame_skip）

这是 Gym 环境的时间步长，即每次 step() 对应的时间间隔。
等于物理仿真时间步长乘以 frame_skip。

术语说明:
    - 时间步长 (Time Step): 每次仿真步进对应的时间间隔
    - 物理时间步长 (Timestep): MuJoCo 物理引擎的时间步长，通常很小（如 0.001s）
    - frame_skip: 每次 Gym step() 执行的物理步进次数，用于加速仿真
    - 控制频率: 1/dt，表示每秒执行多少次控制更新

使用示例:
    ```python
    # 计算控制频率
    REALTIME_STEP = TIME_STEP * FRAME_SKIP * ACTION_SKIP
    CONTROL_FREQ = 1 / REALTIME_STEP  # 50 Hz

    # 在循环中使用
    dt = env.dt  # 获取环境时间步长
    sim_time += dt  # 累计仿真时间
    ```

#### `OrcaGymBaseEnv.agent_num` (property)

Signature:
```python
def agent_num(self) -> int
```

Description:
获取智能体数量

#### `OrcaGymBaseEnv.do_simulation`

Signature:
```python
def do_simulation(self, ctrl, n_frames) -> None
```

Description:
Step the simulation n number of frames and applying a control action.

#### `OrcaGymBaseEnv.close`

Signature:
```python
def close(self)
```

Description:
Close all processes like rendering contexts

#### `OrcaGymBaseEnv.initialize_grpc`

Signature:
```python
def initialize_grpc(self)
```

Description:
Initialize the GRPC communication channel.

#### `OrcaGymBaseEnv.pause_simulation`

Signature:
```python
def pause_simulation(self)
```

Description:
Pause the simulation.

#### `OrcaGymBaseEnv.init_qpos_qvel`

Signature:
```python
def init_qpos_qvel(self)
```

Description:
Init qpos and qvel of the model.

#### `OrcaGymBaseEnv.reset_simulation`

Signature:
```python
def reset_simulation(self)
```

Description:
Reset the simulation.

#### `OrcaGymBaseEnv.set_time_step`

Signature:
```python
def set_time_step(self, time_step)
```

Description:
Set the time step of the simulation.

---
