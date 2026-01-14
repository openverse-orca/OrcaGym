# OrcaGym API Manual: `orca_gym/environment`

> **📖 这是什么文档？**  
> 这是 `orca_gym/environment` 模块的完整 API 参考手册，采用“索引 + 详情”的版式设计，便于快速查找和深入学习。

## 📚 文档说明

### 文档特点

- **索引优先**：每个模块和类都提供索引表格，方便快速浏览和定位
- **详情展开**：点击或展开详情部分，查看完整的方法签名、参数说明和使用示例
- **面向本地环境**：本手册主要覆盖本地环境实现（`OrcaGymLocalEnv`），远程环境（`OrcaGymRemoteEnv`）相关内容已省略
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
| [`orca_gym/environment/orca_gym_local_env.py`](#orca_gymenvironmentorca_gym_local_envpy) | **本地环境实现**：绑定本地 MuJoCo backend 的完整环境实现（最常用） |
| [`orca_gym/environment/async_env/orca_gym_async_agent.py`](#orca_gymenvironmentasync_envorca_gym_async_agentpy) | **异步智能体**：异步环境中的智能体封装 |
| [`orca_gym/environment/async_env/orca_gym_async_env.py`](#orca_gymenvironmentasync_envorca_gym_async_envpy) | **异步环境**：支持多智能体异步执行的环境实现 |
| [`orca_gym/environment/async_env/orca_gym_vector_env.py`](#orca_gymenvironmentasync_envorca_gym_vector_envpy) | **向量化环境**：将多个环境包装为 Gymnasium VectorEnv 接口 |
| [`orca_gym/environment/async_env/subproc_vec_env.py`](#orca_gymenvironmentasync_envsubproc_vec_envpy) | **多进程向量环境**：基于多进程的向量化环境实现 |
| [`orca_gym/environment/async_env/single_agent_env_runner.py`](#orca_gymenvironmentasync_envsingle_agent_env_runnerpy) | **单智能体运行器**：为 RLLib 等框架提供的单智能体环境运行器 |

---

## `orca_gym/environment/async_env/orca_gym_async_agent.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymAsyncAgent` | _No docstring._ |

### Classes（详情）

#### `class OrcaGymAsyncAgent`

<details>
<summary>Class docstring</summary>

_No docstring._

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymAsyncAgent.dt` | `def dt(self) -> float` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.name` | `def name(self) -> str` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.name_space` | `def name_space(self, name) -> str` | _No docstring._ |  |
| `OrcaGymAsyncAgent.name_space_list` | `def name_space_list(self, names) -> list[str]` | _No docstring._ |  |
| `OrcaGymAsyncAgent.joint_names` | `def joint_names(self) -> list[str]` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.actuator_names` | `def actuator_names(self) -> list[str]` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.site_names` | `def site_names(self) -> list[str]` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.sensor_names` | `def sensor_names(self) -> list[str]` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.nu` | `def nu(self) -> int` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.nq` | `def nq(self) -> int` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.nv` | `def nv(self) -> int` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.truncated` | `def truncated(self) -> bool` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.ctrl_start` | `def ctrl_start(self) -> int` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.action_range` | `def action_range(self) -> np.ndarray` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.kps` | `def kps(self) -> float` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.kds` | `def kds(self) -> float` | _No docstring._ | `@property` |
| `OrcaGymAsyncAgent.get_obs` | `def get_obs(self, **kwargs)` | _No docstring._ |  |
| `OrcaGymAsyncAgent.init_ctrl_info` | `def init_ctrl_info(self, actuator_dict, joint_dict) -> None` | _No docstring._ |  |
| `OrcaGymAsyncAgent.get_ctrl_info` | `def get_ctrl_info(self) -> dict` | _No docstring._ |  |
| `OrcaGymAsyncAgent.init_joint_index` | `def init_joint_index(self, qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length)` | Joint index is specific to the agent and is defined in the subclass. |  |
| `OrcaGymAsyncAgent.set_action_space` | `def set_action_space(self) -> None` | Action space is specific to the agent and is defined in the subclass. |  |
| `OrcaGymAsyncAgent.on_step` | `def on_step(self, action, **kwargs)` | Called after each step in the environment. |  |
| `OrcaGymAsyncAgent.step` | `def step(self, action, **kwargs)` | _No docstring._ |  |
| `OrcaGymAsyncAgent.on_reset` | `def on_reset(self, **kwargs)` | Called after each reset in the environment. |  |
| `OrcaGymAsyncAgent.reset` | `def reset(self, np_random, **kwargs)` | _No docstring._ |  |
| `OrcaGymAsyncAgent.is_success` | `def is_success(self, achieved_goal, desired_goal) -> np.float32` | _No docstring._ |  |
| `OrcaGymAsyncAgent.is_terminated` | `def is_terminated(self, achieved_goal, desired_goal) -> bool` | _No docstring._ |  |
| `OrcaGymAsyncAgent.compute_reward` | `def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat` | _No docstring._ |  |
| `OrcaGymAsyncAgent.set_init_state` | `def set_init_state(self, joint_qpos, init_site_pos_quat) -> None` | _No docstring._ |  |
| `OrcaGymAsyncAgent.get_action_size` | `def get_action_size(self) -> int` | Action size can be overridden in the subclass. |  |
| `OrcaGymAsyncAgent.compute_torques` | `def compute_torques(self, qpos_buffer, qvel_buffer) -> np.ndarray` | _No docstring._ |  |
| `OrcaGymAsyncAgent.setup_curriculum` | `def setup_curriculum(self, curriculum) -> None` | _No docstring._ |  |

##### 方法详情

###### `OrcaGymAsyncAgent.dt` (property)

Signature:

```python
def dt(self) -> float
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.name` (property)

Signature:

```python
def name(self) -> str
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.name_space`

Signature:

```python
def name_space(self, name) -> str
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.name_space_list`

Signature:

```python
def name_space_list(self, names) -> list[str]
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.joint_names` (property)

Signature:

```python
def joint_names(self) -> list[str]
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.actuator_names` (property)

Signature:

```python
def actuator_names(self) -> list[str]
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.site_names` (property)

Signature:

```python
def site_names(self) -> list[str]
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.sensor_names` (property)

Signature:

```python
def sensor_names(self) -> list[str]
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.nu` (property)

Signature:

```python
def nu(self) -> int
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.nq` (property)

Signature:

```python
def nq(self) -> int
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.nv` (property)

Signature:

```python
def nv(self) -> int
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.truncated` (property)

Signature:

```python
def truncated(self) -> bool
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.ctrl_start` (property)

Signature:

```python
def ctrl_start(self) -> int
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.action_range` (property)

Signature:

```python
def action_range(self) -> np.ndarray
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.kps` (property)

Signature:

```python
def kps(self) -> float
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.kds` (property)

Signature:

```python
def kds(self) -> float
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.get_obs`

Signature:

```python
def get_obs(self, **kwargs)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.init_ctrl_info`

Signature:

```python
def init_ctrl_info(self, actuator_dict, joint_dict) -> None
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.get_ctrl_info`

Signature:

```python
def get_ctrl_info(self) -> dict
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.init_joint_index`

Signature:

```python
def init_joint_index(self, qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length)
```

<details>
<summary>Docstring</summary>

Joint index is specific to the agent and is defined in the subclass.

</details>


###### `OrcaGymAsyncAgent.set_action_space`

Signature:

```python
def set_action_space(self) -> None
```

<details>
<summary>Docstring</summary>

Action space is specific to the agent and is defined in the subclass.

</details>


###### `OrcaGymAsyncAgent.on_step`

Signature:

```python
def on_step(self, action, **kwargs)
```

<details>
<summary>Docstring</summary>

Called after each step in the environment.
Implement this method in the subclass to perform additional operations.

</details>


###### `OrcaGymAsyncAgent.step`

Signature:

```python
def step(self, action, **kwargs)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.on_reset`

Signature:

```python
def on_reset(self, **kwargs)
```

<details>
<summary>Docstring</summary>

Called after each reset in the environment.
Implement this method in the subclass to perform additional operations.

</details>


###### `OrcaGymAsyncAgent.reset`

Signature:

```python
def reset(self, np_random, **kwargs)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.is_success`

Signature:

```python
def is_success(self, achieved_goal, desired_goal) -> np.float32
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.is_terminated`

Signature:

```python
def is_terminated(self, achieved_goal, desired_goal) -> bool
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.compute_reward`

Signature:

```python
def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.set_init_state`

Signature:

```python
def set_init_state(self, joint_qpos, init_site_pos_quat) -> None
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.get_action_size`

Signature:

```python
def get_action_size(self) -> int
```

<details>
<summary>Docstring</summary>

Action size can be overridden in the subclass.
In most of cases, this is the number of actuators in the robot.
But in some cases, the action size may be different.

</details>


###### `OrcaGymAsyncAgent.compute_torques`

Signature:

```python
def compute_torques(self, qpos_buffer, qvel_buffer) -> np.ndarray
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncAgent.setup_curriculum`

Signature:

```python
def setup_curriculum(self, curriculum) -> None
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>

---

## `orca_gym/environment/async_env/orca_gym_async_env.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymAsyncEnv` | _No docstring._ |

### Classes（详情）

#### `class OrcaGymAsyncEnv`

<details>
<summary>Class docstring</summary>

_No docstring._

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymAsyncEnv.set_obs_space` | `def set_obs_space(self)` | _No docstring._ |  |
| `OrcaGymAsyncEnv.set_action_space` | `def set_action_space(self) -> None` | _No docstring._ |  |
| `OrcaGymAsyncEnv.initialize_agents` | `def initialize_agents(self, entry, *args, **kwargs)` | _No docstring._ |  |
| `OrcaGymAsyncEnv.get_obs` | `def get_obs(self) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], np.ndarray, np.ndarray]` | Observation is environment specific and is defined in the subclass. |  |
| `OrcaGymAsyncEnv.reset_agents` | `def reset_agents(self, agents)` | Do specific reset operations for each agent. It is defined in the subclass. |  |
| `OrcaGymAsyncEnv.step_agents` | `def step_agents(self, action) -> None` | Do specific operations each step in the environment. It is defined in the subclass. |  |
| `OrcaGymAsyncEnv.step` | `def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]` | _No docstring._ |  |
| `OrcaGymAsyncEnv.reset_model` | `def reset_model(self) -> tuple[ObsType, dict[str, np.ndarray]]` | _No docstring._ |  |
| `OrcaGymAsyncEnv.get_observation` | `def get_observation(self, obs=None)` | _No docstring._ |  |
| `OrcaGymAsyncEnv.init_agent_joint_index` | `def init_agent_joint_index(self)` | _No docstring._ |  |
| `OrcaGymAsyncEnv.reorder_agents` | `def reorder_agents(self)` | _No docstring._ |  |
| `OrcaGymAsyncEnv.generate_action_scale_array` | `def generate_action_scale_array(self, ctrl_info) -> np.ndarray` | _No docstring._ |  |
| `OrcaGymAsyncEnv.setup_curriculum` | `def setup_curriculum(self, curriculum) -> None` | _No docstring._ |  |

##### 方法详情

###### `OrcaGymAsyncEnv.set_obs_space`

Signature:

```python
def set_obs_space(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncEnv.set_action_space`

Signature:

```python
def set_action_space(self) -> None
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncEnv.initialize_agents`

Signature:

```python
def initialize_agents(self, entry, *args, **kwargs)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncEnv.get_obs`

Signature:

```python
def get_obs(self) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], np.ndarray, np.ndarray]
```

<details>
<summary>Docstring</summary>

Observation is environment specific and is defined in the subclass.

</details>


###### `OrcaGymAsyncEnv.reset_agents`

Signature:

```python
def reset_agents(self, agents)
```

<details>
<summary>Docstring</summary>

Do specific reset operations for each agent. It is defined in the subclass.

</details>


###### `OrcaGymAsyncEnv.step_agents`

Signature:

```python
def step_agents(self, action) -> None
```

<details>
<summary>Docstring</summary>

Do specific operations each step in the environment. It is defined in the subclass.

</details>


###### `OrcaGymAsyncEnv.step`

Signature:

```python
def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncEnv.reset_model`

Signature:

```python
def reset_model(self) -> tuple[ObsType, dict[str, np.ndarray]]
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncEnv.get_observation`

Signature:

```python
def get_observation(self, obs=None)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncEnv.init_agent_joint_index`

Signature:

```python
def init_agent_joint_index(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncEnv.reorder_agents`

Signature:

```python
def reorder_agents(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncEnv.generate_action_scale_array`

Signature:

```python
def generate_action_scale_array(self, ctrl_info) -> np.ndarray
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncEnv.setup_curriculum`

Signature:

```python
def setup_curriculum(self, curriculum) -> None
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>

---

## `orca_gym/environment/async_env/orca_gym_vector_env.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymVectorEnv` | 实现基于多个Agent的向量环境。 |

### Classes（详情）

#### `class OrcaGymVectorEnv`

> 实现基于多个Agent的向量环境。

<details>
<summary>Class docstring</summary>

实现基于多个Agent的向量环境。
这样可以用一个Mujoco Instance 来同时运行多个 env
最后转换成VectorEnv的接口格式

# ┌────────────┬────────────────┬──────────────────────┐
# │            │ Mujoco Env     │ Robots in Mujoco     │
# ├────────────┼────────────────┼──────────────────────┤
# │ Vector Env │ OrcaGymAsyncEnv│ num_envs             │
# │ RLLib      │ num_env_runners│ num_envs_per_runner  │
# │ LeggedGym  │ subenv_num     │ agent_num            │
# └────────────┴────────────────┴──────────────────────┘


Base class for vectorized environments to run multiple independent copies of the same environment in parallel.

Vector environments can provide a linear speed-up in the steps taken per second through sampling multiple
sub-environments at the same time. Gymnasium contains two generalised Vector environments: :class:`AsyncVectorEnv`
and :class:`SyncVectorEnv` along with several custom vector environment implementations.
For :func:`reset` and :func:`step` batches `observations`, `rewards`,  `terminations`, `truncations` and
`info` for each sub-environment, see the example below. For the `rewards`, `terminations`, and `truncations`,
the data is packaged into a NumPy array of shape `(num_envs,)`. For `observations` (and `actions`, the batching
process is dependent on the type of observation (and action) space, and generally optimised for neural network
input/outputs. For `info`, the data is kept as a dictionary such that a key will give the data for all sub-environment.

For creating environments, :func:`make_vec` is a vector environment equivalent to :func:`make` for easily creating
vector environments that contains several unique arguments for modifying environment qualities, number of environment,
vectorizer type, vectorizer arguments.

Note:
    The info parameter of :meth:`reset` and :meth:`step` was originally implemented before v0.25 as a list
    of dictionary for each sub-environment. However, this was modified in v0.25+ to be a dictionary with a NumPy
    array for each key. To use the old info style, utilise the :class:`DictInfoToList` wrapper.

To avoid having to wait for all sub-environments to terminated before resetting, implementations will autoreset
sub-environments on episode end (`terminated or truncated is True`). As a result, when adding observations
to a replay buffer, this requires knowing when an observation (and info) for each sub-environment are the first
observation from an autoreset. We recommend using an additional variable to store this information such as
``has_autoreset = np.logical_or(terminated, truncated)``.

The Vector Environments have the additional attributes for users to understand the implementation

- :attr:`num_envs` - The number of sub-environment in the vector environment
- :attr:`observation_space` - The batched observation space of the vector environment
- :attr:`single_observation_space` - The observation space of a single sub-environment
- :attr:`action_space` - The batched action space of the vector environment
- :attr:`single_action_space` - The action space of a single sub-environment

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymVectorEnv.reset` | `def reset(self, *, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]` | Reset all parallel environments and return a batch of initial observations and info. |  |
| `OrcaGymVectorEnv.step` | `def step(self, actions) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]` | Take an action for each parallel environment. |  |
| `OrcaGymVectorEnv.render` | `def render(self) -> tuple[RenderFrame, ...] | None` | Returns the rendered frames from the parallel environments. |  |
| `OrcaGymVectorEnv.close` | `def close(self, **kwargs)` | Close all parallel environments and release resources. |  |

##### 方法详情

###### `OrcaGymVectorEnv.reset`

Signature:

```python
def reset(self, *, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]
```

<details>
<summary>Docstring</summary>

Reset all parallel environments and return a batch of initial observations and info.

Args:
    seed: The environment reset seed
    options: If to return the options

Returns:
    A batch of observations and info from the vectorized environment.

Example:
    >>> import gymnasium as gym
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    >>> observations, infos = envs.reset(seed=42)
    >>> observations
    array([[ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ],
           [ 0.01522993, -0.04562247, -0.04799704,  0.03392126],
           [-0.03774345, -0.02418869, -0.00942293,  0.0469184 ]],
          dtype=float32)
    >>> infos
    {}

</details>


###### `OrcaGymVectorEnv.step`

Signature:

```python
def step(self, actions) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]
```

<details>
<summary>Docstring</summary>

Take an action for each parallel environment.

Args:
    actions: Batch of actions with the :attr:`action_space` shape.

Returns:
    Batch of (observations, rewards, terminations, truncations, infos)

Note:
    As the vector environments autoreset for a terminating and truncating sub-environments, this will occur on
    the next step after `terminated or truncated is True`.

Example:
    >>> import gymnasium as gym
    >>> import numpy as np
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    >>> _ = envs.reset(seed=42)
    >>> actions = np.array([1, 0, 1], dtype=np.int32)
    >>> observations, rewards, terminations, truncations, infos = envs.step(actions)
    >>> observations
    array([[ 0.02727336,  0.18847767,  0.03625453, -0.26141977],
           [ 0.01431748, -0.24002443, -0.04731862,  0.3110827 ],
           [-0.03822722,  0.1710671 , -0.00848456, -0.2487226 ]],
          dtype=float32)
    >>> rewards
    array([1., 1., 1.])
    >>> terminations
    array([False, False, False])
    >>> terminations
    array([False, False, False])
    >>> infos
    {}

</details>


###### `OrcaGymVectorEnv.render`

Signature:

```python
def render(self) -> tuple[RenderFrame, ...] | None
```

<details>
<summary>Docstring</summary>

Returns the rendered frames from the parallel environments.

Returns:
    A tuple of rendered frames from the parallel environments

</details>


###### `OrcaGymVectorEnv.close`

Signature:

```python
def close(self, **kwargs)
```

<details>
<summary>Docstring</summary>

Close all parallel environments and release resources.

It also closes all the existing image viewers, then calls :meth:`close_extras` and set
:attr:`closed` as ``True``.

Warnings:
    This function itself does not close the environments, it should be handled
    in :meth:`close_extras`. This is generic for both synchronous and asynchronous
    vectorized environments.

Note:
    This will be automatically called when garbage collected or program exited.

Args:
    **kwargs: Keyword arguments passed to :meth:`close_extras`

</details>

---

## `orca_gym/environment/async_env/single_agent_env_runner.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymAsyncSingleAgentEnvRunner` | Wrap the SingleAgentEnvRunner to support orca gym asynchronous environments. |

### Classes（详情）

#### `class OrcaGymAsyncSingleAgentEnvRunner`

> Wrap the SingleAgentEnvRunner to support orca gym asynchronous environments.

<details>
<summary>Class docstring</summary>

Wrap the SingleAgentEnvRunner to support orca gym asynchronous environments.

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymAsyncSingleAgentEnvRunner.make_env` | `def make_env(self) -> None` | Creates a vectorized gymnasium env and stores it in `self.env`. | `@override(SingleAgentEnvRunner)` |

##### 方法详情

###### `OrcaGymAsyncSingleAgentEnvRunner.make_env`

Signature:

```python
def make_env(self) -> None
```

<details>
<summary>Docstring</summary>

Creates a vectorized gymnasium env and stores it in `self.env`.

Note that users can change the EnvRunner's config (e.g. change
`self.config.env_config`) and then call this method to create new environments
with the updated configuration.

</details>

---

## `orca_gym/environment/async_env/subproc_vec_env.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymAsyncSubprocVecEnv` | Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own |

### Classes（详情）

#### `class OrcaGymAsyncSubprocVecEnv`

> Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own

<details>
<summary>Class docstring</summary>

Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
process, allowing significant speed up when the environment is computationally complex.

For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
number of logical cores on your CPU.

.. warning::

    Only 'forkserver' and 'spawn' start methods are thread-safe,
    which is important when TensorFlow sessions or other non thread-safe
    libraries are used in the parent (see issue #217). However, compared to
    'fork' they incur a small start-up cost and have restrictions on
    global variables. With those methods, users must wrap the code in an
    ``if __name__ == "__main__":`` block.
    For more information, see the multiprocessing documentation.

:param env_fns: Environments to run in subprocesses
:param start_method: method used to start the subprocesses.
       Must be one of the methods returned by multiprocessing.get_all_start_methods().
       Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.


修改baseline3中的SubprocVecEnv，对于每个环境，创建 agent_num 个智能体，每个智能体共享一个环境
通过拼接每个agent的action和obs，来实现异步多环境的功能
agent_num: 每个环境中智能体的数量

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymAsyncSubprocVecEnv.step_async` | `def step_async(self, actions) -> None` | _No docstring._ |  |
| `OrcaGymAsyncSubprocVecEnv.step_wait` | `def step_wait(self) -> VecEnvStepReturn` | _No docstring._ |  |
| `OrcaGymAsyncSubprocVecEnv.reset` | `def reset(self) -> VecEnvObs` | _No docstring._ |  |
| `OrcaGymAsyncSubprocVecEnv.close` | `def close(self) -> None` | _No docstring._ |  |
| `OrcaGymAsyncSubprocVecEnv.get_images` | `def get_images(self) -> Sequence[Optional[np.ndarray]]` | _No docstring._ |  |
| `OrcaGymAsyncSubprocVecEnv.get_attr` | `def get_attr(self, attr_name, indices=None) -> List[Any]` | Return attribute from vectorized environment (see base class). |  |
| `OrcaGymAsyncSubprocVecEnv.set_attr` | `def set_attr(self, attr_name, value, indices=None) -> None` | Set attribute inside vectorized environments (see base class). |  |
| `OrcaGymAsyncSubprocVecEnv.env_method` | `def env_method(self, method_name, *method_args, indices=None, **method_kwargs) -> List[Any]` | Call instance methods of vectorized environments. |  |
| `OrcaGymAsyncSubprocVecEnv.env_is_wrapped` | `def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]` | Check if worker environments are wrapped with a given wrapper |  |
| `OrcaGymAsyncSubprocVecEnv.setup_curriculum` | `def setup_curriculum(self, curriculum_name) -> None` | _No docstring._ |  |

##### 方法详情

###### `OrcaGymAsyncSubprocVecEnv.step_async`

Signature:

```python
def step_async(self, actions) -> None
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncSubprocVecEnv.step_wait`

Signature:

```python
def step_wait(self) -> VecEnvStepReturn
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncSubprocVecEnv.reset`

Signature:

```python
def reset(self) -> VecEnvObs
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncSubprocVecEnv.close`

Signature:

```python
def close(self) -> None
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncSubprocVecEnv.get_images`

Signature:

```python
def get_images(self) -> Sequence[Optional[np.ndarray]]
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymAsyncSubprocVecEnv.get_attr`

Signature:

```python
def get_attr(self, attr_name, indices=None) -> List[Any]
```

<details>
<summary>Docstring</summary>

Return attribute from vectorized environment (see base class).

</details>


###### `OrcaGymAsyncSubprocVecEnv.set_attr`

Signature:

```python
def set_attr(self, attr_name, value, indices=None) -> None
```

<details>
<summary>Docstring</summary>

Set attribute inside vectorized environments (see base class).

</details>


###### `OrcaGymAsyncSubprocVecEnv.env_method`

Signature:

```python
def env_method(self, method_name, *method_args, indices=None, **method_kwargs) -> List[Any]
```

<details>
<summary>Docstring</summary>

Call instance methods of vectorized environments.

</details>


###### `OrcaGymAsyncSubprocVecEnv.env_is_wrapped`

Signature:

```python
def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]
```

<details>
<summary>Docstring</summary>

Check if worker environments are wrapped with a given wrapper

</details>


###### `OrcaGymAsyncSubprocVecEnv.setup_curriculum`

Signature:

```python
def setup_curriculum(self, curriculum_name) -> None
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>

---

## `orca_gym/environment/orca_gym_env.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `RewardType` | _No docstring._ |
| `OrcaGymBaseEnv` | Superclass for all OrcaSim environments. |

### Classes（详情）

#### `class RewardType`

<details>
<summary>Class docstring</summary>

_No docstring._

</details>


#### `class OrcaGymBaseEnv`

> Superclass for all OrcaSim environments.

<details>
<summary>Class docstring</summary>

Superclass for all OrcaSim environments.

</details>


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

###### `OrcaGymBaseEnv.step`

Signature:

```python
def step(self, action) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymBaseEnv.reset_model`

Signature:

```python
def reset_model(self) -> tuple[dict, dict]
```

<details>
<summary>Docstring</summary>

Reset the robot degrees of freedom (qpos and qvel).
Implement this in each subclass.

</details>


###### `OrcaGymBaseEnv.initialize_simulation`

Signature:

```python
def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]
```

<details>
<summary>Docstring</summary>

Initialize MuJoCo simulation data structures mjModel and mjData.

</details>


###### `OrcaGymBaseEnv.render`

Signature:

```python
def render(self) -> Union[NDArray[np.float64], None]
```

<details>
<summary>Docstring</summary>

Render a frame from the MuJoCo simulation as specified by the render_mode.

</details>


###### `OrcaGymBaseEnv.generate_action_space`

Signature:

```python
def generate_action_space(self, bounds) -> Space
```

<details>
<summary>Docstring</summary>

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

</details>


###### `OrcaGymBaseEnv.generate_observation_space`

Signature:

```python
def generate_observation_space(self, obs) -> spaces.Space
```

<details>
<summary>Docstring</summary>

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

</details>


###### `OrcaGymBaseEnv.reset`

Signature:

```python
def reset(self, *, seed=None, options=None)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymBaseEnv.set_seed_value`

Signature:

```python
def set_seed_value(self, seed=None)
```

<details>
<summary>Docstring</summary>

设置随机数种子

</details>


###### `OrcaGymBaseEnv.body`

Signature:

```python
def body(self, name, agent_id=None) -> str
```

<details>
<summary>Docstring</summary>

获取带智能体前缀的 body 名称

</details>


###### `OrcaGymBaseEnv.joint`

Signature:

```python
def joint(self, name, agent_id=None) -> str
```

<details>
<summary>Docstring</summary>

获取带智能体前缀的关节名称

</details>


###### `OrcaGymBaseEnv.actuator`

Signature:

```python
def actuator(self, name, agent_id=None) -> str
```

<details>
<summary>Docstring</summary>

获取带智能体前缀的执行器名称

</details>


###### `OrcaGymBaseEnv.site`

Signature:

```python
def site(self, name, agent_id=None) -> str
```

<details>
<summary>Docstring</summary>

获取带智能体前缀的 site 名称

</details>


###### `OrcaGymBaseEnv.mocap`

Signature:

```python
def mocap(self, name, agent_id=None) -> str
```

<details>
<summary>Docstring</summary>

获取带智能体前缀的 mocap 名称

</details>


###### `OrcaGymBaseEnv.sensor`

Signature:

```python
def sensor(self, name, agent_id=None) -> str
```

<details>
<summary>Docstring</summary>

获取带智能体前缀的传感器名称

</details>


###### `OrcaGymBaseEnv.dt` (property)

Signature:

```python
def dt(self) -> float
```

<details>
<summary>Docstring</summary>

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

</details>


###### `OrcaGymBaseEnv.agent_num` (property)

Signature:

```python
def agent_num(self) -> int
```

<details>
<summary>Docstring</summary>

获取智能体数量

</details>


###### `OrcaGymBaseEnv.do_simulation`

Signature:

```python
def do_simulation(self, ctrl, n_frames) -> None
```

<details>
<summary>Docstring</summary>

Step the simulation n number of frames and applying a control action.

</details>


###### `OrcaGymBaseEnv.close`

Signature:

```python
def close(self)
```

<details>
<summary>Docstring</summary>

Close all processes like rendering contexts

</details>


###### `OrcaGymBaseEnv.initialize_grpc`

Signature:

```python
def initialize_grpc(self)
```

<details>
<summary>Docstring</summary>

Initialize the GRPC communication channel.

</details>


###### `OrcaGymBaseEnv.pause_simulation`

Signature:

```python
def pause_simulation(self)
```

<details>
<summary>Docstring</summary>

Pause the simulation.

</details>


###### `OrcaGymBaseEnv.init_qpos_qvel`

Signature:

```python
def init_qpos_qvel(self)
```

<details>
<summary>Docstring</summary>

Init qpos and qvel of the model.

</details>


###### `OrcaGymBaseEnv.reset_simulation`

Signature:

```python
def reset_simulation(self)
```

<details>
<summary>Docstring</summary>

Reset the simulation.

</details>


###### `OrcaGymBaseEnv.set_time_step`

Signature:

```python
def set_time_step(self, time_step)
```

<details>
<summary>Docstring</summary>

Set the time step of the simulation.

</details>

---

## `orca_gym/environment/orca_gym_local_env.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymLocalEnv` | _No docstring._ |

### Classes（详情）

#### `class OrcaGymLocalEnv`

<details>
<summary>Class docstring</summary>

_No docstring._

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymLocalEnv.initialize_simulation` | `def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]` | 初始化仿真，加载模型并创建模型和数据对象 |  |
| `OrcaGymLocalEnv.initialize_grpc` | `def initialize_grpc(self)` | 初始化 gRPC 通信通道和客户端 |  |
| `OrcaGymLocalEnv.pause_simulation` | `def pause_simulation(self)` | 暂停仿真（采用被动模式） |  |
| `OrcaGymLocalEnv.close` | `def close(self)` | 关闭环境，清理资源 |  |
| `OrcaGymLocalEnv.get_body_manipulation_anchored` | `def get_body_manipulation_anchored(self)` | _No docstring._ |  |
| `OrcaGymLocalEnv.begin_save_video` | `def begin_save_video(self, file_path, capture_mode=CaptureMode.ASYNC)` | _No docstring._ |  |
| `OrcaGymLocalEnv.stop_save_video` | `def stop_save_video(self)` | _No docstring._ |  |
| `OrcaGymLocalEnv.get_next_frame` | `def get_next_frame(self) -> int` | _No docstring._ |  |
| `OrcaGymLocalEnv.get_current_frame` | `def get_current_frame(self) -> int` | _No docstring._ |  |
| `OrcaGymLocalEnv.get_camera_time_stamp` | `def get_camera_time_stamp(self, last_frame_index) -> dict` | _No docstring._ |  |
| `OrcaGymLocalEnv.get_frame_png` | `def get_frame_png(self, image_path)` | _No docstring._ |  |
| `OrcaGymLocalEnv.get_body_manipulation_movement` | `def get_body_manipulation_movement(self)` | _No docstring._ |  |
| `OrcaGymLocalEnv.do_simulation` | `def do_simulation(self, ctrl, n_frames) -> None` | 执行仿真步进：设置控制并步进 n_frames 次，然后同步数据 |  |
| `OrcaGymLocalEnv.render_mode` | `def render_mode(self) -> str` | _No docstring._ | `@property` |
| `OrcaGymLocalEnv.is_subenv` | `def is_subenv(self) -> bool` | _No docstring._ | `@property` |
| `OrcaGymLocalEnv.sync_render` | `def sync_render(self) -> bool` | _No docstring._ | `@property` |
| `OrcaGymLocalEnv.render` | `def render(self)` | _No docstring._ |  |
| `OrcaGymLocalEnv.do_body_manipulation` | `def do_body_manipulation(self)` | _No docstring._ |  |
| `OrcaGymLocalEnv.release_body_anchored` | `def release_body_anchored(self)` | _No docstring._ |  |
| `OrcaGymLocalEnv.anchor_actor` | `def anchor_actor(self, actor_name, anchor_type)` | _No docstring._ |  |
| `OrcaGymLocalEnv.update_anchor_equality_constraints` | `def update_anchor_equality_constraints(self, actor_name, anchor_type)` | _No docstring._ |  |
| `OrcaGymLocalEnv.set_ctrl` | `def set_ctrl(self, ctrl)` | 设置控制输入（执行器命令） |  |
| `OrcaGymLocalEnv.mj_step` | `def mj_step(self, nstep)` | 执行 MuJoCo 仿真步进 |  |
| `OrcaGymLocalEnv.mj_forward` | `def mj_forward(self)` | 执行 MuJoCo 前向计算（更新动力学状态） |  |
| `OrcaGymLocalEnv.mj_jacBody` | `def mj_jacBody(self, jacp, jacr, body_id)` | 计算 body 的雅可比矩阵（位置和旋转） |  |
| `OrcaGymLocalEnv.mj_jacSite` | `def mj_jacSite(self, jacp, jacr, site_name)` | 计算 site 的雅可比矩阵（位置和旋转） |  |
| `OrcaGymLocalEnv.set_time_step` | `def set_time_step(self, time_step)` | 设置仿真时间步长 |  |
| `OrcaGymLocalEnv.update_data` | `def update_data(self)` | 从服务器同步最新的仿真数据 |  |
| `OrcaGymLocalEnv.reset_simulation` | `def reset_simulation(self)` | 重置仿真到初始状态 |  |
| `OrcaGymLocalEnv.init_qpos_qvel` | `def init_qpos_qvel(self)` | 初始化并保存初始关节位置和速度 |  |
| `OrcaGymLocalEnv.query_joint_offsets` | `def query_joint_offsets(self, joint_names) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]` | 查询关节在状态数组中的偏移量（位置、速度、加速度） |  |
| `OrcaGymLocalEnv.query_joint_lengths` | `def query_joint_lengths(self, joint_names)` | 查询关节状态数组的长度（位置、速度、加速度维度） |  |
| `OrcaGymLocalEnv.get_body_xpos_xmat_xquat` | `def get_body_xpos_xmat_xquat(self, body_name_list)` | 获取 body 的位姿（位置、旋转矩阵、四元数） |  |
| `OrcaGymLocalEnv.query_sensor_data` | `def query_sensor_data(self, sensor_names)` | 查询传感器数据 |  |
| `OrcaGymLocalEnv.query_joint_qpos` | `def query_joint_qpos(self, joint_names)` | 查询关节位置 |  |
| `OrcaGymLocalEnv.query_joint_qvel` | `def query_joint_qvel(self, joint_names)` | 查询关节速度 |  |
| `OrcaGymLocalEnv.query_joint_qacc` | `def query_joint_qacc(self, joint_names)` | 查询关节加速度 |  |
| `OrcaGymLocalEnv.jnt_qposadr` | `def jnt_qposadr(self, joint_name)` | 获取关节在 qpos 数组中的起始地址 |  |
| `OrcaGymLocalEnv.jnt_dofadr` | `def jnt_dofadr(self, joint_name)` | 获取关节在 qvel 数组中的起始地址 |  |
| `OrcaGymLocalEnv.query_site_pos_and_mat` | `def query_site_pos_and_mat(self, site_names)` | 查询 site 的位置和旋转矩阵 |  |
| `OrcaGymLocalEnv.query_site_pos_and_quat` | `def query_site_pos_and_quat(self, site_names) -> Dict[str, Dict[str, Union[NDArray[np.float64], NDArray[np.float64]]]]` | 查询 site 的位置和四元数（从旋转矩阵转换） |  |
| `OrcaGymLocalEnv.query_site_size` | `def query_site_size(self, site_names)` | 查询 site 的尺寸 |  |
| `OrcaGymLocalEnv.query_site_pos_and_quat_B` | `def query_site_pos_and_quat_B(self, site_names, base_body_list) -> Dict[str, Dict[str, Union[NDArray[np.float32], NDArray[np.float32]]]]` | 查询 site 相对于基座 body 的位置和四元数（基座坐标系） |  |
| `OrcaGymLocalEnv.set_joint_qpos` | `def set_joint_qpos(self, joint_qpos)` | 设置关节位置 |  |
| `OrcaGymLocalEnv.set_joint_qvel` | `def set_joint_qvel(self, joint_qvel)` | 设置关节速度 |  |
| `OrcaGymLocalEnv.query_site_xvalp_xvalr` | `def query_site_xvalp_xvalr(self, site_names) -> Tuple[Dict[str, NDArray[np.float64]], Dict[str, NDArray[np.float64]]]` | 查询 site 的线速度和角速度（世界坐标系） |  |
| `OrcaGymLocalEnv.query_site_xvalp_xvalr_B` | `def query_site_xvalp_xvalr_B(self, site_names, base_body_list) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, NDArray[np.float32]]]` | 查询 site 相对于基座 body 的线速度和角速度（基座坐标系） |  |
| `OrcaGymLocalEnv.update_equality_constraints` | `def update_equality_constraints(self, eq_list)` | 更新等式约束列表 |  |
| `OrcaGymLocalEnv.set_mocap_pos_and_quat` | `def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict)` | 设置 mocap body 的位置和四元数（用于物体操作） |  |
| `OrcaGymLocalEnv.query_contact_simple` | `def query_contact_simple(self)` | 查询简单接触信息 |  |
| `OrcaGymLocalEnv.set_geom_friction` | `def set_geom_friction(self, geom_friction_dict)` | 设置几何体的摩擦系数 |  |
| `OrcaGymLocalEnv.add_extra_weight` | `def add_extra_weight(self, weight_load_dict)` | 为 body 添加额外重量 |  |
| `OrcaGymLocalEnv.query_contact_force` | `def query_contact_force(self, contact_ids)` | 查询接触力 |  |
| `OrcaGymLocalEnv.get_cfrc_ext` | `def get_cfrc_ext(self)` | 获取外部约束力 |  |
| `OrcaGymLocalEnv.query_actuator_torques` | `def query_actuator_torques(self, actuator_names)` | 查询执行器扭矩 |  |
| `OrcaGymLocalEnv.query_joint_dofadrs` | `def query_joint_dofadrs(self, joint_names)` | _No docstring._ |  |
| `OrcaGymLocalEnv.get_goal_bounding_box` | `def get_goal_bounding_box(self, geom_name)` | _No docstring._ |  |
| `OrcaGymLocalEnv.query_velocity_body_B` | `def query_velocity_body_B(self, ee_body, base_body)` | _No docstring._ |  |
| `OrcaGymLocalEnv.query_position_body_B` | `def query_position_body_B(self, ee_body, base_body)` | _No docstring._ |  |
| `OrcaGymLocalEnv.query_orientation_body_B` | `def query_orientation_body_B(self, ee_body, base_body)` | _No docstring._ |  |
| `OrcaGymLocalEnv.query_joint_axes_B` | `def query_joint_axes_B(self, joint_names, base_body)` | _No docstring._ |  |
| `OrcaGymLocalEnv.query_robot_velocity_odom` | `def query_robot_velocity_odom(self, base_body, initial_base_pos, initial_base_quat)` | _No docstring._ |  |
| `OrcaGymLocalEnv.query_robot_position_odom` | `def query_robot_position_odom(self, base_body, initial_base_pos, initial_base_quat)` | _No docstring._ |  |
| `OrcaGymLocalEnv.query_robot_orientation_odom` | `def query_robot_orientation_odom(self, base_body, initial_base_pos, initial_base_quat)` | _No docstring._ |  |
| `OrcaGymLocalEnv.set_actuator_trnid` | `def set_actuator_trnid(self, actuator_id, trnid)` | _No docstring._ |  |
| `OrcaGymLocalEnv.disable_actuator` | `def disable_actuator(self, actuator_groups)` | _No docstring._ |  |
| `OrcaGymLocalEnv.load_content_file` | `def load_content_file(self, content_file_name, remote_file_dir='', local_file_dir='', temp_file_path=None)` | _No docstring._ |  |

##### 方法详情

###### `OrcaGymLocalEnv.initialize_simulation`

Signature:

```python
def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]
```

<details>
<summary>Docstring</summary>

初始化仿真，加载模型并创建模型和数据对象

</details>


###### `OrcaGymLocalEnv.initialize_grpc`

Signature:

```python
def initialize_grpc(self)
```

<details>
<summary>Docstring</summary>

初始化 gRPC 通信通道和客户端

</details>


###### `OrcaGymLocalEnv.pause_simulation`

Signature:

```python
def pause_simulation(self)
```

<details>
<summary>Docstring</summary>

暂停仿真（采用被动模式）

</details>


###### `OrcaGymLocalEnv.close`

Signature:

```python
def close(self)
```

<details>
<summary>Docstring</summary>

关闭环境，清理资源

</details>


###### `OrcaGymLocalEnv.get_body_manipulation_anchored`

Signature:

```python
def get_body_manipulation_anchored(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.begin_save_video`

Signature:

```python
def begin_save_video(self, file_path, capture_mode=CaptureMode.ASYNC)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.stop_save_video`

Signature:

```python
def stop_save_video(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.get_next_frame`

Signature:

```python
def get_next_frame(self) -> int
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.get_current_frame`

Signature:

```python
def get_current_frame(self) -> int
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.get_camera_time_stamp`

Signature:

```python
def get_camera_time_stamp(self, last_frame_index) -> dict
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.get_frame_png`

Signature:

```python
def get_frame_png(self, image_path)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.get_body_manipulation_movement`

Signature:

```python
def get_body_manipulation_movement(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.do_simulation`

Signature:

```python
def do_simulation(self, ctrl, n_frames) -> None
```

<details>
<summary>Docstring</summary>

执行仿真步进：设置控制并步进 n_frames 次，然后同步数据

这是环境 step() 函数的核心方法，执行一次完整的仿真步进。
包括：设置控制输入、执行物理步进、同步最新状态。

Args:
    ctrl: 控制输入数组，形状 (nu,)，nu 为执行器数量
    n_frames: 步进次数，通常等于 frame_skip

使用示例:
    ```python
    # 在 step 函数中执行仿真
    for _ in range(self._action_skip):
        # 计算扭矩控制
        torque_ctrl = agent.compute_torques(self.data.qpos, self.data.qvel)
        self.set_ctrl(torque_ctrl)
        # 执行仿真步进
        self.do_simulation(self.ctrl, self.frame_skip)
    ```

使用示例:
    ```python
    # 在 step 中执行多次物理步进（decimation）
    for _ in range(self.decimation):
        self.do_simulation(self.ctrl, 1)  # 每次步进1个物理步
    ```

</details>


###### `OrcaGymLocalEnv.render_mode` (property)

Signature:

```python
def render_mode(self) -> str
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.is_subenv` (property)

Signature:

```python
def is_subenv(self) -> bool
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.sync_render` (property)

Signature:

```python
def sync_render(self) -> bool
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.render`

Signature:

```python
def render(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.do_body_manipulation`

Signature:

```python
def do_body_manipulation(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.release_body_anchored`

Signature:

```python
def release_body_anchored(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.anchor_actor`

Signature:

```python
def anchor_actor(self, actor_name, anchor_type)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.update_anchor_equality_constraints`

Signature:

```python
def update_anchor_equality_constraints(self, actor_name, anchor_type)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.set_ctrl`

Signature:

```python
def set_ctrl(self, ctrl)
```

<details>
<summary>Docstring</summary>

设置控制输入（执行器命令）

设置所有执行器的控制值，形状必须为 (nu,)，其中 nu 是执行器数量。
通常在调用 mj_step() 之前设置。

使用示例:
    ```python
    # 在重置时清零控制
    self.ctrl = np.zeros(self.nu)
    self.set_ctrl(self.ctrl)
    self.mj_forward()
    ```

使用示例:
    ```python
    # 准备控制数组
    self.ctrl = np.zeros(self.nu, dtype=np.float32)
    # ... 计算控制值 ...
    self.set_ctrl(self.ctrl)
    ```

</details>


###### `OrcaGymLocalEnv.mj_step`

Signature:

```python
def mj_step(self, nstep)
```

<details>
<summary>Docstring</summary>

执行 MuJoCo 仿真步进

执行 nstep 次物理仿真步进，每次步进的时间为 timestep。
在调用前需要先设置控制输入 (set_ctrl)。

使用示例:
    ```python
    # 在 step 函数中执行仿真
    self.set_ctrl(self.ctrl)
    self.do_simulation(self.ctrl, self.frame_skip)  # 内部调用 mj_step
    ```

</details>


###### `OrcaGymLocalEnv.mj_forward`

Signature:

```python
def mj_forward(self)
```

<details>
<summary>Docstring</summary>

执行 MuJoCo 前向计算（更新动力学状态）

更新所有动力学相关状态，包括位置、速度、加速度、力等。
在设置关节状态、mocap 位置等操作后需要调用，确保状态一致。

使用示例:
    ```python
    # 在初始化时调用，避免 NaN 错误
    self.mj_forward()
    
    # 在设置初始状态后调用
    self.set_ctrl(self.ctrl)
    self.mj_forward()
    
    # 在重置后调用
    self.mj_forward()
    ```

</details>


###### `OrcaGymLocalEnv.mj_jacBody`

Signature:

```python
def mj_jacBody(self, jacp, jacr, body_id)
```

<details>
<summary>Docstring</summary>

计算 body 的雅可比矩阵（位置和旋转）

术语说明:
    - 雅可比矩阵 (Jacobian Matrix): 描述关节速度到 body 速度的线性映射关系
    - jacp: 位置雅可比，形状 (3, nv)，将关节速度映射到 body 的线性速度
    - jacr: 旋转雅可比，形状 (3, nv)，将关节速度映射到 body 的角速度
    - 用途: 用于逆运动学、速度控制、力控制等算法

使用示例:
    ```python
    # 计算末端执行器的雅可比矩阵
    jacp = np.zeros((3, self.model.nv))
    jacr = np.zeros((3, self.model.nv))
    body_id = self.model.body_name2id("end_effector")
    self.mj_jacBody(jacp, jacr, body_id)
    # 计算末端执行器速度: v = jacp @ qvel, omega = jacr @ qvel
    ```

</details>


###### `OrcaGymLocalEnv.mj_jacSite`

Signature:

```python
def mj_jacSite(self, jacp, jacr, site_name)
```

<details>
<summary>Docstring</summary>

计算 site 的雅可比矩阵（位置和旋转）

术语说明:
    - 雅可比矩阵: 详见 mj_jacBody 的说明
    - Site: 标记点，详见 init_site_dict 的说明

使用示例:
    ```python
    # 计算 site 的雅可比矩阵用于速度计算
    query_dict = self.gym.mj_jac_site(["end_effector"])
    jacp = query_dict["end_effector"]["jacp"]  # (3, nv)
    jacr = query_dict["end_effector"]["jacr"]  # (3, nv)
    # 计算 site 速度: v = jacp @ self.data.qvel
    ```

</details>


###### `OrcaGymLocalEnv.set_time_step`

Signature:

```python
def set_time_step(self, time_step)
```

<details>
<summary>Docstring</summary>

设置仿真时间步长

</details>


###### `OrcaGymLocalEnv.update_data`

Signature:

```python
def update_data(self)
```

<details>
<summary>Docstring</summary>

从服务器同步最新的仿真数据

从 OrcaSim 服务器获取最新的 qpos、qvel、qacc 等状态数据，
更新到本地的 self.data 中。在每次仿真步进后自动调用。

使用示例:
    ```python
    # 在 do_simulation 中自动调用
    self._step_orca_sim_simulation(ctrl, n_frames)
    self.gym.update_data()  # 同步最新状态
    
    # 之后可以安全访问 self.data.qpos, self.data.qvel 等
    current_qpos = self.data.qpos.copy()
    ```

</details>


###### `OrcaGymLocalEnv.reset_simulation`

Signature:

```python
def reset_simulation(self)
```

<details>
<summary>Docstring</summary>

重置仿真到初始状态

加载初始帧，同步数据，并重新设置时间步长。
在环境 reset() 时调用，将仿真恢复到初始状态。

使用示例:
    ```python
    # 在 reset 函数中调用
    def reset(self, seed=None, options=None):
        self.reset_simulation()  # 重置到初始状态
        obs, info = self.reset_model()  # 重置模型特定状态
        return obs, info
    ```

</details>


###### `OrcaGymLocalEnv.init_qpos_qvel`

Signature:

```python
def init_qpos_qvel(self)
```

<details>
<summary>Docstring</summary>

初始化并保存初始关节位置和速度

在环境初始化时调用，保存初始状态用于后续重置。
保存的值可以通过 self.init_qpos 和 self.init_qvel 访问。

使用示例:
    ```python
    # 在 __init__ 中调用
    self.model, self.data = self.initialize_simulation()
    self.reset_simulation()
    self.init_qpos_qvel()  # 保存初始状态
    
    # 在 reset_model 中使用
    self.data.qpos[:] = self.init_qpos  # 恢复到初始位置
    self.data.qvel[:] = self.init_qvel  # 恢复到初始速度
    ```

</details>


###### `OrcaGymLocalEnv.query_joint_offsets`

Signature:

```python
def query_joint_offsets(self, joint_names) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

<details>
<summary>Docstring</summary>

查询关节在状态数组中的偏移量（位置、速度、加速度）

</details>


###### `OrcaGymLocalEnv.query_joint_lengths`

Signature:

```python
def query_joint_lengths(self, joint_names)
```

<details>
<summary>Docstring</summary>

查询关节状态数组的长度（位置、速度、加速度维度）

</details>


###### `OrcaGymLocalEnv.get_body_xpos_xmat_xquat`

Signature:

```python
def get_body_xpos_xmat_xquat(self, body_name_list)
```

<details>
<summary>Docstring</summary>

获取 body 的位姿（位置、旋转矩阵、四元数）

返回指定 body 在世界坐标系中的位置、旋转矩阵和四元数。
这是最常用的位姿查询方法，用于获取机器人基座、末端执行器等关键 body 的位姿。

术语说明:
    - 位姿 (Pose): 物体的位置和姿态的组合
    - 位置 (Position): 物体在空间中的坐标 [x, y, z]
    - 旋转矩阵 (Rotation Matrix): 3x3 矩阵，描述物体的旋转姿态
    - 四元数 (Quaternion): [w, x, y, z] 格式，用于表示旋转，避免万向锁问题
    - 世界坐标系: 固定的全局坐标系，所有位置和姿态都相对于此坐标系

Args:
    body_name_list: body 名称列表，如 ["base_link", "end_effector"]

Returns:
    xpos: 位置数组，形状 (len(body_name_list)*3,)，每3个元素为一个 body 的 [x, y, z]
    xmat: 旋转矩阵数组，形状 (len(body_name_list)*9,)，每9个元素为一个 body 的 3x3 矩阵（按行展开）
    xquat: 四元数数组，形状 (len(body_name_list)*4,)，每4个元素为一个 body 的 [w, x, y, z]

使用示例:
    ```python
    # 获取相机 body 的位姿
    camera_pos, _, camera_quat = self.get_body_xpos_xmat_xquat([self._camera_body_name])
    # camera_pos: [x, y, z]
    # camera_quat: [w, x, y, z]
    ```

使用示例:
    ```python
    # 获取基座位置用于计算高度
    base_pos, _, _ = self.get_body_xpos_xmat_xquat([self.base_body_name])
    real_base_z = float(base_pos[2])  # z 坐标
    ```

使用示例:
    ```python
    # 获取锚点 body 的位姿用于物体操作
    anchor_xpos, anchor_xmat, anchor_xquat = self.get_body_xpos_xmat_xquat([self._anchor_body_name])
    ```

</details>


###### `OrcaGymLocalEnv.query_sensor_data`

Signature:

```python
def query_sensor_data(self, sensor_names)
```

<details>
<summary>Docstring</summary>

查询传感器数据

</details>


###### `OrcaGymLocalEnv.query_joint_qpos`

Signature:

```python
def query_joint_qpos(self, joint_names)
```

<details>
<summary>Docstring</summary>

查询关节位置

返回指定关节的当前位置，字典格式，键为关节名称，值为位置值或数组。

使用示例:
    ```python
    # 查询特定关节位置
    joint_pos = self.query_joint_qpos(["joint1", "joint2", "joint3"])
    # 返回: {"joint1": value1, "joint2": value2, "joint3": value3}
    
    # 用于构建观测空间
    obs["joint_pos"] = np.array([joint_pos[name] for name in joint_names])
    ```

</details>


###### `OrcaGymLocalEnv.query_joint_qvel`

Signature:

```python
def query_joint_qvel(self, joint_names)
```

<details>
<summary>Docstring</summary>

查询关节速度

返回指定关节的当前速度，字典格式，键为关节名称，值为速度值或数组。

使用示例:
    ```python
    # 查询关节速度用于观测
    joint_vel = self.query_joint_qvel(["joint1", "joint2"])
    # 返回: {"joint1": vel1, "joint2": vel2}
    
    # 用于计算奖励（速度惩罚）
    vel_penalty = sum(abs(v) for v in joint_vel.values())
    ```

</details>


###### `OrcaGymLocalEnv.query_joint_qacc`

Signature:

```python
def query_joint_qacc(self, joint_names)
```

<details>
<summary>Docstring</summary>

查询关节加速度

返回指定关节的当前加速度，字典格式，键为关节名称，值为加速度值或数组。

使用示例:
    ```python
    # 查询关节加速度
    joint_acc = self.query_joint_qacc(["joint1", "joint2"])
    # 用于分析运动状态或计算动力学
    ```

</details>


###### `OrcaGymLocalEnv.jnt_qposadr`

Signature:

```python
def jnt_qposadr(self, joint_name)
```

<details>
<summary>Docstring</summary>

获取关节在 qpos 数组中的起始地址

返回关节在全局 qpos 数组中的起始索引，用于访问特定关节的位置数据。
不同关节类型占用的位置数量不同（旋转关节1个，自由关节7个等）。

使用示例:
    ```python
    # 获取关节在 qpos 中的地址
    joint_addr = self.jnt_qposadr("joint1")
    joint_nq = self.model.get_joint_byname("joint1")["JointNq"]
    # 提取该关节的位置
    joint_qpos = self.data.qpos[joint_addr:joint_addr+joint_nq]
    ```

</details>


###### `OrcaGymLocalEnv.jnt_dofadr`

Signature:

```python
def jnt_dofadr(self, joint_name)
```

<details>
<summary>Docstring</summary>

获取关节在 qvel 数组中的起始地址

返回关节在全局 qvel 数组中的起始索引，用于访问特定关节的速度数据。
通常等于自由度数量（旋转关节1个，自由关节6个等）。

使用示例:
    ```python
    # 获取关节在 qvel 中的地址
    joint_dofadr = self.jnt_dofadr("joint1")
    joint_nv = self.model.get_joint_byname("joint1")["JointNv"]
    # 提取该关节的速度
    joint_qvel = self.data.qvel[joint_dofadr:joint_dofadr+joint_nv]
    ```

</details>


###### `OrcaGymLocalEnv.query_site_pos_and_mat`

Signature:

```python
def query_site_pos_and_mat(self, site_names)
```

<details>
<summary>Docstring</summary>

查询 site 的位置和旋转矩阵

</details>


###### `OrcaGymLocalEnv.query_site_pos_and_quat`

Signature:

```python
def query_site_pos_and_quat(self, site_names) -> Dict[str, Dict[str, Union[NDArray[np.float64], NDArray[np.float64]]]]
```

<details>
<summary>Docstring</summary>

查询 site 的位置和四元数（从旋转矩阵转换）

返回指定 site 在世界坐标系中的位置和四元数。
Site 通常用于标记末端执行器、目标点等关键位置。

Args:
    site_names: site 名称列表，如 ["end_effector", "target"]

Returns:
    字典，键为 site 名称，值为包含 'xpos' 和 'xquat' 的字典
    - xpos: 位置数组 [x, y, z]
    - xquat: 四元数 [w, x, y, z]

使用示例:
    ```python
    # 查询末端执行器位姿
    ee_site = self.query_site_pos_and_quat(["end_effector"])
    ee_pos = ee_site["end_effector"]["xpos"]  # [x, y, z]
    ee_quat = ee_site["end_effector"]["xquat"]  # [w, x, y, z]
    
    # 用于计算到目标的距离
    target_site = self.query_site_pos_and_quat(["target"])
    distance = np.linalg.norm(ee_pos - target_site["target"]["xpos"])
    ```

</details>


###### `OrcaGymLocalEnv.query_site_size`

Signature:

```python
def query_site_size(self, site_names)
```

<details>
<summary>Docstring</summary>

查询 site 的尺寸

</details>


###### `OrcaGymLocalEnv.query_site_pos_and_quat_B`

Signature:

```python
def query_site_pos_and_quat_B(self, site_names, base_body_list) -> Dict[str, Dict[str, Union[NDArray[np.float32], NDArray[np.float32]]]]
```

<details>
<summary>Docstring</summary>

查询 site 相对于基座 body 的位置和四元数（基座坐标系）

术语说明:
    - 基座坐标系 (Base Frame): 以机器人基座为原点的局部坐标系
    - 相对位姿: 相对于基座的位置和姿态，而不是世界坐标系
    - 用途: 在机器人控制中，通常需要知道末端执行器相对于基座的位置

使用示例:
    ```python
    # 查询末端执行器相对于基座的位置
    ee_pos_B, ee_quat_B = self.query_site_pos_and_quat_B(
        ["end_effector"], 
        ["base_link"]
    )
    # 返回的是相对于基座的位置，用于逆运动学计算
    ```

</details>


###### `OrcaGymLocalEnv.set_joint_qpos`

Signature:

```python
def set_joint_qpos(self, joint_qpos)
```

<details>
<summary>Docstring</summary>

设置关节位置

直接设置关节位置，用于重置或初始化机器人姿态。
设置后需要调用 mj_forward() 更新动力学状态。

使用示例:
    ```python
    # 在重置时设置初始关节位置
    initial_qpos = np.array([0.0, 0.5, -1.0, ...])  # 初始姿态
    self.set_joint_qpos(initial_qpos)
    self.mj_forward()  # 更新状态
    ```

</details>


###### `OrcaGymLocalEnv.set_joint_qvel`

Signature:

```python
def set_joint_qvel(self, joint_qvel)
```

<details>
<summary>Docstring</summary>

设置关节速度

直接设置关节速度，用于重置或初始化机器人运动状态。
设置后需要调用 mj_forward() 更新动力学状态。

使用示例:
    ```python
    # 在重置时清零速度
    initial_qvel = np.zeros(self.model.nv)
    self.set_joint_qvel(initial_qvel)
    self.mj_forward()  # 更新状态
    ```

</details>


###### `OrcaGymLocalEnv.query_site_xvalp_xvalr`

Signature:

```python
def query_site_xvalp_xvalr(self, site_names) -> Tuple[Dict[str, NDArray[np.float64]], Dict[str, NDArray[np.float64]]]
```

<details>
<summary>Docstring</summary>

查询 site 的线速度和角速度（世界坐标系）

</details>


###### `OrcaGymLocalEnv.query_site_xvalp_xvalr_B`

Signature:

```python
def query_site_xvalp_xvalr_B(self, site_names, base_body_list) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, NDArray[np.float32]]]
```

<details>
<summary>Docstring</summary>

查询 site 相对于基座 body 的线速度和角速度（基座坐标系）

术语说明:
    - 线速度 (Linear Velocity): 物体在空间中的移动速度 [vx, vy, vz]
    - 角速度 (Angular Velocity): 物体绕轴旋转的速度 [wx, wy, wz]
    - 基座坐标系: 详见 query_site_pos_and_quat_B 的说明

使用示例:
    ```python
    # 查询末端执行器相对于基座的速度
    linear_vel_B, angular_vel_B = self.query_site_xvalp_xvalr_B(
        ["end_effector"],
        ["base_link"]
    )
    # 用于速度控制或计算速度误差
    ```

</details>


###### `OrcaGymLocalEnv.update_equality_constraints`

Signature:

```python
def update_equality_constraints(self, eq_list)
```

<details>
<summary>Docstring</summary>

更新等式约束列表

</details>


###### `OrcaGymLocalEnv.set_mocap_pos_and_quat`

Signature:

```python
def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict)
```

<details>
<summary>Docstring</summary>

设置 mocap body 的位置和四元数（用于物体操作）

Mocap body 是用于物体操作的虚拟 body，通过设置其位姿可以控制被锚定的物体。
常用于实现抓取、拖拽等操作。

Args:
    mocap_pos_and_quat_dict: 字典，键为 mocap body 名称，值为包含 'pos' 和 'quat' 的字典
        - pos: 位置数组 [x, y, z]
        - quat: 四元数 [w, x, y, z]

使用示例:
    ```python
    # 设置锚点位置用于物体操作
    self.set_mocap_pos_and_quat({
        self._anchor_body_name: {
            "pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
            "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        }
    })
    self.mj_forward()  # 更新状态
    
    # 释放物体时移动到远处
    self.set_mocap_pos_and_quat({
        self._anchor_body_name: {
            "pos": np.array([0.0, 0.0, -1000.0], dtype=np.float64),
            "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        }
    })
    ```

</details>


###### `OrcaGymLocalEnv.query_contact_simple`

Signature:

```python
def query_contact_simple(self)
```

<details>
<summary>Docstring</summary>

查询简单接触信息

</details>


###### `OrcaGymLocalEnv.set_geom_friction`

Signature:

```python
def set_geom_friction(self, geom_friction_dict)
```

<details>
<summary>Docstring</summary>

设置几何体的摩擦系数

</details>


###### `OrcaGymLocalEnv.add_extra_weight`

Signature:

```python
def add_extra_weight(self, weight_load_dict)
```

<details>
<summary>Docstring</summary>

为 body 添加额外重量

</details>


###### `OrcaGymLocalEnv.query_contact_force`

Signature:

```python
def query_contact_force(self, contact_ids)
```

<details>
<summary>Docstring</summary>

查询接触力

</details>


###### `OrcaGymLocalEnv.get_cfrc_ext`

Signature:

```python
def get_cfrc_ext(self)
```

<details>
<summary>Docstring</summary>

获取外部约束力

</details>


###### `OrcaGymLocalEnv.query_actuator_torques`

Signature:

```python
def query_actuator_torques(self, actuator_names)
```

<details>
<summary>Docstring</summary>

查询执行器扭矩

</details>


###### `OrcaGymLocalEnv.query_joint_dofadrs`

Signature:

```python
def query_joint_dofadrs(self, joint_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.get_goal_bounding_box`

Signature:

```python
def get_goal_bounding_box(self, geom_name)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.query_velocity_body_B`

Signature:

```python
def query_velocity_body_B(self, ee_body, base_body)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.query_position_body_B`

Signature:

```python
def query_position_body_B(self, ee_body, base_body)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.query_orientation_body_B`

Signature:

```python
def query_orientation_body_B(self, ee_body, base_body)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.query_joint_axes_B`

Signature:

```python
def query_joint_axes_B(self, joint_names, base_body)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.query_robot_velocity_odom`

Signature:

```python
def query_robot_velocity_odom(self, base_body, initial_base_pos, initial_base_quat)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.query_robot_position_odom`

Signature:

```python
def query_robot_position_odom(self, base_body, initial_base_pos, initial_base_quat)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.query_robot_orientation_odom`

Signature:

```python
def query_robot_orientation_odom(self, base_body, initial_base_pos, initial_base_quat)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.set_actuator_trnid`

Signature:

```python
def set_actuator_trnid(self, actuator_id, trnid)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.disable_actuator`

Signature:

```python
def disable_actuator(self, actuator_groups)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocalEnv.load_content_file`

Signature:

```python
def load_content_file(self, content_file_name, remote_file_dir='', local_file_dir='', temp_file_path=None)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>
