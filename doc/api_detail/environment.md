# API Detailed Reference: `orca_gym/environment`

本页为自动生成的 API 参考（无需导入模块）。

生成规则：

- 仅包含 **public** 符号（不以下划线开头）

- 每个条目包含：**签名（AST 提取）+ docstring（原样收录）**

- 为提升可读性，补充 **中文概览/中文提示**（不改源码 docstring）


---

## `orca_gym/environment/async_env/orca_gym_async_agent.py`

中文概览：环境层：Gymnasium Env 及其本地/异步/向量化实现。

### Classes

<details>
<summary>class OrcaGymAsyncAgent</summary>


_No class docstring._

#### Methods

##### `OrcaGymAsyncAgent.dt`

Signature:

```python
def dt(self) -> float
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.name`

Signature:

```python
def name(self) -> str
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.name_space`

Signature:

```python
def name_space(self, name) -> str
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.name_space_list`

Signature:

```python
def name_space_list(self, names) -> list[str]
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.joint_names`

Signature:

```python
def joint_names(self) -> list[str]
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.actuator_names`

Signature:

```python
def actuator_names(self) -> list[str]
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.site_names`

Signature:

```python
def site_names(self) -> list[str]
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.sensor_names`

Signature:

```python
def sensor_names(self) -> list[str]
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.nu`

Signature:

```python
def nu(self) -> int
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.nq`

Signature:

```python
def nq(self) -> int
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.nv`

Signature:

```python
def nv(self) -> int
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.truncated`

Signature:

```python
def truncated(self) -> bool
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.ctrl_start`

Signature:

```python
def ctrl_start(self) -> int
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.action_range`

Signature:

```python
def action_range(self) -> np.ndarray
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.kps`

Signature:

```python
def kps(self) -> float
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.kds`

Signature:

```python
def kds(self) -> float
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.get_obs`

Signature:

```python
def get_obs(self, **kwargs)
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.init_ctrl_info`

Signature:

```python
def init_ctrl_info(self, actuator_dict, joint_dict) -> None
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.get_ctrl_info`

Signature:

```python
def get_ctrl_info(self) -> dict
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.init_joint_index`

Signature:

```python
def init_joint_index(self, qpos_offset, qvel_offset, qacc_offset, qpos_length, qvel_length, qacc_length)
```

Docstring:

Joint index is specific to the agent and is defined in the subclass.

##### `OrcaGymAsyncAgent.set_action_space`

Signature:

```python
def set_action_space(self) -> None
```

Docstring:

Action space is specific to the agent and is defined in the subclass.

##### `OrcaGymAsyncAgent.on_step`

Signature:

```python
def on_step(self, action, **kwargs)
```

Docstring:

Called after each step in the environment.
Implement this method in the subclass to perform additional operations.

##### `OrcaGymAsyncAgent.step`

Signature:

```python
def step(self, action, **kwargs)
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.on_reset`

Signature:

```python
def on_reset(self, **kwargs)
```

Docstring:

Called after each reset in the environment.
Implement this method in the subclass to perform additional operations.

##### `OrcaGymAsyncAgent.reset`

Signature:

```python
def reset(self, np_random, **kwargs)
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.is_success`

Signature:

```python
def is_success(self, achieved_goal, desired_goal) -> np.float32
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.is_terminated`

Signature:

```python
def is_terminated(self, achieved_goal, desired_goal) -> bool
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.compute_reward`

Signature:

```python
def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.set_init_state`

Signature:

```python
def set_init_state(self, joint_qpos, init_site_pos_quat) -> None
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.get_action_size`

Signature:

```python
def get_action_size(self) -> int
```

Docstring:

Action size can be overridden in the subclass.
In most of cases, this is the number of actuators in the robot.
But in some cases, the action size may be different.

##### `OrcaGymAsyncAgent.compute_torques`

Signature:

```python
def compute_torques(self, qpos_buffer, qvel_buffer) -> np.ndarray
```

Docstring:

_No docstring._

##### `OrcaGymAsyncAgent.setup_curriculum`

Signature:

```python
def setup_curriculum(self, curriculum) -> None
```

Docstring:

_No docstring._

</details>


---

## `orca_gym/environment/async_env/orca_gym_async_env.py`

中文概览：环境层：Gymnasium Env 及其本地/异步/向量化实现。

### Classes

<details>
<summary>class OrcaGymAsyncEnv</summary>


_No class docstring._

#### Methods

##### `OrcaGymAsyncEnv.set_obs_space`

Signature:

```python
def set_obs_space(self)
```

Docstring:

_No docstring._

##### `OrcaGymAsyncEnv.set_action_space`

Signature:

```python
def set_action_space(self) -> None
```

Docstring:

_No docstring._

##### `OrcaGymAsyncEnv.initialize_agents`

Signature:

```python
def initialize_agents(self, entry, *args, **kwargs)
```

Docstring:

_No docstring._

##### `OrcaGymAsyncEnv.get_obs`

Signature:

```python
def get_obs(self) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], np.ndarray, np.ndarray]
```

Docstring:

Observation is environment specific and is defined in the subclass.

##### `OrcaGymAsyncEnv.reset_agents`

Signature:

```python
def reset_agents(self, agents)
```

Docstring:

Do specific reset operations for each agent. It is defined in the subclass.

##### `OrcaGymAsyncEnv.step_agents`

Signature:

```python
def step_agents(self, action) -> None
```

Docstring:

Do specific operations each step in the environment. It is defined in the subclass.

##### `OrcaGymAsyncEnv.step`

Signature:

```python
def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
```

Docstring:

_No docstring._

##### `OrcaGymAsyncEnv.reset_model`

Signature:

```python
def reset_model(self) -> tuple[ObsType, dict[str, np.ndarray]]
```

Docstring:

_No docstring._

##### `OrcaGymAsyncEnv.get_observation`

Signature:

```python
def get_observation(self, obs=None)
```

Docstring:

_No docstring._

##### `OrcaGymAsyncEnv.init_agent_joint_index`

Signature:

```python
def init_agent_joint_index(self)
```

Docstring:

_No docstring._

##### `OrcaGymAsyncEnv.reorder_agents`

Signature:

```python
def reorder_agents(self)
```

Docstring:

_No docstring._

##### `OrcaGymAsyncEnv.generate_action_scale_array`

Signature:

```python
def generate_action_scale_array(self, ctrl_info) -> np.ndarray
```

Docstring:

_No docstring._

##### `OrcaGymAsyncEnv.setup_curriculum`

Signature:

```python
def setup_curriculum(self, curriculum) -> None
```

Docstring:

_No docstring._

</details>


---

## `orca_gym/environment/async_env/orca_gym_vector_env.py`

中文概览：环境层：Gymnasium Env 及其本地/异步/向量化实现。

### Classes

<details>
<summary>class OrcaGymVectorEnv</summary>


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

#### Methods

##### `OrcaGymVectorEnv.reset`

Signature:

```python
def reset(self, *, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]
```

Docstring:

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

##### `OrcaGymVectorEnv.step`

Signature:

```python
def step(self, actions) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]
```

Docstring:

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

##### `OrcaGymVectorEnv.render`

Signature:

```python
def render(self) -> tuple[RenderFrame, ...] | None
```

Docstring:

Returns the rendered frames from the parallel environments.

Returns:
    A tuple of rendered frames from the parallel environments

##### `OrcaGymVectorEnv.close`

Signature:

```python
def close(self, **kwargs)
```

Docstring:

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

中文概览：环境层：Gymnasium Env 及其本地/异步/向量化实现。

### Classes

<details>
<summary>class OrcaGymAsyncSingleAgentEnvRunner</summary>


Wrap the SingleAgentEnvRunner to support orca gym asynchronous environments.

#### Methods

##### `OrcaGymAsyncSingleAgentEnvRunner.make_env`

Signature:

```python
def make_env(self) -> None
```

Docstring:

Creates a vectorized gymnasium env and stores it in `self.env`.

Note that users can change the EnvRunner's config (e.g. change
`self.config.env_config`) and then call this method to create new environments
with the updated configuration.

</details>


---

## `orca_gym/environment/async_env/subproc_vec_env.py`

中文概览：环境层：Gymnasium Env 及其本地/异步/向量化实现。

### Classes

<details>
<summary>class OrcaGymAsyncSubprocVecEnv</summary>


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

#### Methods

##### `OrcaGymAsyncSubprocVecEnv.step_async`

Signature:

```python
def step_async(self, actions) -> None
```

Docstring:

_No docstring._

##### `OrcaGymAsyncSubprocVecEnv.step_wait`

Signature:

```python
def step_wait(self) -> VecEnvStepReturn
```

Docstring:

_No docstring._

##### `OrcaGymAsyncSubprocVecEnv.reset`

Signature:

```python
def reset(self) -> VecEnvObs
```

Docstring:

_No docstring._

##### `OrcaGymAsyncSubprocVecEnv.close`

Signature:

```python
def close(self) -> None
```

Docstring:

_No docstring._

##### `OrcaGymAsyncSubprocVecEnv.get_images`

Signature:

```python
def get_images(self) -> Sequence[Optional[np.ndarray]]
```

Docstring:

_No docstring._

##### `OrcaGymAsyncSubprocVecEnv.get_attr`

Signature:

```python
def get_attr(self, attr_name, indices=None) -> List[Any]
```

Docstring:

Return attribute from vectorized environment (see base class).

##### `OrcaGymAsyncSubprocVecEnv.set_attr`

Signature:

```python
def set_attr(self, attr_name, value, indices=None) -> None
```

Docstring:

Set attribute inside vectorized environments (see base class).

##### `OrcaGymAsyncSubprocVecEnv.env_method`

Signature:

```python
def env_method(self, method_name, *method_args, indices=None, **method_kwargs) -> List[Any]
```

Docstring:

Call instance methods of vectorized environments.

##### `OrcaGymAsyncSubprocVecEnv.env_is_wrapped`

Signature:

```python
def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]
```

Docstring:

Check if worker environments are wrapped with a given wrapper

##### `OrcaGymAsyncSubprocVecEnv.setup_curriculum`

Signature:

```python
def setup_curriculum(self, curriculum_name) -> None
```

Docstring:

_No docstring._

</details>


---

## `orca_gym/environment/orca_gym_env.py`

中文概览：环境层：Gymnasium Env 及其本地/异步/向量化实现。

### Classes

<details>
<summary>class RewardType</summary>


_No class docstring._

</details>

<details>
<summary>class OrcaGymBaseEnv</summary>


Superclass for all OrcaSim environments.

#### Methods

##### `OrcaGymBaseEnv.step`

Signature:

```python
def step(self, action) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]
```

Docstring:

_No docstring._

##### `OrcaGymBaseEnv.reset_model`

Signature:

```python
def reset_model(self) -> tuple[dict, dict]
```

Docstring:

Reset the robot degrees of freedom (qpos and qvel).
Implement this in each subclass.

##### `OrcaGymBaseEnv.initialize_simulation`

Signature:

```python
def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]
```

Docstring:

Initialize MuJoCo simulation data structures mjModel and mjData.

##### `OrcaGymBaseEnv.render`

Signature:

```python
def render(self) -> Union[NDArray[np.float64], None]
```

Docstring:

Render a frame from the MuJoCo simulation as specified by the render_mode.

##### `OrcaGymBaseEnv.generate_action_space`

Signature:

```python
def generate_action_space(self, bounds) -> Space
```

Docstring:

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

##### `OrcaGymBaseEnv.generate_observation_space`

Signature:

```python
def generate_observation_space(self, obs) -> spaces.Space
```

Docstring:

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

##### `OrcaGymBaseEnv.reset`

Signature:

```python
def reset(self, *, seed=None, options=None)
```

Docstring:

_No docstring._

##### `OrcaGymBaseEnv.set_seed_value`

Signature:

```python
def set_seed_value(self, seed=None)
```

Docstring:

设置随机数种子

##### `OrcaGymBaseEnv.body`

Signature:

```python
def body(self, name, agent_id=None) -> str
```

Docstring:

获取带智能体前缀的 body 名称

##### `OrcaGymBaseEnv.joint`

Signature:

```python
def joint(self, name, agent_id=None) -> str
```

Docstring:

获取带智能体前缀的关节名称

##### `OrcaGymBaseEnv.actuator`

Signature:

```python
def actuator(self, name, agent_id=None) -> str
```

Docstring:

获取带智能体前缀的执行器名称

##### `OrcaGymBaseEnv.site`

Signature:

```python
def site(self, name, agent_id=None) -> str
```

Docstring:

获取带智能体前缀的 site 名称

##### `OrcaGymBaseEnv.mocap`

Signature:

```python
def mocap(self, name, agent_id=None) -> str
```

Docstring:

获取带智能体前缀的 mocap 名称

##### `OrcaGymBaseEnv.sensor`

Signature:

```python
def sensor(self, name, agent_id=None) -> str
```

Docstring:

获取带智能体前缀的传感器名称

##### `OrcaGymBaseEnv.dt`

Signature:

```python
def dt(self) -> float
```

Docstring:

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

##### `OrcaGymBaseEnv.agent_num`

Signature:

```python
def agent_num(self) -> int
```

Docstring:

获取智能体数量

##### `OrcaGymBaseEnv.do_simulation`

Signature:

```python
def do_simulation(self, ctrl, n_frames) -> None
```

Docstring:

Step the simulation n number of frames and applying a control action.

##### `OrcaGymBaseEnv.close`

Signature:

```python
def close(self)
```

Docstring:

Close all processes like rendering contexts

##### `OrcaGymBaseEnv.initialize_grpc`

Signature:

```python
def initialize_grpc(self)
```

Docstring:

Initialize the GRPC communication channel.

##### `OrcaGymBaseEnv.pause_simulation`

Signature:

```python
def pause_simulation(self)
```

Docstring:

Pause the simulation.

##### `OrcaGymBaseEnv.init_qpos_qvel`

Signature:

```python
def init_qpos_qvel(self)
```

Docstring:

Init qpos and qvel of the model.

##### `OrcaGymBaseEnv.reset_simulation`

Signature:

```python
def reset_simulation(self)
```

Docstring:

Reset the simulation.

##### `OrcaGymBaseEnv.set_time_step`

Signature:

```python
def set_time_step(self, time_step)
```

Docstring:

Set the time step of the simulation.

</details>


---

## `orca_gym/environment/orca_gym_local_env.py`

中文概览：环境层：Gymnasium Env 及其本地/异步/向量化实现。

### Classes

<details>
<summary>class OrcaGymLocalEnv</summary>


_No class docstring._

#### Methods

##### `OrcaGymLocalEnv.initialize_simulation`

Signature:

```python
def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]
```

Docstring:

初始化仿真，加载模型并创建模型和数据对象

##### `OrcaGymLocalEnv.initialize_grpc`

Signature:

```python
def initialize_grpc(self)
```

Docstring:

初始化 gRPC 通信通道和客户端

##### `OrcaGymLocalEnv.pause_simulation`

Signature:

```python
def pause_simulation(self)
```

Docstring:

暂停仿真（采用被动模式）

##### `OrcaGymLocalEnv.close`

Signature:

```python
def close(self)
```

Docstring:

关闭环境，清理资源

##### `OrcaGymLocalEnv.get_body_manipulation_anchored`

Signature:

```python
def get_body_manipulation_anchored(self)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.begin_save_video`

Signature:

```python
def begin_save_video(self, file_path, capture_mode=CaptureMode.ASYNC)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.stop_save_video`

Signature:

```python
def stop_save_video(self)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.get_next_frame`

Signature:

```python
def get_next_frame(self) -> int
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.get_current_frame`

Signature:

```python
def get_current_frame(self) -> int
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.get_camera_time_stamp`

Signature:

```python
def get_camera_time_stamp(self, last_frame_index) -> dict
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.get_frame_png`

Signature:

```python
def get_frame_png(self, image_path)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.get_body_manipulation_movement`

Signature:

```python
def get_body_manipulation_movement(self)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.do_simulation`

Signature:

```python
def do_simulation(self, ctrl, n_frames) -> None
```

Docstring:

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

##### `OrcaGymLocalEnv.render_mode`

Signature:

```python
def render_mode(self) -> str
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.is_subenv`

Signature:

```python
def is_subenv(self) -> bool
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.sync_render`

Signature:

```python
def sync_render(self) -> bool
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.render`

Signature:

```python
def render(self)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.do_body_manipulation`

Signature:

```python
def do_body_manipulation(self)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.release_body_anchored`

Signature:

```python
def release_body_anchored(self)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.anchor_actor`

Signature:

```python
def anchor_actor(self, actor_name, anchor_type)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.update_anchor_equality_constraints`

Signature:

```python
def update_anchor_equality_constraints(self, actor_name, anchor_type)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.set_ctrl`

Signature:

```python
def set_ctrl(self, ctrl)
```

Docstring:

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

##### `OrcaGymLocalEnv.mj_step`

Signature:

```python
def mj_step(self, nstep)
```

Docstring:

执行 MuJoCo 仿真步进

执行 nstep 次物理仿真步进，每次步进的时间为 timestep。
在调用前需要先设置控制输入 (set_ctrl)。

使用示例:
    ```python
    # 在 step 函数中执行仿真
    self.set_ctrl(self.ctrl)
    self.do_simulation(self.ctrl, self.frame_skip)  # 内部调用 mj_step
    ```

##### `OrcaGymLocalEnv.mj_forward`

Signature:

```python
def mj_forward(self)
```

Docstring:

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

##### `OrcaGymLocalEnv.mj_jacBody`

Signature:

```python
def mj_jacBody(self, jacp, jacr, body_id)
```

Docstring:

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

##### `OrcaGymLocalEnv.mj_jacSite`

Signature:

```python
def mj_jacSite(self, jacp, jacr, site_name)
```

Docstring:

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

##### `OrcaGymLocalEnv.set_time_step`

Signature:

```python
def set_time_step(self, time_step)
```

Docstring:

设置仿真时间步长

##### `OrcaGymLocalEnv.update_data`

Signature:

```python
def update_data(self)
```

Docstring:

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

##### `OrcaGymLocalEnv.reset_simulation`

Signature:

```python
def reset_simulation(self)
```

Docstring:

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

##### `OrcaGymLocalEnv.init_qpos_qvel`

Signature:

```python
def init_qpos_qvel(self)
```

Docstring:

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

##### `OrcaGymLocalEnv.query_joint_offsets`

Signature:

```python
def query_joint_offsets(self, joint_names) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Docstring:

查询关节在状态数组中的偏移量（位置、速度、加速度）

##### `OrcaGymLocalEnv.query_joint_lengths`

Signature:

```python
def query_joint_lengths(self, joint_names)
```

Docstring:

查询关节状态数组的长度（位置、速度、加速度维度）

##### `OrcaGymLocalEnv.get_body_xpos_xmat_xquat`

Signature:

```python
def get_body_xpos_xmat_xquat(self, body_name_list)
```

Docstring:

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

##### `OrcaGymLocalEnv.query_sensor_data`

Signature:

```python
def query_sensor_data(self, sensor_names)
```

Docstring:

查询传感器数据

##### `OrcaGymLocalEnv.query_joint_qpos`

Signature:

```python
def query_joint_qpos(self, joint_names)
```

Docstring:

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

##### `OrcaGymLocalEnv.query_joint_qvel`

Signature:

```python
def query_joint_qvel(self, joint_names)
```

Docstring:

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

##### `OrcaGymLocalEnv.query_joint_qacc`

Signature:

```python
def query_joint_qacc(self, joint_names)
```

Docstring:

查询关节加速度

返回指定关节的当前加速度，字典格式，键为关节名称，值为加速度值或数组。

使用示例:
    ```python
    # 查询关节加速度
    joint_acc = self.query_joint_qacc(["joint1", "joint2"])
    # 用于分析运动状态或计算动力学
    ```

##### `OrcaGymLocalEnv.jnt_qposadr`

Signature:

```python
def jnt_qposadr(self, joint_name)
```

Docstring:

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

##### `OrcaGymLocalEnv.jnt_dofadr`

Signature:

```python
def jnt_dofadr(self, joint_name)
```

Docstring:

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

##### `OrcaGymLocalEnv.query_site_pos_and_mat`

Signature:

```python
def query_site_pos_and_mat(self, site_names)
```

Docstring:

查询 site 的位置和旋转矩阵

##### `OrcaGymLocalEnv.query_site_pos_and_quat`

Signature:

```python
def query_site_pos_and_quat(self, site_names) -> Dict[str, Dict[str, Union[NDArray[np.float64], NDArray[np.float64]]]]
```

Docstring:

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

##### `OrcaGymLocalEnv.query_site_size`

Signature:

```python
def query_site_size(self, site_names)
```

Docstring:

查询 site 的尺寸

##### `OrcaGymLocalEnv.query_site_pos_and_quat_B`

Signature:

```python
def query_site_pos_and_quat_B(self, site_names, base_body_list) -> Dict[str, Dict[str, Union[NDArray[np.float32], NDArray[np.float32]]]]
```

Docstring:

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

##### `OrcaGymLocalEnv.set_joint_qpos`

Signature:

```python
def set_joint_qpos(self, joint_qpos)
```

Docstring:

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

##### `OrcaGymLocalEnv.set_joint_qvel`

Signature:

```python
def set_joint_qvel(self, joint_qvel)
```

Docstring:

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

##### `OrcaGymLocalEnv.query_site_xvalp_xvalr`

Signature:

```python
def query_site_xvalp_xvalr(self, site_names) -> Tuple[Dict[str, NDArray[np.float64]], Dict[str, NDArray[np.float64]]]
```

Docstring:

查询 site 的线速度和角速度（世界坐标系）

##### `OrcaGymLocalEnv.query_site_xvalp_xvalr_B`

Signature:

```python
def query_site_xvalp_xvalr_B(self, site_names, base_body_list) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, NDArray[np.float32]]]
```

Docstring:

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

##### `OrcaGymLocalEnv.update_equality_constraints`

Signature:

```python
def update_equality_constraints(self, eq_list)
```

Docstring:

更新等式约束列表

##### `OrcaGymLocalEnv.set_mocap_pos_and_quat`

Signature:

```python
def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict)
```

Docstring:

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

##### `OrcaGymLocalEnv.query_contact_simple`

Signature:

```python
def query_contact_simple(self)
```

Docstring:

查询简单接触信息

##### `OrcaGymLocalEnv.set_geom_friction`

Signature:

```python
def set_geom_friction(self, geom_friction_dict)
```

Docstring:

设置几何体的摩擦系数

##### `OrcaGymLocalEnv.add_extra_weight`

Signature:

```python
def add_extra_weight(self, weight_load_dict)
```

Docstring:

为 body 添加额外重量

##### `OrcaGymLocalEnv.query_contact_force`

Signature:

```python
def query_contact_force(self, contact_ids)
```

Docstring:

查询接触力

##### `OrcaGymLocalEnv.get_cfrc_ext`

Signature:

```python
def get_cfrc_ext(self)
```

Docstring:

获取外部约束力

##### `OrcaGymLocalEnv.query_actuator_torques`

Signature:

```python
def query_actuator_torques(self, actuator_names)
```

Docstring:

查询执行器扭矩

##### `OrcaGymLocalEnv.query_joint_dofadrs`

Signature:

```python
def query_joint_dofadrs(self, joint_names)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.get_goal_bounding_box`

Signature:

```python
def get_goal_bounding_box(self, geom_name)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.query_velocity_body_B`

Signature:

```python
def query_velocity_body_B(self, ee_body, base_body)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.query_position_body_B`

Signature:

```python
def query_position_body_B(self, ee_body, base_body)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.query_orientation_body_B`

Signature:

```python
def query_orientation_body_B(self, ee_body, base_body)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.query_joint_axes_B`

Signature:

```python
def query_joint_axes_B(self, joint_names, base_body)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.query_robot_velocity_odom`

Signature:

```python
def query_robot_velocity_odom(self, base_body, initial_base_pos, initial_base_quat)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.query_robot_position_odom`

Signature:

```python
def query_robot_position_odom(self, base_body, initial_base_pos, initial_base_quat)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.query_robot_orientation_odom`

Signature:

```python
def query_robot_orientation_odom(self, base_body, initial_base_pos, initial_base_quat)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.set_actuator_trnid`

Signature:

```python
def set_actuator_trnid(self, actuator_id, trnid)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.disable_actuator`

Signature:

```python
def disable_actuator(self, actuator_groups)
```

Docstring:

_No docstring._

##### `OrcaGymLocalEnv.load_content_file`

Signature:

```python
def load_content_file(self, content_file_name, remote_file_dir='', local_file_dir='', temp_file_path=None)
```

Docstring:

_No docstring._

</details>


---

## `orca_gym/environment/orca_gym_remote_env.py`

中文概览：环境层：Gymnasium Env 及其本地/异步/向量化实现。

### Classes

<details>
<summary>class OrcaGymRemoteEnv</summary>


_No class docstring._

#### Methods

##### `OrcaGymRemoteEnv.initialize_simulation`

Signature:

```python
def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.do_simulation`

Signature:

```python
def do_simulation(self, ctrl, n_frames) -> None
```

Docstring:

Step the simulation n number of frames and applying a control action.

##### `OrcaGymRemoteEnv.set_qpos_qvel`

Signature:

```python
def set_qpos_qvel(self, qpos, qvel)
```

Docstring:

Set the joints position qpos and velocity qvel of the model.

Note: `qpos` and `qvel` is not the full physics state for all mujoco models/environments https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate

##### `OrcaGymRemoteEnv.render`

Signature:

```python
def render(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.get_observation`

Signature:

```python
def get_observation(self, obs=None)
```

Docstring:

Return the current environment observation as a dictionary, unless obs is not None.
This function should process the raw environment observation to align with the input expected by the policy model.
For example, it should cast an image observation to float with value range 0-1 and shape format [C, H, W].

##### `OrcaGymRemoteEnv.close`

Signature:

```python
def close(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.get_body_com_dict`

Signature:

```python
def get_body_com_dict(self, body_name_list) -> Dict[str, Dict[str, NDArray[np.float64]]]
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.get_body_com_xpos_xmat`

Signature:

```python
def get_body_com_xpos_xmat(self, body_name_list) -> Tuple[NDArray[np.float64], NDArray[np.float64]]
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.get_body_com_xpos_xmat_list`

Signature:

```python
def get_body_com_xpos_xmat_list(self, body_name_list) -> Tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.get_body_xpos_xmat_xquat`

Signature:

```python
def get_body_xpos_xmat_xquat(self, body_name_list)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.get_geom_xpos_xmat`

Signature:

```python
def get_geom_xpos_xmat(self, geom_name_list)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.initialize_grpc`

Signature:

```python
def initialize_grpc(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.pause_simulation`

Signature:

```python
def pause_simulation(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.init_qpos_qvel`

Signature:

```python
def init_qpos_qvel(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.reset_simulation`

Signature:

```python
def reset_simulation(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.set_ctrl`

Signature:

```python
def set_ctrl(self, ctrl)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.mj_forward`

Signature:

```python
def mj_forward(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.set_time_step`

Signature:

```python
def set_time_step(self, time_step)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_joint_qpos`

Signature:

```python
def query_joint_qpos(self, joint_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_joint_qvel`

Signature:

```python
def query_joint_qvel(self, joint_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.set_joint_qpos`

Signature:

```python
def set_joint_qpos(self, joint_qpos)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_cfrc_ext`

Signature:

```python
def query_cfrc_ext(self, body_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_actuator_force`

Signature:

```python
def query_actuator_force(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.load_keyframe`

Signature:

```python
def load_keyframe(self, keyframe_name)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_joint_limits`

Signature:

```python
def query_joint_limits(self, joint_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_body_velocities`

Signature:

```python
def query_body_velocities(self, body_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_actuator_gain_prm`

Signature:

```python
def query_actuator_gain_prm(self, actuator_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.set_actuator_gain_prm`

Signature:

```python
def set_actuator_gain_prm(self, gain_prm_set_list)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_actuator_bias_prm`

Signature:

```python
def query_actuator_bias_prm(self, actuator_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.set_actuator_bias_prm`

Signature:

```python
def set_actuator_bias_prm(self, bias_prm_set_list)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_mocap_pos_and_quat`

Signature:

```python
def query_mocap_pos_and_quat(self, mocap_body_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.set_mocap_pos_and_quat`

Signature:

```python
def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_site_pos_and_mat`

Signature:

```python
def query_site_pos_and_mat(self, site_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_site_pos_and_quat`

Signature:

```python
def query_site_pos_and_quat(self, site_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_site_xvalp_xvalr`

Signature:

```python
def query_site_xvalp_xvalr(self, site_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.update_equality_constraints`

Signature:

```python
def update_equality_constraints(self, eq_list)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_all_geoms`

Signature:

```python
def query_all_geoms(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_opt_config`

Signature:

```python
def query_opt_config(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.set_opt_config`

Signature:

```python
def set_opt_config(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_contact_simple`

Signature:

```python
def query_contact_simple(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_contact`

Signature:

```python
def query_contact(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_contact_force`

Signature:

```python
def query_contact_force(self, contact_ids)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.mj_jac`

Signature:

```python
def mj_jac(self, body_point_list, compute_jacp=True, compute_jacr=True)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.calc_full_mass_matrix`

Signature:

```python
def calc_full_mass_matrix(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_qfrc_bias`

Signature:

```python
def query_qfrc_bias(self)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_subtree_com`

Signature:

```python
def query_subtree_com(self, body_name)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.set_geom_friction`

Signature:

```python
def set_geom_friction(self, geom_name_list, friction_list)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_sensor_data`

Signature:

```python
def query_sensor_data(self, sensor_names)
```

Docstring:

_No docstring._

##### `OrcaGymRemoteEnv.query_joint_offsets`

Signature:

```python
def query_joint_offsets(self, joint_names) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Docstring:

_No docstring._

</details>
