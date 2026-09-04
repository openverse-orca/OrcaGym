# 🏋️ Gymnasium Interface

OrcaGym strictly follows the Gymnasium specification, providing a standard RL environment interface.

## Environment Base Class

```
gymnasium.Env
  └── OrcaGymEulerEnv      # 👈 Recommended: current primary path
        └── Your environment class
```

## OrcaGymEulerEnv

`OrcaGymEulerEnv` is the recommended base class for writing custom environments. It encapsulates the core simulation functionality, allowing you to focus on task logic.

### Constructor Parameters

```python
class MyEnv(OrcaGymEulerEnv):
    def __init__(self, ...):
        super().__init__(
            frame_skip: int,          # Number of physics steps per step() call
            orcagym_addr: str,        # gRPC server address
            agent_names: list[str],   # List of agent names
            time_step: float,         # Physics time step (seconds)
            *,
            model_xml_path: str | None = None,   # Local model path (offline mode)
            skip_grpc_load: bool = False,        # True → offline mode
            render_mode: str = "human",          # "human" / "none"
            sync_render: bool = False,           # Whether to render synchronously
            **kwargs,
        )
```

### Abstract Methods That Must Be Implemented

Every subclass must override the following three methods:

```python
def step(self, action) -> tuple:
    """Execute one simulation step, returning obs, reward, terminated, truncated, info"""
    ...

def reset_model(self) -> tuple:
    """Reset the robot state to its initial pose, returning (obs, info)"""
    ...

def _get_obs(self):
    """Build the observation"""
    ...
```

### Key Attributes

```python
env.data          # Complete state read-only view
env.model         # Model structure information
env.sim_config    # Solver configuration
env.ctrl          # Current control input

@property
env.dt            # Environment time step = timestep × frame_skip
```

### Key Methods

```python
# Simulation stepping — core method
env.do_simulation(ctrl, n_frames)

# State setting
env.set_joint_qpos(qpos)
env.set_joint_qvel(qvel)

# Forward computation (update derived quantities)
env.mj_forward()

# Apply external forces
env.apply_body_force(body_name, force, torque)

# Mocap control
env.set_mocap_pos_and_quat(mocap_dict)
```

## Observation and Action Spaces

### Action Space

OrcaGym automatically generates the `action_space` from the actuator control ranges:

```python
# internal: action_space = spaces.Box(low=ctrl_range[:,0], high=ctrl_range[:,1])
print(env.action_space)  # Box(low=-1.0, high=1.0, shape=(nu,), float32)
```

### Observation Space

The observation space is defined by you in `_get_obs()`:

```python
def _get_obs(self):
    """Build the observation, return np.ndarray or dict"""
    obs = np.concatenate([self.data.qpos, self.data.qvel])
    return obs.astype(np.float32)

# observation_space is automatically inferred on first reset
```

### Supported Space Types

| Type | Python Class | Description |
|------|-----------|------|
| Box Continuous | `spaces.Box` | Observation/action is a numpy array (directly supported by `generate_action_space` / `generate_observation_space`) |
| Dict Space | `spaces.Dict` | Observation is a dictionary (multimodal) (directly supported by `generate_observation_space`) |

> Note: `generate_action_space` / `generate_observation_space` only directly support `Box` and `Dict`. General Gymnasium capabilities (such as the `Discrete` discrete space) are not within OrcaGym's auto-generation scope; if needed, you can manually construct `action_space` / `observation_space` in subclasses.

## Creating an Environment

### Direct Instantiation (Recommended)

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

### Via gym.make

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

### Via Registration

```python
gym.register(
    id="MyTask-v0",
    entry_point="my_package:MyTaskEnv",
    max_episode_steps=1000,
)
env = gym.make("MyTask-v0", **specific_kwargs)
```

## Key Conventions

1. **`env.dt`** is the policy control period, not MuJoCo's `timestep`
2. **`frame_skip`** determines how many physics steps are executed per `step()`
3. **Observations should be built in `_get_obs()`**; data is already updated after `do_simulation()`
4. **In multi-agent setups**, body/joint/actuator names are automatically prefixed with the agent name (via `self.body()` and similar methods)
5. **`do_simulation()`** returns after `env.data` has been automatically synchronized
6. **State configuration** is accessed through `env.sim_config`
