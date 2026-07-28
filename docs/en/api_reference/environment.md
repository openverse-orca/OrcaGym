# 🌍 Environment API

Gymnasium environment interface, providing a standard reinforcement learning environment abstraction.

## Main Classes

| Class | Description |
|----|------|
| **`OrcaGymEulerEnv`** | Recommended environment base class |
| `OrcaGymVectorEnv` | Vectorized parallel environment |

---

## OrcaGymEulerEnv

`OrcaGymEulerEnv` is the recommended base class for writing custom environments. It encapsulates the simulation core, letting you focus on task logic.

### Constructor Parameters

```python
class OrcaGymEulerEnv:
    def __init__(
        self,
        frame_skip: int,           # Number of physics steps per step()
        orcagym_addr: str,         # gRPC server address (e.g. "localhost:50051")
        agent_names: list[str],    # List of agent names
        time_step: float,          # Physics simulation time step (seconds)
        *,
        model_xml_path: str | None = None,   # Local XML path (offline mode)
        skip_grpc_load: bool = False,        # True → offline mode
        render_mode: str = "human",          # "human" / "none"
        sync_render: bool = False,
        **kwargs,
    )
```

### Public Properties

```python
data: OrcaGymDataView          # Complete read-only state view
model: OrcaGymModel            # Model structure information
sim_config: SimConfig          # Solver parameter configuration
ctrl: np.ndarray               # Current control input array
init_qpos: np.ndarray          # Cached initial generalized coordinates
init_qvel: np.ndarray          # Cached initial generalized velocity
frame_skip: int                # Number of physics steps per step()
seed: int                      # Random seed

@property
dt: float                      # Environment time step = sim_config.timestep × frame_skip
```

### Simulation Control

```python
def do_simulation(ctrl: np.ndarray, n_frames: int) -> None
```
Core stepping method. Set control → step n_frames → auto-sync state. After calling, `self.data` is the latest state. ctrl must have shape `(nu,)`.

```python
def set_ctrl(ctrl: np.ndarray) -> None
def mj_step(nstep: int) -> None
def mj_forward() -> None
```
`set_ctrl` sets the control input without stepping; `mj_step` / `mj_forward` are low-level simulation controls, typically not needed directly.

### State Setting

```python
def set_joint_qpos(qpos: np.ndarray) -> None   # Set generalized coordinates (full)
def set_joint_qvel(qvel: np.ndarray) -> None   # Set generalized velocity (full)
```
After setting, call `mj_forward()` to update derived quantities.

### Force Application

```python
def apply_body_force(body_name: str, force: np.ndarray, torque: np.ndarray) -> None
def clear_body_force(body_name: str) -> None
def clear_all_forces() -> None
def mj_apply_force_at_site(site_name: str, force: np.ndarray, torque: np.ndarray) -> None
def mj_clear_xfrc_applied_for_site(site_name: str) -> None
```

### Mocap and Geometry Settings

```python
def set_mocap_pos_and_quat(mocap_pos_and_quat_dict: dict) -> None
def set_geom_friction(geom_friction_dict: dict) -> None
def add_extra_weight(weight_load_dict: dict) -> None
```

### State Queries (by name, no ID needed)

```python
# Joint queries
def query_joint_qpos(joint_names: list[str]) -> dict[str, np.ndarray]
def query_joint_qvel(joint_names: list[str]) -> dict[str, np.ndarray]
def query_joint_qacc(joint_names: list[str]) -> dict[str, np.ndarray]
def query_joint_offsets(joint_names: list[str]) -> dict[str, np.ndarray]   # Joint offsets
def query_joint_lengths(joint_names: list[str]) -> dict[str, np.ndarray]   # Joint lengths
def query_joint_dofadrs(joint_names: list[str]) -> dict[str, int]           # Joint DOF start addresses
def jnt_qposadr(joint_name: str) -> int
def jnt_dofadr(joint_name: str) -> int

# Body pose
def get_body_xpos_xmat_xquat(body_name_list: list[str]) -> dict
def get_body_xpos_xmat_xquat_xvel(body_name_list: list[str]) -> dict

# Site queries
def query_site_pos_and_mat(site_names: list[str]) -> dict
def query_site_size(site_names: list[str]) -> dict[str, np.ndarray]

# Sensor / Actuator / Contact
def query_sensor_data(sensor_names: list[str]) -> dict[str, np.ndarray]
def query_actuator_torques(actuator_names: list[str]) -> dict[str, np.ndarray]
def query_contact_simple() -> list[dict]
def query_contact_force(contact_ids: list[int]) -> dict[int, np.ndarray]
def get_cfrc_ext() -> np.ndarray
def get_goal_bounding_box(geom_name: str) -> np.ndarray   # Geom bounding box half-size (3,)
def body_subtree_mass(body_name: str) -> float
```

### Base Frame Coordinate Transformations

```python
def query_site_pos_and_quat_B(site_names, base_body_list) -> dict
def query_site_xvalp_xvalr(site_names) -> tuple[dict, dict]
def query_site_xvalp_xvalr_B(site_names, base_body_list) -> tuple[dict, dict]
def query_velocity_body_B(ee_body, base_body) -> np.ndarray       # 6D velocity (base frame)
def query_position_body_B(ee_body, base_body) -> np.ndarray       # 3D position (base frame)
def query_orientation_body_B(ee_body, base_body) -> np.ndarray    # Quaternion (base frame)
def query_joint_axes_B(joint_names, base_body) -> dict            # Joint axis directions (base frame)
```

### Odometry Queries

```python
def query_robot_velocity_odom(base_body, initial_base_pos, initial_base_quat) -> tuple
def query_robot_position_odom(base_body, initial_base_pos, initial_base_quat) -> np.ndarray
def query_robot_orientation_odom(base_body, initial_base_pos, initial_base_quat) -> np.ndarray
```

### Jacobian

```python
def mj_jacBody(jacp: np.ndarray, jacr: np.ndarray, body_name: str) -> None
def mj_jacSite(jacp: np.ndarray, jacr: np.ndarray, site_name: str) -> None
def mj_jac_site(site_names: list[str]) -> dict[str, dict]
```

### Equality Constraint Primitives (Recommended)

> The legacy `update_equality_constraints` / `anchor_actor` etc. have been removed. Use the following atomic primitives instead:

```python
def equality_find_slot_by_body(body_name: str) -> int          # Find equality constraint slot containing the given body, returns -1 if not found
def equality_constraint(slot: int) -> dict                     # Read complete data of a single equality constraint
def equality_update(
    slot: int,
    *,
    eq_type: int | None = None,       # mjtEq type constant
    obj1_name: str | None = None,     # New obj1 body name
    obj2_name: str | None = None,     # New obj2 body name
    data: np.ndarray | None = None,   # Constraint data (mjNEQDATA,)
    active: bool | None = None,       # Whether active
    solref: np.ndarray | None = None, # Solver reference parameters (2,)
    solimp: np.ndarray | None = None, # Solver impedance parameters (5,)
    forward: bool = True,             # Whether to call mj_forward after writing
) -> None
```

### Read-Only Queries

```python
def geom_friction(geom_name: str) -> np.ndarray   # Geom friction coefficients (3,) [sliding, torsion, rolling]
```

### Studio Interaction

```python
def render() -> np.ndarray | None              # Render to Studio
def begin_save_video(file_path, capture_mode=0) -> None
def stop_save_video() -> None
def get_current_frame() -> int
def get_next_frame() -> int
def get_camera_time_stamp(last_frame_index) -> dict
def get_frame_png(image_path) -> None
def load_content_file(content_file_name, **kwargs) -> None
```

### Camera Sensor Configuration

```python
def set_camera_sensor_info(
    actor_name: str,
    capture_rgb: bool,
    capture_depth: bool,
    save_mp4_file: bool = False,
    use_dds: bool = False,
    **kwargs,   # Optional extension parameters: capture_normal, width, height, vertical_fov, near_clip, far_clip, gamma, color_port, depth_port, ...
) -> None
def make_camera_viewport_active(actor_name: str, entity_name: str) -> None
```

### Studio Bridge

```python
def studio_bridge()   # Returns the OrcaStudio bridge object (K9 method access pattern)
```

### Namespace Resolution (Multi-Agent)

Inherited from `OrcaGymEnvMixin`, used to automatically prefix entity names with agent names in multi-agent environments.

```python
@property
agent_num: int                          # Number of agents

def body(name: str, agent_id: int | None = None) -> str       # body name → "agent_name_body_name"
def joint(name: str, agent_id: int | None = None) -> str      # Joint name
def actuator(name: str, agent_id: int | None = None) -> str   # Actuator name
def site(name: str, agent_id: int | None = None) -> str       # Site name
def mocap(name: str, agent_id: int | None = None) -> str      # Mocap name
def sensor(name: str, agent_id: int | None = None) -> str     # Sensor name
```

> When `agent_id=None`, defaults to the first agent (index 0).

**Example:**
```python
# Single-agent environment (agent_names=["robot_1"])
env.body("pelvis")    # → "robot_1_pelvis"
env.joint("leg_l_1")  # → "robot_1_leg_l_1"

# Multi-agent environment (agent_names=["robot_1", "robot_2"])
env.body("pelvis", agent_id=1)  # → "robot_2_pelvis"
```

### Action/Observation Space Generation

Inherited from `OrcaGymEnvMixin`, used to conveniently generate Gymnasium-compliant action and observation spaces.

```python
def generate_action_space(bounds: np.ndarray) -> gym.Space      # (nu, 2) → Box action space
def generate_observation_space(obs: np.ndarray | dict) -> gym.Space  # Generate observation space from a sample observation
```

`generate_action_space` automatically handles ±inf bounds (clamped to the float32 representable range) to avoid gymnasium overflow warnings.

**Example:**
```python
# Generate action space from actuator control ranges
ctrlrange = self.model.get_actuator_ctrlrange()  # (nu, 2)
self.action_space = self.generate_action_space(ctrlrange)

# Generate observation space from a sample observation
obs_sample = self._get_obs()
self.observation_space = self.generate_observation_space(obs_sample)
```

### Random Seed

```python
def set_seed_value(seed: int) -> list[int]     # Set the random seed, returns a seed list
```

After setting, use `self.np_random` to access the `RandomState` instance.

### Reset (Gymnasium Standard Interface)

```python
def reset(*, seed: int | None = None, options: dict | None = None) -> tuple[ObsType, dict]
```

Provided by `OrcaGymEnvMixin`, orchestration order: `set_seed_value` → `reset_simulation` → `reset_model` → `render`.

> Subclasses should override `reset_model()` rather than overriding `reset()` directly.

### Abstract Methods (subclasses must implement)

```python
def step(action: np.ndarray) -> tuple[ObsType, float, bool, bool, dict]
def reset_model() -> tuple[np.ndarray | dict, dict]
def _get_obs() -> np.ndarray | dict
```

### Lifecycle Methods

```python
def initialize_grpc()
def initialize_simulation()     # Load model
def reset_simulation()          # Reset state
def init_qpos_qvel()            # Cache initial state
def set_time_step(time_step)    # Set time step
def pause_simulation()
def close()                     # Close connection
```

### Usage Example

```python
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class MyRobotEnv(OrcaGymEulerEnv):
    """Minimal environment: Box observation + Box action, offline mode."""

    def __init__(self, model_xml_path: str):
        super().__init__(
            frame_skip=5,
            orcagym_addr="localhost:50051",
            agent_names=["robot_1"],
            time_step=0.001,
            model_xml_path=model_xml_path,
            skip_grpc_load=True,   # Offline mode
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        obs_sample = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
        ]).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = False
        truncated = False
        info = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs: np.ndarray) -> float:
        return 0.0  # Replace with your reward function

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
        self.set_joint_qpos(qpos)
        self.set_joint_qvel(qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}


# Usage
if __name__ == "__main__":
    env = MyRobotEnv(model_xml_path="/path/to/scene.xml")
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()
```

---

## OrcaGymVectorEnv

Vectorized environment, running multiple agents in parallel within a single MuJoCo instance. Inherits from Gymnasium `VectorEnv`.

> ⚠️ Note: `num_envs` must be an integer multiple of 32 (internally grouped into sets of 32 agents).

### Constructor Parameters

```python
class OrcaGymVectorEnv(VectorEnv):
    def __init__(
        self,
        num_envs: int,           # Total number of parallel environments (must be a multiple of 32)
        worker_index: int,       # Worker process index
        entry_point: str,        # Import path for the environment class (e.g. "my_package.envs:MyEnv")
        **kwargs,                # Additional parameters passed to the environment constructor
    )
```

### Public Properties

```python
num_envs: int                        # Number of parallel environments
observation_space: gym.Space         # Batched observation space
single_observation_space: gym.Space  # Single-environment observation space
action_space: gym.Space              # Batched action space
single_action_space: gym.Space       # Single-environment action space
```

### Main Methods

```python
def reset(*, seed=None, options=None) -> tuple[ObsType, list[dict]]
def step(actions: ActType) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, list[dict]]
def render() -> None
def close() -> None
```

### Return Value Descriptions

| Return Value | Shape/Type | Description |
|--------|-----------|------|
| `observations` | `(num_envs, *obs_shape)` | Observations for all environments |
| `rewards` | `(num_envs,)` | Rewards for all environments |
| `terminated` | `(num_envs,) bool` | Whether terminated |
| `truncated` | `(num_envs,) bool` | Whether truncated |
| `infos` | `list[dict]` | Info dict for each environment |

---

## RewardType

```python
class RewardType:
    SPARSE = "sparse"
    DENSE = "dense"
```
