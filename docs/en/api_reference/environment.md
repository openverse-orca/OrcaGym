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
ctrl: np.ndarray               # The getter returns the current actuator_force; the setter sets the control input
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
Core stepping method. Set control → step n_frames → auto-sync state. After calling, `self.data` is the latest state. `ctrl` must have shape `(nu,)`.

```python
def mj_step(nstep: int) -> None
def mj_forward() -> None
```
Low-level simulation controls, typically not needed directly.

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
def query_joint_dofadrs(joint_names: list[str]) -> dict[str, int]          # Joint DOF start addresses
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

> Note: `mj_jacBody` / `mj_jacSite` signatures are `(jacp, jacr, body_name)` — no `*` separator. The first two parameters are caller-preallocated arrays written in-place.

### Equality Constraints and Body Manipulation

Env-layer public primitives (programmatic manipulation should be orchestrated using the following methods):

```python
def equality_find_slot_by_body(body_name: str) -> int          # Find the equality constraint slot containing the given body; returns -1 if not found
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

> ⚠️ **Note**: The following methods do not exist at the Env layer:
> - `update_equality_constraints` / `modify_equality_objects`: only exist in
>   `OrcaGymEuler` (gym layer) and `MuJoCoSimCore` (sim layer).
>   The `modify_equality_objects` signature is
>   `modify_equality_objects(eq_ids: list[int], obj1_ids=None, obj2_ids=None)`,
>   with parameters being int lists, not names.
> - `update_anchor_equality_constraints` / `anchor_actor` /
>   `release_body_anchored` / `do_body_manipulation`: these are Env-layer internal `_`-prefixed
>   methods (`_anchor_actor` / `_release_body_anchored` / `_do_body_manipulation`),
>   used by the UI grasping state machine driven internally by `render()`, and should not be called directly.
>   Programmatic manipulation should use `equality_find_slot_by_body` + `equality_constraint` +
>   `equality_update` + `set_mocap_pos_and_quat` primitive orchestration.

### Read-Only Queries

```python
def geom_friction(geom_name: str) -> np.ndarray   # Geom friction coefficients (3,) [sliding, torsion, rolling]
```

### Studio Interaction

```python
def render(simulate_index: int = -1, request_idr: bool = False) -> np.ndarray | None
    # Render to Studio. simulate_index is passed through to the engine camera
    # pipeline for frame alignment. -1 means server-side auto-increment (default).
    # Pass a monotonically increasing value >=0 when client-side recording is enabled.
    # request_idr=True asks the engine to output an IDR keyframe on this render
    # (use at a recording-segment start, together with save_streaming's internal
    # default truncation so the video starts at a keyframe).

# The following methods are deprecated (engine-side MP4 recording RPCs removed).
# Calling them emits a DeprecationWarning:
def begin_save_video(file_path, capture_mode=0) -> None       # [Deprecated] no-op
def stop_save_video() -> None                                  # [Deprecated] no-op
def get_current_frame() -> int                                 # [Deprecated] returns -1
def get_next_frame() -> int                                    # [Deprecated] returns 0
def get_camera_time_stamp(last_frame_index) -> dict            # [Deprecated] returns {}

def get_frame_png(image_path) -> None
def load_content_file(content_file_name, **kwargs) -> None
```

### Camera Recording API (Client-side PyAV remux)

```python
def save_streaming(
    camera_name: str,
    camera_type: str,
    file_path: str,
    start_simulate_index: int,
    end_simulate_index: int,
) -> Future[RemuxResult]
    # Save the specified camera's [start, end] range as an MP4. **Non-blocking**,
    # returns a Future.
    # Operates via the VideoRecorderManager unified interface: idempotently starts
    # the recorder and registers a range-save task in that camera's waiting queue.
    # Each range task carries its own start/end independently, so multiple ranges
    # can be registered simultaneously without interfering with each other.
    # When the receiver thread sees a frame with simulate_index >= end, a save
    # worker thread remuxes asynchronously without blocking the receiver thread
    # or the caller thread.
    # This tolerates the physics-step -> render -> frame-capture latency (e.g.
    # saving 0-500 while the buffer only holds 490 frames; the task waits until
    # frame 500 arrives before saving).
    # Internally truncates to the first keyframe in the range
    # (``truncate_to_keyframe=True``) so the video starts at a keyframe (use with
    # a recording-start ``render(request_idr=True)``).
    # env.close() automatically saves any unfinished recording tasks (blocking
    # until remux finishes).
    # Prerequisite: ``start_streaming`` must be called first.
    # Internally remux_range uses timestamp_ns as PTS time base (not fixed FPS),
    # and returns RemuxResult (with frame_indices / timestamps_ns frame-number
    # ↔ physical-index mapping).

def set_render_fps(fps: int) -> None
    # Sets the render frame rate (render FPS), controlling how often render()
    # invokes the engine: with sync_render=True one frame is rendered every
    # 1/fps physics steps; with sync_render=False one frame every 1/fps seconds.

def set_sync_render(enabled: bool) -> None
    # Enables/disables synchronous rendering. Enable it (enabled=True) when
    # recording for frame alignment, so render() invokes the engine every physics
    # step and forwards simulate_index.

def set_video_recorder_manager(manager: VideoRecorderManager | None) -> None
    # Injects a VideoRecorderManager instance. The environment layer forwards all
    # camera property query/set and recording operations to this manager. When
    # None, the next camera/recording call lazily creates it via
    # CreateVideoRecorderManager(self.stub, self.loop).
```

Underlying ``VideoRecorderManager`` unified interface (``orca_gym.recorder`` module).
Camera property query/set and the streaming state machine are implemented centrally
by ``VideoRecorderManager``; the environment layers (``OrcaGymLocalEnv`` /
``OrcaGymEulerEnv``) only forward calls. It directly implements them on the
injected gRPC capability stub (``GrpcServiceStub``).

```python
from orca_gym.recorder import CreateVideoRecorderManager, RemuxResult
from concurrent.futures import Future

# stub is the gRPC capability stub (GrpcServiceStub), providing interfaces for
# camera property query/set + streaming state machine; it may be None (recording
# only, no camera configuration).
# loop is the owning env's event loop (self.loop), used to bridge the stub's
# async interfaces synchronously.
manager = CreateVideoRecorderManager(stub=env.stub, loop=env.loop)
manager.start_recorder(camera_name, color_port=7070)      # idempotent start
future: Future[RemuxResult] = manager.save_streaming(
    camera_name, file_path, start_idx, end_idx
)  # register a range-save task, non-blocking
result: RemuxResult = future.result()                    # wait for save to finish (optional)
# result.file_path / result.frame_count / result.frame_indices / result.timestamps_ns
manager.stop_all_and_save() -> dict[str, RemuxResult]    # env.close() auto-save (blocking wait)
```

> Task queue abstraction: each save task (``RecordingTask``) in the waiting queue
> independently carries a trigger callback (``trigger_fn``) and execution logic
> (``execute``). The trigger condition is a callback
> ``(task, current_simulate_index) -> bool``, polled per-frame by the receiver
> thread, making it easy to extend new task types (e.g. triggered by timestamp
> or frame count).

### Real-time video visualization (``VideoStreamViewer``)

``VideoRecorderManager`` provides real-time visualization: it launches an
**independent subprocess** that establishes its own WebSocket connection to
receive the H.264 stream, decodes it, and renders it with matplotlib. The
subprocess is fully decoupled from the main process — it does not read the
recorder's rolling buffer, and never blocks the receiver thread, the save
worker, or the main simulation thread.

```python
from orca_gym.recorder import CreateVideoRecorderManager, VideoStreamViewer

manager = CreateVideoRecorderManager(stub=env.stub, loop=env.loop)
manager.start_recorder(camera_name, color_port=7070)   # ensure the recorder is running

# Non-blocking start of the visualization window (subprocess connects to
# WebSocket and renders with matplotlib)
viewer: VideoStreamViewer = manager.start_viewer(camera_name, window_name=None)
viewer.is_running

manager.get_viewer(camera_name)        # get the viewer (None if not started)
manager.stop_viewer(camera_name)       # stop one camera's window
manager.stop_all_viewers()             # stop all windows
manager.get_viewer_stats()             # stats for all windows
```

Using ``VideoStreamViewer`` standalone (without the manager):

```python
from orca_gym.recorder import VideoStreamViewer

viewer = VideoStreamViewer(recorder, window_name="Camera")
viewer.start()
# ... in the simulation loop ...
viewer.stop()
```

Requires ``matplotlib`` / ``numpy`` / ``av`` / ``websockets`` / ``opencv-python``.
Close the window via the window close button or by calling ``viewer.stop()`` /
``manager.stop_viewer()`` (internally signals the subprocess to exit via
``stop_event``).

### Camera Property Query/Set + Streaming State Machine

```python
def get_camera_names() -> list[str]
def get_camera_properties(camera_name: str) -> GetCameraPropertiesResponse
def set_camera_properties(
    camera_name: str,
    **kwargs,   # Optional parameters: capture_rgb, capture_depth, capture_normal, capture_object_color, random_object_color, use_nvenc, nvenc_gpu_index, width, height, vertical_fov, near_clip, far_clip, gamma, color_port, depth_port, use_dds, dds_topic, dds_stream_id
) -> None
def set_streaming_enabled(camera_name: str, enabled: bool) -> None
def make_camera_viewport_active(actor_name: str, entity_name: str) -> None
```

State machine constraints:
- `camera_name` can be enumerated via `get_camera_names()`
- `set_camera_properties` is only allowed in `Idle` state; in `Streaming` state, call `set_streaming_enabled(False)` first to return to `Idle` before setting properties
- After `set_streaming_enabled(True)` enters `Streaming` state, the corresponding ports (e.g., 7070/7071) start listening and streaming
- Client-side PyAV recording is controlled by `save_streaming`, orthogonal to this group of interfaces (but requires `set_streaming_enabled(True)` first)

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

> When `agent_id=None`, it defaults to the first agent (index 0).

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

`generate_action_space` automatically handles ±inf bounds (clamped to the float32 representable range) to avoid Gymnasium overflow warnings.

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

### Abstract Methods (subclasses must implement these)

```python
def step(action: np.ndarray) -> tuple[ObsType, float, bool, bool, dict]
def reset_model() -> tuple[dict, dict]
def _get_obs() -> np.ndarray | dict
```

### Lifecycle Methods

```python
def initialize_grpc()
def initialize_simulation()     # Load the model
def reset_simulation()          # Reset state
def init_qpos_qvel()            # Cache the initial state
def set_time_step(time_step)    # Set the time step
def pause_simulation()
def close()                     # Close the connection
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
        return 0.0  # Replace with your own reward function

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

Vectorized environment, running multiple environments in parallel.

```python
class OrcaGymVectorEnv(VectorEnv):
    def __init__(self, num_envs: int, worker_index: int, entry_point: str, **kwargs)
    def step(actions) -> tuple[obs, rewards, terminations, truncations, infos]
    def reset(*, seed=None, options=None) -> tuple[obs, infos]
```

---

## RewardType

Module path: `from orca_gym.environment.orca_gym_env import RewardType`

```python
class RewardType:
    SPARSE = "sparse"
    DENSE = "dense"
```
