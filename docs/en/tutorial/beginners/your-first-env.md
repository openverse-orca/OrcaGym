# 🏗️ Your First Environment — Writing Your Own Environment Class

In the previous section, we ran a simulation loop with a minimal environment. In this section, you will learn how to **write a complete environment class** to control the simulation.

## Why Write an Environment Class?

To control the simulation (physics stepping, reading state), you need to inherit from an environment base class.

**Use `OrcaGymEulerEnv`** (the new Euler main path). `OrcaGymLocalEnv` is in maintenance mode.

An environment class = the "driver" for the scene:

```
MuJoCo XML -> OrcaGymEulerEnv -> Drive simulation (loop execution)
```

## Minimal Environment Skeleton (recommended)

An environment class needs to implement **3 core methods**:

```
__init__()        — initialization (set action_space, observation_space)
step(action)      — execute one simulation step, returns (obs, reward, terminated, truncated, info)
reset_model()     — reset to initial state
_get_obs()        — collect observation data
```

> **Key difference from older RL environments**: `step()` returns a 5-tuple (Gymnasium standard), not the old 4-tuple.
> `terminated` indicates task completion/failure (e.g., robot fell over), `truncated` indicates timeout truncation (reached max_episode_steps).

Below is a **runnable** complete environment (offline mode, no Studio needed):

```python
"""
my_first_env.py — A minimal custom environment.
Simplified from [OrcaPlayground examples/euler/01_hello_euler/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/01_hello_euler.
"""

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class MyFirstEnv(OrcaGymEulerEnv):
    """Minimal environment: observation = joint position + velocity, action = torque control, reward = 0."""

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

    def __init__(self, model_xml_path, **kwargs):
        # -- Parent class initialization (autonomous lifecycle orchestration) --
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            skip_grpc_load=kwargs.pop("skip_grpc_load", True),  # default offline
            **kwargs,
        )

        # -- Action space --
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # -- Observation space --
        obs_sample = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32
        )

    # -- Observation acquisition --
    def _get_obs(self) -> np.ndarray:
        """Return current observation. Zero-copy direct read from data."""
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
        ]).astype(np.float32)

    # -- Simulation stepping --
    def step(self, action: np.ndarray):
        """
        Execute one simulation step.
        action: shape (nu,), each value in [-1, 1]
        Returns: (obs, reward, terminated, truncated, info) — Gymnasium 5-tuple
        """
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)

        # 1. Execute simulation: do_simulation internally auto-syncs data
        self.do_simulation(action, self.frame_skip)

        # 2. Get new observation
        obs = self._get_obs()

        # 3. Reward & termination condition
        reward = 0.0  # always 0 here (replace with your reward function)
        terminated = False  # never terminates here (replace with your termination condition)
        truncated = False
        info = {"time": float(self.data.time)}

        return obs, reward, terminated, truncated, info

    # -- Reset --
    def reset_model(self):
        """Return to the initial state."""
        # Canonical approach: set state via set_joint_qpos / set_joint_qvel
        self.set_joint_qpos(self.init_qpos)  # restore initial qpos defined in XML
        self.set_joint_qvel(self.init_qvel)  # restore initial qvel defined in XML
        self.mj_forward()   # update derived quantities (body pose, sensors, etc.)
        self._sync_view()   # sync to DataView (env.data)
        return self._get_obs(), {}


# ============================================================
# Usage
# ============================================================
if __name__ == "__main__":
    SCENE_XML = "/path/to/your/scene.xml"

    env = MyFirstEnv(model_xml_path=SCENE_XML)
    obs, _ = env.reset()
    print(f"nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")
    print(f"obs.shape={obs.shape}")

    for i in range(10):
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f" Step {i}: reward={reward:.3f}, time={_:.4f}s")

    env.close()
```

## Core Concepts Explained

### `do_simulation` — One-Stop Simulation Stepping

```python
self.do_simulation(ctrl, self.frame_skip)
```

This single line is equivalent to:

```python
# Internally delegates to self._gym public methods:
# _gym.step_with_coupling(ctrl, n_frames, dt)
#   -> _sim.set_ctrl(ctrl) + _sim.step(n_frames)
# _gym.sync_to_view()
#   -> data is automatically synced to the latest state
```

> **Key advantage**: After `do_simulation()` returns, `self.data` is already updated automatically — no need to call `update_data()` manually.

### `self.data` — Complete State Read-Only View

`self.data` is `OrcaGymDataView`, providing a zero-copy read-only view:

| Attribute | Meaning | Shape |
|-----------|---------|-------|
| `self.data.qpos` | Generalized positions | `(nq,)` |
| `self.data.qvel` | Generalized velocities | `(nv,)` |
| `self.data.qacc` | Generalized accelerations | `(nv,)` |
| `self.data.time` | Simulation time | scalar |
| `self.data.xfrc_applied` | External forces (read-only) | `(nbody, 6)` |

> ⚠️ `self.data.qpos` is a zero-copy view — just read it directly. If you need to save historical values, call `.copy()`.

### `self.model` — Model Structure Queries

| Attribute/Method | Meaning |
|------------------|---------|
| `self.model.nq` | Generalized coordinate dimension |
| `self.model.nv` | Generalized velocity dimension |
| `self.model.nu` | Number of actuators |
| `self.model.body_name2id(name)` | Body name -> ID |
| `self.model.joint_name2id(name)` | Joint name -> ID |
| `self.model.get_joint_dict()` | Dictionary of all joint info |
| `self.model.get_geom_dict()` | Dictionary of all geometry info |

### State Writing

After modifying state, you **must call `mj_forward()`**; otherwise, derived quantities (body poses, sensors, etc.) will not be updated:

```python
# ✅ Correct
qpos = self.data.qpos.copy()
qpos[0] = 0.5  # modify joint 0 angle
self.set_joint_qpos(qpos)   # canonical write
self.mj_forward()            # <- required! Update derived quantities
self._sync_view()            # sync to DataView

# ❌ Wrong: directly writing data.qpos (read-only view)
# self.data.qpos[0] = 0.5  # violates encapsulation contract

# ❌ Wrong: missing mj_forward
# self.set_joint_qpos(qpos)
# # body_xpos will read the old value at this point
```

### Standard `reset_model` Pattern

```python
def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
    qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
    self.set_joint_qpos(qpos)   # canonical write
    self.set_joint_qvel(qvel)   # canonical write
    self.mj_forward()           # update derived quantities
    self._sync_view()           # sync DataView
    return self._get_obs(), {}
```

- `self.init_qpos` / `self.init_qvel`: initial state cached by the parent class after `initialize_simulation()`
- `self.np_random`: random number generator created by the parent class via `set_seed_value()`

### Environment Lifecycle

```
MyFirstEnv(model_xml_path=..., skip_grpc_load=True)
 └── OrcaGymEulerEnv.__init__()
     ├── initialize_grpc()          # skip in offline mode
     ├── pause_simulation()
     ├── set_time_step(time_step)
     ├── initialize_simulation()    # load model_xml -> create MuJoCo instance
     ├── reset_simulation()         # reset_data + sync_to_view
     └── init_qpos_qvel()           # cache init_qpos / init_qvel

env.reset() [from OrcaGymEnvMixin]
 ├── reset_simulation() -> restore initial state
 └── reset_model() -> your custom reset logic

env.step(action) <- repeat N times
 ├── do_simulation(ctrl, frame_skip)   # step + auto sync
 ├── _get_obs()
 └── return (obs, reward, terminated, truncated, info)

env.close()
 └── close gRPC channel (no-op in offline mode)
```

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Action dimension mismatch` | `action.shape` != `(nu,)` | Check `action.reshape(env.model.nu)` |
| Observation data is "wrong" | Reading data without `mj_forward()` first | Confirm `mj_forward()` is called in `reset_model` |
| Observation is all NaN | Read derived quantities before `mj_forward()` | Use `do_simulation()` instead of manual operations |
| `AttributeError: 'OrcaGymEulerEnv' object has no attribute 'gym'` | Using old API | There is no `env.gym`; use `env.data` / `env.model` |
| `TypeError: step() returns 5 values` | Old code unpacking only 4 values | Gymnasium standard: `obs, reward, terminated, truncated, info = env.step(action)` |

## Next Step

The environment class is written. Next, learn about **simulation state data layout and synchronization rules**: [📐 State Management](state-management.md).
