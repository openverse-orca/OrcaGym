# 🔌 Hello World — Running Your First Simulation

Goal: Run a minimal simulation loop in 5 minutes (no OrcaStudio required, offline mode).

---

## Prerequisites

- OrcaGym installed (you can `from orca_gym.environment.euler import OrcaGymEulerEnv`)
- A simple MuJoCo XML scene file (this section uses a built-in example)

---

## Complete Code

Below is a **runnable** minimal example. Save it as `hello_orcagym.py`:

```python
"""
hello_orcagym.py — OrcaGym minimal example

Function: Create an environment -> drive with random actions for 200 steps -> close
Prerequisite: No OrcaStudio required (offline mode, skip_grpc_load=True)
"""
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class HelloEnv(OrcaGymEulerEnv):
    """Minimal environment: drive simulation with random actions, print state."""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            skip_grpc_load=True,  # offline mode, no Studio required
            **kwargs,
        )
        # Action space = Box
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return self.data.qpos.copy().astype(np.float32)


# ============================================================
# Usage
# ============================================================
if __name__ == "__main__":
    # Scene XML path (replace with your local scene file)
    SCENE_XML = "/path/to/your/scene.xml"

    print("=" * 60)
    print("Hello OrcaGym — Offline Mode Minimal Example")
    print("=" * 60)

    # 1. Create environment (offline mode, loads local XML directly)
    env = HelloEnv(model_xml_path=SCENE_XML, frame_skip=5, time_step=0.002)
    print(f"[1/4] Environment created: nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")

    # 2. Verify state access
    print(f"[2/4] State access: qpos.shape={env.data.qpos.shape}, time={env.data.time:.4f}")

    # 3. reset
    obs, info = env.reset()
    print(f"[3/4] reset successful: obs.shape={obs.shape}")

    # 4. Step loop (random actions, 200 steps)
    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (step + 1) % 50 == 0:
            print(f"[4/4] step {step + 1}/200: obs={obs}, reward={reward:.4f}")

    print(f"[4/4] Step loop complete: total reward={total_reward:.4f}")
    env.close()
    print("=" * 60)
    print("Hello OrcaGym complete!")
```

Run:

```bash
python hello_orcagym.py
```

---

## Line-by-Line Explanation

### Environment Construction

```python
class HelloEnv(OrcaGymEulerEnv):
    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=5,                # each step() advances 5 physics steps
            orcagym_addr="localhost:50051",  # gRPC address (not needed in offline mode)
            agent_names=["agent0"],      # list of agent names
            time_step=0.002,             # each physics step is 0.002 seconds
            model_xml_path=model_xml_path,   # local MuJoCo XML scene path
            skip_grpc_load=True,         # True = offline mode
        )
```

Key parameters:
- `model_xml_path`: Path to the local MuJoCo XML scene file. In offline mode, the scene is loaded directly from this path.
- `skip_grpc_load=True`: Skip the gRPC connection, pure local MuJoCo simulation.
- `frame_skip=5`: The physics engine advances 5 steps per `step()` call. Control frequency = 1 / (time_step * frame_skip) = 100 Hz.
- `time_step=0.002`: Each physics step is 0.002 seconds.

### Three Abstract Methods You Must Implement

```python
def step(self, action):       # Execute one simulation step -> returns (obs, reward, terminated, truncated, info)
def reset_model(self):        # Reset to initial state -> returns (obs, info)
def _get_obs(self) -> dict | np.ndarray:  # Collect observation data
```

### Core Loop

```python
obs, info = env.reset()                                                  # Return to initial state
obs, reward, terminated, truncated, info = env.step(action)              # Advance one step
```

| Variable | Meaning | Type |
|----------|---------|------|
| `obs` | Observation data (returned by `_get_obs()`) | `np.ndarray` or `dict` |
| `reward` | Reward | `float` |
| `terminated` | Whether the task is complete/failed (e.g., fell over) | `bool` |
| `truncated` | Whether truncated due to timeout (reached max_episode_steps) | `bool` |
| `info` | Additional debug information | `dict` |

### Offline vs. Online Mode

| Mode | `skip_grpc_load` | Studio Required | Use Case |
|------|------------------|-----------------|----------|
| Offline (recommended for beginners) | `True` | No | Training, quick testing, local development |
| Online | `False` (default) | Yes | Visualization, human observation, video recording |

---

## FAQ

### `FileNotFoundError: model XML not found`

**Cause**: The XML file pointed to by `model_xml_path` does not exist.

**Solution**: Verify the file path is correct. You can obtain example scene files from [OrcaPlayground](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler).

### `ModuleNotFoundError: No module named 'orca_gym'`

**Cause**: OrcaGym is not installed.

**Solution**: Follow the [Installation Guide](../getting-started/installation.md) to install.

### `env.render()` does not work in offline mode?

In offline mode, `render()` is a no-op (because no Studio is connected). For visualization, use online mode (`skip_grpc_load=False`) and launch OrcaStudio.

---

## Next Step

You have run your first minimal simulation! Next, learn how to **put things in the scene**: [🎬 Scene Setup](scene-setup.md).
