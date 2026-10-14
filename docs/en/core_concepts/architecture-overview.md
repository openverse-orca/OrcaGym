# 🏗️ Architecture Overview

This document describes the overall architecture of the `OrcaGymEulerEnv` + `OrcaGymEuler` system from a layered perspective, clarifying the responsibilities and API boundaries of each layer to help developers determine "which layer to develop in" and "which layer to maintain."

For component design details, API contracts, and encapsulation isolation mechanisms, see [architecture.md](architecture.md).

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  User Code                                                       │
│  Business environment subclasses, task definitions,              │
│  reward functions, observation construction                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Inherit OrcaGymEulerEnv, use env.data / env.sim_config
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  RL Training Framework (RSL-RL / SB3)                            │
│  Policy training, rollout scheduling, obs / action / reward flow │
└───────────────────────────┬─────────────────────────────────────┘
                            │ env.reset() / env.step() / env.do_simulation()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Environment Layer: OrcaGymEulerEnv                              │
│  gym.Env implementation, public API contract, OrcaGymEnvMixin   │
│  .data / .model / .sim_config / .apply_body_force() / .query_*()│
└───────────────────────────┬─────────────────────────────────────┘
                            │ Composition (not inheritance), delegates to _gym
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Simulation Core Layer: OrcaGymEuler (Facade)                    │
│  MuJoCoSimCore / ModelRegistry / SimConfig                      │
│  Selects one of two mutually exclusive backends at runtime       │
└───────┬───────────────────────────────────────┬─────────────────┘
        │                                       │
        │  backend="mujoco"                     │  backend="euler"
        ▼                                       ▼
┌───────────────────────────────┐  ┌──────────────────────────────┐
│  MuJoCo Backend (CPU)          │  │  Euler Backend (GPU)         │
│  MjModel / MjData / mj_step    │  │  Euler engine (autonomous)   │
│  opt.* solver parameters       │  │  Exposes MuJoCo-style API    │
│  Pure MuJoCo, no orchestration │  │  D2H extraction (qpos/xpos)  │
└───────────────┬───────────────┘  └──────────────┬───────────────┘
                │                                 │
                │ sync_to_view()                  │ D2H extraction (qpos/xpos etc.)
                ▼                                 ▼
        ┌──────────────────────────────────────────────┐
        │  OrcaGymDataView (unified state view)        │
        │  env.data reads are consistent,              │
        │  backend differences are shielded            │
        └──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  External Renderer (optional, bypass system)                     │
│  Consumes qpos / sim_time snapshots,                             │
│  does not participate in the physics stepping main path          │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Project | Python Package | Responsibility |
|------|------|----------|------|
| User Code | Business repo | — | Business environment subclasses, reward functions, observation construction |
| RL Training Framework | RSL-RL / SB3 | — | Policy training, rollout scheduling |
| Environment Layer | OrcaGym | `orca_gym` | gym.Env implementation, public API contract, MuJoCo semantic interface |
| Simulation Core Layer | OrcaGym | `orca_gym` | Simulation core Facade + backend selection and delegation |
| MuJoCo Backend | MuJoCo | `mujoco` | CPU rigid-body dynamics solving (open-source standard) |
| Euler Backend | Euler | — | GPU physics simulation (autonomous engine) |
| External Renderer | OrcaStudio / OrcaLab | — | Rendering, scene sync, interaction (bypass system, optional) |

### Mutually Exclusive Dual-Backend Selection

OrcaGym, as a simulation framework, integrates two **independent** physics backends and selects one at runtime:

- **MuJoCo Backend** (open-source standard): Pure CPU MuJoCo rigid-body simulation, no orchestration or coupling.
- **Euler Backend** (Orca team in-house): Euler operates as a complete physics engine autonomously. OrcaGym drives simulation through the MuJoCo-style API provided by Euler, and extracts `qpos`/`xpos` and other data to CPU via the D2H interface for rendering.

The two paths are mutually exclusive: loading one backend does not involve the other. Backend selection is transparent to user code; public APIs such as `env.data` / `env.do_simulation()` behave consistently.

> **Current implementation status**: The `_euler` field is a placeholder (always `None`), `has_euler()` always returns `False`, and only the MuJoCo backend is currently available. Euler backend integration will be implemented in a future version.

> **External renderer is a bypass system**: It only consumes `qpos`/`sim_time` snapshots and does not participate in the physics stepping main path. The environment can still step normally when the renderer is absent.

---

## API Layers and Role Definitions

### User Development Layer

User code interacts only with the following two API layers and **must not penetrate into lower-level internal objects** (`_mjModel`/`_mjData`/`_sim`, etc.):

| API | Source | Purpose |
|-----|------|------|
| `env.data` | `OrcaGymDataView` | Read state such as `qpos`/`qvel`/`body_xpos(name)` |
| `env.model` | `OrcaGymModel` | Query the model structure (dimensions, name mappings) |
| `env.sim_config` | `SimConfig` | Configure timestep / integrator / iterations / gravity |
| `env.ctrl` | `np.ndarray` | Set the control input |
| `env.do_simulation(ctrl, n)` | — | Simulation stepping |
| `env.set_joint_qpos()` / `env.apply_body_force()` / `env.clear_body_force()` | — | State writing, external force injection |
| `env.body()` / `env.joint()` / `env.actuator()` / `env.site()` | `OrcaGymEnvMixin` | Namespace resolution (automatically adds the agent prefix) |
| `env.render()` | — | External renderer interaction |
| `gym.Env` standard interface | Gymnasium | `reset()` / `step()` / `observation_space` / `action_space` |

**User Development Paradigm**:

```python
class MyTaskEnv(OrcaGymEulerEnv):
    def _get_obs(self) -> dict:
        # ✅ Use public API
        return {
            "qpos": self.data.qpos.copy(),
            "body_pos": self.data.body_xpos("link1"),
        }

    def compute_reward(self) -> float:
        # ✅ Use public API
        return float(self.data.body_xpos("target")[2])

    def _apply_disturbance(self):
        # ✅ Use public API
        self.apply_body_force("link1", force=[0, 0, 10], torque=[0, 0, 0])
```

### Developer Maintenance Layer

| Layer | Maintained By | Maintenance Content |
|------|--------|---------|
| **Environment Layer** `OrcaGymEulerEnv` | OrcaGym Team | gym.Env implementation, public API contract, Mixin public methods |
| **Simulation Core Layer** `OrcaGymEuler` and subcomponents | OrcaGym Team | Facade delegation, `MuJoCoSimCore` / `ModelRegistry` / `SimConfig`, backend selection |
| **MuJoCo Backend** MuJoCo | Upstream | `mujoco` library |
| **Euler Backend** Euler | Euler Team | Autonomous physics engine, exposes MuJoCo-style API |
| **External Renderer** OrcaStudio / OrcaLab | respective teams | Renderer, interaction logic |

**Developer Extension Principle**: When the public API does not meet user needs, add public methods in `OrcaGymEulerEnv` (delegating to `_gym` public API), or add field accessors in `OrcaGymDataView`. **Do not guide users to bypass the wall and access internal objects.**

---

## Key Call Flows

### step Main Path

```
User Code / RL Framework
    │ env.step(action)  or  env.do_simulation(ctrl, n_frames)
    ▼
OrcaGymEulerEnv
    │ delegates _gym.do_simulation()
    ▼
OrcaGymEuler
    │ delegates stepping to the selected backend
    ▼
┌─────────────────────────────────────────────────────┐
│  MuJoCo Backend             │  Euler Backend         │
│  _sim.set_ctrl()            │  Euler-driven stepping │
│  _sim.step(nstep)           │  (physics autonomous)  │
│  mj_step × nstep            │                        │
└─────────────────────────────────────────────────────┘
    │
    │ State sync to view (MuJoCo: sync_to_view / Euler: D2H extraction)
    ▼
OrcaGymDataView  ←── env.data reads are consistent
```

### Rendering Bypass

```
User Code
    │ env.render()
    ▼
OrcaGymEulerEnv
    │ delegates renderer to consume state snapshot
    ▼
External renderer (independent process/machine)
    │ Scene sync, rendering, video frame capture
```

The rendering path is **completely decoupled** from the physics stepping path: the renderer only consumes `qpos`/`sim_time` snapshots and does not touch physics stepping.

### State Writing and External Force Injection

```
User Code
    │ env.apply_body_force(name, force, torque)
    ▼
OrcaGymEulerEnv
    │ delegates _gym.apply_body_force()
    ▼
OrcaGymEuler
    │ writes external force via the selected backend (mechanism is backend-specific)
    ▼
Backend internal (invisible to users)
```

External force injection is **explicit and traceable**: injected via the public API, with the force application mechanism handled internally by the backend.

---

## Encapsulation Boundary

```
User-Visible                  │  User-Invisible (Internal)
─────────────────────────────┼──────────────────────────────────
env.data (DataView)           │  env._gym
env.model (OrcaGymModel)      │  env._gym._sim
env.sim_config (SimConfig)    │  env._gym._sim._mjModel / _mjData
env.ctrl                      │  env._gym._studio
env.do_simulation()           │  env._gym._registry
env.apply_body_force()        │  env._gym._euler
env.query_*()                 │  env._gym._opt
env.body() / joint() / ...    │
```

- **Left column**: Public API (L1), intended for users and AI to use, visible in IDE autocompletion
- **Right column**: Internal components (L2/L3), multi-layer isolation: `OrcaGymEuler.__getattribute__` actively intercepts access (accessing `_sim`/`_studio`/`_mjData`, etc. directly raises `AttributeError`), supplemented by the `_` prefix convention + ruff SLF001 static checking + AGENTS.md constraints, prohibiting external access

For detailed contracts and isolation mechanisms, see [architecture.md](architecture.md) sections 6-7.
