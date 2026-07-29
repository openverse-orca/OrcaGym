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
│  MuJoCoSimCore / OrcaStudioBridge / ModelRegistry /             │
│  SimConfig / EulerOrchestrator                                  │
└──────┬─────────────────────────────────────────────┬────────────┘
       │                                             │
       │ mj_step / mj_forward                        │ gRPC communication
       ▼                                             ▼
┌─────────────────────────────────┐  ┌─────────────────────────────┐
│  MuJoCo Runtime (rigid-body     │  │  OrcaStudio System           │
│  solver)                        │  │  Rendering, scene sync,      │
│  MjModel / MjData / mj_step     │  │  video saving                │
│  opt.* solver parameters        │  │  Object manipulation,        │
│                                 │  │  camera control              │
└─────────────────────────────────┘  └─────────────────────────────┘
       │
       │ External force coupling / sync cycle (SyncCycleConfig)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Engine Layer: Euler Runtime (orca.euler)                        │
│  Multi-physics simulation, Model / State / Control,              │
│  solver scheduling, zero-copy coupling                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ import orca.flow
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Framework Layer: OrcaFlow (orca.flow)                           │
│  GPU programming framework, multi-backend compilation,           │
│  flow.kernel / flow.array                                       │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Project | Python Package | Responsibility |
|------|------|----------|------|
| User Code | Business repo | — | Business environment subclasses, reward functions, observation construction |
| RL Training Framework | RSL-RL / SB3 | — | Policy training, rollout scheduling |
| Environment Layer | OrcaGym | `orca_gym` | gym.Env implementation, public API contract, MuJoCo semantic interface |
| Simulation Core Layer | OrcaGym | `orca_gym` | Simulation core Facade + subcomponent orchestration |
| Rigid-Body Runtime | MuJoCo | `mujoco` | Rigid-body dynamics solving |
| Non-Rigid-Body Runtime | Euler | `orca.euler` | Multi-physics simulation, solver scheduling, zero-copy coupling |
| Studio System | OrcaStudio | — | Rendering, scene sync, interaction (bypass system) |
| Framework Layer | OrcaFlow | `orca.flow` | GPU programming framework, multi-backend compilation |

> **OrcaStudio is a bypass system**: It communicates with the simulation core via gRPC, does not participate in the `mj_step` main path, and does not affect the physics simulation. The environment can still step normally when Studio is absent.

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
| `env.render()` / `env.begin_save_video()` | — | Studio interaction |
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
| **Simulation Core Layer** `OrcaGymEuler` and subcomponents | OrcaGym Team | Facade orchestration, `MuJoCoSimCore` / `OrcaStudioBridge` / `ModelRegistry` / `SimConfig` / `EulerOrchestrator` |
| **Rigid-Body Runtime** MuJoCo | Upstream | `mujoco` library |
| **Non-Rigid-Body Runtime** Euler | Euler Team | Model/State/Control, solver, coupling orchestration |
| **Framework Layer** OrcaFlow | Flow Team | GPU kernel compilation, multi-backend scheduling |
| **Studio System** OrcaStudio | Studio Team | Renderer, gRPC service, interaction logic |

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
    │ _sim.set_ctrl() → _sim.step(nstep)
    ▼
MuJoCoSimCore
    │ mj_step × nstep
    ▼
MuJoCo Runtime  ←── (when EulerOrchestrator enabled) ── Euler Runtime / OrcaFlow
    │
    │ _sim.sync_to_view()
    ▼
OrcaGymDataView  ←── env.data reads are consistent
```

### Rendering Bypass

```
User Code
    │ env.render()
    ▼
OrcaGymEulerEnv
    │ delegates _studio.render(qpos, sim_time)
    ▼
OrcaStudioBridge  ──gRPC──►  OrcaStudio System (independent process/machine)
                                    │ Scene sync, rendering, video frame capture
```

The rendering path is **completely decoupled** from the physics stepping path: Studio only consumes `qpos`/`sim_time` snapshots and does not touch `mj_step`.

### State Writing and External Force Injection

```
User Code
    │ env.apply_body_force(name, force, torque)
    ▼
OrcaGymEulerEnv
    │ delegates _gym.apply_body_force()
    ▼
OrcaGymEuler
    │ _sim.apply_body_force(body_id, force, torque)
    │ (optional) _euler.notify_external_force(...)
    ▼
MuJoCoSimCore
    │ Write to xfrc_applied (internal detail, invisible to users)
```

External force injection is **explicit and traceable**: when `EulerOrchestrator` is enabled, it can perceive force injections, ensuring MuJoCo-Euler coupling consistency.

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
- **Right column**: Internal components (L2/L3), `_` prefix convention + ruff SLF001 static checking + AGENTS.md constraints prohibit external access

For detailed contracts and isolation mechanisms, see [architecture.md](architecture.md) sections 6-7.
