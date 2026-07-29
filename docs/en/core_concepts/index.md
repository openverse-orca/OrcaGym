# 🧠 Core Concepts

Understanding the key concepts in OrcaGym will help you use the environment more effectively.

## Key Object Relationships

When using OrcaGym, you interact with the simulation world through the environment object:

```
env (Gymnasium Environment)
  ├── model (OrcaGymModel)    — Model information (structure, immutable)
  ├── data  (OrcaGymDataView) — Simulation state (dynamic, changes every step)
  └── sim_config              — Simulation parameter configuration
```

---

## Concept Quick Reference

| Concept | Description |
|------|------|
| **Model** | Static model information (geometry, joints, actuators, sensors) |
| **Data** | Dynamic simulation state (position, velocity, acceleration, time) |
| **SimConfig** | Simulation parameters (time step, solver settings, gravity) |
| **qpos** | Generalized coordinates (position), length `nq` |
| **qvel** | Generalized velocity, length `nv` |
| **ctrl** | Control input, length `nu` |
| **Equality Constraints** | Connection constraints used for object manipulation |
| **Mocap Body** | A special body that can be "controlled" by setting its pose |

---

## Simulation = Model + Data

OrcaGym divides the simulation world into two parts:

| Concept | Type | Analogy | Example |
|------|------|------|------|
| `env.model` | `OrcaGymModel` | The robot's **manual** (never changes) | How many joints there are, what each joint is called |
| `env.data` | `OrcaGymDataView` | The robot's **current state** (changes every step) | How many degrees each joint has rotated, what its velocity is |

```python
# model — static, describes the structure
print(env.model.nq)            # Number of position variables
print(env.model.nv)            # Number of velocity variables
print(env.model.nu)            # Number of control dimensions

# data — dynamic, reflects the current state
print(env.data.qpos)           # Current position → changes after each simulation step
print(env.data.qvel)           # Current velocity
print(env.data.time)           # Simulation time
```

---

## Simulation Time

```
time_step  = 0.001 seconds    ← Time per physics engine step
frame_skip = 20               ← Number of physics steps per step() call
dt = 0.001 × 20 = 0.02 seconds ← How often your control command updates (50Hz)
```

Control frequency: `control_hz = 1.0 / dt`

---

## Dimension Conventions

| Variable | Length | Meaning |
|------|------|------|
| `model.nq` | Number of generalized coordinates | qpos length |
| `model.nv` | Number of degrees of freedom | qvel length |
| `model.nu` | Number of actuators | Control input length |

---

## Joint Types and qpos/qvel Dimensions

Different joint types occupy different numbers of elements in `qpos` and `qvel`:

| Joint Type | qpos Size | qvel Size | Example |
|----------|-----------|-----------|------|
| FREE | 7 (3 pos + 4 quat) | 6 (3 lin + 3 ang) | Free-flying body |
| BALL | 4 (quaternion) | 3 (angular velocity) | Ball joint |
| HINGE | 1 (angle) | 1 (angular velocity) | Revolute joint |
| SLIDE | 1 (displacement) | 1 (linear velocity) | Sliding joint |

---

## Suggested Reading Order

1. [Model / Data / Config](model-data-opt.md) — Understand the three core data objects
2. [Gymnasium Interface](gym-interface.md) — Understand the standard RL interface
3. [Data Flow](data-flow.md) — Understand how data flows through the simulation
4. [System Architecture](architecture.md) — Understand the overall layered design, API boundaries, component design, encapsulation isolation, and the migration guide
