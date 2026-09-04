# 🎓 Beginners' Guide

Welcome to the OrcaGym beginners' tutorial! This series starts from **zero** and guides you through OrcaGym step by step.

---

## Who Is This For?

- **New to OrcaGym** — developers who want to get started from scratch
- **New to robot simulation** — those who want to understand the basic concepts of a simulation environment
- **Students or engineers with Python basics** — those who want to learn robot simulation

## Prerequisites

| Knowledge | Description |
|-----------|-------------|
| Python basics | Familiar with functions, classes, `import` |
| NumPy basics | Understand `np.array` creation and basic operations |
| Robotics basics | Understand concepts like "joint", "end-effector" (optional, explained in the tutorial) |

> **No RL (Reinforcement Learning) background required!** This tutorial focuses on simulation itself.

---

## Learning Path

We have designed a **progressive** learning path. Each step introduces only one new concept:

```
🔌 Hello World — Understand what a minimal simulation loop looks like
 │
 └── 🎬 Scene Setup — Learn to place robots, objects, and lights in a scene
 │
 └── 🔧 MuJoCo Backend — Model loading, stepping control, solver configuration
 │
 └── 🏗️ Your First Environment — Learn to write your own environment class
 │
 └── 📐 State Management — Understand qpos/qvel/qacc data layout and sync rules
 │
 └── 📡 Reading State — Learn to query joint angles and body poses by name
 │
 └── 🦾 Making the Robot Move — Understand qpos/qvel and control joints
 │
 └── 📷 Camera & Vision — Get RGB-D camera images
 │
 └── 🎮 Simple Controller — Write a PD controller
 │
 └── 🏆 Build a Task — Combine all knowledge to complete a reach task
 │
 └── 🧠 PPO Training — Train a controller with SB3 RL
```

| Chapter | New Concept | Estimated Time |
|---------|-------------|----------------|
| [🔌 Hello World](hello-world.md) | Offline mode, `do_simulation`, `data.qpos` | 5 min |
| [🎬 Scene Setup](scene-setup.md) | `OrcaGymScene`, `Actor`, asset placement | 15 min |
| [🔧 MuJoCo Backend](mujoco-backend.md) | Model loading, `sim_config`, stepping control | 15 min |
| [🏗️ Your First Environment](your-first-env.md) | Inherit `OrcaGymEulerEnv`, implement `step`/`reset_model`/`_get_obs` | 15 min |
| [📐 State Management](state-management.md) | qpos/qvel/qacc layout, `mj_forward` sync, zero-copy view | 15 min |
| [📡 Reading State](state-queries.md) | `query_joint_qpos`, `get_body_xpos_xmat_xquat`, sensors | 15 min |
| [🦾 Making the Robot Move](move-a-joint.md) | `qpos`/`qvel`, `set_joint_qpos`, `do_simulation` | 20 min |
| [📷 Camera & Vision](camera-and-vision.md) | `CameraWrapper`, RGB-D image acquisition | 15 min |
| [🎮 Simple Controller](simple-controller.md) | PD controller principles, parameter tuning | 20 min |
| [🏆 Build a Task](build-a-task.md) | Combine everything: perception -> decision -> control, complete a reach goal | 30 min |
| [🧠 PPO Training](ppo-training.md) | SB3 PPO, reward function design, offline training | 30 min |

---

## Key Concepts at a Glance

### Simulation = Model + Data

OrcaGym divides the simulation world into two parts:

| Concept | Type | Analogy | Example |
|---------|------|---------|---------|
| `env.model` | `OrcaGymModel` | The robot's **manual** (static) | How many joints, what each joint is named |
| `env.data` | `OrcaGymDataView` | The robot's **current state** (changes each step) | Joint angle, joint velocity |

```python
# model — static, describes structure
print(env.model.nq) # total number of position variables
print(env.model.nv) # total number of velocity variables
print(env.model.nu) # total number of actuators (control dimensions)

# data — dynamic, reflects current state (OrcaGymDataView, zero-copy read-only view)
print(env.data.qpos) # current position → changes with each simulation step
print(env.data.qvel) # current velocity
print(env.data.time) # simulation time
```

### Simulation Time

```
time_step = 0.001 seconds ← time per physics engine step (small, for accuracy)
frame_skip = 20 ← number of physics steps per step() call
dt = 0.001 * 20 = 0.02 seconds ← control command update interval (50Hz)
```

### Environment Class Hierarchy

```
gymnasium.Env
 └── OrcaGymEulerEnv # 👈 recommended (current main path)
 ├── Composes OrcaGymEuler simulation core
 ├── .data -> OrcaGymDataView (full state zero-copy read-only view)
 ├── .model -> OrcaGymModel (model structure info)
 ├── .sim_config -> SimConfig (solver configuration)
 └── .ctrl -> np.ndarray (current control input)

 └── OrcaGymLocalEnv # maintenance mode, not recommended for new projects
```

> **Recommendation**: `OrcaGymEulerEnv` is the recommended entry point for new projects. `OrcaGymLocalEnv` is in maintenance mode and is being phased out.
> See [OrcaPlayground examples/euler/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler) for runnable examples.

---

## Next Step

Start with [🔌 Hello World](hello-world.md) and run your first simulation in 5 minutes!
