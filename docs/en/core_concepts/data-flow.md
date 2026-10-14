# 🔄 Data Flow

Understanding how simulation data flows through OrcaGym is key to using the environment correctly.

## Core Data Flow

```
           RL Policy
               │
               ▼ action (nu,)
    ┌──────────────────────┐
    │  OrcaGymEulerEnv      │
    │   ┌────────────────┐  │
    │   │ do_simulation() │  │  ← Core stepping method
    │   │   ├─ Set control │  │
    │   │   ├─ Physics step│  │
    │   │   └─ Sync state  │  │
    │   └───────┬────────┘  │
    │           ▼            │
    │   Simulation state     │
    │   qpos/qvel/time/...   │
    │           │            │
    │   ┌───────▼────────┐  │
    │   │ _get_obs()     │──┼──▶ obs
    │   │ subclass step() │──┼──▶ reward / terminated / truncated
    │   └────────────────┘  │
    └──────────────────────┘
               │
               ▼ obs
           RL Policy
```

## What Happens Inside step()

```python
# When you call env.step(action):
env.step(action)
  │
  ├─▶ do_simulation(ctrl, n_frames)
  │     ├─▶ Set control input ctrl
  │     └─▶ Execute n_frames physics simulation steps
  │
  ├─▶ _get_obs()          # Build the observation
  ├─▶ Compute reward / terminated / truncated  # Implemented by the subclass within its own step() logic
  │
  └─▶ Return (obs, reward, terminated, truncated, info)
```

> Note: `compute_reward()` is not a public method of `OrcaGymEulerEnv`. The computation of reward and termination conditions is done by subclasses within their own `step()` implementation (`OrcaGymEulerEnv.step` is an abstract method that subclasses must override).

## State Synchronization Rules

After modifying the simulation state, certain operations must be performed to ensure data consistency:

| Modification Action | Required Operation |
|----------|----------|
| `set_joint_qpos()` | `mj_forward()` |
| `set_joint_qvel()` | `mj_forward()` |
| `set_mocap_pos_and_quat()` | `mj_forward()` |

> ⚠️ **Important**: After `do_simulation()` returns, the data is already automatically synchronized. You only need to manually call `mj_forward()` when you manually modify the state.

## Why mj_forward() Is Needed After Modifying qpos

MuJoCo has many "derived quantities" (body poses, sensor values, contact forces, etc.) that need `mj_forward` to be refreshed. Only modifying `qpos` without calling `forward` leads to:

- Body poses being NaN or stale values
- Incorrect sensor data
- Inconsistent contact state

```python
# ✅ Correct
env.set_joint_qpos(qpos)
env.mj_forward()          # Refresh all derived quantities

# Now it is safe to read
body_pos = env.data.body_xpos("end_effector")

# ❌ Wrong — forgot forward
env.set_joint_qpos(...)
body_pos = env.data.body_xpos("end_effector")  # May be NaN!
```

## Common Data Sync Issues

| Symptom | Possible Cause | Solution |
|------|----------|----------|
| Reading stale state | No sync after modifying state | `do_simulation()` auto-syncs; manual operations need `mj_forward()` |
| Pose NaN | `mj_forward()` not called after modifying qpos | Call `mj_forward()` after modifying the state |
| Sensor values unchanged | sensordata depends on forward | Read sensors after `mj_forward()` |
| Abnormal contact forces | Contact refreshes after step | Read contact immediately after step |
