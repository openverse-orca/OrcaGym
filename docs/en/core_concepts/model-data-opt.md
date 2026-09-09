# 📊 Model / Data / Config

OrcaGym organizes simulation information into three clear layers.

---

## OrcaGymModel — Static Model Information

`OrcaGymModel` contains all information that **does not change during simulation**, accessible via `env.model`:

```python
model = env.model

# Dimension information
print(model.nq)          # Number of generalized coordinates
print(model.nv)          # Number of degrees of freedom
print(model.nu)          # Number of actuators
print(len(model.get_body_names()))    # Total number of bodies
print(len(model.get_joint_dict()))    # Total number of joints

# Name-to-ID mappings
body_id = model.body_name2id("base_link")
joint_id = model.joint_name2id("shoulder")
actuator_id = model.actuator_name2id("shoulder_actuator")

# Get the actuator control range (for building action_space)
ctrl_range = model.get_actuator_ctrlrange()  # shape: (nu, 2)

# List all body names
body_names = model.get_body_names()
```

### Information Stored in Model

| Information Type | Contents |
|----------|----------|
| Body | Mass, inertia, parent-child relationships, pose |
| Joint | Type, range, axis, stiffness, damping |
| Actuator | Control range, gear ratio, transmission type |
| Geom | Shape, friction, contact parameters |
| Site | Marker position, size |
| Sensor | Type, dimensions |
| Eq | Equality constraint type, target objects |
| Mocap | Mocap body mappings |

---

## OrcaGymDataView — Dynamic Simulation State

`OrcaGymDataView` is a read-only view of the simulation state, accessible via `env.data`. It is automatically updated after `do_simulation()`.

```python
data = env.data

# Core state
qpos = data.qpos               # (nq,) generalized coordinates
qvel = data.qvel               # (nv,) generalized velocity
qacc = data.qacc               # (nv,) generalized acceleration
time = data.time               # Simulation time (scalar)

# External forces and constraints
xfrc_applied = data.xfrc_applied       # External forces (read-only)
cfrc_ext = data.cfrc_ext              # External constraint forces (nbody, 6)
contact = data.contact                 # Contact list

# Query body/site by name (no need to know the ID)
body_pos = data.body_xpos("torso_link")       # (3,) world coordinates
body_quat = data.body_xquat("torso_link")    # (4,) [w,x,y,z]
body_vel = data.body_cvel("torso_link")       # (6,) [ang(3), lin(3)]
site_pos = data.site_xpos("imu")             # (3,)
geom_pos = data.geom_xpos("box_geom")        # (3,)
mass = data.body_subtree_mass("torso_link")   # float
```

### When to Read State

```
do_simulation(ctrl, n_frames)   ← Simulation step
  └─▶ Stepping complete, env.data automatically updated
  └─▶ Can directly read env.data.qpos, etc.
```

> ⚠️ **Important**: After `do_simulation()` returns, `env.data` is already automatically synchronized (internally calling `sync_to_view`) and can be read directly. The base fields of `env.data` (`qpos`/`qvel`, etc.) are zero-copy views, while derived quantities (`body_xpos`/`site_xpos`, etc.) are read on demand from the live `_mjData`, so after `mj_forward()` executes, derived quantities reflect the latest values (`mj_forward` updates `_mjData` in place). However, `env.data.time` is a float copy (not a view) that is only refreshed on `sync_to_view` — `mj_forward` does not change `time`, so it has no effect; if you need to read `time` after `mj_step`, use `do_simulation()` or manually call `sync_to_view`.

---

## SimConfig — Solver Configuration

`SimConfig` provides a read/write interface for simulation parameters, accessible via `env.sim_config`. Changes are written immediately to `mj_model.opt`, and take effect on the next `mj_step`.

```python
sim_config = env.sim_config

# Read/write parameters
sim_config.timestep = 0.002     # Physics time step
sim_config.iterations = 100     # Solver iteration count
sim_config.integrator = 1       # Integrator (0=Euler, 1=RK4, 2=IMPLICIT, 3=IMPLICITFAST)
sim_config.gravity = np.array([0., 0., -9.81])  # Gravity

# Batch configuration
sim_config.load_from_dict({
    "integrator": 0,
    "iterations": 100,
})

# Export the configuration
config_dict = sim_config.to_dict()
```

---

## Environment Time Step

```python
env.dt = env.sim_config.timestep * env.frame_skip
```

Control frequency: `control_hz = 1.0 / env.dt`

For example, `timestep = 0.001, frame_skip = 20` → `dt = 0.020s, control_hz = 50 Hz`

---

## Joint Types and qpos/qvel Dimensions

Different joint types occupy different numbers of elements in `qpos` and `qvel`:

| Joint Type | qpos Size | qvel Size | Example |
|----------|-----------|-----------|------|
| FREE | 7 (3 pos + 4 quat) | 6 (3 lin + 3 ang) | Free-flying body |
| BALL | 4 (quaternion) | 3 (angular velocity) | Ball joint |
| HINGE | 1 (angle) | 1 (angular velocity) | Revolute joint |
| SLIDE | 1 (displacement) | 1 (linear velocity) | Sliding joint |
