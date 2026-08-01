# 📐 State Management

Managing state (qpos/qvel/qacc) in MuJoCo simulation is key to using OrcaGym correctly.

> See [OrcaPlayground examples/euler/04_query_api/](https://github.com/OrcaGym/OrcaPlayground) and [06_jacobian/](https://github.com/OrcaGym/OrcaPlayground) for complete runnable code.

## State Data Layout

```
qpos (generalized coordinates):
 [body0_free_pos_xyz, body0_free_quat_wxyz, joint0_qpos, joint1_qpos, ...]
 length = model.nq

qvel (generalized velocities):
 [body0_free_lin_vel, body0_free_ang_vel, joint0_qvel, joint1_qvel, ...]
 length = model.nv

qacc (generalized accelerations):
 same layout as qvel
 length = model.nv
```

## Getting State

### Getting Global State

```python
# env.data is an OrcaGymDataView (zero-copy view, read-only)
# ⚠️ The zero-copy view updates automatically with simulation steps. Call .copy() to preserve historical values.
qpos = env.data.qpos          # (nq,) — zero-copy view, read-only
qvel = env.data.qvel          # (nv,)
qacc = env.data.qacc          # (nv,)
qfrc_bias = env.data.qfrc_bias  # (nv,)
time = env.data.time          # float, simulation time

# Use .copy() when you need to save historical values or modify
qpos_snapshot = env.data.qpos.copy()
qvel_snapshot = env.data.qvel.copy()
```

### Getting Specific Joint State

```python
# Query specific joints by name
joint_names = ["g1_left_knee_joint", "g1_right_knee_joint"]

# Position
qpos_dict = env.query_joint_qpos(joint_names)
# → {"g1_left_knee_joint": array([0.523]), "g1_right_knee_joint": array([0.518]), ...}

# Velocity
qvel_dict = env.query_joint_qvel(joint_names)

# Acceleration
qacc_dict = env.query_joint_qacc(joint_names)
```

### Getting Joint Index Information

```python
# Address of a single joint in the global array
qpos_adr = env.jnt_qposadr("g1_left_knee_joint")   # starting index in qpos
dof_adr = env.jnt_dofadr("g1_left_knee_joint")      # starting index in qvel/qacc

# Slice from global array by address
knee_angle = env.data.qpos[qpos_adr]                 # hinge joint qpos length = 1
```

> **Note**: `env.data.qpos` is a **global** array (containing all body free dofs and joint qpos).
> In multi-body scenes, you cannot directly use `data.qpos[7:]` to access G1 joints — you must use `jnt_qposadr`
> to stitch together segments by each joint's address. For example, the qpos addresses of G1's 29 revolute joints may not be contiguous with `data.qpos[7:]`.

## Setting State

### Setting Joint Positions

```python
# Full assignment (set the complete qpos array, length = model.nq)
qpos = env.data.qpos.copy()
qpos[env.jnt_qposadr("g1_left_knee_joint")] = 0.6
env.set_joint_qpos(qpos)

# ⚠️ Important: must call mj_forward() after setting to update derived quantities
env.mj_forward()
```

### Setting Joint Velocities

```python
qvel = env.data.qvel.copy()
qvel[env.jnt_dofadr("g1_left_knee_joint")] = 0.1
env.set_joint_qvel(qvel)
env.mj_forward()
```

### Resetting to Initial State

```python
# Use init_qpos/init_qvel in reset_model
def reset_model(self):
    qpos = self.init_qpos.copy() + noise
    self.set_joint_qpos(qpos)
    self.mj_forward()
    self._sync_view()
    return self._get_obs(), {}
```

## Getting Body Poses

```python
# Query a single body by name (via env.data)
body_pos = env.data.body_xpos("g1_pelvis")    # (3,) world position
body_quat = env.data.body_xquat("g1_pelvis")  # (4,) [w, x, y, z]
body_mat = env.data.body_xmat("g1_pelvis")    # (9,) 3×3 rotation matrix row-major

# Batch query (recommended: returns full poses for multiple bodies at once)
body_dict = env.get_body_xpos_xmat_xquat(["g1_pelvis", "g1_torso_link"])
for name, pose in body_dict.items():
    pos = pose["xpos"]    # np.array([x, y, z])
    mat = pose["xmat"]    # np.array((3, 3)) — 3×3 rotation matrix (already reshaped)
    quat = pose["xquat"]  # np.array([w, x, y, z])

# Common: get pelvis height
pelvis_z = float(body_dict["g1_pelvis"]["xpos"][2])
```

## Getting Sensor Data

```python
# Query sensor data (by name)
sensor_data = env.query_sensor_data([
    "g1_imu_quat",
    "g1_imu_gyro",
])

imu_quat = sensor_data["g1_imu_quat"]  # (4,) orientation quaternion
imu_gyro = sensor_data["g1_imu_gyro"]  # (3,) angular velocity
```

## Golden Rule of State Synchronization

> ⚠️ **Modify state → mj_forward → _sync_view → then read data**

```python
# ✅ Correct state modification flow (Euler API)
env.set_joint_qpos(new_qpos)       # 1. Modify state
env.mj_forward()                    # 2. Refresh derived quantities (body poses, sensors, etc.)
env._sync_view()                    # 3. Sync to DataView
current_qpos = env.data.qpos        # 4. Read (zero-copy view reflects latest values)

# ✅ do_simulation auto-syncs (recommended)
env.do_simulation(ctrl, n_frames)   # Internally: set_ctrl → mj_step → _sync_view
current_qpos = env.data.qpos        # Read directly

# ✅ When you need to save a snapshot
snapshot = env.data.qpos.copy()     # copy() creates an independent copy, unaffected by later updates
```

## Common Mistakes

| Mistake | Consequence | Fix |
|------|------|------|
| Not calling `mj_forward()` after modifying qpos | Body pose/sensor data is stale | Add `mj_forward()` |
| Reading data without syncing the view | Reading stale data | Call `_sync_view()` |
| Saving a reference without `.copy()` | Data gets overwritten by subsequent simulation steps | Use `data.qpos.copy()` |
| Wrong array dimensions | ValueError | Use `jnt_qposadr` to check address and length |
| Reading body pose without calling `mj_forward()` | Reading stale pose | Must call `mj_forward()` after modifying qpos |
| Using `data.qpos[7:]` directly in multi-body scenes | Reading data from other bodies | Use `jnt_qposadr` to stitch segments joint by joint |
