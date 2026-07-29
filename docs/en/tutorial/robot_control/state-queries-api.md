# 📡 State Query API — Reading Joints, Bodies, and Sensors

This section covers how to use OrcaGym's **state query API**, covering joint states, body poses, sensors, actuator torques, contact information, and more.

> For complete runnable code, see [OrcaPlayground examples/euler/04_query_api/](https://github.com/OrcaGym/OrcaPlayground).

---

## Complete Example: The Big Picture

Below is a **runnable** complete example demonstrating the usage of all query APIs.
It is recommended to read through it once, then review the step-by-step explanation that follows.

```python
"""State Query API Complete Demo

Features: Demonstrates the full set of state query APIs provided by OrcaGymEulerEnv

Usage (offline mode, no Studio required):
    python query_demo.py
"""
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class QueryDemoEnv(OrcaGymEulerEnv):
    """Environment that demonstrates all query APIs"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 20),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["g1"]),
            time_step=kwargs.pop("time_step", 0.001),
            model_xml_path=model_xml_path,
            **kwargs,
        )

    def demo_all_queries(self):
        """Demonstrate the full set of query APIs"""
        self.reset()
        agent = self.agent_name
        print(f"Agent name: {agent}\n")

        # ─── 1. Joint Queries ───
        print("=" * 50)
        print("1. Joint Queries")
        print("=" * 50)

        knee = f"{agent}_left_knee_joint"
        hip = f"{agent}_left_hip_pitch_joint"

        # Query position by name
        qpos = self.query_joint_qpos([knee, hip])
        print(f"  {knee}: {qpos[knee]} rad")
        print(f"  {hip}:   {qpos[hip]} rad")

        # Query velocity by name
        qvel = self.query_joint_qvel([knee, hip])
        print(f"  {knee} velocity: {qvel[knee]} rad/s")

        # Get joint address in global arrays
        qpos_adr = self.jnt_qposadr(knee)
        dof_adr = self.jnt_dofadr(knee)
        print(f"  {knee} qpos address: {qpos_adr}, dof address: {dof_adr}")

        # ─── 2. Body Pose Queries ───
        print("\n" + "=" * 50)
        print("2. Body Pose Queries")
        print("=" * 50)

        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        # Return format: {"g1_pelvis": {"xpos": array, "xmat": array, "xquat": array}}
        p = pelvis[f"{agent}_pelvis"]
        print(f"  pelvis position: [{p['xpos'][0]:.3f}, {p['xpos'][1]:.3f}, {p['xpos'][2]:.3f}]")
        print(f"  pelvis height: {p['xpos'][2]:.3f}m")
        print(f"  pelvis quaternion: [{p['xquat'][0]:.3f}, {p['xquat'][1]:.3f}, {p['xquat'][2]:.3f}, {p['xquat'][3]:.3f}]")

        # Can also query a single body by name via env.data
        pelvis_z = self.data.body_xpos(f"{agent}_pelvis")[2]
        print(f"  (env.data) pelvis z: {pelvis_z:.3f}m")

        # ─── 3. Sensor Queries ───
        print("\n" + "=" * 50)
        print("3. Sensor Queries")
        print("=" * 50)

        imu = self.query_sensor_data([f"{agent}_imu_quat", f"{agent}_imu_gyro"])
        print(f"  IMU quaternion: {imu[f'{agent}_imu_quat']}")
        print(f"  IMU angular velocity: {imu[f'{agent}_imu_gyro']}")

        # ─── 4. Actuator Torque Queries ───
        print("\n" + "=" * 50)
        print("4. Actuator Torque Queries")
        print("=" * 50)

        actuator_names = [f"{agent}_left_knee", f"{agent}_right_knee"]
        torques = self.query_actuator_torques(actuator_names)
        print(f"  Left knee torque: {torques[f'{agent}_left_knee']}")
        print(f"  Right knee torque: {torques[f'{agent}_right_knee']}")

        # ─── 5. Contact Queries ───
        print("\n" + "=" * 50)
        print("5. Contact Queries")
        print("=" * 50)

        contacts = self.query_contact_simple()
        # Returns: [{"geom1": 12, "geom2": 34, ...}, ...]
        print(f"  Active contacts: {len(contacts)}")
        if contacts:
            # Get contact forces (by list index)
            contact_ids = list(range(len(contacts)))
            forces = self.query_contact_force(contact_ids)
            max_normal = max(abs(f[0]) for f in forces.values())
            print(f"  Max normal force: {max_normal:.1f}N")
            # Show first 3 contacts
            for i, c in enumerate(contacts[:3]):
                f = forces[i][:3]
                print(f"    Contact {i}: geom{c['geom1']}↔geom{c['geom2']}, force={np.linalg.norm(f):.1f}N")

        # ─── 6. Mass Queries ───
        print("\n" + "=" * 50)
        print("6. Mass Queries")
        print("=" * 50)

        torso_mass = self.body_subtree_mass(f"{agent}_torso_link")
        print(f"  Torso subtree total mass: {torso_mass:.2f}kg")

        # ─── 7. Base Frame Transform ───
        print("\n" + "=" * 50)
        print("7. Base Frame Transform")
        print("=" * 50)

        # torso_link position in pelvis coordinate frame
        torso_B = self.query_position_body_B(
            f"{agent}_torso_link", f"{agent}_pelvis"
        )
        print(f"  Torso in pelvis frame: {torso_B}")
        print(f"  Torso above pelvis: {torso_B[2]:.3f}m (z component)")

        # ─── 8. Site Queries ───
        print("\n" + "=" * 50)
        print("8. Site Queries")
        print("=" * 50)

        imu_site = self.query_site_pos_and_mat([f"{agent}_imu"])
        site_pos = imu_site[f"{agent}_imu"]["xpos"]
        print(f"  IMU site position: {site_pos}")

        # Site velocity
        xvalp, xvalr = self.query_site_xvalp_xvalr([f"{agent}_imu"])
        print(f"  IMU site linear velocity: {xvalp[f'{agent}_imu']}")
        print(f"  IMU site angular velocity: {xvalr[f'{agent}_imu']}")

        # ─── 9. Direct Reads from env.data ───
        print("\n" + "=" * 50)
        print("9. env.data Zero-Copy View")
        print("=" * 50)

        print(f"  data.qpos.shape: {self.data.qpos.shape}")
        print(f"  data.qvel.shape: {self.data.qvel.shape}")
        print(f"  data.time:       {self.data.time:.4f}s")
        print(f"  model.nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")

        print("\n All query API demos complete")

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        return self._get_obs(), 0.0, False, False, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.data.qpos.copy()


if __name__ == "__main__":
    import sys
    env = QueryDemoEnv(
        model_xml_path=sys.argv[1] if len(sys.argv) > 1 else "/path/to/scene.xml",
        skip_grpc_load=True,   # Offline mode
    )
    env.demo_all_queries()
    env.close()
```

---

## Step-by-Step Explanation

### Query API Overview

`OrcaGymEulerEnv` provides the following public query methods, **all accessed by name, no need to memorize IDs**:

| Category | Method | Return Type | Description |
|----------|--------|-------------|-------------|
| **Joints** | `query_joint_qpos(names)` | `dict[str, np.ndarray]` | Joint position |
| | `query_joint_qvel(names)` | `dict[str, np.ndarray]` | Joint velocity |
| | `query_joint_qacc(names)` | `dict[str, np.ndarray]` | Joint acceleration |
| | `jnt_qposadr(name)` | `int` | Start address in qpos |
| | `jnt_dofadr(name)` | `int` | Start address in qvel/qacc |
| **Bodies** | `get_body_xpos_xmat_xquat(names)` | `dict[str, dict]` | World pose (position + matrix + quaternion) |
| | `get_body_xpos_xmat_xquat_xvel(names)` | `dict[str, dict]` | Pose + linear velocity |
| **Sites** | `query_site_pos_and_mat(names)` | `dict[str, dict]` | Site pose |
| | `query_site_xvalp_xvalr(names)` | `tuple[dict, dict]` | Site velocity (linear + angular) |
| **Sensors** | `query_sensor_data(names)` | `dict[str, np.ndarray]` | Sensor readings |
| **Actuators** | `query_actuator_torques(names)` | `dict[str, np.ndarray]` | Torques |
| **Contacts** | `query_contact_simple()` | `list[dict]` | Contact pair list |
| | `query_contact_force(ids)` | `dict[int, np.ndarray]` | Contact force (6D) |
| **Mass** | `body_subtree_mass(name)` | `float` | Total subtree mass |
| **Base Transform** | `query_position_body_B(ee, base)` | `np.ndarray(3,)` | Relative position |

### 1. Joint Queries

```python
# Prepare joint names (must include agent prefix)
agent = "g1"
joint_names = [f"{agent}_left_knee_joint", f"{agent}_right_knee_joint"]

# Query
qpos = env.query_joint_qpos(joint_names)
# → {"g1_left_knee_joint": array([0.523]), "g1_right_knee_joint": array([0.518])}

qvel = env.query_joint_qvel(joint_names)
# → {"g1_left_knee_joint": array([-0.1]), ...}
```

**Implementation principle**: `query_joint_qpos` internally slices from `data.qpos` by address using `jnt_qposadr`:

```python
# Internally equivalent to:
for jn in joint_names:
    addr = env.jnt_qposadr(jn)
    result[jn] = env.data.qpos[addr:addr + 1]  # hinge joint length = 1
```

> **Important**: `env.data.qpos` is a **global** array (contains all body DOFs + all joints).
> In multi-body scenes, you **cannot** use `data.qpos[7:]` to directly get G1 joints; you must
> concatenate joint by joint using `jnt_qposadr`.

### 2. Body Pose Queries

```python
pelvis = env.get_body_xpos_xmat_xquat(["g1_pelvis", "g1_torso_link"])

# Return format:
# {
#   "g1_pelvis": {
#     "xpos": np.array([0.0, 0.0, 0.78]),   # World position (3,)
#     "xmat": np.array([...]),                # Rotation matrix (3, 3) (already reshaped)
#     "xquat": np.array([1.0, 0, 0, 0]),     # Quaternion [w,x,y,z] (4,)
#   },
#   "g1_torso_link": { ... }
# }

# Common: get pelvis height
pelvis_z = float(pelvis["g1_pelvis"]["xpos"][2])
```

You can also query single bodies via `env.data`:
```python
env.data.body_xpos("g1_pelvis")    # (3,)  returns position only
env.data.body_xquat("g1_pelvis")   # (4,)  returns quaternion only
```

### 3. Sensor Queries

```python
sensor_data = env.query_sensor_data(["g1_imu_quat", "g1_imu_gyro"])

# Get results by name
imu_quat = sensor_data["g1_imu_quat"]  # (4,) orientation quaternion
imu_gyro = sensor_data["g1_imu_gyro"]  # (3,) angular velocity
```

> Sensor data is only updated after `mj_forward()` or `do_simulation()`.
> Modifying qpos without calling forward before reading sensors → you will read stale data.

### 4. Contact Queries

```python
# Step 1: Get contact list
contacts = env.query_contact_simple()
# → [{"geom1": 12, "geom2": 34, "dist": 0.001, "pos": [...], "frame": [...]}, ...]

# Step 2: Get contact forces (by list index, not by some id field in the contact dict)
contact_ids = list(range(len(contacts)))
forces = env.query_contact_force(contact_ids)
# → {0: array([normal, shear1, shear2, torque1, torque2, torque3]), 1: ...}

# Component 0 is the normal force
max_normal = max(abs(f[0]) for f in forces.values())
```

> **Note**: The dict keys returned by `query_contact_simple()` are **lowercase** `"geom1"` / `"geom2"`,
> not uppercase `"Geom1"` / `"Geom2"`.

### 5. Base Frame Transform

```python
# Pure NumPy transform, does not depend on MuJoCo
torso_in_pelvis = env.query_position_body_B("g1_torso_link", "g1_pelvis")
# → array([x, y, z])  — torso_link position in pelvis coordinate frame
```

### 6. `env.data` Zero-Copy View

`env.data` is an `OrcaGymDataView`, providing a **zero-copy read-only view**. Data is automatically updated as simulation steps:

```python
env.data.qpos          # (nq,) generalized coordinates
env.data.qvel          # (nv,) generalized velocities
env.data.qacc          # (nv,) generalized accelerations
env.data.time          # float simulation time
env.data.qfrc_bias     # (nv,) bias forces
env.data.xfrc_applied  # (nbody, 6) applied external forces (read-only)
```

> `env.data.qpos` is a zero-copy view. If you need to save historical values, call `.copy()`.

---

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `KeyError: 'g1_left_knee_joint'` | Joint name missing agent prefix | Use `f"{agent}_{suffix}"` to concatenate |
| `data.qpos[7:]` gives wrong values | Addresses not contiguous in multi-body scenes | Slice joint by joint using `jnt_qposadr` |
| `query_contact_force` returns empty | No contacts right after loading | Step a few frames first to let the robot touch the ground |
| Sensor data is stale | Forgot to call `mj_forward()` | Must forward after `set_joint_qpos` |

---

## Next Steps

Now that you've mastered state queries, learn how to **apply external forces and write state**: [🔄 External Force Application and IK](../physics/force-apply.md).
