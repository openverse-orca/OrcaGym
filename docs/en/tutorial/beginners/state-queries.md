# 📡 Reading State — Joints, Bodies, Sensors

In the previous section we only read `self.data.qpos` and `self.data.qvel`. In this section, you will learn to use the **query API** provided by OrcaGym to obtain richer state information.

> See [OrcaPlayground examples/euler/04_query_api/](https://github.com/OrcaGym/OrcaPlayground) for complete runnable code.

---

## Complete Example: See the Big Picture First

Below is a **runnable** complete example that constructs a `StateDumper` debugging tool, demonstrating the usage of all state query APIs.

```python
"""State reading complete demo: joints, bodies, sites, sensors, actuator torques"""
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class StateDumper:
    """Debugging tool: one-click dump of all current states of the environment"""

    def __init__(self, env):
        self.env = env

    def dump(self):
        env = self.env
        print("=" * 60)
        print(f"Simulation time: {env.data.time:.3f}s")

        # -- Joints --
        joint_names = list(env.model.get_joint_dict().keys())[:5]
        qpos = env.query_joint_qpos(joint_names)
        print("\nJoint positions (first 5):")
        for name in joint_names:
            print(f"  {name}: {qpos[name]}")

        # -- End effector --
        try:
            ee_site = env.site("end_effector")
            sites = env.query_site_pos_and_quat([ee_site])
            ee_pos = sites[ee_site]["xpos"]
            print(f"\nEnd-effector position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
        except Exception:
            print("\n(no end_effector site)")

        # -- Contacts --
        contacts = env.query_contact_simple()
        print(f"\nNumber of contact points: {len(contacts)}")

        print("=" * 60)


class StateQueryDemo(OrcaGymEulerEnv):
    """Environment demonstrating all state query APIs"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            **kwargs,
        )
        self._dumper = StateDumper(self)

    # --- Query methods ---

    def check_joints(self):
        """Query positions and velocities of all joints"""
        joint_names = list(self.model.get_joint_dict().keys())
        qpos = self.query_joint_qpos(joint_names)
        qvel = self.query_joint_qvel(joint_names)

        print(f"Total {len(joint_names)} joints:")
        for name in joint_names[:5]:  # show first 5
            pos = qpos[name]
            vel = qvel[name]
            pos_str = f"{pos[0]:.3f}" if len(pos) == 1 else f"{pos}"
            print(f"  {name:25s}: pos={pos_str}, vel={vel}")
        return qpos, qvel

    def inspect_joint(self, joint_name):
        """View detailed info of a single joint"""
        info = self.model.get_joint_byname(joint_name)
        print(f"Joint: {joint_name}")
        print(f"  Type: {info['Type']}")          # hinge / slide / free / ball
        print(f"  Limited: {info['Limited']}")
        if info['Limited']:
            print(f"  Range: [{info['Range'][0]:.3f}, {info['Range'][1]:.3f}] rad")

        qpos_addr = self.jnt_qposadr(joint_name)  # starting position in qpos
        dof_addr = self.jnt_dofadr(joint_name)     # starting position in qvel
        print(f"  qpos address: {qpos_addr}, qvel address: {dof_addr}")

    def check_body_pose(self):
        """Query positions and orientations of key bodies"""
        body_names = list(self.model.get_body_names())
        print(f"Total {len(body_names)} bodies (first 10):")
        for name in body_names[:10]:
            print(f"  - {name}")

        # Batch query (recommended)
        if len(body_names) >= 2:
            body_dict = self.get_body_xpos_xmat_xquat(body_names[:2])
            for name, pose in body_dict.items():
                print(f"\n{name}:")
                print(f"  Position: {pose['xpos']}")
                print(f"  Quaternion: {pose['xquat']}")

    def check_end_effector(self):
        """Query end-effector pose and velocity"""
        ee_site = self.site("end_effector")
        site_data = self.query_site_pos_and_mat([ee_site])
        ee = site_data[ee_site]
        print(f"End-effector (site: {ee_site}):")
        print(f"  Position: {ee['xpos']}")
        print(f"  Rotation matrix: {ee['xmat']}")

        # Velocity
        linear_vel, angular_vel = self.query_site_xvalp_xvalr([ee_site])
        print(f"  Linear velocity: {linear_vel[ee_site]}")
        print(f"  Angular velocity: {angular_vel[ee_site]}")

    def read_sensors(self):
        """Read sensor data"""
        sensor_names = list(self.model.gen_sensor_dict().keys())
        print(f"Sensor list ({len(sensor_names)} total):")
        for name in sensor_names:
            info = self.model.gen_sensor_dict()[name]
            print(f"  {name}: type={info['Type']}, dim={info['Dim']}")

        if sensor_names:
            sensor_data = self.query_sensor_data(sensor_names[:3])
            for name, data in sensor_data.items():
                print(f"  {name}: {data}")

    def read_actuator_torques(self):
        """View actuator torques"""
        names = [self.model.actuator_id2name(i) for i in range(self.model.nu)]
        torques = self.query_actuator_torques(names[:3])
        for name, t in torques.items():
            print(f"  {name}: {t}")

    # --- Demo entry point ---

    def demo(self):
        self.reset()
        print(f"Environment: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}\n")

        self._dumper.dump()
        self.check_joints()
        self.check_body_pose()

    # --- Gymnasium interface ---
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
    env = StateQueryDemo(
        model_xml_path="/path/to/scene.xml",
        skip_grpc_load=True,  # offline mode
    )
    env.demo()
    env.close()
```

---

## Section-by-Section Explanation

### Query API Overview

OrcaGym provides APIs that query by **name** (no need to remember IDs):

| Query Target | Method | Returns |
|--------------|--------|---------|
| Joint position | `query_joint_qpos(names)` | `dict[str, array]` |
| Joint velocity | `query_joint_qvel(names)` | `dict[str, array]` |
| Body pose | `get_body_xpos_xmat_xquat(names)` | `dict[str, dict]` |
| Body position (single) | `env.data.body_xpos(name)` | `(3,)` |
| Site pose | `query_site_pos_and_quat(names)` | `dict[str, dict]` |
| Site velocity | `query_site_xvalp_xvalr(names)` | `tuple[dict, dict]` |
| Sensor | `query_sensor_data(names)` | `dict[str, array]` |
| Actuator torque | `query_actuator_torques(names)` | `dict[str, array]` |

### 1. Joint Queries

```python
qpos = env.query_joint_qpos(["robot_0_joint1", "robot_0_joint2"])
# -> {"robot_0_joint1": array([0.52]), "robot_0_joint2": array([-0.31])}

qvel = env.query_joint_qvel(["robot_0_joint1", "robot_0_joint2"])
```

**View detailed joint info**:
```python
info = env.model.get_joint_byname("robot_0_joint1")
# -> {"Type": "hinge", "Limited": True, "Range": [-3.14, 3.14], ...}
```

**Joint address in the global array**:
```python
qpos_addr = env.jnt_qposadr("robot_0_joint1")   # index in qpos
dof_addr = env.jnt_dofadr("robot_0_joint1")      # index in qvel
```

### 2. Body Pose Queries

```python
# Batch query (recommended: fetch multiple bodies at once)
body_dict = env.get_body_xpos_xmat_xquat(["base_link", "ee_link"])
for name, pose in body_dict.items():
    print(f"{name}: pos={pose['xpos']}, quat={pose['xquat']}")

# Single query (via env.data)
pos = env.data.body_xpos("base_link")     # (3,) world position
quat = env.data.body_xquat("base_link")   # (4,) [w,x,y,z]
```

### 3. Site Queries

Sites are marker points in MuJoCo, typically marking the **end-effector**, **IMU position**, etc.:

```python
# Pose
sites = env.query_site_pos_and_quat(["robot_0_end_effector"])
ee = sites["robot_0_end_effector"]
print(f"End-effector position: {ee['xpos']}")

# Velocity (linear + angular)
lin_vel, ang_vel = env.query_site_xvalp_xvalr(["robot_0_end_effector"])
```

### 4. Sensor Queries

```python
# List all sensors
sensor_names = list(env.model.gen_sensor_dict().keys())

# Read data
data = env.query_sensor_data(["imu_acc", "imu_gyro"])
```

Common sensor types:

| MuJoCo Type | Purpose | Dimension |
|-------------|---------|-----------|
| `accelerometer` | Linear acceleration | (3,) |
| `gyro` | Angular velocity | (3,) |
| `force_torque` | 6-axis force/torque | (6,) |
| `jointpos` | Joint position | (1,) |
| `jointvel` | Joint velocity | (1,) |

### Timing of State Updates

```
mj_forward() or do_simulation()
        │
        v
  All derived quantities updated (sensors, contact forces, body poses...)
        │
        v
  Your query methods <- now you can read the latest values
```

> ⚠️ **Must forward/step before querying**
> ```python
> # ✅ Correct
> env.do_simulation(ctrl, frame_skip)   # internal step + sync
> pos = env.query_joint_qpos(["joint_0"])
> 
> # ❌ Wrong — may read stale data
> env.set_joint_qpos(...)
> pos = env.query_joint_qpos(["joint_0"])  # haven't forwarded yet!
> ```

---

## Next Step

Now you can read state. Next, learn how to **make the robot move**: [🦾 Making the Robot Move](move-a-joint.md).
