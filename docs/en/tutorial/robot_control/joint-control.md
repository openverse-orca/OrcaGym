# 🦿 Joint Control

Low-level joint control interface, directly operating MuJoCo actuators.

## set_ctrl — Lowest-Level Control

```python
# Directly set control values for all actuators
ctrl = np.array([0.1, -0.2, 0.0, ...], dtype=np.float64) # (nu,)
env.set_ctrl(ctrl)
env.mj_step(n_frames)
env._sync_view() 
```

## Atomic Operations via do_simulation (Recommended)

```python
# do_simulation = set_ctrl + mj_step + automatic data sync
env.do_simulation(ctrl, n_frames=env.frame_skip)
```

## Joint Position Control

`set_joint_qpos` accepts a **full qpos array** (`np.ndarray`, shape `(nq,)`), not a dict.
You need to write each joint into the corresponding address of the qpos array, then set it as a whole:

```python
# Build the full qpos array and set it as a whole
qpos = env.data.qpos.copy()
qpos[env.jnt_qposadr("shoulder_joint")] = 0.5
qpos[env.jnt_qposadr("elbow_joint")] = -0.3
qpos[env.jnt_qposadr("wrist_joint")] = 1.2
env.set_joint_qpos(qpos)

# Must forward
env.mj_forward()
```

## Joint Velocity Control

`set_joint_qvel` also accepts a **full qvel array** (`np.ndarray`, shape `(nv,)`):

```python
qvel = env.data.qvel.copy()
qvel[env.jnt_dofadr("shoulder_joint")] = 0.1
qvel[env.jnt_dofadr("elbow_joint")] = -0.05
env.set_joint_qvel(qvel)

env.mj_forward()
```

## JointController — PD Control

```python
from orca_gym.utils.joint_controller import JointController

# Create a PD controller for each joint
controllers = {
	"shoulder": JointController(Kp=100.0, Ki=0.1, Kd=10.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)),
	"elbow": JointController(Kp=100.0, Ki=0.1, Kd=10.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)),
	"wrist": JointController(Kp=100.0, Ki=0.1, Kd=10.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)),
}

# Compute control torque (each joint computed independently)
# Note: the ctrl array is indexed by actuator, not by joint.
# joint_name2id returns the joint id, which cannot be used directly as a ctrl index;
# use actuator_name2id to get the actuator id.
ctrl = np.zeros(env.model.nu)
target_angles = {"shoulder_actuator": 0.5, "elbow_actuator": -0.3, "wrist_actuator": 1.2}
# Assume actuator names correspond one-to-one with joint names (per model XML); joint names are used to look up dof addresses
joint_of_actuator = {
	"shoulder_actuator": "shoulder_joint",
	"elbow_actuator": "elbow_joint",
	"wrist_actuator": "wrist_joint",
}
for actuator_name, target in target_angles.items():
	actuator_id = env.model.actuator_name2id(actuator_name)
	joint_name = joint_of_actuator[actuator_name]
	dof_adr = env.jnt_dofadr(joint_name)
	ctrl[actuator_id] = controllers[joint_name.replace("_joint", "")].compute_torque(
		target_qpos=target,
		current_qpos=env.data.qpos[dof_adr],
		current_qvel=env.data.qvel[dof_adr],
		dt=env.dt,
	)

# Apply (do_simulation auto-syncs data)
env.do_simulation(ctrl, env.frame_skip)
```

## PD Parameter Tuning

| Parameter | Function | Typical Value |
|-----------|----------|---------------|
| kp | Proportional gain → stiffness/response speed | 10 ~ 500 |
| kd | Derivative gain → damping/stability | 1 ~ 50 |

- kp too large → oscillation
- kp too small → slow tracking
- kd too large → sluggish response
- kd too small → underdamped

## Low-Pass Filtering

```python
from orca_gym.utils.low_pass_filter import LowPassFilter

# Create filter
filter = LowPassFilter(alpha=0.1, initial_output=np.zeros(env.model.nu))

# Filter ctrl at each step
raw_ctrl = compute_raw_ctrl(...)
smooth_ctrl = filter.apply(raw_ctrl)
env.do_simulation(smooth_ctrl, env.frame_skip)
```

## Joint Limit Checking

```python
def check_joint_limits(env):
 """Check whether all joints are within limits"""
 for joint_name in list(env.model.get_joint_dict().keys()):
 joint_info = env.model.get_joint_byname(joint_name)
 if not joint_info["Limited"]:
 continue
 
 qpos = env.query_joint_qpos([joint_name])[joint_name]
 low, high = joint_info["Range"]
 
 if qpos[0] < low or qpos[0] > high:
 print(f"Warning: {joint_name} out of range: "
 f"{qpos[0]:.3f} not in [{low:.3f}, {high:.3f}]")
```
