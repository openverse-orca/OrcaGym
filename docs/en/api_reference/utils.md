# 🔧 Utils API

Utility functions and controllers, providing inverse kinematics, joint control, rotation conversion, and other practical tools.

## Classes and Functions Overview

| Name | Description |
|------|------|
| `InverseKinematicsController` | Jacobian-based inverse kinematics solver |
| `JointController` | PID + velocity feedforward joint torque controller |
| `pd_control` | Simple PD controller function |
| `LowPassFilter` | First-order low-pass filter |
| `RewardPrinter` | Training reward statistics and printing |
| `rotations` | Rotation representation conversion utilities |

---

## InverseKinematicsController

Inverse kinematics solver based on Jacobian matrix + damped least squares method.

> ⚠️ **Note**: This controller depends on the `RobomimicEnv` adapter (`orca_gym.adapters.robomimic`) and is not applicable to the raw `OrcaGymEulerEnv`. Wrap your environment via the Robomimic adapter before use.

### Constructor

```python
class InverseKinematicsController:
    def __init__(
        self,
        env: RobomimicEnv,          # Environment wrapped by the RobomimicEnv adapter
        site_id: int,               # ID of the end-effector site
        dof_indices: list[int],     # List of DOF indices for controlled joints
        lamba_value: float = 1e-3,  # Damping coefficient (note: actual spelling in code is lamba, not lambda)
        alpha_value: float = 0.2,   # Step size scaling factor
    )
```

### Methods

```python
def set_goal(pos: np.ndarray, quat: np.ndarray)         # Set target pose
def set_lambda(lambda_value: float)                     # Set damping coefficient
def set_alpha(alpha_value: float)                       # Set step size
def compute_inverse_kinematics() -> np.ndarray          # Compute incremental joint angles dq
```

### Usage Example

```python
from orca_gym.utils.inverse_kinematics_controller import InverseKinematicsController
import numpy as np

ik = InverseKinematicsController(
    env=my_env,
    site_id=end_effector_site_id,
    dof_indices=arm_dof_indices,
    lamba_value=1e-3,
    alpha_value=0.2,
)

# Set target pose
ik.set_goal(
    pos=np.array([0.5, 0.0, 0.3]),
    quat=np.array([1.0, 0.0, 0.0, 0.0]),
)

# Compute inverse kinematics each step
dq = ik.compute_inverse_kinematics()
current_qpos += dq
```

---

## JointController

PID + velocity feedforward single-joint torque controller with integral anti-windup.

### Constructor

```python
class JointController:
    def __init__(
        self,
        Kp: float = 10.0,            # Position proportional gain
        Ki: float = 0.1,             # Integral gain
        Kd: float = 2.0,             # Velocity derivative gain
        Kv: float = 5.0,             # Velocity error feedback gain
        max_speed: float = 80.0,     # Maximum allowed target speed (rad/s)
        ctrlrange: tuple = (-80, 80),# Actuator torque limits (min, max) Nm
    )
```

### Methods

```python
def compute_torque(
    self,
    target_qpos: float,        # Target angle (rad)
    current_qpos: float,       # Current angle (rad)
    current_qvel: float,       # Current velocity (rad/s)
    dt: float,                 # Simulation step size (s)
) -> float                    # Output torque (Nm)
```

### Usage Example

```python
from orca_gym.utils.joint_controller import JointController

controller = JointController(
    Kp=50.0, Ki=0.5, Kd=3.0, Kv=8.0,
    max_speed=100.0, ctrlrange=(-100, 100),
)

dt = 0.001
for target in target_trajectory:
    torque = controller.compute_torque(
        target_qpos=target,
        current_qpos=current_angle,
        current_qvel=current_velocity,
        dt=dt,
    )
    ctrl[actuator_id] = torque
```

---

## pd_control

Simple PD controller function.

```python
def pd_control(
    target_q: np.ndarray,      # Target position
    q: np.ndarray,             # Current position
    kp: float | np.ndarray,    # Position gain
    target_dq: np.ndarray,     # Target velocity
    dq: np.ndarray,            # Current velocity
    kd: float | np.ndarray,    # Velocity gain
) -> np.ndarray               # Output torque
```

Formula: `torque = (target_q - q) × kp + (target_dq - dq) × kd`

---

## LowPassFilter

First-order exponential smoothing low-pass filter.

```python
class LowPassFilter:
    def __init__(self, alpha: float, initial_output: np.ndarray)
    def apply(self, x: np.ndarray) -> np.ndarray
```

Formula: `output[t] = alpha × x[t] + (1 - alpha) × output[t-1]`

- `alpha`: Smoothing coefficient (0, 1], 1 = no filtering, near 0 = strong filtering

---

## RewardPrinter

Reward statistics and printing utility for training.

```python
class RewardPrinter:
    PRINT_DETAIL = True
    def __init__(self, buffer_size: int = 100)
    def print_reward(self, message: str, reward: float = 0, coeff: float = 1.0)
```

---

## Rotation Utilities (`rotations`)

All functions support **batch operations**, with angles in **radians**.

### Conventions

- **Quaternion format**: `[w, x, y, z]` (MuJoCo standard)
- **Matrix format**: 3×3 rotation matrix

### Conversion Functions

| Function | Description |
|------|------|
| `mat2quat(mat)` | 3×3 matrix → `[w, x, y, z]` |
| `quat2mat(quat)` | `[w, x, y, z]` → 3×3 matrix |
| `euler2mat(euler)` | Euler angles → 3×3 matrix |
| `mat2euler(mat)` | 3×3 matrix → Euler angles |
| `euler2quat(euler)` | Euler angles → `[w, x, y, z]` |
| `quat2euler(quat)` | `[w, x, y, z]` → Euler angles |
| `quat2axisangle(quat)` | Quaternion → `(axis, theta)` axis-angle representation |

### Quaternion Operations

```python
rotations.quat_mul(q1, q2)              # Quaternion multiplication q1 * q2
rotations.quat_conjugate(q)             # Quaternion conjugate
rotations.quat_identity()               # Identity quaternion [1, 0, 0, 0]
rotations.quat_rot_vec(q, v0)           # Rotate a vector by a quaternion
```

### Euler Angle Operations

```python
rotations.subtract_euler(e1, e2)        # Euler angle difference
rotations.euler2point_euler(euler)      # Euler angles → point representation
rotations.point_euler2euler(pe)         # Point representation → Euler angles
rotations.normalize_angles(angles)      # Normalize angles to [-pi, pi]
rotations.round_to_straight_angles(angles)  # Round to right angles
```

### Other Utilities

```python
rotations.quat2point_quat(quat)         # Quaternion → point representation
rotations.point_quat2quat(pq)           # Point representation → quaternion
rotations.get_parallel_rotations()      # Get set of parallel rotations
rotations.unit_vector(data, axis, out)  # Unit vector
rotations.quat_slerp(q0, q1, fraction, shortestpath=True)  # Spherical linear interpolation
```

### Usage Example

```python
from orca_gym.utils import rotations
import numpy as np

# Quaternion → matrix
mat = rotations.quat2mat(np.array([1.0, 0.0, 0.0, 0.0]))

# Euler angles → quaternion
quat = rotations.euler2quat(np.array([0.0, np.pi/2, 0.0]))

# Spherical interpolation
q0 = np.array([1.0, 0.0, 0.0, 0.0])
q1 = rotations.euler2quat(np.array([np.pi/2, 0.0, 0.0]))
q_mid = rotations.quat_slerp(q0, q1, 0.5)

# Rotate a vector with a quaternion
v = np.array([1.0, 0.0, 0.0])
v_rot = rotations.quat_rot_vec(quat, v)

# Batch operations
eulers = np.array([[0.0, 0.5, 0.0], [np.pi/4, 0.0, 0.0], [0.0, 0.0, -np.pi/2]])
quats = rotations.euler2quat(eulers)  # (3, 4)
```
