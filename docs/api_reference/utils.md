# 🔧 Utils API

工具函数和控制器，提供逆运动学、关节控制、旋转转换等实用工具。

## 类与函数概览

| 名称 | 说明 |
|------|------|
| `InverseKinematicsController` | 基于雅可比的逆运动学求解器 |
| `JointController` | PID+速度前馈的关节力矩控制器 |
| `pd_control` | 简单 PD 控制器函数 |
| `LowPassFilter` | 一阶低通滤波器 |
| `RewardPrinter` | 训练奖励统计与打印 |
| `rotations` | 旋转表示转换工具集 |

---

## InverseKinematicsController

基于雅可比矩阵 + 阻尼最小二乘法的逆运动学求解器。

### 构造

```python
class InverseKinematicsController:
    def __init__(
        self,
        env: RobomimicEnv,          # 环境对象
        site_id: int,               # 末端执行器 site 的 ID
        dof_indices: list[int],     # 受控关节的 DOF 索引列表
        lamba_value: float = 1e-3,  # 阻尼系数
        alpha_value: float = 0.2,   # 步长缩放系数
    )
```

### 方法

```python
def set_goal(pos: np.ndarray, quat: np.ndarray)         # 设置目标位姿
def set_lambda(lambda_value: float)                     # 设置阻尼系数
def set_alpha(alpha_value: float)                       # 设置步长
def compute_inverse_kinematics() -> np.ndarray          # 计算增量关节角度 dq
```

### 使用示例

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

# 设置目标位姿
ik.set_goal(
    pos=np.array([0.5, 0.0, 0.3]),
    quat=np.array([1.0, 0.0, 0.0, 0.0]),
)

# 每步计算逆运动学
dq = ik.compute_inverse_kinematics()
current_qpos += dq
```

---

## JointController

带积分抗饱和的 PID + 速度前馈单关节力矩控制器。

### 构造

```python
class JointController:
    def __init__(
        self,
        Kp: float = 10.0,            # 位置比例增益
        Ki: float = 0.1,             # 积分增益
        Kd: float = 2.0,             # 速度微分增益
        Kv: float = 5.0,             # 速度误差反馈增益
        max_speed: float = 80.0,     # 最大允许目标速度 (rad/s)
        ctrlrange: tuple = (-80, 80),# 驱动器力矩范围 (min, max) Nm
    )
```

### 方法

```python
def compute_torque(
    self,
    target_qpos: float,        # 目标角度 (rad)
    current_qpos: float,       # 当前角度 (rad)
    current_qvel: float,       # 当前速度 (rad/s)
    dt: float,                 # 仿真步长 (s)
) -> float                    # 输出力矩 (Nm)
```

### 使用示例

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

简单 PD 控制器函数。

```python
def pd_control(
    target_q: np.ndarray,      # 目标位置
    q: np.ndarray,             # 当前位置
    kp: float | np.ndarray,    # 位置增益
    target_dq: np.ndarray,     # 目标速度
    dq: np.ndarray,            # 当前速度
    kd: float | np.ndarray,    # 速度增益
) -> np.ndarray               # 输出力矩
```

计算公式: `torque = (target_q - q) × kp + (target_dq - dq) × kd`

---

## LowPassFilter

一阶指数平滑低通滤波器。

```python
class LowPassFilter:
    def __init__(self, alpha: float, initial_output: np.ndarray)
    def apply(self, x: np.ndarray) -> np.ndarray
```

公式: `output[t] = alpha × x[t] + (1 - alpha) × output[t-1]`

- `alpha`: 平滑系数 (0, 1]，1 = 不过滤，接近 0 = 强滤波

---

## RewardPrinter

训练过程中的奖励统计与打印工具。

```python
class RewardPrinter:
    PRINT_DETAIL = True
    def __init__(self, buffer_size: int = 100)
    def print_reward(self, message: str, reward: float = 0, coeff: float = 1.0)
```

---

## 旋转工具 (`rotations`)

所有函数支持 **batch 操作**，角度单位为 **弧度**。

### 约定

- **四元数格式**: `[w, x, y, z]`（MuJoCo 标准）
- **矩阵格式**: 3×3 旋转矩阵

### 转换函数

| 函数 | 说明 |
|------|------|
| `mat2quat(mat)` | 3×3 矩阵 → `[w, x, y, z]` |
| `quat2mat(quat)` | `[w, x, y, z]` → 3×3 矩阵 |
| `euler2mat(euler)` | 欧拉角 → 3×3 矩阵 |
| `mat2euler(mat)` | 3×3 矩阵 → 欧拉角 |
| `euler2quat(euler)` | 欧拉角 → `[w, x, y, z]` |
| `quat2euler(quat)` | `[w, x, y, z]` → 欧拉角 |
| `quat2axisangle(quat)` | 四元数 → `(axis, theta)` 轴角表示 |

### 四元数运算

```python
rotations.quat_mul(q1, q2)              # 四元数乘法 q1 * q2
rotations.quat_conjugate(q)             # 四元数共轭
rotations.quat_identity()               # 单位四元数 [1, 0, 0, 0]
rotations.quat_slerp(q0, q1, fraction)  # 球面线性插值
rotations.quat_rot_vec(q, v)            # 用四元数旋转向量 v
```

### 角度处理

```python
rotations.normalize_angles(angles)               # 归一化到 [-π, π]
rotations.subtract_euler(e1, e2)                 # 欧拉角差值
rotations.round_to_straight_angles(angles)       # 舍入到最近的 90° 倍数
```

### 使用示例

```python
from orca_gym.utils import rotations
import numpy as np

# 四元数 → 矩阵
mat = rotations.quat2mat(np.array([1.0, 0.0, 0.0, 0.0]))

# 欧拉角 → 四元数
quat = rotations.euler2quat(np.array([0.0, np.pi/2, 0.0]))

# 球面插值
q0 = np.array([1.0, 0.0, 0.0, 0.0])
q1 = rotations.euler2quat(np.array([np.pi/2, 0.0, 0.0]))
q_mid = rotations.quat_slerp(q0, q1, 0.5)

# 用四元数旋转向量
v = np.array([1.0, 0.0, 0.0])
v_rot = rotations.quat_rot_vec(quat, v)

# Batch 操作
eulers = np.array([[0.0, 0.5, 0.0], [np.pi/4, 0.0, 0.0], [0.0, 0.0, -np.pi/2]])
quats = rotations.euler2quat(eulers)  # (3, 4)
```
