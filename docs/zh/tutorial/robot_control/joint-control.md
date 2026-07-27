# 🦿 关节控制

底层的关节控制接口，直接操作 MuJoCo 执行器。

## set_ctrl —— 最底层控制

```python
# 直接设置所有执行器的控制值
ctrl = np.array([0.1, -0.2, 0.0, ...], dtype=np.float64) # (nu,)
env.set_ctrl(ctrl)
env.mj_step(n_frames)
env._sync_view() 
```

## 通过 do_simulation 原子化操作（推荐）

```python
# do_simulation = set_ctrl + mj_step + 自动同步 data
env.do_simulation(ctrl, n_frames=env.frame_skip)
```

## 关节位置控制

```python
# 直接设置关节目标位置
env.set_joint_qpos({
 "shoulder_joint": np.array([0.5]),
 "elbow_joint": np.array([-0.3]),
 "wrist_joint": np.array([1.2]),
})

# 必须 forward
env.mj_forward()
```

## 关节速度控制

```python
env.set_joint_qvel({
 "shoulder_joint": np.array([0.1]),
 "elbow_joint": np.array([-0.05]),
})

env.mj_forward()
```

## JointController — PD 控制

```python
from orca_utils.joint_controller import JointController

# 为每个关节创建一个 PD 控制器
controllers = {
 "shoulder": JointController(Kp=100.0, Ki=0.1, Kd=10.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)),
 "elbow": JointController(Kp=100.0, Ki=0.1, Kd=10.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)),
 "wrist": JointController(Kp=100.0, Ki=0.1, Kd=10.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)),
}

# 计算控制力矩（每个关节独立计算）
ctrl = np.zeros(env.model.nu)
target_angles = {"shoulder": 0.5, "elbow": -0.3, "wrist": 1.2}
for joint_name, target in target_angles.items():
 joint_id = env.model.joint_name2id(joint_name)
 dof_adr = env.jnt_dofadr(joint_name)
 ctrl[joint_id] = controllers[joint_name].compute_torque(
 target_qpos=target,
 current_qpos=env.data.qpos[dof_adr],
 current_qvel=env.data.qvel[dof_adr],
 dt=env.dt,
 )

# 应用（do_simulation 自动同步 data）
env.do_simulation(ctrl, env.frame_skip)
```

## PD 参数调优

| 参数 | 作用 | 典型值 |
|------|------|--------|
| kp | 比例增益 → 刚性/响应速度 | 10 ~ 500 |
| kd | 微分增益 → 阻尼/稳定性 | 1 ~ 50 |

- kp 太大 → 振荡
- kp 太小 → 跟踪慢
- kd 太大 → 响应迟钝
- kd 太小 → 欠阻尼

## 低通滤波

```python
from orca_utils.low_pass_filter import LowPassFilter

# 创建滤波器
filter = LowPassFilter(alpha=0.1, initial_output=np.zeros(env.model.nu))

# 在每步对 ctrl 滤波
raw_ctrl = compute_raw_ctrl(...)
smooth_ctrl = filter.apply(raw_ctrl)
env.do_simulation(smooth_ctrl, env.frame_skip)
```

## 关节限位检查

```python
def check_joint_limits(env):
 """检查所有关节是否在限位内"""
 for joint_name in list(env.model.get_joint_dict().keys()):
 joint_info = env.model.get_joint_byname(joint_name)
 if not joint_info["Limited"]:
 continue
 
 qpos = env.query_joint_qpos([joint_name])[joint_name]
 low, high = joint_info["Range"]
 
 if qpos[0] < low or qpos[0] > high:
 print(f"警告: {joint_name} 超出范围: "
 f"{qpos[0]:.3f} ∉ [{low:.3f}, {high:.3f}]")
```
