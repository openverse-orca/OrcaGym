# 🦿 关节控制

底层的关节控制接口，直接操作 MuJoCo 执行器。

## set_ctrl —— 最底层控制

```python
# 直接设置所有执行器的控制值
ctrl = np.array([0.1, -0.2, 0.0, ...], dtype=np.float64) # (nu,)
env.set_ctrl(ctrl)
env.mj_step(n_frames)
# 注意：set_ctrl + mj_step 不会自动同步 DataView。
# 推荐使用 do_simulation（下方），它内部封装了 set_ctrl + mj_step + sync_to_view。
# 如需手动同步，子类内部可调用 self._sync_view()（私有方法，仅限子类内部使用）。
```

## 通过 do_simulation 原子化操作（推荐）

```python
# do_simulation = set_ctrl + mj_step + 自动同步 data
env.do_simulation(ctrl, n_frames=env.frame_skip)
```

## 关节位置控制

`set_joint_qpos` 接受**全量 qpos 数组**（`np.ndarray`，形状 `(nq,)`），不接受字典。
需要按关节逐个写入 qpos 数组的对应地址，再整体设置：

```python
# 构造完整 qpos 数组后整体设置
qpos = env.data.qpos.copy()
qpos[env.jnt_qposadr("shoulder_joint")] = 0.5
qpos[env.jnt_qposadr("elbow_joint")] = -0.3
qpos[env.jnt_qposadr("wrist_joint")] = 1.2
env.set_joint_qpos(qpos)

# 必须 forward
env.mj_forward()
```

## 关节速度控制

`set_joint_qvel` 同样接受**全量 qvel 数组**（`np.ndarray`，形状 `(nv,)`）：

```python
qvel = env.data.qvel.copy()
qvel[env.jnt_dofadr("shoulder_joint")] = 0.1
qvel[env.jnt_dofadr("elbow_joint")] = -0.05
env.set_joint_qvel(qvel)

env.mj_forward()
```

## JointController — PD 控制

```python
from orca_gym.utils.joint_controller import JointController

# 为每个关节创建一个 PD 控制器
controllers = {
 "shoulder": JointController(Kp=100.0, Ki=0.1, Kd=10.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)),
 "elbow": JointController(Kp=100.0, Ki=0.1, Kd=10.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)),
 "wrist": JointController(Kp=100.0, Ki=0.1, Kd=10.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)),
}

# 计算控制力矩（每个关节独立计算）
# 注意：ctrl 数组按执行器（actuator）索引，而非关节索引。
# joint_name2id 返回关节 id，不能直接用作 ctrl 索引；
# 需用 actuator_name2id 获取执行器 id。
ctrl = np.zeros(env.model.nu)
target_angles = {"shoulder_actuator": 0.5, "elbow_actuator": -0.3, "wrist_actuator": 1.2}
# 假设执行器名与关节名一一对应（按模型 XML 定义），关节名用于查 dof 地址
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
from orca_gym.utils.low_pass_filter import LowPassFilter

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
