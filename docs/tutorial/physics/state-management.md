# 📐 状态管理

管理 MuJoCo 仿真中的状态（qpos/qvel/qacc）是正确使用 OrcaGym 的关键。

> 完整可运行代码见 [OrcaPlayground examples/euler/04_query_api/](https://github.com/OrcaGym/OrcaPlayground) 和 [06_jacobian/](https://github.com/OrcaGym/OrcaPlayground)。

## 状态数据布局

```
qpos (广义坐标):
 [body0_free_pos_xyz, body0_free_quat_wxyz, joint0_qpos, joint1_qpos, ...]
 长度 = model.nq

qvel (广义速度):
 [body0_free_lin_vel, body0_free_ang_vel, joint0_qvel, joint1_qvel, ...]
 长度 = model.nv

qacc (广义加速度):
 与 qvel 相同布局
 长度 = model.nv
```

## 获取状态

### 获取全局状态

```python
# env.data 是 OrcaGymDataView（零拷贝视图，只读）
# ⚠️ 零拷贝视图会随仿真步进自动更新，若需保存历史值请调用 .copy()
qpos = env.data.qpos          # (nq,) — 零拷贝视图，只读
qvel = env.data.qvel          # (nv,)
qacc = env.data.qacc          # (nv,)
qfrc_bias = env.data.qfrc_bias  # (nv,)
time = env.data.time          # float，仿真时间

# 需要保存历史值或修改时，使用 .copy()
qpos_snapshot = env.data.qpos.copy()
qvel_snapshot = env.data.qvel.copy()
```

### 获取特定关节状态

```python
# 按名称查询特定关节
joint_names = ["g1_left_knee_joint", "g1_right_knee_joint"]

# 位置
qpos_dict = env.query_joint_qpos(joint_names)
# → {"g1_left_knee_joint": array([0.523]), "g1_right_knee_joint": array([0.518]), ...}

# 速度
qvel_dict = env.query_joint_qvel(joint_names)

# 加速度
qacc_dict = env.query_joint_qacc(joint_names)
```

### 获取关节索引信息

```python
# 单个关节在全局数组中的地址
qpos_adr = env.jnt_qposadr("g1_left_knee_joint")   # qpos 中的起始索引
dof_adr = env.jnt_dofadr("g1_left_knee_joint")      # qvel/qacc 中的起始索引

# 从全局数组按地址切片
knee_angle = env.data.qpos[qpos_adr]                 # 铰链关节 qpos 长度 = 1
```

> **注意**：`env.data.qpos` 是**全局**数组（包含所有 body 的自由度和关节 qpos）。
> 在多 body 场景中，不能直接 `data.qpos[7:]` 访问 G1 关节 —— 必须通过 `jnt_qposadr`
> 按各关节地址逐段拼接。例如 G1 的 29 个旋转关节的 qpos 地址可能与 `data.qpos[7:]` 不连续。

## 设置状态

### 设置关节位置

```python
# 全量设置（设置完整的 qpos 数组，长度 = model.nq）
qpos = env.data.qpos.copy()
qpos[env.jnt_qposadr("g1_left_knee_joint")] = 0.6
env.set_joint_qpos(qpos)

# ⚠️ 重要：设置后必须 mj_forward() 更新派生量
env.mj_forward()
```

### 设置关节速度

```python
qvel = env.data.qvel.copy()
qvel[env.jnt_dofadr("g1_left_knee_joint")] = 0.1
env.set_joint_qvel(qvel)
env.mj_forward()
```

### 重置到初始状态

```python
# 在 reset_model 中使用 init_qpos/init_qvel
def reset_model(self):
    qpos = self.init_qpos.copy() + noise
    self.set_joint_qpos(qpos)
    self.mj_forward()
    self._sync_view()
    return self._get_obs(), {}
```

## 获取 Body 位姿

```python
# 按名称查询单个 body（通过 env.data）
body_pos = env.data.body_xpos("g1_pelvis")    # (3,) 世界位置
body_quat = env.data.body_xquat("g1_pelvis")  # (4,) [w, x, y, z]
body_mat = env.data.body_xmat("g1_pelvis")    # (9,) 3×3 旋转矩阵按行展开

# 批量查询（推荐：一次返回多个 body 的完整位姿）
body_dict = env.get_body_xpos_xmat_xquat(["g1_pelvis", "g1_torso_link"])
for name, pose in body_dict.items():
    pos = pose["xpos"]    # np.array([x, y, z])
    mat = pose["xmat"]    # np.array(9) — 3×3 矩阵按行展开
    quat = pose["xquat"]  # np.array([w, x, y, z])

# 常用：获取 pelvis 高度
pelvis_z = float(body_dict["g1_pelvis"]["xpos"][2])
```

## 获取 Sensor 数据

```python
# 查询传感器数据（按名称）
sensor_data = env.query_sensor_data([
    "g1_imu_quat",
    "g1_imu_gyro",
])

imu_quat = sensor_data["g1_imu_quat"]  # (4,) 姿态四元数
imu_gyro = sensor_data["g1_imu_gyro"]  # (3,) 角速度
```

## 状态同步黄金法则

> ⚠️ **修改状态 → mj_forward → _sync_view → 再读数据**

```python
# ✅ 正确的状态修改流程（Euler API）
env.set_joint_qpos(new_qpos)       # 1. 修改状态
env.mj_forward()                    # 2. 刷新派生量（body 位姿、传感器等）
env._sync_view()                    # 3. 同步到 DataView
current_qpos = env.data.qpos        # 4. 读取（零拷贝视图，反映最新值）

# ✅ do_simulation 后自动同步（推荐）
env.do_simulation(ctrl, n_frames)   # 内部自动: set_ctrl → mj_step → _sync_view
current_qpos = env.data.qpos        # 直接读即可

# ✅ 需要快照保存时
snapshot = env.data.qpos.copy()     # copy() 创建独立副本，不受后续更新影响
```

## 常见错误

| 错误 | 后果 | 修正 |
|------|------|------|
| 修改 qpos 后不 `mj_forward()` | body 位姿/传感器数据为旧值 | 加 `mj_forward()` |
| 不同步数据就读 data | 读到旧数据 | 调 `_sync_view()` |
| 不用 `.copy()` 就保存引用 | 数据被后续仿真步进覆盖 | 需要 `data.qpos.copy()` |
| 数组维度不对 | ValueError | 用 `jnt_qposadr` 检查地址和长度 |
| 不调用 `mj_forward()` 就读 body 位姿 | 读到旧位姿 | 修改 qpos 后必须 `mj_forward()` |
| 多 body 场景直接用 `data.qpos[7:]` | 读到其他 body 的数据 | 用 `jnt_qposadr` 逐关节切片拼接 |
