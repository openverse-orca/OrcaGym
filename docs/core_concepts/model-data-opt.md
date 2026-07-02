# 📊 Model / Data / Config

OrcaGym 将仿真信息分为三个清晰的层级。

---

## OrcaGymModel — 静态模型信息

`OrcaGymModel` 包含所有**在仿真过程中不变**的信息，可通过 `env.model` 访问：

```python
model = env.model

# 维度信息
print(model.nq)          # 广义坐标数
print(model.nv)          # 自由度数
print(model.nu)          # 执行器数
print(model.nbody)       # body 总数
print(model.njnt)        # 关节总数

# 名称到 ID 的映射
body_id = model.body_name2id("base_link")
joint_id = model.joint_name2id("shoulder")
actuator_id = model.actuator_name2id("shoulder_actuator")

# 获取执行器控制范围（用于构建 action_space）
ctrl_range = model.get_actuator_ctrlrange()  # shape: (nu, 2)

# 列出所有 body 名称
body_names = model.get_body_names()
```

### Model 中存储的信息

| 信息类型 | 包含内容 |
|----------|----------|
| Body | 质量、惯性、父子关系、位姿 |
| Joint | 类型、范围、轴、刚度、阻尼 |
| Actuator | 控制范围、齿轮比、传输类型 |
| Geom | 形状、摩擦、碰撞参数 |
| Site | 标记点位置、尺寸 |
| Sensor | 类型、维度 |
| Eq | 等式约束类型、目标对象 |
| Mocap | mocap body 映射 |

---

## OrcaGymDataView — 动态仿真状态

`OrcaGymDataView` 是仿真状态的只读视图，可通过 `env.data` 访问。`do_simulation()` 后自动更新。

```python
data = env.data

# 核心状态
qpos = data.qpos               # (nq,) 广义坐标
qvel = data.qvel               # (nv,) 广义速度
qacc = data.qacc               # (nv,) 广义加速度
time = data.time               # 仿真时间（标量）

# 外力与约束
xfrc_applied = data.xfrc_applied       # 外力（只读）
cfrc_ext = data.cfrc_ext              # 外部约束力 (nbody, 6)
contact = data.contact                 # 接触列表

# 按名称查询 body/site（无需知道 ID）
body_pos = data.body_xpos("torso_link")       # (3,) 世界坐标
body_quat = data.body_xquat("torso_link")    # (4,) [w,x,y,z]
body_vel = data.body_cvel("torso_link")       # (6,) [ang(3), lin(3)]
site_pos = data.site_xpos("imu")             # (3,)
geom_pos = data.geom_xpos("box_geom")        # (3,)
mass = data.body_subtree_mass("torso_link")   # float
```

### 状态读取的时机

```
do_simulation(ctrl, n_frames)   ← 仿真步进
  └─▶ 步进完成，env.data 自动更新
  └─▶ 可直接读取 env.data.qpos 等
```

> ⚠️ **重要**：`do_simulation()` 返回后 `env.data` 已自动同步。手动调用 `mj_forward()` 后如需读取 `env.data`，需要调用 `env._sync_view()` 同步。

---

## SimConfig — 求解器配置

`SimConfig` 提供仿真参数的读写接口，通过 `env.sim_config` 访问。修改在下次仿真步进时生效。

```python
sim_config = env.sim_config

# 读写参数
sim_config.timestep = 0.002     # 物理时间步长
sim_config.iterations = 100     # 求解器迭代次数
sim_config.integrator = 1       # 积分器（0=Euler, 1=RK4）
sim_config.gravity = np.array([0., 0., -9.81])  # 重力

# 批量设置
sim_config.load_from_dict({
    "integrator": 0,
    "iterations": 100,
})

# 导出配置
config_dict = sim_config.to_dict()
```

---

## 环境时间步长

```python
env.dt = env.sim_config.timestep * env.frame_skip
```

控制频率：`control_hz = 1.0 / env.dt`

例如 `timestep = 0.001, frame_skip = 20` → `dt = 0.020s, control_hz = 50 Hz`

---

## 关节类型与 qpos/qvel 维度

不同关节类型在 `qpos` 和 `qvel` 中占用不同数量的元素：

| 关节类型 | qpos 大小 | qvel 大小 | 示例 |
|----------|-----------|-----------|------|
| FREE | 7 (3 pos + 4 quat) | 6 (3 lin + 3 ang) | 自由飞行体 |
| BALL | 4 (quaternion) | 3 (angular velocity) | 球关节 |
| HINGE | 1 (angle) | 1 (angular velocity) | 旋转关节 |
| SLIDE | 1 (displacement) | 1 (linear velocity) | 滑动关节 |
