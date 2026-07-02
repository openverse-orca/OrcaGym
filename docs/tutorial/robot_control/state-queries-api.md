# 📡 状态查询 API — 读取关节、Body、传感器

本节介绍如何使用 OrcaGym 的**状态查询 API**，覆盖关节状态、Body 位姿、传感器、执行器力矩、接触信息等。

> 完整可运行代码见 [OrcaPlayground examples/euler/04_query_api/](https://github.com/OrcaGym/OrcaPlayground)。

---

## 完整示例：先看全貌

下面是一个**可以直接运行**的完整示例，演示了所有查询 API 的用法。
建议先通读一遍，再看后面的逐段解释。

```python
"""状态查询 API 完整演示

功能：演示 OrcaGymEulerEnv 提供的全套状态查询 API

用法（离线模式，不需要 Studio）:
    python query_demo.py
"""
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class QueryDemoEnv(OrcaGymEulerEnv):
    """演示所有查询 API 的环境"""

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
        """演示全套查询 API"""
        self.reset()
        agent = self.agent_name
        print(f"Agent 名称: {agent}\n")

        # ─── 1. 关节查询 ───
        print("=" * 50)
        print("1. 关节查询")
        print("=" * 50)

        knee = f"{agent}_left_knee_joint"
        hip = f"{agent}_left_hip_pitch_joint"

        # 按名称查询位置
        qpos = self.query_joint_qpos([knee, hip])
        print(f"  {knee}: {qpos[knee]} rad")
        print(f"  {hip}:   {qpos[hip]} rad")

        # 按名称查询速度
        qvel = self.query_joint_qvel([knee, hip])
        print(f"  {knee} 速度: {qvel[knee]} rad/s")

        # 获取关节在全局数组中的地址
        qpos_adr = self.jnt_qposadr(knee)
        dof_adr = self.jnt_dofadr(knee)
        print(f"  {knee} qpos 地址: {qpos_adr}, dof 地址: {dof_adr}")

        # ─── 2. Body 位姿查询 ───
        print("\n" + "=" * 50)
        print("2. Body 位姿查询")
        print("=" * 50)

        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        # 返回格式: {"g1_pelvis": {"xpos": array, "xmat": array, "xquat": array}}
        p = pelvis[f"{agent}_pelvis"]
        print(f"  pelvis 位置: [{p['xpos'][0]:.3f}, {p['xpos'][1]:.3f}, {p['xpos'][2]:.3f}]")
        print(f"  pelvis 高度: {p['xpos'][2]:.3f}m")
        print(f"  pelvis 四元数: [{p['xquat'][0]:.3f}, {p['xquat'][1]:.3f}, {p['xquat'][2]:.3f}, {p['xquat'][3]:.3f}]")

        # 也可通过 env.data 按名称查单个 body
        pelvis_z = self.data.body_xpos(f"{agent}_pelvis")[2]
        print(f"  (env.data) pelvis z: {pelvis_z:.3f}m")

        # ─── 3. 传感器查询 ───
        print("\n" + "=" * 50)
        print("3. 传感器查询")
        print("=" * 50)

        imu = self.query_sensor_data([f"{agent}_imu_quat", f"{agent}_imu_gyro"])
        print(f"  IMU 四元数: {imu[f'{agent}_imu_quat']}")
        print(f"  IMU 角速度: {imu[f'{agent}_imu_gyro']}")

        # ─── 4. 执行器力矩查询 ───
        print("\n" + "=" * 50)
        print("4. 执行器力矩查询")
        print("=" * 50)

        actuator_names = [f"{agent}_left_knee", f"{agent}_right_knee"]
        torques = self.query_actuator_torques(actuator_names)
        print(f"  左膝力矩: {torques[f'{agent}_left_knee']}")
        print(f"  右膝力矩: {torques[f'{agent}_right_knee']}")

        # ─── 5. 接触查询 ───
        print("\n" + "=" * 50)
        print("5. 接触查询")
        print("=" * 50)

        contacts = self.query_contact_simple()
        # 返回: [{"geom1": 12, "geom2": 34, ...}, ...]
        print(f"  活跃接触数: {len(contacts)}")
        if contacts:
            # 获取接触力（按列表索引）
            contact_ids = list(range(len(contacts)))
            forces = self.query_contact_force(contact_ids)
            max_normal = max(abs(f[0]) for f in forces.values())
            print(f"  最大法向力: {max_normal:.1f}N")
            # 显示前 3 个接触
            for i, c in enumerate(contacts[:3]):
                f = forces[i][:3]
                print(f"    接触 {i}: geom{c['geom1']}↔geom{c['geom2']}, 力={np.linalg.norm(f):.1f}N")

        # ─── 6. 质量查询 ───
        print("\n" + "=" * 50)
        print("6. 质量查询")
        print("=" * 50)

        torso_mass = self.body_subtree_mass(f"{agent}_torso_link")
        print(f"  躯干子树总质量: {torso_mass:.2f}kg")

        # ─── 7. 基座坐标系变换 ───
        print("\n" + "=" * 50)
        print("7. 基座坐标系变换")
        print("=" * 50)

        # torso_link 在 pelvis 坐标系中的位置
        torso_B = self.query_position_body_B(
            f"{agent}_torso_link", f"{agent}_pelvis"
        )
        print(f"  躯干在骨盆坐标系: {torso_B}")
        print(f"  躯干在骨盆上方: {torso_B[2]:.3f}m (z 分量)")

        # ─── 8. Site 查询 ───
        print("\n" + "=" * 50)
        print("8. Site 查询")
        print("=" * 50)

        imu_site = self.query_site_pos_and_mat([f"{agent}_imu"])
        site_pos = imu_site[f"{agent}_imu"]["xpos"]
        print(f"  IMU site 位置: {site_pos}")

        # Site 速度
        xvalp, xvalr = self.query_site_xvalp_xvalr([f"{agent}_imu"])
        print(f"  IMU site 线速度: {xvalp[f'{agent}_imu']}")
        print(f"  IMU site 角速度: {xvalr[f'{agent}_imu']}")

        # ─── 9. 从 env.data 直接读取 ───
        print("\n" + "=" * 50)
        print("9. env.data 零拷贝视图")
        print("=" * 50)

        print(f"  data.qpos.shape: {self.data.qpos.shape}")
        print(f"  data.qvel.shape: {self.data.qvel.shape}")
        print(f"  data.time:       {self.data.time:.4f}s")
        print(f"  model.nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")

        print("\n✅ 所有查询 API 演示完成")

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
        skip_grpc_load=True,   # 离线模式
    )
    env.demo_all_queries()
    env.close()
```

---

## 逐段解释

### 查询 API 总览

`OrcaGymEulerEnv` 提供以下公共查询方法，**全部按名称访问，无需记 id**：

| 类别 | 方法 | 返回类型 | 说明 |
|------|------|----------|------|
| **关节** | `query_joint_qpos(names)` | `dict[str, np.ndarray]` | 关节位置 |
| | `query_joint_qvel(names)` | `dict[str, np.ndarray]` | 关节速度 |
| | `query_joint_qacc(names)` | `dict[str, np.ndarray]` | 关节加速度 |
| | `jnt_qposadr(name)` | `int` | qpos 中的起始地址 |
| | `jnt_dofadr(name)` | `int` | qvel/qacc 中的起始地址 |
| **Body** | `get_body_xpos_xmat_xquat(names)` | `dict[str, dict]` | 世界位姿（位置+矩阵+四元数） |
| | `get_body_xpos_xmat_xquat_xvel(names)` | `dict[str, dict]` | 位姿 + 线速度 |
| **Site** | `query_site_pos_and_mat(names)` | `dict[str, dict]` | Site 位姿 |
| | `query_site_xvalp_xvalr(names)` | `tuple[dict, dict]` | Site 速度（线+角） |
| **传感器** | `query_sensor_data(names)` | `dict[str, np.ndarray]` | 传感器读数 |
| **执行器** | `query_actuator_torques(names)` | `dict[str, np.ndarray]` | 力矩 |
| **接触** | `query_contact_simple()` | `list[dict]` | 接触对列表 |
| | `query_contact_force(ids)` | `dict[int, np.ndarray]` | 接触力（6D） |
| **质量** | `body_subtree_mass(name)` | `float` | 子树总质量 |
| **基座变换** | `query_position_body_B(ee, base)` | `np.ndarray(3,)` | 相对位置 |

### 1. 关节查询

```python
# 准备关节名称（必须带 agent 前缀）
agent = "g1"
joint_names = [f"{agent}_left_knee_joint", f"{agent}_right_knee_joint"]

# 查询
qpos = env.query_joint_qpos(joint_names)
# → {"g1_left_knee_joint": array([0.523]), "g1_right_knee_joint": array([0.518])}

qvel = env.query_joint_qvel(joint_names)
# → {"g1_left_knee_joint": array([-0.1]), ...}
```

**实现原理**：`query_joint_qpos` 内部通过 `jnt_qposadr` 从 `data.qpos` 按地址切片：

```python
# 内部等价于：
for jn in joint_names:
    addr = env.jnt_qposadr(jn)
    result[jn] = env.data.qpos[addr:addr + 1]  # 铰链关节长度 = 1
```

> ⚠️ **重要**：`env.data.qpos` 是**全局**数组（包含所有 body 的自由度 + 所有关节）。
> 多 body 场景中**不能**用 `data.qpos[7:]` 直接取 G1 关节，必须通过 `jnt_qposadr`
> 逐关节按地址拼接。

### 2. Body 位姿查询

```python
pelvis = env.get_body_xpos_xmat_xquat(["g1_pelvis", "g1_torso_link"])

# 返回格式：
# {
#   "g1_pelvis": {
#     "xpos": np.array([0.0, 0.0, 0.78]),   # 世界位置 (3,)
#     "xmat": np.array([...]),                # 旋转矩阵 (9,) 按行展开
#     "xquat": np.array([1.0, 0, 0, 0]),     # 四元数 [w,x,y,z] (4,)
#   },
#   "g1_torso_link": { ... }
# }

# 常用：获取 pelvis 高度
pelvis_z = float(pelvis["g1_pelvis"]["xpos"][2])
```

也可通过 `env.data` 单查：
```python
env.data.body_xpos("g1_pelvis")    # (3,)  只返回位置
env.data.body_xquat("g1_pelvis")   # (4,)  只返回四元数
```

### 3. 传感器查询

```python
sensor_data = env.query_sensor_data(["g1_imu_quat", "g1_imu_gyro"])

# 按名称取结果
imu_quat = sensor_data["g1_imu_quat"]  # (4,) 姿态四元数
imu_gyro = sensor_data["g1_imu_gyro"]  # (3,) 角速度
```

> ⚠️ 传感器数据在 `mj_forward()` 或 `do_simulation()` 后才更新。
> 修改 qpos 后不 forward 就读传感器 → 读到旧数据。

### 4. 接触查询

```python
# 第一步：获取接触列表
contacts = env.query_contact_simple()
# → [{"geom1": 12, "geom2": 34, "dist": 0.001, "pos": [...], "frame": [...]}, ...]

# 第二步：获取接触力（按列表索引，不是 contact 字典中的某个 id 字段）
contact_ids = list(range(len(contacts)))
forces = env.query_contact_force(contact_ids)
# → {0: array([normal, shear1, shear2, torque1, torque2, torque3]), 1: ...}

# 第 0 分量是法向力
max_normal = max(abs(f[0]) for f in forces.values())
```

> **注意**：`query_contact_simple()` 返回的字典 key 是**小写** `"geom1"` / `"geom2"`，
> 不是大写 `"Geom1"` / `"Geom2"`。

### 5. 基座坐标系变换

```python
# 纯 NumPy 变换，不依赖 MuJoCo
torso_in_pelvis = env.query_position_body_B("g1_torso_link", "g1_pelvis")
# → array([x, y, z])  — torso_link 在 pelvis 坐标系中的位置
```

### 6. `env.data` 零拷贝视图

`env.data` 是 `OrcaGymDataView`，提供**零拷贝只读视图**。数据随仿真步进自动更新：

```python
env.data.qpos          # (nq,) 广义坐标
env.data.qvel          # (nv,) 广义速度
env.data.qacc          # (nv,) 广义加速度
env.data.time          # float 仿真时间
env.data.qfrc_bias     # (nv,) 偏置力
env.data.xfrc_applied  # (nbody, 6) 施加的外力（只读）
```

> ⚠️ `env.data.qpos` 是零拷贝视图。若需保存历史值，调用 `.copy()`。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `KeyError: 'g1_left_knee_joint'` | 关节名缺少 agent 前缀 | 用 `f"{agent}_{suffix}"` 拼接 |
| `data.qpos[7:]` 得到错误值 | 多 body 场景地址不连续 | 用 `jnt_qposadr` 逐关节切片 |
| `query_contact_force` 返回空 | 刚加载时无接触 | 先步进几步让机器人触地 |
| 传感器数据是旧值 | 没调 `mj_forward()` | `set_joint_qpos` 后必须 forward |

---

## 下一步

掌握了状态查询，接下来学习如何**施加外力和写入状态**：[🔄 外力应用与 IK](../physics/force-apply.md)。
