# 📡 读取状态 — 关节、body、传感器

上一节我们只读了 `self.data.qpos` 和 `self.data.qvel`。这一节，你将学会用 OrcaGym 提供的**查询 API** 来获取更丰富的状态信息。

> 完整可运行代码见 [OrcaPlayground examples/euler/04_query_api/](https://github.com/OrcaGym/OrcaPlayground)。

---

## 完整示例：先看全貌

下面是一个**可以直接运行**的完整示例，构造了一个 `StateDumper` 调试工具，
演示了所有状态查询 API 的用法。

```python
"""状态读取完整演示：关节、Body、Site、传感器、执行器力矩"""
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class StateDumper:
    """调试工具：一键打印环境当前所有状态"""

    def __init__(self, env):
        self.env = env

    def dump(self):
        env = self.env
        print("=" * 60)
        print(f"仿真时间: {env.data.time:.3f}s")

        # ── 关节 ──
        joint_names = list(env.model.get_joint_dict().keys())[:5]
        qpos = env.query_joint_qpos(joint_names)
        print("\n关节位置（前 5 个）:")
        for name in joint_names:
            print(f"  {name}: {qpos[name]}")

        # ── 末端执行器 ──
        try:
            ee_site = env.site("end_effector")
            sites = env.query_site_pos_and_quat([ee_site])
            ee_pos = sites[ee_site]["xpos"]
            print(f"\n末端位置: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
        except Exception:
            print("\n(无 end_effector site)")

        # ── 接触 ──
        contacts = env.query_contact_simple()
        print(f"\n接触点数量: {len(contacts)}")

        print("=" * 60)


class StateQueryDemo(OrcaGymEulerEnv):
    """演示所有状态查询 API 的环境"""

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

    # ─── 查询方法 ───

    def check_joints(self):
        """查询所有关节的位置和速度"""
        joint_names = list(self.model.get_joint_dict().keys())
        qpos = self.query_joint_qpos(joint_names)
        qvel = self.query_joint_qvel(joint_names)

        print(f"共 {len(joint_names)} 个关节:")
        for name in joint_names[:5]:  # 显示前 5 个
            pos = qpos[name]
            vel = qvel[name]
            pos_str = f"{pos[0]:.3f}" if len(pos) == 1 else f"{pos}"
            print(f"  {name:25s}: pos={pos_str}, vel={vel}")
        return qpos, qvel

    def inspect_joint(self, joint_name):
        """查看单个关节的详细信息"""
        info = self.model.get_joint_byname(joint_name)
        print(f"关节: {joint_name}")
        print(f"  类型: {info['Type']}")          # hinge / slide / free / ball
        print(f"  有限位: {info['Limited']}")
        if info['Limited']:
            print(f"  范围: [{info['Range'][0]:.3f}, {info['Range'][1]:.3f}] rad")

        qpos_addr = self.jnt_qposadr(joint_name)  # qpos 中的起始位置
        dof_addr = self.jnt_dofadr(joint_name)     # qvel 中的起始位置
        print(f"  qpos 地址: {qpos_addr}, qvel 地址: {dof_addr}")

    def check_body_pose(self):
        """查询关键 body 的位置和姿态"""
        body_names = list(self.model.get_body_names())
        print(f"共 {len(body_names)} 个 body（前 10 个）:")
        for name in body_names[:10]:
            print(f"  - {name}")

        # 批量查询（推荐）
        if len(body_names) >= 2:
            body_dict = self.get_body_xpos_xmat_xquat(body_names[:2])
            for name, pose in body_dict.items():
                print(f"\n{name}:")
                print(f"  位置: {pose['xpos']}")
                print(f"  四元数: {pose['xquat']}")

    def check_end_effector(self):
        """查询末端执行器的位姿和速度"""
        ee_site = self.site("end_effector")
        site_data = self.query_site_pos_and_quat([ee_site])
        ee = site_data[ee_site]
        print(f"末端执行器 (site: {ee_site}):")
        print(f"  位置: {ee['xpos']}")
        print(f"  四元数: {ee['xquat']}")

        # 速度
        linear_vel, angular_vel = self.query_site_xvalp_xvalr([ee_site])
        print(f"  线速度: {linear_vel[ee_site]}")
        print(f"  角速度: {angular_vel[ee_site]}")

    def read_sensors(self):
        """读取传感器数据"""
        sensor_names = list(self.model.gen_sensor_dict().keys())
        print(f"传感器列表 ({len(sensor_names)} 个):")
        for name in sensor_names:
            info = self.model.gen_sensor_dict()[name]
            print(f"  {name}: type={info['Type']}, dim={info['Dim']}")

        if sensor_names:
            sensor_data = self.query_sensor_data(sensor_names[:3])
            for name, data in sensor_data.items():
                print(f"  {name}: {data}")

    def read_actuator_torques(self):
        """查看执行器力矩"""
        names = [self.model.actuator_id2name(i) for i in range(self.model.nu)]
        torques = self.query_actuator_torques(names[:3])
        for name, t in torques.items():
            print(f"  {name}: {t}")

    # ─── 演示入口 ───

    def demo(self):
        self.reset()
        print(f"环境: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}\n")

        self._dumper.dump()
        self.check_joints()
        self.check_body_pose()

    # ─── Gymnasium 接口 ───
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
        skip_grpc_load=True,  # 离线模式
    )
    env.demo()
    env.close()
```

---

## 逐段解释

### 查询 API 总览

OrcaGym 提供按**名称**查询的 API（无需记 id）：

| 查询对象 | 方法 | 返回 |
|----------|------|------|
| 关节位置 | `query_joint_qpos(names)` | `dict[str, array]` |
| 关节速度 | `query_joint_qvel(names)` | `dict[str, array]` |
| Body 位姿 | `get_body_xpos_xmat_xquat(names)` | `dict[str, dict]` |
| Body 位置（单查） | `env.data.body_xpos(name)` | `(3,)` |
| Site 位姿 | `query_site_pos_and_quat(names)` | `dict[str, dict]` |
| Site 速度 | `query_site_xvalp_xvalr(names)` | `tuple[dict, dict]` |
| 传感器 | `query_sensor_data(names)` | `dict[str, array]` |
| 执行器力矩 | `query_actuator_torques(names)` | `dict[str, array]` |

### 1. 关节查询

```python
qpos = env.query_joint_qpos(["robot_0_joint1", "robot_0_joint2"])
# → {"robot_0_joint1": array([0.52]), "robot_0_joint2": array([-0.31])}

qvel = env.query_joint_qvel(["robot_0_joint1", "robot_0_joint2"])
```

**查看关节详细信息**：
```python
info = env.model.get_joint_byname("robot_0_joint1")
# → {"Type": "hinge", "Limited": True, "Range": [-3.14, 3.14], ...}
```

**关节在全局数组中的地址**：
```python
qpos_addr = env.jnt_qposadr("robot_0_joint1")   # 在 qpos 中的索引
dof_addr = env.jnt_dofadr("robot_0_joint1")      # 在 qvel 中的索引
```

### 2. Body 位姿查询

```python
# 批量查询（推荐：一次返回多个 body）
body_dict = env.get_body_xpos_xmat_xquat(["base_link", "ee_link"])
for name, pose in body_dict.items():
    print(f"{name}: pos={pose['xpos']}, quat={pose['xquat']}")

# 单个查询（通过 env.data）
pos = env.data.body_xpos("base_link")     # (3,) 世界位置
quat = env.data.body_xquat("base_link")   # (4,) [w,x,y,z]
```

### 3. Site 查询

Site 是 MuJoCo 中的标记点，通常标记**末端执行器**、**IMU 位置**等：

```python
# 位姿
sites = env.query_site_pos_and_quat(["robot_0_end_effector"])
ee = sites["robot_0_end_effector"]
print(f"末端位置: {ee['xpos']}")

# 速度（线速度 + 角速度）
lin_vel, ang_vel = env.query_site_xvalp_xvalr(["robot_0_end_effector"])
```

### 4. 传感器查询

```python
# 列出所有传感器
sensor_names = list(env.model.gen_sensor_dict().keys())

# 读取数据
data = env.query_sensor_data(["imu_acc", "imu_gyro"])
```

常用传感器类型：
| MuJoCo 类型 | 用途 | 维度 |
|-------------|------|------|
| `accelerometer` | 线加速度 | (3,) |
| `gyro` | 角速度 | (3,) |
| `force_torque` | 六维力/力矩 | (6,) |
| `jointpos` | 关节位置 | (1,) |
| `jointvel` | 关节速度 | (1,) |

### 状态更新的时机

```
mj_forward() 或 do_simulation()
        │
        ▼
  所有派生量更新（传感器、接触力、body位姿...）
        │
        ▼
  你的查询方法 ← 现在可以读到最新值
```

> ⚠️ **查询前必须先 forward/step**
> ```python
> # ✅ 正确
> env.do_simulation(ctrl, frame_skip)   # 内部 step + sync
> pos = env.query_joint_qpos(["joint_0"])
> 
> # ❌ 错误 — 读到的可能是旧数据
> env.set_joint_qpos(...)
> pos = env.query_joint_qpos(["joint_0"])  # 还没 forward！
> ```

---

## 下一步

能读状态了，接下来学习**让机器人动起来**：[🦾 让机器人动起来](move-a-joint.md)。
