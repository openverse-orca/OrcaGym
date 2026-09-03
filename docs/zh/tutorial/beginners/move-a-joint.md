# 🦾 让机器人动起来 — 控制单个关节

前面我们只是"看"，这一节开始**让机器人动起来**。我们从最简单的开始：理解 `qpos`/`qvel`，手动控制关节。

> 本节基于 [OrcaPlayground examples/euler/01_hello_euler/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/01_hello_euler) 的场景和样例进行讲解，
> 其场景 XML 为 [assets/scenes/simple_pendulum.xml](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/assets/scenes/simple_pendulum.xml)。
> 样例代码 (`simple_env.py`) 使用直接下标访问，本节为展示**按名称查询和设置**的写法进行了改写。

本节示例**沿用该 XML**，关键名称对照如下（这些名称在 XML 中定义，后续读写都用它们）：

| 元素 | XML 中的 `name` | 含义 |
|------|------------------|------|
| body | `pendulum` | 摆杆本体 |
| joint | `hinge` | 绕 Y 轴的铰链关节 |
| geom | `arm` | 摆杆几何体 |
| site | `tip` | 摆杆末端站点 |
| actuator | `hinge_motor` | 电机执行器（`joint="hinge"`） |

---

## 完整示例：先看全貌

下面是一个**可以直接运行**的完整示例，展示三种控制/查询关节状态的方法。
**统一使用名称**（而非下标）查询和设置，便于把代码和 XML 对应起来。

```python
"""关节控制完整演示：力矩驱动 → 按名称设位置 → 按名称查询"""
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

# simple_pendulum.xml 中的名称
JOINT_NAME   = "hinge"        # <joint name="hinge">
ACTUATOR_NAME = "hinge_motor"  # <motor name="hinge_motor">
BODY_NAME    = "pendulum"     # <body name="pendulum">
SITE_NAME    = "tip"          # <site name="tip">


class JointControlDemo(OrcaGymEulerEnv):
    """关节控制演示环境（离线模式，无需 Studio）"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            skip_grpc_load=kwargs.pop("skip_grpc_load", True),
            **kwargs,
        )

    # ─── 方法 1：力矩驱动（经过物理）⭐ 推荐 ───
    def demo_torque_drive(self, actuator_name, steps=200):
        """用恒定力矩驱动关节，观察它在重力+惯性下的自然运动。

        这是"经过物理"的方式：力矩 → 加速度 → 速度 → 位置
        """
        # 查执行器力矩范围（按名称查，无需记下标）
        ctrlrange = self.model.get_actuator_ctrlrange()
        act_id = self.model.actuator_name2id(actuator_name)
        max_torque = ctrlrange[act_id, 1]
        print(f"执行器 {actuator_name} 力矩范围: "
              f"[{ctrlrange[act_id, 0]:.1f}, {max_torque:.1f}] N·m")

        for i in range(steps):
            ctrl = np.zeros(self.model.nu, dtype=np.float64)

            # 前半段正向力矩，后半段反向 → 观察往复运动
            if i < steps // 2:
                ctrl[act_id] = 0.3 * max_torque   # 30% 正向
            else:
                ctrl[act_id] = -0.3 * max_torque  # 30% 反向

            self.do_simulation(ctrl, self.frame_skip)

            if i % 20 == 0:
                # 按名称查关节状态（dict 返回，直观对应）
                qpos = self.query_joint_qpos([JOINT_NAME])
                qvel = self.query_joint_qvel([JOINT_NAME])
                pos = float(qpos[JOINT_NAME][0])
                vel = float(qvel[JOINT_NAME][0])
                print(f"  Step {i:3d}: pos={pos:+.4f} rad, "
                      f"vel={vel:+.4f} rad/s, torque={ctrl[act_id]:+.2f}")

    # ─── 方法 2：按名称设位置（正弦摆动，适合 reset）───
    def demo_wiggle(self, joint_name, amplitude=0.5, steps=200):
        """让关节做正弦摆动。直接设 qpos 方式，不经过物理。"""
        # 先用名称查初始位置
        init_pos = float(self.query_joint_qpos([joint_name])[joint_name][0])
        print(f"关节 {joint_name} 初始位置: {init_pos:.3f} rad")

        for i in range(steps):
            target_angle = amplitude * np.sin(i * 0.1)

            # 合规写入：copy → 按名称定位 → 修改 → set → forward
            new_qpos = self.data.qpos.copy()
            qpos_addr = self.jnt_qposadr(joint_name)  # 按名称查地址
            new_qpos[qpos_addr] = target_angle

            self.set_joint_qpos(new_qpos)             # 全量写入
            self.set_joint_qvel(np.zeros(self.model.nv))
            self.mj_forward()                         # 更新派生量
            self._sync_view()                          # 同步 DataView

            if i % 20 == 0:
                # 再用名称查询验证
                actual = float(self.query_joint_qpos([joint_name])[joint_name][0])
                print(f"  Step {i:3d}: 目标={target_angle:+.3f}, "
                      f"实际={actual:+.3f}")

    # ─── 方法 3：按名称查 body / site 位姿 ───
    def demo_query_body_site(self, body_name, site_name):
        """展示 body / site 的按名称查询（名称同样来自 XML）。"""
        # Body 位姿（dict 返回，键即 XML 中的 body name）
        body_pose = self.get_body_xpos_xmat_xquat([body_name])
        bp = body_pose[body_name]
        print(f"Body '{body_name}':")
        print(f"  位置: {bp['xpos']}")
        print(f"  四元数: {bp['xquat']}")

        # Site 位姿
        site_pose = self.query_site_pos_and_mat([site_name])
        sp = site_pose[site_name]
        print(f"Site '{site_name}':")
        print(f"  位置: {sp['xpos']}")

    # ─── 工具：打印 qpos 布局 ───
    def print_qpos_layout(self):
        """打印 qpos 布局，帮助理解每个关节占几个元素"""
        offset = 0
        for name in self.model.get_joint_dict().keys():
            # 按名称查每个关节在 qpos 中的起始地址
            qpos_addr = self.jnt_qposadr(name)
            # 不同关节类型的 qpos 长度不同（hinge/slide=1, ball=4, free=7）
            info = self.model.get_joint_byname(name)
            nq = 1  # 默认 hinge/slide
            print(f"  qpos[{qpos_addr:2d}:{qpos_addr+nq:2d}]  {name}  (nq={nq})")
            offset = qpos_addr + nq

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
        # 按名称读取，组装观测向量
        qpos = self.query_joint_qpos([JOINT_NAME])[JOINT_NAME]
        qvel = self.query_joint_qvel([JOINT_NAME])[JOINT_NAME]
        return np.concatenate([qpos, qvel]).astype(np.float32)


if __name__ == "__main__":
    env = JointControlDemo(
        model_xml_path="tests/orca_gym/environment/euler/fixtures/simple_pendulum.xml",
        skip_grpc_load=True,  # 离线模式
    )
    env.reset()

    print("=" * 50)
    print("1. 力矩驱动（经过物理）")
    print("=" * 50)
    env.demo_torque_drive(actuator_name=ACTUATOR_NAME, steps=100)

    print("\n" + "=" * 50)
    print("2. 直接设位置（正弦摆动）")
    print("=" * 50)
    env.demo_wiggle(joint_name=JOINT_NAME, amplitude=0.5, steps=100)

    print("\n" + "=" * 50)
    print("3. Body / Site 查询")
    print("=" * 50)
    env.demo_query_body_site(body_name=BODY_NAME, site_name=SITE_NAME)

    print("\n" + "=" * 50)
    print("4. qpos 布局")
    print("=" * 50)
    env.print_qpos_layout()

    env.close()
```

---

## 逐段解释

### 核心概念：qpos 和 qvel

MuJoCo 用两个数组描述整个仿真世界：

```
qpos = [关节0角度, 关节1角度, ..., 自由物体位姿(xyz+qwxyz)]
       长度 = model.nq（广义坐标数）

qvel = [关节0角速度, 关节1角速度, ..., 自由物体速度(v+ω)]
       长度 = model.nv（自由度数）
```

不同关节类型在 qpos 中占的元素数不同：

| 关节类型 | qpos 元素数 | 含义 |
|----------|------------|------|
| `hinge`（旋转） | 1 | 旋转角度（弧度） |
| `slide`（滑动） | 1 | 滑动距离（米） |
| `ball`（球） | 4 | 四元数 [w, x, y, z] |
| `free`（自由） | 7 | [x, y, z, qw, qx, qy, qz] |

> **名称 vs 下标**：XML 中每个 `<joint>`、`<body>`、`<site>`、`<actuator>`、`<sensor>` 都有 `name` 属性。
> OrcaGym 的查询 API 全部支持按名称访问（`query_joint_qpos(names)`、`get_body_xpos_xmat_xquat(names)` 等），
> 避免你记住易变的下标。下方"状态写入"需要先 `jnt_qposadr(name)` 拿到地址，再改 qpos 副本。

### 方法 1：力矩驱动（推荐）⭐

```python
ctrl = np.zeros(env.model.nu)
ctrl[act_id] = 0.3 * max_torque   # 施加 30% 最大力矩
env.do_simulation(ctrl, env.frame_skip)
```

**原理**：力矩 → 加速度 → 速度 → 位置。这是"经过物理"的方式——关节在
重力、惯性、摩擦力等物理效应下自然运动，而非瞬移到目标位置。

**适用场景**：正常的仿真控制、RL 训练。这是**推荐的标准方式**。

### 方法 2：按名称设位置（适合 reset）

```python
qpos = env.data.qpos.copy()            # 1. 复制当前 qpos
addr = env.jnt_qposadr("hinge")        # 2. 按名称查起始地址
qpos[addr] = target_angle              # 3. 修改副本
env.set_joint_qpos(qpos)               # 4. 全量写入（合规）
env.mj_forward()                       # 5. 必须！更新派生量
env._sync_view()                        # 6. 同步到 DataView
```

> ⚠️ **这个方法不经过物理！** 关节瞬移到目标角度，不经历加速/减速过程。
> 适用场景：**重置环境**（快速设定初始姿态）、调试。
>
> ⚠️ Euler 路径下 `set_joint_qpos(qpos)` 接受**全量 qpos 数组**（不是按名称的 dict）。
> 若只改一个关节，仍需先 copy 全量 qpos，再按 `jnt_qposadr(name)` 定位修改，最后整体写回。

### 方法 3：按名称查 body / site

```python
# Body 位姿（名称来自 XML 中 <body name="pendulum">）
body_pose = env.get_body_xpos_xmat_xquat(["pendulum"])
# → {"pendulum": {"xpos": ..., "xmat": ..., "xquat": ...}}

# Site 位姿（名称来自 XML 中 <site name="tip">）
site_pose = env.query_site_pos_and_mat(["tip"])
# → {"tip": {"xpos": ..., "xmat": ...}}
```

Body、Site、Sensor、Actuator 的名称都来自 XML 的 `name` 属性，
代码中直接用名称作键查询，读起来一目了然。

### 状态写入的黄金法则

```
1. copy()                   ← 复制当前 qpos（data.qpos 是只读零拷贝视图）
2. jnt_qposadr(name)         ← 按名称查起始地址
3. 修改副本中对应切片
4. set_joint_qpos(qpos_copy) ← 全量合规写入
5. mj_forward()              ← 必须！更新派生量
6. _sync_view()              ← 同步到 DataView
```

跳过第 5 步 → body 位姿/传感器读到的仍是旧值。

### 安全提示

- 设置过大的关节角度可能导致**自碰撞**
- 设置过大的力矩可能导致仿真**不稳定**（数值爆炸）
- 建议先用小幅度（±0.5 rad 以内）测试
- 仿真中损坏没有后果——大胆试！

---

## 下一步

能控制关节了。接下来学习如何**写 PD 控制器**：[🎮 简单控制器](simple-controller.md)。
