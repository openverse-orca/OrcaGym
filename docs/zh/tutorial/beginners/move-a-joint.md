# 🦾 让机器人动起来 — 控制单个关节

前面我们只是"看"，这一节开始**让机器人动起来**。我们从最简单的开始：理解 `qpos`/`qvel`，手动控制关节。

> 完整可运行代码见 [OrcaPlayground examples/euler/01_hello_euler/](https://github.com/OrcaGym/OrcaPlayground)。

---

## 完整示例：先看全貌

下面是一个**可以直接运行**的完整示例，展示了三种控制和查询关节状态的方法。
建议先通读一遍，再看后面的逐段解释。

```python
"""关节控制完整演示：力矩驱动 → 直接设位置 → 按名称设位置"""
import time
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class JointControlDemo(OrcaGymEulerEnv):
    """关节控制演示环境"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            **kwargs,
        )

    # ─── 方法 1：力矩驱动（经过物理）⭐ 推荐 ───
    def demo_torque_drive(self, joint_index=0, steps=200):
        """用恒定力矩驱动关节，观察它在重力+惯性下的自然运动。

        这是"经过物理"的方式：力矩 → 加速度 → 速度 → 位置
        """
        ctrlrange = self.model.get_actuator_ctrlrange()
        max_torque = ctrlrange[joint_index, 1]
        print(f"关节 {joint_index} 力矩范围: "
              f"[{ctrlrange[joint_index, 0]:.1f}, {max_torque:.1f}] N·m")

        for i in range(steps):
            ctrl = np.zeros(self.model.nu, dtype=np.float64)

            # 前半段正向力矩，后半段反向 → 观察往复运动
            if i < steps // 2:
                ctrl[joint_index] = 0.3 * max_torque   # 30% 正向
            else:
                ctrl[joint_index] = -0.3 * max_torque  # 30% 反向

            self.do_simulation(ctrl, self.frame_skip)

            if i % 20 == 0:
                pos = self.data.qpos[joint_index]
                vel = self.data.qvel[joint_index]
                print(f"  Step {i:3d}: pos={pos:+.4f} rad, "
                      f"vel={vel:+.4f} rad/s, torque={ctrl[joint_index]:+.2f}")

    # ─── 方法 2：直接设位置（正弦摆动，适合 reset）───
    def demo_wiggle(self, joint_index=0, amplitude=0.5, steps=200):
        """让关节做正弦摆动。直接设 qpos 方式，不经过物理。"""
        print(f"关节 {joint_index} 初始位置: {self.data.qpos[joint_index]:.3f} rad")

        for i in range(steps):
            target_angle = amplitude * np.sin(i * 0.1)

            # 合规写入：copy → 修改 → set → forward
            new_qpos = self.data.qpos.copy()
            new_qpos[joint_index] = target_angle
            self.set_joint_qpos(new_qpos)
            self.set_joint_qvel(np.zeros(self.model.nv))
            self.mj_forward()
            self._sync_view()

            if i % 20 == 0:
                actual = self.data.qpos[joint_index]
                print(f"  Step {i:3d}: 目标={target_angle:+.3f}, "
                      f"实际={actual:+.3f}")

    # ─── 方法 3：按名称设位置 ───
    def demo_set_named_joint(self, joint_name, target_angle):
        """按关节名称（而非索引）设置位置。"""
        qpos = self.data.qpos.copy()
        qpos_addr = self.jnt_qposadr(joint_name)
        qpos[qpos_addr] = target_angle

        self.set_joint_qpos(qpos)
        self.mj_forward()
        self._sync_view()

        # 验证
        actual = self.query_joint_qpos([joint_name])[joint_name]
        print(f"{joint_name}: 目标={target_angle:.3f}, 实际={actual[0]:.3f}")

    # ─── 工具：打印 qpos 布局 ───
    def print_qpos_layout(self):
        """打印 qpos 布局，帮助理解每个关节占几个元素"""
        offset = 0
        for i in range(self.model.njnt):
            name = self.model.joint_id2name(i)
            info = self.model.get_joint_byname(name)
            nq = info.get("NQ", 1)
            print(f"  qpos[{offset:2d}:{offset+nq:2d}]  {name}  (nq={nq})")
            offset += nq

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
    env = JointControlDemo(
        model_xml_path="/path/to/scene.xml",
        skip_grpc_load=True,  # 离线模式
    )
    env.reset()

    print("=" * 50)
    print("1. 力矩驱动（经过物理）")
    print("=" * 50)
    env.demo_torque_drive(joint_index=0, steps=100)

    print("\n" + "=" * 50)
    print("2. 直接设位置（正弦摆动）")
    print("=" * 50)
    env.demo_wiggle(joint_index=0, amplitude=0.5, steps=100)

    print("\n" + "=" * 50)
    print("3. qpos 布局")
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

### 方法 1：力矩驱动（推荐）⭐

```python
ctrl = np.zeros(env.model.nu)
ctrl[joint_index] = 0.3 * max_torque   # 施加 30% 最大力矩
env.do_simulation(ctrl, env.frame_skip)
```

**原理**：力矩 → 加速度 → 速度 → 位置。这是"经过物理"的方式——关节在
重力、惯性、摩擦力等物理效应下自然运动，而非瞬移到目标位置。

**适用场景**：正常的仿真控制、RL 训练。这是**推荐的标准方式**。

### 方法 2：直接设位置（适合 reset）

```python
qpos = env.data.qpos.copy()        # 1. 复制
qpos[joint_index] = target_angle   # 2. 修改副本
env.set_joint_qpos(qpos)           # 3. 合规写入
env.mj_forward()                   # 4. 必须！更新派生量
env._sync_view()                   # 5. 同步到 DataView
```

> ⚠️ **这个方法不经过物理！** 关节瞬移到目标角度，不经历加速/减速过程。
> 适用场景：**重置环境**（快速设定初始姿态）、调试。

### 方法 3：按名称设位置

```python
qpos = env.data.qpos.copy()
addr = env.jnt_qposadr("robot_0_joint1")  # 按名称查地址
qpos[addr] = target_angle
env.set_joint_qpos(qpos)
env.mj_forward()
```

当你知道关节**名字**（而非索引）时使用。`jnt_qposadr(name)` 返回该关节在
qpos 数组中的起始地址。

### 状态写入的黄金法则

```
1. copy()  ← 复制当前 qpos（data.qpos 是只读零拷贝视图）
2. 修改副本
3. set_joint_qpos(qpos_copy)  ← 合规写入
4. mj_forward()               ← 必须！更新派生量
5. _sync_view()               ← 同步到 DataView
```

跳过第 4 步 → body 位姿/传感器读到的仍是旧值。

### 安全提示

- 设置过大的关节角度可能导致**自碰撞**
- 设置过大的力矩可能导致仿真**不稳定**（数值爆炸）
- 建议先用小幅度（±0.5 rad 以内）测试
- 仿真中损坏没有后果——大胆试！

---

## 下一步

能控制关节了。接下来学习如何**写 PD 控制器**：[🎮 简单控制器](simple-controller.md)。
