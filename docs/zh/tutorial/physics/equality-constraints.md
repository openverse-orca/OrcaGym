# 🔗 等式约束

MuJoCo 的等式约束是 OrcaGym 中实现物体抓取和操作的核心机制。

> 完整可运行代码见 [OrcaPlayground examples/euler/05_force_apply/](https://github.com/OrcaGym/OrcaPlayground) 和 [09_body_manipulation/](https://github.com/OrcaGym/OrcaPlayground)。

---

## 完整示例：先看全貌

下面是一个完整的抓取→移动→释放演示：

```python
"""等式约束完整演示：抓取 → 移动 → 释放"""
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class GraspDemo(OrcaGymEulerEnv):
    """演示 mocap + weld 约束抓取物体"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 20),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["g1"]),
            time_step=kwargs.pop("time_step", 0.001),
            model_xml_path=model_xml_path,
            **kwargs,
        )

    def demo_grasp_and_move(self):
        """完整演示：抓取物体 → 移动到目标 → 释放"""
        agent = self.agent_name
        object_name = f"{agent}_manipulation_box"
        ctrl = np.zeros(self.model.nu)

        # ─── 第 1 步：抓取 ───
        print("第 1 步：抓取物体...")
        self.anchor_actor(object_name, "weld")
        print(f"  ✅ {object_name} 已锚定（WELD 约束）")

        # ─── 第 2 步：移动 ───
        target_pos = np.array([0.7, 0.0, 0.5])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        print(f"\n第 2 步：移动物体到 {target_pos}...")

        self.set_mocap_pos_and_quat({
            "ActorManipulator_Anchor": {
                "pos": target_pos,
                "quat": target_quat,
            }
        })
        self.mj_forward()

        # 步进让约束生效
        for _ in range(10):
            self.do_simulation(ctrl, self.frame_skip)

        # 验证：物体已跟随到目标
        box = self.get_body_xpos_xmat_xquat([object_name])
        box_pos = box[object_name]["xpos"]
        dist = np.linalg.norm(box_pos - target_pos)
        print(f"  物体当前位置: {box_pos}")
        print(f"  距目标: {dist:.4f}m")
        print(f"  {'✅ 物体已到达目标' if dist < 0.05 else '⚠️ 未到达'}")

        # ─── 第 3 步：释放 ───
        print(f"\n第 3 步：释放物体...")
        self.release_body_anchored()
        self.mj_forward()
        print("  ✅ 物体已释放")

        # ─── 第 4 步：查看约束信息 ───
        print(f"\n当前等式约束:")
        eq_list = self.model.get_eq_list()
        for eq in eq_list:
            print(f"  type={eq['eq_type']}, obj1={eq['obj1_id']}, "
                  f"obj2={eq['obj2_id']}, active={eq['active']}")

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
    env = GraspDemo(
        model_xml_path="/path/to/scene.xml",
        skip_grpc_load=False,
    )
    env.reset()
    env.demo_grasp_and_move()
    env.close()
```

---

## 逐段解释

### 什么是等式约束

等式约束强制两个 body 之间满足某种运动学关系：

| 约束类型 | 效果 | 自由度 |
|----------|------|--------|
| `mjEQ_WELD` | 完全固定（位置 + 姿态），像焊接在一起 | 0 DOF |
| `mjEQ_CONNECT` (BALL) | 固定位置，允许旋转，像球关节 | 3 DOF (旋转) |

在 OrcaGym 中，等式约束通常配合 **mocap body** 使用：
```
用户设置 mocap 位姿 → WELD 约束 → 被锚定的物体跟随移动
```

### 1. 锚定物体 — `anchor_actor`

```python
env.anchor_actor("target_object", "weld")
```

这一行做了三件事：
1. 读取物体当前的世界位姿
2. 将 mocap body 移到该位姿
3. 在 mocap 和物体之间建立 WELD 等式约束

锚定类型：
```python
from orca_core.orca_gym_local import AnchorType

AnchorType.WELD   # 焊接 — 完全固定（位置+姿态）
AnchorType.BALL   # 球关节 — 固定位置，允许旋转
AnchorType.NONE   # 无锚定
```

### 2. 移动物体 — Mocap 位姿设置

```python
env.set_mocap_pos_and_quat({
    "ActorManipulator_Anchor": {
        "pos": np.array([0.7, 0.0, 0.5]),          # 目标位置 [x, y, z]
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),    # 目标四元数 [w, x, y, z]
    }
})
env.mj_forward()  # ← 必须！更新派生量
```

**Mocap body** 是 MuJoCo 的特殊 body（`body_mocapid != -1`）：
- 可以**直接设置位姿**，不受力/动力学影响
- 像"看不见的手"一样移动
- 配合 WELD 约束，被锚定的物体会自动跟随

**回读验证**（通过 `env.data` 零拷贝视图）：
```python
read_pos = env.data.mocap_pos("mocap_name")    # (3,)
read_quat = env.data.mocap_quat("mocap_name")  # (4,) [w, x, y, z]
```

### 3. 释放物体 — `release_body_anchored`

```python
env.release_body_anchored()
env.mj_forward()
```

解除 WELD 约束，物体恢复自由（受重力影响下落）。

### 4. 等式约束管理

**查看约束**：
```python
eq_list = env.model.get_eq_list()
for eq in eq_list:
    print(f"type={eq['eq_type']}, obj1={eq['obj1_id']}, "
          f"obj2={eq['obj2_id']}, active={eq['active']}")
```

**修改约束关联对象**（按名称，自动解析 id）：
```python
env.modify_equality_objects(
    eq_ids=[0],                              # 等式约束索引
    obj1_names=["ActorManipulator_Anchor"],  # 新 obj1
    obj2_names=["target_object"],            # 新 obj2
)
```

**停用约束**：
```python
env.update_equality_constraints([{
    "type": 0, "obj1_id": -1, "obj2_id": -1,
    "data": np.zeros(7),
}])
```

### 5. UI 交互中的锚定

在 OrcaStudio UI 中拖拽物体时，系统自动处理锚定：

```python
# render() 内部自动调用 do_body_manipulation()
# 可通过 studio_bridge() 检测 UI 操作
bridge = env.studio_bridge()
body_name, anchor_type = bridge.get_body_manipulation_anchored()
if body_name is not None:
    delta_pos, delta_quat = bridge.get_body_manipulation_movement()
    print(f"用户正在拖拽: {body_name}, 位移: {delta_pos}")
```

---

## 完整工作流总结

```
抓取:  anchor_actor("object", "weld")
         ↓
移动:  set_mocap_pos_and_quat({mocap: {pos, quat}})
         ↓
      mj_forward()
         ↓
      do_simulation(ctrl, n_frames)  ← 约束生效，物体跟随
         ↓
释放:  release_body_anchored()
```

---

## 下一步

掌握了等式约束，接下来学习如何**施加外力和 IK**：[🔄 外力应用与 IK](../physics/force-apply.md)。
