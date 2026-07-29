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
        import mujoco
        agent = self._agent_names[0]
        object_name = f"{agent}_manipulation_box"
        mocap_name = "ActorManipulator_Anchor"  # 旧关卡；新关卡为 ORCA_MANIPULATOR_<uuid>_Anchor
        ctrl = np.zeros(self.model.nu)

        # ─── 第 1 步：抓取（用公共原语编排，等价于 Local 的 anchor_actor）───
        print("第 1 步：抓取物体...")
        # 1.1 查找含 mocap 的等式约束槽位
        slot = self.equality_find_slot_by_body(mocap_name)
        if slot == -1:
            raise ValueError(f"模型中无含 {mocap_name} 的 equality 槽位")
        # 1.2 保存原始约束快照（释放时恢复）
        original_eq = self.equality_constraint(slot)
        # 1.3 对齐 mocap 位姿到物体当前位姿（避免下一帧拉扯）
        obj_pose = self.get_body_xpos_xmat_xquat([object_name])[object_name]
        self.set_mocap_pos_and_quat({
            mocap_name: {"pos": obj_pose["xpos"], "quat": obj_pose["xquat"]}
        })
        # 1.4 写入 WELD 约束（type/obj，内部 mj_forward）
        mocap_id = self.model.body_name2id(mocap_name)
        if original_eq["obj1_id"] == mocap_id:
            new_obj1_name, new_obj2_name = mocap_name, object_name
        else:
            new_obj1_name, new_obj2_name = object_name, mocap_name
        self.equality_update(
            slot,
            eq_type=mujoco.mjtEq.mjEQ_WELD,
            obj1_name=new_obj1_name,
            obj2_name=new_obj2_name,
        )
        print(f"  ✅ {object_name} 已锚定（WELD 约束）")

        # ─── 第 2 步：移动 ───
        target_pos = np.array([0.7, 0.0, 0.5])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        print(f"\n第 2 步：移动物体到 {target_pos}...")

        self.set_mocap_pos_and_quat({
            mocap_name: {
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

        # ─── 第 3 步：释放（用公共原语恢复原始约束）───
        print(f"\n第 3 步：释放物体...")
        slot = self.equality_find_slot_by_body(object_name)
        if slot != -1:
            self.equality_update(
                slot,
                eq_type=original_eq["type"],
                obj1_name=self.model.body_id2name(original_eq["obj1_id"]),
                obj2_name=self.model.body_id2name(original_eq["obj2_id"]),
                data=original_eq["data"],
            )
        self.mj_forward()
        print("  ✅ 物体已释放")

        # ─── 第 4 步：查看约束信息 ───
        print(f"\n当前等式约束:")
        for i in range(self._gym.n_equality()):
            eq = self.equality_constraint(i)
            print(f"  type={eq['type']}, obj1={eq['obj1_id']}, "
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

### 1. 锚定物体 — 公共原语编排

> ⚠️ **Euler 路径**：`OrcaGymEulerEnv` **没有** `anchor_actor` / `release_body_anchored` 公共方法（这两个仅在 Local 体系 `OrcaGymLocalEnv` 中提供）。程序化操作应仿照 UI 抓取内部方法 `_anchor_actor` / `_release_body_anchored` 的编排模式，使用以下公共原语组合实现：

```python
# 等价于 Local 体系的 env.anchor_actor("target_object", "weld")
import mujoco

mocap_name = "ActorManipulator_Anchor"   # 场景中的 mocap body
object_name = "target_object"

# 1. 查找含 mocap 的等式约束槽位
slot = env.equality_find_slot_by_body(mocap_name)
# 2. 读取原始约束（释放时恢复）
original_eq = env.equality_constraint(slot)
# 3. 对齐 mocap 位姿到物体当前位姿（避免下一帧拉扯）
obj_pose = env.get_body_xpos_xmat_xquat([object_name])[object_name]
env.set_mocap_pos_and_quat({
    mocap_name: {"pos": obj_pose["xpos"], "quat": obj_pose["xquat"]}
})
# 4. 写入 WELD 约束
env.equality_update(
    slot,
    eq_type=mujoco.mjtEq.mjEQ_WELD,
    obj1_name=mocap_name,
    obj2_name=object_name,
)
```

这一组操作做了三件事：
1. 读取物体当前的世界位姿
2. 将 mocap body 移到该位姿
3. 在 mocap 和物体之间建立 WELD 等式约束

约束类型常量（来自 `mujoco.mjtEq`，无需额外导入 `AnchorType`）：
```python
import mujoco

mujoco.mjtEq.mjEQ_WELD      # 焊接 — 完全固定（位置+姿态）
mujoco.mjtEq.mjEQ_CONNECT   # 球关节 — 固定位置，允许旋转
mujoco.mjtEq.mjEQ_JOINT     # 关节耦合
```

> 📝 **Local 体系**：若使用 `OrcaGymLocalEnv`，可直接调用 `env.anchor_actor(name, AnchorType.WELD)`，`AnchorType` 从 `orca_gym.core.orca_gym_local` 导入。Euler 路径不提供此便捷封装，需用上面的原语编排。

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

### 3. 释放物体 — 恢复原始约束

> ⚠️ **Euler 路径**：`OrcaGymEulerEnv` 没有 `release_body_anchored` 公共方法。释放时用 `equality_update` 恢复抓取前保存的原始约束字段：

```python
# 等价于 Local 体系的 env.release_body_anchored()
slot = env.equality_find_slot_by_body(object_name)
if slot != -1:
    env.equality_update(
        slot,
        eq_type=original_eq["type"],
        obj1_name=env.model.body_id2name(original_eq["obj1_id"]),
        obj2_name=env.model.body_id2name(original_eq["obj2_id"]),
        data=original_eq["data"],
    )
env.mj_forward()
```

解除 WELD 约束，物体恢复自由（受重力影响下落）。

### 4. 等式约束管理

**查看约束**（两种等价途径）：
```python
# 途径 A：通过 env.equality_constraint(slot) 逐个读取（返回键为 type）
for slot in range(env._gym.n_equality()):
    eq = env.equality_constraint(slot)
    print(f"type={eq['type']}, obj1={eq['obj1_id']}, "
          f"obj2={eq['obj2_id']}, active={eq['active']}")

# 途径 B：通过 env.model.get_eq_list() 读取初始快照（返回键为 eq_type）
eq_list = env.model.get_eq_list()
for eq in eq_list:
    print(f"type={eq['eq_type']}, obj1={eq['obj1_id']}, "
          f"obj2={eq['obj2_id']}, active={eq['active']}")
```

> ⚠️ **键名差异**：`env.equality_constraint(slot)` 返回的字典键名为 `type`；`env.model.get_eq_list()` 返回的字典键名为 `eq_type`。两者均对应 MuJoCo 的 `eq_type` 字段，只是命名不同。

**修改约束关联对象**（Euler 路径用 `equality_update`，按名称自动解析 id）：
```python
# Euler 路径：env.equality_update(slot, obj1_name=..., obj2_name=...)
env.equality_update(
    0,                                        # 等式约束槽位索引
    obj1_name="ActorManipulator_Anchor",      # 新 obj1（自动解析为 id）
    obj2_name="target_object",                # 新 obj2（自动解析为 id）
)
```

> 📝 **Local 体系**：`OrcaGymLocalEnv` 提供 `env.modify_equality_objects(eq_ids, obj1_ids, obj2_ids)`（参数为 id 列表，gym 层 API）。Euler 路径已用 `equality_update` 覆盖此功能，且按名称传入更直观。

**停用约束**（通过 `equality_update` 设 `active=False`）：
```python
env.equality_update(0, active=False)   # 停用槽位 0 的约束
env.equality_update(0, active=True)    # 重新激活
```

> ⚠️ Euler 路径已删除 `env.update_equality_constraints(eq_list)` 公共方法（该方法保留在 gym 层 `OrcaGymEuler` / SimCore 作为 `equality_update` 的底层实现）。Env 层统一用 `equality_update(slot, ...)` 逐槽位更新。

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
抓取:  equality_find_slot_by_body(mocap) → equality_constraint(slot) 保存原始
       → set_mocap_pos_and_quat 对齐位姿 → equality_update(WELD, mocap, object)
         ↓
移动:  set_mocap_pos_and_quat({mocap: {pos, quat}})
         ↓
      mj_forward()
         ↓
      do_simulation(ctrl, n_frames)  ← 约束生效，物体跟随
         ↓
释放:  equality_find_slot_by_body(object) → equality_update 恢复 original_eq 字段
```

---

## 下一步

掌握了等式约束，接下来学习如何**施加外力和 IK**：[🔄 外力应用与 IK](../physics/force-apply.md)。
