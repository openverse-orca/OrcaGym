# 🎭 Mocap 控制

Mocap (Motion Capture) body 是 MuJoCo 中的特殊 body，可以通过直接设置位姿来操控。

## 什么是 Mocap Body

- MuJoCo 中 `body_mocapid != -1` 的 body
- 可以**直接设置位姿**而不受力/动力学影响
- 常配合等式约束 (WELD/CONNECT) 实现抓取/拖拽
- 典型用途：锚点、虚拟手、工具附着点

## 查找 Mocap Body

Euler 体系下 `OrcaGymEulerEnv` 未直接暴露 mocap 名称列表的公共方法。
当前可通过 `env.data.mocap_pos(name)` 验证某个 mocap body 是否存在
（不存在会抛 KeyError），或通过 `env.model.get_body_names()` 遍历查找
（mocap body 通常带 `Anchor` 等后缀）。

```python
# 通过 env.data 读取已知 mocap body 的位姿（若名称不存在会抛错）
mocap_pos = env.data.mocap_pos("ActorManipulator_Anchor")  # (3,)
mocap_quat = env.data.mocap_quat("ActorManipulator_Anchor")  # (4,)
```

> 注：默认关卡的 UI 抓取专用 mocap body 名为
> `ORCA_MANIPULATOR_<uuid>_Anchor`，旧关卡为 `ActorManipulator_Anchor`。

## 设置 Mocap 位姿

```python
# 直接设置 mocap body 的世界坐标位姿
env.set_mocap_pos_and_quat({
    "ActorManipulator_Anchor": {
        "pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
        "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }
})

# 必须 forward
env.mj_forward()
```

## 读取 Mocap 位姿

```python
# 通过 env.data 读取 mocap 位姿
mocap_pos = env.data.mocap_pos("ActorManipulator_Anchor")  # (3,)
mocap_quat = env.data.mocap_quat("ActorManipulator_Anchor")  # (4,)
```

## Mocap + 等式约束 = 物体操作

Euler 体系下 `OrcaGymEulerEnv` 不提供 `anchor_actor` / `release_body_anchored`
高层公共方法（这两个方法仅在 Local 体系存在）。程序化操作需用**公共等式约束原语**编排，
与 UI 抓取内部方法 `_anchor_actor` / `_release_body_anchored` 的编排模式一致：

- `equality_find_slot_by_body(body_name)` — 查找含指定 body 的等式约束槽位
- `equality_constraint(slot)` — 读取槽位完整数据（用于快照/恢复）
- `equality_update(slot, ...)` — 原子写入槽位字段（type/obj1/obj2/data/active…）
- `set_mocap_pos_and_quat(...)` — 对齐 mocap 位姿到目标 body

```python
import mujoco

# ── 1. 锚定物体 ──
anchor_mocap_name = "ActorManipulator_Anchor"  # 或 ORCA_MANIPULATOR_<uuid>_Anchor
actor_name = "target_object"

# 查找含 anchor mocap 的等式约束槽位
slot = env.equality_find_slot_by_body(anchor_mocap_name)
if slot == -1:
    raise ValueError(f"模型中无含 {anchor_mocap_name} 的 equality 槽位")

# 保存原始约束快照（释放时恢复）
original_eq = env.equality_constraint(slot)

# 对齐 mocap 位姿到 actor 当前位姿（避免下一帧拉扯）
actor_pose = env.get_body_xpos_xmat_xquat([actor_name])[actor_name]
env.set_mocap_pos_and_quat({
    anchor_mocap_name: {
        "pos": actor_pose["xpos"],
        "quat": actor_pose["xquat"],
    }
})

# 确定改 obj1 还是 obj2（mocap 一端保持，另一端改为 actor）
mocap_id = env.model.body_name2id(anchor_mocap_name)
if original_eq["obj1_id"] == mocap_id:
    new_obj1_name = anchor_mocap_name
    new_obj2_name = actor_name
else:
    new_obj1_name = actor_name
    new_obj2_name = anchor_mocap_name

# 写入约束（type/obj，内部 mj_forward）
env.equality_update(
    slot,
    eq_type=mujoco.mjtEq.mjEQ_WELD,
    obj1_name=new_obj1_name,
    obj2_name=new_obj2_name,
)

# ── 2. 移动锚点 → 物体跟随 ──
env.set_mocap_pos_and_quat({
    anchor_mocap_name: {
        "pos": new_target_pos,
        "quat": new_target_quat,
    }
})
env.mj_forward()

# ── 3. 释放（从快照恢复原始约束） ──
slot = env.equality_find_slot_by_body(actor_name)
if slot != -1:
    env.equality_update(
        slot,
        eq_type=original_eq["type"],
        obj1_name=env.model.body_id2name(original_eq["obj1_id"]),
        obj2_name=env.model.body_id2name(original_eq["obj2_id"]),
        data=original_eq["data"],
    )
```

> 注：`equality_update` 的 `obj1_name` / `obj2_name` 应为**已含 agent 前缀**
> 的完整 body 名称（本原语不做名称空间解析）。

## 轨迹跟踪示例

```python
def follow_trajectory(env, trajectory: list[np.ndarray], duration: float):
    """让锚点跟随一条轨迹"""
    steps = int(duration / env.dt)

    for i in range(steps):
        t = i / steps
        idx = min(int(t * len(trajectory)), len(trajectory) - 1)
        target_pos = trajectory[idx]

        env.set_mocap_pos_and_quat({
            "ActorManipulator_Anchor": {
                "pos": target_pos,
                "quat": np.array([1, 0, 0, 0]),
            }
        })

        env.mj_forward()
        env.render()
```
