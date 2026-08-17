# 🎭 物体操作

在 OrcaGym/OrcaStudio 中操作场景物体。

## 锚点系统

OrcaGym 使用 **Mocap 锚点 + 等式约束** 系统来操作物体：

```
用户/代码 → 设置 mocap 位姿 → 
 WELD/CONNECT 约束 → 物体跟随锚点移动
```

## Python 代码操作

Euler 体系下 `OrcaGymEulerEnv` 不提供 `anchor_actor` / `release_body_anchored`
高层公共方法（这两个方法仅在 Local 体系存在）。程序化操作需用**公共等式约束原语**编排：
`equality_find_slot_by_body` + `equality_constraint` + `equality_update` +
`set_mocap_pos_and_quat`。完整示例见 [🎭 Mocap 控制](../robot_control/mocap-control.md)。

简化的移动锚点片段：

```python
# 移动锚点（mocap body 名称需在模型中存在）
env.set_mocap_pos_and_quat({
    "ActorManipulator_Anchor": {
        "pos": np.array([0.5, 0.0, 0.8]),
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
})

# 释放需通过 equality_update 从原始快照恢复约束（见 Mocap 控制文档）
```

## UI 拖拽操作

在 OrcaStudio UI 中拖拽物体时，Euler 体系下需通过 `env.studio_bridge()`
（返回 `OrcaStudioBridge`）的 **async** 方法查询状态，再用事件循环驱动：

```python
# 检测 UI 操作（async 方法，需通过 env.loop 驱动）
body_name, anchor_type = env.loop.run_until_complete(
    env.studio_bridge().get_body_manipulation_anchored()
)
if body_name is not None:
    # 返回 dict，含 "delta_pos" 和 "delta_quat" 键（非 tuple）
    movement = env.loop.run_until_complete(
        env.studio_bridge().get_body_manipulation_movement()
    )
    delta_pos = movement["delta_pos"]
    delta_quat = movement["delta_quat"]
    print(f"物体 {body_name} 移动了 {delta_pos}")
```

> 注：`OrcaGymEulerEnv` 内部的 `_do_body_manipulation` 已封装上述流程，
> 由 `render()` 自动驱动，用户通常无需手动调用。

## 包围盒查询

```python
# 返回 np.ndarray(3,)，geom 的半尺寸 (hx, hy, hz)，非字典
bbox = env.get_goal_bounding_box("target_object")
print(f"半尺寸: {bbox}")  # 如 [0.05, 0.05, 0.1]
```
