# 🎭 物体操作

在 OrcaGym/OrcaStudio 中操作场景物体。

## 锚点系统

OrcaGym 使用 **Mocap 锚点 + 等式约束** 系统来操作物体：

```
用户/代码 → 设置 mocap 位姿 → 
 WELD/CONNECT 约束 → 物体跟随锚点移动
```

## Python 代码操作

```python
# 锚定物体
env.anchor_actor("target_object", AnchorType.WELD)

# 移动锚点（mocap body 名称需在模型中存在）
env.set_mocap_pos_and_quat({
    "ActorManipulator_Anchor": {
        "pos": np.array([0.5, 0.0, 0.8]),
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
})

# 释放
env.release_body_anchored()
```

## UI 拖拽操作

在 OrcaStudio UI 中拖拽物体时：

```python
# 检测 UI 操作
body_name, anchor_type = env.get_body_manipulation_anchored()
if body_name is not None:
 delta_pos, delta_quat = env.get_body_manipulation_movement()
 print(f"物体 {body_name} 移动了 {delta_pos}")
```

## 包围盒查询

```python
# 计算物体的轴对齐包围盒
bbox = env.get_goal_bounding_box("target_object")
print(f"包围盒: min={bbox['min']}, max={bbox['max']}, size={bbox['size']}")
```
