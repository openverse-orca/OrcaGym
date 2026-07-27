# 🎭 Mocap 控制

Mocap (Motion Capture) body 是 MuJoCo 中的特殊 body，可以通过直接设置位姿来操控。

## 什么是 Mocap Body

- MuJoCo 中 `body_mocapid != -1` 的 body
- 可以**直接设置位姿**而不受力/动力学影响
- 常配合等式约束 (WELD/CONNECT) 实现抓取/拖拽
- 典型用途：锚点、虚拟手、工具附着点

## 查找 Mocap Body

```python
# 查看模型中所有 mocap body
mocap_dict = env.model.get_mocap_dict()
for name, mocap_id in mocap_dict.items():
 print(f"Mocap: {name} (id={mocap_id})")

# 也可通过 _gym 查询
mocap_names = env._mocap_body_names()
```

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
mocap_pos = env.data.mocap_pos("ActorManipulator_Anchor") # (3,)
mocap_quat = env.data.mocap_quat("ActorManipulator_Anchor") # (4,)
```

## Mocap + 等式约束 = 物体操作

```python
# ：使用高层 API
# 1. 锚定物体 — 自动查询位姿 + 设置 mocap + 建立约束
env.anchor_actor("target_object", "weld")

# 2. 移动锚点 → 物体跟随
env.set_mocap_pos_and_quat({
 "ActorManipulator_Anchor": {
 "pos": new_target_pos,
 "quat": new_target_quat,
 }
})
env.mj_forward()

# 3. 释放
env.release_body_anchored()
```

### 低级控制（需要时）

```python
# 修改等式约束关联对象
env.modify_equality_objects(
 eq_ids=[0],
 obj2_names=["target_object"], # 将 obj2 从旧 body 改为目标 body
)

# 更新约束
env.update_equality_constraints(eq_list)
```

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
