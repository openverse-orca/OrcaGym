# 🤝 接触感知

利用接触力信息作为机器人的"触觉感知"。

> 接触查询 API 详见 [📡 状态查询 API](../robot_control/state-queries-api.md)。

## 接触作为触觉信号

接触力可以提供丰富的信息：

- **抓取检测** — 是否有接触
- **力控** — 维持目标接触力
- **表面识别** — 接触法向方向
- **碰撞检测** — 意外碰撞

## 接触查询流水线

```python
# 1. 获取所有活跃接触
contacts = env.query_contact_simple()
# 返回: [{"geom1": 12, "geom2": 34, "dist": ..., "pos": ..., "frame": ...}, ...]

# 2. 筛选感兴趣的接触（注意 key 为小写 "geom1"/"geom2"）
target_body_id = env.model.body_name2id("robot_finger")
finger_contacts = [
    c for c in contacts
    if env.model.get_geom_body_id(c["geom1"]) == target_body_id
    or env.model.get_geom_body_id(c["geom2"]) == target_body_id
]

# 3. 获取接触力（按列表索引，不是 contact 字典中的某个 ID 字段）
if finger_contacts:
    contact_ids = list(range(len(contacts)))
    forces = env.query_contact_force(contact_ids)
    # 取对应接触的力
    for i, c in enumerate(finger_contacts):
        contact_idx = contacts.index(c)
        f = forces[contact_idx]
        normal_force = f[0]  # 第 0 分量是法向力
```

## 抓取检测

```python
def is_grasped(env, finger_names: list[str], object_name: str) -> bool:
    """检查手指是否与目标物体有接触"""
    contacts = env.query_contact_simple()
    object_id = env.model.body_name2id(object_name)
    finger_ids = [env.model.body_name2id(f) for f in finger_names]

    for c in contacts:
        geom1_body = env.model.get_geom_body_id(c["geom1"])
        geom2_body = env.model.get_geom_body_id(c["geom2"])
        bodies = {geom1_body, geom2_body}
        if object_id in bodies and any(f in bodies for f in finger_ids):
            return True
    return False
```

## 力控

```python
def force_control(env, target_force: float = 10.0):
    """简单的力控：维持目标接触力"""
    contacts = env.query_contact_simple()

    if not contacts:
        return np.zeros(env.model.nu)  # 没有接触

    contact_ids = list(range(len(contacts)))
    forces = env.query_contact_force(contact_ids)
    total_force = sum(np.linalg.norm(f[:3]) for f in forces.values())

    # PID 力控
    force_error = target_force - total_force
    correction = force_error * 0.01

    ctrl = np.zeros(env.model.nu)
    # ... 将 correction 分配到相应执行器
    return ctrl
```

## 接触信息汇总

```python
def contact_summary(env) -> dict:
    """生成接触摘要"""
    contacts = env.query_contact_simple()

    summary = {
        "total_contacts": len(contacts),
        "body_pairs": set(),
        "total_force": 0.0,
        "max_force": 0.0,
    }

    if contacts:
        contact_ids = list(range(len(contacts)))
        forces = env.query_contact_force(contact_ids)

        for i, c in enumerate(contacts):
            f = forces[i][:3]  # 线力部分（接触坐标系）
            magnitude = np.linalg.norm(f)

            summary["total_force"] += magnitude
            summary["max_force"] = max(summary["max_force"], magnitude)
            summary["body_pairs"].add((
                env.model.get_geom_body_id(c["geom1"]),
                env.model.get_geom_body_id(c["geom2"]),
            ))

    return summary
```

## 站立检测

```python
def is_standing(env, min_normal_force: float = 50.0) -> bool:
    """检查机器人是否站立（足部接触法向力是否足够）"""
    contacts = env.query_contact_simple()
    if not contacts:
        return False
    contact_ids = list(range(len(contacts)))
    forces = env.query_contact_force(contact_ids)
    # 接触坐标系第 0 分量是法向力，站立时显著为正
    max_normal = max(abs(f[0]) for f in forces.values())
    return max_normal > min_normal_force
```
