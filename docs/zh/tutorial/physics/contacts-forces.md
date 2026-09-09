# 💥 接触与力

OrcaGym 提供全面的接触和力查询接口，用于奖励计算、调试和分析。

> 完整可运行代码见 [OrcaPlayground examples/euler/06_force_apply/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/06_force_apply)（`force_apply.py` + `force_apply_env.py`）。

---

## 实战样例：G1 站立接触力验证

下面是 [Lesson 6](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/06_force_apply) 中的真实验证流程：G1 人形机器人站立时，足部与地面接触，查询接触对和接触法向力。

```python
"""接触力查询验证（摘自 force_apply_env.py 阶段一）"""

# ── step 0：G1 直立，足部触地 ──
contacts = env.query_contact_simple()
print(f"接触对数量: {len(contacts)}")  # 实际约 25 对（双脚与地面多 geom 接触）

# 验证：至少有 1 个接触（站立触地）
assert len(contacts) >= 1, "G1 站立时应与地面有接触"

# 查询接触力（需要接触索引列表）
contact_ids = list(range(len(contacts)))
forces = env.query_contact_force(contact_ids)
# → {0: array([normal, shear1, shear2, torque1, torque2, torque3]), ...}

# 接触力前 3 分量在接触坐标系下，第 0 分量为法向力
max_normal = max(abs(f[0]) for f in forces.values())
print(f"最大法向力: {max_normal:.1f}N")  # 实际约 109931.9N（G1 整重 ~343N）

# 验证：法向力显著（> 50N，证明足部真实触地）
assert max_normal > 50.0, "接触法向力应大于 50N"
```

**关键点**：

- `query_contact_simple()` 返回所有活跃接触对，每个含 `geom1`/`geom2` ID
- `query_contact_force(contact_ids)` 返回 6D 力（接触坐标系：法向 + 切向 + 力矩）
- 第 0 分量 = 法向力，是判断"是否真实接触"的关键指标

---

## 接触力可视化（Studio 自动绘制）

在 Euler 在线模式下，`env.render()` 会自动构建接触快照并传给 OrcaStudio 绘制：

```python
# 正常步进 + 渲染
env.do_simulation(ctrl, env.frame_skip)
env.render(simulate_index=step_idx)
#                ↑ 内部自动：
#   1. _build_contact_data() 收集所有接触的 pos + world_force
#   2. 传给 Studio，在 3D 视口中绘制接触力向量
```

**绘制内容**：

- 接触点位置（3D 箭头起点）
- 接触力向量（世界坐标系，箭头方向和长度反映力的大小）
- 力已从接触坐标系转换到世界坐标系（`frame.T @ force`）

> 无需手动调用绘制接口——`render()` 自动处理。离线模式下无绘制（不连接 Studio）。

---

## 施加外力

样例 step 10 对 G1 pelvis 施加 500N 向上力，验证 `apply_body_force`：

```python
"""外力应用（摘自 force_apply_env.py 阶段二）"""

# ── step 10：记录 pelvis 初始高度，施加 500N 向上力 ──
agent = env._agent_names[0]  # 如 "g1_29dof_camera_usda"
pelvis_body = f"{agent}_pelvis"

pelvis = env.get_body_xpos_xmat_xquat([pelvis_body])
z_before = float(pelvis[pelvis_body]["xpos"][2])  # 实际 ~0.7864m
print(f"施力前 pelvis 高度: {z_before:.4f}m")

env.apply_body_force(
    pelvis_body,
    force=np.array([0.0, 0.0, 500.0]),   # 500N 向上（> G1 整重 ~343N）
    torque=np.array([0.0, 0.0, 0.0]),
)

# ── step 30：验证 pelvis 上升 + xfrc 记录 ──
for _ in range(20):  # 20 控制周期（0.4s）让力生效
    env.do_simulation(np.zeros(env.model.nu), env.frame_skip)

pelvis = env.get_body_xpos_xmat_xquat([pelvis_body])
z_after = float(pelvis[pelvis_body]["xpos"][2])  # 实际 ~1.1777m
print(f"施力后 pelvis 高度: {z_after:.4f}m (Δ={z_after - z_before:.4f}m)")
# 验证：pelvis 上升 > 1cm
assert z_after > z_before + 0.01
```

**为什么施力在 pelvis 而非 torso**：G1 采用力控 motor 执行器，`ctrl=0` 时关节无力矩，零控下腰部关节松弛，torso 施力难以经由松弛关节传递到 pelvis。直接对 pelvis 施力可可靠验证 API 并产生可见的整机抬起效果。

### 验证 xfrc_applied 记录

```python
# xfrc_applied 是 DataView 只读视图，按 body_id 索引
body_id = env.model.body_name2id(pelvis_body)
xfrc = env.data.xfrc_applied[body_id, :3]
print(f"xfrc 记录: {xfrc}")  # [0.0, 0.0, 500.0]
assert np.any(xfrc != 0), "xfrc 应记录施加的力"
```

### 清除外力

```python
# ── step 30 末尾：验证上升后立即清力 ──
env.clear_body_force(pelvis_body)

# ── step 35：验证 xfrc 已归零 ──
xfrc = env.data.xfrc_applied[body_id, :3]
print(f"清力后 xfrc: {xfrc}")  # [0.0, 0.0, 0.0]
assert np.all(xfrc == 0), "清力后 xfrc 应为零"

# step 50：全清（烟雾测试）
env.clear_all_forces()
assert np.all(env.data.xfrc_applied == 0), "全清后所有 xfrc 应为零"
```

---

## 摩擦系数设置

样例 step 50 对 G1 geom 设置摩擦系数：

```python
"""摩擦设置（摘自 force_apply_env.py 阶段三）"""

# 从 model.get_geom_dict() 动态获取 geom 名称（含 agent 前缀 + GUID 后缀）
geom_dict = env.model.get_geom_dict()
g1_geom = next(name for name in geom_dict if name.startswith(f"{agent}_"))

env.set_geom_friction(
    {g1_geom: np.array([0.8, 0.005, 0.0001])}
    # [滑动摩擦, 扭转摩擦, 滚动摩擦]
)
print(f"已设置 {g1_geom} 摩擦系数")
```

---

## Body 外部约束力

查询所有 body 上的外部约束力（含接触反力、等式约束力等）：

```python
cfrc_ext = env.get_cfrc_ext()  # shape: (nbody, 6)
# 每行: [mx, my, mz, fx, fy, fz] — MuJoCo spatial vector 布局（力矩在前，力在后）

# 找出受力最大的 body（按线性力大小）
max_idx = np.argmax(np.linalg.norm(cfrc_ext[:, 3:], axis=1))
print(f"受力最大的 body ID: {max_idx}, 力: {cfrc_ext[max_idx, 3:]}")
```

---

## 碰撞检测

检查两个 body 之间是否碰撞：

```python
def detect_collision(env, body_a, body_b):
    """检查两个 body 之间是否碰撞"""
    contacts = env.query_contact_simple()
    id_a = env.model.body_name2id(body_a)
    id_b = env.model.body_name2id(body_b)

    for c in contacts:
        g1 = env.model.get_geom_body_id(c["geom1"])
        g2 = env.model.get_geom_body_id(c["geom2"])
        if (g1 == id_a and g2 == id_b) or (g1 == id_b and g2 == id_a):
            return True
    return False

# 检查 G1 左右脚是否碰撞
left_foot = f"{agent}_left_ankle_roll_link"
right_foot = f"{agent}_right_ankle_roll_link"
print(f"左脚↔右脚碰撞: {detect_collision(env, left_foot, right_foot)}")
```

---

## 奖励函数中的接触

```python
def contact_reward(env):
    """奖励适度的接触力"""
    contacts = env.query_contact_simple()
    if not contacts:
        return -1.0          # 无接触 = 惩罚

    contact_ids = list(range(len(contacts)))
    forces = env.query_contact_force(contact_ids)
    total_force = sum(np.linalg.norm(f[:3]) for f in forces.values())

    if total_force < 100:    return 0.5   # 轻度接触
    elif total_force < 500:  return 1.0   # 理想接触
    else:                    return -0.5  # 过度用力
```

---

## 接触坐标系说明

`query_contact_force` 返回的力在**接触坐标系**下表示：

| 分量 | 含义 |
|------|------|
| `[0]` | 法向力（垂直于接触面） |
| `[1:3]` | 切向力（摩擦力） |
| `[3:6]` | 力矩分量 |

> ⚠️ **注意**：`query_contact_simple()` 返回的字典 key 是**小写** `"geom1"` / `"geom2"`。

---

## API 速查

| 操作 | API | 说明 |
|------|-----|------|
| 获取接触列表 | `env.query_contact_simple()` | 返回 `list[dict]`，key 小写 |
| 获取接触力 | `env.query_contact_force(ids)` | 6D 力，接触坐标系 |
| 获取约束力 | `env.get_cfrc_ext()` | (nbody, 6)，世界坐标系 |
| 施加外力 | `env.apply_body_force(name, f, τ)` | 世界坐标系 |
| 清除外力 | `env.clear_body_force(name)` | 清除指定 body |
| 清除全部 | `env.clear_all_forces()` | 清除所有外力 |
| 设置摩擦 | `env.set_geom_friction({name: arr})` | [滑动, 扭转, 滚动] |
| 接触力绘制 | `env.render()` | 自动构建接触快照传 Studio 绘制 |

---

## 下一步

掌握了接触和力，接下来学习如何用**等式约束抓取物体**：[🔗 等式约束](equality-constraints.md)。
