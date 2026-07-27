# 💥 接触与力

OrcaGym 提供全面的接触和力查询接口，用于奖励计算、调试和分析。

> 完整可运行代码见 [OrcaPlayground examples/euler/05_force_apply/](https://github.com/OrcaGym/OrcaPlayground)。

---

## 完整示例：先看全貌

下面是一个完整的接触与力分析演示，展示了接触检测、接触力查询、外力施加、碰撞检测和站立检测：

```python
"""接触与力完整演示"""
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class ContactForceDemo(OrcaGymEulerEnv):
    """演示接触检测、力查询和外力施加"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 20),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["g1"]),
            time_step=kwargs.pop("time_step", 0.001),
            model_xml_path=model_xml_path,
            **kwargs,
        )

    # ─── 工具函数 ───

    def analyze_contacts(self):
        """分析当前所有接触"""
        contacts = self.query_contact_simple()
        if not contacts:
            print("  没有活跃接触")
            return

        print(f"  活跃接触数: {len(contacts)}")
        contact_ids = list(range(len(contacts)))
        forces = self.query_contact_force(contact_ids)

        # 显示前 5 个接触
        for i, c in enumerate(contacts[:5]):
            f = forces[i]
            f_linear = f[:3]
            f_magnitude = np.linalg.norm(f_linear)
            print(f"    接触 {i}: geom{c['geom1']}↔geom{c['geom2']}, "
                  f"力={f_magnitude:.2f}N, 方向={f_linear}")

        # 最大法向力
        max_normal = max(abs(f[0]) for f in forces.values())
        print(f"  最大法向力: {max_normal:.1f}N")

    def detect_collision(self, body_a, body_b):
        """检查两个 body 之间是否碰撞"""
        contacts = self.query_contact_simple()
        id_a = self.model.body_name2id(body_a)
        id_b = self.model.body_name2id(body_b)

        for c in contacts:
            g1 = self.model.get_geom_body_id(c["geom1"])
            g2 = self.model.get_geom_body_id(c["geom2"])
            if (g1 == id_a and g2 == id_b) or (g1 == id_b and g2 == id_a):
                return True
        return False

    def is_standing(self, min_force=50.0):
        """检查机器人是否站立（足部触地力足够）"""
        contacts = self.query_contact_simple()
        if not contacts:
            return False
        contact_ids = list(range(len(contacts)))
        forces = self.query_contact_force(contact_ids)
        max_normal = max(abs(f[0]) for f in forces.values())
        return max_normal > min_force

    # ─── 演示流程 ───

    def demo(self):
        self.reset()
        agent = self.agent_name
        ctrl = np.zeros(self.model.nu)

        # 先步进几步让机器人稳定触地
        for _ in range(5):
            self.do_simulation(ctrl, self.frame_skip)

        # ─── 1. 接触检测 ───
        print("=" * 50)
        print("1. 接触检测（G1 站立触地）")
        print("=" * 50)
        contacts = self.query_contact_simple()
        print(f"  接触对数量: {len(contacts)}")
        print(f"  站立状态: {'✅ 站立' if self.is_standing() else '⚠️ 未站立'}")
        self.analyze_contacts()

        # ─── 2. Body 外部约束力 ───
        print("\n" + "=" * 50)
        print("2. Body 外部约束力")
        print("=" * 50)
        cfrc_ext = self.get_cfrc_ext()
        max_idx = np.argmax(np.linalg.norm(cfrc_ext[:, :3], axis=1))
        print(f"  受力最大的 body ID: {max_idx}, 力: {cfrc_ext[max_idx, :3]}")

        # ─── 3. 施加外力抬起 pelvis ───
        print("\n" + "=" * 50)
        print("3. 施加外力抬起机器人")
        print("=" * 50)

        pelvis_body = f"{agent}_pelvis"
        pelvis = self.get_body_xpos_xmat_xquat([pelvis_body])
        z_before = float(pelvis[pelvis_body]["xpos"][2])
        print(f"  施力前 pelvis 高度: {z_before:.3f}m")

        # 施加 500N 向上力
        self.apply_body_force(
            pelvis_body,
            force=np.array([0.0, 0.0, 500.0]),
            torque=np.array([0.0, 0.0, 0.0]),
        )

        # 步进让力生效
        for _ in range(20):
            self.do_simulation(ctrl, self.frame_skip)

        pelvis = self.get_body_xpos_xmat_xquat([pelvis_body])
        z_after = float(pelvis[pelvis_body]["xpos"][2])
        print(f"  施力后 pelvis 高度: {z_after:.3f}m (Δ={z_after - z_before:.3f}m)")

        # 验证 xfrc_applied
        body_id = self.model.body_name2id(pelvis_body)
        xfrc = self.data.xfrc_applied[body_id, :3]
        print(f"  xfrc_applied 记录的力: {xfrc}")

        # ─── 4. 清除外力 ───
        print("\n" + "=" * 50)
        print("4. 清除外力")
        print("=" * 50)

        self.clear_body_force(pelvis_body)
        xfrc = self.data.xfrc_applied[body_id, :3]
        print(f"  清力后 xfrc: {xfrc}")
        assert np.all(xfrc == 0), "清力后 xfrc 应为零"
        print("  ✅ clear_body_force 成功")

        self.clear_all_forces()  # 全清（烟雾测试）
        print("  ✅ clear_all_forces 成功")

        # ─── 5. 碰撞检测测试 ───
        print("\n" + "=" * 50)
        print("5. 碰撞检测")
        print("=" * 50)
        left_foot = f"{agent}_left_ankle_roll_link"
        right_foot = f"{agent}_right_ankle_roll_link"
        # 地面 body 名称取决于 XML 定义
        print(f"  左脚↔右脚碰撞: {self.detect_collision(left_foot, right_foot)}")

        print("\n✅ 所有接触与力演示完成")

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
    env = ContactForceDemo(
        model_xml_path="/path/to/scene.xml",
        skip_grpc_load=False,
    )
    env.demo()
    env.close()
```

---

## 逐段解释

### 1. 接触检测

```python
contacts = env.query_contact_simple()
# → [{"geom1": 12, "geom2": 34, "dist": ..., "pos": ..., "frame": ...}, ...]
```

`query_contact_simple()` 返回所有当前活跃的接触对。每个接触是一个字典，
包含两个碰撞 geom 的 ID、穿透距离、接触点位置和接触坐标系。

> **注意**：字典 key 是**小写** `"geom1"` / `"geom2"`，不是大写。

**获取接触力**（需要两步）：

```python
# 第 1 步：构造接触 ID 列表（按接触列表索引）
contact_ids = list(range(len(contacts)))

# 第 2 步：查询接触力
forces = env.query_contact_force(contact_ids)
# → {0: array([normal, shear1, shear2, torque1, torque2, torque3]), ...}

# 接触坐标系：第 0 分量 = 法向力
max_normal = max(abs(f[0]) for f in forces.values())
```

**接触坐标系**：接触力在**接触坐标系**下表示。
- 第 0 分量：法向力（垂直于接触面）
- 第 1-2 分量：切向力（摩擦力）
- 第 3-5 分量：力矩分量

### 2. Body 外部约束力

```python
cfrc_ext = env.get_cfrc_ext()  # shape: (nbody, 6)
# 每行: [fx, fy, fz, mx, my, mz] — 作用在每个 body 上的外部约束力

# 找出受力最大的 body
max_idx = np.argmax(np.linalg.norm(cfrc_ext[:, :3], axis=1))
print(f"受力最大: body {max_idx}, 力={cfrc_ext[max_idx, :3]}")
```

### 3. 施加外力

```python
env.apply_body_force(
    "g1_pelvis",                          # body 名称
    force=np.array([0.0, 0.0, 500.0]),    # 力 (N)，世界坐标系
    torque=np.array([0.0, 0.0, 0.0]),     # 力矩 (N·m)
)
```

**原理**：直接在 MuJoCo 的 `xfrc_applied` 数组中写入力/力矩。
力作用在 body 质心，力矩绕 body 质心。下一次 `mj_step()` 时参与动力学计算。

**验证**：`env.data.xfrc_applied[body_id, :3]` — 前 3 个分量为力。

**清除**：
```python
env.clear_body_force("body_name")   # 清除单个
env.clear_all_forces()              # 清除全部
```

### 4. 外力物理原理

在 site 点施力时，等效到 body 中心的力和力矩：

- **力不变**：F_body = F
- **附加扭矩**：τ = r × F（r = site_pos - body_pos）
- **总扭矩**：τ_total = r × F + τ_user

这意味着对 site 施力会产生额外的力矩，等效于对 body 中心施同样的力 + 力臂力矩。

### 5. 奖励函数中的接触

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

## API 速查

| 操作 | API | 说明 |
|------|-----|------|
| 获取接触列表 | `env.query_contact_simple()` | 返回 `list[dict]`，key 小写 |
| 获取接触力 | `env.query_contact_force(ids)` | 6D 力，接触坐标系 |
| 获取约束力 | `env.get_cfrc_ext()` | (nbody, 6)，世界坐标系 |
| 施加外力 | `env.apply_body_force(name, f, τ)` | 世界坐标系 |
| 清除外力 | `env.clear_body_force(name)` | 清除指定 body |
| 清除全部 | `env.clear_all_forces()` | 清除所有外力 |

---

## 下一步

掌握了接触和力，接下来学习如何用**等式约束抓取物体**：[🔗 等式约束](equality-constraints.md)。
