# 🔄 外力应用、状态写入与 IK — 控制仿真状态

本节介绍如何**施加外力、写入状态、使用雅可比矩阵和逆运动学（IK）**控制仿真。

> 完整可运行代码见 [OrcaPlayground examples/euler/05_force_apply/](https://github.com/OrcaGym/OrcaPlayground) 和 [06_jacobian/](https://github.com/OrcaGym/OrcaPlayground)。

---

## 完整示例：先看全貌

下面是一个**可以直接运行**的完整示例，展示了外力施加、状态写入、雅可比计算和 IK 的所有核心操作。
建议先通读一遍，再看后面的逐段解释。

```python
"""完整示例：外力 + IK 控制 G1 人形机器人

功能：
  1. 对 pelvis 施加向上力抬起机器人
  2. 清除外力并验证
  3. 设置摩擦系数
  4. 用 mocap + weld 约束拖拽物体
  5. 用阻尼最小二乘 IK 抬起左脚

前提：需要 OrcaStudio 在线运行，加载含 G1 的场景
"""
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


# ================================================================
# G1 关节定义（29 个旋转关节）
# ================================================================
G1_ROT_JOINT_SUFFIXES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class ForceAndIKDemo(OrcaGymEulerEnv):
    """外力 + IK 演示环境"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 20),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["g1"]),
            time_step=kwargs.pop("time_step", 0.001),
            model_xml_path=model_xml_path,
            **kwargs,
        )

    def demo_force_apply(self):
        """演示 1：施加外力抬起机器人"""
        agent = self.agent_name

        # 记录初始高度
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        z_before = float(pelvis[f"{agent}_pelvis"]["xpos"][2])
        print(f"初始 pelvis 高度: {z_before:.3f}m")

        # 施加 500N 向上力
        self.apply_body_force(
            f"{agent}_pelvis",
            force=np.array([0.0, 0.0, 500.0]),
            torque=np.array([0.0, 0.0, 0.0]),
        )

        # 步进 20 个控制周期让力生效
        ctrl = np.zeros(self.model.nu)
        for _ in range(20):
            self.do_simulation(ctrl, self.frame_skip)

        # 验证上升
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        z_after = float(pelvis[f"{agent}_pelvis"]["xpos"][2])
        print(f"施力后 pelvis 高度: {z_after:.3f}m (Δ={z_after - z_before:.3f}m)")

        # 验证 xfrc_applied 记录了力
        body_id = self.model.body_name2id(f"{agent}_pelvis")
        xfrc = self.data.xfrc_applied[body_id, :3]
        print(f"xfrc_applied 记录的力: {xfrc}")

        # 清除外力
        self.clear_body_force(f"{agent}_pelvis")
        xfrc = self.data.xfrc_applied[body_id, :3]
        assert np.all(xfrc == 0), "清力后 xfrc 应为零"
        print("✅ 外力已清除")

    def demo_mocap_drag(self):
        """演示 2：mocap + weld 约束拖拽物体"""
        agent = self.agent_name

        # 设置 mocap 目标位姿
        target_pos = np.array([0.7, 0.0, 0.5])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.set_mocap_pos_and_quat({
            f"{agent}_TestMocapAnchor": {
                "pos": target_pos,
                "quat": target_quat,
            }
        })

        # 回读验证写入一致
        read_pos = self.data.mocap_pos(f"{agent}_TestMocapAnchor")
        read_quat = self.data.mocap_quat(f"{agent}_TestMocapAnchor")
        assert np.allclose(read_pos, target_pos, atol=1e-6), "mocap 位置回读不一致"
        assert np.allclose(read_quat, target_quat, atol=1e-6), "mocap 四元数回读不一致"
        print(f"✅ mocap 写入回读一致: pos={read_pos}")

        # 步进让 weld 约束生效 → 物体跟随 mocap
        ctrl = np.zeros(self.model.nu)
        for _ in range(10):
            self.do_simulation(ctrl, self.frame_skip)

        # 验证物体已跟随
        box = self.get_body_xpos_xmat_xquat([f"{agent}_manipulation_box"])
        box_pos = box[f"{agent}_manipulation_box"]["xpos"]
        print(f"物体位置: {box_pos} (目标: {target_pos})")
        print(f"✅ weld 约束驱动物体跟随 mocap")

    def demo_ik_lift_foot(self):
        """演示 3：阻尼最小二乘 IK 抬起左脚"""
        agent = self.agent_name
        foot_body = f"{agent}_left_ankle_roll_link"

        # --- 准备 G1 关节信息 ---
        joint_names = [f"{agent}_{s}" for s in G1_ROT_JOINT_SUFFIXES]
        dof_adrs = [self.jnt_dofadr(jn) for jn in joint_names]
        qpos_adrs = [self.jnt_qposadr(jn) for jn in joint_names]

        # G1 关节在全局 dof 中的列范围（多 body 场景不能直接用 [7:]）
        v_min, v_max = min(dof_adrs), max(dof_adrs)
        g1_joint_cols = slice(v_min, v_max + 1)

        # 关节限位
        jdict = self.model.get_joint_dict()
        jnt_lo = np.array([
            jdict[jn]["Range"][0] if jdict[jn]["Limited"] else -np.inf
            for jn in joint_names
        ])
        jnt_hi = np.array([
            jdict[jn]["Range"][1] if jdict[jn]["Limited"] else np.inf
            for jn in joint_names
        ])

        # --- 阶段 1：预设微蹲姿态（避免反关节路径）---
        preset = {
            f"{agent}_left_knee_joint": 0.6,
            f"{agent}_left_hip_pitch_joint": -0.3,
            f"{agent}_left_ankle_pitch_joint": -0.3,
            f"{agent}_right_knee_joint": 0.6,
            f"{agent}_right_hip_pitch_joint": -0.3,
            f"{agent}_right_ankle_pitch_joint": -0.3,
        }
        qpos = self.data.qpos.copy()
        for jn, val in preset.items():
            qpos[self.jnt_qposadr(jn)] = val
        self.set_joint_qpos(qpos)
        self.mj_forward()
        print("阶段 1 完成：预设微蹲姿态")

        # --- 阶段 2：IK 迭代抬左脚 ---
        DAMPING = 0.05
        STEP = 0.05
        ITERS = 80
        ATOL = 0.02

        foot_pos = self.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
        target = foot_pos + np.array([0.0, 0.05, 0.10])  # 抬高 ~10cm

        jacr = np.zeros((3, self.model.nv))
        for i in range(ITERS):
            # 雅可比
            jacp_foot = np.zeros((3, self.model.nv))
            self.mj_jacBody(jacp_foot, jacr, body_name=foot_body)
            cur = self.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
            delta = target - cur

            # 阻尼最小二乘：dq = J^T (J J^T + λ²I)^(-1) Δx
            jac_leg = jacp_foot[:, g1_joint_cols]
            dq = jac_leg.T @ np.linalg.inv(
                jac_leg @ jac_leg.T + DAMPING**2 * np.eye(3)
            ) @ delta

            # 合规写入 + 限位 clamp
            qpos = self.data.qpos.copy()
            for j, qadr in enumerate(qpos_adrs):
                qpos[qadr] = np.clip(qpos[qadr] + dq[j] * STEP, jnt_lo[j], jnt_hi[j])
            self.set_joint_qpos(qpos)
            self.mj_forward()

            err = np.linalg.norm(delta)
            if err < ATOL:
                print(f"IK 收敛于迭代 {i + 1}，误差 {err:.4f}m")
                break

        final = self.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
        print(f"左脚: 初始={foot_pos}, 最终={final}, 目标={target}")
        print(f"误差: {np.linalg.norm(final - target):.4f}m")

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


# ================================================================
# 运行
# ================================================================
if __name__ == "__main__":
    env = ForceAndIKDemo(
        model_xml_path="/path/to/g1_29dof_camera.xml",
        skip_grpc_load=False,  # 在线模式连接 Studio
    )
    env.reset()

    env.demo_force_apply()
    env.demo_mocap_drag()
    env.demo_ik_lift_foot()

    env.close()
```

---

## 逐段解释

### 1. 外力应用

```python
env.apply_body_force(
    "g1_pelvis",                          # body 名称（含 agent 前缀）
    force=np.array([0.0, 0.0, 500.0]),    # 力 (N)，世界坐标系
    torque=np.array([0.0, 0.0, 0.0]),     # 力矩 (N·m)，世界坐标系
)
```

**原理**：`apply_body_force` 直接在 MuJoCo 的 `xfrc_applied` 数组中写入力/力矩。
力作用在 body 的质心，力矩绕 body 质心。这些力会在下一次 `mj_step()` 时参与动力学计算。

**验证**：通过 `env.data.xfrc_applied[body_id, :3]` 可以读到当前施加的力（零拷贝只读视图）：
```python
body_id = env.model.body_name2id("g1_pelvis")
xfrc = env.data.xfrc_applied[body_id, :3]  # [fx, fy, fz]
```

**清除**：
```python
env.clear_body_force("g1_pelvis")   # 清除单个 body 的力
env.clear_all_forces()              # 清除所有外力
```

> **注意**：G1 使用力控 motor，`ctrl=0` 时关节无力矩。施加外力时应选择能直接受力的 body（如 pelvis），
> 避免对松软关节链上的 body 施力（力会被关节吸收）。

### 2. Mocap 拖拽

**Mocap body** 是 MuJoCo 中的特殊 body（`body_mocapid != -1`），可以**直接设置位姿**，
不受力/动力学影响。配合 **WELD 等式约束**，可以像"用看不见的手拖拽"一样移动普通 body。

```python
# 写入 mocap 位姿
env.set_mocap_pos_and_quat({
    "mocap_name": {
        "pos": np.array([x, y, z]),
        "quat": np.array([w, x, y, z]),
    }
})

# 回读验证
read_pos = env.data.mocap_pos("mocap_name")    # (3,)
read_quat = env.data.mocap_quat("mocap_name")  # (4,)
```

**完整拖拽流程**：
1. `anchor_actor("object", "weld")` — 建立 WELD 约束连接 mocap 和物体
2. `set_mocap_pos_and_quat(...)` — 移动 mocap → 物体跟随
3. `do_simulation(...)` — 步进让约束生效
4. `release_body_anchored()` — 释放

### 3. 状态写入

**合规的状态写入方式**（W1 规则）：`copy → 修改 → set → forward`

```python
# ❌ 错误：直接写 data.qpos（只读视图）
# self.data.qpos[0] = 0.5

# ✅ 正确
qpos = env.data.qpos.copy()       # 1. 复制
qpos[addr] = new_value             # 2. 修改副本
env.set_joint_qpos(qpos)           # 3. 合规写入
env.mj_forward()                   # 4. 必须！更新派生量
env._sync_view()                   # 5. 同步到 DataView
```

> ⚠️ **关键**：修改 qpos/qvel 后**必须调用 `mj_forward()`**。不调用的话，
> `get_body_xpos_xmat_xquat` 等读到的 body 位姿仍然是旧值。

### 4. 雅可比矩阵

**`mj_jacBody`** — 计算指定 body 的平移和旋转雅可比：

```python
nv = env.model.nv
jacp = np.zeros((3, nv))   # 平移雅可比 (3, nv) — 原地写入
jacr = np.zeros((3, nv))   # 旋转雅可比 (3, nv)

env.mj_jacBody(jacp, jacr, body_name="g1_pelvis")

# 数学关系：
# jacp @ qvel = body 世界坐标线速度
# jacr @ qvel = body 世界坐标角速度
```

**`mj_jacSite`** — 计算 site 点的雅可比：

```python
jacp_site = np.zeros((3, env.model.nv))
env.mj_jacSite(jacp_site, jacr_site, site_name="g1_imu")

# 验证一致性
xvalp, _ = env.query_site_xvalp_xvalr(["g1_imu"])
expected = jacp_site @ env.data.qvel       # jac @ qvel 应等于查询速度
assert np.allclose(xvalp["g1_imu"], expected, atol=1e-4)
```

### 5. 阻尼最小二乘 IK

**为什么需要阻尼？** 标准雅可比伪逆 `J⁺ = J^T(J J^T)^(-1)` 在奇点附近会发散
（`J J^T` 接近奇异，伪逆元素趋于无穷大）。阻尼最小二乘法加入正则化项 λ²I：

```
dq = J^T (J J^T + λ²I)^(-1) Δx
```

- **λ 太小** → 接近伪逆，奇点处不稳定
- **λ 太大** → 收敛慢，但稳定

**为什么需要两阶段？**

G1 默认 `qpos=0` 时膝盖完全伸直。从伸直状态出发，纯数学 IK 解可能让膝盖**向后弯**
（反关节方向）——这在数学上正确但物理上不可行。

**阶段 1 — 预设微蹲**：膝盖前弯 +0.6 rad、髋前屈 -0.3 rad、踝背屈 -0.3 rad。
这样 IK 从已弯曲状态出发，解会自然地沿关节正向继续弯曲抬脚。

**阶段 2 — IK 迭代**：
```
每步迭代:
  1. mj_jacBody → 脚部雅可比 jacp_foot
  2. jac_leg = jacp_foot[:, g1_dof_min:g1_dof_max+1]  ← 只取 G1 关节列
  3. dq = jac_leg^T @ (jac_leg @ jac_leg^T + λ²I)^(-1) @ Δx  ← DLS
  4. q ← clip(q + dq·step, jnt_lo, jnt_hi)  ← 限位 clamp
  5. set_joint_qpos + mj_forward
  6. 检查收敛: ||Δx|| < ATOL?
```

**多 body 场景的 dof 列**：G1 关节在全局 dof 数组中的位置不是 `[7:]`。
必须通过 `jnt_dofadr` 逐个获取各关节的 dof 地址，构造 `[min, max+1]` 范围。

```python
dof_adrs = [env.jnt_dofadr(jn) for jn in joint_names]
v_min, v_max = min(dof_adrs), max(dof_adrs)
g1_joint_cols = slice(v_min, v_max + 1)  # 正确的 G1 关节列范围
```

### 6. 关节限位 clamp

从 `model.get_joint_dict()` 读取限位信息，每次 IK 迭代后 clamp：

```python
jdict = env.model.get_joint_dict()
jnt_lo = np.array([
    jdict[jn]["Range"][0] if jdict[jn]["Limited"] else -np.inf
    for jn in joint_names
])
jnt_hi = np.array([
    jdict[jn]["Range"][1] if jdict[jn]["Limited"] else np.inf
    for jn in joint_names
])

# 每次迭代后
for j, qadr in enumerate(qpos_adrs):
    qpos[qadr] = np.clip(qpos[qadr] + dq[j] * STEP, jnt_lo[j], jnt_hi[j])
```

---

## 参数速查

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `DAMPING` | 阻尼系数 — 越大越稳定，越小收敛越快 | 0.01–0.1 |
| `STEP` | 每步最大关节变化（弧度） | 0.02–0.1 |
| `ITERS` | 最大迭代次数 | 50–200 |
| `ATOL` | 收敛阈值（m） | 0.01–0.05 |

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| IK 不收敛 | 阻尼太小 / 目标太远 | 增大 DAMPING，减小目标偏移 |
| 关节反方向弯曲 | 起始姿态不合适 | 预设微蹲姿态 |
| 雅可比全为零 | 未 `mj_forward` / body 名错误 | 确保 forward + 含 agent 前缀 |
| 多 body 场景 IK 异常 | dof 列范围用错 | 用 `jnt_dofadr` 取 [min, max] |
| `mj_forward()` 后位姿不对 | `set_joint_qpos` 传了不完整的 qpos | 传长度 `nq` 的完整数组 |

---

## 下一步

掌握了状态控制和 IK，接下来学习如何**让 G1 行走**：[🦿 关节控制](../robot_control/joint-control.md)。
