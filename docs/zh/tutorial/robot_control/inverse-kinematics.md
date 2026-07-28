# 🦾 逆运动学（IK）

逆运动学（Inverse Kinematics）将末端执行器的目标位姿转换为关节角度。

> 完整可运行代码见 [OrcaPlayground examples/euler/06_jacobian/](https://github.com/OrcaGym/OrcaPlayground)。

---

## 什么是 IK？

```
正运动学 (FK): 关节角度 → 末端位姿（唯一解）
逆运动学 (IK): 末端位姿 → 关节角度（可能有多个解或无解）
```

IK 的三大挑战：
1. **冗余**：末端 6 DOF，关节可能更多 → 无穷多解
2. **奇点**：某些姿态下雅可比退化 → 微小末端位移需要无限大关节速度
3. **关节限位**：解必须满足物理限位

---

## 完整示例：先看全貌

下面是一个**完整的 IK 示例**，将 G1 左脚抬高约 10cm。
它使用**阻尼最小二乘法 + 关节限位 clamp + 两阶段策略**。

```python
"""阻尼最小二乘 IK：G1 左脚抬高 ~10cm"""
import numpy as np


# G1 的 29 个旋转关节后缀
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


def damped_least_squares_ik(env, foot_suffix="left_ankle_roll_link",
                            offset=np.array([0.0, 0.05, 0.10]),
                            damping=0.05, step=0.05, iters=80, atol=0.02):
    """阻尼最小二乘 IK：将脚部 body 移动到目标偏移位置。

    两阶段：
      1. 预设微蹲姿态 — 避免从伸直状态走反关节路径
      2. IK 迭代 — 阻尼最小二乘 + 关节限位 clamp

    Args:
        env: OrcaGymEulerEnv 实例
        foot_suffix: 脚部 body 后缀（不含 agent 前缀）
        offset: 目标偏移 [dx, dy, dz]（世界坐标系，米）
        damping: 阻尼系数（越大越稳定，越小收敛越快）
        step: 每步最大关节变化（弧度）
        iters: 最大迭代次数
        atol: 收敛阈值（米）

    Returns:
        最终脚部位置 (3,) np.ndarray
    """
    agent = env.agent_name
    foot_body = f"{agent}_{foot_suffix}"

    # ── 准备 G1 关节信息 ──
    joint_names = [f"{agent}_{s}" for s in G1_ROT_JOINT_SUFFIXES]
    dof_adrs = [env.jnt_dofadr(jn) for jn in joint_names]
    qpos_adrs = [env.jnt_qposadr(jn) for jn in joint_names]

    # G1 关节在全局 dof 中的列范围（多 body 不能用 [7:]！）
    v_min, v_max = min(dof_adrs), max(dof_adrs)
    g1_joint_cols = slice(v_min, v_max + 1)

    # 关节限位
    jdict = env.model.get_joint_dict()
    jnt_lo = np.array([
        jdict[jn]["Range"][0] if jdict[jn]["Limited"] else -np.inf
        for jn in joint_names
    ])
    jnt_hi = np.array([
        jdict[jn]["Range"][1] if jdict[jn]["Limited"] else np.inf
        for jn in joint_names
    ])

    # ═══════════════════════════════════════════
    # 阶段 1：预设微蹲姿态
    # ═══════════════════════════════════════════
    # 从伸直状态出发，纯 DLS 可能走"膝盖后弯"的反关节路径。
    # 预设膝盖前弯 + 髋前屈 + 踝背屈，让 IK 从合理姿态出发。
    preset = {
        f"{agent}_left_knee_joint": 0.6,
        f"{agent}_left_hip_pitch_joint": -0.3,
        f"{agent}_left_ankle_pitch_joint": -0.3,
        f"{agent}_right_knee_joint": 0.6,
        f"{agent}_right_hip_pitch_joint": -0.3,
        f"{agent}_right_ankle_pitch_joint": -0.3,
    }
    qpos = env.data.qpos.copy()
    for jn, val in preset.items():
        qpos[env.jnt_qposadr(jn)] = val
    env.set_joint_qpos(qpos)
    env.mj_forward()
    print(f"  阶段 1 完成：预设微蹲姿态")

    # ═══════════════════════════════════════════
    # 阶段 2：阻尼最小二乘 IK 迭代
    # ═══════════════════════════════════════════
    foot_pos = env.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
    target = foot_pos + offset
    print(f"  阶段 2 开始：目标={target}，初始={foot_pos}")

    jacr = np.zeros((3, env.model.nv))
    for i in range(iters):
        # (a) 计算脚部雅可比
        jacp_foot = np.zeros((3, env.model.nv))
        env.mj_jacBody(jacp_foot, jacr, body_name=foot_body)

        # (b) 当前误差
        cur = env.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
        delta = target - cur

        # (c) 阻尼最小二乘：dq = J^T (J J^T + λ²I)^(-1) Δx
        jac_leg = jacp_foot[:, g1_joint_cols]
        dq = jac_leg.T @ np.linalg.inv(
            jac_leg @ jac_leg.T + damping**2 * np.eye(3)
        ) @ delta

        # (d) 合规写入 + 限位 clamp
        qpos = env.data.qpos.copy()
        for j, qadr in enumerate(qpos_adrs):
            qpos[qadr] = np.clip(
                qpos[qadr] + dq[j] * step, jnt_lo[j], jnt_hi[j]
            )
        env.set_joint_qpos(qpos)
        env.mj_forward()

        # (e) 收敛判定
        err = np.linalg.norm(delta)
        if err < atol:
            print(f"  ✅ IK 收敛于迭代 {i + 1}，误差 {err:.4f}m")
            break
    else:
        print(f"  ⚠️ IK 未收敛，最终误差 {err:.4f}m")

    return env.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # env 是你的 OrcaGymEulerEnv 实例（已 reset）
    final_pos = damped_least_squares_ik(
        env,
        foot_suffix="left_ankle_roll_link",
        offset=np.array([0.0, 0.05, 0.10]),  # y+5cm, z+10cm
    )
    print(f"左脚最终位置: {final_pos}")
```

---

## 逐段解释

### 阻尼最小二乘原理

标准雅可比伪逆 `J⁺ = J^T(J J^T)^(-1)` 在奇点附近会发散。
**阻尼最小二乘法（Damped Least Squares）** 加入正则化项 λ²I：

```
dq = J^T (J J^T + λ²I)^(-1) Δx
```

| λ 值 | 行为 |
|------|------|
| λ = 0 | 退化为标准伪逆，奇点处不稳定 |
| λ 小 | 收敛快，但奇点附近可能震荡 |
| λ 大 | 解更平滑稳定，但收敛慢 |

### IK 迭代流程

```
每步迭代:
  1. mj_jacBody → 脚部雅可比 jacp_foot (3, nv)
  2. jac_leg = jacp_foot[:, g1_dof_min:g1_dof_max+1]  ← 只取 G1 关节列
  3. Δx = target - current_xpos
  4. dq = J^T (J J^T + λ²I)^(-1) Δx   ← 阻尼最小二乘
  5. q ← clip(q + dq·step, jnt_lo, jnt_hi)  ← 限位 clamp
  6. set_joint_qpos(q) + mj_forward()
  7. 检查 ||Δx|| < ATOL?
```

### 为什么需要两阶段？

G1 默认 qpos=0 时膝盖**完全伸直**。从伸直状态出发，纯数学 DLS 解可能让膝盖**向后弯**
（反关节方向）——这在数学上正确但物理上不可行。

**阶段 1 — 预设微蹲**：
- 膝盖前弯 +0.6 rad（≈34°）
- 髋前屈 -0.3 rad
- 踝背屈 -0.3 rad（补偿使脚底水平）
- 双腿对称

这样 IK 从已弯曲状态出发，解会自然地沿关节正向继续弯曲抬脚。

**阶段 2 — IK 迭代抬脚**：阻尼最小二乘 + 限位 clamp，80 次迭代收敛到 ~2cm 精度。

### 多 body 场景的 dof 列

G1 关节在**全局 dof 数组**中的列范围**不是 `[7:]`**！多 body 场景中
（G1 + 操纵物体 + 玩具），其他 body 的 dof 可能穿插其中。

**正确做法**：通过 `jnt_dofadr` 逐个获取各关节的 dof 地址，构造 `[min, max+1]` 范围：

```python
dof_adrs = [env.jnt_dofadr(jn) for jn in joint_names]
g1_joint_cols = slice(min(dof_adrs), max(dof_adrs) + 1)
```

### 关节限位 clamp

每次 IK 迭代后 clamp 到关节限位，防止生成不可达的姿态：

```python
jdict = env.model.get_joint_dict()
for jn in joint_names:
    lo, hi = jdict[jn]["Range"]       # [lower, upper]
    limited = jdict[jn]["Limited"]    # True/False

# clamp
qpos[qadr] = np.clip(qpos[qadr] + dq * step, lo, hi)
```

---

## 参数速查

| 参数 | 作用 | 推荐值 | 调优 |
|------|------|--------|------|
| `damping` | 数值稳定性 | 0.01–0.1 | 不收敛时↑，太慢时↓ |
| `step` | 每步关节变化 | 0.02–0.1 | 震荡时↓ |
| `iters` | 最大迭代次数 | 50–200 | 目标远时↑ |
| `atol` | 收敛阈值 (m) | 0.01–0.05 | 精度要求高时↓ |

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| IK 不收敛 | 阻尼太小/目标太远 | 增大 damping，减小 offset |
| 关节反方向弯 | 从伸直状态出发 | 预设微蹲姿态 |
| 雅可比全为零 | 未 forward/body 名错误 | 确保 mj_forward + agent 前缀 |
| 多 body 场景异常 | dof 列范围错 | 用 jnt_dofadr 取 [min, max] |

---

## 下一步

掌握了 IK，接下来学习如何**施加外力**和完整的力+IK 工作流：[🔄 外力应用与 IK](../physics/force-apply.md)。
