# 🎮 简单控制器 — PD 控制

前面我们用手动设置 `qpos` 或恒定力矩来驱动机器人。这一节，你将学会写一个**PD 控制器**——这是机器人控制中最基础也最常用的控制器。

> 本节结合 [OrcaPlayground Lesson 8 - Locomotion](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/08_locomotion) 的 G1 行走样例进行讲解，这是 PD 控制链路的核心课程。

---

## 什么是 PD 控制器？

PD = Proportional（比例）+ Derivative（微分）

```
力矩 = Kp × (目标位置 - 当前位置) + Kd × (目标速度 - 当前速度)
  τ  = Kp ×      Δpos            + Kd ×      Δvel
```

**直觉理解**：

- **P 项（比例）**：离目标越远，力气越大 —— 像一根弹簧把你拉向目标
- **D 项（微分）**：速度越快，阻力越大 —— 像阻尼器防止你冲过头

```
目标位置 ──→ [Kp × 位置误差] ──→ ┐
                                ├──→ 力矩 ──→ 关节
当前速度 ──→ [Kd × 速度误差] ──→ ┘
```

---

## 从零实现一个 PD 控制器

```python
"""
simple_pd_controller.py — 从零实现的 PD 控制器
"""

import numpy as np


class SimplePDController:
    """
    多关节 PD 控制器。

    对每个关节独立计算力矩:
        torque = kp * (target - current_pos) + kd * (0 - current_vel)

    参数:
        kp: 比例增益 — 越大跟踪越快，但可能振荡
        kd: 微分增益 — 越大越稳定，但可能迟钝

    单位:
        kp: N·m/rad（旋转关节）—— 每弧度位置误差产生的力矩
        kd: N·m·s/rad（旋转关节）—— 每弧度/秒速度误差产生的力矩
    """

    def __init__(self, nu: int, kp: float = 100.0, kd: float = 10.0):
        """
        Args:
            nu: 执行器数量（关节数）
            kp: 比例增益，典型值 50~500
            kd: 微分增益，典型值 5~50
        """
        self.nu = nu
        self.kp = np.full(nu, kp, dtype=np.float64)  # 每个关节可以有不同的增益
        self.kd = np.full(nu, kd, dtype=np.float64)

    def compute(self, target_qpos, current_qpos, current_qvel):
        """
        计算控制力矩。

        Args:
            target_qpos: 目标关节位置 (nu,) —— "想去哪"
            current_qpos: 当前关节位置 (nq,) —— "现在在哪"
            current_qvel: 当前关节速度 (nv,) —— "现在多快"

        Returns:
            torque: 控制力矩 (nu,) —— "该用多大劲"
        """
        pos_error = target_qpos - current_qpos[:self.nu]
        vel_error = np.zeros(self.nu) - current_qvel[:self.nu]
        torque = self.kp * pos_error + self.kd * vel_error
        return torque.astype(np.float64)


# ============================================================
# 在环境中使用 PD 控制器
# ============================================================

def demo_pd_control(env, target_angle: float = 0.8, steps: int = 150):
    """
    演示：用 PD 控制器驱动关节 0 到目标角度。

    观察关节如何平滑地到达目标（而不是瞬移）。

    注意：这里每步只调用一次 do_simulation，属于"开环简化"演示。
    对于 frame_skip > 1 或高动态场景（如人形机器人行走），
    建议使用后文介绍的"闭环 PD"写法（每物理步重算 tau）。
    """
    nu = env.model.nu
    pd = SimplePDController(nu=nu, kp=150.0, kd=12.0)

    # 目标关节位置 —— 只有关节 0 动，其他保持原位
    target_qpos = env.data.qpos[:nu].copy()
    target_qpos[0] = target_angle  # 关节 0 转到 target_angle 弧度

    print(f"目标角度: {target_qpos[0]:.3f} rad")
    print(f"初始角度: {env.data.qpos[0]:.3f} rad")

    for i in range(steps):
        # PD 计算力矩（每步重算，简化演示用）
        ctrl = pd.compute(
            target_qpos=target_qpos,
            current_qpos=env.data.qpos,
            current_qvel=env.data.qvel,
        )

        # 执行仿真（开环：一次 tau，步进 frame_skip 次）
        env.do_simulation(ctrl, env.frame_skip)
        env.render()

        # 打印进度
        pos_error = abs(target_qpos[0] - env.data.qpos[0])
        if i % 15 == 0:
            print(f"  Step {i:3d}: pos={env.data.qpos[0]:+.4f}, "
                  f"error={pos_error:.4f}, torque={ctrl[0]:+.2f}")

        # 到达目标后提前退出
        if pos_error < 0.001:
            print(f"  ✅ 到达目标！耗时 {i} 步")
            break

    print(f"最终角度: {env.data.qpos[0]:.3f} rad")
```

---

## 实战样例：G1 行走的 PD 控制链路

上面的 `SimplePDController` 展示了基本原理。真实机器人控制中，每个关节往往需要**独立的 Kp/Kd**，并且需要 **力矩限位**（clip）防止过载。

[OrcaPlayground Lesson 8](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/08_locomotion) 中的 G1 人形机器人行走控制就是典型的 PD 控制链路：

```
策略输出 q_target (29维) ──→ PD 控制器 ──→ tau 力矩 ──→ motor 执行器 ──→ 行走
                              ↑
                     Kp, Kd, 力矩限位（来自 YAML 配置）
```

### 1. 每个关节独立的 Kp/Kd

G1 共 29 个关节（腿 12 + 腰 3 + 双臂 14），不同部位负载差异大，因此每关节配置独立的 PD 增益。配置位于 [assets/g1/config/g1_29dof_hist.yaml](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/assets/g1/config/g1_29dof_hist.yaml)：

```yaml
# 腿部（髋 + 膝 + 踝）：承重，需要大刚度
JOINT_KP: [
    100, 100, 100, 200, 20, 20,     # 左腿：髋 yaw/roll/pitch=100, 膝=200, 踝=20
    100, 100, 100, 200, 20, 20,     # 右腿：同上
    400, 400, 400,                  # 腰部：yaw/roll/pitch=400（核心稳定）
    90, 60, 20, 60, 4, 4, 0,        # 左臂：肩 pitch=90, roll=60, yaw=20, 肘=60, 腕=4/4/0
    90, 60, 20, 60, 4, 4, 4         # 右臂：同左臂
]

JOINT_KD: [
    2.5, 2.5, 2.5, 5, 0.2, 0.1,     # 左腿阻尼
    2.5, 2.5, 2.5, 5, 0.2, 0.1,     # 右腿阻尼
    5.0, 5.0, 5.0,                  # 腰部阻尼
    2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2,  # 左臂阻尼
    2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2   # 右臂阻尼
]
```

**规律一眼可见**：

| 部位 | Kp | Kd | 物理意义 |
|------|-----|-----|---------|
| 腰部 | 400 | 5.0 | 承担上身重量，刚度最大 |
| 膝关节 | 200 | 5.0 | 单腿承重，需要大刚度 |
| 髋部 | 100 | 2.5 | 摆腿驱动 |
| 肩部 | 90 / 60 | 2.0 / 1.0 | 手臂较轻，增益较低 |
| 腕部 | 0~4 | 0.2 | 末端灵活，几乎不发力 |

### 2. PD 控制器实现（带力矩限位）

`G1Locomotion.compute_tau` 是核心 PD 控制方法：

```python
def compute_tau(self, q_target, dof_pos, dof_vel):
    """
    PD 控制器：位置目标 q_target → 力矩 tau

        tau = Kp * (q_target - q) + Kd * (0 - qd)

    前馈力矩 tau_ff=0，目标速度 dq_target=0（与 G1 训练侧一致）
    """
    tau = self.joint_kp * (q_target - dof_pos) - self.joint_kd * dof_vel
    # 力矩限位：防止超过电机额定输出
    tau = np.clip(tau, -self.motor_effort_limit, self.motor_effort_limit)
    return tau
```

与上面的 `SimplePDController` 的区别：

| 项 | `SimplePDController` | `G1Locomotion.compute_tau` |
|----|---------------------|--------------------------|
| Kp / Kd | 标量（所有关节共用） | 每关节独立（向量） |
| 力矩限位 | 无 | `np.clip` 到 `motor_effort_limit` |
| 目标速度 | 默认为 0 | 默认为 0（前馈 `tau_ff=0`） |

> 💡 **力矩限位** 是真实机器人控制中不可或缺的安全机制，防止瞬时大电流损坏电机或导致机器人失稳。G1 的 `motor_effort_limit` 同样配置在 yaml 中。

### 3. 闭环 PD：每物理步重算 tau

开环 PD（一次算 tau，循环步进多次）会累积误差导致失稳。闭环 PD 在**每个物理步**都重新读取关节状态并重算力矩：

```python
def step(self, action):
    """以 frame_skip=1 多次调用 do_simulation 实现精细 PD 控制"""
    action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
    for _ in range(self.frame_skip):
        ctrl = self._pd_controller(action)   # 每步重算 tau
        self.do_simulation(ctrl, 1)           # 单步物理仿真
    obs = self._get_obs()
    reward = self._compute_reward(obs, action)
    ...
```

`_pd_controller` 是一个 hook，子类按场景复写：

```python
class G1LocomotionEnv(G1BaseEnv):
    def _pd_controller(self, target):
        """复写为 G1 行走的 PD 控制"""
        dof_pos, dof_vel = self._loco.read_joint_state(self)  # 读当前关节状态
        return self._loco.compute_tau(target, dof_pos, dof_vel)
```

**闭环 vs 开环对比**：

| 方式 | 做法 | 问题 |
|------|------|------|
| 开环 PD | `tau = pd.compute(target, q₀, v₀)` → `do_simulation(tau, frame_skip)` | frame_skip 步内状态已变，tau 仍基于旧 obs，累积误差 |
| 闭环 PD ⭐ | 每物理步重读 obs 重算 tau | tau 始终基于最新状态，稳定跟踪 |

> 💡 闭环 PD 是 OrcaPlayground G1 系列样例（Lesson 6~10）的标准写法，也是人形机器人控制的事实标准。你在自己的环境中也建议采用这种方式。

---

## PD 参数调优

调 PD 参数就像调热水龙头——太热加点冷水，太冷加点热水：

| 现象 | 原因 | 调整 |
|------|------|------|
| 关节振荡、来回抖动 | Kp 太大 | ⬇️ 减小 Kp，⬆️ 增大 Kd |
| 响应慢、跟不上目标 | Kp 太小 | ⬆️ 增大 Kp |
| 到达后停不住、微微振 | Kd 太小 | ⬆️ 增大 Kd |
| 像在蜂蜜里动、太迟钝 | Kd 太大 | ⬇️ 减小 Kd |

**推荐的调优流程**：

```python
# 第 1 步：Kd=0，逐渐增大 Kp 直到关节开始轻微振荡
pd = SimplePDController(nu=nu, kp=50, kd=0)   # 试
pd = SimplePDController(nu=nu, kp=100, kd=0)  # 再试
pd = SimplePDController(nu=nu, kp=200, kd=0)  # 振荡了！回退到 150

# 第 2 步：固定 Kp=150，逐渐增大 Kd 直到振荡消失
pd = SimplePDController(nu=nu, kp=150, kd=5)   # 试
pd = SimplePDController(nu=nu, kp=150, kd=10)  # 好多了
pd = SimplePDController(nu=nu, kp=150, kd=15)  # 有点迟钝，回退到 12
```

### 不同场景的参考值

| 场景 | Kp | Kd | 说明 |
|------|-----|-----|------|
| 轻量机械臂 (≤2kg) | 80~150 | 8~15 | 惯性小，低增益即可 |
| 重型机械臂 (≥10kg) | 200~500 | 20~50 | 需要更大的力 |
| 高精度任务 | 300~500 | 30~50 | 需要快速响应 |
| 人机协作 | 50~100 | 15~30 | 安全优先，不能太"硬" |
| 人形机器人腰部 | 400 | 5.0 | 承担上身重量，刚度最大（参考 G1） |
| 人形机器人腕部 | 0~4 | 0.2 | 末端灵活，几乎不发力（参考 G1） |

> 💡 复杂机器人建议**每关节独立配置 Kp/Kd**，参考 G1 的 yaml 配置方式，按部位（髋/膝/腰/肩/腕）分组调整。

---

## 控制器在环境中的位置

```
你的 step() 方法
│
├── action (策略/程序的输出)
│     │
│     ├── 位置控制: action = 目标关节角度
│     │   └── PD.compute(target=action, ...) → 力矩
│     │
│     ├── 增量控制: action = 角度增量
│     │   └── target = current + action
│     │       └── PD.compute(target, ...) → 力矩
│     │
│     └── 力矩控制: action 就是力矩
│         └── 直接 do_simulation(action, ...)
│
└── do_simulation(ctrl, frame_skip)
```

**闭环写法**（推荐，参考 G1 样例）：

```python
def step(self, action):
    action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
    for _ in range(self.frame_skip):
        ctrl = self._pd_controller(action)  # 每物理步重算
        self.do_simulation(ctrl, 1)
    return self._get_obs(), reward, terminated, truncated, info
```

---

## 完整示例：位置控制环境

```python
class PositionControlEnv(OrcaGymEulerEnv):
    """动作 = 目标关节角度，内部用 PD 转化为力矩"""

    def __init__(self, frame_skip, orcagym_addr, agent_names, time_step, **kwargs):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        # 创建 PD 控制器
        self._pd = SimplePDController(nu=self.model.nu, kp=150.0, kd=12.0)

        # 动作空间 = 关节限位范围
        ranges = []
        for i in range(self.model.nu):
            joint_name = self.model.joint_id2name(i)
            info = self.model.get_joint_byname(joint_name)
            if info.get("Limited", False):
                ranges.append(info["Range"])
            else:
                ranges.append([-3.14, 3.14])

        self.action_space = spaces.Box(
            low=np.array([r[0] for r in ranges], dtype=np.float32),
            high=np.array([r[1] for r in ranges], dtype=np.float32),
        )
        obs = self._get_obs()
        self.observation_space = spaces.Dict({
            "joint_pos": spaces.Box(-np.inf, np.inf, shape=(self.model.nq,)),
            "joint_vel": spaces.Box(-np.inf, np.inf, shape=(self.model.nv,)),
        })

    def _get_obs(self):
        return {
            "joint_pos": self.data.qpos.copy().astype(np.float32),
            "joint_vel": self.data.qvel.copy().astype(np.float32),
        }

    def step(self, action):
        # action = 目标关节角度
        ctrl = self._pd.compute(
            target_qpos=action,
            current_qpos=self.data.qpos,
            current_qvel=self.data.qvel,
        )
        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        tracking_error = np.mean(np.abs(action - self.data.qpos[:self.model.nu]))
        reward = -tracking_error  # 追踪越准，奖励越高
        return obs, reward, False, False, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}
```

---

## 下一步

你学会了如何精确控制关节。现在把前面学的所有知识组合起来，**搭建一个完整的任务**：[🏆 搭建一个任务](build-a-task.md)。
