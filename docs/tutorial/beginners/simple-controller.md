# 🎮 简单控制器 — PD 控制

前面我们用手动设置 `qpos` 或恒定力矩来驱动机器人。这一节，你将学会写一个**PD 控制器**——这是机器人控制中最基础也最常用的控制器。

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
目标位置 ──→ [Kp × 误差] ──→ ┐
                              ├──→ 力矩 ──→ 关节
当前速度 ──→ [Kd × 误差] ──→ ┘
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
    """
    nu = env.model.nu
    pd = SimplePDController(nu=nu, kp=150.0, kd=12.0)

    # 目标关节位置 —— 只有关节 0 动，其他保持原位
    target_qpos = env.data.qpos[:nu].copy()
    target_qpos[0] = target_angle  # 关节 0 转到 target_angle 弧度

    print(f"目标角度: {target_qpos[0]:.3f} rad")
    print(f"初始角度: {env.data.qpos[0]:.3f} rad")

    for i in range(steps):
        # PD 计算力矩
        ctrl = pd.compute(
            target_qpos=target_qpos,
            current_qpos=env.data.qpos,
            current_qvel=env.data.qvel,
        )

        # 执行仿真
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
