# 🏆 搭建一个任务 — 组合所有知识

这是新手教程的收官之作。我们将把前面学到的所有知识组合起来，搭建一个完整的任务：

> **让机械臂末端到达随机指定的目标点。**

**不使用 RL**——我们将手动编写控制逻辑（PD 控制器 + 简单的轨迹规划）。

---

## 任务定义

```
┌─────────────────────────────────────────┐
│ 任务：Reach Target │
│ │
│ 输入（观测）： │
│ - 关节角度 │
│ - 末端执行器位置 │
│ - 目标位置 │
│ │
│ 输出（动作）： │
│ - 每个关节的目标角度 │
│ │
│ 成功条件： │
│ - 末端到目标的距离 < 5mm │
│ │
│ 内部控制器： │
│ - PD 控制器（目标角度 → 力矩） │
└─────────────────────────────────────────┘
```

---

## 完整代码

```python
"""
reach_target_task.py — 完整的机械臂到达任务

整合了前面教程的所有知识：
- 环境子类化
- 状态查询
- 关节控制
- PD 控制器

运行方式:
 python reach_target_task.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


# ============================================================
# PD 控制器
# ============================================================
class PDController:
 """多关节 PD 控制器"""

 def __init__(self, nu: int, kp: float = 150.0, kd: float = 12.0):
 self.nu = nu
 self.kp = np.full(nu, kp, dtype=np.float64)
 self.kd = np.full(nu, kd, dtype=np.float64)

 def compute(self, target_qpos, current_qpos, current_qvel):
 pos_error = target_qpos - current_qpos[:self.nu]
 vel_error = np.zeros(self.nu) - current_qvel[:self.nu]
 return (self.kp * pos_error + self.kd * vel_error).astype(np.float64)


# ============================================================
# 主任务环境
# ============================================================
class ReachTargetTask(OrcaGymEulerEnv):
 """
 机械臂末端到达目标点任务。

 这个环境展示了如何将多个组件（状态查询、PD控制、奖励计算）
 组合成一个完整的工作任务。
 """

 # ── 任务参数 ──
 SUCCESS_THRESHOLD = 0.005 # 到达阈值（5mm）
 MAX_DELTA = 0.05 # 每步关节角度最大变化
 GOAL_WORKSPACE = np.array([0.2, -0.3, 0.1, # [x_min, y_min, z_min,
 0.6, 0.3, 0.5]) # x_max, y_max, z_max]

 def __init__(self, frame_skip, orcagym_addr, agent_names, time_step, **kwargs):
 super().__init__(
 frame_skip=frame_skip,
 orcagym_addr=orcagym_addr,
 agent_names=agent_names,
 time_step=time_step,
 **kwargs,
 )

 # ── 任务状态 ──
 self._goal_pos = np.zeros(3) # 目标位置（每 episode 随机采样）
 self._step_count = 0 # 当前步数

 # ── PD 控制器 ──
 self._pd = PDController(nu=self.model.nu, kp=150.0, kd=12.0)

 # ── 初始化空间 ──
 self.action_space = spaces.Box(
 low=-self.MAX_DELTA, high=self.MAX_DELTA,
 shape=(self.model.nu,), dtype=np.float32,
 )
 obs_sample = self._get_obs()
 self.observation_space = spaces.Dict({
 key: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32)
 for key, v in obs_sample.items()
 })

 print(f"ReachTargetTask 就绪: nu={self.model.nu}, "
 f"dt={self.dt:.4f}s, 控制频率={1.0/self.dt:.1f}Hz")

 # ================================================================
 # 观测
 # ================================================================
 def _get_obs(self) -> dict:
 """收集观测：关节状态 + 末端位姿 + 目标位置"""
 ee_site = self.site("end_effector")
 sites = self.query_site_pos_and_quat([ee_site])
 ee_pos = sites[ee_site]["xpos"].copy()

 dist = np.linalg.norm(ee_pos - self._goal_pos)

 return {
 "joint_pos": self.data.qpos.copy().astype(np.float32),
 "joint_vel": self.data.qvel.copy().astype(np.float32),
 "ee_pos": ee_pos.astype(np.float32),
 "goal_pos": self._goal_pos.astype(np.float32),
 "dist_to_goal": np.array([dist], dtype=np.float32),
 }

 # ================================================================
 # 仿真步进
 # ================================================================
 def step(self, action: np.ndarray):
 """
 执行一步仿真。

 action: 关节角度增量 (nu,)，每个值 ∈ [-MAX_DELTA, MAX_DELTA]
 """
 self._step_count += 1

 # 1. 限制增量幅度
 action = np.clip(action, -self.MAX_DELTA, self.MAX_DELTA)

 # 2. 目标位置 = 当前位置 + 增量
 target_qpos = self.data.qpos[:self.model.nu] + action

 # 3. PD 控制器 → 力矩 → 仿真步进
 ctrl = self._pd.compute(target_qpos, self.data.qpos, self.data.qvel)
 self.do_simulation(ctrl, self.frame_skip)

 # 4. 获取新观测
 obs = self._get_obs()

 # 5. 计算奖励
 reward, reached = self._compute_reward(obs, action)

 # 6. 检查截断
 truncated = self._step_count >= 300

 info = {
 "dist": obs["dist_to_goal"].item(),
 "step": self._step_count,
 }

 return obs, reward, reached, truncated, info

 # ================================================================
 # 奖励函数
 # ================================================================
 def _compute_reward(self, obs, action):
 """
 奖励设计：
 - 距离惩罚（密集）：每步都给，越近越好
 - 成功奖励（稀疏）：到达时给一次大额奖励
 - 动作平滑惩罚：不鼓励剧烈动作
 """
 dist = obs["dist_to_goal"].item()

 dist_reward = -dist
 reached = dist < self.SUCCESS_THRESHOLD
 success_reward = 50.0 if reached else 0.0
 action_penalty = -0.01 * np.sum(action ** 2)

 reward = dist_reward + success_reward + action_penalty
 return reward, reached

 # ================================================================
 # 重置
 # ================================================================
 def reset_model(self):
 """重置任务：采样新目标位置"""
 self._step_count = 0

 # 在工作空间内随机采样目标位置
 self._goal_pos = self.np_random.uniform(
 low=self.GOAL_WORKSPACE[:3],
 high=self.GOAL_WORKSPACE[3:],
 )

 self.set_joint_qpos(self.init_qpos)
 self.set_joint_qvel(self.init_qvel)
 self.mj_forward()
 self._sync_view()

 return self._get_obs(), {"goal": self._goal_pos.copy()}


# ============================================================
# 注册环境
# ============================================================
ENV_ID = "ReachTargetTask-v1"
if ENV_ID not in gym.envs.registry:
 gym.register(
 id=ENV_ID,
 entry_point="reach_target_task:ReachTargetTask",
 kwargs={
 'frame_skip': 20,
 'orcagym_addr': "localhost:50051",
 'agent_names': ["robot_0"],
 'time_step': 0.001,
 },
 max_episode_steps=300,
 )


# ============================================================
# 手动控制演示（非 RL）
# ============================================================
def manual_control_demo():
 """
 用手写的"朝向目标"策略来控制机械臂。

 策略逻辑非常简单：
 1. 朝目标方向移动
 2. 到达后停止

 这展示了如何在不使用 RL 的情况下，用手动规则完成仿真任务。
 """
 print("=" * 60)
 print(" 手动控制演示：机械臂到达目标")
 print("=" * 60)

 env = gym.make(ENV_ID)

 obs, info = env.reset()
 goal = info["goal"]
 print(f"\n🎯 目标位置: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")

 total_reward = 0

 for step_idx in range(300):
 ee_pos = obs["ee_pos"]
 direction = goal - ee_pos
 dist = np.linalg.norm(direction)

 if dist < env.unwrapped.SUCCESS_THRESHOLD:
 print(f"\n✅ 到达目标！耗时 {step_idx} 步，误差 {dist*1000:.1f}mm")
 break

 # 简单策略：随机探索 + 保留好的方向
 action = np.random.randn(env.unwrapped.model.nu) * 0.02

 obs, reward, terminated, truncated, info = env.step(action)
 env.render()
 total_reward += reward

 if step_idx % 30 == 0:
 print(f" Step {step_idx:3d}: dist={dist*1000:5.1f}mm, "
 f"ee=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}], "
 f"reward={reward:+.3f}")

 if terminated:
 break

 if step_idx == 299:
 print(f"\n⏰ 未在 300 步内到达目标，最终误差 {dist*1000:.1f}mm")

 print(f"\n总奖励: {total_reward:.1f}")
 env.close()


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
 manual_control_demo()
```

---

## 代码结构回顾

```
ReachTargetTask
│
├── __init__()
│ ├── super().__init__() ← 连接仿真、初始化
│ ├── self._goal_pos ← 任务特有状态
│ ├── self._pd = PDController() ← 底层控制器
│ └── 定义动作空间和观测空间
│
├── reset_model()
│ ├── 采样新目标位置
│ └── 返回初始观测
│
├── step(action)
│ ├── target = current + action ← 解析动作
│ ├── ctrl = pd.compute(target, ...) ← PD → 力矩
│ ├── do_simulation(ctrl, ...) ← 物理仿真
│ ├── obs = _get_obs() ← 收集观测
│ └── _compute_reward(obs, action) ← 计算奖励
│
└── _get_obs()
 ├── self.data.qpos / qvel ← 关节状态
 ├── query_site_pos_and_quat() ← 末端位姿
 └── 计算到目标的距离
```

---

## 你学会了什么

回顾整个新手教程，你掌握了：

| 教程 | 知识点 | 在本任务中的体现 |
|------|--------|-----------------|
| Hello World | 环境概念、`step`/`reset` | `gym.register()`、`gym.make()` |
| 场景搭建 | Actor 摆放、资产添加 | （任务开始前搭建的场景） |
| 第一个环境 | 继承 `OrcaGymEulerEnv` | `class ReachTargetTask(OrcaGymEulerEnv)` |
| 读取状态 | 查询关节/Body/Site | `query_site_pos_and_quat()` |
| 控制关节 | qpos/qvel 操作 | `self.data.qpos[:nu] + action` |
| PD 控制器 | 目标角度→力矩 | `self._pd.compute()` |
| **本任务** | **组合一切** | **完整的 reach 任务** |

---

## 扩展方向

掌握了基础后，你可以尝试：

1. **加入相机观测** — 在 `_get_obs()` 中添加相机图像
2. **更智能的控制** — 用 IK 解算器替代简单探索
3. **加入物体操作** — 在场景中放一个方块，让机械臂推动它
4. **RL 训练** — 用 PPO 训练策略替代手动控制（见 [🧠 PPO 训练](ppo-training.md)）

---

> 🎉 **恭喜！**你已经从零开始，学会了 OrcaGym 的核心使用方式。现在去创造你自己的机器人任务吧！
