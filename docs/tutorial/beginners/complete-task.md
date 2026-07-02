# 🏆 完整任务 — 从零构建一个 RL 任务

本篇将整合前面学到的所有知识，从零构建一个完整的强化学习任务：

> **任务**：控制机械臂末端执行器到达随机目标点（Reach Target）

---

## 任务设计

### 任务描述

- **目标**：控制机械臂使末端执行器（end-effector）移动到随机采样的目标位置
- **动作**：增量关节位置控制（每个关节每步最多变化 0.05 弧度）
- **观测**：关节位置、速度、末端位姿、目标位置
- **奖励**：到达目标给正奖励 + 距离惩罚 + 能耗惩罚
- **终止**：到达目标（`terminated`）或超时（`truncated`）

### 奖励函数设计

好的奖励函数是 RL 成功的关键。我们采用**密集奖励 + 成功奖励**的组合：

```python
# 1. 距离奖励（密集）：每步都给予，引导靠近目标
dist_reward = -distance_to_goal

# 2. 成功奖励（稀疏）：到达目标时给大额奖励
success_reward = 100.0 if distance < threshold else 0.0

# 3. 能耗惩罚（正则化）：惩罚过大的动作
action_penalty = -0.01 * np.sum(action ** 2)

# 总奖励
reward = dist_reward + success_reward + action_penalty
```

---

## 完整代码

```python
"""
reach_target_env.py — 机械臂到达目标任务的完整环境

可以直接用于 Stable-Baselines3、RLlib 等 RL 框架训练。
"""

import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


# ================================================================
# PD 控制器
# ================================================================
class PDController:
 """简单的多关节 PD 控制器"""

 def __init__(self, nu: int, kp: float = 150.0, kd: float = 15.0):
 self.nu = nu
 self.kp = np.full(nu, kp, dtype=np.float64)
 self.kd = np.full(nu, kd, dtype=np.float64)

 def compute(self, target_qpos, current_qpos, current_qvel):
 """计算 PD 力矩"""
 pos_error = target_qpos - current_qpos[:self.nu]
 vel_error = np.zeros(self.nu) - current_qvel[:self.nu]
 return (self.kp * pos_error + self.kd * vel_error).astype(np.float64)


# ================================================================
# 主环境类
# ================================================================
class ReachTargetEnv(OrcaGymEulerEnv):
 """
 机械臂末端到达目标点任务。

 ## 动作空间
 Box(low=-0.05, high=0.05, shape=(nu,))
 每步的关节角度增量（弧度）

 ## 观测空间
 Dict(
 joint_pos: Box(shape=(nq,)) — 关节位置
 joint_vel: Box(shape=(nv,)) — 关节速度
 ee_pos: Box(shape=(3,)) — 末端执行器位置
 goal_pos: Box(shape=(3,)) — 目标位置
 dist_to_goal: Box(shape=(1,)) — 到目标的距离
 )

 ## 奖励
 reward = -dist_to_goal # 距离惩罚（密集）
 + success_bonus # 到达奖励（稀疏）
 - action_penalty # 动作惩罚（正则化）

 ## 终止条件
 - 到达目标：dist_to_goal < 1cm → terminated = True
 - 超时：由 TimeLimit wrapper 自动处理 → truncated = True
 """

 # 任务超参数
 SUCCESS_THRESHOLD = 0.01 # 到达目标的距离阈值（米），1cm
 SUCCESS_BONUS = 100.0 # 到达目标的奖励
 ACTION_PENALTY_COEF = 0.01 # 动作惩罚系数
 MAX_DELTA_PER_STEP = 0.05 # 每步最大关节角度变化（弧度）
 PD_KP = 150.0 # PD 比例增益
 PD_KD = 15.0 # PD 微分增益

 # 目标采样空间 [x_min, y_min, z_min, x_max, y_max, z_max]
 GOAL_WORKSPACE = np.array([0.2, -0.3, 0.1, 0.6, 0.3, 0.5])

 def __init__(
 self,
 frame_skip: int,
 orcagym_addr: str,
 agent_names: list[str],
 time_step: float,
 max_episode_steps: Optional[int] = 200,
 **kwargs,
 ):
 super().__init__(
 frame_skip=frame_skip,
 orcagym_addr=orcagym_addr,
 agent_names=agent_names,
 time_step=time_step,
 **kwargs,
 )

 self._goal_pos = np.zeros(3)
 self._step_count = 0
 self._max_episode_steps = max_episode_steps

 self._pd = PDController(
 nu=self.model.nu,
 kp=self.PD_KP,
 kd=self.PD_KD,
 )

 # 动作空间
 self.action_space = spaces.Box(
 low=-self.MAX_DELTA_PER_STEP,
 high=self.MAX_DELTA_PER_STEP,
 shape=(self.model.nu,), dtype=np.float32,
 )

 # 观测空间
 obs_sample = self._get_obs()
 self.observation_space = spaces.Dict({
 key: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32)
 for key, v in obs_sample.items()
 })

 print(f"ReachTargetEnv 初始化完成: "
 f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}, "
 f"控制频率={1.0/self.dt:.1f}Hz")

 def _get_obs(self) -> dict:
 """收集当前状态作为观测"""
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

 def step(self, action: np.ndarray):
 self._step_count += 1

 action = np.clip(action, -self.MAX_DELTA_PER_STEP, self.MAX_DELTA_PER_STEP)

 current_qpos = self.data.qpos[:self.model.nu].copy()
 target_qpos = current_qpos + action

 ctrl = self._pd.compute(
 target_qpos=target_qpos,
 current_qpos=self.data.qpos,
 current_qvel=self.data.qvel,
 )

 self.do_simulation(ctrl, self.frame_skip)

 obs = self._get_obs()
 reward, terminated, info = self._compute_reward(obs, action)
 truncated = self._step_count >= self._max_episode_steps

 info["step"] = self._step_count
 info["ctrl_norm"] = float(np.linalg.norm(ctrl))

 return obs, reward, terminated, truncated, info

 def _compute_reward(self, obs: dict, action: np.ndarray):
 dist = obs["dist_to_goal"].item()

 dist_reward = -dist
 terminated = dist < self.SUCCESS_THRESHOLD
 success_reward = self.SUCCESS_BONUS if terminated else 0.0
 action_penalty = -self.ACTION_PENALTY_COEF * np.sum(action ** 2)

 reward = dist_reward + success_reward + action_penalty

 info = {
 "dist_to_goal": dist,
 "dist_reward": dist_reward,
 "success_reward": success_reward,
 "action_penalty": action_penalty,
 }

 return reward, terminated, info

 def reset_model(self) -> tuple:
 ws = self.GOAL_WORKSPACE
 self._goal_pos = self.np_random.uniform(
 low=ws[:3],
 high=ws[3:],
 )
 self._step_count = 0

 self.set_joint_qpos(self.init_qpos)
 self.set_joint_qvel(self.init_qvel)
 self.mj_forward()
 self._sync_view()

 obs = self._get_obs()
 info = {"goal": self._goal_pos.copy()}

 return obs, info


# ================================================================
# 注册环境
# ================================================================
ENV_ID = "ReachTarget-v0"

if ENV_ID not in gym.envs.registry:
 gym.register(
 id=ENV_ID,
 entry_point="reach_target_env:ReachTargetEnv",
 kwargs={
 'frame_skip': 20,
 'orcagym_addr': "localhost:50051",
 'agent_names': ["robot_0"],
 'time_step': 0.001,
 'max_episode_steps': 200,
 },
 max_episode_steps=200,
 )
```

---

## 使用环境

### 交互式测试

```python
import gymnasium as gym
import numpy as np

env = gym.make("ReachTarget-v0")

obs, info = env.reset()
print(f"初始目标位置: {info['goal']}")

total_reward = 0.0
for step_idx in range(200):
 action = env.action_space.sample()
 obs, reward, terminated, truncated, info = env.step(action)
 env.render()
 total_reward += reward

 if step_idx % 20 == 0:
 print(f" Step {step_idx:3d}: "
 f"reward={reward:8.4f}, "
 f"dist={info['dist_to_goal']:.4f}")

 if terminated:
 print(f" ✅ 到达目标！在第 {step_idx} 步")
 break

 if truncated:
 print(f" ⏰ 超时截断，最终距离: {info['dist_to_goal']:.4f}")
 break

print(f"\nEpisode 总奖励: {total_reward:.2f}")
env.close()
```

### 与 Stable-Baselines3 集成

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import reach_target_env # 触发 gym.register()

env = gym.make("ReachTarget-v0")
env = Monitor(env)

eval_env = gym.make("ReachTarget-v0")
eval_env = Monitor(eval_env)

model = PPO(
 policy="MultiInputPolicy",
 env=env,
 learning_rate=3e-4,
 n_steps=2048,
 batch_size=64,
 n_epochs=10,
 gamma=0.99,
 verbose=1,
 tensorboard_log="./logs/",
)

eval_callback = EvalCallback(
 eval_env,
 best_model_save_path="./logs/best_model/",
 log_path="./logs/eval/",
 eval_freq=5000,
)

print("🚀 开始训练...")
model.learn(
 total_timesteps=1_000_000,
 callback=eval_callback,
 progress_bar=True,
)

model.save("reach_target_ppo")
print("✅ 训练完成！")
env.close()
eval_env.close()
```

---

## 调优指南

### 奖励函数调优

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 策略不动（零动作） | 距离惩罚太大 | 减少距离惩罚系数，增大成功奖励 |
| 策略剧烈振荡 | 动作惩罚太小 | 增大 `ACTION_PENALTY_COEF` |
| 策略永远到不了目标 | 成功奖励不够大 | 增大 `SUCCESS_BONUS` |
| 训练不稳定 | 奖励尺度太大 | 对奖励做归一化 |

### 推荐的奖励尺度

```python
dist_reward ∈ [-1, 0] # 距离惩罚（占主导）
success_reward ∈ [10, 100] # 成功奖励（一次性）
action_penalty ∈ [-0.1, 0] # 动作惩罚（小量）
```

---

## 回顾与总结

恭喜你完成了新手教程的全部内容！让我们回顾一下你学到的知识：

| 教程 | 核心知识点 |
|------|-----------|
| [🚀 Hello World](hello-world.md) | 最简仿真循环、`step()`/`reset()` |
| [🏗️ 第一个环境](your-first-env.md) | 继承 `OrcaGymEulerEnv`、实现 `_get_obs()` |
| [👁️ 观测与动作](observation-action.md) | 观测空间设计、动作空间类型 |
| [🎮 简单控制器](simple-controller.md) | PD 控制器原理与实现 |
| [🏆 完整任务](complete-task.md) | 奖励函数设计、终止条件、SB3 集成 |

---

> 🎉 **恭喜！**你已经具备了使用 OrcaGym 构建自定义 RL 环境的能力。现在去创造你自己的机器人任务吧！
