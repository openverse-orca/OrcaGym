# 👁️ 观测与动作 — 设计观测空间和动作空间

在本篇教程中，你将深入理解如何为你的环境设计**观测空间**和**动作空间**。这是构建有效 RL 环境的关键步骤。

---

## 观测空间 (Observation Space)

观测空间定义了 RL 策略在每个时间步能"看到"什么信息。

### 设计观测的原则

好的观测设计遵循以下原则：

1. **充分性**：观测中应包含完成任务所需的所有信息（关节位置、目标位置等）
2. **紧凑性**：只包含必要信息，多余的维度会增加策略学习难度
3. **尺度合理性**：不同特征的数值范围应尽量在相近的数量级
4. **可获取性**：所有观测值必须能从仿真状态中可靠地计算出来

### 示例：为机械臂环境设计观测

#### 最简观测：只有关节状态

```python
def _get_obs(self) -> dict:
 """最基础的观测：关节位置 + 速度"""
 return {
 "joint_pos": self.data.qpos.copy(), # (nq,) 广义位置
 "joint_vel": self.data.qvel.copy(), # (nv,) 广义速度
 }
```

#### 进阶观测：加入末端执行器信息

```python
def _get_obs(self) -> dict:
 """包含末端执行器位姿的观测"""

 # 1. 基础关节状态
 joint_pos = self.data.qpos.copy()
 joint_vel = self.data.qvel.copy()

 # 2. 末端执行器 (End-Effector) 位姿
 ee_site_name = self.site("end_effector") # 自动加上 agent 前缀
 ee_site = self.query_site_pos_and_quat([ee_site_name])

 # ee_site 返回格式: {site_name: {"xpos": array([x,y,z]), "xquat": array([w,x,y,z])}}
 ee_pos = ee_site[ee_site_name]["xpos"] # 末端位置 (3,)
 ee_quat = ee_site[ee_site_name]["xquat"] # 末端姿态四元数 (4,)

 # 3. 末端执行器速度（用于捕捉运动趋势）
 ee_linear_vel, ee_angular_vel = self.query_site_xvalp_xvalr([ee_site_name])
 ee_vel = ee_linear_vel[ee_site_name] # 线速度 (3,)
 ee_angvel = ee_angular_vel[ee_site_name] # 角速度 (3,)

 return {
 "joint_pos": joint_pos, # (nq,)
 "joint_vel": joint_vel, # (nv,)
 "ee_pos": ee_pos, # (3,)
 "ee_quat": ee_quat, # (4,)
 "ee_vel": ee_vel, # (3,)
 "ee_angvel": ee_angvel, # (3,)
 }
```

#### 完整观测：加入目标和传感器

```python
def _get_obs(self) -> dict:
 """完整的任务观测：包含目标位置和传感器数据"""

 # 关节状态
 joint_pos = self.data.qpos.copy()
 joint_vel = self.data.qvel.copy()

 # 末端执行器在世界坐标系中的位姿
 ee_site_name = self.site("end_effector")
 ee_site = self.query_site_pos_and_quat([ee_site_name])

 # 末端执行器相对于基座的位姿（对固定基座机械臂很有用）
 base_name = self.body("base_link")
 ee_pos_B_dict = self.query_site_pos_and_quat_B(
 [ee_site_name], [base_name]
 )

 # 目标位置（在 reset 中随机采样）
 goal_pos = self._goal_pos # (3,)，在 reset_model 中设置

 # 到目标的距离（给策略一个直接的误差信号）
 ee_pos = ee_site[ee_site_name]["xpos"]
 distance_to_goal = np.linalg.norm(ee_pos - goal_pos)

 obs = {
 "joint_pos": joint_pos,
 "joint_vel": joint_vel,
 "ee_pos": ee_pos,
 "ee_pos_base_frame": ee_pos_B_dict[ee_site_name]["xpos"],
 "goal_pos": goal_pos,
 }

 return obs
```

### 观测空间的类型

| 观测类型 | Space 类型 | 示例 |
|----------|-----------|------|
| 字典观测（推荐） | `spaces.Dict` | `{"joint_pos": Box, "joint_vel": Box}` |
| 单一数组 | `spaces.Box` | `Box(low=-inf, high=inf, shape=(13,))` |

!!! tip "推荐使用字典观测"
 字典观测比单一数组更易于：
 - 调试（可以按名称查看各分量）
 - 扩展（添加新观测不改变已有维度）
 - 归一化（可以对不同键使用不同的归一化策略）

---

## 动作空间 (Action Space)

动作空间定义了策略能输出什么动作来控制机器人。

### 动作空间的三种常见设计

#### 1. 力矩控制（Torque Control）

最底层的方式 —— 直接控制每个关节的力矩：

```python
def _set_action_space(self):
 """力矩控制：动作是各关节的目标力矩"""
 ctrlrange = self.model.get_actuator_ctrlrange() # (nu, 2)
 self.action_space = spaces.Box(
 low=ctrlrange[:, 0], high=ctrlrange[:, 1], dtype=np.float32
 )
```

#### 2. 位置控制（Position Control）

动作作为目标关节位置，由 PD 控制器转换为力矩：

```python
def _set_action_space(self):
 """位置控制：动作是目标关节角度"""
 joint_ranges = np.array([
 [-3.14, 3.14], # joint_0: ±180°
 [-1.57, 1.57], # joint_1: ±90°
 # ... 根据你的机器人设置
 ])
 self.action_space = spaces.Box(
 low=joint_ranges[:, 0], high=joint_ranges[:, 1], dtype=np.float32
 )
```

在 `step()` 中使用 PD 控制器（详见 [🎮 简单控制器](simple-controller.md)）：

```python
def step(self, action): # action 是目标关节角度
 ctrl = self._pd.compute(
 target_qpos=action,
 current_qpos=self.data.qpos,
 current_qvel=self.data.qvel,
 )
 self.do_simulation(ctrl, self.frame_skip)
```

#### 3. 增量控制（Delta Control）

动作是相对于当前位置的偏移：

```python
def step(self, action): # action ∈ [-0.1, 0.1]，表示关节角度变化
 max_delta = 0.1 # 每步最多变化 0.1 弧度
 delta = np.clip(action, -max_delta, max_delta)

 # 目标位置 = 当前位置 + 增量
 target_qpos = self.data.qpos[:self.model.nu] + delta

 # 用 PD 控制器追踪目标位置
 ctrl = self._pd.compute(target_qpos, self.data.qpos, self.data.qvel)
 self.do_simulation(ctrl, self.frame_skip)
```

### 动作空间设计对比

| 控制方式 | 优点 | 缺点 | 适用场景 |
|----------|------|------|----------|
| 力矩控制 | 最灵活、最高带宽 | 难以训练、需要大量探索 | 精细操作、高动态任务 |
| 位置控制 | 易于训练、行为平滑 | 响应带宽受限 | 拾放、装配等准静态任务 |
| 增量控制 | 平滑、安全 | 速度受限 | 需要平滑轨迹的任务 |

> **新手建议**：从**位置控制**或**增量控制**开始。力矩控制虽然灵活，但对 RL 策略来说探索难度大得多。

---

## 完整示例：整合观测和动作

下面是一个完整的环境，展示了观测和动作空间的综合设计：

```python
"""
reach_env.py — 一个机械臂到达任务的环境

观测：关节位置、速度、末端位姿、目标位置
动作：增量关节位置控制
"""

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class ReachEnv(OrcaGymEulerEnv):
 """机械臂末端到达指定目标点的任务"""

 def __init__(self, frame_skip, orcagym_addr, agent_names, time_step, **kwargs):
 super().__init__(
 frame_skip=frame_skip,
 orcagym_addr=orcagym_addr,
 agent_names=agent_names,
 time_step=time_step,
 **kwargs,
 )

 # 任务相关：目标位置
 self._goal_pos = np.zeros(3)

 # 动作空间：增量位置控制 [-0.05, 0.05] 弧度/步
 self.action_space = spaces.Box(
 low=-0.05, high=0.05, shape=(self.model.nu,), dtype=np.float32
 )

 obs_sample = self._get_obs()
 self.observation_space = spaces.Dict({
 key: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32)
 for key, v in obs_sample.items()
 })

 def _get_obs(self):
 """收集丰富的观测信息"""
 ee_site = self.site("end_effector")
 sites = self.query_site_pos_and_quat([ee_site])

 ee_pos = sites[ee_site]["xpos"]
 ee_quat = sites[ee_site]["xquat"]

 dist = np.linalg.norm(ee_pos - self._goal_pos)

 return {
 "joint_pos": self.data.qpos.copy().astype(np.float32),
 "joint_vel": self.data.qvel.copy().astype(np.float32),
 "ee_pos": ee_pos.astype(np.float32),
 "goal_pos": self._goal_pos.astype(np.float32),
 "dist_to_goal": np.array([dist], dtype=np.float32),
 }

 def step(self, action):
 # 增量控制：当前 qpos + 动作偏移
 target_qpos = self.data.qpos[:self.model.nu] + action

 # 用简单 PD 计算力矩
 pos_error = target_qpos - self.data.qpos[:self.model.nu]
 vel_error = -self.data.qvel[:self.model.nv]
 ctrl = pos_error * 100.0 + vel_error * 10.0

 # 执行仿真
 self.do_simulation(ctrl, self.frame_skip)

 obs = self._get_obs()
 dist = obs["dist_to_goal"].item()
 reward = -dist # 越近奖励越大
 terminated = dist < 0.01 # 距离小于 1cm 视为成功
 truncated = False

 return obs, reward, terminated, truncated, {"distance": dist}

 def reset_model(self):
 """重置机器人并随机采样新目标"""
 self._goal_pos = self.np_random.uniform(
 low=[0.2, -0.3, 0.1],
 high=[0.6, 0.3, 0.5],
 )

 self.set_joint_qpos(self.init_qpos)
 self.set_joint_qvel(self.init_qvel)
 self.mj_forward()
 self._sync_view()

 return self._get_obs(), {"goal": self._goal_pos}
```

---

## 观测与动作的调试技巧

### 1. 检查观测的有效性

```python
def _validate_obs(obs):
 """确保观测中没有 NaN 或 Inf"""
 for key, val in obs.items():
 if np.any(np.isnan(val)):
 print(f"⚠️ NaN in obs['{key}']")
 if np.any(np.isinf(val)):
 print(f"⚠️ Inf in obs['{key}']")
 print(f" obs['{key}']: shape={val.shape}, "
 f"range=[{val.min():.3f}, {val.max():.3f}]")
```

### 2. 随机动作探索

在开发阶段，用随机动作测试环境是否稳定：

```python
env = ReachEnv(...)
obs, _ = env.reset()
for i in range(200):
 action = env.action_space.sample() # 随机动作
 obs, reward, terminated, truncated, _ = env.step(action)
 env.render()
 if terminated or truncated:
 obs, _ = env.reset()
```

---

## 下一步

你已经掌握了观测和动作空间的设计。现在学习如何编写控制器： [🎮 简单控制器](simple-controller.md)。
