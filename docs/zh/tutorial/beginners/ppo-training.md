# 🧠 PPO 训练 — 用强化学习训练倒立摆

本节教你如何基于 Stable Baselines3 PPO 算法，在 OrcaGym 环境中训练一个倒立摆保持直立。

> 完整可运行代码见 [OrcaPlayground examples/euler/03_rl_ppo/](https://github.com/OrcaGym/OrcaPlayground)。

---

## 前提

- 已完成 [🏗️ 第一个环境](your-first-env.md)
- 安装了 stable-baselines3：`pip install stable-baselines3`

---

## 环境设计

我们训练一个**单铰链倒立摆**（Gymnasium Pendulum-v1 风格）：

- **场景**：一个铰链关节 + 一根摆杆（本地 MuJoCo XML）
- **观测**：`[cos(theta), sin(theta), theta_dot]` — 3 维 Box
- **动作**：`[torque]` — 1 维 Box，范围 `[-1, 1]`
- **奖励**：`-(theta² + 0.1*theta_dot² + 0.001*action²)` — 越接近 0 越好
- **终止**：无（连续控制任务），200 步后 truncate

### 完整环境代码

```python
"""simple_env.py — 单铰链倒立摆环境"""

import os
from typing import Any

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


# 场景 XML 路径（请替换为你的路径）
_SCENE_XML = os.path.join(os.path.dirname(__file__), "simple_pendulum.xml")


class SimpleEulerEnv(OrcaGymEulerEnv):
 """单铰链倒立摆环境。theta=0 为直立位置（目标）。"""

 metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}
 MAX_EPISODE_STEPS = 200

 def __init__(
 self,
 orcagym_addr: str = "localhost:50051",
 time_step: float = 0.002,
 frame_skip: int = 5,
 skip_grpc_load: bool = True, # 训练用离线模式
 **kwargs,
 ):
 super().__init__(
 frame_skip=frame_skip,
 orcagym_addr=orcagym_addr,
 agent_names=["agent0"],
 time_step=time_step,
 model_xml_path=_SCENE_XML,
 skip_grpc_load=skip_grpc_load,
 **kwargs,
 )
 self._step_count = 0

 # 动作空间：1 维，[-1, 1]
 self.action_space = spaces.Box(
 low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
 )
 # 观测空间：3 维，[cos, sin, theta_dot]
 obs = self._get_obs()
 self.observation_space = spaces.Box(
 low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
 )

 def step(self, action: np.ndarray):
 action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
 self.do_simulation(action, self.frame_skip)
 obs = self._get_obs()
 theta = float(self.data.qpos[0])
 theta_dot = float(self.data.qvel[0])
 # Pendulum-v1 标准 cost
 reward = float(-(theta**2 + 0.1 * theta_dot**2 + 0.001 * float(action[0])**2))
 terminated = False
 self._step_count += 1
 truncated = self._step_count >= self.MAX_EPISODE_STEPS
 info: dict[str, Any] = {"time": float(self.data.time)}
 return obs, reward, terminated, truncated, info

 def reset_model(self):
 qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
 qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
 self.set_joint_qpos(qpos)
 self.set_joint_qvel(qvel)
 self.mj_forward()
 self._sync_view()
 self._step_count = 0
 return self._get_obs(), {}

 def _get_obs(self) -> np.ndarray:
 theta = float(self.data.qpos[0])
 theta_dot = float(self.data.qvel[0])
 # cos/sin 编码避免 2π 周期性问题
 return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
```

### 奖励函数设计要点

| 项 | 含义 | 系数 |
|----|------|------|
| `theta²` | 偏离直立位置惩罚 | 1.0 |
| `0.1 * theta_dot²` | 速度惩罚（鼓励平稳） | 0.1 |
| `0.001 * action²` | 动作幅度惩罚（节能） | 0.001 |

- `theta=0` 时 `cos(theta)=1, sin(theta)=0`，奖励最大（趋近 0）
- 随机动作下 reward 为大负数，训练后应逐步趋近 0

---

## PPO 训练代码

```python
"""train_ppo.py — SB3 PPO 训练倒立摆"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from simple_env import SimpleEulerEnv

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def train(total_timesteps: int = 100000, device: str = "cuda"):
 # 1. 创建环境（离线模式，最高效）
 env = SimpleEulerEnv(
 orcagym_addr="localhost:50051",
 time_step=0.002,
 frame_skip=5,
 skip_grpc_load=True, # 离线训练，不需要 Studio
 )
 env = Monitor(env) # 包装以记录 episode reward
 print(f"环境: obs={env.observation_space.shape}, action={env.action_space.shape}")

 # 2. 创建 PPO 模型
 model = PPO(
 policy="MlpPolicy",
 env=env,
 learning_rate=3e-4,
 n_steps=2048,
 batch_size=64,
 gamma=0.99,
 gae_lambda=0.95,
 clip_range=0.2,
 ent_coef=0.0,
 vf_coef=0.5,
 max_grad_norm=0.5,
 device=device,
 verbose=1,
 )

 # 3. 训练
 model.learn(total_timesteps=total_timesteps)

 # 4. 保存模型
 os.makedirs(_MODEL_DIR, exist_ok=True)
 model_path = os.path.join(_MODEL_DIR, "ppo_pendulum.zip")
 model.save(model_path)
 print(f"模型已保存: {model_path}")

 env.close()
 return model_path


def evaluate(model_path: str, episodes: int = 5):
 """评估：默认用 online human 模式可视化。"""
 # 评估用在线模式以便渲染观察
 env = SimpleEulerEnv(
 orcagym_addr="localhost:50051",
 time_step=0.002,
 frame_skip=5,
 skip_grpc_load=False, # 在线评估，连接 Studio 可视化
 )
 model = PPO.load(model_path, env=env)

 for ep in range(episodes):
 obs, _ = env.reset()
 ep_reward = 0.0
 for step in range(200):
 action, _ = model.predict(obs, deterministic=True)
 obs, reward, terminated, truncated, _ = env.step(action)
 ep_reward += reward
 env.render() # Studio 视口实时显示
 if terminated or truncated:
 break
 print(f" episode {ep + 1}: reward={ep_reward:.4f}, steps={step + 1}")

 env.close()


if __name__ == "__main__":
 parser = argparse.ArgumentParser()
 parser.add_argument("--total-timesteps", type=int, default=100000)
 parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
 parser.add_argument("--eval", action="store_true")
 parser.add_argument("--model-path", default=None)
 args = parser.parse_args()

 if args.eval:
 model_path = args.model_path or os.path.join(_MODEL_DIR, "ppo_pendulum.zip")
 evaluate(model_path)
 else:
 train(args.total_timesteps, args.device)
```

运行：

```bash
# 训练（100k 步，约 2-3 分钟）
python train_ppo.py --total-timesteps 100000

# 快速验证（20k 步，约 30 秒）
python train_ppo.py --total-timesteps 20000

# 评估（需要 OrcaStudio 启动并加载 pendulum 场景）
python train_ppo.py --eval --eval-episodes 5
```

---

## 训练日志解读

训练过程中 reward 应**从大负数逐步趋近 0**：

```
| rollout/ | |
| ep_len_mean | 200 | ← episode 长度（固定 200）
| ep_rew_mean | -50 → -5 | ← reward 从 -50 逐渐上升到 -5
| time/ | |
| fps | ~2000 | ← 离线模式 FPS 很高
```

- 如果 reward 一直在 -100 以下 → 学习率太高或环境有问题
- 如果 reward 快速到达 -1 左右 → 训练成功，摆杆能稳定直立

---

## 关键技巧

### 观测编码

用 `[cos(theta), sin(theta)]` 而非直接 `theta`，因为：
- `theta=0` 和 `theta=2π` 是同一个物理姿态
- 直接用角度值，网络需要学习 2π 周期性 → 很难
- cos/sin 编码天然处理了这个问题

### 离线 vs 在线训练

| 模式 | `skip_grpc_load` | FPS | 用途 |
|------|------------------|-----|------|
| 离线训练 | `True` | ~2000+ | 训练（最高效） |
| 在线评估 | `False` | ~50（RTF=1.0） | 可视化评估 |

> 训练始终用离线模式。评估时才连接 Studio 看效果。

### `Monitor` 包装

```python
from stable_baselines3.common.monitor import Monitor
env = Monitor(env)
```

`Monitor` 会自动记录 episode reward 和长度，方便查看训练曲线。

---

## 常见问题

### `UserWarning: You are trying to run PPO on the GPU`

SB3 检测到 GPU 可用，默认用 GPU 但对 MLP 策略推荐 CPU。**可忽略**，GPU 训练实际上更快。

### 训练时 reward 不上升

检查：
1. 奖励函数符号：应该是负 cost（越接近 0 越好），不是正 reward
2. `reset_model` 中初始姿态随机范围是否太大
3. 学习率是否合适（推荐 3e-4）

---

## 下一步

训练好了控制器，接下来学习如何**查询更多仿真状态**：[📡 状态查询 API](../robot_control/state-queries-api.md)。
