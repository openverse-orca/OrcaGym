# 🧠 PPO Training — Training an Inverted Pendulum with Reinforcement Learning

This section shows you how to train an inverted pendulum to stay upright in an OrcaGym environment using the Stable Baselines3 PPO algorithm.

> See [OrcaPlayground examples/euler/03_rl_ppo/](https://github.com/OrcaGym/OrcaPlayground) for complete runnable code.

---

## Prerequisites

- Completed [🏗️ Your First Environment](your-first-env.md)
- stable-baselines3 installed: `pip install stable-baselines3`

---

## Environment Design

We train a **single-hinge inverted pendulum** (Gymnasium Pendulum-v1 style):

- **Scene**: one hinge joint + one rod (local MuJoCo XML)
- **Observation**: `[cos(theta), sin(theta), theta_dot]` — 3-dimensional Box
- **Action**: `[torque]` — 1-dimensional Box, range `[-1, 1]`
- **Reward**: `-(theta^2 + 0.1*theta_dot^2 + 0.001*action^2)` — the closer to 0, the better
- **Termination**: none (continuous control task), truncate after 200 steps

### Complete Environment Code

```python
"""simple_env.py — Single-hinge inverted pendulum environment."""

import os
from typing import Any

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


# Scene XML path (replace with your path)
_SCENE_XML = os.path.join(os.path.dirname(__file__), "simple_pendulum.xml")


class SimpleEulerEnv(OrcaGymEulerEnv):
    """Single-hinge inverted pendulum environment. theta=0 is the upright position (target)."""

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}
    MAX_EPISODE_STEPS = 200

    def __init__(
        self,
        orcagym_addr: str = "localhost:50051",
        time_step: float = 0.002,
        frame_skip: int = 5,
        skip_grpc_load: bool = True,  # use offline mode for training
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

        # Action space: 1-dimensional, [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        # Observation space: 3-dimensional, [cos, sin, theta_dot]
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
        # Pendulum-v1 standard cost
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
        # cos/sin encoding avoids the 2*pi periodicity problem
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
```

### Reward Function Design Key Points

| Term | Meaning | Coefficient |
|------|---------|-------------|
| `theta^2` | Penalty for deviating from upright | 1.0 |
| `0.1 * theta_dot^2` | Velocity penalty (encourages smoothness) | 0.1 |
| `0.001 * action^2` | Action magnitude penalty (energy saving) | 0.001 |

- When `theta=0`, `cos(theta)=1, sin(theta)=0`, reward is maximum (approaching 0)
- Under random actions, the reward is a large negative number; after training it should gradually approach 0

---

## PPO Training Code

```python
"""train_ppo.py — SB3 PPO training for inverted pendulum."""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from simple_env import SimpleEulerEnv

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def train(total_timesteps: int = 100000, device: str = "cuda"):
    # 1. Create environment (offline mode, most efficient)
    env = SimpleEulerEnv(
        orcagym_addr="localhost:50051",
        time_step=0.002,
        frame_skip=5,
        skip_grpc_load=True,  # offline training, no Studio needed
    )
    env = Monitor(env)  # wrap to record episode reward
    print(f"Environment: obs={env.observation_space.shape}, action={env.action_space.shape}")

    # 2. Create PPO model
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

    # 3. Train
    model.learn(total_timesteps=total_timesteps)

    # 4. Save model
    os.makedirs(_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(_MODEL_DIR, "ppo_pendulum.zip")
    model.save(model_path)
    print(f"Model saved: {model_path}")

    env.close()
    return model_path


def evaluate(model_path: str, episodes: int = 5):
    """Evaluate: default to online human mode for visualization."""
    # Use online mode for evaluation to enable rendering and observation.
    env = SimpleEulerEnv(
        orcagym_addr="localhost:50051",
        time_step=0.002,
        frame_skip=5,
        skip_grpc_load=False,  # online evaluation, connect to Studio for visualization
    )
    model = PPO.load(model_path, env=env)

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            env.render()  # Studio viewport real-time display
            if terminated or truncated:
                break
        print(f"  episode {ep + 1}: reward={ep_reward:.4f}, steps={step + 1}")

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

Run:

```bash
# Train (100k steps, ~2-3 minutes)
python train_ppo.py --total-timesteps 100000

# Quick verification (20k steps, ~30 seconds)
python train_ppo.py --total-timesteps 20000

# Evaluate (requires OrcaStudio running with the pendulum scene loaded)
python train_ppo.py --eval
```

---

## Interpreting Training Logs

During training, reward should **gradually rise from large negative numbers toward 0**:

```
| rollout/           |             |
| ep_len_mean        | 200         | <- episode length (fixed at 200)
| ep_rew_mean        | -50 -> -5   | <- reward gradually rises from -50 to -5
| time/              |             |
| fps                | ~2000       | <- offline mode FPS is very high
```

- If reward stays below -100 -> learning rate too high or environment issue
- If reward quickly reaches around -1 -> training successful, the pendulum can balance upright

---

## Key Techniques

### Observation Encoding

Use `[cos(theta), sin(theta)]` rather than raw `theta`, because:
- `theta=0` and `theta=2*pi` describe the same physical pose
- With raw angle values, the network would need to learn the 2*pi periodicity -> very difficult
- cos/sin encoding naturally handles this issue

### Offline vs. Online Training

| Mode | `skip_grpc_load` | FPS | Purpose |
|------|------------------|-----|---------|
| Offline training | `True` | ~2000+ | Training (most efficient) |
| Online evaluation | `False` | ~50 (RTF=1.0) | Visual evaluation |

> Always use offline mode for training. Only connect to Studio when evaluating to see the results.

### `Monitor` Wrapper

```python
from stable_baselines3.common.monitor import Monitor
env = Monitor(env)
```

`Monitor` automatically records episode reward and length, making it easy to view training curves.

---

## FAQ

### `UserWarning: You are trying to run PPO on the GPU`

SB3 detects that a GPU is available and defaults to GPU, but for MLP policies CPU is recommended. **Can be ignored** — GPU training is actually faster.

### Reward Does Not Increase During Training

Check:
1. Reward function sign: should be negative cost (closer to 0 is better), not positive reward
2. Whether the initial pose randomization range in `reset_model` is too large
3. Whether the learning rate is appropriate (3e-4 recommended)

---

## Next Step

You have trained a controller. Next, learn how to **query more simulation state**: [📡 State Query API](../robot_control/state-queries-api.md).
