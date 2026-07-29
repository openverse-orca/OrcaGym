# 🏆 Complete Task — Building an RL Task from Scratch

This section integrates all the knowledge from the previous tutorials to build a complete reinforcement learning task from scratch:

> **Task**: Control a robot arm end-effector to reach a random target point (Reach Target)

---

## Task Design

### Task Description

- **Goal**: Control the robot arm so that the end-effector moves to a randomly sampled target position
- **Action**: Delta joint position control (each joint changes at most 0.05 radians per step)
- **Observation**: Joint positions, velocities, end-effector pose, target position
- **Reward**: Positive reward for reaching the target + distance penalty + energy penalty
- **Termination**: Target reached (`terminated`) or timeout (`truncated`)

### Reward Function Design

A good reward function is key to RL success. We use a **dense reward + success reward** combination:

```python
# 1. Distance reward (dense): given every step, guides toward the target
dist_reward = -distance_to_goal

# 2. Success reward (sparse): large bonus when the target is reached
success_reward = 100.0 if distance < threshold else 0.0

# 3. Energy penalty (regularization): penalizes excessive actions
action_penalty = -0.01 * np.sum(action ** 2)

# Total reward
reward = dist_reward + success_reward + action_penalty
```

---

## Complete Code

```python
"""
reach_target_env.py — Complete environment for the robot arm reach-target task

Can be used directly with RL frameworks such as Stable-Baselines3 and RLlib for training.
"""

import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


# ================================================================
# PD Controller
# ================================================================
class PDController:
    """Simple multi-joint PD controller."""

    def __init__(self, nu: int, kp: float = 150.0, kd: float = 15.0):
        self.nu = nu
        self.kp = np.full(nu, kp, dtype=np.float64)
        self.kd = np.full(nu, kd, dtype=np.float64)

    def compute(self, target_qpos, current_qpos, current_qvel):
        """Compute PD torques."""
        pos_error = target_qpos - current_qpos[:self.nu]
        vel_error = np.zeros(self.nu) - current_qvel[:self.nu]
        return (self.kp * pos_error + self.kd * vel_error).astype(np.float64)


# ================================================================
# Main Environment Class
# ================================================================
class ReachTargetEnv(OrcaGymEulerEnv):
    """
    Robot arm end-effector reach-target task.

    ## Action Space
    Box(low=-0.05, high=0.05, shape=(nu,))
    Joint angle delta per step (radians)

    ## Observation Space
    Dict(
        joint_pos: Box(shape=(nq,))   — joint positions
        joint_vel: Box(shape=(nv,))   — joint velocities
        ee_pos: Box(shape=(3,))       — end-effector position
        goal_pos: Box(shape=(3,))     — target position
        dist_to_goal: Box(shape=(1,)) — distance to target
    )

    ## Reward
    reward = -dist_to_goal       # distance penalty (dense)
           + success_bonus       # reach reward (sparse)
           - action_penalty      # action penalty (regularization)

    ## Termination Conditions
    - Target reached: dist_to_goal < 1 cm -> terminated = True
    - Timeout: auto-handled by TimeLimit wrapper -> truncated = True
    """

    # Task hyperparameters
    SUCCESS_THRESHOLD = 0.01      # distance threshold for reaching target (meters), 1 cm
    SUCCESS_BONUS = 100.0         # reward for reaching the target
    ACTION_PENALTY_COEF = 0.01    # action penalty coefficient
    MAX_DELTA_PER_STEP = 0.05     # max joint angle change per step (radians)
    PD_KP = 150.0                 # PD proportional gain
    PD_KD = 15.0                  # PD derivative gain

    # Target sampling workspace [x_min, y_min, z_min, x_max, y_max, z_max]
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

        # Action space
        self.action_space = spaces.Box(
            low=-self.MAX_DELTA_PER_STEP,
            high=self.MAX_DELTA_PER_STEP,
            shape=(self.model.nu,), dtype=np.float32,
        )

        # Observation space
        obs_sample = self._get_obs()
        self.observation_space = spaces.Dict({
            key: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32)
            for key, v in obs_sample.items()
        })

        print(f"ReachTargetEnv initialized: "
              f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}, "
              f"control frequency={1.0/self.dt:.1f}Hz")

    def _get_obs(self) -> dict:
        """Collect current state as observation."""
        ee_site = self.site("end_effector")
        sites = self.query_site_pos_and_mat([ee_site])
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
# Register Environment
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

## Using the Environment

### Interactive Testing

```python
import gymnasium as gym
import numpy as np

env = gym.make("ReachTarget-v0")

obs, info = env.reset()
print(f"Initial target position: {info['goal']}")

total_reward = 0.0
for step_idx in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    total_reward += reward

    if step_idx % 20 == 0:
        print(f"  Step {step_idx:3d}: "
              f"reward={reward:8.4f}, "
              f"dist={info['dist_to_goal']:.4f}")

    if terminated:
        print(f"  ✅ Target reached! At step {step_idx}")
        break

    if truncated:
        print(f"  ⏰ Timeout truncated, final distance: {info['dist_to_goal']:.4f}")
        break

print(f"\nEpisode total reward: {total_reward:.2f}")
env.close()
```

### Integration with Stable-Baselines3

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import reach_target_env  # triggers gym.register()

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

print("🚀 Starting training...")
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback,
    progress_bar=True,
)

model.save("reach_target_ppo")
print("✅ Training complete!")
env.close()
eval_env.close()
```

---

## Tuning Guide

### Reward Function Tuning

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| Policy does nothing (zero actions) | Distance penalty too large | Reduce distance penalty coefficient, increase success reward |
| Policy oscillates violently | Action penalty too small | Increase `ACTION_PENALTY_COEF` |
| Policy never reaches the target | Success reward not large enough | Increase `SUCCESS_BONUS` |
| Training is unstable | Reward scale too large | Normalize the reward |

### Recommended Reward Scales

```python
dist_reward in [-1, 0]       # distance penalty (dominant)
success_reward in [10, 100]  # success reward (one-time)
action_penalty in [-0.1, 0]  # action penalty (small)
```

---

## Review and Summary

Congratulations on completing all the content of the beginners' tutorial! Let's review what you have learned:

| Tutorial | Core Knowledge |
|----------|---------------|
| [🚀 Hello World](hello-world.md) | Minimal simulation loop, `step()`/`reset()` |
| [🏗️ Your First Environment](your-first-env.md) | Inheriting `OrcaGymEulerEnv`, implementing `_get_obs()` |
| [👁️ Observation & Action](observation-action.md) | Observation space design, action space types |
| [🎮 Simple Controller](simple-controller.md) | PD controller principles and implementation |
| [🏆 Complete Task](complete-task.md) | Reward function design, termination conditions, SB3 integration |

---

> 🎉 **Congratulations!** You can now build custom RL environments with OrcaGym. Now go and create your own robot tasks!
