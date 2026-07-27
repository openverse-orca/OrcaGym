# 🏆 Build a Task — Combining All Your Knowledge

This is the capstone of the beginners' tutorial. We will combine all the knowledge from the previous sections to build a complete task:

> **Make the robot arm end-effector reach a randomly specified target point.**

**No RL** — we will write the control logic manually (PD controller + simple trajectory planning).

---

## Task Definition

```
+-------------------------------------------------+
| Task: Reach Target                              |
|                                                 |
| Input (observation):                            |
|   - Joint angles                                |
|   - End-effector position                       |
|   - Target position                             |
|                                                 |
| Output (action):                                |
|   - Target angle for each joint                 |
|                                                 |
| Success condition:                              |
|   - End-effector to target distance < 5mm       |
|                                                 |
| Internal controller:                            |
|   - PD controller (target angles -> torque)     |
+-------------------------------------------------+
```

---

## Complete Code

```python
"""
reach_target_task.py — Complete robot arm reach task

Integrates all the knowledge from the previous tutorials:
- Environment subclassing
- State queries
- Joint control
- PD controller

Usage:
    python reach_target_task.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


# ============================================================
# PD Controller
# ============================================================
class PDController:
    """Multi-joint PD controller"""

    def __init__(self, nu: int, kp: float = 150.0, kd: float = 12.0):
        self.nu = nu
        self.kp = np.full(nu, kp, dtype=np.float64)
        self.kd = np.full(nu, kd, dtype=np.float64)

    def compute(self, target_qpos, current_qpos, current_qvel):
        pos_error = target_qpos - current_qpos[:self.nu]
        vel_error = np.zeros(self.nu) - current_qvel[:self.nu]
        return (self.kp * pos_error + self.kd * vel_error).astype(np.float64)


# ============================================================
# Main Task Environment
# ============================================================
class ReachTargetTask(OrcaGymEulerEnv):
    """
    Robot arm end-effector reach-target task.

    This environment demonstrates how to combine multiple components
    (state queries, PD control, reward computation) into a complete,
    working task.
    """

    # -- Task parameters --
    SUCCESS_THRESHOLD = 0.005          # reach threshold (5mm)
    MAX_DELTA = 0.05                   # max joint angle change per step
    GOAL_WORKSPACE = np.array([0.2, -0.3, 0.1,   # [x_min, y_min, z_min,
                               0.6,  0.3, 0.5])  #  x_max, y_max, z_max]

    def __init__(self, frame_skip, orcagym_addr, agent_names, time_step, **kwargs):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        # -- Task state --
        self._goal_pos = np.zeros(3)   # target position (randomly sampled each episode)
        self._step_count = 0           # current step count

        # -- PD controller --
        self._pd = PDController(nu=self.model.nu, kp=150.0, kd=12.0)

        # -- Initialize spaces --
        self.action_space = spaces.Box(
            low=-self.MAX_DELTA, high=self.MAX_DELTA,
            shape=(self.model.nu,), dtype=np.float32,
        )
        obs_sample = self._get_obs()
        self.observation_space = spaces.Dict({
            key: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32)
            for key, v in obs_sample.items()
        })

        print(f"ReachTargetTask ready: nu={self.model.nu}, "
              f"dt={self.dt:.4f}s, control frequency={1.0/self.dt:.1f}Hz")

    # ================================================================
    # Observation
    # ================================================================
    def _get_obs(self) -> dict:
        """Collect observation: joint state + end-effector pose + target position"""
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
    # Simulation Step
    # ================================================================
    def step(self, action: np.ndarray):
        """
        Execute one simulation step.

        action: joint angle deltas (nu,), each value in [-MAX_DELTA, MAX_DELTA]
        """
        self._step_count += 1

        # 1. Clamp delta magnitude
        action = np.clip(action, -self.MAX_DELTA, self.MAX_DELTA)

        # 2. Target position = current position + delta
        target_qpos = self.data.qpos[:self.model.nu] + action

        # 3. PD controller -> torque -> simulation step
        ctrl = self._pd.compute(target_qpos, self.data.qpos, self.data.qvel)
        self.do_simulation(ctrl, self.frame_skip)

        # 4. Get new observation
        obs = self._get_obs()

        # 5. Compute reward
        reward, reached = self._compute_reward(obs, action)

        # 6. Check truncation
        truncated = self._step_count >= 300

        info = {
            "dist": obs["dist_to_goal"].item(),
            "step": self._step_count,
        }

        return obs, reward, reached, truncated, info

    # ================================================================
    # Reward Function
    # ================================================================
    def _compute_reward(self, obs, action):
        """
        Reward design:
        - Distance penalty (dense): given every step, closer is better
        - Success reward (sparse): large bonus when target is reached
        - Action smoothness penalty: discourage jerky movements
        """
        dist = obs["dist_to_goal"].item()

        dist_reward = -dist
        reached = dist < self.SUCCESS_THRESHOLD
        success_reward = 50.0 if reached else 0.0
        action_penalty = -0.01 * np.sum(action ** 2)

        reward = dist_reward + success_reward + action_penalty
        return reward, reached

    # ================================================================
    # Reset
    # ================================================================
    def reset_model(self):
        """Reset task: sample a new target position"""
        self._step_count = 0

        # Randomly sample target position within workspace
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
# Register Environment
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
# Manual Control Demo (Non-RL)
# ============================================================
def manual_control_demo():
    """
    Control the robot arm with a hand-written "move toward target" strategy.

    The strategy logic is very simple:
    1. Move in the direction of the target
    2. Stop once reached

    This demonstrates how to complete a simulation task with manual rules,
    without using RL.
    """
    print("=" * 60)
    print(" Manual Control Demo: Robot Arm Reach Target")
    print("=" * 60)

    env = gym.make(ENV_ID)

    obs, info = env.reset()
    goal = info["goal"]
    print(f"\n🎯 Target position: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")

    total_reward = 0

    for step_idx in range(300):
        ee_pos = obs["ee_pos"]
        direction = goal - ee_pos
        dist = np.linalg.norm(direction)

        if dist < env.unwrapped.SUCCESS_THRESHOLD:
            print(f"\n✅ Target reached! Took {step_idx} steps, error {dist*1000:.1f}mm")
            break

        # Simple strategy: random exploration + keep good directions
        action = np.random.randn(env.unwrapped.model.nu) * 0.02

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward

        if step_idx % 30 == 0:
            print(f"  Step {step_idx:3d}: dist={dist*1000:5.1f}mm, "
                  f"ee=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}], "
                  f"reward={reward:+.3f}")

        if terminated:
            break

    if step_idx == 299:
        print(f"\n⏰ Did not reach target within 300 steps, final error {dist*1000:.1f}mm")

    print(f"\nTotal reward: {total_reward:.1f}")
    env.close()


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    manual_control_demo()
```

---

## Code Structure Review

```
ReachTargetTask
│
├── __init__()
│   ├── super().__init__()              <- connect simulation, initialize
│   ├── self._goal_pos                  <- task-specific state
│   ├── self._pd = PDController()       <- low-level controller
│   └── Define action space and observation space
│
├── reset_model()
│   ├── Sample new target position
│   └── Return initial observation
│
├── step(action)
│   ├── target = current + action       <- parse action
│   ├── ctrl = pd.compute(target, ...)  <- PD -> torque
│   ├── do_simulation(ctrl, ...)        <- physics simulation
│   ├── obs = _get_obs()                <- collect observation
│   └── _compute_reward(obs, action)    <- compute reward
│
└── _get_obs()
    ├── self.data.qpos / qvel           <- joint state
    ├── query_site_pos_and_quat()       <- end-effector pose
    └── Compute distance to target
```

---

## What You Have Learned

Looking back at the entire beginners' tutorial, you have mastered:

| Tutorial | Knowledge | How It Appears in This Task |
|----------|-----------|----------------------------|
| Hello World | Environment concepts, `step`/`reset` | `gym.register()`, `gym.make()` |
| Scene Setup | Actor placement, asset adding | (Scene built before the task starts) |
| Your First Environment | Inheriting `OrcaGymEulerEnv` | `class ReachTargetTask(OrcaGymEulerEnv)` |
| Reading State | Query joints/bodies/sites | `query_site_pos_and_quat()` |
| Controlling Joints | qpos/qvel operations | `self.data.qpos[:nu] + action` |
| PD Controller | Target angles -> torque | `self._pd.compute()` |
| **This Task** | **Combining everything** | **Complete reach task** |

---

## Extension Directions

Now that you have the foundations, you can try:

1. **Add camera observations** — include camera images in `_get_obs()`
2. **Smarter control** — replace the simple exploration with an IK solver
3. **Add object manipulation** — place a cube in the scene and have the robot arm push it
4. **RL training** — train a policy with PPO instead of manual control (see [🧠 PPO Training](ppo-training.md))

---

> 🎉 **Congratulations!** You have learned the core usage of OrcaGym from scratch. Now go create your own robot tasks!
