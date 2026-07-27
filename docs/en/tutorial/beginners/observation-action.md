# 👁️ Observation & Action — Designing Observation and Action Spaces

In this tutorial, you will gain a deep understanding of how to design **observation spaces** and **action spaces** for your environment. This is a critical step in building effective RL environments.

---

## Observation Space

The observation space defines what information the RL policy can "see" at each time step.

### Principles of Observation Design

Good observation design follows these principles:

1. **Sufficiency**: The observation should contain all information needed to complete the task (joint positions, target positions, etc.)
2. **Compactness**: Only include necessary information; extra dimensions increase the difficulty of policy learning
3. **Reasonable Scaling**: The numerical ranges of different features should be of similar orders of magnitude
4. **Accessibility**: All observation values must be reliably computable from the simulation state

### Example: Designing Observations for a Robot Arm Environment

#### Minimal Observation: Joint State Only

```python
def _get_obs(self) -> dict:
    """Most basic observation: joint position + velocity"""
    return {
        "joint_pos": self.data.qpos.copy(),  # (nq,) generalized positions
        "joint_vel": self.data.qvel.copy(),  # (nv,) generalized velocities
    }
```

#### Advanced Observation: Adding End-Effector Information

```python
def _get_obs(self) -> dict:
    """Observation including end-effector pose"""

    # 1. Basic joint state
    joint_pos = self.data.qpos.copy()
    joint_vel = self.data.qvel.copy()

    # 2. End-effector pose
    ee_site_name = self.site("end_effector")  # auto-prepends agent prefix
    ee_site = self.query_site_pos_and_quat([ee_site_name])

    # ee_site return format: {site_name: {"xpos": array([x,y,z]), "xquat": array([w,x,y,z])}}
    ee_pos = ee_site[ee_site_name]["xpos"]   # end-effector position (3,)
    ee_quat = ee_site[ee_site_name]["xquat"] # end-effector orientation quaternion (4,)

    # 3. End-effector velocity (to capture motion trends)
    ee_linear_vel, ee_angular_vel = self.query_site_xvalp_xvalr([ee_site_name])
    ee_vel = ee_linear_vel[ee_site_name]     # linear velocity (3,)
    ee_angvel = ee_angular_vel[ee_site_name] # angular velocity (3,)

    return {
        "joint_pos": joint_pos,     # (nq,)
        "joint_vel": joint_vel,     # (nv,)
        "ee_pos": ee_pos,           # (3,)
        "ee_quat": ee_quat,         # (4,)
        "ee_vel": ee_vel,           # (3,)
        "ee_angvel": ee_angvel,     # (3,)
    }
```

#### Complete Observation: Adding Goals and Sensors

```python
def _get_obs(self) -> dict:
    """Complete task observation: includes target position and sensor data"""

    # Joint state
    joint_pos = self.data.qpos.copy()
    joint_vel = self.data.qvel.copy()

    # End-effector pose in world coordinates
    ee_site_name = self.site("end_effector")
    ee_site = self.query_site_pos_and_quat([ee_site_name])

    # End-effector pose relative to base (useful for fixed-base robot arms)
    base_name = self.body("base_link")
    ee_pos_B_dict = self.query_site_pos_and_quat_B(
        [ee_site_name], [base_name]
    )

    # Target position (randomly sampled in reset)
    goal_pos = self._goal_pos  # (3,), set in reset_model

    # Distance to target (gives the policy a direct error signal)
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

### Observation Space Types

| Observation Type | Space Type | Example |
|------------------|------------|---------|
| Dictionary observation (recommended) | `spaces.Dict` | `{"joint_pos": Box, "joint_vel": Box}` |
| Single array | `spaces.Box` | `Box(low=-inf, high=inf, shape=(13,))` |

!!! tip "Dictionary observations are recommended"
    Dictionary observations are easier than single arrays for:
    - Debugging (you can inspect each component by name)
    - Extension (adding new observations does not change existing dimensions)
    - Normalization (you can use different normalization strategies for different keys)

---

## Action Space

The action space defines what actions the policy can output to control the robot.

### Three Common Action Space Designs

#### 1. Torque Control

The lowest-level approach — directly control the torque of each joint:

```python
def _set_action_space(self):
    """Torque control: action is the target torque for each joint"""
    ctrlrange = self.model.get_actuator_ctrlrange()  # (nu, 2)
    self.action_space = spaces.Box(
        low=ctrlrange[:, 0], high=ctrlrange[:, 1], dtype=np.float32
    )
```

#### 2. Position Control

Action as target joint positions, converted to torques by a PD controller:

```python
def _set_action_space(self):
    """Position control: action is the target joint angle"""
    joint_ranges = np.array([
        [-3.14, 3.14],   # joint_0: +/-180 degrees
        [-1.57, 1.57],   # joint_1: +/-90 degrees
        # ... configure based on your robot
    ])
    self.action_space = spaces.Box(
        low=joint_ranges[:, 0], high=joint_ranges[:, 1], dtype=np.float32
    )
```

Using a PD controller in `step()` (see [🎮 Simple Controller](simple-controller.md) for details):

```python
def step(self, action):  # action is the target joint angle
    ctrl = self._pd.compute(
        target_qpos=action,
        current_qpos=self.data.qpos,
        current_qvel=self.data.qvel,
    )
    self.do_simulation(ctrl, self.frame_skip)
```

#### 3. Delta Control

Action is an offset relative to the current position:

```python
def step(self, action):  # action in [-0.1, 0.1], representing joint angle change
    max_delta = 0.1  # at most 0.1 radians change per step
    delta = np.clip(action, -max_delta, max_delta)

    # Target position = current position + delta
    target_qpos = self.data.qpos[:self.model.nu] + delta

    # Use PD controller to track target position
    ctrl = self._pd.compute(target_qpos, self.data.qpos, self.data.qvel)
    self.do_simulation(ctrl, self.frame_skip)
```

### Action Space Design Comparison

| Control Mode | Pros | Cons | Use Case |
|--------------|------|------|----------|
| Torque control | Most flexible, highest bandwidth | Hard to train, requires extensive exploration | Fine manipulation, highly dynamic tasks |
| Position control | Easy to train, smooth behavior | Limited response bandwidth | Pick-and-place, assembly and other quasi-static tasks |
| Delta control | Smooth, safe | Speed limited | Tasks requiring smooth trajectories |

> **Beginner's advice**: Start with **position control** or **delta control**. Torque control, while flexible, is much harder for an RL policy to explore.

---

## Complete Example: Integrating Observation and Action

Below is a complete environment demonstrating the integrated design of observation and action spaces:

```python
"""
reach_env.py — A robot arm reach task environment

Observation: joint positions, velocities, end-effector pose, target position
Action: delta joint position control
"""

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class ReachEnv(OrcaGymEulerEnv):
    """Task: move the robot arm end-effector to a specified target point"""

    def __init__(self, frame_skip, orcagym_addr, agent_names, time_step, **kwargs):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        # Task-related: target position
        self._goal_pos = np.zeros(3)

        # Action space: delta position control [-0.05, 0.05] radians/step
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(self.model.nu,), dtype=np.float32
        )

        obs_sample = self._get_obs()
        self.observation_space = spaces.Dict({
            key: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32)
            for key, v in obs_sample.items()
        })

    def _get_obs(self):
        """Collect rich observation information"""
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
        # Delta control: current qpos + action offset
        target_qpos = self.data.qpos[:self.model.nu] + action

        # Compute torque using simple PD
        pos_error = target_qpos - self.data.qpos[:self.model.nu]
        vel_error = -self.data.qvel[:self.model.nv]
        ctrl = pos_error * 100.0 + vel_error * 10.0

        # Execute simulation
        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        dist = obs["dist_to_goal"].item()
        reward = -dist  # closer = higher reward
        terminated = dist < 0.01  # distance less than 1cm considered success
        truncated = False

        return obs, reward, terminated, truncated, {"distance": dist}

    def reset_model(self):
        """Reset robot and randomly sample a new target"""
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

## Debugging Tips for Observations and Actions

### 1. Validate Observation Validity

```python
def _validate_obs(obs):
    """Ensure no NaN or Inf in observations"""
    for key, val in obs.items():
        if np.any(np.isnan(val)):
            print(f"⚠️ NaN in obs['{key}']")
        if np.any(np.isinf(val)):
            print(f"⚠️ Inf in obs['{key}']")
        print(f"  obs['{key}']: shape={val.shape}, "
              f"range=[{val.min():.3f}, {val.max():.3f}]")
```

### 2. Random Action Exploration

During development, test environment stability with random actions:

```python
env = ReachEnv(...)
obs, _ = env.reset()
for i in range(200):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
```

---

## Next Step

You have mastered observation and action space design. Now learn how to write a controller: [🎮 Simple Controller](simple-controller.md).
