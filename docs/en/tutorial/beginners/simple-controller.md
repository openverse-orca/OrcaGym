# 🎮 Simple Controller — PD Control

Previously we drove the robot by manually setting `qpos` or a constant torque. In this section, you will learn to write a **PD controller** — the most fundamental and most commonly used controller in robot control.

> This section is explained using the G1 walking example from [OrcaPlayground Lesson 8 - Locomotion](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/08_locomotion), which is the core lesson of the PD control chain.

---

## What is a PD Controller?

PD = Proportional + Derivative

```
torque = Kp × (target position - current position) + Kd × (target velocity - current velocity)
  τ  = Kp ×      Δpos            + Kd ×      Δvel
```

**Intuitive understanding**:

- **P term (proportional)**: the farther from the target, the harder it pulls — like a spring pulling you toward the target
- **D term (derivative)**: the faster the motion, the larger the resistance — like a damper preventing you from overshooting

```
target position ──→ [Kp × position error] ──→ ┐
                                ├──→ torque ──→ joint
current velocity ──→ [Kd × velocity error] ──→ ┘
```

---

## Implementing a PD Controller from Scratch

```python
"""
simple_pd_controller.py — a PD controller implemented from scratch
"""

import numpy as np


class SimplePDController:
    """
    Multi-joint PD controller.

    Computes torque independently for each joint:
        torque = kp * (target - current_pos) + kd * (0 - current_vel)

    Parameters:
        kp: proportional gain — the larger, the faster the tracking, but may oscillate
        kd: derivative gain — the larger, the more stable, but may feel sluggish

    Units:
        kp: N·m/rad (revolute joint) — torque produced per radian of position error
        kd: N·m·s/rad (revolute joint) — torque produced per radian/second of velocity error
    """

    def __init__(self, nu: int, kp: float = 100.0, kd: float = 10.0):
        """
        Args:
            nu: number of actuators (joints)
            kp: proportional gain, typical value 50~500
            kd: derivative gain, typical value 5~50
        """
        self.nu = nu
        self.kp = np.full(nu, kp, dtype=np.float64)  # each joint can have a different gain
        self.kd = np.full(nu, kd, dtype=np.float64)

    def compute(self, target_qpos, current_qpos, current_qvel):
        """
        Compute control torque.

        Args:
            target_qpos: target joint position (nu,) — "where to go"
            current_qpos: current joint position (nq,) — "where it is now"
            current_qvel: current joint velocity (nv,) — "how fast now"

        Returns:
            torque: control torque (nu,) — "how hard to push"
        """
        pos_error = target_qpos - current_qpos[:self.nu]
        vel_error = np.zeros(self.nu) - current_qvel[:self.nu]
        torque = self.kp * pos_error + self.kd * vel_error
        return torque.astype(np.float64)


# ============================================================
# Use the PD controller in an environment
# ============================================================

def demo_pd_control(env, target_angle: float = 0.8, steps: int = 150):
    """
    Demo: drive joint 0 to a target angle with a PD controller.

    Observe how the joint arrives at the target smoothly (instead of teleporting).

    Note: only one do_simulation call per step here, which is an "open-loop simplified" demo.
    For frame_skip > 1 or highly dynamic scenes (such as humanoid walking),
    use the "closed-loop PD" approach introduced later (recompute tau at every physics step).
    """
    nu = env.model.nu
    pd = SimplePDController(nu=nu, kp=150.0, kd=12.0)

    # Target joint position — only joint 0 moves, the others stay in place
    target_qpos = env.data.qpos[:nu].copy()
    target_qpos[0] = target_angle  # turn joint 0 to target_angle radians

    print(f"target angle: {target_qpos[0]:.3f} rad")
    print(f"initial angle: {env.data.qpos[0]:.3f} rad")

    for i in range(steps):
        # PD computes torque (recomputed every step, for the simplified demo)
        ctrl = pd.compute(
            target_qpos=target_qpos,
            current_qpos=env.data.qpos,
            current_qvel=env.data.qvel,
        )

        # Run simulation (open-loop: one tau, step frame_skip times)
        env.do_simulation(ctrl, env.frame_skip)
        env.render()

        # Print progress
        pos_error = abs(target_qpos[0] - env.data.qpos[0])
        if i % 15 == 0:
            print(f"  Step {i:3d}: pos={env.data.qpos[0]:+.4f}, "
                  f"error={pos_error:.4f}, torque={ctrl[0]:+.2f}")

        # Exit early after reaching the target
        if pos_error < 0.001:
            print(f"  ✅ reached the target! took {i} steps")
            break

    print(f"final angle: {env.data.qpos[0]:.3f} rad")
```

---

## Real-World Example: G1 Walking PD Control Chain

The `SimplePDController` above demonstrates the basic principle. In real robot control, each joint often needs **independent Kp/Kd**, and needs **torque limiting** (clip) to prevent overload.

The G1 humanoid walking control in [OrcaPlayground Lesson 8](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/08_locomotion) is a typical PD control chain:

```
policy output q_target (29-dim) ──→ PD controller ──→ tau torque ──→ motor actuator ──→ walking
                              ↑
                     Kp, Kd, torque limit (from YAML config)
```

### 1. Independent Kp/Kd per Joint

The G1 has 29 joints in total (12 legs + 3 waist + 14 two arms), and different body parts carry very different loads, so each joint is configured with independent PD gains. The config lives in [assets/g1/config/g1_29dof_hist.yaml](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/assets/g1/config/g1_29dof_hist.yaml):

```yaml
# legs (hip + knee + ankle): load-bearing, need high stiffness
JOINT_KP: [
    100, 100, 100, 200, 20, 20,     # left leg: hip pitch/roll/yaw=100, knee=200, ankle=20
    100, 100, 100, 200, 20, 20,     # right leg: same as above
    400, 400, 400,                  # waist: yaw/roll/pitch=400 (core stability)
    90, 60, 20, 60, 4, 4, 0,        # left arm: shoulder pitch=90, roll=60, yaw=20, elbow=60, wrist=4/4/0
    90, 60, 20, 60, 4, 4, 4         # right arm: same as left (wrist yaw=4 instead of 0)
]

JOINT_KD: [
    2.5, 2.5, 2.5, 5, 0.2, 0.1,     # left leg damping
    2.5, 2.5, 2.5, 5, 0.2, 0.1,     # right leg damping
    5.0, 5.0, 5.0,                  # waist damping
    2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2,  # left arm damping
    2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2   # right arm damping
]
```

**The pattern is obvious at a glance**:

| Body part | Kp | Kd | Physical meaning |
|-----------|-----|-----|------------------|
| waist | 400 | 5.0 | carries upper body weight, highest stiffness |
| knee | 200 | 5.0 | single-leg load-bearing, needs high stiffness |
| hip | 100 | 2.5 | leg swing drive |
| shoulder | 90 / 60 | 2.0 / 1.0 | arms are light, lower gains |
| wrist | 0~4 | 0.2 | end-effector flexibility, barely exerts force |

### 2. PD Controller Implementation (with Torque Limiting)

`G1Locomotion.compute_tau` is the core PD control method:

```python
def compute_tau(self, q_target, dof_pos, dof_vel):
    """
    PD controller: position target q_target → torque tau

        tau = Kp * (q_target - q) + Kd * (0 - qd)

    Feedforward torque tau_ff=0, target velocity dq_target=0 (consistent with the G1 training side)
    """
    tau = self.joint_kp * (q_target - dof_pos) - self.joint_kd * dof_vel
    # torque limit: prevent exceeding the motor's rated output
    tau = np.clip(tau, -self.motor_effort_limit, self.motor_effort_limit)
    return tau
```

Differences from the `SimplePDController` above:

| Item | `SimplePDController` | `G1Locomotion.compute_tau` |
|------|---------------------|--------------------------|
| Kp / Kd | scalar (shared by all joints) | per-joint independent (vector) |
| torque limit | none | `np.clip` to `motor_effort_limit` |
| target velocity | default 0 | default 0 (feedforward `tau_ff=0`) |

> 💡 **Torque limiting** is an indispensable safety mechanism in real robot control, preventing momentary large currents from damaging motors or destabilizing the robot. The G1's `motor_effort_limit` is likewise configured in the yaml.

### 3. Closed-Loop PD: Recompute tau Every Physics Step

Open-loop PD (compute tau once, then step multiple times) accumulates error and destabilizes. Closed-loop PD re-reads the joint state and recomputes torque at **every physics step**:

```python
def step(self, action):
    """Call do_simulation multiple times with frame_skip=1 for fine-grained PD control"""
    action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
    for _ in range(self.frame_skip):
        ctrl = self._pd_controller(action)   # recompute tau every step
        self.do_simulation(ctrl, 1)           # single physics step
    obs = self._get_obs()
    reward = self._compute_reward(obs, action)
    ...
```

`_pd_controller` is a hook that subclasses override per scene:

```python
class G1LocomotionEnv(G1BaseEnv):
    def _pd_controller(self, target):
        """Override with G1 walking PD control"""
        dof_pos, dof_vel = self._loco.read_joint_state(self)  # read current joint state
        return self._loco.compute_tau(target, dof_pos, dof_vel)
```

**Closed-loop vs open-loop comparison**:

| Approach | How | Problem |
|----------|-----|---------|
| open-loop PD | `tau = pd.compute(target, q₀, v₀)` → `do_simulation(tau, frame_skip)` | state already changed within frame_skip steps, tau still based on old obs, accumulated error |
| closed-loop PD ⭐ | re-read obs and recompute tau every physics step | tau always based on the latest state, stable tracking |

> 💡 Closed-loop PD is the standard approach in the OrcaPlayground G1 series of examples (Lesson 6~10), and is also the de facto standard for humanoid robot control. You are advised to adopt it in your own environments too.

---

## PD Parameter Tuning

Tuning PD parameters is like adjusting a hot-water tap — too hot, add some cold water; too cold, add some hot water:

| Symptom | Cause | Adjustment |
|---------|-------|------------|
| joint oscillation, back-and-forth jitter | Kp too large | ⬇️ decrease Kp, ⬆️ increase Kd |
| slow response, can't keep up with target | Kp too small | ⬆️ increase Kp |
| won't settle after arriving, slight vibration | Kd too small | ⬆️ increase Kd |
| moves like in honey, too sluggish | Kd too large | ⬇️ decrease Kd |

**Recommended tuning workflow**:

```python
# Step 1: Kd=0, gradually increase Kp until the joint starts to oscillate slightly
pd = SimplePDController(nu=nu, kp=50, kd=0)   # try
pd = SimplePDController(nu=nu, kp=100, kd=0)  # try again
pd = SimplePDController(nu=nu, kp=200, kd=0)  # it oscillated! back off to 150

# Step 2: fix Kp=150, gradually increase Kd until oscillation disappears
pd = SimplePDController(nu=nu, kp=150, kd=5)   # try
pd = SimplePDController(nu=nu, kp=150, kd=10)  # much better
pd = SimplePDController(nu=nu, kp=150, kd=15)  # a bit sluggish, back off to 12
```

### Reference Values for Different Scenarios

| Scenario | Kp | Kd | Notes |
|----------|-----|-----|-------|
| lightweight arm (≤2kg) | 80~150 | 8~15 | low inertia, low gains suffice |
| heavy-duty arm (≥10kg) | 200~500 | 20~50 | needs larger force |
| high-precision tasks | 300~500 | 30~50 | needs fast response |
| human-robot collaboration | 50~100 | 15~30 | safety first, can't be too "stiff" |
| humanoid waist | 400 | 5.0 | carries upper body weight, highest stiffness (see G1) |
| humanoid wrist | 0~4 | 0.2 | end-effector flexibility, barely exerts force (see G1) |

> 💡 For complex robots, it is recommended to configure **per-joint independent Kp/Kd**, referring to the G1 yaml configuration, tuning by body part (hip/knee/waist/shoulder/wrist).

---

## Where the Controller Sits in the Environment

```
your step() method
│
├── action (policy / program output)
│     │
│     ├── position control: action = target joint angle
│     │   └── PD.compute(target=action, ...) → torque
│     │
│     ├── delta control: action = angle delta
│     │   └── target = current + action
│     │       └── PD.compute(target, ...) → torque
│     │
│     └── torque control: action is the torque
│         └── directly do_simulation(action, ...)
│
└── do_simulation(ctrl, frame_skip)
```

**Closed-loop approach** (recommended, see the G1 example):

```python
def step(self, action):
    action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
    for _ in range(self.frame_skip):
        ctrl = self._pd_controller(action)  # recompute every physics step
        self.do_simulation(ctrl, 1)
    return self._get_obs(), reward, terminated, truncated, info
```

---

## Complete Example: A Position Control Environment

```python
class PositionControlEnv(OrcaGymEulerEnv):
    """Action = target joint angles, internally converted to torque by PD"""

    def __init__(self, frame_skip, orcagym_addr, agent_names, time_step, **kwargs):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        # Create the PD controller
        self._pd = SimplePDController(nu=self.model.nu, kp=150.0, kd=12.0)

        # Action space = joint limit ranges
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
        # action = target joint angles
        ctrl = self._pd.compute(
            target_qpos=action,
            current_qpos=self.data.qpos,
            current_qvel=self.data.qvel,
        )
        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        tracking_error = np.mean(np.abs(action - self.data.qpos[:self.model.nu]))
        reward = -tracking_error  # the closer the tracking, the higher the reward
        return obs, reward, False, False, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}
```

---

## Next Step

You've learned how to precisely control joints. Now combine everything you've learned so far and **build a complete task**: [🏆 Build a Task](build-a-task.md).
