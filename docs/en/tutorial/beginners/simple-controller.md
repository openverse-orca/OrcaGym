# 🎮 Simple Controller — PD Control

Previously we drove the robot by manually setting `qpos` or applying constant torque. In this section, you will learn to write a **PD controller** — the most fundamental and commonly used controller in robotics.

---

## What Is a PD Controller?

PD = Proportional + Derivative

```
torque = Kp * (target position - current position) + Kd * (target velocity - current velocity)
   tau = Kp *            delta_pos              + Kd *            delta_vel
```

**Intuition**:

- **P term (Proportional)**: The farther from the target, the stronger the force — like a spring pulling you toward the target
- **D term (Derivative)**: The faster the velocity, the stronger the resistance — like a damper preventing overshoot

```
target position --> [Kp * error] --> +
                                      |--> torque --> joint
current velocity -> [Kd * error] --> +
```

---

## Implementing a PD Controller from Scratch

```python
"""
simple_pd_controller.py — PD controller implemented from scratch
"""

import numpy as np


class SimplePDController:
    """
    Multi-joint PD controller.

    Computes torque independently for each joint:
        torque = kp * (target - current_pos) + kd * (0 - current_vel)

    Parameters:
        kp: proportional gain — larger = faster tracking, but may oscillate
        kd: derivative gain — larger = more stable, but may become sluggish
    """

    def __init__(self, nu: int, kp: float = 100.0, kd: float = 10.0):
        """
        Args:
            nu: number of actuators (joints)
            kp: proportional gain, typical range 50~500
            kd: derivative gain, typical range 5~50
        """
        self.nu = nu
        self.kp = np.full(nu, kp, dtype=np.float64)  # each joint can have different gains
        self.kd = np.full(nu, kd, dtype=np.float64)

    def compute(self, target_qpos, current_qpos, current_qvel):
        """
        Compute control torques.

        Args:
            target_qpos: target joint positions (nu,) — "where you want to go"
            current_qpos: current joint positions (nq,) — "where you are now"
            current_qvel: current joint velocities (nv,) — "how fast you are now"

        Returns:
            torque: control torques (nu,) — "how much force to apply"
        """
        pos_error = target_qpos - current_qpos[:self.nu]
        vel_error = np.zeros(self.nu) - current_qvel[:self.nu]
        torque = self.kp * pos_error + self.kd * vel_error
        return torque.astype(np.float64)


# ============================================================
# Using the PD controller in an environment
# ============================================================

def demo_pd_control(env, target_angle: float = 0.8, steps: int = 150):
    """
    Demo: use a PD controller to drive joint 0 to a target angle.

    Observe how the joint smoothly reaches the target (instead of teleporting).
    """
    nu = env.model.nu
    pd = SimplePDController(nu=nu, kp=150.0, kd=12.0)

    # Target joint position — only joint 0 moves, others stay at current position
    target_qpos = env.data.qpos[:nu].copy()
    target_qpos[0] = target_angle  # joint 0 moves to target_angle radians

    print(f"Target angle: {target_qpos[0]:.3f} rad")
    print(f"Initial angle: {env.data.qpos[0]:.3f} rad")

    for i in range(steps):
        # PD computes torque
        ctrl = pd.compute(
            target_qpos=target_qpos,
            current_qpos=env.data.qpos,
            current_qvel=env.data.qvel,
        )

        # Execute simulation
        env.do_simulation(ctrl, env.frame_skip)
        env.render()

        # Print progress
        pos_error = abs(target_qpos[0] - env.data.qpos[0])
        if i % 15 == 0:
            print(f"  Step {i:3d}: pos={env.data.qpos[0]:+.4f}, "
                  f"error={pos_error:.4f}, torque={ctrl[0]:+.2f}")

        # Exit early when target is reached
        if pos_error < 0.001:
            print(f"  ✅ Target reached! Took {i} steps")
            break

    print(f"Final angle: {env.data.qpos[0]:.3f} rad")
```

---

## PD Parameter Tuning

Tuning PD parameters is like adjusting a shower faucet — too hot, add some cold; too cold, add some hot:

| Symptom | Cause | Adjustment |
|---------|-------|------------|
| Joint oscillates, shakes back and forth | Kp too large | Decrease Kp, increase Kd |
| Slow response, can't keep up with target | Kp too small | Increase Kp |
| Can't stop after reaching, slight vibration | Kd too small | Increase Kd |
| Moves like through honey, too sluggish | Kd too large | Decrease Kd |

**Recommended tuning process**:

```python
# Step 1: Kd=0, gradually increase Kp until joint begins to oscillate slightly
pd = SimplePDController(nu=nu, kp=50, kd=0)   # try
pd = SimplePDController(nu=nu, kp=100, kd=0)  # try again
pd = SimplePDController(nu=nu, kp=200, kd=0)  # oscillating! Back off to 150

# Step 2: Fix Kp=150, gradually increase Kd until oscillation disappears
pd = SimplePDController(nu=nu, kp=150, kd=5)   # try
pd = SimplePDController(nu=nu, kp=150, kd=10)  # much better
pd = SimplePDController(nu=nu, kp=150, kd=15)  # a bit sluggish, back off to 12
```

### Reference Values for Different Scenarios

| Scenario | Kp | Kd | Notes |
|----------|-----|-----|-------|
| Lightweight arm (<=2kg) | 80~150 | 8~15 | Low inertia, low gain suffices |
| Heavy-duty arm (>=10kg) | 200~500 | 20~50 | Needs more force |
| High-precision tasks | 300~500 | 30~50 | Needs fast response |
| Human-robot collaboration | 50~100 | 15~30 | Safety first, can't be too "stiff" |

---

## Where the Controller Fits in the Environment

```
Your step() method
│
├── action (output of policy/program)
│     │
│     ├── Position control: action = target joint angles
│     │   └── PD.compute(target=action, ...) -> torque
│     │
│     ├── Delta control: action = angle delta
│     │   └── target = current + action
│     │       └── PD.compute(target, ...) -> torque
│     │
│     └── Torque control: action is the torque itself
│         └── directly do_simulation(action, ...)
│
└── do_simulation(ctrl, frame_skip)
```

---

## Complete Example: Position Control Environment

```python
class PositionControlEnv(OrcaGymEulerEnv):
    """Action = target joint angles, internally converted to torque via PD"""

    def __init__(self, frame_skip, orcagym_addr, agent_names, time_step, **kwargs):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        # Create PD controller
        self._pd = SimplePDController(nu=self.model.nu, kp=150.0, kd=12.0)

        # Action space = joint limit range
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
        reward = -tracking_error  # better tracking = higher reward
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

You have learned how to precisely control joints. Now combine all the knowledge from previous sections to **build a complete task**: [🏆 Build a Task](build-a-task.md).
