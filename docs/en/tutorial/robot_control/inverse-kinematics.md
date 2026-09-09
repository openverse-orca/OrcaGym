# 🦾 Inverse Kinematics (IK)

Inverse Kinematics converts the target pose of an end-effector into joint angles.

> For complete runnable code, see [OrcaPlayground examples/euler/07_jacobian/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler/07_jacobian).

---

## What is IK?

```
Forward Kinematics (FK): joint angles → end-effector pose (unique solution)
Inverse Kinematics (IK): end-effector pose → joint angles (may have multiple or no solutions)
```

IK presents three major challenges:
1. **Redundancy**: end-effector has 6 DOF, joints may have more → infinitely many solutions
2. **Singularities**: Jacobian degenerates in certain poses → tiny end-effector displacement requires infinite joint velocity
3. **Joint Limits**: solutions must satisfy physical limits

---

## Complete Example: The Big Picture

Below is a **complete IK example** that lifts the G1 left foot by approximately 10cm.
It uses **damped least squares + joint limit clamping + a two-phase strategy**.

```python
"""Damped Least Squares IK: G1 left foot lifted ~10cm"""
import numpy as np


# G1's 29 revolute joint suffixes
G1_ROT_JOINT_SUFFIXES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def damped_least_squares_ik(env, foot_suffix="left_ankle_roll_link",
                            offset=np.array([0.0, 0.05, 0.10]),
                            damping=0.05, step=0.05, iters=80, atol=0.02):
    """Damped least squares IK: move the foot body to a target offset position.

    Two phases:
      1. Preset slight-crouch pose — avoid paths that violate the joint's natural direction from a fully extended state
      2. IK iteration — damped least squares + joint limit clamping

    Args:
        env: OrcaGymEulerEnv instance
        foot_suffix: foot body suffix (without agent prefix)
        offset: target offset [dx, dy, dz] (world frame, meters)
        damping: damping coefficient (larger = more stable, smaller = faster convergence)
        step: maximum joint change per step (radians)
        iters: maximum number of iterations
        atol: convergence threshold (meters)

    Returns:
        Final foot position (3,) np.ndarray
    """
    agent = env._agent_names[0]
    foot_body = f"{agent}_{foot_suffix}"

    # ── Prepare G1 joint info ──
    joint_names = [f"{agent}_{s}" for s in G1_ROT_JOINT_SUFFIXES]
    dof_adrs = [env.jnt_dofadr(jn) for jn in joint_names]
    qpos_adrs = [env.jnt_qposadr(jn) for jn in joint_names]

    # Column range of G1 joints in the global DOF array (must NOT use [7:] in multi-body scenes!)
    v_min, v_max = min(dof_adrs), max(dof_adrs)
    g1_joint_cols = slice(v_min, v_max + 1)

    # Joint limits
    jdict = env.model.get_joint_dict()
    jnt_lo = np.array([
        jdict[jn]["Range"][0] if jdict[jn]["Limited"] else -np.inf
        for jn in joint_names
    ])
    jnt_hi = np.array([
        jdict[jn]["Range"][1] if jdict[jn]["Limited"] else np.inf
        for jn in joint_names
    ])

    # ═══════════════════════════════════════════
    # Phase 1: Preset slight-crouch pose
    # ═══════════════════════════════════════════
    # Starting from a fully extended state, pure DLS may take a path that
    # violates the joint's natural direction (e.g., knee bending backward).
    # Preset forward knee bend + hip flexion + ankle dorsiflexion so IK
    # starts from a reasonable pose.
    preset = {
        f"{agent}_left_knee_joint": 0.6,
        f"{agent}_left_hip_pitch_joint": -0.3,
        f"{agent}_left_ankle_pitch_joint": -0.3,
        f"{agent}_right_knee_joint": 0.6,
        f"{agent}_right_hip_pitch_joint": -0.3,
        f"{agent}_right_ankle_pitch_joint": -0.3,
    }
    qpos = env.data.qpos.copy()
    for jn, val in preset.items():
        qpos[env.jnt_qposadr(jn)] = val
    env.set_joint_qpos(qpos)
    env.mj_forward()
    print(f"  Phase 1 complete: preset slight-crouch pose")

    # ═══════════════════════════════════════════
    # Phase 2: Damped least squares IK iteration
    # ═══════════════════════════════════════════
    foot_pos = env.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
    target = foot_pos + offset
    print(f"  Phase 2 start: target={target}, initial={foot_pos}")

    jacr = np.zeros((3, env.model.nv))
    for i in range(iters):
        # (a) Compute foot Jacobian
        jacp_foot = np.zeros((3, env.model.nv))
        env.mj_jacBody(jacp_foot, jacr, body_name=foot_body)

        # (b) Current error
        cur = env.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
        delta = target - cur

        # (c) Damped least squares: dq = J^T (J J^T + lambda^2 I)^(-1) delta_x
        jac_leg = jacp_foot[:, g1_joint_cols]
        dq = jac_leg.T @ np.linalg.inv(
            jac_leg @ jac_leg.T + damping**2 * np.eye(3)
        ) @ delta

        # (d) Write compliantly + limit clamping
        qpos = env.data.qpos.copy()
        for j, qadr in enumerate(qpos_adrs):
            qpos[qadr] = np.clip(
                qpos[qadr] + dq[j] * step, jnt_lo[j], jnt_hi[j]
            )
        env.set_joint_qpos(qpos)
        env.mj_forward()

        # (e) Convergence check
        err = np.linalg.norm(delta)
        if err < atol:
            print(f"  IK converged at iteration {i + 1}, error {err:.4f}m")
            break
    else:
        print(f"  IK did not converge, final error {err:.4f}m")

    return env.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]


# ============================================================
# Usage Example
# ============================================================
if __name__ == "__main__":
    # env is your OrcaGymEulerEnv instance (already reset)
    final_pos = damped_least_squares_ik(
        env,
        foot_suffix="left_ankle_roll_link",
        offset=np.array([0.0, 0.05, 0.10]),  # y+5cm, z+10cm
    )
    print(f"Left foot final position: {final_pos}")
```

---

## Step-by-Step Explanation

### Damped Least Squares Principle

The standard Jacobian pseudoinverse `J^+ = J^T(J J^T)^(-1)` diverges near singularities.
**Damped Least Squares (DLS)** adds a regularization term `lambda^2 I`:

```
dq = J^T (J J^T + lambda^2 I)^(-1) delta_x
```

| lambda value | Behavior |
|--------------|----------|
| lambda = 0 | Degrades to standard pseudoinverse, unstable near singularities |
| lambda small | Fast convergence, but may oscillate near singularities |
| lambda large | Smoother, more stable solution, but slower convergence |

### IK Iteration Workflow

```
Each iteration:
  1. mj_jacBody → foot Jacobian jacp_foot (3, nv)
  2. jac_leg = jacp_foot[:, g1_dof_min:g1_dof_max+1]  ← only take G1 joint columns
  3. delta_x = target - current_xpos
  4. dq = J^T (J J^T + lambda^2 I)^(-1) delta_x   ← damped least squares
  5. q ← clip(q + dq * step, jnt_lo, jnt_hi)  ← limit clamping
  6. set_joint_qpos(q) + mj_forward()
  7. Check ||delta_x|| < ATOL?
```

### Why Two Phases?

At G1's default `qpos=0`, the knees are **fully extended**. Starting from an extended state,
a pure mathematical DLS solution may cause the knees to **bend backward**
(opposite to the joint's natural direction) — this is mathematically correct but physically infeasible.

**Phase 1 — Preset Slight Crouch**:
- Knee forward bend +0.6 rad (~34 degrees)
- Hip flexion -0.3 rad
- Ankle dorsiflexion -0.3 rad (compensates to keep foot sole level)
- Applied symmetrically to both legs

This way, IK starts from an already-bent state, and the solution naturally continues
bending forward along the joint's positive direction to lift the foot.

**Phase 2 — IK Iteration to Lift Foot**: Damped least squares + limit clamping,
80 iterations converging to ~2cm precision.

### DOF Columns in Multi-Body Scenarios

The column range of G1 joints in the **global DOF array** is **NOT `[7:]`**! In multi-body scenes
(G1 + manipulated object + toy), other bodies' DOFs may be interleaved.

**Correct approach**: Get each joint's DOF address via `jnt_dofadr`, then construct the `[min, max+1]` range:

```python
dof_adrs = [env.jnt_dofadr(jn) for jn in joint_names]
g1_joint_cols = slice(min(dof_adrs), max(dof_adrs) + 1)
```

### Joint Limit Clamping

After each IK iteration, clamp to joint limits to prevent generating unreachable poses:

```python
jdict = env.model.get_joint_dict()
for jn in joint_names:
    lo, hi = jdict[jn]["Range"]       # [lower, upper]
    limited = jdict[jn]["Limited"]    # True/False

# clamp
qpos[qadr] = np.clip(qpos[qadr] + dq * step, lo, hi)
```

---

## Parameter Quick Reference

| Parameter | Function | Recommended | Tuning |
|-----------|----------|-------------|--------|
| `damping` | Numerical stability | 0.01–0.1 | Increase if not converging, decrease if too slow |
| `step` | Joint change per step | 0.02–0.1 | Decrease if oscillating |
| `iters` | Max iterations | 50–200 | Increase for distant targets |
| `atol` | Convergence threshold (m) | 0.01–0.05 | Decrease for higher precision |

## Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| IK does not converge | Damping too small / target too far | Increase damping, reduce offset |
| Joint bends in the wrong direction | Starting from fully extended state | Preset slight-crouch pose |
| Jacobian is all zeros | No forward call / wrong body name | Ensure mj_forward + agent prefix |
| Multi-body scene anomalies | Wrong DOF column range | Use jnt_dofadr for [min, max] |

---

## Next Steps

Now that you've mastered IK, learn how to **apply external forces** and the complete force + IK workflow: [🔄 External Force Application and IK](../physics/force-apply.md).
