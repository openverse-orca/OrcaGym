# 🔄 Force Application, State Writing and IK — Controlling Simulation State

This section covers how to **apply external forces, write state, use Jacobian matrices, and inverse kinematics (IK)** to control the simulation.

> See [OrcaPlayground examples/euler/05_force_apply/](https://github.com/OrcaGym/OrcaPlayground) and [06_jacobian/](https://github.com/OrcaGym/OrcaPlayground) for complete runnable code.

---

## Complete Example: Overview First

Below is a **directly runnable** complete example demonstrating all core operations: external force application, state writing, Jacobian computation, and IK.
It is recommended to read through it first, then review the section-by-section explanation.

```python
"""Complete Example: External Force + IK Control for G1 Humanoid Robot

Features:
  1. Apply upward force to pelvis to lift the robot
  2. Clear external forces and verify
  3. Set friction coefficients
  4. Drag an object using mocap + weld constraint
  5. Lift the left foot using damped least-squares IK

Prerequisites: OrcaStudio must be running online, with a scene containing G1 loaded
"""
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


# ================================================================
# G1 Joint Definitions (29 revolute joints)
# ================================================================
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


class ForceAndIKDemo(OrcaGymEulerEnv):
    """External Force + IK Demo Environment"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 20),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["g1"]),
            time_step=kwargs.pop("time_step", 0.001),
            model_xml_path=model_xml_path,
            **kwargs,
        )

    def demo_force_apply(self):
        """Demo 1: Apply external force to lift the robot"""
        agent = self.agent_name

        # Record initial height
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        z_before = float(pelvis[f"{agent}_pelvis"]["xpos"][2])
        print(f"Initial pelvis height: {z_before:.3f}m")

        # Apply 500N upward force
        self.apply_body_force(
            f"{agent}_pelvis",
            force=np.array([0.0, 0.0, 500.0]),
            torque=np.array([0.0, 0.0, 0.0]),
        )

        # Step 20 control cycles to let the force take effect
        ctrl = np.zeros(self.model.nu)
        for _ in range(20):
            self.do_simulation(ctrl, self.frame_skip)

        # Verify the lift
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        z_after = float(pelvis[f"{agent}_pelvis"]["xpos"][2])
        print(f"Pelvis height after force: {z_after:.3f}m (Δ={z_after - z_before:.3f}m)")

        # Verify that xfrc_applied recorded the force
        body_id = self.model.body_name2id(f"{agent}_pelvis")
        xfrc = self.data.xfrc_applied[body_id, :3]
        print(f"Force recorded in xfrc_applied: {xfrc}")

        # Clear external force
        self.clear_body_force(f"{agent}_pelvis")
        xfrc = self.data.xfrc_applied[body_id, :3]
        assert np.all(xfrc == 0), "xfrc should be zero after clearing force"
        print("✅ External force cleared")

    def demo_mocap_drag(self):
        """Demo 2: Drag an object using mocap + weld constraint"""
        agent = self.agent_name

        # Set mocap target pose
        target_pos = np.array([0.7, 0.0, 0.5])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.set_mocap_pos_and_quat({
            f"{agent}_TestMocapAnchor": {
                "pos": target_pos,
                "quat": target_quat,
            }
        })

        # Read back to verify write consistency
        read_pos = self.data.mocap_pos(f"{agent}_TestMocapAnchor")
        read_quat = self.data.mocap_quat(f"{agent}_TestMocapAnchor")
        assert np.allclose(read_pos, target_pos, atol=1e-6), "mocap position readback inconsistent"
        assert np.allclose(read_quat, target_quat, atol=1e-6), "mocap quaternion readback inconsistent"
        print(f"✅ mocap write/read consistent: pos={read_pos}")

        # Step to let the weld constraint take effect → object follows mocap
        ctrl = np.zeros(self.model.nu)
        for _ in range(10):
            self.do_simulation(ctrl, self.frame_skip)

        # Verify the object has followed
        box = self.get_body_xpos_xmat_xquat([f"{agent}_manipulation_box"])
        box_pos = box[f"{agent}_manipulation_box"]["xpos"]
        print(f"Object position: {box_pos} (target: {target_pos})")
        print(f"✅ weld constraint drives object to follow mocap")

    def demo_ik_lift_foot(self):
        """Demo 3: Damped least-squares IK to lift the left foot"""
        agent = self.agent_name
        foot_body = f"{agent}_left_ankle_roll_link"

        # --- Prepare G1 joint info ---
        joint_names = [f"{agent}_{s}" for s in G1_ROT_JOINT_SUFFIXES]
        dof_adrs = [self.jnt_dofadr(jn) for jn in joint_names]
        qpos_adrs = [self.jnt_qposadr(jn) for jn in joint_names]

        # Column range of G1 joints in global dof (cannot use [7:] directly in multi-body scenes)
        v_min, v_max = min(dof_adrs), max(dof_adrs)
        g1_joint_cols = slice(v_min, v_max + 1)

        # Joint limits
        jdict = self.model.get_joint_dict()
        jnt_lo = np.array([
            jdict[jn]["Range"][0] if jdict[jn]["Limited"] else -np.inf
            for jn in joint_names
        ])
        jnt_hi = np.array([
            jdict[jn]["Range"][1] if jdict[jn]["Limited"] else np.inf
            for jn in joint_names
        ])

        # --- Phase 1: Preset a slight squat pose (avoids anti-joint paths) ---
        preset = {
            f"{agent}_left_knee_joint": 0.6,
            f"{agent}_left_hip_pitch_joint": -0.3,
            f"{agent}_left_ankle_pitch_joint": -0.3,
            f"{agent}_right_knee_joint": 0.6,
            f"{agent}_right_hip_pitch_joint": -0.3,
            f"{agent}_right_ankle_pitch_joint": -0.3,
        }
        qpos = self.data.qpos.copy()
        for jn, val in preset.items():
            qpos[self.jnt_qposadr(jn)] = val
        self.set_joint_qpos(qpos)
        self.mj_forward()
        print("Phase 1 complete: preset slight squat pose")

        # --- Phase 2: IK iteration to lift the left foot ---
        DAMPING = 0.05
        STEP = 0.05
        ITERS = 80
        ATOL = 0.02

        foot_pos = self.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
        target = foot_pos + np.array([0.0, 0.05, 0.10])  # lift ~10cm

        jacr = np.zeros((3, self.model.nv))
        for i in range(ITERS):
            # Jacobian
            jacp_foot = np.zeros((3, self.model.nv))
            self.mj_jacBody(jacp_foot, jacr, body_name=foot_body)
            cur = self.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
            delta = target - cur

            # Damped least-squares: dq = J^T (J J^T + λ²I)^(-1) Δx
            jac_leg = jacp_foot[:, g1_joint_cols]
            dq = jac_leg.T @ np.linalg.inv(
                jac_leg @ jac_leg.T + DAMPING**2 * np.eye(3)
            ) @ delta

            # Compliant write + limit clamp
            qpos = self.data.qpos.copy()
            for j, qadr in enumerate(qpos_adrs):
                qpos[qadr] = np.clip(qpos[qadr] + dq[j] * STEP, jnt_lo[j], jnt_hi[j])
            self.set_joint_qpos(qpos)
            self.mj_forward()

            err = np.linalg.norm(delta)
            if err < ATOL:
                print(f"IK converged at iteration {i + 1}, error {err:.4f}m")
                break

        final = self.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
        print(f"Left foot: initial={foot_pos}, final={final}, target={target}")
        print(f"Error: {np.linalg.norm(final - target):.4f}m")

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        return self._get_obs(), 0.0, False, False, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.data.qpos.copy()


# ================================================================
# Run
# ================================================================
if __name__ == "__main__":
    env = ForceAndIKDemo(
        model_xml_path="/path/to/g1_29dof_camera.xml",
        skip_grpc_load=False,  # Online mode, connect to Studio
    )
    env.reset()

    env.demo_force_apply()
    env.demo_mocap_drag()
    env.demo_ik_lift_foot()

    env.close()
```

---

## Section-by-Section Explanation

### 1. External Force Application

```python
env.apply_body_force(
    "g1_pelvis",                          # body name (includes agent prefix)
    force=np.array([0.0, 0.0, 500.0]),    # force (N), world frame
    torque=np.array([0.0, 0.0, 0.0]),     # torque (N·m), world frame
)
```

**Principle**: `apply_body_force` directly writes force/torque into MuJoCo's `xfrc_applied` array.
The force acts at the body's center of mass, and the torque acts about the body's center of mass. These forces participate in the dynamics computation on the next `mj_step()`.

**Verification**: The currently applied force can be read via `env.data.xfrc_applied[body_id, :3]` (zero-copy read-only view):
```python
body_id = env.model.body_name2id("g1_pelvis")
xfrc = env.data.xfrc_applied[body_id, :3]  # [fx, fy, fz]
```

**Clearing**:
```python
env.clear_body_force("g1_pelvis")   # Clear force on a single body
env.clear_all_forces()              # Clear all external forces
```

> **Note**: G1 uses force-controlled motors, so joints produce zero torque when `ctrl=0`. When applying external forces, choose a body that can directly receive force (e.g., pelvis),
> and avoid applying force to bodies on a limp joint chain (the force will be absorbed by the joints).

### 2. Mocap Dragging

**Mocap bodies** are special bodies in MuJoCo (`body_mocapid != -1`) that can have their pose **directly set**,
unaffected by forces/dynamics. Combined with a **WELD equality constraint**, they can move regular bodies as if "dragged by an invisible hand."

```python
# Write mocap pose
env.set_mocap_pos_and_quat({
    "mocap_name": {
        "pos": np.array([x, y, z]),
        "quat": np.array([w, x, y, z]),
    }
})

# Read back and verify
read_pos = env.data.mocap_pos("mocap_name")    # (3,)
read_quat = env.data.mocap_quat("mocap_name")  # (4,)
```

**Complete Dragging Workflow** (under the Euler path, `anchor_actor`/`release_body_anchored` are not public Env API; programmatic operations should follow the UI-grasp internal method orchestration pattern using these public primitives):
1. `equality_find_slot_by_body(env.body("mocap_anchor"))` — find the equality constraint slot containing the mocap
2. `equality_constraint(slot)` — save original constraint snapshot (restore on release)
3. `set_mocap_pos_and_quat(...)` — align mocap pose to the object's current pose
4. `equality_update(slot, eq_type=mjtEq.mjEQ_WELD, obj1_name=..., obj2_name=...)` — establish WELD constraint
5. `set_mocap_pos_and_quat(...)` — move mocap → object follows
6. `do_simulation(...)` — step to let the constraint take effect
7. `equality_update(slot, ...)` — restore original constraint from snapshot (release)

> Note: `anchor_actor` / `release_body_anchored` are public methods only in the Local system (`OrcaGymLocalEnv`). Under the Euler system they are UI-grasp internal methods (`_anchor_actor` / `_release_body_anchored`) and should not be called directly.

### 3. State Writing

**Compliant State Writing Pattern** (W1 rule): `copy → modify → set → forward`

```python
# ❌ Wrong: directly writing data.qpos (read-only view)
# self.data.qpos[0] = 0.5

# ✅ Correct
qpos = env.data.qpos.copy()       # 1. Copy
qpos[addr] = new_value             # 2. Modify the copy
env.set_joint_qpos(qpos)           # 3. Compliant write
env.mj_forward()                   # 4. Required! Update derived quantities
env._sync_view()                   # 5. Sync to DataView
```

> ⚠️ **Critical**: After modifying qpos/qvel, you **must call `mj_forward()`**. Without it,
> body poses read by `get_body_xpos_xmat_xquat` etc. will still be the old values.

### 4. Jacobian Matrices

**`mj_jacBody`** — Computes the translational and rotational Jacobians for a specified body:

```python
nv = env.model.nv
jacp = np.zeros((3, nv))   # translational Jacobian (3, nv) — written in-place
jacr = np.zeros((3, nv))   # rotational Jacobian (3, nv)

env.mj_jacBody(jacp, jacr, body_name="g1_pelvis")

# Mathematical relationship:
# jacp @ qvel = body world-frame linear velocity
# jacr @ qvel = body world-frame angular velocity
```

**`mj_jacSite`** — Computes the Jacobian for a site point:

```python
jacp_site = np.zeros((3, env.model.nv))
env.mj_jacSite(jacp_site, jacr_site, site_name="g1_imu")

# Verify consistency
xvalp, _ = env.query_site_xvalp_xvalr(["g1_imu"])
expected = jacp_site @ env.data.qvel       # jac @ qvel should equal the queried velocity
assert np.allclose(xvalp["g1_imu"], expected, atol=1e-4)
```

### 5. Damped Least-Squares IK

**Why damping is needed?** The standard Jacobian pseudoinverse `J⁺ = J^T(J J^T)^(-1)` diverges near singularities
(`J J^T` becomes nearly singular, pseudoinverse elements tend toward infinity). Damped least-squares adds a regularization term λ²I:

```
dq = J^T (J J^T + λ²I)^(-1) Δx
```

- **λ too small** → approaches pseudoinverse, unstable at singularities
- **λ too large** → slow convergence, but stable

**Why two phases are needed?**

At G1's default `qpos=0`, the knees are fully extended. Starting from the extended pose, a purely mathematical IK solution may cause the knee to **bend backward**
(anti-joint direction) — mathematically correct but physically infeasible.

**Phase 1 — Preset slight squat**: knee forward bend +0.6 rad, hip forward flexion -0.3 rad, ankle dorsiflexion -0.3 rad.
This way IK starts from an already bent state, and the solution naturally continues bending forward along the joint to lift the foot.

**Phase 2 — IK iteration**:
```
Each iteration:
  1. mj_jacBody → foot Jacobian jacp_foot
  2. jac_leg = jacp_foot[:, g1_dof_min:g1_dof_max+1]  ← only G1 joint columns
  3. dq = jac_leg^T @ (jac_leg @ jac_leg^T + λ²I)^(-1) @ Δx  ← DLS
  4. q ← clip(q + dq·step, jnt_lo, jnt_hi)  ← limit clamp
  5. set_joint_qpos + mj_forward
  6. Check convergence: ||Δx|| < ATOL?
```

**DOF columns in multi-body scenes**: G1 joint positions in the global dof array are not `[7:]`.
You must get each joint's dof address individually via `jnt_dofadr` and construct the `[min, max+1]` range.

```python
dof_adrs = [env.jnt_dofadr(jn) for jn in joint_names]
v_min, v_max = min(dof_adrs), max(dof_adrs)
g1_joint_cols = slice(v_min, v_max + 1)  # Correct G1 joint column range
```

### 6. Joint Limit Clamping

Read limit information from `model.get_joint_dict()` and clamp after each IK iteration:

```python
jdict = env.model.get_joint_dict()
jnt_lo = np.array([
    jdict[jn]["Range"][0] if jdict[jn]["Limited"] else -np.inf
    for jn in joint_names
])
jnt_hi = np.array([
    jdict[jn]["Range"][1] if jdict[jn]["Limited"] else np.inf
    for jn in joint_names
])

# After each iteration
for j, qadr in enumerate(qpos_adrs):
    qpos[qadr] = np.clip(qpos[qadr] + dq[j] * STEP, jnt_lo[j], jnt_hi[j])
```

---

## Parameter Quick Reference

| Parameter | Purpose | Recommended Value |
|------|------|--------|
| `DAMPING` | Damping coefficient — larger = more stable, smaller = faster convergence | 0.01–0.1 |
| `STEP` | Max joint change per step (radians) | 0.02–0.1 |
| `ITERS` | Max iteration count | 50–200 |
| `ATOL` | Convergence threshold (m) | 0.01–0.05 |

## Common Issues

| Symptom | Cause | Solution |
|------|------|------|
| IK does not converge | Damping too small / target too far | Increase DAMPING, reduce target offset |
| Joint bends in wrong direction | Inappropriate starting pose | Preset a slight squat pose |
| Jacobian is all zeros | No `mj_forward` / wrong body name | Ensure forward + include agent prefix |
| IK abnormal in multi-body scenes | Wrong dof column range | Use `jnt_dofadr` to get [min, max] |
| Pose incorrect after `mj_forward()` | `set_joint_qpos` received incomplete qpos | Pass a full array of length `nq` |

---

## Next Steps

Now that you understand state control and IK, learn how to **make G1 walk**: [🦿 Joint Control](../robot_control/joint-control.md).
