# 🎮 Robot Control

OrcaGym provides a rich set of robot control interfaces, from low-level joint control to high-level inverse kinematics.

> For complete runnable code, see [OrcaPlayground examples/euler/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler) (Lessons 4-9).

## Control Hierarchy

```
High-Level API
  ├── Damped Least Squares IK      ── end-effector pose → joint positions (with limit clamping)
  ├── PD Controller                ── target position → control torque
  ├── ONNX Walking Policy          ── observation → target joint angles (G1 walking)
  └── Mocap Control                ── directly set mocap pose + WELD constraint driving

Low-Level API
  ├── do_simulation()              ── set ctrl + step + auto-sync (recommended)
  ├── set_joint_qpos()             ── set joint positions
  ├── apply_body_force()           ── apply external force
  └── mj_forward()                 ── refresh derived quantities
```

## Chapter Navigation

- [📡 State Query API](state-queries-api.md) — Complete API for joints, bodies, sensors, and contact queries
- [🎯 Action Space](action-space.md) — Action space definition and types
- [👁️ Observation Space](observation-space.md) — Observation construction and normalization
- [🦿 Joint Control](joint-control.md) — set_ctrl, PD controller, low-pass filtering
- [🦾 Inverse Kinematics](inverse-kinematics.md) — Damped least squares IK + joint limits
- [🎭 Mocap Control](mocap-control.md) — Mocap body + WELD constraint = object dragging
- [🔄 External Force Application and IK](../physics/force-apply.md) — External force application + Jacobian + IK complete workflow
