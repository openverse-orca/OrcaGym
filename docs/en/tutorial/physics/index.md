# ⚛️ Physics Simulation

OrcaGym's physics simulation is based on the MuJoCo engine, providing high-precision rigid body dynamics and constraint solving.

> See [OrcaPlayground examples/euler/](https://github.com/OrcaGym/OrcaPlayground) for complete runnable code.

## Chapter Navigation

- [🔧 MuJoCo Backend](mujoco-backend.md) — Model loading, solver configuration, stepping control
- [📐 State Management](state-management.md) — qpos/qvel/qacc, state setting and reading
- [💥 Contacts and Forces](contacts-forces.md) — Contact detection, force querying, external force injection
- [🔄 Force Application and IK](force-apply.md) — External force application, state writing, Jacobians, inverse kinematics
- [🔗 Equality Constraints](equality-constraints.md) — WELD/CONNECT constraints, mocap anchoring, grasp operations
- [🧤 Soft Bodies and Flexibles](soft-bodies.md) — MuJoCo Flex system (experimental)

## Quick Reference

| Operation | API | Description |
|------|-----|------|
| Step + sync (recommended) | `env.do_simulation(ctrl, n)` | Atomic operation, auto-syncs data |
| Advance n physics steps | `env.mj_step(n)` | Execute n physics steps |
| Forward update | `env.mj_forward()` | Refresh derived quantities (body poses, sensors) |
| Sync view | `env._sync_view()` | Internal method; `do_simulation()` already auto-syncs, public users usually do not need to call this manually |
| Jacobian matrix | `env.mj_jacBody(jacp, jacr, name)` | Position/rotation Jacobian |
| Site Jacobian | `env.mj_jacSite(jacp, jacr, name)` | Site point Jacobian |
| Apply external force | `env.apply_body_force(name, f, τ)` | Apply force/torque to a body |
| Clear body force | `env.clear_body_force(name)` | Clear external force on a specific body |
| Clear all forces | `env.clear_all_forces()` | Clear all external forces |
| Set friction | `env.set_geom_friction({name: arr})` | Set geom friction coefficients |
| Set mocap | `env.set_mocap_pos_and_quat({name: {...}})` | Set mocap body pose |
| Set solver | `env.sim_config.timestep = 0.002` | Timestep configuration |

## Key Physics Engine Parameters

Configure via `env.sim_config`:

| Parameter | Default | Description |
|------|--------|------|
| `timestep` | 0.001 (G1) / 0.002 | Physics simulation step size (seconds) |
| `iterations` | 100 | Solver iteration count |
| `integrator` | 0 (Euler) | Integrator type (0=Euler, 1=RK4) |
| `gravity` | `[0, 0, -9.81]` | Gravity vector |
| `tolerance` | 1e-8 | Solver tolerance |

## G1 Humanoid Robot Standard Configuration

The G1 Euler examples (Lessons 4-9) use the following standard parameters:

| Parameter | Value | Description |
|------|-----|------|
| `time_step` | 0.001s | Physics step 1ms (MuJoCo 1000Hz) |
| `frame_skip` | 20 | 20 physics steps per control cycle |
| `dt` | 0.02s | Control frequency 50Hz |
| `integrator` | 0 (Euler) | Semi-implicit Euler integration |
