# 📖 API Reference

OrcaGym API documentation to help you quickly find the interfaces you need.

## Module Index

| Module | Description |
|------|------|
| [🧬 Core API](core.md) | Simulation core, model, state, solver configuration |
| [🌍 Environment API](environment.md) | Gymnasium environment base class and methods |
| [🎬 Scene API](scene.md) | Scene management, Actor, lights, materials |
| [📷 Sensor API](sensor.md) | Camera, sensor data acquisition |
| [🔧 Utils API](utils.md) | Inverse kinematics, joint control, rotation utilities |

## Quick Navigation

### Getting Started

1. Start with `OrcaGymEulerEnv` in the [Environment API](environment.md) — this is the main entry point for writing environments
2. Learn about Model, Data, and SimConfig in the [Core API](core.md)
3. Check out controllers and rotation utilities in the [Utils API](utils.md)

### Find by Task

| Task | Related API |
|------|----------|
| Create robot training environment | `OrcaGymEulerEnv` |
| Read simulation state | `env.data` → `env.data.qpos` / `env.data.body_xpos(name)` |
| Set solver parameters | `env.sim_config` → `env.sim_config.timestep = 0.002` |
| Query body/site state | `env.query_*()` / `env.get_body_*()` |
| Apply external force | `env.apply_body_force(name, force, torque)` |
| Grasp/drag objects | Mocap + equality constraints |
| Camera image capture | `start_streaming` / `show_camera` / `CameraWrapper` |
| Place objects in scene | `OrcaGymScene` |
| Inverse kinematics control | `InverseKinematicsController` |
| Joint torque control | `JointController` |
| Rotation/pose conversion | `rotations` |
| Record video | `save_streaming` / `start_streaming` |

## Key Concepts Quick Reference

| Concept | Description | See |
|------|------|------|
| **Body** | Rigid body, the basic unit of physics simulation | [Core API](core.md) |
| **Joint** | Joint, a constraint connecting bodies | [Core API](core.md) |
| **Actuator** | Actuator, the element that drives the robot | [Core API](core.md) |
| **Geom** | Geometry, a collision detection shape | [Core API](core.md) |
| **Site** | Marker point, does not participate in physics simulation | [Core API](core.md) |
| **Sensor** | Sensor, measures physical quantities | [Sensor API](sensor.md) |
| **Mocap Body** | Freely movable virtual body | [Core API](core.md) |
| **Equality Constraint** | Equality constraint, connects two bodies | [Core API](core.md) |
| **qpos/qvel/qacc** | Generalized coordinates/velocity/acceleration | [Core API](core.md) |
| **Frame Skip** | Number of physics steps per `step()` | [Environment API](environment.md) |

## API Usage Conventions

The following are recommended practices for using the OrcaGym API:

| ✅ Recommended | ❌ Avoid |
|---------|--------|
| `env.data.qpos` | Direct access to MuJoCo internal data structures |
| `env.data.body_xpos("link1")` | Access body via internal ID |
| `env.sim_config.timestep = 0.002` | Directly modify raw solver parameters |
| `env.apply_body_force("link1", f, tau)` | Directly write to external force arrays |
| `env.do_simulation(ctrl, n_frames)` | Manual stepping and syncing |
| `env.query_joint_qpos([...])` | Indirect access to internal data structures |
