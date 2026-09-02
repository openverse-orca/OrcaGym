# 🔧 MuJoCo Backend

This document describes how to use OrcaGym's **MuJoCo backend**. The MuJoCo backend is the open-source standard path (CPU, pure rigid-body) in OrcaGym's dual-backend architecture, selected via `SimConfig.backend="mujoco"`. Model loading is automatically completed during environment initialization.

> See [OrcaPlayground examples/euler/](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/euler) for complete runnable code.

## Model Loading

Model loading is automatically handled during environment initialization — no manual steps required:

```python
# Offline mode: load from a local XML file
env = MyEnv(
    model_xml_path="path/to/scene.xml",
    skip_grpc_load=True,   # True = offline mode, loads local XML directly
)

# Online mode: fetch model XML from OrcaStudio via gRPC
env = MyEnv(
    model_xml_path="path/to/scene.xml",
    skip_grpc_load=False,  # False (default) = online mode
)
```

Internal flow:
1. Load XML → create MuJoCo instance (`mjModel` + `mjData`)
2. Query and populate Model / Data information
3. Initialize all dictionaries (body, joint, actuator, ...)
4. Cache initial state into `init_qpos` / `init_qvel`

## Resource Caching

Mesh and hfield files that MuJoCo models depend on are cached in the `~/.orcagym/tmp/` directory.

## Simulation Control

### Stepping Control

```python
# ✅ Recommended: do_simulation (atomic operation, auto-syncs data)
env.do_simulation(ctrl, n_frames=20)
# Equivalent to: set_ctrl → mj_step(20) → _sync_view

# Manual control (requires manual syncing)
env.set_ctrl(ctrl)
env.mj_step(nstep=20)
env._sync_view()           # Sync state view

# Forward computation (refreshes derived quantities, does not advance time)
env.mj_forward()

# Pure MuJoCo stepping
env.mj_step(nstep=20)
```

### ctrl Setting

```python
# ctrl is the model actuator control input, length = model.nu
ctrl = np.zeros(env.model.nu, dtype=np.float64)
env.do_simulation(ctrl, env.frame_skip)
```

## Solver Configuration

Read and write solver parameters via `env.sim_config`:

```python
# Read/write individual parameters
env.sim_config.timestep = 0.002
env.sim_config.iterations = 100
env.sim_config.integrator = 0       # 0=Euler, 1=RK4
env.sim_config.gravity = np.array([0., 0., -9.81])

# Batch configuration
env.sim_config.load_from_dict({
    "integrator": 0,
    "iterations": 100,
    "tolerance": 1e-8,
})

# Export configuration
config = env.sim_config.to_dict()
```

### Key Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `timestep` | float | 0.002 | Physics step size (seconds) |
| `iterations` | int | 100 | Solver iteration count |
| `integrator` | int | 0 | 0=Euler, 1=RK4 |
| `gravity` | ndarray | [0,0,-9.81] | Gravitational acceleration |
| `tolerance` | float | 1e-8 | Solver tolerance |

## Timestep vs Control Frequency

```
Physics step (timestep)    = 0.001 s  # Time per physics engine step (default)
Control step (frame_skip)  = 20       # How many physics steps per step() call
Environment step (dt)      = 0.02 s   # timestep × frame_skip
Control frequency          = 50 Hz    # 1 / dt
```

```python
print(f"Physics timestep: {env.sim_config.timestep:.4f}s")
print(f"frame_skip: {env.frame_skip}")
print(f"Control step: {env.dt:.4f}s")
print(f"Control frequency: {1.0/env.dt:.1f}Hz")
```

### G1 Standard Configuration

The G1 humanoid robot uses the following standard parameters:

| Parameter | Value | Description |
|------|-----|------|
| `time_step` | 0.001 | Physics step 1ms |
| `frame_skip` | 20 | 20 physics steps per control cycle |
| `dt` | 0.02s | Control frequency 50Hz |

## Debugging and Profiling

```python
# View contact count (use query_contact_simple to get the contact list length)
contacts = env.query_contact_simple()
print(f"Contact count: {len(contacts)}")

# View model information (OrcaGymModel does not expose nbody/njnt/nsite;
# use len() to get dictionary sizes; ngeom/nq/nv/nu are fields in model_info)
print(f"nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")
print(f"ngeom={env.model.ngeom}")
print(f"nbody={len(env.model.get_body_names())}")
print(f"njnt={len(env.model.get_joint_dict())}")
```

---

## Next Step

Now that you understand backend configuration, learn how to **write an environment class** to control this scene: [🏗️ Your First Environment](your-first-env.md).
