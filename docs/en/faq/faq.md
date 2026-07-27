# ❓ FAQ

## Installation and Configuration

### Q: Importing `orca_gym` fails after installation?

```bash
# Verify installation
pip show orca-gym

# If not found, reinstall
pip install orca-gym --force-reinstall
```

### Q: MuJoCo cannot find the GLFW library?

```bash
# Linux
sudo apt-get install libglfw3 libglew2.2 libosmesa6

# macOS
brew install glfw glew
```

### Q: How to connect to OrcaStudio/OrcaLab?

1. Download and install from [orca3d.cn](http://orca3d.cn/)
2. Open the software and click the "Run" button
3. The default gRPC address is `localhost:50051`

## Running Simulations

### Q: Simulation fails to start with "Load local env failed"?

Common causes:
1. Scene configuration errors (duplicate joint/body names)
2. Overlapping initial model poses
3. Missing mesh/texture resources -- wait a few seconds and retry
4. OrcaStudio/OrcaLab is not running properly

### Q: Data shows NaN?

```python
# Check if mj_forward() was called after modifying the state
env.set_joint_qpos(...)
env.mj_forward()   # <-- This step is required

# Sync to DataView
env._sync_view()

# Now reading data will return correct values
print(env.data.qpos)
```

### Q: Reading old data after stepping?

`do_simulation()` internally auto-syncs data; after it returns, `env.data` contains the latest state. If you manually modified the state, remember to call `mj_forward()` and `_sync_view()`.

### Q: How to improve simulation speed?

1. Set `render_mode="none"`
2. Increase `timestep` (e.g., 0.002 or 0.005)
3. Reduce `frame_skip`
4. Use vectorized environments
5. Simplify collision geometries

### Q: How to choose between remote mode and local mode?

| Scenario | Recommended Mode | Reason |
|----------|-----------------|--------|
| Development & debugging | Local | Direct MuJoCo connection, no network latency |
| Single-machine training | Local + vectorized | Multi-process parallelism |
| Large-scale distributed | Remote | Simulation on server side, training on client side |
| Requires PhysX backend | Remote | PhysX is only available on the server side |

## Environment Development

### Q: How to customize an environment?

Inherit from `OrcaGymEulerEnv` and implement the `step()`, `reset_model()`, and `_get_obs()` methods.

```python
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

class MyEnv(OrcaGymEulerEnv):
    def step(self, action): ...
    def reset_model(self): ...
    def _get_obs(self): ...
```

### Q: Where does the action_space dimensionality come from?

It comes from the number of actuators in the model (`model.nu`):

```python
print(f"Number of actuators: {env.model.nu}")
print(f"Action space: {env.action_space}")
# Box(low=-1.0, high=1.0, shape=(nu,), float32)
```

### Q: How to add sensors to observations?

```python
def _get_obs(self):
    # Joint state
    proprio = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    # Sensor data
    sensors = self.query_sensor_data(["imu_acc", "imu_gyro"])

    return np.concatenate([
        proprio,
        sensors["imu_acc"],
        sensors["imu_gyro"],
    ]).astype(np.float32)
```

## Migration

### Q: Key differences when migrating from Isaac Gym?

| Isaac Gym | OrcaGym |
|-----------|---------|
| `VecEnv` interface | `Gymnasium.Env` interface |
| PyTorch Tensor batch operations | NumPy Array per-instance operations |
| GPU single-process 4096 envs | Multi-process vectorization |
| RSL-RL | Stable-Baselines3 / RLlib |
| PhysX | MuJoCo (local) |

### Q: Migrating from native MuJoCo environments?

1. Change the base class to `OrcaGymEulerEnv`
2. Use `env.data` instead of directly accessing MuJoCo internal data
3. Use `env.sim_config` instead of directly accessing solver parameters
4. Use `env.apply_body_force()` instead of directly writing external forces

## Other

### Q: How to contribute?

See the [Contributing Guide](https://github.com/openverse-orca/OrcaGym#贡献).

### Q: How to get help?

Please submit issues via [GitHub Issues](https://github.com/openverse-orca/OrcaGym/issues).
