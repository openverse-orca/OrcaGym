# 🧬 Why Choose OrcaGym

Among the many robot simulation platforms, what are OrcaGym's unique advantages?

## Comparison with Mainstream Simulation Platforms

| Feature | OrcaGym | Isaac Gym | MuJoCo (Native) | PyBullet | SAPIEN |
|---------|---------|-----------|-----------------|----------|--------|
| Gymnasium API | ✅ Fully compatible | ❌ Custom VecEnv | Requires manual wrapping | ✅ | ❌ |
| Multi-physics backend | ✅ MuJoCo/PhysX/ODE | ❌ PhysX only | ❌ MuJoCo only | ❌ Bullet only | ❌ PhysX only |
| Distributed deployment | ✅ Native support | ❌ Single machine | ❌ Single machine | ❌ Single machine | ❌ Single machine |
| Ray tracing | ✅ | ❌ | ❌ | ❌ | ✅ |
| GPU acceleration | ✅ (via backend) | ✅ (native) | ❌ (CPU) | ❌ (CPU) | ✅ |
| Multi-agent | ✅ Native | ⚠️ Manual setup | ⚠️ Manual setup | ⚠️ Manual setup | ✅ |
| Visual editor | ✅ OrcaStudio | ❌ | ❌ | ❌ | ❌ |
| Open source | ✅ MIT | ✅ Non-commercial | ✅ Apache 2.0 | ✅ | ✅ |

## Core Advantages in Detail

### 1. Standardized RL Interface

OrcaGym strictly follows the Gymnasium specification, which means:

- Existing RL algorithm libraries can run **without modification**
- `env.step()`, `env.reset()`, and other calls behave exactly as expected
- Supports both `Dict` and `Box` observation spaces

```python
# Use with any Gymnasium-compatible RL library
from stable_baselines3 import PPO

env = gym.make("YourEnv-v0", ...)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

### 2. Cloud-Native Distributed Architecture

Unlike single-machine simulators, OrcaGym natively supports:

- **Local Mode**: Drive MuJoCo directly within the Python process, suitable for development and debugging
- **Remote Mode**: Python client connects to remote OrcaStudio/OrcaLab, suitable for large-scale training
- **Hybrid Mode**: Training on remote, policy execution locally

```
Development: Local Mode → Rapid iteration
Deployment: Remote Mode → Elastic scaling
```

### 3. Visualization and Debugging Ecosystem

OrcaGym is deeply integrated with OrcaStudio/OrcaLab, providing:

- Real-time 3D scene visualization
- Interactive object drag-and-drop manipulation
- Real-time joint/sensor data monitoring
- Video recording and playback

## Use Cases

| Scenario | Why Recommended |
|----------|-----------------|
| **Legged Robot Control** | High-precision contact model + standard RL interface |
| **Robotic Arm Manipulation** | Inverse kinematics + equality constraints + Mocap control |
| **Multi-Agent Collaboration** | Native multi-agent + asynchronous environments |
| **Visual RL** | Ray tracing + RGB-D sensors |
| **Large-Scale Distributed Training** | Multi-node scaling |
| **Robotics Education** | Standardized API + rich examples |

## Limitations

- The core package is CPU-driven (MuJoCo backend); GPU acceleration requires OrcaStudio/OrcaLab
- Remote mode depends on the OrcaStudio/OrcaLab server
- The community is still in its early stages, with fewer third-party examples
