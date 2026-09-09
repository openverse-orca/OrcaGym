# 🐋 What is OrcaGym

OrcaGym is an **open-source, cloud-native robot simulation platform** that provides robot simulation environments that are fully compatible with the OpenAI Gym / Gymnasium interface.

## In a Nutshell

**OrcaGym = Gymnasium API + Dual Physics Backends (MuJoCo / Euler) + Distributed Communication**

## Core Positioning

Traditional robot simulation solutions often face trade-offs between fidelity and computational efficiency. OrcaGym bridges this gap through:

1. **Standardized Interface**: Fully compatible with the Gymnasium API, enabling zero-cost migration of existing RL algorithms
2. **Dual Physics Backends**: The core package supports two mutually independent paths — MuJoCo (open-source, CPU, pure rigid-body) and Euler (in-house, GPU) — selected via `SimConfig.backend`
3. **Cloud-Native Architecture**: Enables hybrid local/remote deployment
4. **Extensible Rendering**: High-quality rendering such as ray tracing can be accessed via external visualization tools (e.g., OrcaStudio/OrcaLab)

## Key Features

### 🎮 Gymnasium API Compatibility

```python
import gymnasium as gym

env = gym.make("YourEnv-v0", frame_skip=5, orcagym_addr="localhost:50051")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

Seamlessly integrates with mainstream RL libraries such as Stable-Baselines3, RLlib, and CleanRL.

### ⚡ Dual Physics Backends

OrcaGym adopts a dual-backend architecture with two mutually independent paths, selected via `SimConfig.backend`:

| Backend | Positioning | Characteristics | Use Case |
|---------|-------------|-----------------|----------|
| **MuJoCo** | Open-source standard path | High-precision rigid-body dynamics (CPU) | Legged robots, robotic arm manipulation, rapid prototyping |
| **Euler** | In-house path (Orca team) | GPU accelerated | Large-scale parallel training, high-fidelity simulation |

> **Current implementation status**: The MuJoCo backend is fully functional; the Euler backend is a reserved interface (`_euler=None`), integration plan TBD.

### 🌐 Distributed Deployment

Supports flexible deployment ranging from local development to large-scale remote training:
- **Local Mode**: Run MuJoCo directly within the Python process, suitable for development and debugging
- **Remote Mode**: Connect to a remote server for physics computation and rendering, suitable for large-scale training

### 📷 Sensors and Perception

- IMU (accelerometer, gyroscope)
- Force/torque sensors
- RGB-D cameras
- Contact force sensors

### 🤖 Multi-Agent Support

Natively supports homogeneous/heterogeneous multi-agent scenarios, where agents can operate independently or collaboratively.

## License

MIT License — Fully open source, free for both academic and commercial use.
