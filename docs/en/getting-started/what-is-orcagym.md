# 🐋 What is OrcaGym

OrcaGym is an **open-source, cloud-native robot simulation platform** that provides robot simulation environments that are fully compatible with the OpenAI Gym / Gymnasium interface.

## In a Nutshell

**OrcaGym = Gymnasium API + MuJoCo Physics Engine + Distributed Communication + OrcaStudio/OrcaLab Cloud Platform**

## Core Positioning

Traditional robot simulation solutions often face trade-offs between fidelity and computational efficiency. OrcaGym bridges this gap through:

1. **Standardized Interface**: Fully compatible with the Gymnasium API, enabling zero-cost migration of existing RL algorithms
2. **Multiple Physics Backends**: The core package locally supports MuJoCo/MuJoCoWarp/Euler; PhysX and ODE can be accessed via OrcaStudio/OrcaLab
3. **Cloud-Native Architecture**: Enables hybrid local/remote deployment
4. **Realistic Rendering**: Ray tracing provides high-quality observations for visual RL tasks (provided by the OrcaStudio/OrcaLab server)

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

### ⚡ Multiple Physics Backends

| Backend | Availability | Characteristics | Use Case |
|---------|--------------|-----------------|----------|
| **MuJoCo** | Core package (local) | High-precision rigid-body dynamics (CPU) | Legged robots, robotic arm manipulation |
| **MuJoCoWarp** | Core package (local) | GPU (CUDA) accelerated parallelism | Large-scale parallel training |
| **Euler** | Core package (local) | Multi-vendor GPU multi-physics engine | High-fidelity multi-physics simulation |
| **PhysX** | Remote only (OrcaStudio/OrcaLab) | GPU accelerated, massively parallel | Swarm simulation, complex scenes |
| **ODE** | Remote only (OrcaStudio/OrcaLab) | Open-source, general-purpose | Rapid prototyping, educational use |

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
