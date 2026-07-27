# 🛠️ Installation Guide

## System Requirements

| Dependency | Version Requirement |
|------------|---------------------|
| Python | >= 3.9 (recommended 3.12) |
| pip | >= 21.0 |
| Operating System | Ubuntu 20.04+ / Windows 10+ / macOS 12+ |

## Install from PyPI (Recommended)

```bash
# Install core package
pip install orca-gym

# Or install with optional dependencies
pip install orca-gym[rl]          # Reinforcement learning training (Stable-Baselines3, etc.)
pip install orca-gym[imitation]   # Imitation learning
pip install orca-gym[devices]     # Input device support
pip install orca-gym[sensors]     # Cameras and sensors
pip install orca-gym[all]         # All optional dependencies
```

## Optional Dependencies

| Group | Contents | Use Case |
|-------|----------|----------|
| `[rl]` | stable-baselines3, torch | RL training |
| `[imitation]` | robomimic related | Imitation learning |
| `[devices]` | pygame, inputs | Gamepad/keyboard control |
| `[sensors]` | opencv-python, av, websockets | Camera vision |

## Configure OrcaStudio / OrcaLab

Download and install [OrcaStudio](http://orca3d.cn/) or OrcaLab to access:

- Visual scene editing
- Remote simulation service
- Multi-physics backend support

## Verify Installation

```bash
# Check import
python -c "import orca_gym; print('OrcaGym installed successfully!')"

# Check version
python -c "import orca_gym; print(orca_gym.__version__)"
```

## Common Installation Issues

### Issue: MuJoCo import failure

```bash
# Ensure mujoco is installed
pip install mujoco>=3.3.0

# Linux users may need to install additional system dependencies
sudo apt-get install libglfw3 libglew2.2 libosmesa6
```

### Issue: gRPC version conflict

```bash
# Reinstall
pip install grpcio grpcio-tools --force-reinstall
```

### Issue: mesh/hfield resource download failure

This is normal — mesh and texture files are downloaded on demand when the simulation first starts. Just ensure a stable network connection.
