# 🛠️ Installation Guide

## System Requirements

| Dependency | Version Requirement |
|------------|---------------------|
| Python | >= 3.10 (recommended 3.12) |
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
| `[rl]` | stable-baselines3, sb3_contrib, tensorboard | RL training |
| `[imitation]` | h5py, opencv-python, tqdm | Imitation learning |
| `[robomimic]` | h5py, termcolor, opencv-python | robomimic toolchain |
| `[devices]` | pygame | Gamepad/keyboard control |
| `[sensors]` | opencv-python, av, websockets, matplotlib, pillow | Camera vision |

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
pip show orca-gym
```

## Common Installation Issues

### Issue: MuJoCo import failure

```bash
# Ensure mujoco is installed (version pinned to match orca-gym)
pip install mujoco==3.7.0

# Linux users may need to install additional system dependencies
# Note: libglew package name varies by Ubuntu version
#   - Ubuntu 22.04+: libglew2.2
#   - Ubuntu 20.04: libglew2.1
sudo apt-get install libglfw3 libglew2.2 libosmesa6
```

### Issue: gRPC version conflict

```bash
# Reinstall
pip install grpcio grpcio-tools --force-reinstall
```

### Issue: mesh/hfield resource download failure

This is normal — mesh and texture files are downloaded on demand when the simulation first starts. Just ensure a stable network connection.
