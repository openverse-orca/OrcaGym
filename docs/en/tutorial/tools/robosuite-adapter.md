# 🔧 Robosuite Adapter

OrcaGym provides a Robosuite adapter for robot manipulation tasks.

## Module Structure

```
orca_gym/adapters/robosuite/
├── __init__.py
├── macros.py # Macro definitions and constants
├── controllers/ # Controllers (OSC/IK/joint position/torque/velocity, etc.)
│   ├── config/ # Controller configuration JSON
│   ├── interpolators/ # Interpolators
│   ├── base_controller.py
│   ├── controller_factory.py
│   ├── ik.py
│   ├── joint_pos.py
│   ├── joint_tor.py
│   ├── joint_vel.py
│   └── osc.py
├── devices/ # Input devices (keyboard/spacemouse, etc.)
│   ├── device.py
│   ├── keyboard.py
│   └── spacemouse.py
└── utils/
    ├── control_utils.py # Control utilities
    ├── errors.py # Error definitions
    ├── binding_utils.py # Binding utilities
    ├── robot_utils.py # Robot utilities
    ├── placement_samplers.py # Object placement sampling
    └── log_utils.py # Logging utilities
```

## Usage

The Robosuite adapter provides a set of utility functions for:

- Robot model binding
- Object placement strategies
- Control signal generation
- Error handling

```python
from orca_gym.adapters.robosuite import macros
from orca_gym.adapters.robosuite.utils import control_utils, robot_utils

# Use Robosuite-style tools
```

## Relationship with Robomimic

The Robosuite adapter is typically used in conjunction with the Robomimic adapter:

- **Robosuite** → Environment abstraction and tools
- **Robomimic** → Datasets and algorithms
- **OrcaGym** → Underlying simulation engine
