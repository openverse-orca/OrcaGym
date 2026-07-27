# 🧠 Robomimic Adapter

OrcaGym provides a Robomimic adapter for imitation learning.

## Environment Adaptation

```python
from orca_adapters.robomimic import robomimic_env

class MyTask(RobomimicEnv):
 """Create a Robomimic-compatible environment"""
 def __init__(self, ...):
 super().__init__(...)
```

## Dataset Utilities

```python
from orca_adapters.robomimic import dataset_util

# Process HDF5 datasets
```

## Task Definition

```python
# Custom task class
from orca_adapters.robomimic.task import ...

class MyRobomimicTask(RobomimicEnv):
 # Implement the interface required by Robomimic
 pass
```
