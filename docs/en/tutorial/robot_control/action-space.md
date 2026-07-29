# 🎯 Action Space

The action space defines the control commands that an agent can output to the environment.

## Automatic Generation

OrcaGym automatically generates the action space from MuJoCo actuator control ranges:

```python
class MyEnv(OrcaGymEulerEnv):
    def __init__(self, ...):
        super().__init__(...)
        
        # Method 1: Automatic generation
        ctrl_range = self.model.get_actuator_ctrlrange()  # (nu, 2)
        self.action_space = self.generate_action_space(ctrl_range)
        
        # Method 2: Manual definition
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )
```

## Action Space Types

| Type | Description | Use Case |
|------|-------------|----------|
| Position Control | Action = target joint position | Robot arms |
| Velocity Control | Action = target joint velocity | Mobile robots |
| Torque Control | Action = target torque | Direct force control |
| Incremental Control | Action = offset from current position | Fine manipulation |

## Understanding Action Dimensions

```python
# Information for each actuator
actuator_dict = env.model.get_actuator_dict()

for name, info in actuator_dict.items():
    print(f"{name}:")
    print(f"  Type: {info['TrnType']}")
    print(f"  Ctrl Range: {info['CtrlRange']}")
    print(f"  Force Range: {info['ForceRange']}")
    print(f"  Gear Ratio: {info['GearRatio']}")
    print(f"  Associated Joint: {info['JointName']}")
```

## Action Scaling

```python
def scale_action(action, low, high):
    """Scale action from [-1, 1] to [low, high]"""
    return low + (action + 1.0) * 0.5 * (high - low)

# Scale using the actual actuator ranges
ctrl_range = env.model.get_actuator_ctrlrange()
low = ctrl_range[:, 0]
high = ctrl_range[:, 1]
scaled_action = scale_action(normalized_action, low, high)
```

## Action Space and step()

```python
# step() accepts actions matching action_space
action = env.action_space.sample()     # Correct: from action_space
obs, reward, _, _, _ = env.step(action)

# Or pass ctrl directly
ctrl = np.zeros(env.model.nu)           # Custom
env.do_simulation(ctrl, env.frame_skip) # Direct control
```
