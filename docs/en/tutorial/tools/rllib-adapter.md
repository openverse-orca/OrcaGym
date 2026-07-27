# 🤖 RLlib Adapter

OrcaGym provides an RLlib adapter that supports distributed RL training.

## Integration

```python
from orca_adapters.rllib import appo_catalog

# Train using the APPO algorithm
```

### Metrics Callback

```python
from orca_adapters.rllib import metrics_callback
```

## Configuration

Specify the OrcaGym environment in your RLlib configuration:

```python
config = {
 "env": "YourOrcaGymEnv-v0",
 "env_config": {
 "frame_skip": 20,
 "orcagym_addr": "localhost:50051",
 "agent_names": ["agent0"],
 "time_step": 0.001,
 },
 # RLlib-specific configuration...
}
```

## Training Example

The OrcaGym + RLlib training loop uses RLlib's standard workflow; simply register the environment as an OrcaGym environment.
