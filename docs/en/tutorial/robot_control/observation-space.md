# 👁️ Observation Space

The observation space defines the information that an agent perceives from the environment.

## Building Observations

### Basic Observation (Joint State)

```python
def _get_obs(self):
    qpos = self.data.qpos.copy()
    qvel = self.data.qvel.copy()
    return np.concatenate([qpos, qvel]).astype(np.float32)
```

### Extended Observation (+ Sensors)

```python
def _get_obs(self):
    qpos = self.data.qpos.copy()
    qvel = self.data.qvel.copy()
    
    # IMU data
    imu = self.query_sensor_data(["imu_acc", "imu_gyro"])
    
    # End-effector pose (returns flat arrays: xpos, xmat, xquat)
    xpos, xmat, xquat = self.get_body_xpos_xmat_xquat(["ee_link"])
    ee_pos = xpos[:3]  # First 3 elements are ee_link position
    
    return np.concatenate([
        qpos, qvel, 
        imu["imu_acc"], imu["imu_gyro"],
        ee_pos
    ]).astype(np.float32)
```

### Dictionary Observation (Multimodal)

```python
def _get_obs(self):
    return {
        "proprio": np.concatenate([
            self.data.qpos.copy(), 
            self.data.qvel.copy()
        ]).astype(np.float32),
        "vision": self.get_camera_image("front_camera"),  # Requires custom camera capture
        "force": self.query_sensor_data(["ft_sensor"])["ft_sensor"],
    }
```

## Automatic Observation Space Inference

```python
# Automatically inferred on first reset
obs = self._get_obs()
self.observation_space = self.generate_observation_space(obs)

# For numpy observations → spaces.Box
# For dict observations → spaces.Dict
```

## Observation Normalization

```python
class MyEnv(OrcaGymEulerEnv):
    def __init__(self, ...):
        super().__init__(...)
        
        # Define normalization parameters
        self._obs_mean = np.array([...])  # Task-dependent
        self._obs_std = np.array([...])
    
    def _get_obs(self):
        raw_obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ])
        return ((raw_obs - self._obs_mean) / self._obs_std).astype(np.float32)
```

## Observation Construction Best Practices

1. **Use `copy()`** to avoid subsequent updates overwriting data
2. **Keep float32** for PyTorch/TensorFlow compatibility
3. **Fixed observation dimensions** — shape must not change during inference
4. **Consider historical information** — may need to stack multiple frames of observations
