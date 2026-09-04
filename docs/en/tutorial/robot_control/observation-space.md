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

    # End-effector pose (returns a nested dict: body_name -> {"xpos", "xmat", "xquat"})
    result = self.get_body_xpos_xmat_xquat(["ee_link"])
    ee_pos = result["ee_link"]["xpos"]  # (3,) end-effector position

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
        # Camera image: first start streaming via start_streaming, then get frames through VideoRecorderManager
        # "vision": env.get_recorder_manager().get_frame(...),  # see the camera streaming section
        "force": self.query_sensor_data(["ft_sensor"])["ft_sensor"],
    }
```

> Note: camera image capture requires first starting the stream via `env.start_streaming(camera_name, ...)`,
> then obtaining the `VideoRecorderManager` through `env.get_recorder_manager()` to read frames.
> There is no `get_camera_image` method. See [API Reference · Sensors](../../api_reference/sensor.md).

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
