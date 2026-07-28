# 👁️ 观测空间

观测空间定义了智能体从环境中感知到的信息。

## 构建观测

### 基本观测（关节状态）

```python
def _get_obs(self):
    qpos = self.data.qpos.copy()
    qvel = self.data.qvel.copy()
    return np.concatenate([qpos, qvel]).astype(np.float32)
```

### 扩展观测（+ 传感器）

```python
def _get_obs(self):
    qpos = self.data.qpos.copy()
    qvel = self.data.qvel.copy()
    
    # IMU 数据
    imu = self.query_sensor_data(["imu_acc", "imu_gyro"])
    
    # 末端位姿（返回 flat 数组: xpos, xmat, xquat）
    xpos, xmat, xquat = self.get_body_xpos_xmat_xquat(["ee_link"])
    ee_pos = xpos[:3]  # 前 3 个元素为 ee_link 的位置
    
    return np.concatenate([
        qpos, qvel, 
        imu["imu_acc"], imu["imu_gyro"],
        ee_pos
    ]).astype(np.float32)
```

### 字典观测（多模态）

```python
def _get_obs(self):
    return {
        "proprio": np.concatenate([
            self.data.qpos.copy(), 
            self.data.qvel.copy()
        ]).astype(np.float32),
        "vision": self.get_camera_image("front_camera"),  # 需要自定义相机采集
        "force": self.query_sensor_data(["ft_sensor"])["ft_sensor"],
    }
```

## 自动推断观测空间

```python
# 首次 reset 时自动推断
obs = self._get_obs()
self.observation_space = self.generate_observation_space(obs)

# 对 numpy 观测 → spaces.Box
# 对 dict 观测 → spaces.Dict
```

## 观测归一化

```python
class MyEnv(OrcaGymEulerEnv):
    def __init__(self, ...):
        super().__init__(...)
        
        # 定义归一化参数
        self._obs_mean = np.array([...])  # 根据任务确定
        self._obs_std = np.array([...])
    
    def _get_obs(self):
        raw_obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ])
        return ((raw_obs - self._obs_mean) / self._obs_std).astype(np.float32)
```

## 观测构建最佳实践

1. **使用 `copy()`** 避免后续更新覆盖数据
2. **保持 float32** 以兼容 PyTorch/TensorFlow
3. **固定观测维度** — 推理时不能改变 shape
4. **考虑历史信息** — 可能需要堆叠多帧观测
