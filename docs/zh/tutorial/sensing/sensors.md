# 🖲️ 传感器系统

OrcaGym 封装了 MuJoCo 的传感器系统，提供便捷的传感器查询接口。

## 传感器模型

```python
# 查看所有传感器
sensor_dict = env.model.gen_sensor_dict()
for name, info in sensor_dict.items():
 print(f"{name}: type={info['Type']}, dim={info['Dim']}, adr={info['Adr']}")
```

## 查询传感器数据

```python
# 按名称查询（通用 env.query_sensor_data）
sensor_data = env.query_sensor_data([
 "imu_accelerometer",
 "imu_gyro", 
 "force_torque_sensor",
 "joint_pos_sensor",
])

accel = sensor_data["imu_accelerometer"] # (3,) 加速度
gyro = sensor_data["imu_gyro"] # (3,) 角速度
ft = sensor_data["force_torque_sensor"] # (6,) 力+扭矩
joint_pos = sensor_data["joint_pos_sensor"] # (1,) 关节角度
```

## 传感器数据布局

每个传感器在 MuJoCo 的 `sensordata` 数组中占据连续的一段：

```
sensordata: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ft_fx, ft_fy, ...]
 └─── accelerometer ───┘ └────── gyro ───────┘ └─ force_torque ─┘
 adr=0, dim=3 adr=3, dim=3 adr=6, dim=6
```

## 常用传感器使用

### IMU (惯性测量单元)

```python
def read_imu(env):
 imu_data = env.query_sensor_data(["imu_acc", "imu_gyro"])
 acc = imu_data["imu_acc"] # 线加速度 (body frame)
 gyro = imu_data["imu_gyro"] # 角速度 (body frame)
 return acc, gyro
```

### 力/扭矩传感器

```python
def read_wrist_ft(env):
 ft_data = env.query_sensor_data(["wrist_force_torque"])
 ft = ft_data["wrist_force_torque"] # [fx, fy, fz, mx, my, mz]
 force = ft[:3]
 torque = ft[3:]
 return force, torque
```

### 关节位置/速度传感器

```python
def read_joint_sensors(env):
 pos = env.query_sensor_data(["joint0_pos", "joint1_pos", "joint2_pos"])
 vel = env.query_sensor_data(["joint0_vel", "joint1_vel", "joint2_vel"])
 return pos, vel
```

## 传感器数据的更新时机

⚠️ 传感器数据在 `mj_forward()` 或 `mj_step()` 后才会更新。

```python
# ✅ 正确 — do_simulation 自动 step + sync_to_view
env.do_simulation(ctrl, n_frames)
sensor = env.query_sensor_data(...)     # 读最新值

# ✅ 正确 — 手动步进
env.mj_step(nstep)                      # step 包含 forward
env._sync_view()
sensor = env.query_sensor_data(...)

# ❌ 错误 — 修改状态后没有 forward
env.set_joint_qpos(...)
sensor = env.query_sensor_data(...)     # 旧数据！还没 forward
```

## 传感器噪声

MuJoCo 传感器支持内置噪声：

```python
# 查看传感器噪声设置
for name, info in env.model.gen_sensor_dict().items():
 if info['Noise'] > 0:
 print(f"{name}: 噪声标准差 = {info['Noise']}")
```
