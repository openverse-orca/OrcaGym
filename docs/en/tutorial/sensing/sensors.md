# 🖲️ Sensor System

OrcaGym wraps MuJoCo's sensor system, providing a convenient sensor query interface.

## Sensor Model

```python
# View all sensors
sensor_dict = env.model.gen_sensor_dict()
for name, info in sensor_dict.items():
    print(f"{name}: type={info['Type']}, dim={info['Dim']}, adr={info['Adr']}")
```

## Querying Sensor Data

```python
# Query by name (via the generic env.query_sensor_data)
sensor_data = env.query_sensor_data([
    "imu_accelerometer",
    "imu_gyro", 
    "force_torque_sensor",
    "joint_pos_sensor",
])

accel = sensor_data["imu_accelerometer"]  # (3,) acceleration
gyro = sensor_data["imu_gyro"]            # (3,) angular velocity
ft = sensor_data["force_torque_sensor"]   # (6,) force + torque
joint_pos = sensor_data["joint_pos_sensor"]  # (1,) joint angle
```

## Sensor Data Layout

Each sensor occupies a contiguous segment in MuJoCo's `sensordata` array:

```
sensordata: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ft_fx, ft_fy, ...]
             └─── accelerometer ───┘ └────── gyro ───────┘ └─ force_torque ─┘
             adr=0, dim=3             adr=3, dim=3          adr=6, dim=6
```

## Common Sensor Usage

### IMU (Inertial Measurement Unit)

```python
def read_imu(env):
    imu_data = env.query_sensor_data(["imu_acc", "imu_gyro"])
    acc = imu_data["imu_acc"]    # linear acceleration (body frame)
    gyro = imu_data["imu_gyro"]  # angular velocity (body frame)
    return acc, gyro
```

### Force/Torque Sensor

```python
def read_wrist_ft(env):
    ft_data = env.query_sensor_data(["wrist_force_torque"])
    ft = ft_data["wrist_force_torque"]  # [fx, fy, fz, mx, my, mz]
    force = ft[:3]
    torque = ft[3:]
    return force, torque
```

### Joint Position/Velocity Sensor

```python
def read_joint_sensors(env):
    pos = env.query_sensor_data(["joint0_pos", "joint1_pos", "joint2_pos"])
    vel = env.query_sensor_data(["joint0_vel", "joint1_vel", "joint2_vel"])
    return pos, vel
```

## Sensor Data Update Timing

⚠️ Sensor data is only updated after calling `mj_forward()` or `mj_step()`.

```python
# ✅ Correct — do_simulation automatically steps + sync_to_view
env.do_simulation(ctrl, n_frames)
sensor = env.query_sensor_data(...)     # read latest values

# ✅ Correct — manual stepping
env.mj_step(nstep)                      # mj_step includes the forward pass
env._sync_view()
sensor = env.query_sensor_data(...)

# ❌ Incorrect — no mj_forward() call after modifying state
env.set_joint_qpos(...)
sensor = env.query_sensor_data(...)     # stale data! mj_forward() hasn't been called
```

## Sensor Noise

MuJoCo sensors support built-in noise:

```python
# View sensor noise settings
for name, info in env.model.gen_sensor_dict().items():
    if info['Noise'] > 0:
        print(f"{name}: noise std dev = {info['Noise']}")
```
