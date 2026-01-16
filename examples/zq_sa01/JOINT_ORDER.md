# ZQ SA01 关节顺序和观察空间映射

## 关节定义顺序

### 关节名称列表（按索引）

```python
joint_names = [
    'leg_l1_joint',  # 索引 0:  左髋侧摆 (Hip Roll Left)
    'leg_l2_joint',  # 索引 1:  左髋旋转 (Hip Yaw Left)
    'leg_l3_joint',  # 索引 2:  左髋俯仰 (Hip Pitch Left)
    'leg_l4_joint',  # 索引 3:  左膝关节 (Knee Left)
    'leg_l5_joint',  # 索引 4:  左踝俯仰 (Ankle Pitch Left)
    'leg_l6_joint',  # 索引 5:  左踝侧摆 (Ankle Roll Left)
    'leg_r1_joint',  # 索引 6:  右髋侧摆 (Hip Roll Right)
    'leg_r2_joint',  # 索引 7:  右髋旋转 (Hip Yaw Right)
    'leg_r3_joint',  # 索引 8:  右髋俯仰 (Hip Pitch Right)
    'leg_r4_joint',  # 索引 9:  右膝关节 (Knee Right)
    'leg_r5_joint',  # 索引 10: 右踝俯仰 (Ankle Pitch Right)
    'leg_r6_joint',  # 索引 11: 右踝侧摆 (Ankle Roll Right)
]
```

### 关节类型说明

| 索引 | 关节名 | 中文名 | 英文名 | 自由度类型 | 默认角度 |
|------|--------|--------|--------|-----------|---------|
| 0 | leg_l1 | 左髋侧摆 | Hip Roll Left | Revolute (侧向) | 0.0 |
| 1 | leg_l2 | 左髋旋转 | Hip Yaw Left | Revolute (旋转) | 0.0 |
| 2 | leg_l3 | 左髋俯仰 | Hip Pitch Left | Revolute (前后) | -0.24 |
| 3 | leg_l4 | 左膝关节 | Knee Left | Revolute (前后) | 0.48 |
| 4 | leg_l5 | 左踝俯仰 | Ankle Pitch Left | Revolute (前后) | -0.24 |
| 5 | leg_l6 | 左踝侧摆 | Ankle Roll Left | Revolute (侧向) | 0.0 |
| 6 | leg_r1 | 右髋侧摆 | Hip Roll Right | Revolute (侧向) | 0.0 |
| 7 | leg_r2 | 右髋旋转 | Hip Yaw Right | Revolute (旋转) | 0.0 |
| 8 | leg_r3 | 右髋俯仰 | Hip Pitch Right | Revolute (前后) | -0.24 |
| 9 | leg_r4 | 右膝关节 | Knee Right | Revolute (前后) | 0.48 |
| 10 | leg_r5 | 右踝俯仰 | Ankle Pitch Right | Revolute (前后) | -0.24 |
| 11 | leg_r6 | 右踝侧摆 | Ankle Roll Right | Revolute (侧向) | 0.0 |

### 默认站立姿态

```python
default_dof_pos = [
    0.0,   0.0,  -0.24, 0.48, -0.24, 0.0,  # 左腿
    0.0,   0.0,  -0.24, 0.48, -0.24, 0.0   # 右腿
]

# 物理意义：
# 髋俯仰: -0.24 rad ≈ -13.75° (髋关节略微后倾)
# 膝关节:  0.48 rad ≈  27.50° (膝盖弯曲)
# 踝俯仰: -0.24 rad ≈ -13.75° (脚踝补偿，保持平衡)
# 净效果: 机器人保持稍微下蹲的站立姿态
```

## 观察空间映射

### 环境返回的观察（45维，不含相位）

```python
single_obs = env.unwrapped._get_obs()  # 45维

结构:
[0:3]    命令 (cmd_vx, cmd_vy, cmd_dyaw)
[3:15]   关节位置偏差 (dof_pos - default_dof_pos) × 1.0
[15:27]  关节速度 (dof_vel × 0.05)  ← 注意已缩放！
[27:39]  上一步动作 (last_action)
[39:42]  角速度 (ang_vel)
[42:45]  欧拉角 (euler)
```

### 添加相位后的观察（47维）

```python
single_obs_with_phase = np.concatenate([
    [sin_phase, cos_phase],  # 2维
    single_obs               # 45维
])  # 总共 47维

结构:
[0:2]    相位 (sin(2π×phase), cos(2π×phase))
[2:5]    命令 (cmd_vx, cmd_vy, cmd_dyaw)
[5:17]   关节位置偏差 (12维)
[17:29]  关节速度 (12维，已 ×0.05)
[29:41]  上一步动作 (12维)
[41:44]  角速度 (3维)
[44:47]  欧拉角 (3维)
```

### 完整输入（705维）

```python
obs = np.concatenate(list(hist_obs))  # 15帧 × 47维 = 705维

结构:
[0:47]     t-14 时刻的观察
[47:94]    t-13 时刻的观察
...
[658:705]  t 时刻（当前）的观察
```

## 详细的关节映射表

### 在单帧观察中（47维）的位置

| 关节<br>索引 | 关节名 | 中文名 | 位置偏差<br>obs索引 | 速度<br>obs索引 | 上一步动作<br>obs索引 | 动作<br>输出索引 |
|-------------|--------|--------|-------------------|----------------|---------------------|----------------|
| **左腿** |
| 0 | leg_l1 | 左髋侧摆 | [5] | [17] | [29] | [0] |
| 1 | leg_l2 | 左髋旋转 | [6] | [18] | [30] | [1] |
| 2 | leg_l3 | 左髋俯仰 | [7] | [19] | [31] | [2] |
| 3 | leg_l4 | 左膝关节 | [8] | [20] | [32] | [3] |
| 4 | leg_l5 | 左踝俯仰 | [9] | [21] | [33] | [4] |
| 5 | leg_l6 | 左踝侧摆 | [10] | [22] | [34] | [5] |
| **右腿** |
| 6 | leg_r1 | 右髋侧摆 | [11] | [23] | [35] | [6] |
| 7 | leg_r2 | 右髋旋转 | [12] | [24] | [36] | [7] |
| 8 | leg_r3 | 右髋俯仰 | [13] | [25] | [37] | [8] |
| 9 | leg_r4 | 右膝关节 | [14] | [26] | [38] | [9] |
| 10 | leg_r5 | 右踝俯仰 | [15] | [27] | [39] | [10] |
| 11 | leg_r6 | 右踝侧摆 | [16] | [28] | [40] | [11] |

### 其他观察变量

| obs索引 | 维度 | 变量名 | 含义 | 缩放/公式 |
|---------|------|--------|------|----------|
| [0] | 1 | sin_phase | 相位正弦 | sin(2π × phase) |
| [1] | 1 | cos_phase | 相位余弦 | cos(2π × phase) |
| [2] | 1 | cmd_vx | 前进速度命令 | m/s |
| [3] | 1 | cmd_vy | 侧向速度命令 | m/s |
| [4] | 1 | cmd_dyaw | 转向角速度命令 | rad/s |
| [41] | 1 | ang_vel_x | 基座角速度 Roll | rad/s × 1.0 |
| [42] | 1 | ang_vel_y | 基座角速度 Pitch | rad/s × 1.0 |
| [43] | 1 | ang_vel_z | 基座角速度 Yaw | rad/s × 1.0 |
| [44] | 1 | euler_x | 基座欧拉角 Roll | rad × 1.0 |
| [45] | 1 | euler_y | 基座欧拉角 Pitch | rad × 1.0 |
| [46] | 1 | euler_z | 基座欧拉角 Yaw | rad × 1.0 |

## 数据流向

### 1. 环境内部流程

```python
# 1. 查询真实关节状态
joint_qpos = env.query_joint_qpos(joint_names)  # 真实位置
joint_qvel = env.query_joint_qvel(joint_names)  # 真实速度

dof_pos = [joint_qpos[jn] for jn in joint_names]  # 按顺序提取
dof_vel = [joint_qvel[jn] for jn in joint_names]

# 2. 构建观察
obs[5:17] = dof_pos - default_dof_pos  # 位置偏差
obs[17:29] = dof_vel * 0.05             # 速度（缩放）
```

### 2. 策略推理流程

```python
# 1. 构建完整观察（705维）
obs_history = deque(maxlen=15)
for i in range(15):
    obs_history.append(single_obs_with_phase_47d)
obs_full = np.concatenate(list(obs_history))  # 705维

# 2. 策略推理
action = policy(obs_full)  # 输入 705维 → 输出 12维

# action 是关节位置偏移量，不是力矩！
```

### 3. PD控制流程（在环境内部）

```python
def do_simulation(action, n_frames):
    # 1. 计算目标位置
    target_q = action * 0.5 + default_dof_pos
    
    # 2. 查询当前状态
    q_current = query_joint_qpos(joint_names)
    dq_current = query_joint_qvel(joint_names)
    
    # 3. PD控制计算力矩
    tau = kps * (target_q - q_current) - kds * dq_current
    tau = np.clip(tau, -200, 200)
    
    # 4. 施加力矩
    set_ctrl(tau)
    for _ in range(n_frames):
        mj_step()
```

## 代码示例

### 访问特定关节的观察

```python
# 获取单帧观察（47维）
single_obs_with_phase = [sin_phase, cos_phase] + env._get_obs()

# 访问左膝关节 (leg_l4, 索引3)
left_knee_pos_error = single_obs_with_phase[5 + 3]  # = single_obs_with_phase[8]
left_knee_vel_scaled = single_obs_with_phase[17 + 3]  # = single_obs_with_phase[20]
left_knee_last_action = single_obs_with_phase[29 + 3]  # = single_obs_with_phase[32]

# 还原真实速度
left_knee_vel_real = left_knee_vel_scaled / 0.05

# 访问右髋俯仰 (leg_r3, 索引8)
right_hip_pitch_error = single_obs_with_phase[5 + 8]  # = single_obs_with_phase[13]
right_hip_pitch_vel = single_obs_with_phase[17 + 8] / 0.05
```

### 批量访问关节数据

```python
# 获取所有关节位置偏差
joint_pos_errors = single_obs_with_phase[5:17]  # 12维

# 获取所有关节速度（缩放后）
joint_vels_scaled = single_obs_with_phase[17:29]  # 12维

# 还原真实速度
joint_vels_real = joint_vels_scaled / 0.05

# 获取左腿数据
left_leg_pos_errors = single_obs_with_phase[5:11]   # 6维
left_leg_vels = single_obs_with_phase[17:23] / 0.05

# 获取右腿数据
right_leg_pos_errors = single_obs_with_phase[11:17]  # 6维
right_leg_vels = single_obs_with_phase[23:29] / 0.05
```

### 验证关节顺序

```python
import numpy as np

# 环境中的关节名称
joint_names_str = [
    'leg_l1_joint', 'leg_l2_joint', 'leg_l3_joint',
    'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint',
    'leg_r1_joint', 'leg_r2_joint', 'leg_r3_joint',
    'leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint'
]

# 打印当前状态
single_obs = env.unwrapped._get_obs()
single_obs_with_phase = np.concatenate([[0, 1], single_obs])  # 添加假相位

print("关节状态:")
print(f"{'索引':<4} {'关节名':<15} {'位置偏差':<12} {'速度(缩放)':<12} {'速度(真实)':<12}")
print("-" * 60)

for i, name in enumerate(joint_names_str):
    pos_error = single_obs_with_phase[5 + i]
    vel_scaled = single_obs_with_phase[17 + i]
    vel_real = vel_scaled / 0.05
    
    print(f"{i:<4} {name:<15} {pos_error:>11.4f} {vel_scaled:>11.4f} {vel_real:>11.4f}")
```

## 常见错误

### ❌ 错误 1: 使用观察中的偏差计算力矩

```python
# 错误！观察中的是位置偏差，不能直接用于PD控制
joint_pos_error = single_obs[5:17]
tau = kp * joint_pos_error - kd * vel  # ❌ 错误！
```

**原因**: PD控制需要：`kp * (target - current)`，而观察中已经是偏差了

**正确做法**: 让环境内部处理PD控制，不要手动计算

### ❌ 错误 2: 索引错位

```python
# 错误！single_obs 只有45维，没有包含相位
joint_pos = single_obs[5:17]  # ❌ 错误！应该是 [3:15]

# 正确：single_obs_with_phase 有47维，包含相位
joint_pos = single_obs_with_phase[5:17]  # ✓ 正确
```

### ❌ 错误 3: 速度缩放错误

```python
# 错误！速度已经缩放过了
vel_scaled = single_obs[17:29]
vel_real = vel_scaled * 20  # ❌ 错误！应该是除以0.05

# 正确
vel_real = vel_scaled / 0.05  # ✓ 正确
```

### ❌ 错误 4: 覆盖策略输出

```python
# 错误！策略输出的是位置偏移，不应该被覆盖
action = policy(obs)  # 策略输出
tau = calculate_torque(...)
action = tau  # ❌ 错误！覆盖了策略输出

# 正确：直接使用策略输出
action = policy(obs)
action = np.clip(action, -18, 18)
obs, _, _, _, _ = env.step(action)  # ✓ 环境内部会处理PD控制
```

## PD 控制参数

### 刚度和阻尼

```python
kps = np.array([
    50, 50, 70, 70, 20, 20,  # 左腿: 髋侧,髋旋,髋俯,膝,踝俯,踝侧
    50, 50, 70, 70, 20, 20   # 右腿: 同上
], dtype=np.float64)

kds = np.array([
    5.0, 5.0, 7.0, 7.0, 2.0, 2.0,  # 左腿
    5.0, 5.0, 7.0, 7.0, 2.0, 2.0   # 右腿
], dtype=np.float64)

# 关系: kd ≈ kp / 10
```

### 参数说明

| 关节类型 | Kp | Kd | 说明 |
|---------|----|----|------|
| 髋侧摆/旋转 | 50 | 5.0 | 较软，允许侧向调整 |
| 髋俯仰/膝关节 | 70 | 7.0 | 较硬，支撑体重 |
| 踝关节 | 20 | 2.0 | 最软，平衡调整 |

## 总结

### 关键点

1. **关节顺序**: 先左腿6个关节，再右腿6个关节
2. **观察索引**: 
   - 位置偏差: [5:17]
   - 速度(缩放): [17:29]
   - 动作: [29:41]
3. **速度缩放**: 观察中的速度已经 ×0.05
4. **策略输出**: 是位置偏移，不是力矩
5. **PD控制**: 在环境内部自动处理

### 正确的使用流程

```python
# 1. 获取观察
single_obs = env.unwrapped._get_obs()  # 45维
single_obs_with_phase = [sin, cos] + single_obs  # 47维

# 2. 构建历史
hist_obs.append(single_obs_with_phase)
obs_full = np.concatenate(list(hist_obs))  # 705维

# 3. 策略推理
action = policy(obs_full)  # 12维位置偏移
action = np.clip(action, -18, 18)

# 4. 执行（环境内部会做PD控制）
obs, reward, done, truncated, info = env.step(action)
```

**不要手动计算力矩！让环境处理！**

