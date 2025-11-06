# 腰部控制功能说明

## 概述

本功能为OrcaGym仿真环境添加了通过Pico手柄控制机器人腰部旋转的能力。腰部控制使用独立的PD控制器，避免了与手臂控制的冲突。

## 功能特性

- **双重控制方式**：支持左手柄和右手柄摇杆控制
- **安全范围限制**：腰部旋转角度限制在±90度范围内
- **可调灵敏度**：支持动态调整控制灵敏度
- **一键重置**：组合键快速重置到初始位置
- **稳定控制**：使用PD控制器确保平滑稳定的运动

## 控制方式

### 主要控制
- **左手柄摇杆按下 + X轴左右移动**：控制腰部旋转
- **右手柄摇杆按下 + X轴左右移动**：备用控制方式

### 特殊功能
- **左手柄A键 + 右手柄B键**：重置腰部到初始位置

## 技术实现

### 核心类和方法

#### DualArmRobot类新增方法

1. **set_waist_control(target_angle)**
   - 使用Position控制器设置腰部目标角度
   - 支持平滑过渡控制
   - 参数：target_angle - 目标角度（弧度）

2. **set_waist_joystick_control(joystick_state)**
   - 处理手柄输入并控制腰部旋转
   - 参数：joystick_state - 手柄状态字典

3. **reset_waist_control()**
   - 重置腰部控制到初始位置
   - 重置所有相关变量

4. **set_waist_control_sensitivity(sensitivity)**
   - 设置控制灵敏度
   - 参数：sensitivity - 灵敏度值 [0.1, 2.0]

5. **set_waist_smooth_transition(enabled)**
   - 设置是否启用平滑过渡
   - 参数：enabled - 是否启用平滑过渡

6. **set_waist_max_velocity(max_velocity)**
   - 设置腰部最大角速度
   - 参数：max_velocity - 最大角速度 (rad/s)

7. **get_waist_angle()**
   - 获取当前腰部角度
   - 返回：当前角度（弧度）

8. **get_waist_target_angle()**
   - 获取目标腰部角度
   - 返回：目标角度（弧度）

### 控制参数

- **最大旋转角度**：±π/2 弧度（±90度）
- **默认灵敏度**：0.5
- **最大角速度**：1.0 rad/s
- **平滑过渡**：启用
- **控制器类型**：Position控制器（内置PID）

## 配置要求

### XML配置文件修改

1. **关节参数优化**：
   ```xml
   <joint name="waist_yaw_joint" damping="0.1" frictionloss="0.01" armature="0.01"/>
   ```

2. **执行器参数调整**：
   ```xml
   <position name="P_waist" kp="10" kv="5" ctrlrange="-3.14 3.14" forcerange="-50 50"/>
   <motor name="M_waist" ctrlrange="-50 50"/>
   ```

### Python配置文件修改

在`d12_waist_config.py`中添加腰部配置：
```python
"waist": {
    "joint_name": "waist_yaw_joint",
    "neutral_joint_value": 0.0,
    "motor_name": "M_waist",
    "position_name": "P_waist",
}
```

## 使用示例

### 基本使用

```python
# 在遥操作模式下，腰部控制会自动启用
# 通过手柄摇杆控制腰部旋转

# 调整控制灵敏度
robot.set_waist_control_sensitivity(0.8)

# 配置平滑过渡
robot.set_waist_smooth_transition(True)
robot.set_waist_max_velocity(1.5)  # 1.5 rad/s

# 获取当前状态
current_angle = robot.get_waist_angle()
target_angle = robot.get_waist_target_angle()

# 手动重置
robot.reset_waist_control()
```

### 高级控制

```python
# 直接设置目标角度
robot.set_waist_control(np.pi/4)  # 45度

# 配置不同控制模式
robot.set_waist_smooth_transition(False)  # 禁用平滑过渡，快速响应
robot.set_waist_max_velocity(2.0)  # 提高最大角速度

# 动态调整参数
for sensitivity in [0.3, 0.5, 0.8, 1.0]:
    robot.set_waist_control_sensitivity(sensitivity)
    # 执行控制逻辑...
```

### Position控制器优势

```python
# Position控制器相比Motor控制器的优势：
# 1. 内置PID控制，无需手动调参
# 2. 更稳定的控制响应
# 3. 支持平滑过渡控制
# 4. 自动限制角度范围
# 5. 适合遥操作应用
```

## 故障排除

### 常见问题

1. **腰部不响应控制**
   - 检查手柄连接状态
   - 确认摇杆按下状态
   - 验证腰部关节配置

2. **控制过于敏感或迟钝**
   - 调整灵敏度参数
   - 检查PD控制增益设置

3. **腰部运动不稳定**
   - 检查关节阻尼参数
   - 验证执行器配置
   - 确认没有双重控制冲突

### 调试信息

启用调试输出：
```python
# 在_read_config方法中已包含调试信息
print("waist joint name: ", self._waist_joint_name)
print("waist actuator id: ", self._waist_actuator_id)
# ... 其他调试信息
```

## 安全注意事项

1. **角度限制**：腰部旋转角度被限制在±90度范围内
2. **力矩限制**：控制力矩被限制在±10 N·m范围内
3. **平滑控制**：使用PD控制器确保平滑稳定的运动
4. **紧急停止**：松开摇杆后腰部会保持当前位置

## 更新日志

- **v1.0.0**：初始版本，支持基本腰部控制功能
- 添加双重控制方式支持
- 实现安全范围限制
- 提供可调灵敏度功能
- 添加一键重置功能

## 相关文件

- `envs/manipulation/dual_arm_robot.py` - 主要实现文件
- `envs/manipulation/robots/configs/d12_waist_config.py` - 配置文件
- `d12_waist_fixed.xml` - 修复后的XML配置文件
- `waist_control_example.py` - 使用示例

