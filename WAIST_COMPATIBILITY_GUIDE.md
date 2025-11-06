# 腰部关节兼容性指南

## 概述

本指南说明如何使用支持腰部关节兼容性的机器人控制系统。系统现在可以自动检测配置中是否包含腰部关节，并相应地调整行为。

## 兼容性特性

### ✅ 支持的配置类型

1. **带腰部关节的机器人**
   - 配置中包含 `waist` 部分
   - 支持腰部旋转控制
   - 末端位置跟随腰部转动

2. **不带腰部关节的机器人**
   - 配置中不包含 `waist` 部分
   - 使用传统双臂控制
   - 末端位置基于固定基座

### 🔧 自动检测机制

系统会自动检测配置中是否包含腰部关节：

```python
# 自动检测逻辑
self._has_waist = "waist" in config and config["waist"] is not None
```

## 配置示例

### 带腰部关节的配置

```python
# d12_waist_config.py
d12_waist_config = {
    "robot_type": "dual_arm",
    "base": {
        "base_body_name": "base_link",
        "base_joint_name": "base_joint",
        "dummy_joint_name": "dummy_joint",
    },
    "right_arm": {
        "joint_names": ["right_shoulder_pitch_joint", ...],
        "neutral_joint_values": [-0.67, -0.72, ...],
        "motor_names": ["M_arm_r_01", "M_arm_r_02", ...],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["left_shoulder_pitch_joint", ...],
        "neutral_joint_values": [-0.67, 0.72, ...],
        "motor_names": ["M_arm_l_01", "M_arm_l_02", ...],
        "ee_center_site_name": "ee_center_site",
    },
    "waist": {  # 关键：包含waist部分
        "joint_name": "waist_yaw_joint",
        "neutral_joint_value": 0.0,
        "position_name": "P_waist",
    },
}
```

### 不带腰部关节的配置

```python
# dual_arm_no_waist_config.py
dual_arm_no_waist_config = {
    "robot_type": "dual_arm",
    "base": {
        "base_body_name": "base_link",
        "base_joint_name": "base_joint",
        "dummy_joint_name": "dummy_joint",
    },
    "right_arm": {
        "joint_names": ["right_shoulder_pitch_joint", ...],
        "neutral_joint_values": [-0.67, -0.72, ...],
        "motor_names": ["M_arm_r_01", "M_arm_r_02", ...],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["left_shoulder_pitch_joint", ...],
        "neutral_joint_values": [-0.67, 0.72, ...],
        "motor_names": ["M_arm_l_01", "M_arm_l_02", ...],
        "ee_center_site_name": "ee_center_site",
    },
    # 注意：没有waist部分
}
```

## 功能差异

### 带腰部关节的机器人

**可用功能**：
- ✅ 腰部旋转控制
- ✅ 手柄遥操作腰部控制
- ✅ 末端位置跟随腰部转动
- ✅ 腰部关节数据记录
- ✅ 坐标系转换考虑腰部转动

**控制方法**：
```python
# 设置腰部角度
robot.set_waist_control(target_angle=0.5)

# 手柄控制
robot.set_waist_joystick_control(joystick_state)

# 获取腰部角度
waist_angle = robot.get_waist_angle()

# 重置腰部
robot.reset_waist_control()
```

### 不带腰部关节的机器人

**可用功能**：
- ✅ 传统双臂控制
- ✅ 末端位置基于固定基座
- ✅ 标准坐标系转换
- ❌ 腰部相关功能被忽略

**行为**：
```python
# 这些调用会被安全忽略
robot.set_waist_control(target_angle=0.5)  # 无效果
robot.set_waist_joystick_control(joystick_state)  # 无效果
waist_angle = robot.get_waist_angle()  # 返回 0.0
```

## 实现细节

### 1. 配置检测

```python
# 在 _read_config 方法中
self._has_waist = "waist" in config and config["waist"] is not None

if self._has_waist:
    # 初始化腰部关节相关变量
    self._waist_joint_name = self._env.joint(config["waist"]["joint_name"], id)
    # ... 其他腰部初始化
else:
    # 设置默认值
    self._waist_joint_name = None
    # ... 其他默认值
```

### 2. 安全的方法调用

```python
def get_waist_angle(self) -> float:
    """获取当前腰部角度，如果没有腰部关节则返回0"""
    if self._has_waist and self._waist_jnt_address is not None:
        return self._env.data.qpos[self._waist_jnt_address]
    else:
        return 0.0

def set_waist_control(self, target_angle: float = 0.0) -> None:
    """设置腰部控制，如果没有腰部关节则忽略"""
    if not self._has_waist or self._waist_actuator_id is None:
        return
    # ... 腰部控制逻辑
```

### 3. 坐标系转换

```python
def _local_to_global(self, local_pos, local_quat):
    """坐标系转换，考虑腰部转动（如果有腰部关节）"""
    base_link_pos, _, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)

    if self._has_waist:
        # 有腰部关节：组合基座和腰部旋转
        waist_angle = self.get_waist_angle()
        waist_quat = create_waist_quaternion(waist_angle)
        combined_quat = rotations.quat_mul(base_link_quat, waist_quat)
    else:
        # 无腰部关节：直接使用基座旋转
        combined_quat = base_link_quat
    
    # 应用转换
    global_pos = base_link_pos + rotations.quat_rot_vec(combined_quat, local_pos)
    global_quat = rotations.quat_mul(combined_quat, local_quat)
    return global_pos, global_quat
```

## 使用指南

### 1. 选择配置

**带腰部关节**：
```python
from envs.manipulation.robots.configs.d12_waist_config import d12_waist_config
# 使用 d12_waist_config
```

**不带腰部关节**：
```python
from envs.manipulation.robots.configs.dual_arm_no_waist_config import dual_arm_no_waist_config
# 使用 dual_arm_no_waist_config
```

### 2. 检查腰部支持

```python
# 检查机器人是否支持腰部关节
if robot._has_waist:
    print("机器人支持腰部关节")
    waist_angle = robot.get_waist_angle()
    robot.set_waist_control(0.5)
else:
    print("机器人不支持腰部关节")
```

### 3. 安全的功能调用

```python
# 这些调用在任何配置下都是安全的
robot.set_waist_control(target_angle)  # 有腰部时有效，无腰部时忽略
robot.set_waist_joystick_control(state)  # 有腰部时有效，无腰部时忽略
waist_angle = robot.get_waist_angle()  # 总是返回有效值
```

## 测试和验证

### 运行兼容性测试

```bash
python test_waist_compatibility.py
```

### 测试内容

1. **配置结构测试**
   - 带腰部配置验证
   - 不带腰部配置验证

2. **功能兼容性测试**
   - 方法存在性检查
   - 安全调用验证

3. **数学正确性测试**
   - 坐标系转换验证
   - 角度计算验证

## 迁移指南

### 从无腰部到有腰部

1. 在配置中添加 `waist` 部分
2. 确保XML文件包含腰部关节和控制器
3. 重新运行系统，自动检测腰部支持

### 从有腰部到无腰部

1. 从配置中移除 `waist` 部分
2. 系统自动切换到无腰部模式
3. 腰部相关功能调用被安全忽略

## 注意事项

### 1. 性能影响

- 有腰部关节：轻微性能开销（角度查询和坐标转换）
- 无腰部关节：无额外开销

### 2. 向后兼容性

- ✅ 现有无腰部配置无需修改
- ✅ 现有代码无需修改
- ✅ 新功能可选使用

### 3. 调试建议

```python
# 检查腰部支持状态
print(f"机器人有腰部关节: {robot._has_waist}")
print(f"当前腰部角度: {robot.get_waist_angle()}")

# 检查配置加载
if robot._has_waist:
    print(f"腰部关节名称: {robot._waist_joint_name}")
    print(f"腰部控制器ID: {robot._waist_actuator_id}")
```

## 总结

通过兼容性支持，您现在可以：

1. **无缝切换**：在带腰部和不带腰部的机器人之间切换
2. **代码复用**：同一套代码支持两种配置
3. **安全调用**：腰部相关功能在无腰部时安全忽略
4. **自动检测**：系统自动检测配置类型并调整行为

这使得系统更加灵活和健壮，支持更广泛的机器人配置需求。

