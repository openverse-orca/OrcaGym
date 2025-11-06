# 腰部转动对末端位置影响实现说明

## 概述

本实现修改了机器人末端执行器位置计算的方法，使其能够跟随腰部关节的转动。当腰部关节(`waist_yaw_joint`)转动时，末端执行器在全局坐标系中的位置会相应地发生变化，但在局部坐标系中保持相对稳定的位置关系。

## 问题分析

### 原始问题
- 末端执行器位置计算基于固定的`base_link`坐标系
- 腰部关节转动时，末端位置计算没有考虑腰部的旋转
- 导致末端位置在全局坐标系中不会跟随腰部转动

### 解决方案
- 在坐标系转换中引入腰部关节角度
- 将腰部旋转与基座旋转组合
- 确保末端位置计算考虑腰部的转动

## 修改的方法

### 1. `_local_to_global()` 方法 (`dual_arm_robot.py:617`)

**功能**: 将局部坐标系位姿转换为全局坐标系

**修改内容**:
```python
def _local_to_global(self, local_pos: np.ndarray, local_quat: np.ndarray) -> tuple:
    base_link_pos, _, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)
    
    # 获取腰部关节角度
    waist_angle = self.get_waist_angle()
    
    # 创建腰部转动的四元数 (绕Z轴旋转)
    waist_quat = np.array([
        np.cos(waist_angle / 2),  # w
        0,                        # x
        0,                        # y
        np.sin(waist_angle / 2)   # z
    ])
    
    # 组合基座和腰部的旋转
    combined_quat = rotations.quat_mul(base_link_quat, waist_quat)
    
    global_pos = base_link_pos + rotations.quat_rot_vec(combined_quat, local_pos)
    global_quat = rotations.quat_mul(combined_quat, local_quat)
    return global_pos, global_quat
```

### 2. `_global_to_local()` 方法 (`dual_arm_robot.py:645`)

**功能**: 将全局坐标系位姿转换为局部坐标系

**修改内容**:
```python
def _global_to_local(self, global_pos: np.ndarray, global_quat: np.ndarray) -> tuple:
    base_link_pos, _, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)
    
    # 获取腰部关节角度
    waist_angle = self.get_waist_angle()
    
    # 创建腰部转动的四元数 (绕Z轴旋转)
    waist_quat = np.array([
        np.cos(waist_angle / 2),  # w
        0,                        # x
        0,                        # y
        np.sin(waist_angle / 2)   # z
    ])
    
    # 组合基座和腰部的旋转
    combined_quat = rotations.quat_mul(base_link_quat, waist_quat)
    combined_quat_inv = rotations.quat_conjugate(combined_quat)
    
    local_pos = rotations.quat_rot_vec(combined_quat_inv, global_pos - base_link_pos)
    local_quat = rotations.quat_mul(combined_quat_inv, global_quat)
    return local_pos, local_quat
```

### 3. `query_site_pos_and_quat_B()` 方法 (`orca_gym_local_env.py:429`)

**功能**: 查询相对于基座坐标系的末端位置和姿态

**修改内容**:
```python
def query_site_pos_and_quat_B(self, site_names, base_body_list, waist_angle=0.0):
    # ... 原有代码 ...
    
    # 创建腰部转动的旋转矩阵 (绕Z轴旋转)
    rot_waist = R.from_euler('z', waist_angle, degrees=False)
    
    # 组合基座和腰部的旋转
    rot_combined = rot_base * rot_waist
    rot_combined_inv = rot_combined.inv()
    
    # 应用组合旋转
    relative_rot_ee = rot_combined_inv * rot_ee
    relative_pos_ee = rot_combined_inv.apply(ee_pos - base_pos)
```

### 4. `query_site_xvalp_xvalr_B()` 方法 (`orca_gym_local_env.py:474`)

**功能**: 查询相对于基座坐标系的末端速度

**修改内容**:
```python
def query_site_xvalp_xvalr_B(self, site_names, base_body_list, waist_angle=0.0):
    # ... 原有代码 ...
    
    # 创建腰部转动的旋转矩阵 (绕Z轴旋转)
    rot_waist = R.from_euler('z', waist_angle, degrees=False)
    waist_mat = rot_waist.as_matrix()
    
    # 组合基座和腰部的旋转矩阵
    combined_mat = base_mat @ waist_mat
    
    # 应用组合旋转矩阵
    linear_vel_B = combined_mat.T @ (ee_xvalp - base_xvalp)
    angular_vel_B = combined_mat.T @ (ee_xvalr - base_xvalr)
```

### 5. `get_obs()` 方法 (`dual_arm_robot.py:275`)

**功能**: 获取观测数据

**修改内容**:
```python
def get_obs(self) -> dict:
    # 获取当前腰部关节角度
    waist_angle = self.get_waist_angle()
    
    # 传递腰部角度参数，使末端位置跟着腰部转动
    ee_sites = self._env.query_site_pos_and_quat_B([self._ee_site_l, self._ee_site_r], self._base_body_name, waist_angle)
    ee_xvalp, ee_xvalr = self._env.query_site_xvalp_xvalr_B([self._ee_site_l, self._ee_site_r], self._base_body_name, waist_angle)
    # ... 其余代码 ...
```

## 数学原理

### 坐标系转换

1. **组合旋转**: `R_combined = R_base * R_waist`
2. **位置转换**: `pos_global = pos_base + R_combined * pos_local`
3. **姿态转换**: `quat_global = R_combined * quat_local`

### 腰部旋转矩阵

腰部关节绕Z轴旋转，旋转矩阵为：
```
R_waist = [cos(θ) -sin(θ)  0]
          [sin(θ)  cos(θ)  0]
          [0       0       1]
```

其中θ是腰部关节角度。

### 四元数表示

腰部旋转的四元数表示为：
```
q_waist = [cos(θ/2), 0, 0, sin(θ/2)]
```

## 使用效果

### 1. 末端位置跟随腰部转动
- 当腰部转动时，末端执行器在全局坐标系中的位置会相应变化
- 在局部坐标系中保持相对稳定的位置关系

### 2. 观测数据一致性
- 观测数据中的末端位置会反映腰部的转动
- 速度计算也会考虑腰部的转动

### 3. 控制逻辑正确性
- 遥操作时的坐标系转换正确
- 回放时的动作执行正确

## 测试验证

### 运行示例脚本
```bash
python waist_rotation_example.py
```

该脚本会：
1. 演示腰部转动对末端位置的影响
2. 显示数学原理和实现细节
3. 生成可视化图表

### 验证要点
1. 末端位置是否跟随腰部转动
2. 局部坐标系中的相对位置是否稳定
3. 观测数据是否一致
4. 控制逻辑是否正确

## 注意事项

### 1. 性能影响
- 每次计算都需要获取腰部关节角度
- 增加了四元数运算的复杂度
- 对整体性能影响较小

### 2. 数值稳定性
- 使用四元数避免万向锁问题
- 确保角度范围在合理范围内
- 注意浮点数精度问题

### 3. 兼容性
- 向后兼容，不影响现有功能
- 腰部角度参数有默认值0.0
- 可以逐步迁移到新实现

## 相关文件

- `envs/manipulation/dual_arm_robot.py` - 主要修改文件
- `orca_gym/environment/orca_gym_local_env.py` - 环境接口修改
- `waist_rotation_example.py` - 示例和测试脚本
- `WAIST_ROTATION_IMPLEMENTATION.md` - 本说明文档

## 更新日志

- **v1.0.0**: 初始实现，添加腰部转动对末端位置的影响
- 修改坐标系转换方法
- 更新观测数据计算
- 添加示例和测试脚本
