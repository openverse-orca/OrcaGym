# 抬腿高度控制说明

## 📋 总结

**抬腿高度主要在模型中控制**（通过训练学习），但训练时通过代码中的奖励函数来引导模型学习合适的抬腿高度。

---

## 1. 控制机制

### 1.1 训练时（代码控制）

在训练阶段，抬腿高度通过**奖励函数**来引导：

**配置文件** (`Lite3_config.py`):
```python
"reward_coeff" : {
    "feet_swing_height" : 0,    # 奖励系数（Lite3中为0，表示不启用）
    ...
}

"foot_leg_period" : {
    "swing_height" : 0.03,      # 目标抬腿高度：0.03米（3厘米）
    ...
}
```

**奖励函数** (`legged_robot.py`):
```python
def _compute_reward_feet_swing_height(self, coeff) -> SupportsFloat:
    # 避免无命令时踏步
    if self._command["lin_vel"][0] == 0.0 and self._command["lin_vel"][1] == 0.0:
        return 0.0
    # 惩罚足底高度偏离目标高度
    swing_height = self._foot_leg_period["swing_height"]  # 0.03m
    contact = self._feet_contact[:] > 0
    pos_error = np.square(self._foot_height[:] - swing_height) * ~contact
    
    reward = -np.sum(pos_error) * coeff * self.dt
    return reward
```

**工作原理**：
- 计算足底实际高度与目标高度（0.03m）的平方误差
- 只在足底未接触地面时（swing phase）计算误差
- 误差越小，奖励越高（惩罚越小）
- 通过强化学习，模型学习到如何控制关节角度以达到目标抬腿高度

### 1.2 推理时（模型控制）

在推理阶段（使用ONNX模型），抬腿高度由**策略模型直接输出**：

**模型输出**：
- 策略模型输出12维动作（关节角度偏移）
- 动作通过PD控制器转换为关节角度
- 关节角度决定足底位置和抬腿高度

**代码流程**：
```python
# 1. 策略推理
policy_action = policy(lite3_obs)  # [12] 维动作

# 2. 转换为目标关节位置
target_dof_pos = policy_action * action_scale + dof_pos_default

# 3. PD控制计算力矩
torque = Kp * (target_pos - current_pos) + Kd * (target_vel - current_vel)

# 4. 执行动作，足底位置由关节角度决定
env.step(action)
```

---

## 2. 当前配置

### 2.1 Lite3配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **feet_swing_height奖励系数** | 0 | 训练时未启用抬腿高度奖励 |
| **swing_height目标高度** | 0.03m | 目标抬腿高度（3厘米） |
| **compute_foot_height** | True | 计算足底高度（用于奖励） |

### 2.2 其他机器人对比

| 机器人 | feet_swing_height系数 | swing_height目标 | 说明 |
|--------|----------------------|-----------------|------|
| **Lite3** | 0 | 0.03m | 未启用奖励 |
| **G1** | 20 | 0.1m | 强烈鼓励抬腿高度 |
| **A01B** | 未设置 | - | 未使用 |

---

## 3. 如何调整抬腿高度

### 3.1 训练时调整（需要重新训练）

如果要改变抬腿高度，需要修改训练配置：

```python
# 在Lite3_config.py中修改
"reward_coeff" : {
    "feet_swing_height" : 10,  # 启用奖励，系数越大越重要
}

"foot_leg_period" : {
    "swing_height" : 0.05,     # 修改目标高度（例如5厘米）
}
```

然后重新训练模型。

### 3.2 推理时调整（当前ONNX模型）

**无法直接调整**，因为：
- ONNX模型已经训练完成，抬腿高度已经编码在模型中
- 模型输出的动作是固定的，无法改变

**可能的间接方法**：
1. **修改action_scale**：改变动作范围可能影响抬腿高度，但不推荐
2. **修改PD控制参数**：可能影响响应速度，但不会改变目标高度
3. **重新训练模型**：使用不同的swing_height目标值训练新模型

---

## 4. 抬腿高度的影响因素

### 4.1 训练时影响因素

1. **奖励函数**：
   - `feet_swing_height` 系数：越大越鼓励达到目标高度
   - `swing_height` 目标值：期望的抬腿高度

2. **其他奖励项**：
   - `feet_air_time`：足底离地时间
   - `feet_slip`：足底滑动惩罚
   - `follow_command_linvel`：跟随命令速度

3. **动作空间**：
   - `action_scale`：动作范围影响关节角度变化幅度
   - 关节角度范围：限制最大抬腿高度

### 4.2 推理时影响因素

1. **模型输出**：策略模型直接决定关节角度
2. **PD控制**：Kp和Kd影响关节响应速度
3. **物理约束**：关节角度限制、碰撞检测等

---

## 5. 检查当前抬腿高度

### 5.1 代码中查看

在 `legged_robot.py` 中可以打印足底高度：

```python
# 在 _compute_reward_feet_swing_height 中
print("Feet swing height: ", self._foot_height, 
      "Swing height: ", swing_height, 
      "Contact: ", contact)
```

### 5.2 可视化查看

在仿真中可以直接观察机器人的抬腿高度。

---

## 6. 总结

| 阶段 | 控制方式 | 可调整性 |
|------|---------|---------|
| **训练时** | 奖励函数引导 | ✅ 可调整（需重新训练） |
| **推理时** | 模型直接输出 | ❌ 不可调整（模型已固定） |

**结论**：
- **抬腿高度主要在模型中控制**（通过训练学习）
- **训练时通过代码中的奖励函数引导**模型学习合适的抬腿高度
- **当前ONNX模型**的抬腿高度已经固定，无法直接调整
- 如需改变抬腿高度，需要**重新训练模型**并修改训练配置

---

**文档版本**: v1.0  
**最后更新**: 2024

