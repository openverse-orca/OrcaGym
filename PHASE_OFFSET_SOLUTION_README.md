# 相位偏移解决方案说明

## 问题描述

在使用多对象动画时，发现两个物体距离会越来越近，直至接触到。这是因为所有对象都使用相同的曲线参数化距离，导致目标位置逐渐收敛。

## 根本原因

1. **共享曲线距离**：所有对象都使用相同的 `curbesdist` 来更新位置
2. **相同目标位置**：从相同的曲线距离计算目标位置
3. **缺乏相位差**：没有为不同对象设置不同的起始相位

## 解决方案：相位偏移

### 核心思想
为每个对象设置不同的相位偏移（`phase_offset`），使它们在曲线上保持固定距离。

### 实现原理

```python
# 原始问题代码：
obj.curbesdist += movedist  # 所有对象累积相同距离
pos = bezier_path.get_position(obj.curbesdist)  # 相同目标位置

# 修复后代码：
obj.original_curbesdist += movedist  # 原始距离
obj.curbesdist = obj.original_curbesdist + obj.phase_offset  # 加相位偏移
pos = bezier_path.get_position(obj.curbesdist)  # 不同目标位置
```

## 主要修改

### 1. AnimatedObject 类增强
```python
class AnimatedObject:
    def __init__(self, joint_name, bezier_path, start_from_current_position=True, phase_offset=0.0):
        # ... 其他属性 ...
        self.phase_offset = phase_offset  # 相位偏移
        self.original_curbesdist = 0  # 原始曲线距离
```

### 2. 自动相位偏移计算
```python
def add_animated_object(self, joint_name, start_from_current_position=None, phase_offset=None):
    # 自动计算相位偏移
    if phase_offset is None:
        phase_offset = len(self.animated_objects) * (self.bezierpath.total_length * 0.1)
```

### 3. 距离更新逻辑
```python
# 更新原始距离
obj.original_curbesdist += movedist
# 计算带相位偏移的距离
obj.curbesdist = obj.original_curbesdist + obj.phase_offset
```

## 使用方法

### 1. 自动相位偏移（推荐）
```python
# 系统会自动为每个对象计算相位偏移
env.add_animated_object("object_1")
env.add_animated_object("object_2")  # 自动偏移10%曲线长度
env.add_animated_object("object_3")  # 自动偏移20%曲线长度
```

### 2. 手动设置相位偏移
```python
# 手动设置不同的相位偏移
env.add_animated_object("object_1", phase_offset=0.0)
env.add_animated_object("object_2", phase_offset=curve_length * 0.2)  # 20%偏移
env.add_animated_object("object_3", phase_offset=curve_length * 0.5)  # 50%偏移
```

### 3. 当前配置
```python
# 当前代码中的配置
self.add_animated_object("openloong_gripper_2f85_fix_base_usda_J_box", phase_offset=0.0)
self.add_animated_object("openloong_gripper_2f85_fix_base_usda_J1_box", phase_offset=self.bezierpath.total_length * 0.2)
```

## 效果说明

### 修复前
- 两个对象从不同位置开始
- 逐渐向同一点收敛
- 最终接触或重叠

### 修复后
- 两个对象保持固定距离
- 第一个对象在曲线起点开始
- 第二个对象在20%曲线长度处开始
- 始终保持20%曲线长度的距离

## 参数调优

### 相位偏移大小
```python
# 小偏移（对象较近）
phase_offset = curve_length * 0.1  # 10%

# 中等偏移（推荐）
phase_offset = curve_length * 0.2  # 20%

# 大偏移（对象较远）
phase_offset = curve_length * 0.5  # 50%
```

### 自动计算规则
```python
# 当前自动计算规则：每个对象间隔10%曲线长度
phase_offset = len(self.animated_objects) * (self.bezierpath.total_length * 0.1)
```

## 调试信息

修复后的代码会输出详细的调试信息：
```
joint_name: object_1, movedist: 0.05, original_curbesdist: 1.2, curbesdist: 1.2, phase_offset: 0.0
joint_name: object_2, movedist: 0.05, original_curbesdist: 1.2, curbesdist: 3.4, phase_offset: 2.2
```

## 注意事项

1. **相位偏移单位**：使用曲线长度的百分比，确保在不同曲线长度下都能正常工作
2. **重置逻辑**：重置时保持相位偏移不变
3. **性能影响**：相位偏移计算开销很小，不影响性能
4. **向后兼容**：如果不设置相位偏移，默认为0，保持原有行为

## 扩展功能

### 动态调整相位偏移
```python
# 可以在运行时调整相位偏移
obj.phase_offset = new_offset
```

### 最小距离约束
```python
# 可以添加最小距离约束
min_distance = 0.5  # 最小距离0.5米
if distance_between_objects < min_distance:
    # 调整相位偏移
    obj.phase_offset += adjustment
```

这个解决方案确保了多个对象在曲线上运动时保持固定距离，避免了收敛问题。
