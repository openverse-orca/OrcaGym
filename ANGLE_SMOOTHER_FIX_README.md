# AngleSmoother 多对象共享问题修复说明

## 问题描述

在多对象动画系统中，所有 `AnimatedObject` 实例共享同一个 `BezierPath` 实例，而 `BezierPath` 中的 `AngleSmoother` 也是单例的。这导致：

1. **状态污染**：一个对象的角度跳跃会影响其他对象的角度平滑
2. **角度偏移混乱**：不同对象的角度偏移会相互干扰
3. **不可预测的行为**：角度平滑逻辑变得不可预测

## 根本原因

```python
# 问题代码结构：
class BezierPath:
    def __init__(self, json_path):
        self.smoother = AngleSmoother()  # 单例，被所有对象共享

# 所有 AnimatedObject 使用同一个 BezierPath 实例
self.bezierpath = BezierPath("path.json")  # 共享实例
```

## 解决方案

### 核心思想
为每个 `AnimatedObject` 创建独立的 `AngleSmoother` 实例，避免状态共享。

### 主要修改

#### 1. AnimatedObject 类增强
```python
class AnimatedObject:
    def __init__(self, joint_name: str, bezier_path: BezierPath, start_from_current_position: bool = True):
        # ... 其他属性 ...
        
        # 为每个对象创建独立的 AngleSmoother
        from bezierdata import AngleSmoother
        self.angle_smoother = AngleSmoother()
```

#### 2. 重置时清理状态
```python
def reset(self, current_pos=None):
    # ... 其他重置逻辑 ...
    
    # 重置独立的 AngleSmoother
    self.angle_smoother.reset()
```

#### 3. BezierPath 支持禁用内置 AngleSmoother
```python
def update_position(self, current_distance, speed, dt, use_angle_smoother=True):
    # ... 位置计算 ...
    
    if use_angle_smoother:
        # 使用内置的 AngleSmoother（向后兼容）
        direction = [0,0, self.smoother.smooth_angle(direction[2])]
    else:
        # 返回原始方向，让调用者自己处理角度平滑
        direction = [0, 0, direction[2]]
```

#### 4. 在 step_modelanim 中使用独立 AngleSmoother
```python
# 更新位置（禁用内置的 AngleSmoother）
pos, distance, direction = obj.bezier_path.update_position(
    obj.curbesdist, self.animspeed, passtime, use_angle_smoother=False
)

# 使用对象独立的 AngleSmoother 处理角度
smoothed_direction = [0, 0, obj.angle_smoother.smooth_angle(direction[2])]

# 计算角速度（使用平滑后的角度）
dirdiff = smoothed_direction - curdir_unwrapped
```

## 修复效果

### 修复前
- 所有对象共享同一个 `AngleSmoother` 状态
- 对象A的角度跳跃影响对象B
- 角度偏移累积和混乱
- 不可预测的角度平滑行为

### 修复后
- 每个对象有独立的 `AngleSmoother` 状态
- 对象间角度平滑完全独立
- 角度偏移不会相互干扰
- 可预测和稳定的角度平滑行为

## 技术细节

### 状态隔离
```python
# 每个对象维护独立的状态
obj1.angle_smoother.prev_angle = angle1
obj1.angle_smoother.offset = offset1

obj2.angle_smoother.prev_angle = angle2  # 独立状态
obj2.angle_smoother.offset = offset2     # 独立状态
```

### 向后兼容
- `BezierPath.update_position()` 默认使用内置 `AngleSmoother`
- 通过 `use_angle_smoother=False` 参数禁用
- 现有单对象代码无需修改

### 内存开销
- 每个 `AnimatedObject` 增加一个 `AngleSmoother` 实例
- 开销很小：每个实例只有 2 个浮点数属性
- 对于 10 个对象，额外内存开销 < 1KB

## 使用示例

### 基本使用（自动处理）
```python
# 创建多个对象，每个都有独立的 AngleSmoother
env.add_animated_object("object_1")
env.add_animated_object("object_2")
env.add_animated_object("object_3")

# 角度平滑完全独立，不会相互干扰
```

### 调试角度平滑
```python
# 可以单独调试每个对象的角度平滑
for joint_name, obj in env.animated_objects.items():
    print(f"对象 {joint_name} 的角度平滑状态:")
    print(f"  prev_angle: {obj.angle_smoother.prev_angle}")
    print(f"  offset: {obj.angle_smoother.offset}")
```

## 注意事项

1. **状态独立性**：每个对象的角度平滑状态完全独立
2. **重置行为**：重置对象时会同时重置其 `AngleSmoother`
3. **内存管理**：移除对象时会自动清理其 `AngleSmoother`
4. **性能影响**：每个对象独立处理角度平滑，性能影响很小

## 验证方法

### 检查状态独立性
```python
# 运行一段时间后检查状态
for joint_name, obj in env.animated_objects.items():
    print(f"对象 {joint_name}: prev_angle={obj.angle_smoother.prev_angle}, offset={obj.angle_smoother.offset}")
```

### 观察角度平滑效果
- 每个对象的角度变化应该平滑
- 不同对象的角度跳跃不会相互影响
- 角度偏移不会累积或混乱

这个修复确保了多对象动画系统中角度平滑的独立性和稳定性。
