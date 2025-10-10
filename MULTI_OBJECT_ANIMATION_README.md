# 多对象动画系统使用说明

## 概述

现在 `step_modelanim` 函数已经支持多个对象沿曲线运动。每个对象都有独立的状态管理，可以同时或分别控制多个物体的动画。

## 主要特性

1. **多对象支持**：可以同时为多个关节/物体设置动画
2. **独立状态管理**：每个对象都有独立的位置、角度、速度等状态
3. **灵活控制**：可以动态添加/移除动画对象
4. **向后兼容**：保持原有单对象功能

## 核心类

### AnimatedObject 类
管理单个动画对象的状态：
- `joint_name`: 关节名称
- `bezier_path`: 贝塞尔路径实例
- `curbesdist`: 在曲线上的距离
- `targetpos`: 目标位置
- `lastpos`: 上一帧位置
- `last_curdir`: 上一帧角度
- `is_active`: 是否激活

## 使用方法

### 1. 基本使用（自动添加默认对象）
```python
# 创建环境时，会自动添加默认的动画对象
env = DualArmEnv(...)

# 对象会自动开始动画
```

### 2. 手动添加多个对象
```python
# 添加第一个对象
env.add_animated_object("joint_name_1")

# 添加第二个对象
env.add_animated_object("joint_name_2")

# 添加第三个对象，并指定从当前位置开始
env.add_animated_object("joint_name_3", start_from_current_position=True)
```

### 3. 移除对象
```python
# 移除指定对象
env.remove_animated_object("joint_name_1")
```

### 4. 设置全局起始模式
```python
# 所有对象都从当前位置开始
env.set_animation_start_mode(start_from_current_position=True)

# 所有对象都从曲线起点开始
env.set_animation_start_mode(start_from_current_position=False)
```

## 完整示例

```python
import numpy as np
from envs.manipulation.dual_arm_env import DualArmEnv

# 创建环境
env = DualArmEnv(
    frame_skip=1,
    reward_type="dense",
    orcagym_addr="localhost:50051",
    agent_names=["robot1"],
    pico_ports=[8000],
    time_step=0.01,
    run_mode=RunMode.TELEOPERATION,
    action_type=ActionType.END_EFFECTOR_OSC,
    ctrl_device=ControlDevice.VR,
    control_freq=50,
    sample_range=0.1,
    task_config_dict={"type": "pick_and_place"}
)

# 添加多个动画对象
env.add_animated_object("object_joint_1")
env.add_animated_object("object_joint_2") 
env.add_animated_object("object_joint_3")

# 设置所有对象从当前位置开始
env.set_animation_start_mode(start_from_current_position=True)

# 运行仿真
for step in range(1000):
    action = np.zeros(env.action_space.shape[0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 检查动画状态
    for joint_name, obj in env.animated_objects.items():
        if obj.is_active:
            print(f"对象 {joint_name}: 距离={obj.curbesdist:.3f}/{obj.bezier_path.total_length:.3f}")
```

## 高级用法

### 1. 为不同对象设置不同的起始模式
```python
# 添加对象1，从当前位置开始
env.add_animated_object("joint_1", start_from_current_position=True)

# 添加对象2，从曲线起点开始
env.add_animated_object("joint_2", start_from_current_position=False)
```

### 2. 动态控制对象
```python
# 在运行时添加新对象
env.add_animated_object("new_object_joint")

# 在运行时移除对象
env.remove_animated_object("old_object_joint")
```

### 3. 检查对象状态
```python
for joint_name, obj in env.animated_objects.items():
    print(f"对象: {joint_name}")
    print(f"  激活状态: {obj.is_active}")
    print(f"  当前位置: {obj.lastpos}")
    print(f"  目标位置: {obj.targetpos}")
    print(f"  曲线距离: {obj.curbesdist}")
    print(f"  是否重置: {obj.breset}")
```

## 注意事项

1. **关节名称**：确保关节名称在仿真环境中存在
2. **性能考虑**：对象数量过多可能影响性能
3. **错误处理**：系统会自动跳过无效的关节
4. **路径共享**：所有对象共享同一个贝塞尔路径

## 技术细节

### 状态管理
每个 `AnimatedObject` 实例维护：
- 位置状态（当前位置、目标位置、上一帧位置）
- 角度状态（当前角度、累积角度、角度跳跃检测）
- 动画状态（是否激活、是否重置、曲线距离）

### 动画循环
1. 遍历所有激活的动画对象
2. 获取每个对象的当前位置和角度
3. 计算移动距离和角度变化
4. 更新曲线上的位置
5. 计算目标位置和速度
6. 设置关节速度
7. 检查是否需要重置

### 错误处理
- 自动跳过无效的关节名称
- 捕获并报告处理错误
- 继续处理其他对象

## 迁移指南

### 从单对象到多对象
1. 现有代码无需修改，默认行为保持不变
2. 使用 `add_animated_object()` 添加更多对象
3. 使用 `remove_animated_object()` 移除不需要的对象

### 性能优化
1. 只添加需要的对象
2. 及时移除不需要的对象
3. 考虑使用不同的路径文件
