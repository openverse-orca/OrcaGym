# 无手类型配置说明

## 概述

无手类型（`hand_type="none"` 或 `"no_hand"`）用于没有手部执行器的双臂机器人。该类机器人只包含手臂部分，适用于不需要抓取功能的场景。

## 使用场景

- **推拉操作**：只需要手臂推拉物体，不需要抓取
- **拨动操作**：拨动开关、按钮等
- **测试调试**：测试手臂控制算法，不需要手部干扰
- **简化操作**：某些任务只需要精确的手臂位置控制

## 配置结构

### no_hand_config.py

```python
no_hand_config = {
    "hand_type": "none",  # 或 "no_hand"
    # 注意：不包含 left_hand 和 right_hand 字段
}
```

### 在主配置中使用

如果主配置（如 `openloong_config`）要使用无手类型：

```python
openloong_config = {
    "robot_type": "dual_arm",
    "base": {...},
    "right_arm": {...},
    "left_arm": {...},
    "hand_type": "none",  # ← 指定无手类型
    # 不包含 left_hand 和 right_hand
}
```

## 机器人类

### OpenLoongNoHandFixBase

**位置**: `envs/manipulation/robots/openloong_no_hand_fix_base.py`

**特点**:
- 继承自 `DualArmRobot`
- 不初始化任何手部执行器
- 手部相关方法提供空实现
- 力反馈返回零值

**实现的方法**:

| 方法 | 实现 |
|------|------|
| `set_gripper_ctrl_l()` | 空实现，设置 `_grasp_value_l = 0.0` |
| `set_gripper_ctrl_r()` | 空实现，设置 `_grasp_value_r = 0.0` |
| `set_l_hand_actuator_ctrl()` | 空实现 |
| `set_r_hand_actuator_ctrl()` | 空实现 |
| `update_force_feedback()` | 发送零力反馈 `(0.0, 0.0)` |
| `set_wheel_ctrl()` | 空实现 |

## 使用方法

### 方法1: 使用 no_hand 配置

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'no_hand' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

### 方法2: 在主配置中指定 hand_type

如果主配置包含 `hand_type="none"`：

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'your_config_with_no_hand' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

## 工作流程

```
用户命令: --robot_config 'no_hand'
    ↓
加载配置: no_hand_config
    ↓
检查 hand_type: "none"
    ↓
查找 robot_entry: get_robot_entry_by_hand_type("none")
    ↓
返回: "envs.manipulation.robots.openloong_no_hand_fix_base:OpenLoongNoHandFixBase"
    ↓
创建机器人: OpenLoongNoHandFixBase(...)
    ↓
初始化: 只初始化手臂，不初始化手部 ✅
```

## 调试输出

运行时会看到：

```
[create_agent] 创建机器人 ID=0, name='...'
[create_agent] 从字典获取的配置: no_hand
[create_agent] 从配置中获取 hand_type: none
[create_agent] 根据 hand_type='none' 选择 robot_entry: envs.manipulation.robots.openloong_no_hand_fix_base:OpenLoongNoHandFixBase
[OpenLoongNoHandFixBase.init_agent] 机器人 name='...', robot_config_name=no_hand
[OpenLoongNoHandFixBase.init_agent] 无手类型机器人，不初始化手部执行器
```

## 与其他类型的对比

| 特性 | hand | gripper | none |
|------|------|---------|------|
| 手部执行器 | ✅ 有（11个关节） | ✅ 有（1个执行器） | ❌ 无 |
| 抓取功能 | ✅ 支持 | ✅ 支持 | ❌ 不支持 |
| 力反馈 | ✅ 支持 | ✅ 支持 | ❌ 零值 |
| 适用场景 | 精细操作 | 简单抓取 | 推拉操作 |

## 注意事项

1. **动作空间**: 无手类型的动作空间不包含手部控制，只有手臂控制

2. **观测空间**: 观测空间中的 `grasp_value_l` 和 `grasp_value_r` 始终为 0.0

3. **力反馈**: 如果使用 VR 设备，力反馈会始终为零

4. **配置兼容性**: 无手配置不需要包含 `left_hand` 和 `right_hand` 字段

5. **向后兼容**: 如果配置中没有 `hand_type`，系统会使用基于 `name` 的方式查找 `robot_entry`

## 示例：创建自定义无手配置

```python
# my_no_hand_config.py
my_no_hand_config = {
    "robot_type": "dual_arm",
    "base": {
        "base_body_name": "base_link",
        "base_joint_name": "base_joint",
        "dummy_joint_name": "dummy_joint",
    },
    "right_arm": {
        "joint_names": [...],
        "neutral_joint_values": [...],
        "motor_names": [...],
        "position_names": [...],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": [...],
        "neutral_joint_values": [...],
        "motor_names": [...],
        "position_names": [...],
        "ee_center_site_name": "ee_center_site",
    },
    "hand_type": "none",  # ← 指定无手类型
    # 不包含 left_hand 和 right_hand
}
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `envs/manipulation/robots/configs/no_hand_config.py` | 无手配置示例 |
| `envs/manipulation/robots/openloong_no_hand_fix_base.py` | 无手机器人类实现 |
| `envs/manipulation/dual_arm_env.py` | `get_robot_entry_by_hand_type` 函数支持 "none" 和 "no_hand" |

## 总结

✅ 支持无手类型配置  
✅ 自动选择 `OpenLoongNoHandFixBase` 类  
✅ 手部方法提供空实现  
✅ 适用于不需要抓取功能的场景  
✅ 完全向后兼容  

现在您可以使用无手类型来处理只需要手臂操作的任务了！

