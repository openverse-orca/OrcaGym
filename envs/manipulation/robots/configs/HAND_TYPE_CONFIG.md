# Hand Type 配置说明

## 概述

现在手部配置支持 `hand_type` 参数，通过该参数可以自动确定使用哪个 `robot_entry`（机器人类）。

## 配置结构

### hand_config.py

```python
hand_config = {
    "hand_type": "hand",  # 手部类型：'hand' 表示灵巧手
    "left_hand": {...},
    "right_hand": {...},
}
```

### gripper_2f85_config.py

```python
gripper_2f85_config = {
    "hand_type": "gripper",  # 手部类型：'gripper' 表示夹爪
    "left_hand": {...},
    "right_hand": {...},
}
```

## hand_type 值

| hand_type | 说明 | 对应的 robot_entry | 直接映射 |
|-----------|------|-------------------|---------|
| `"hand"` | 灵巧手 | `OpenLoongHandFixBase` | ✅ `hand_type_entries["hand"]` |
| `"gripper"` | 夹爪（固定底座） | `OpenLoongGripperFixBase` | ✅ `hand_type_entries["gripper"]` |
| `"gripper_mobile"` | 夹爪（移动底座） | `OpenLoongGripperMobileBase` | ✅ `hand_type_entries["gripper_mobile"]` |
| `"none"` 或 `"no_hand"` | 无手 | `OpenLoongNoHandFixBase` | ✅ `hand_type_entries["none"]` |

**注意**: 现在可以直接通过类型名在 `hand_type_entries` 字典中查找，无需遍历匹配！

## 工作原理

### 1. 配置加载流程

```
用户命令: --robot_config 'hand'
    ↓
加载配置: hand_config
    ↓
检查配置: hand_config["hand_type"] = "hand"
    ↓
查找 robot_entry: get_robot_entry_by_hand_type("hand")
    ↓
返回: "envs.manipulation.robots.openloong_hand_fix_base:OpenLoongHandFixBase"
    ↓
创建机器人: OpenLoongHandFixBase(...)
```

### 2. 函数说明

#### `get_robot_entry_by_hand_type(hand_type, prefer_fix_base=True)`

根据 `hand_type` **直接**查找对应的 `robot_entry`。

**参数**:
- `hand_type`: 手部类型
  - `"hand"`: 灵巧手
  - `"gripper"`: 夹爪（固定底座，默认）
  - `"gripper_mobile"`: 移动底座夹爪
  - `"none"` 或 `"no_hand"`: 无手
- `prefer_fix_base`: 是否优先选择 `fix_base`（仅对 `gripper` 有效，如果指定了 `"gripper_mobile"` 则忽略）

**返回值**:
- `robot_entry` 字符串，如果未找到则返回 `None`

**逻辑**:
- **直接查找**: 首先在 `hand_type_entries` 字典中直接查找类型名
- **智能匹配**: 如果没有直接匹配，根据 `prefer_fix_base` 参数智能选择（仅对 `"gripper"`）

**hand_type_entries 映射**:
```python
hand_type_entries = {
    "hand": "envs.manipulation.robots.openloong_hand_fix_base:OpenLoongHandFixBase",
    "gripper": "envs.manipulation.robots.openloong_gripper_fix_base:OpenLoongGripperFixBase",
    "gripper_mobile": "envs.manipulation.robots.openloong_gripper_mobile_base:OpenLoongGripperMobileBase",
    "none": "envs.manipulation.robots.openloong_no_hand_fix_base:OpenLoongNoHandFixBase",
    "no_hand": "envs.manipulation.robots.openloong_no_hand_fix_base:OpenLoongNoHandFixBase",
}
```

### 3. create_agent 流程

```python
def create_agent(self, id, name):
    # 1. 获取配置名称
    robot_config_name = self._robot_configs.get("__all__", None)
    
    # 2. 如果提供了配置，尝试从配置中获取 hand_type
    if robot_config_name:
        config = get_robot_config(name, robot_config_name)
        if "hand_type" in config:
            hand_type = config["hand_type"]
            entry = get_robot_entry_by_hand_type(hand_type)
    
    # 3. 如果没有通过 hand_type 找到，使用基于 name 的方式
    if entry is None:
        entry = get_robot_entry(name)
    
    # 4. 创建机器人实例
    ...
```

## 使用示例

### 示例1: 使用 hand 配置

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'hand' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

**结果**:
- 加载 `hand_config`
- `hand_type = "hand"`
- 选择 `OpenLoongHandFixBase`
- 使用 `hand_config` 中的手部配置

### 示例2: 使用 gripper 配置

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'gripper_2f85' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

**结果**:
- 加载 `gripper_2f85_config`
- `hand_type = "gripper"`
- 选择 `OpenLoongGripperFixBase`（优先 fix_base）
- 使用 `gripper_2f85_config` 中的手部配置

### 示例3: 使用无手配置

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'no_hand' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

**结果**:
- 加载 `no_hand_config`
- `hand_type = "none"`
- 选择 `OpenLoongNoHandFixBase`
- 不初始化手部执行器

### 示例4: 配置中包含 hand_type

如果主配置（如 `openloong_config`）中包含手部配置和 `hand_type`：

```python
openloong_config = {
    "robot_type": "dual_arm",
    "base": {...},
    "right_arm": {...},
    "left_arm": {...},
    "hand_type": "hand",  # ← 添加 hand_type
    "left_hand": {...},   # ← 添加手部配置
    "right_hand": {...},  # ← 添加手部配置
}
```

**结果**:
- 加载 `openloong_config`
- `hand_type = "hand"`
- 选择 `OpenLoongHandFixBase`
- 使用 `openloong_config` 中的手部配置

## 向后兼容性

✅ **完全兼容**：如果不提供 `hand_type` 或配置中没有 `hand_type`，系统会使用原来的基于 `name` 的方式查找 `robot_entry`。

## 添加新配置

### 步骤1: 创建配置文件

```python
# my_hand_config.py
my_hand_config = {
    "hand_type": "hand",  # 或 "gripper"
    "left_hand": {...},
    "right_hand": {...},
}
```

### 步骤2: 注册配置

配置会自动被 `robot_config_registry` 发现和加载。

### 步骤3: 使用配置

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'my_hand' \
    ...
```

系统会自动根据 `hand_type` 选择正确的 `robot_entry`。

## 调试信息

运行时会输出以下调试信息：

```
[create_agent] 创建机器人 ID=0, name='...'
[create_agent] 从字典获取的配置: hand
[create_agent] 从配置中获取 hand_type: hand
[create_agent] 根据 hand_type='hand' 选择 robot_entry: envs.manipulation.robots.openloong_hand_fix_base:OpenLoongHandFixBase
```

## 注意事项

1. **hand_type 优先级**: 如果配置中有 `hand_type`，会优先使用它来确定 `robot_entry`，而不是基于 `name` 的方式。

2. **配置完整性**: 确保配置中包含 `hand_type` 字段，否则会回退到基于 `name` 的方式。

3. **robot_entry 匹配**: 确保 `robot_entries` 字典中有对应的条目，否则会抛出错误。

4. **gripper 优先级**: 对于 `gripper` 类型，默认优先选择 `fix_base`，如果需要 `mobile_base`，可以修改 `prefer_fix_base` 参数。

## 无手类型说明

### 使用场景

无手类型适用于：
- 只需要手臂操作，不需要抓取功能的场景
- 推、拉、拨等不需要抓取的操作
- 测试和调试手臂控制算法

### 配置示例

```python
no_hand_config = {
    "hand_type": "none",  # 或 "no_hand"
    # 注意：不包含 left_hand 和 right_hand 字段
}
```

### 机器人类

`OpenLoongNoHandFixBase` 类：
- 继承自 `DualArmRobot`
- 不初始化任何手部执行器
- 手部相关方法提供空实现
- 力反馈返回零值

### 手部方法

无手类型的手部方法都是空实现：
- `set_gripper_ctrl_l()`: 空实现
- `set_gripper_ctrl_r()`: 空实现
- `set_l_hand_actuator_ctrl()`: 空实现
- `set_r_hand_actuator_ctrl()`: 空实现
- `update_force_feedback()`: 发送零力反馈

## 相关文件

| 文件 | 说明 |
|------|------|
| `envs/manipulation/robots/configs/hand_config.py` | 灵巧手配置（hand_type="hand"） |
| `envs/manipulation/robots/configs/gripper_2f85_config.py` | 夹爪配置（hand_type="gripper"） |
| `envs/manipulation/robots/configs/no_hand_config.py` | 无手配置（hand_type="none"） |
| `envs/manipulation/robots/openloong_no_hand_fix_base.py` | 无手机器人类 |
| `envs/manipulation/dual_arm_env.py` | `get_robot_entry_by_hand_type` 函数和 `create_agent` 方法 |

## 总结

✅ 支持通过 `hand_type` 参数自动确定 `robot_entry`  
✅ 保持向后兼容性  
✅ 提供清晰的调试信息  
✅ 支持灵活配置扩展  

现在您可以通过在配置中添加 `hand_type` 参数来控制使用哪个机器人类！

