# Hand Type 直接映射改进

## 概述

改进了 `robot_entries` 的查找机制，现在可以通过类型名直接查找对应的入口，无需遍历匹配。

## 改进内容

### 之前的方式（遍历匹配）

```python
def get_robot_entry_by_hand_type(hand_type: str):
    if hand_type == "hand":
        # 需要遍历所有条目，查找包含 "hand" 的
        for robot_name, entry in robot_entries.items():
            if "hand" in robot_name and "fix_base" in robot_name and "no_hand" not in robot_name:
                return entry
    ...
```

**问题**:
- 需要遍历所有条目
- 字符串匹配可能出错（如 "hand" 和 "no_hand" 的冲突）
- 性能较低
- 不够直观

### 现在的方式（直接映射）

```python
hand_type_entries = {
    "hand": "envs.manipulation.robots.openloong_hand_fix_base:OpenLoongHandFixBase",
    "gripper": "envs.manipulation.robots.openloong_gripper_fix_base:OpenLoongGripperFixBase",
    "gripper_mobile": "envs.manipulation.robots.openloong_gripper_mobile_base:OpenLoongGripperMobileBase",
    "none": "envs.manipulation.robots.openloong_no_hand_fix_base:OpenLoongNoHandFixBase",
    "no_hand": "envs.manipulation.robots.openloong_no_hand_fix_base:OpenLoongNoHandFixBase",
}

def get_robot_entry_by_hand_type(hand_type: str):
    # 直接查找，O(1) 时间复杂度
    if hand_type in hand_type_entries:
        return hand_type_entries[hand_type]
    ...
```

**优势**:
- ✅ O(1) 时间复杂度，性能更好
- ✅ 直接映射，不会出错
- ✅ 代码更清晰，易于维护
- ✅ 支持更多类型（如 `gripper_mobile`）

## 数据结构

### robot_entries（基于名称）

用于基于机器人名称的查找（向后兼容）：

```python
robot_entries = {
    "openloong_hand_fix_base": "...",
    "openloong_gripper_2f85_fix_base": "...",
    "openloong_gripper_2f85_mobile_base": "...",
    "openloong_no_hand_fix_base": "...",
}
```

### hand_type_entries（基于类型）

用于基于手部类型的直接查找：

```python
hand_type_entries = {
    "hand": "envs.manipulation.robots.openloong_hand_fix_base:OpenLoongHandFixBase",
    "gripper": "envs.manipulation.robots.openloong_gripper_fix_base:OpenLoongGripperFixBase",
    "gripper_mobile": "envs.manipulation.robots.openloong_gripper_mobile_base:OpenLoongGripperMobileBase",
    "none": "envs.manipulation.robots.openloong_no_hand_fix_base:OpenLoongNoHandFixBase",
    "no_hand": "envs.manipulation.robots.openloong_no_hand_fix_base:OpenLoongNoHandFixBase",
}
```

## 使用示例

### 示例1: 直接通过类型名查找

```python
from envs.manipulation.dual_arm_env import hand_type_entries

# 直接查找
entry = hand_type_entries["hand"]
# 返回: "envs.manipulation.robots.openloong_hand_fix_base:OpenLoongHandFixBase"

entry = hand_type_entries["gripper"]
# 返回: "envs.manipulation.robots.openloong_gripper_fix_base:OpenLoongGripperFixBase"

entry = hand_type_entries["none"]
# 返回: "envs.manipulation.robots.openloong_no_hand_fix_base:OpenLoongNoHandFixBase"
```

### 示例2: 通过函数查找

```python
from envs.manipulation.dual_arm_env import get_robot_entry_by_hand_type

# 使用函数查找（内部使用直接映射）
entry = get_robot_entry_by_hand_type("hand")
# 直接返回: "envs.manipulation.robots.openloong_hand_fix_base:OpenLoongHandFixBase"

entry = get_robot_entry_by_hand_type("gripper")
# 直接返回: "envs.manipulation.robots.openloong_gripper_fix_base:OpenLoongGripperFixBase"

entry = get_robot_entry_by_hand_type("gripper_mobile")
# 直接返回: "envs.manipulation.robots.openloong_gripper_mobile_base:OpenLoongGripperMobileBase"
```

### 示例3: 在配置中使用

```python
# 配置文件中
my_config = {
    "hand_type": "hand",  # 直接使用类型名
    ...
}

# create_agent 中
hand_type = config["hand_type"]  # "hand"
entry = get_robot_entry_by_hand_type(hand_type)  # 直接查找，O(1)
```

## 性能对比

| 操作 | 之前（遍历） | 现在（直接映射） |
|------|------------|----------------|
| 时间复杂度 | O(n) | O(1) |
| 查找 "hand" | 需要遍历所有条目 | 直接字典查找 |
| 查找 "gripper" | 需要遍历所有条目 | 直接字典查找 |
| 查找 "none" | 需要遍历所有条目 | 直接字典查找 |

## 向后兼容性

✅ **完全兼容**：

1. **基于名称的查找**：`get_robot_entry(name)` 仍然使用 `robot_entries`，行为不变
2. **基于类型的查找**：`get_robot_entry_by_hand_type(hand_type)` 优先使用直接映射，如果没有找到则回退到智能匹配
3. **现有代码**：无需修改，自动享受性能提升

## 添加新类型

### 步骤1: 添加机器人类

创建新的机器人类文件，如 `openloong_custom_hand_fix_base.py`

### 步骤2: 添加到 robot_entries

```python
robot_entries = {
    ...
    "openloong_custom_hand_fix_base": "envs.manipulation.robots.openloong_custom_hand_fix_base:OpenLoongCustomHandFixBase",
}
```

### 步骤3: 添加到 hand_type_entries

```python
hand_type_entries = {
    ...
    "custom_hand": "envs.manipulation.robots.openloong_custom_hand_fix_base:OpenLoongCustomHandFixBase",
}
```

### 步骤4: 在配置中使用

```python
custom_config = {
    "hand_type": "custom_hand",
    ...
}
```

## 调试信息

运行时会输出：

```
[get_robot_entry_by_hand_type] 直接通过类型名 'hand' 找到 entry: envs.manipulation.robots.openloong_hand_fix_base:OpenLoongHandFixBase
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `envs/manipulation/dual_arm_env.py` | `hand_type_entries` 字典和 `get_robot_entry_by_hand_type` 函数 |
| `envs/manipulation/robots/configs/HAND_TYPE_CONFIG.md` | 详细使用文档 |

## 总结

✅ 添加了 `hand_type_entries` 直接映射字典  
✅ `get_robot_entry_by_hand_type` 现在使用 O(1) 直接查找  
✅ 支持更多类型（如 `gripper_mobile`）  
✅ 完全向后兼容  
✅ 性能更好，代码更清晰  

现在可以通过类型名直接查找对应的入口了！

