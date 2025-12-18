# 修复：config_name 为 None 的问题

## 问题描述

运行脚本时：
```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'dexforce_w1' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

在 `get_config_for_robot` 函数中打印出 `config_name` 为 `None`。

## 根本原因

当用户只指定 `--robot_config` 参数时，代码会：
1. 获取 `args.agent_names`（默认值：`'openloong_gripper_2f85_fix_base_usda'`）
2. 创建配置字典：`{agent_name: config_name}`

**问题**：如果实际创建的机器人名称与 `args.agent_names` 不匹配，`create_agent` 方法中 `self._robot_configs.get(name, None)` 就会返回 `None`。

例如：
- 用户指定 `--robot_config 'dexforce_w1'`
- 代码创建字典：`{'openloong_gripper_2f85_fix_base_usda': 'dexforce_w1'}`
- 但实际机器人名称可能是 `'dexforce_w1_gripper_usda'`
- 结果：`robot_configs.get('dexforce_w1_gripper_usda', None)` 返回 `None`

## 解决方案

### 方案1：使用特殊键 `"__all__"` 表示全局配置

当用户只指定 `--robot_config` 时，使用 `{"__all__": config_name}` 作为配置字典，表示所有机器人都使用这个配置。

**修改文件**: `examples/openpi/run_dual_arm_sim_with_config.py`

```python
elif args.robot_config:
    # 所有机器人使用相同配置
    # 使用特殊键 "__all__" 表示所有机器人都使用这个配置
    robot_configs = {"__all__": args.robot_config}
```

### 方案2：在 `create_agent` 中检查 `"__all__"` 键

**修改文件**: `envs/manipulation/dual_arm_env.py`

```python
def create_agent(self, id, name):
    # ...
    # 首先检查是否有 "__all__" 特殊键
    robot_config_name = self._robot_configs.get("__all__", None)
    
    # 如果没有 "__all__"，则尝试使用机器人名称作为键
    if robot_config_name is None:
        robot_config_name = self._robot_configs.get(name, None)
    
    # 如果还是没有，但有单一配置值，使用它
    if robot_config_name is None and self._robot_configs:
        if len(self._robot_configs) == 1:
            robot_config_name = list(self._robot_configs.values())[0]
    # ...
```

### 方案3：添加调试信息

添加详细的调试信息，帮助用户了解配置传递过程：

```python
print(f"[create_agent] 创建机器人 ID={id}, name='{name}'")
print(f"[create_agent] 当前配置字典: {self._robot_configs}")
print(f"[create_agent] 从字典获取的配置: {robot_config_name}")
print(f"[create_agent] 最终使用的配置名称: {robot_config_name}")
```

## 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `examples/openpi/run_dual_arm_sim_with_config.py` | 使用 `"__all__"` 键表示全局配置 |
| `envs/manipulation/dual_arm_env.py` | 检查 `"__all__"` 键，添加调试信息 |

## 工作流程

### 之前（有问题）

```
用户: --robot_config 'dexforce_w1'
      ↓
代码: {'openloong_gripper_2f85_fix_base_usda': 'dexforce_w1'}
      ↓
create_agent(name='actual_robot_name')
      ↓
robot_configs.get('actual_robot_name', None) → None ❌
```

### 现在（已修复）

```
用户: --robot_config 'dexforce_w1'
      ↓
代码: {'__all__': 'dexforce_w1'}
      ↓
create_agent(name='actual_robot_name')
      ↓
robot_configs.get('__all__', None) → 'dexforce_w1' ✅
```

## 验证步骤

### 1. 测试基本功能

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'dexforce_w1' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

应该看到：
```
所有机器人使用配置: dexforce_w1
配置映射: {'__all__': 'dexforce_w1'}
[create_agent] 创建机器人 ID=0, name='...'
[create_agent] 当前配置字典: {'__all__': 'dexforce_w1'}
[create_agent] 从字典获取的配置: dexforce_w1
[create_agent] 最终使用的配置名称: dexforce_w1
get_config_for_robot: ... dexforce_w1
```

### 2. 测试多机器人配置

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names "robot1 robot2" \
    --robot_configs "robot1:dexforce_w1,robot2:openloong" \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

应该看到：
```
使用自定义机器人配置: {'robot1': 'dexforce_w1', 'robot2': 'openloong'}
```

### 3. 测试自动推断

```bash
python run_dual_arm_sim_with_config.py \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

应该看到：
```
未指定配置，将根据机器人名称自动推断配置
```

## 调试信息说明

现在代码会输出详细的调试信息：

1. **配置映射信息**：显示如何将命令行参数映射到配置字典
2. **创建机器人信息**：显示每个机器人的创建过程和配置查找
3. **最终配置**：显示最终使用的配置名称

这些信息可以帮助您：
- 了解配置是如何传递的
- 诊断配置不匹配的问题
- 验证配置是否正确应用

## 常见问题

### Q: 为什么使用 `"__all__"` 而不是直接使用配置名称？

A: 因为配置字典的键必须是机器人名称，但当我们不知道实际机器人名称时，使用特殊键 `"__all__"` 可以表示"所有机器人都使用这个配置"。

### Q: 如果我想为不同机器人指定不同配置怎么办？

A: 使用 `--robot_configs` 参数：
```bash
--robot_configs "robot1:dexforce_w1,robot2:openloong"
```

### Q: 调试信息太多怎么办？

A: 调试信息可以帮助诊断问题。如果不需要，可以注释掉 `print` 语句，或者添加一个 `--verbose` 参数来控制。

## 总结

✅ 修复了 `config_name` 为 `None` 的问题  
✅ 使用 `"__all__"` 特殊键支持全局配置  
✅ 添加了详细的调试信息  
✅ 保持了向后兼容性  

现在 `--robot_config` 参数应该能正常工作了！

