# 修复：robot_config_name 参数传递问题

## 问题描述

执行脚本后，虽然 `create_agent` 中正确获取到了配置 `dexforce_w1`，但在 `get_config_for_robot` 中 `config_name` 仍然是 `None`。

输出示例：
```
[create_agent] 创建机器人 ID=0, name='openloong_gripper_2f85_fix_base_usda'
[create_agent] 当前配置字典: {'__all__': 'dexforce_w1'}
[create_agent] 从字典获取的配置: dexforce_w1
[create_agent] 最终使用的配置名称: dexforce_w1
get_config_for_robot:  openloong_gripper_2f85_fix_base_usda None  ← 问题在这里！
```

## 根本原因

问题出在机器人类（如 `OpenLoongGripperFixBase`）的 `__init__` 方法签名上：

**问题代码**：
```python
class OpenLoongGripperFixBase(DualArmRobot):
    def __init__(self, env: DualArmEnv, id: int, name: str) -> None:  # ← 缺少 robot_config_name 参数
        super().__init__(env, id, name)  # ← 没有传递 robot_config_name
```

当 `create_agent` 尝试传递 `robot_config_name` 参数时：
```python
agent = class_type(self, id, name, robot_config_name=robot_config_name)
```

由于 `OpenLoongGripperFixBase.__init__` 不接受 `robot_config_name` 参数，会抛出 `TypeError`，然后代码回退到：
```python
except TypeError:
    agent = class_type(self, id, name)  # ← 不传递 robot_config_name
```

结果：`robot_config_name` 没有被传递，导致 `self._robot_config_name` 为 `None`。

## 解决方案

修改所有机器人类的 `__init__` 方法，添加 `robot_config_name` 参数并传递给父类。

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `envs/manipulation/robots/openloong_gripper_fix_base.py` | 添加 `robot_config_name` 参数 |
| `envs/manipulation/robots/openloong_gripper_mobile_base.py` | 添加 `robot_config_name` 参数 |
| `envs/manipulation/robots/openloong_hand_fix_base.py` | 添加 `robot_config_name` 参数 |
| `envs/manipulation/dual_arm_robot.py` | 添加调试信息 |

### 修改示例

**修改前**：
```python
class OpenLoongGripperFixBase(DualArmRobot):
    def __init__(self, env: DualArmEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)
```

**修改后**：
```python
class OpenLoongGripperFixBase(DualArmRobot):
    def __init__(self, env: DualArmEnv, id: int, name: str, robot_config_name: str = None) -> None:
        super().__init__(env, id, name, robot_config_name=robot_config_name)
```

## 工作流程

### 之前（有问题）

```
create_agent 获取配置: 'dexforce_w1'
      ↓
尝试传递: OpenLoongGripperFixBase(env, id, name, robot_config_name='dexforce_w1')
      ↓
TypeError: __init__() got an unexpected keyword argument 'robot_config_name'
      ↓
回退到: OpenLoongGripperFixBase(env, id, name)
      ↓
robot_config_name 丢失 → None ❌
```

### 现在（已修复）

```
create_agent 获取配置: 'dexforce_w1'
      ↓
传递: OpenLoongGripperFixBase(env, id, name, robot_config_name='dexforce_w1')
      ↓
成功创建，robot_config_name='dexforce_w1' ✅
      ↓
传递给父类: DualArmRobot(env, id, name, robot_config_name='dexforce_w1')
      ↓
self._robot_config_name = 'dexforce_w1' ✅
      ↓
get_config_for_robot(name, 'dexforce_w1') ✅
```

## 验证步骤

### 1. 运行脚本

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'dexforce_w1' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

### 2. 检查输出

应该看到：
```
[create_agent] 创建机器人 ID=0, name='openloong_gripper_2f85_fix_base_usda'
[create_agent] 当前配置字典: {'__all__': 'dexforce_w1'}
[create_agent] 从字典获取的配置: dexforce_w1
[create_agent] 最终使用的配置名称: dexforce_w1
[DualArmRobot.__init__] 机器人 name='openloong_gripper_2f85_fix_base_usda', robot_config_name=dexforce_w1  ← 应该显示配置名称
[DualArmRobot.init_agent] 机器人 name='openloong_gripper_2f85_fix_base_usda', robot_config_name=dexforce_w1  ← 应该显示配置名称
get_config_for_robot:  openloong_gripper_2f85_fix_base_usda dexforce_w1  ← 不再是 None！
```

## 调试信息说明

现在代码会在以下位置输出调试信息：

1. **create_agent**: 显示配置字典和获取的配置名称
2. **DualArmRobot.__init__**: 显示接收到的 `robot_config_name` 参数
3. **DualArmRobot.init_agent**: 显示调用 `get_robot_config` 时的参数
4. **get_config_for_robot**: 显示最终查找配置时的参数

这些信息可以帮助您：
- 追踪配置传递的完整路径
- 诊断参数传递问题
- 验证配置是否正确应用

## 注意事项

### 1. 向后兼容性

所有修改都保持了向后兼容性：
- `robot_config_name` 参数有默认值 `None`
- 如果不传递该参数，行为与之前完全一致（自动推断配置）

### 2. 其他机器人类

如果将来添加新的机器人类，请确保：
- `__init__` 方法包含 `robot_config_name: str = None` 参数
- 将 `robot_config_name` 传递给父类 `super().__init__()`

### 3. 调试信息

调试信息在生产环境中可能不需要。如果需要移除，可以：
- 注释掉 `print` 语句
- 使用日志级别控制
- 添加 `--verbose` 参数

## 相关文件

| 文件 | 说明 |
|------|------|
| `envs/manipulation/robots/openloong_gripper_fix_base.py` | OpenLoong 固定底座夹爪机器人 |
| `envs/manipulation/robots/openloong_gripper_mobile_base.py` | OpenLoong 移动底座夹爪机器人 |
| `envs/manipulation/robots/openloong_hand_fix_base.py` | OpenLoong 固定底座灵巧手机器人 |
| `envs/manipulation/dual_arm_robot.py` | 双臂机器人基类 |

## 总结

✅ 修复了所有机器人类的 `__init__` 方法签名  
✅ 确保 `robot_config_name` 参数正确传递  
✅ 添加了详细的调试信息  
✅ 保持了向后兼容性  

现在 `robot_config_name` 应该能正确传递到 `get_config_for_robot` 函数了！

