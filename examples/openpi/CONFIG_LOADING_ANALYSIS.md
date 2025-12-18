# 配置加载过程分析

## 问题描述

运行脚本：
```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'd12' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

期望加载 `d12_config`，但实际加载了 `gripper_2f85_config`。

## 配置加载流程

### 1. 命令行参数解析

**文件**: `examples/openpi/run_dual_arm_sim_with_config.py`

```python
# 用户指定: --robot_config 'd12'
args.robot_config = 'd12'

# 创建配置字典
robot_configs = {"__all__": "d12"}
args.robot_configs_dict = {"__all__": "d12"}
```

**输出**:
```
所有机器人使用配置: d12
配置映射: {'__all__': 'd12'}
传递给 run_dual_arm_sim 的 robot_configs: {'__all__': 'd12'}
```

### 2. 环境初始化

**文件**: `orca_gym/scripts/dual_arm_manipulation.py`

```python
# 传递 robot_configs 到 register_env
robot_configs = {"__all__": "d12"}

# 注册环境
env_id, kwargs = register_env(..., robot_configs=robot_configs)
```

### 3. 创建机器人代理

**文件**: `envs/manipulation/dual_arm_env.py`

```python
def create_agent(self, id, name):
    # name = 'openloong_gripper_2f85_fix_base_usda'
    
    # 获取配置名称
    robot_config_name = self._robot_configs.get("__all__", None)
    # robot_config_name = "d12" ✅
    
    # 创建机器人实例
    agent = OpenLoongGripperFixBase(self, id, name, robot_config_name="d12")
```

**输出**:
```
[create_agent] 创建机器人 ID=0, name='openloong_gripper_2f85_fix_base_usda'
[create_agent] 当前配置字典: {'__all__': 'd12'}
[create_agent] 从字典获取的配置: d12
[create_agent] 最终使用的配置名称: d12
```

### 4. DualArmRobot 初始化

**文件**: `envs/manipulation/dual_arm_robot.py`

```python
class DualArmRobot(AgentBase):
    def __init__(self, env, id, name, robot_config_name="d12"):
        self._robot_config_name = "d12"  # ✅ 正确保存
        self.init_agent(id)
```

**输出**:
```
[DualArmRobot.__init__] 机器人 name='openloong_gripper_2f85_fix_base_usda', robot_config_name=d12
```

### 5. DualArmRobot.init_agent - 加载手臂配置

**文件**: `envs/manipulation/dual_arm_robot.py`

```python
def init_agent(self, id: int):
    # 调用 get_robot_config
    config = get_robot_config(self._name, self._robot_config_name)
    # config = get_robot_config('openloong_gripper_2f85_fix_base_usda', 'd12')
    # ✅ 正确返回 d12_config
    
    self._read_config(config, id)  # 读取手臂配置（base, right_arm, left_arm）
```

**输出**:
```
[DualArmRobot.init_agent] 机器人 name='openloong_gripper_2f85_fix_base_usda', robot_config_name=d12
get_config_for_robot:  openloong_gripper_2f85_fix_base_usda d12
```

**配置内容**:
```python
# d12_config 包含：
{
    "robot_type": "dual_arm",
    "base": {...},
    "right_arm": {
        "joint_names": [...],
        "motor_names": [...],
        # 没有 "position_names"（被注释掉了）
    },
    "left_arm": {...}
    # ❌ 没有 "left_hand" 和 "right_hand" 配置！
}
```

### 6. OpenLoongGripperFixBase.init_agent - 加载手部配置

**文件**: `envs/manipulation/robots/openloong_gripper_fix_base.py`

```python
class OpenLoongGripperFixBase(DualArmRobot):
    # ❌ 问题在这里：硬编码导入
    from envs.manipulation.robots.configs.gripper_2f85_config import gripper_2f85_config as config
    
    def init_agent(self, id: int):
        super().init_agent(id)  # ✅ 已加载 d12_config（手臂部分）
        
        # ❌ 问题：直接使用硬编码的 config（gripper_2f85_config）
        self._l_hand_actuator_names = [self._env.actuator(config["left_hand"]["actuator_names"][0], id)]
        # config 是 gripper_2f85_config，不是 d12_config！
```

**输出**:
```
[OpenLoongGripperFixBase.init_agent] 机器人 name='openloong_gripper_2f85_fix_base_usda', robot_config_name=d12
```

**问题**:
- `super().init_agent(id)` 正确加载了 `d12_config`（用于手臂）
- 但之后使用了硬编码的 `gripper_2f85_config`（用于手部）
- 即使指定了 `--robot_config 'd12'`，手部仍然使用 `gripper_2f85_config`

## 根本原因

### 问题1: 硬编码的手部配置导入

**文件**: `envs/manipulation/robots/openloong_gripper_fix_base.py:4`

```python
# ❌ 硬编码导入
from envs.manipulation.robots.configs.gripper_2f85_config import gripper_2f85_config as config
```

这导致无论指定什么配置，手部都使用 `gripper_2f85_config`。

### 问题2: 配置结构分离

- **手臂配置** (`d12_config`, `openloong_config`): 只包含 `base`, `right_arm`, `left_arm`
- **手部配置** (`gripper_2f85_config`, `hand_config`): 只包含 `left_hand`, `right_hand`

这种分离设计导致：
- 手臂配置和手部配置是独立的
- `OpenLoongGripperFixBase` 需要同时使用两种配置
- 当前实现硬编码了手部配置

### 问题3: 配置不完整

`d12_config` 不包含手部配置，所以即使加载了 `d12_config`，也无法用于手部。

## 配置加载流程图

```
用户命令: --robot_config 'd12'
    ↓
创建配置字典: {"__all__": "d12"}
    ↓
create_agent: robot_config_name = "d12"
    ↓
DualArmRobot.__init__: self._robot_config_name = "d12"
    ↓
DualArmRobot.init_agent:
    get_robot_config(name, "d12")
    ↓
    返回 d12_config ✅
    ↓
    _read_config(d12_config)  ← 加载手臂配置 ✅
    ↓
OpenLoongGripperFixBase.init_agent:
    super().init_agent(id)  ← 已加载 d12_config（手臂）✅
    ↓
    使用硬编码的 config 变量
    ↓
    config = gripper_2f85_config  ← 硬编码导入 ❌
    ↓
    使用 config["left_hand"]  ← 使用 gripper_2f85_config ❌
```

## 解决方案

### 方案1: 修改 OpenLoongGripperFixBase 使用父类配置

**问题**: 父类加载的配置可能不包含手部配置。

**解决**: 需要检查配置中是否有手部配置，如果没有则使用默认配置。

### 方案2: 统一配置结构

**问题**: 手臂配置和手部配置分离。

**解决**: 将手部配置合并到主配置中，或者创建组合配置。

### 方案3: 支持配置组合

**问题**: 需要同时使用手臂配置和手部配置。

**解决**: 允许配置包含 `hand_config` 字段，或者支持配置继承/组合。

## 当前状态

### ✅ 正确工作的部分

1. 命令行参数解析 ✅
2. 配置字典创建 ✅
3. 配置传递到 `create_agent` ✅
4. `DualArmRobot` 加载手臂配置 ✅

### ❌ 有问题的部分

1. `OpenLoongGripperFixBase` 硬编码手部配置 ❌
2. `d12_config` 不包含手部配置 ❌
3. 配置结构分离导致无法统一管理 ❌

## 已实施的修复方案

### ✅ 修复1: 在 DualArmRobot 中存储配置

**文件**: `envs/manipulation/dual_arm_robot.py`

```python
def init_agent(self, id: int):
    config = get_robot_config(self._name, self._robot_config_name)
    # 存储配置，以便子类访问
    self._config = config  # ← 新增
    self._read_config(config, id)
    ...
```

### ✅ 修复2: OpenLoongGripperFixBase 检查主配置

**文件**: `envs/manipulation/robots/openloong_gripper_fix_base.py`

```python
def init_agent(self, id: int):
    super().init_agent(id)
    
    # 尝试从主配置中获取手部配置，如果没有则使用默认配置
    if hasattr(self, '_config') and "left_hand" in self._config and "right_hand" in self._config:
        hand_config = self._config
        print("使用主配置中的手部配置")
    else:
        hand_config = default_hand_config  # gripper_2f85_config
        print("主配置中没有手部配置，使用默认配置")
    
    # 使用 hand_config 而不是硬编码的 config
    self._l_hand_actuator_names = [self._env.actuator(hand_config["left_hand"]["actuator_names"][0], id)]
    ...
```

### 当前行为

1. **如果主配置包含手部配置**：使用主配置中的手部配置 ✅
2. **如果主配置不包含手部配置**：使用默认的 `gripper_2f85_config` ✅

**对于 `d12_config`**：
- `d12_config` 不包含手部配置
- 因此会使用默认的 `gripper_2f85_config`
- 这是**预期的行为**，因为 `d12_config` 确实没有手部配置

### 如何让 d12_config 使用不同的手部配置？

**选项1**: 在 `d12_config.py` 中添加手部配置

```python
d12_config = {
    "robot_type": "dual_arm",
    "base": {...},
    "right_arm": {...},
    "left_arm": {...},
    # 添加手部配置
    "left_hand": {
        "actuator_names": ["d12_left_fingers_actuator"],
        "body_names": ["d12_left_pad", "d12_right_pad"],
    },
    "right_hand": {
        "actuator_names": ["d12_right_fingers_actuator"],
        "body_names": ["d12_left_pad", "d12_right_pad"],
    }
}
```

**选项2**: 创建组合配置

创建一个新的配置文件，组合 `d12_config` 和所需的手部配置。

### 长期方案：统一配置结构

1. 将手部配置合并到主配置中
2. 或者创建配置组合机制
3. 或者支持配置继承

## 验证步骤

### 1. 检查当前配置加载

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'd12' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

观察输出：
- `[create_agent] 最终使用的配置名称: d12` ✅
- `get_config_for_robot: ... d12` ✅
- 但手部仍使用 `gripper_2f85_config` ❌

### 2. 检查配置内容

```python
from envs.manipulation.robots.configs.robot_config_registry import get_robot_config

# 检查 d12_config
d12 = get_robot_config("test", config_name="d12")
print("d12_config keys:", d12.keys())
# 输出: dict_keys(['robot_type', 'base', 'right_arm', 'left_arm'])
# ❌ 没有 'left_hand' 和 'right_hand'

# 检查 gripper_2f85_config
gripper = get_robot_config("test", config_name="gripper_2f85")
print("gripper_2f85_config keys:", gripper.keys())
# 输出: dict_keys(['left_hand', 'right_hand'])
# ✅ 只有手部配置
```

## 总结

**问题根源**:
1. `OpenLoongGripperFixBase` 硬编码导入了 `gripper_2f85_config`
2. `d12_config` 不包含手部配置
3. 配置结构分离（手臂配置和手部配置分开）

**当前行为**:
- 手臂部分：正确使用 `d12_config` ✅
- 手部部分：强制使用 `gripper_2f85_config` ❌

**需要修复**:
- 修改 `OpenLoongGripperFixBase` 以支持从主配置中获取手部配置
- 或者统一配置结构，将手部配置合并到主配置中

