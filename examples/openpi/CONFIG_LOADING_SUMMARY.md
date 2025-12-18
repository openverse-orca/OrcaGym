# 配置加载过程总结

## 问题

运行 `--robot_config 'd12'` 时，手部配置加载了 `gripper_2f85_config` 而不是 `d12_config` 中的配置。

## 原因分析

### 1. 配置结构分离

- **手臂配置** (`d12_config`, `openloong_config`): 只包含 `base`, `right_arm`, `left_arm`
- **手部配置** (`gripper_2f85_config`): 只包含 `left_hand`, `right_hand`

`d12_config` **不包含手部配置**，所以无法用于手部。

### 2. 硬编码的手部配置

`OpenLoongGripperFixBase` 类硬编码导入了 `gripper_2f85_config`：

```python
# 旧代码
from envs.manipulation.robots.configs.gripper_2f85_config import gripper_2f85_config as config

def init_agent(self, id: int):
    super().init_agent(id)  # 加载 d12_config（手臂）
    # 使用硬编码的 config（gripper_2f85_config）
    self._l_hand_actuator_names = [self._env.actuator(config["left_hand"]["actuator_names"][0], id)]
```

## 配置加载流程

```
用户命令: --robot_config 'd12'
    ↓
1. 创建配置字典: {"__all__": "d12"}
    ↓
2. create_agent: robot_config_name = "d12"
    ↓
3. DualArmRobot.__init__: self._robot_config_name = "d12"
    ↓
4. DualArmRobot.init_agent:
   - get_robot_config(name, "d12") → 返回 d12_config ✅
   - self._config = d12_config  ← 存储配置 ✅
   - _read_config(d12_config)  ← 加载手臂配置 ✅
    ↓
5. OpenLoongGripperFixBase.init_agent:
   - super().init_agent(id)  ← 已加载 d12_config（手臂）✅
   - 检查 self._config 是否有手部配置
   - d12_config 没有手部配置 ❌
   - 使用默认的 gripper_2f85_config ✅（回退机制）
```

## 已实施的修复

### 修复1: 存储配置

**文件**: `envs/manipulation/dual_arm_robot.py`

```python
def init_agent(self, id: int):
    config = get_robot_config(self._name, self._robot_config_name)
    self._config = config  # ← 存储配置，供子类访问
    self._read_config(config, id)
    ...
```

### 修复2: 智能配置选择

**文件**: `envs/manipulation/robots/openloong_gripper_fix_base.py`

```python
def init_agent(self, id: int):
    super().init_agent(id)
    
    # 优先使用主配置中的手部配置，如果没有则使用默认配置
    if hasattr(self, '_config') and "left_hand" in self._config and "right_hand" in self._config:
        hand_config = self._config  # 使用主配置
    else:
        hand_config = default_hand_config  # 使用默认配置（gripper_2f85_config）
    
    # 使用 hand_config 而不是硬编码的 config
    self._l_hand_actuator_names = [self._env.actuator(hand_config["left_hand"]["actuator_names"][0], id)]
    ...
```

## 当前行为

### 场景1: 使用 d12_config

```bash
--robot_config 'd12'
```

**结果**:
- ✅ 手臂配置：使用 `d12_config`
- ✅ 手部配置：使用 `gripper_2f85_config`（因为 `d12_config` 没有手部配置）

**输出**:
```
[OpenLoongGripperFixBase.init_agent] 主配置中没有手部配置，使用默认配置: gripper_2f85_config
```

### 场景2: 使用包含手部配置的配置

如果将来创建包含手部配置的配置：

```python
my_config = {
    "robot_type": "dual_arm",
    "base": {...},
    "right_arm": {...},
    "left_arm": {...},
    "left_hand": {...},  # ← 有手部配置
    "right_hand": {...},  # ← 有手部配置
}
```

**结果**:
- ✅ 手臂配置：使用 `my_config`
- ✅ 手部配置：使用 `my_config` 中的手部配置

**输出**:
```
[OpenLoongGripperFixBase.init_agent] 使用主配置中的手部配置
```

## 如何为 d12_config 添加手部配置？

### 方法1: 修改 d12_config.py

在 `envs/manipulation/robots/configs/d12_config.py` 中添加：

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

### 方法2: 创建组合配置

创建一个新配置文件 `d12_with_custom_hand_config.py`：

```python
from .d12_config import d12_config
from .gripper_2f85_config import gripper_2f85_config

d12_with_custom_hand_config = {
    **d12_config,
    **gripper_2f85_config,  # 合并手部配置
}
```

## 验证

运行脚本并观察输出：

```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'd12' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

应该看到：
```
[create_agent] 最终使用的配置名称: d12
[DualArmRobot.init_agent] robot_config_name=d12
get_config_for_robot: ... d12
[OpenLoongGripperFixBase.init_agent] 主配置中没有手部配置，使用默认配置: gripper_2f85_config
```

## 总结

✅ **已修复**:
- `DualArmRobot` 现在存储配置供子类访问
- `OpenLoongGripperFixBase` 优先使用主配置中的手部配置
- 如果主配置没有手部配置，回退到默认配置

✅ **当前行为**:
- `d12_config` 不包含手部配置，所以使用默认的 `gripper_2f85_config`
- 这是**预期的行为**，因为 `d12_config` 确实没有手部配置

✅ **未来扩展**:
- 如果配置包含手部配置，会自动使用
- 支持灵活的配置组合

## 相关文件

| 文件 | 说明 |
|------|------|
| `envs/manipulation/dual_arm_robot.py` | 存储配置供子类访问 |
| `envs/manipulation/robots/openloong_gripper_fix_base.py` | 智能选择手部配置 |
| `envs/manipulation/robots/configs/d12_config.py` | D12 机器人配置（无手部配置） |
| `envs/manipulation/robots/configs/gripper_2f85_config.py` | 默认手部配置 |

