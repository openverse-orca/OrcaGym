# 机器人配置系统使用指南

## 概述

新的机器人配置系统允许您通过运行参数灵活地指定机器人配置，而不是硬编码在代码中。这使得支持多种机器人配置变得更加简单和灵活。

## 主要特性

1. **动态配置加载**：配置在运行时自动加载，无需修改代码
2. **灵活的配置指定**：可以通过命令行参数、配置文件或环境变量指定配置
3. **自动配置推断**：如果不显式指定配置，系统会根据机器人名称自动推断
4. **多机器人支持**：支持为不同的机器人指定不同的配置

## 可用配置

当前系统支持以下配置：

- `openloong` - OpenLoong 人形机器人配置
- `d12` - D12 机器人配置
- `hand` - 灵巧手配置
- `gripper_2f85` - 2F85 夹爪配置

### 查看所有可用配置

```bash
python examples/imitation/run_openloong_with_config.py --list_configs
```

## 使用方法

### 1. 使用默认配置（自动推断）

如果不指定配置，系统会根据机器人名称自动推断配置：

```bash
python examples/imitation/run_openloong_sim.py \
    --agent_name openloong_hand_fix_base \
    --run_mode teleoperation
```

在这个例子中，系统会自动匹配 `openloong_hand_fix_base` 到 `openloong` 配置。

### 2. 显式指定配置

为所有机器人使用相同的配置：

```bash
python examples/imitation/run_openloong_with_config.py \
    --agent_name openloong_hand_fix_base \
    --robot_config openloong \
    --run_mode teleoperation
```

### 3. 为不同机器人指定不同配置

当您有多个机器人时，可以为每个机器人指定不同的配置：

```bash
python examples/imitation/run_openloong_with_config.py \
    --agent_name "robot1 robot2" \
    --robot_configs "robot1:openloong,robot2:d12" \
    --run_mode teleoperation
```

### 4. 在 Python 代码中使用

```python
from envs.manipulation.robots.configs.robot_config_registry import (
    get_robot_config,
    list_available_configs
)

# 列出所有可用配置
configs = list_available_configs()
print(f"可用配置: {configs}")

# 获取特定配置
config = get_robot_config("openloong_hand_fix_base")
print(f"配置内容: {config}")

# 显式指定配置名称
config = get_robot_config("my_robot", config_name="openloong")
```

### 5. 在环境初始化时传递配置

```python
import gymnasium as gym
from gymnasium.envs.registration import register

# 注册环境时传递机器人配置
kwargs = {
    'frame_skip': 20,
    'reward_type': 'sparse',
    'orcagym_addr': 'localhost:50051',
    'agent_names': ['robot1', 'robot2'],
    'pico_ports': ['9870', '9871'],
    'time_step': 0.001,
    'run_mode': 'teleoperation',
    'action_type': 'end_effector',
    'ctrl_device': 'vr',
    'control_freq': 50,
    'sample_range': 0.0,
    'task_config_dict': {},
    'robot_configs': {
        'robot1': 'openloong',
        'robot2': 'd12'
    }  # 关键参数：指定机器人配置
}

gym.register(
    id='DualArmEnv-v0',
    entry_point='envs.manipulation.dual_arm_env:DualArmEnv',
    kwargs=kwargs,
)

env = gym.make('DualArmEnv-v0')
```

## 配置文件结构

每个机器人配置文件应包含以下结构：

```python
robot_config = {
    "robot_type": "dual_arm",  # 机器人类型
    "base": {
        "base_body_name": "base_link",
        "base_joint_name": "base_joint",
        "dummy_joint_name": "dummy_joint",
    },
    "right_arm": {
        "joint_names": [...],          # 关节名称列表
        "neutral_joint_values": [...], # 中性位置
        "motor_names": [...],          # 电机名称列表
        "position_names": [...],       # 位置控制器名称列表
        "ee_center_site_name": "...",  # 末端执行器中心点名称
    },
    "left_arm": {
        "joint_names": [...],
        "neutral_joint_values": [...],
        "motor_names": [...],
        "position_names": [...],
        "ee_center_site_name": "...",
    },
}
```

## 添加新配置

要添加新的机器人配置：

1. 在 `envs/manipulation/robots/configs/` 目录下创建新的配置文件，例如 `my_robot_config.py`

2. 定义配置字典：

```python
my_robot_config = {
    "robot_type": "dual_arm",
    "base": {
        "base_body_name": "base_link",
        "base_joint_name": "base_joint",
        "dummy_joint_name": "dummy_joint",
    },
    "right_arm": {
        "joint_names": ["joint1", "joint2", ...],
        "neutral_joint_values": [0.0, 0.0, ...],
        "motor_names": ["motor1", "motor2", ...],
        "position_names": ["pos1", "pos2", ...],
        "ee_center_site_name": "ee_center_r",
    },
    "left_arm": {
        "joint_names": ["joint1", "joint2", ...],
        "neutral_joint_values": [0.0, 0.0, ...],
        "motor_names": ["motor1", "motor2", ...],
        "position_names": ["pos1", "pos2", ...],
        "ee_center_site_name": "ee_center_l",
    },
}
```

3. 配置会自动被注册表发现并加载

4. 使用新配置：

```bash
python examples/imitation/run_openloong_with_config.py \
    --agent_name my_robot \
    --robot_config my_robot
```

## 配置命名规范

为了让自动推断工作得更好，建议遵循以下命名规范：

- 配置文件名：`<robot_type>_config.py`
- 配置变量名：`<robot_type>_config`
- 配置名称：`<robot_type>`

例如：
- 文件名：`openloong_config.py`
- 变量名：`openloong_config`
- 配置名称：`openloong`

## 故障排除

### 配置未找到

如果遇到 "Robot configuration not found" 错误：

1. 检查配置文件是否在正确的目录下
2. 检查配置文件的命名是否正确
3. 使用 `--list_configs` 查看所有可用配置
4. 确保配置文件中的变量名与文件名匹配

### 配置加载失败

如果配置加载失败：

1. 检查配置文件的语法是否正确
2. 确保所有必需的字段都存在
3. 查看错误信息中的详细堆栈跟踪

## 示例

### 示例1：单机器人遥操作

```bash
python examples/imitation/run_openloong_with_config.py \
    --orcagym_address localhost:50051 \
    --agent_name openloong_hand_fix_base \
    --robot_config openloong \
    --run_mode teleoperation \
    --action_type end_effector \
    --ctrl_device vr \
    --record_length 60
```

### 示例2：多机器人回放

```bash
python examples/imitation/run_openloong_with_config.py \
    --agent_name "robot1 robot2" \
    --robot_configs "robot1:openloong,robot2:d12" \
    --run_mode playback \
    --dataset ./records/demo.hdf5
```

### 示例3：训练模仿学习策略

```bash
python examples/imitation/run_openloong_with_config.py \
    --agent_name openloong_gripper_2f85_fix_base \
    --robot_config openloong \
    --run_mode imitation \
    --dataset ./records/training_data.hdf5 \
    --algo bc
```

## API 参考

### `RobotConfigRegistry`

配置注册表类，管理所有机器人配置。

#### 方法

- `register_config(config_name: str, config_dict: dict)` - 注册一个配置
- `get_config(config_name: str) -> dict` - 获取指定名称的配置
- `get_config_for_robot(robot_name: str, config_name: str = None) -> dict` - 根据机器人名称获取配置
- `list_available_configs() -> list` - 列出所有可用配置

### 便捷函数

- `get_robot_config(robot_name: str, config_name: str = None) -> dict` - 获取机器人配置
- `list_available_configs() -> list` - 列出所有可用配置

## 最佳实践

1. **优先使用显式配置**：在生产环境中，建议显式指定配置而不是依赖自动推断
2. **配置版本控制**：将配置文件纳入版本控制系统
3. **文档化配置**：为每个配置添加注释说明其用途和特点
4. **测试新配置**：添加新配置后，确保进行充分测试
5. **配置验证**：在关键应用中，添加配置验证逻辑确保配置的正确性

## 向后兼容性

新的配置系统完全向后兼容。如果您的代码不传递 `robot_configs` 参数，系统会自动根据机器人名称推断配置，行为与之前保持一致。

