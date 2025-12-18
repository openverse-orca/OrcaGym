# 机器人配置系统改进 - 修改说明

## 修改日期
2025-12-18

## 问题描述
之前的机器人配置是硬编码在 `dual_arm_robot.py` 中的，所有机器人都使用相同的配置。这导致：
1. 无法灵活指定不同的机器人配置
2. 添加新机器人配置需要修改代码
3. 不支持通过运行参数动态配置机器人

## 解决方案
实现了一个灵活的机器人配置注册表系统，支持：
1. 动态加载和注册机器人配置
2. 通过命令行参数指定配置
3. 自动推断配置（向后兼容）
4. 为不同机器人指定不同配置

## 修改的文件

### 1. 新增文件

#### `envs/manipulation/robots/configs/robot_config_registry.py`
**描述**: 机器人配置注册表模块

**主要功能**:
- `RobotConfigRegistry` 类：管理所有机器人配置
- `get_robot_config()` 函数：获取机器人配置
- `list_available_configs()` 函数：列出所有可用配置
- 自动发现和加载配置文件

**关键特性**:
- 支持显式指定配置名称
- 支持根据机器人名称自动推断配置
- 提供友好的错误信息
- 完全向后兼容

#### `envs/manipulation/robots/configs/README.md`
**描述**: 配置系统使用指南

**内容**:
- 系统概述和特性介绍
- 详细的使用方法和示例
- 配置文件结构说明
- 添加新配置的步骤
- API 参考文档
- 故障排除指南
- 最佳实践建议

#### `envs/manipulation/robots/configs/test_config_registry.py`
**描述**: 配置系统测试脚本

**测试内容**:
- 列出所有可用配置
- 通过名称获取配置
- 自动推断配置
- 无效配置处理
- 配置内容完整性检查

#### `examples/imitation/run_openloong_with_config.py`
**描述**: 支持配置参数的示例运行脚本

**新增功能**:
- `--list_configs`: 列出所有可用配置
- `--robot_config`: 为所有机器人指定统一配置
- `--robot_configs`: 为不同机器人指定不同配置

#### `orca_gym/scripts/openloong_manipulation.py`
**描述**: OpenLoong 操作脚本（dual_arm_manipulation 的别名）

**用途**: 向后兼容，使现有脚本能继续工作

### 2. 修改的文件

#### `envs/manipulation/dual_arm_robot.py`

**修改前**:
```python
from envs.manipulation.robots.configs.openloong_config import openloong_config
robot_config = {
    "openloong_hand_fix_base" : openloong_config,
    "openloong_gripper_2f85_fix_base" : openloong_config,
    "openloong_gripper_2f85_mobile_base" : openloong_config,
}
def get_robot_config(robot_name: str):
    for key in robot_config.keys():
        if key in robot_name:
            return robot_config[key]
    raise ValueError(f"Robot configuration for {robot_name} not found in robot_config dictionary.")

class DualArmRobot(AgentBase):
    def __init__(self, env: DualArmEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)

    def init_agent(self, id: int):
        config = get_robot_config(self._name)
        ...
```

**修改后**:
```python
from envs.manipulation.robots.configs.robot_config_registry import get_robot_config

class DualArmRobot(AgentBase):
    def __init__(self, env: DualArmEnv, id: int, name: str, robot_config_name: str = None) -> None:
        """
        初始化双臂机器人
        
        Args:
            env: 环境对象
            id: 机器人ID
            name: 机器人名称
            robot_config_name: 可选的配置名称，如果不提供则根据机器人名称自动推断
        """
        super().__init__(env, id, name)
        self._robot_config_name = robot_config_name

    def init_agent(self, id: int):
        config = get_robot_config(self._name, self._robot_config_name)
        ...
```

**关键变化**:
- 移除了硬编码的配置字典
- 使用新的配置注册表系统
- `DualArmRobot.__init__` 接受可选的 `robot_config_name` 参数
- `init_agent` 使用新的 `get_robot_config` 函数

#### `envs/manipulation/dual_arm_env.py`

**修改前**:
```python
def __init__(
    self,
    frame_skip: int,        
    reward_type: str,
    orcagym_addr: str,
    agent_names: list,
    pico_ports: list,
    time_step: float,
    run_mode: RunMode,
    action_type: ActionType,
    ctrl_device: ControlDevice,
    control_freq: int,
    sample_range: float,
    task_config_dict: dict,
    **kwargs,
):
    ...

def create_agent(self, id, name):
    entry = get_robot_entry(name)
    module_name, class_name = entry.rsplit(":", 1)
    module = importlib.import_module(module_name)
    class_type = getattr(module, class_name)
    agent = class_type(self, id, name)
    return agent
```

**修改后**:
```python
def __init__(
    self,
    frame_skip: int,        
    reward_type: str,
    orcagym_addr: str,
    agent_names: list,
    pico_ports: list,
    time_step: float,
    run_mode: RunMode,
    action_type: ActionType,
    ctrl_device: ControlDevice,
    control_freq: int,
    sample_range: float,
    task_config_dict: dict,
    robot_configs: dict = None,  # 新增参数
    **kwargs,
):
    """
    初始化双臂环境
    
    Args:
        ...
        robot_configs: 可选的机器人配置字典，格式为 {agent_name: config_name}
                      例如: {"robot1": "openloong", "robot2": "d12"}
                      如果不提供，将根据机器人名称自动推断配置
    """
    ...
    self._robot_configs = robot_configs or {}

def create_agent(self, id, name):
    entry = get_robot_entry(name)
    module_name, class_name = entry.rsplit(":", 1)
    module = importlib.import_module(module_name)
    class_type = getattr(module, class_name)
    
    # 获取该机器人的配置名称（如果提供）
    robot_config_name = self._robot_configs.get(name, None)
    
    # 尝试传递 robot_config_name 参数，如果类不支持则使用默认方式
    try:
        agent = class_type(self, id, name, robot_config_name=robot_config_name)
    except TypeError:
        # 兼容旧版本不支持 robot_config_name 参数的类
        agent = class_type(self, id, name)
    
    return agent
```

**关键变化**:
- `__init__` 接受新的 `robot_configs` 参数
- `create_agent` 支持传递配置名称给机器人类
- 保持向后兼容，支持不接受配置参数的旧版本机器人类

#### `orca_gym/scripts/dual_arm_manipulation.py`

**修改**:
```python
def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_names : str,
                 pico_ports : str,
                 run_mode : str, 
                 action_type : str,
                 ctrl_device : str,
                 max_episode_steps : int,
                 sample_range : float,
                 action_step : int,
                 camera_config : Dict[str, Any],
                 task_config_dict: Dict[str, Any] = None,
                 robot_configs: Dict[str, str] = None) -> str:  # 新增参数
    ...
    kwargs = {
        ...
        'robot_configs': robot_configs  # 新增
    }
    ...
```

**关键变化**:
- `register_env` 接受新的 `robot_configs` 参数
- 将 `robot_configs` 添加到环境的 kwargs 中

## 使用示例

### 示例 1：使用默认配置
```bash
python examples/imitation/run_openloong_sim.py \
    --agent_name openloong_hand_fix_base \
    --run_mode teleoperation
```
系统会自动推断使用 `openloong` 配置。

### 示例 2：显式指定配置
```bash
python examples/imitation/run_openloong_with_config.py \
    --agent_name openloong_hand_fix_base \
    --robot_config openloong \
    --run_mode teleoperation
```

### 示例 3：多机器人不同配置
```bash
python examples/imitation/run_openloong_with_config.py \
    --agent_name "robot1 robot2" \
    --robot_configs "robot1:openloong,robot2:d12" \
    --run_mode teleoperation
```

### 示例 4：列出所有配置
```bash
python examples/imitation/run_openloong_with_config.py --list_configs
```

### 示例 5：在代码中使用
```python
from envs.manipulation.robots.configs.robot_config_registry import get_robot_config

# 获取配置
config = get_robot_config("openloong_hand_fix_base")

# 或显式指定配置名称
config = get_robot_config("my_robot", config_name="openloong")
```

## 测试结果

运行 `python envs/manipulation/robots/configs/test_config_registry.py` 的结果：

```
============================================================
机器人配置注册表测试套件
============================================================

✓ 测试1: 列出所有可用配置 - 通过
✓ 测试2: 通过名称获取配置 - 通过
✓ 测试3: 自动推断配置 - 通过
✓ 测试4: 无效配置处理 - 通过
✓ 测试5: 配置内容完整性 - 通过

============================================================
所有测试完成!
============================================================
```

## 向后兼容性

所有修改都保持了向后兼容：

1. **现有脚本无需修改**：如果不传递 `robot_configs` 参数，系统会自动根据机器人名称推断配置
2. **旧版本机器人类**：`create_agent` 方法使用 try-except 机制，兼容不接受 `robot_config_name` 参数的旧类
3. **默认行为不变**：在不显式指定配置时，行为与之前完全一致

## 优势

1. **灵活性**：可以轻松为不同机器人指定不同配置
2. **可扩展性**：添加新配置只需创建配置文件，无需修改代码
3. **易用性**：支持命令行参数和 Python API
4. **可维护性**：配置与代码分离，更易维护
5. **可测试性**：提供完整的测试脚本
6. **文档完善**：提供详细的使用指南

## 下一步改进建议

1. **配置验证**：添加配置格式验证，确保配置的正确性
2. **配置文件支持**：支持从 YAML/JSON 文件加载配置
3. **配置继承**：支持配置继承，减少重复
4. **性能优化**：缓存已加载的配置，提高性能
5. **Web 界面**：开发 Web 界面用于可视化配置管理

## 相关文档

- [配置系统使用指南](README.md)
- [测试脚本](test_config_registry.py)
- [示例脚本](../../examples/imitation/run_openloong_with_config.py)

