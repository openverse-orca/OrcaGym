# 🔧 Robosuite 适配器

OrcaGym 提供 Robosuite 适配器用于机器人操控任务。

## 模块结构

```
orca_gym/adapters/robosuite/
├── __init__.py
├── macros.py # 宏定义和常量
├── controllers/ # 控制器（OSC/IK/关节位置/力矩/速度等）
│ ├── config/ # 控制器配置 JSON
│ ├── interpolators/ # 插值器
│ ├── base_controller.py
│ ├── controller_factory.py
│ ├── ik.py
│ ├── joint_pos.py
│ ├── joint_tor.py
│ ├── joint_vel.py
│ └── osc.py
├── devices/ # 输入设备（键盘/空间鼠标等）
│ ├── device.py
│ ├── keyboard.py
│ └── spacemouse.py
└── utils/
 ├── control_utils.py # 控制工具
 ├── errors.py # 错误定义
 ├── binding_utils.py # 绑定工具
 ├── robot_utils.py # 机器人工具
 ├── placement_samplers.py # 物体放置采样
 └── log_utils.py # 日志工具
```

## 使用方式

Robosuite 适配器提供了一系列工具函数，用于：

- 机器人模型绑定
- 物体放置策略
- 控制信号生成
- 错误处理

```python
from orca_gym.adapters.robosuite import macros
from orca_gym.adapters.robosuite.utils import control_utils, robot_utils

# 使用 Robosuite 风格的工具
```

## 与 Robomimic 的关系

Robosuite 适配器通常与 Robomimic 适配器配合使用：

- **Robosuite** → 环境抽象和工具
- **Robomimic** → 数据集和算法
- **OrcaGym** → 底层仿真引擎
