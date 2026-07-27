# 🧠 Robomimic 适配器

OrcaGym 提供 Robomimic 适配器用于模仿学习。

## 环境适配

```python
from orca_adapters.robomimic import robomimic_env

class MyTask(RobomimicEnv):
 """创建 Robomimic 兼容的环境"""
 def __init__(self, ...):
 super().__init__(...)
```

## 数据集工具

```python
from orca_adapters.robomimic import dataset_util

# 处理 HDF5 数据集
```

## 任务定义

```python
# 自定义任务类
from orca_adapters.robomimic.task import ...

class MyRobomimicTask(RobomimicEnv):
 # 实现 Robomimic 要求的接口
 pass
```
