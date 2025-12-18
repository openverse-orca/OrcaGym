# Python 导入路径问题说明

## 问题描述

在 `orca_gym/scripts/dual_arm_manipulation.py` 文件中，可以直接使用：

```python
from envs.manipulation.dual_arm_env import DualArmEnv, ControlDevice
```

但在其他环境下运行时会报错：`ModuleNotFoundError: No module named 'envs'`

## 原因分析

### 为什么在当前环境下能工作？

1. **开发模式安装**：如果使用 `pip install -e .` 安装项目，项目根目录会被自动添加到 Python 的 `sys.path` 中
2. **PYTHONPATH 环境变量**：可能已经设置了 `PYTHONPATH` 环境变量指向项目根目录
3. **工作目录**：当前工作目录可能就是项目根目录，Python 会自动将当前目录添加到搜索路径

### 为什么在其他环境下会报错？

1. **未以开发模式安装**：只通过 `pip install orca-gym` 安装，`envs` 目录不在包中（根据 `pyproject.toml`，`envs*` 被明确排除）
2. **PYTHONPATH 未设置**：没有设置环境变量指向项目根目录
3. **工作目录不对**：不在项目根目录下运行脚本

## 解决方案

### 方案 1：开发模式安装（推荐）⭐

这是最推荐的方式，适用于开发和测试：

```bash
cd /home/orcash/OrcaGym/OrcaGym
pip install -e .
```

这样安装后，项目根目录会被添加到 Python 路径，所有脚本都可以直接导入 `envs` 和 `examples`。

### 方案 2：设置 PYTHONPATH 环境变量

在运行脚本前设置环境变量：

```bash
export PYTHONPATH="${PYTHONPATH}:/home/orcash/OrcaGym/OrcaGym"
python your_script.py
```

或者在脚本中动态添加：

```python
import sys
import os
from pathlib import Path

# 获取项目根目录（假设脚本在项目内）
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 现在可以导入 envs
from envs.manipulation.dual_arm_env import DualArmEnv
```

### 方案 3：修改导入方式（不推荐）

如果 `envs` 目录在项目根目录下，可以尝试相对导入，但这需要确保包结构正确：

```python
# 不推荐，因为 envs 不在 orca_gym 包内
# 这种方式在开发模式下可能工作，但不够清晰
```

### 方案 4：在脚本开头自动添加路径

修改 `dual_arm_manipulation.py`，在文件开头添加路径设置：

```python
import sys
import os
from pathlib import Path

# 自动检测并添加项目根目录到 Python 路径
_script_dir = Path(__file__).parent  # orca_gym/scripts
_package_dir = _script_dir.parent    # orca_gym
_project_root = _package_dir.parent  # OrcaGym (项目根目录)

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 现在可以安全导入
from envs.manipulation.dual_arm_env import DualArmEnv, ControlDevice
```

## 项目结构说明

```
OrcaGym/                    # 项目根目录
├── orca_gym/              # 核心包（会被安装到 site-packages）
│   └── scripts/
│       └── dual_arm_manipulation.py  # 当前文件
├── envs/                  # 环境实现（不在包中，需要手动添加到路径）
│   └── manipulation/
│       └── dual_arm_env.py
├── examples/              # 示例代码（不在包中）
└── pyproject.toml         # 配置：exclude = ["examples*", "envs*", ...]
```

根据 `pyproject.toml` 的配置：

```toml
[tool.setuptools.packages.find]
include = ["orca_gym*"]
exclude = ["examples*", "envs*", "doc*", "3rd_party*", "tests*"]
```

`envs` 和 `examples` 目录**不会**被打包到 PyPI 包中，所以必须通过开发模式安装或手动设置路径来访问。

## 最佳实践建议

1. **开发环境**：使用 `pip install -e .` 安装，这样所有路径都会自动配置好
2. **生产环境**：如果只需要核心功能，使用 `pip install orca-gym`，但需要手动处理 `envs` 和 `examples` 的路径
3. **CI/CD**：在 CI 脚本中明确设置 `PYTHONPATH` 或使用开发模式安装

## 验证方法

检查 Python 路径是否包含项目根目录：

```python
import sys
print("Python 路径:")
for p in sys.path:
    print(f"  - {p}")

# 检查是否能导入
try:
    from envs.manipulation.dual_arm_env import DualArmEnv
    print("✅ 可以导入 envs")
except ImportError as e:
    print(f"❌ 无法导入 envs: {e}")
```



















