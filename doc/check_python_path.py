#!/usr/bin/env python3
"""
检查当前 Python 路径的几种方法
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("方法 1: 使用 sys.path（推荐）")
print("=" * 60)
print("当前 Python 搜索路径:")
for i, path in enumerate(sys.path, 1):
    print(f"  {i}. {path}")

print("\n" + "=" * 60)
print("方法 2: 使用 os.environ 查看 PYTHONPATH")
print("=" * 60)
pythonpath = os.environ.get('PYTHONPATH', '未设置')
if pythonpath:
    print(f"PYTHONPATH: {pythonpath}")
else:
    print("PYTHONPATH: 未设置")

print("\n" + "=" * 60)
print("方法 3: 检查特定路径是否存在")
print("=" * 60)
project_root = Path(__file__).parent.parent  # 假设脚本在 doc/ 目录下
print(f"项目根目录: {project_root}")
print(f"是否在 sys.path 中: {str(project_root) in sys.path}")

# 检查 envs 目录
envs_dir = project_root / "envs"
print(f"\nenvs 目录: {envs_dir}")
print(f"是否存在: {envs_dir.exists()}")
print(f"是否可导入: ", end="")
try:
    import envs
    print("✅ 可以导入 envs 模块")
except ImportError:
    print("❌ 无法导入 envs 模块")

print("\n" + "=" * 60)
print("方法 4: 当前工作目录")
print("=" * 60)
print(f"当前工作目录: {os.getcwd()}")

print("\n" + "=" * 60)
print("方法 5: 脚本所在目录")
print("=" * 60)
print(f"脚本文件路径: {__file__}")
print(f"脚本所在目录: {Path(__file__).parent}")
print(f"脚本的绝对路径: {Path(__file__).absolute()}")

print("\n" + "=" * 60)
print("方法 6: 检查特定模块的路径")
print("=" * 60)
try:
    import orca_gym
    print(f"orca_gym 模块路径: {orca_gym.__file__}")
    print(f"orca_gym 包目录: {Path(orca_gym.__file__).parent}")
except ImportError:
    print("orca_gym 模块未安装")

print("\n" + "=" * 60)
print("快速检查：envs 是否可以导入")
print("=" * 60)
try:
    from envs.manipulation.dual_arm_env import DualArmEnv
    print("✅ 成功导入: from envs.manipulation.dual_arm_env import DualArmEnv")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n建议解决方案:")
    print("  1. 运行: pip install -e .")
    print("  2. 或设置: export PYTHONPATH=\"${PYTHONPATH}:$(pwd)\"")

