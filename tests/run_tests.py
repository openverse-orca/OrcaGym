#!/usr/bin/env python
"""OrcaGym 测试一键运行脚本。

基于 unittest，支持全量运行和按组件运行。组件按 `tests/orca_gym/` 下的
相对路径标识（如 `core/euler`、`environment/euler`），脚本自动发现含
`test_*.py` 的目录作为可运行组件。

用法:
    # 全量运行
    python tests/run_tests.py

    # 按组件运行（精确匹配叶子组件）
    python tests/run_tests.py --component core/euler

    # 按前缀运行（匹配 core 下所有组件，含未来新增的）
    python tests/run_tests.py --component core

    # 列出所有可用组件
    python tests/run_tests.py --list

    # 详细输出
    python tests/run_tests.py -v

    # 多个组件
    python tests/run_tests.py --component core/euler --component environment/euler

运行环境: orca conda 环境（MuJoCo 3.7.0）。CPU 测试可在 sandbox 内运行；
GPU 依赖的测试需通过 TRAE 命令白名单旁路 sandbox（见 AGENTS.md 规则 3）。
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

# 脚本所在目录: tests/
_TESTS_DIR = Path(__file__).resolve().parent
# 仓库根目录: tests/ 的父目录
_REPO_ROOT = _TESTS_DIR.parent
# 测试根包目录: tests/orca_gym/
_TEST_ROOT = _TESTS_DIR / "orca_gym"


def _ensure_import_path() -> None:
    """将仓库根加入 sys.path，确保 `import orca_gym...` 可用。"""
    repo_root = str(_REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def discover_components() -> list[str]:
    """发现所有含 test_*.py 的组件目录，返回相对路径标识列表。

    组件标识为相对 `tests/orca_gym/` 的 POSIX 路径（如 `core/euler`）。
    仅返回直接包含 test_*.py 的目录，避免父目录与子目录重复。
    """
    if not _TEST_ROOT.is_dir():
        return []

    components: list[str] = []
    for test_file in _TEST_ROOT.rglob("test_*.py"):
        rel = test_file.parent.relative_to(_TEST_ROOT)
        # 跳过 fixtures 等非测试目录（test_*.py 不应放在那里）
        if rel.parts and rel.parts[0] == "fixtures":
            continue
        ident = rel.as_posix() if rel.parts else "."
        if ident not in components:
            components.append(ident)
    components.sort()
    return components


def _matches_component(component: str, available: list[str]) -> list[str]:
    """返回与指定组件标识匹配的可用组件列表。

    支持精确匹配和前缀匹配：
    - `core/euler` 精确匹配 `core/euler`
    - `core` 匹配 `core/euler`、`core/xxx` 等所有 core 下组件
    """
    exact = {c for c in available if c == component}
    prefix = {c for c in available if c.startswith(component + "/")}
    matched = sorted(exact | prefix)
    return matched


def build_suite(components: list[str] | None) -> unittest.TestSuite:
    """构建测试套件。

    Args:
        components: 指定组件列表；None 表示全量运行。
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    available = discover_components()
    if not available:
        print("警告: 未发现任何测试组件（tests/orca_gym/ 下无 test_*.py）")
        return suite

    if components is None:
        # 全量运行：从测试根包顶层 discover
        suite = loader.discover(
            start_dir=str(_TEST_ROOT),
            pattern="test_*.py",
            top_level_dir=str(_REPO_ROOT),
        )
        return suite

    # 按组件运行
    selected: list[str] = []
    for comp in components:
        matched = _matches_component(comp, available)
        if not matched:
            print(f"警告: 未找到组件 '{comp}'（可用: {', '.join(available) or '无'}）")
            continue
        for m in matched:
            if m not in selected:
                selected.append(m)

    for comp in selected:
        target_dir = _TEST_ROOT / Path(*comp.split("/")) if comp != "." else _TEST_ROOT
        comp_suite = loader.discover(
            start_dir=str(target_dir),
            pattern="test_*.py",
            top_level_dir=str(_REPO_ROOT),
        )
        suite.addTest(comp_suite)

    return suite


def run(components: list[str] | None, verbose: bool) -> int:
    """运行测试，返回退出码（0=成功，1=失败）。"""
    _ensure_import_path()

    suite = build_suite(components)
    if suite.countTestCases() == 0:
        print("无可运行测试用例。")
        return 1

    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout,
    )
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OrcaGym 测试一键运行脚本（unittest）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--component", "-c",
        action="append",
        default=None,
        metavar="PATH",
        help="按组件运行，组件标识为 tests/orca_gym/ 下的相对路径（如 core/euler）。"
             "可多次指定。支持前缀匹配（如 core 匹配 core 下所有组件）。",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用组件后退出。",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出（每个用例一行）。",
    )
    args = parser.parse_args()

    if args.list:
        components = discover_components()
        if not components:
            print("未发现任何组件。")
        else:
            print("可用组件（tests/orca_gym/ 下相对路径）:")
            for c in components:
                print(f"  {c}")
        return 0

    return run(components=args.component, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
