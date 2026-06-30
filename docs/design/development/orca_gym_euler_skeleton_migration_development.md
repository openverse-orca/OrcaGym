# OrcaGym Euler 骨架变更实施方案：直接继承 gym.Env + OrcaGymEnvMixin

## 1. 文档定位

本文是 `OrcaGymEulerEnv` 骨架变更的实施指导，对应架构文档 §4.1/§5.1/§5.9/§11.1/§12 的最新规约。

**变更目标**：将 `OrcaGymEulerEnv` 的继承链从 `OrcaGymBaseEnv` 切换为 `OrcaGymEnvMixin, gym.Env`，删除 `_BLOCKED_ATTRS`/`__getattr__`/`__setattr__`/`_SHIELDED_ATTRS` 补丁机制，通过 `OrcaGymEnvMixin` 共享公共方法。

**上游约束**：架构文档 `docs/design/architecture/orca_gym_euler_architecture.md`（§5.9 OrcaGymEnvMixin 定义、§7.1 M0-M7 多层封装隔离机制、§7.2 ruff SLF001 配置、§7.3 AGENTS.md 内容要求、§12.3 K14 继承链约束）。

**前提条件**：阶段二功能填充已完成（生命周期、步进、状态设置、渲染方法均已真实实现），本方案不修改这些已填充的功能逻辑，仅变更继承结构和隔离机制。

**实施次序**：阶段 0 先行落地 ruff SLF001 静态检查与 AGENTS.md AI 行为约束（架构 §7.2/§7.3），**一次性覆盖 6 个自研仓库**（OrcaGym / OrcaFlow / OrcaEuler / OrcaPlayground / OrcaManipulation / OrcaLab），为后续阶段 1-3 的编码提供静态检查基线；阶段 1-3 在此基础上完成 OrcaGym 继承链切换与端到端验证。

---

## 2. 变更范围

### 2.1 新增文件

| 文件 | 仓库 | 内容 | 阶段 |
|------|------|------|------|
| `tests/orca_gym/test_ruff_config.py` | OrcaGym | ruff SLF001 配置与可执行性测试 | 阶段 0 |
| `tests/orca_gym/test_agents_md.py` | OrcaGym | AGENTS.md API 隔离章节内容校验测试 | 阶段 0 |
| `tests/test_ruff_config.py` | OrcaFlow/OrcaPlayground/OrcaManipulation | 同上（各仓库根 `tests/`） | 阶段 0 |
| `tests/test_agents_md.py` | OrcaFlow/OrcaPlayground/OrcaManipulation | 同上 | 阶段 0 |
| `orca/euler/tests/test_ruff_config.py` | OrcaEuler | 同上 | 阶段 0 |
| `orca/euler/tests/test_agents_md.py` | OrcaEuler | 同上 | 阶段 0 |
| `test/test_ruff_config.py` | OrcaLab | 同上（OrcaLab 测试目录为 `test/`，单数） | 阶段 0 |
| `test/test_agents_md.py` | OrcaLab | 同上 | 阶段 0 |
| `pyproject.toml` | OrcaPlayground | 新建，仅含 `[tool.ruff.lint]` 段（不破坏 `setup.py`） | 阶段 0 |
| `AGENTS.md` | OrcaManipulation | 新建，含测试环境规则 + API 隔离章节 | 阶段 0 |
| `AGENTS.md` | OrcaLab | 新建，含测试环境规则 + API 隔离章节（适配 OrcaLab 编辑器/UI 仓） | 阶段 0 |
| `orca_gym/environment/orca_gym_env_mixin.py` | OrcaGym | `OrcaGymEnvMixin` 类，迁移自 `OrcaGymBaseEnv` 的 10 个公共方法 | 阶段 1 |
| `tests/orca_gym/environment/test_orca_gym_env_mixin.py` | OrcaGym | Mixin 单元测试 | 阶段 1 |

### 2.2 修改文件

| 文件 | 仓库 | 变更内容 | 阶段 |
|------|------|---------|------|
| `AGENTS.md` | OrcaGym | 删除过时的 `__getattr__`/`_BLOCKED_ATTRS`/M1-M6 描述，新增"API 隔离强制"章节 | 阶段 0 |
| `AGENTS.md` | OrcaFlow/OrcaEuler | 追加 "API Isolation Enforcement" 英文章节 | 阶段 0 |
| `AGENTS.md` | OrcaPlayground | 新增"API 隔离强制"章节（适配示例仓） | 阶段 0 |
| `pyproject.toml` | OrcaGym | 新增 `[tool.ruff.lint]` SLF001 配置，`dev` 依赖增加 `ruff` | 阶段 0 |
| `pyproject.toml` | OrcaFlow/OrcaEuler | `select` 追加 `"SLF001"`，`dev` 增加 `ruff`，OrcaFlow 加 `exclude` | 阶段 0 |
| `pyproject.toml` | OrcaManipulation | 新增 `[tool.ruff.lint]` 段 | 阶段 0 |
| `pyproject.toml` | OrcaLab | 新增 `[tool.ruff.lint]` 段，`dev` 依赖增加 `ruff` | 阶段 0 |
| `orca_gym/environment/euler/orca_gym_euler_env.py` | OrcaGym | 继承链切换、删除补丁机制、重写 `__init__`、更新 `__dir__` | 阶段 2 |
| `tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py` | OrcaGym | 删除 K2/K10 旧测试，新增 K14 测试 | 阶段 2 |

### 2.3 不修改的文件

| 文件 | 原因 |
|------|------|
| `orca_gym/environment/orca_gym_env.py`（`OrcaGymBaseEnv`） | 原有继承体系不动（架构 §4.3），Local 体系仍使用 |
| `orca_gym/core/euler/*.py` | Gym 层及子组件不变更 |
| `tests/orca_gym/environment/euler/test_simple_euler_env_compliance.py` | SimpleEulerEnv 合规测试不依赖继承链 |
| `tests/orca_gym/core/euler/*.py` | Gym 层测试不变更 |

---

## 3. 实施阶段

### 阶段 0：基础设施先行（AGENTS.md + ruff）—— 六仓库统一实施

**目标**：在编码变更前，先落地架构文档 §7.2（ruff SLF001 静态检查）和 §7.3（AGENTS.md AI 行为约束）的要求，为后续阶段 1-3 的编码提供静态检查基线和 AI 行为约束。

**上游约束**：架构文档 §7.1（M1 ruff SLF001、M2 AGENTS.md）、§7.2（ruff 配置）、§7.3（AGENTS.md 内容要求）。

**实施范围**：阶段 0 一次性处理以下 6 个自研仓库，统一落地 ruff SLF001 配置与 AGENTS.md API 隔离章节：

| 仓库 | 配置文件 | ruff 现状 | AGENTS.md 现状 | fork 目录 |
|------|---------|----------|---------------|----------|
| OrcaGym | `pyproject.toml` | 无 ruff 段 | 有（中文，含过时 `__getattr__`/`_BLOCKED_ATTRS`/M1-M6） | 无 |
| OrcaFlow | `pyproject.toml` | `select=["E","F","W","I"]`，无 SLF001 | 有（英文，Warp fork 主题） | `orca/flow/src/warp_fork/`（232 报警） |
| OrcaEuler | `pyproject.toml` | `select=["E","F","W","I"]`，无 SLF001 | 有（英文，Euler 框架主题） | 无 |
| OrcaPlayground | `setup.py`（无 pyproject.toml） | 无 ruff 配置 | 有（中文，示例仓，引用 OrcaGym 架构） | 无 |
| OrcaManipulation | `pyproject.toml` | 无 ruff 段 | **无**（仅 `DEVELOPER_GUIDE.md`） | 无 |
| OrcaLab | `pyproject.toml` | 无 ruff 段（`dev` 含 `flake8`/`black`/`mypy`） | **无** | 无（自研代码在 `orcalab/`） |

#### 3.0.1 各仓库统一配置 ruff SLF001

SLF001 基于类型作用域判断"内部访问 vs 外部穿墙"：类内部访问 `self._private` 不报警，外部对象访问 `obj._private` 报警。无需逐文件配置即可精准识别。

**统一配置模板**（写入各仓库配置文件的 `[tool.ruff.lint]` 段）：

```toml
[tool.ruff.lint]
select = ["SLF001"]

[tool.ruff.lint.per-file-ignores]
# 测试文件允许白盒访问（测试就是要测内部）
"tests/**" = ["SLF001"]
# __init__.py 允许 re-export
"**/__init__.py" = ["SLF001", "F401"]
```

> - OrcaFlow/OrcaEuler 已有 `select = ["E", "F", "W", "I"]`，在原列表追加 `"SLF001"` 即可（合并为 `select = ["E", "F", "W", "I", "SLF001"]`），保留既有规则。
> - **OrcaLab 测试目录为 `test/`（单数，非 `tests/`）**，其 `per-file-ignores` 须写 `"test/**" = ["SLF001"]`，模板中的 `"tests/**"` 不匹配 OrcaLab。

**各仓库配置落点**：

| 仓库 | 配置文件 | 操作 |
|------|---------|------|
| OrcaGym | `pyproject.toml` | 新增 `[tool.ruff.lint]` 段；`dev` 依赖追加 `ruff` |
| OrcaFlow | `pyproject.toml` | `select` 追加 `"SLF001"`；`dev` 追加 `ruff` |
| OrcaEuler | `pyproject.toml` | `select` 追加 `"SLF001"`；`dev` 追加 `ruff` |
| OrcaPlayground | 新建 `pyproject.toml`（仅含 `[tool.ruff.lint]` 段，不破坏现有 `setup.py`） | 新增 ruff 配置 |
| OrcaManipulation | `pyproject.toml` | 新增 `[tool.ruff.lint]` 段 |
| OrcaLab | `pyproject.toml` | 新增 `[tool.ruff.lint]` 段，`per-file-ignores` 用 `"test/**"`；`dev` 追加 `ruff` |

**第三方 fork 目录排除原则**：

SLF001 只约束自研代码。第三方 fork 代码遵循上游实现风格，存在大量同框架跨模块访问 `_` 前缀属性（如 `module._find_kernel()`、`self._setup_nnz_transfer()`），这些是框架内部协作，SLF001 无法与"外部穿墙"区分。

各仓库须执行以下流程：

1. **识别 fork 目录**：扫描仓库内的第三方 fork 代码（通常带 `LICENSE`/`NOTICE`/`apply_namespace.sh` 等上游标记）
2. **基线扫描**：对 fork 目录单独执行 `ruff check --select SLF001`，确认报警来自上游风格而非本仓库引入
3. **配置排除**：在 `[tool.ruff.lint.exclude]` 中排除该目录

```toml
[tool.ruff.lint.exclude]
# 第三方 fork 代码遵循上游风格，不强制 SLF001
# 仅 OrcaFlow 需要配置：
"orca/flow/src/warp_fork/"
```

> 已知 fork 目录基线（截至扫描日）：
>
> | 仓库 | fork 目录 | SLF001 报警数 | 来源 |
> |------|----------|--------------|------|
> | OrcaFlow | `orca/flow/src/warp_fork/` | 232 | Warp 项目 |
>
> 报警集中在 `python/context.py`（~70+）、`python/sparse.py`（~80+）、`python/codegen.py`（~29）、`python/tape.py`（~6）等，均为 Warp 上游内部协作访问，排除即可。其余 5 仓库无 fork 目录，无需配置 `exclude`。

#### 3.0.2 各仓库统一配置 AGENTS.md API 隔离章节

架构 §7.3 要求每个自研仓库根目录配置 `AGENTS.md`，含"API 隔离强制"章节。各仓库现有 AGENTS.md 风格不同（OrcaGym/OrcaPlayground 中文、OrcaFlow/OrcaEuler 英文、OrcaManipulation/OrcaLab 无），处理方式如下：

**OrcaGym**（更新现有）：

- 删除规则 2 中"跳过 `__getattr__` 拦截机制"禁止项、"`_BLOCKED_ATTRS`"相关描述（机制已删除）
- 将"M1-M6 六层机制"更新为"M0-M7 多层封装隔离机制"（架构 §7.1）
- 新增"规则 4：API 隔离强制"章节（内容见下）

**OrcaPlayground**（更新现有）：

- 规则 2 引用的"M0-M7 多层封装隔离机制"已是最新（无需改过时描述）
- 新增"规则 4：API 隔离强制"章节，内容适配示例仓（禁止穿墙访问 `env._gym._sim._mjData` 等，引导走 `env.data.*`/`env.sim_config.*` 公共 API）

**OrcaFlow / OrcaEuler**（追加章节）：

- 现有 AGENTS.md 为英文、聚焦 Warp fork / Euler 框架主题，保留不动
- 追加"API Isolation Enforcement"章节（英文），约束 AI 不得穿墙访问自研类的 `_` 前缀属性（如 Euler 的 `solver._xxx`、Flow 的内部 `_` 属性），并要求执行 `ruff check --select SLF001`

**OrcaManipulation**（新建）：

- 仓库目前无 `AGENTS.md`，从模板创建
- 含测试环境规则（`orca` conda 环境）+ "规则 2：API 隔离强制"章节（适配操作仓：禁止穿墙访问 OrcaGym env 的 `_` 前缀属性，走公共 API）

**OrcaLab**（新建）：

- 仓库目前无 `AGENTS.md`，从模板创建
- 含测试环境规则（`orca` conda 环境）+ "API 隔离强制"章节（适配 OrcaLab 编辑器/UI 仓：约束 AI 不得穿墙访问 `orcalab/` 自研类的 `_` 前缀属性，如内部 service/bus 的 `_` 成员；OrcaLab 通过 `orca-gym` 依赖调用 OrcaGym 时走公共 API，禁止 `env._gym._sim._mjData` 穿墙），并要求执行 `ruff check --select SLF001 orcalab/`

**"API 隔离强制"章节模板**（OrcaGym 版，其余仓库按自身语言和组件调整）：

```markdown
## 规则 4：API 隔离强制

本仓库采用 `_` 前缀社区约定 + ruff SLF001 静态检查 + `__dir__` 控制，
引导 AI 和用户走公共 API（架构 §7）。

### 禁止穿墙访问

不得访问以下 `_` 前缀内部属性（类内部合法的 `self._xxx` 委托除外）：

- `env._gym` / `env._stub` / `env._channel` / `env._studio_bridge`
- `env._gym._sim` / `env._gym._sim._mjData` / `env._gym._sim._mjModel`
- 任何自研类的 `_` 前缀属性

> `env.gym` / `env.stub` / `env.channel` 在 `OrcaGymEulerEnv` 中不存在
> （直接继承 `gym.Env`，Python 原生 `AttributeError`）。

### 必须使用公共 API

| 操作 | 正确 | 禁止 |
|------|------|------|
| 读取状态 | `env.data.qpos` / `env.data.body_xpos(name)` / `env.query_*()` | `env._gym._sim._mjData.qpos` |
| 写入状态 | `env.set_joint_qpos()` / `env.apply_body_force()` | `env._gym._sim._mjData.xfrc_applied[...]` |
| 步进 | `env.do_simulation(ctrl, n_frames)` / `env.step()` | `env._gym._sim._mjData.step()` |
| 求解器配置 | `env.sim_config.timestep = 0.002` | `env._gym._sim._mjModel.opt.timestep = 0.002` |

### 必须执行 ruff

提交代码前必须执行，零报警方可提交：

    <conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca_gym/

### 缺失功能时扩展公共方法

若公共 API 不满足需求，**暂停并提交用户决策**，不得自行穿墙访问内部属性。
扩展方式：
- 在 `OrcaGymEulerEnv` 增加公共方法（委托到 `_gym` 公共 API）
- 在 `OrcaGymEuler` 增加公共方法（委托到 `_sim` 公共 API）
- 在 `OrcaGymDataView` 增加字段访问器
```

> 各仓库的 ruff 扫描目标路径不同（OrcaGym: `orca_gym/`、OrcaFlow: `orca/`、OrcaEuler: `orca/`、OrcaPlayground: `envs/`+`examples/`、OrcaManipulation: `envs/`+`examples/`、OrcaLab: `orcalab/`），AGENTS.md 的 ruff 命令示例按各自路径填写。

#### 3.0.3 建立现有代码的 ruff 扫描基线

配置完成后，对各仓库自研代码执行 ruff 扫描，记录基线报警（阶段二功能填充时可能引入的 `self._gym._sim` 访问）：

```bash
# OrcaGym（Euler 子目录，阶段 2/3 重点）
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca_gym/environment/euler/ orca_gym/core/euler/
# OrcaFlow
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca/
# OrcaEuler
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca/
# OrcaPlayground
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 envs/ examples/
# OrcaManipulation
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 envs/ examples/
# OrcaLab
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orcalab/
```

> 若仓库根配置文件已含 `[tool.ruff.lint]` 且 `exclude`，ruff 会自动跳过 fork 目录（如 OrcaFlow 的 `warp_fork/`），无需额外参数。加 `--statistics` 可查看报警分布。

**基线处理原则**：

- 若现有代码存在 SLF001 报警，记录到基线清单（报警位置 + 数量）
- 阶段 2 切换继承链时，同步修复 OrcaGym 基线报警（改为公共 API 委托）；其余仓库基线报警按需修复
- 阶段 3 验收时，OrcaGym 零报警方为通过；其余仓库阶段 0 验收即记录基线，后续各自迭代消化
- **fork 目录报警不纳入基线**：已通过 `exclude` 排除，验收时不计入

#### 3.0.4 测试

各仓库在自身 `tests/` 下新建两个测试文件，校验本仓库的 ruff 配置与 AGENTS.md 内容。测试逻辑统一，仅配置文件路径和 AGENTS.md 路径按仓库调整。

**测试文件落点**：

| 仓库 | ruff 配置测试 | AGENTS.md 测试 |
|------|--------------|---------------|
| OrcaGym | `tests/orca_gym/test_ruff_config.py` | `tests/orca_gym/test_agents_md.py` |
| OrcaFlow | `tests/test_ruff_config.py` | `tests/test_agents_md.py` |
| OrcaEuler | `orca/euler/tests/test_ruff_config.py` | `orca/euler/tests/test_agents_md.py` |
| OrcaPlayground | `tests/test_ruff_config.py` | `tests/test_agents_md.py` |
| OrcaManipulation | `tests/test_ruff_config.py` | `tests/test_agents_md.py` |
| OrcaLab | `test/test_ruff_config.py`（`test/` 单数） | `test/test_agents_md.py` |

> OrcaFlow/OrcaEuler 用 `unittest`（遵循其 AGENTS.md 约定），测试类与断言与下方一致，仅 `Path(__file__).resolve().parents[N]` 的层数按仓库目录深度调整，指向各自根目录的配置文件。

**`test_ruff_config.py`**（以 OrcaGym 为例，`parents[3]` 指向仓库根）：

```python
"""ruff SLF001 配置与可执行性测试。"""

import subprocess
import sys
import unittest
from pathlib import Path


class TestRuffConfig(unittest.TestCase):
    """ruff 配置与可执行性。"""

    @classmethod
    def setUpClass(cls):
        cls.pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
        cls.content = cls.pyproject.read_text()

    def test_ruff_installed(self):
        """ruff 已安装且可执行。"""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, "ruff 未安装")
        self.assertIn("ruff", result.stdout.lower())

    def test_ruff_config_has_slf001(self):
        """配置文件已配置 SLF001。"""
        self.assertIn("[tool.ruff.lint]", self.content)
        self.assertIn("SLF001", self.content)

    def test_ruff_tests_ignored(self):
        """测试目录已配置 SLF001 忽略。"""
        # OrcaGym/OrcaFlow/OrcaEuler/OrcaPlayground/OrcaManipulation 用 "tests/**"
        # OrcaLab 测试目录为单数 test/，用 "test/**"
        self.assertTrue(
            "tests/**" in self.content or "test/**" in self.content
        )

    def test_ruff_init_ignored(self):
        """__init__.py 已配置忽略。"""
        self.assertIn("__init__.py", self.content)

    def test_ruff_exclude_section_exists(self):
        """配置文件含 [tool.ruff.lint.exclude] 段（第三方 fork 排除）。

        OrcaFlow 必须含此段并排除 warp_fork/；
        其余仓库此段可为空但段头应存在（统一模板）。
        """
        self.assertIn("[tool.ruff.lint.exclude]", self.content)
```

> OrcaPlayground 用 `setup.py` 无 `pyproject.toml`，阶段 0 为其新建仅含 ruff 配置的 `pyproject.toml`，测试指向该文件。

**`test_agents_md.py`**（以 OrcaGym 为例，校验中文 AGENTS.md）：

```python
"""AGENTS.md API 隔离章节内容校验测试。"""

import unittest
from pathlib import Path


class TestAgentsMd(unittest.TestCase):
    """AGENTS.md 内容约束。"""

    @classmethod
    def setUpClass(cls):
        cls.content = (Path(__file__).resolve().parents[3] / "AGENTS.md").read_text()

    def test_has_api_isolation_section(self):
        """AGENTS.md 包含 API 隔离强制章节。"""
        # 中文仓库（OrcaGym/OrcaPlayground/OrcaManipulation/OrcaLab）匹配中文标题
        # 英文仓库（OrcaFlow/OrcaEuler）匹配 "API Isolation Enforcement"
        self.assertTrue(
            "API 隔离强制" in self.content
            or "API Isolation Enforcement" in self.content
        )

    def test_has_ruff_requirement(self):
        """AGENTS.md 要求执行 ruff SLF001。"""
        self.assertIn("ruff check", self.content)
        self.assertIn("SLF001", self.content)

    def test_has_public_api_table(self):
        """AGENTS.md 含"正确 vs 禁止"公共 API 对照表。

        各仓库 API 例子不同：
        - OrcaGym/OrcaPlayground/OrcaManipulation：env.data.qpos / env.sim_config
        - OrcaFlow/OrcaEuler/OrcaLab：各自组件的公共 API
        统一校验对照表标记（中文"正确"/"禁止"或英文"Correct"/"Forbidden"）。
        """
        has_cn = "正确" in self.content and "禁止" in self.content
        has_en = "Correct" in self.content and "Forbidden" in self.content
        self.assertTrue(has_cn or has_en, "缺少公共 API 对照表")

    def test_no_legacy_getattr_description(self):
        """AGENTS.md 不再描述 __getattr__ 拦截机制（仅 OrcaGym 需清理）。"""
        self.assertNotIn("__getattr__ 拦截", self.content)
        self.assertNotIn("_BLOCKED_ATTRS", self.content)

    def test_mechanism_version_updated(self):
        """机制描述更新为 M0-M7（仅 OrcaGym 原含 M1-M6）。"""
        self.assertNotIn("M1-M6 六层机制", self.content)
```

> - OrcaFlow/OrcaEuler 的 AGENTS.md 为英文且原本不含过时 `__getattr__`/`_BLOCKED_ATTRS`/M1-M6 描述，`test_no_legacy_getattr_description` 和 `test_mechanism_version_updated` 对它们恒真通过。
> - `test_has_api_isolation_section` 允许匹配英文标题 "API Isolation Enforcement"。
> - `test_has_public_api_table` 用通用对照表标记校验，不绑定具体 API 名，兼容 OrcaFlow/OrcaEuler/OrcaLab 等非 env 仓库。
> - OrcaManipulation/OrcaLab 新建 AGENTS.md 后同样通过上述全部断言。

#### 3.0.5 验收标准

| 验收项 | 验证方式 | 适用仓库 |
|--------|---------|---------|
| 配置文件含 `[tool.ruff.lint]` + SLF001 | `test_ruff_config_has_slf001` | 全部 6 仓库 |
| 测试目录 SLF001 忽略 | `test_ruff_tests_ignored` | 全部 6 仓库 |
| `__init__.py` 忽略 | `test_ruff_init_ignored` | 全部 6 仓库 |
| 配置文件含 `[tool.ruff.lint.exclude]` 段 | `test_ruff_exclude_section_exists` | 全部 6 仓库 |
| 第三方 fork 目录已排除 | `exclude` 含 `warp_fork/` | 仅 OrcaFlow |
| ruff 可执行 | `test_ruff_installed` | 全部 6 仓库 |
| AGENTS.md 含 API 隔离章节 | `test_has_api_isolation_section` | 全部 6 仓库 |
| AGENTS.md 含 ruff 要求 | `test_has_ruff_requirement` | 全部 6 仓库 |
| AGENTS.md 含公共 API 对照表 | `test_has_public_api_table` | 全部 6 仓库 |
| AGENTS.md 无过时 `__getattr__`/`_BLOCKED_ATTRS` | `test_no_legacy_getattr_description` | 全部 6 仓库 |
| AGENTS.md 机制版本更新 | `test_mechanism_version_updated` | 全部 6 仓库 |
| 现有代码基线已记录（不含 fork 目录报警） | ruff 扫描输出（人工记录报警清单） | 全部 6 仓库 |

**运行命令**（各仓库根目录执行）：

```bash
# OrcaGym
<conda-base>/envs/orca/bin/python -m pytest tests/orca_gym/test_ruff_config.py tests/orca_gym/test_agents_md.py -v
# OrcaFlow / OrcaEuler / OrcaPlayground / OrcaManipulation（各自根目录）
<conda-base>/envs/orca/bin/python -m unittest tests.test_ruff_config tests.test_agents_md -v
# OrcaLab（测试目录为 test/ 单数）
<conda-base>/envs/orca/bin/python -m unittest test.test_ruff_config test.test_agents_md -v
```

---

### 阶段 1：新建 OrcaGymEnvMixin

**目标**：从 `OrcaGymBaseEnv` 抽取与仿真引擎无关的公共方法到独立 Mixin 文件，不修改任何现有代码。

#### 3.1.1 新建 `orca_gym/environment/orca_gym_env_mixin.py`

**迁移方法清单**（从 `orca_gym/environment/orca_gym_env.py` 复制）：

| 方法 | 类型 | 依赖字段 |
|------|------|---------|
| `body(name, agent_id=None)` | 名称空间 | `self._agent_names` |
| `joint(name, agent_id=None)` | 名称空间 | `self._agent_names` |
| `actuator(name, agent_id=None)` | 名称空间 | `self._agent_names` |
| `site(name, agent_id=None)` | 名称空间 | `self._agent_names` |
| `mocap(name, agent_id=None)` | 名称空间 | `self._agent_names` |
| `sensor(name, agent_id=None)` | 名称空间 | `self._agent_names` |
| `_name_with_agent0(name)` | 辅助 | `self._agent_names` |
| `_name_with_agent(agent_id, name)` | 辅助 | `self._agent_names` |
| `generate_action_space(bounds)` | 空间生成 | 无 |
| `generate_observation_space(obs)` | 空间生成 | 无 |
| `reset(*, seed, options)` | reset 编排 | `self.reset_simulation()`/`self.reset_model()`/`self.render()`/`self.set_seed_value()` |
| `set_seed_value(seed)` | 随机种子 | 无 |
| `_get_reset_info()` | reset 辅助 | 无 |
| `agent_num` | property | `self._agent_names` |

**Mixin 设计要点**：

- 不定义 `__init__`，子类自行初始化 `_agent_names`/`frame_skip` 等字段
- `reset` 方法调用 `super().reset(seed=seed)`（走 MRO 到 `gym.Env.reset`），然后调用 `self.reset_simulation()`/`self.reset_model()`/`self.render()`（由 Env 子类提供）
- `dt` property **不迁入** Mixin（Euler 和 Local 实现不同）

**Mixin 骨架**：

```python
"""OrcaGym 环境公共方法 Mixin。

提供名称空间解析、动作/观测空间生成、reset 编排等方法。
不定义 __init__，不持有状态，子类自行初始化 _agent_names 等字段。
"""

from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Space


class OrcaGymEnvMixin:
    """OrcaGym 环境公共方法 Mixin（名称空间、空间生成、reset 编排）。

    子类必须提供以下字段/方法：
        - self._agent_names: list[str]
        - self.reset_simulation() -> None
        - self.reset_model() -> tuple[dict, dict]
        - self.render() -> Any
    """

    # --- 名称空间解析 ---
    def body(self, name: str, agent_id: int = None) -> str: ...
    def joint(self, name: str, agent_id: int = None) -> str: ...
    def actuator(self, name: str, agent_id: int = None) -> str: ...
    def site(self, name: str, agent_id: int = None) -> str: ...
    def mocap(self, name: str, agent_id: int = None) -> str: ...
    def sensor(self, name: str, agent_id: int = None) -> str: ...

    # --- 辅助 ---
    def _name_with_agent0(self, name: str) -> str: ...
    def _name_with_agent(self, agent_id: int, name: str) -> str: ...
    @property
    def agent_num(self) -> int: ...

    # --- 空间生成 ---
    def generate_action_space(self, bounds: NDArray[np.float64]) -> Space: ...
    def generate_observation_space(self, obs: Union[Dict[str, Any], np.ndarray]) -> Space: ...

    # --- reset 编排 ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None): ...
    def set_seed_value(self, seed: int = None) -> list: ...
    def _get_reset_info(self) -> Dict[str, float]: ...
```

**方法实现**：直接从 `orca_gym/environment/orca_gym_env.py` 对应方法复制函数体，不改逻辑。

#### 3.1.2 测试：`tests/orca_gym/environment/test_orca_gym_env_mixin.py`

```python
"""OrcaGymEnvMixin 单元测试。

验证 Mixin 方法存在性、可独立调用、不依赖引擎特定字段。
"""

import unittest
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

from orca_gym.environment.orca_gym_env_mixin import OrcaGymEnvMixin


class _DummyEnv(OrcaGymEnvMixin):
    """最小化 Env 桩，仅提供 Mixin 依赖的字段/方法。"""

    def __init__(self, agent_names: list[str]):
        self._agent_names = agent_names
        self._reset_called = False
        self._reset_model_called = False
        self._render_called = False

    def reset_simulation(self):
        self._reset_called = True

    def reset_model(self):
        self._reset_model_called = True
        return {"obs": np.zeros(3)}, {}

    def render(self):
        self._render_called = True
        return None


class TestMixinStructure(unittest.TestCase):
    """Mixin 结构约束。"""

    def test_mixin_has_no_init(self):
        """Mixin 不定义 __init__。"""
        self.assertNotIn("__init__", OrcaGymEnvMixin.__dict__)

    def test_mixin_methods_exist(self):
        """Mixin 包含全部 10 个公共方法。"""
        expected = [
            "body", "joint", "actuator", "site", "mocap", "sensor",
            "_name_with_agent0", "_name_with_agent",
            "generate_action_space", "generate_observation_space",
            "reset", "set_seed_value", "_get_reset_info",
            "agent_num",
        ]
        for name in expected:
            with self.subTest(method=name):
                self.assertTrue(hasattr(OrcaGymEnvMixin, name),
                                f"Mixin 缺少方法 '{name}'")


class TestMixinNamespace(unittest.TestCase):
    """名称空间解析。"""

    def test_body_with_agent0_prefix(self):
        env = _DummyEnv(["agent0", "agent1"])
        self.assertEqual(env.body("torso"), "agent0_torso")

    def test_body_with_agent_id(self):
        env = _DummyEnv(["agent0", "agent1"])
        self.assertEqual(env.body("torso", agent_id=1), "agent1_torso")

    def test_body_no_agent_names(self):
        env = _DummyEnv([])
        self.assertEqual(env.body("torso"), "torso")

    def test_all_namespace_methods_work(self):
        env = _DummyEnv(["agent0"])
        for method in ["body", "joint", "actuator", "site", "mocap", "sensor"]:
            with self.subTest(method=method):
                result = getattr(env, method)("test_name")
                self.assertEqual(result, "agent0_test_name")


class TestMixinSpaceGeneration(unittest.TestCase):
    """动作/观测空间生成。"""

    def test_generate_action_space(self):
        env = _DummyEnv(["agent0"])
        bounds = np.array([[0.0, 1.0], [-1.0, 1.0]])
        space = env.generate_action_space(bounds)
        self.assertIsInstance(space, Box)
        self.assertEqual(space.shape, (2,))

    def test_generate_observation_space_array(self):
        env = _DummyEnv(["agent0"])
        obs = np.zeros(5)
        space = env.generate_observation_space(obs)
        self.assertIsInstance(space, Box)
        self.assertEqual(space.shape, (5,))

    def test_generate_observation_space_dict(self):
        env = _DummyEnv(["agent0"])
        obs = {"a": np.zeros(3), "b": np.zeros(2)}
        space = env.generate_observation_space(obs)
        self.assertIsInstance(space, DictSpace)
        self.assertIn("a", space.spaces)
        self.assertIn("b", space.spaces)


class TestMixinReset(unittest.TestCase):
    """reset 编排。"""

    def test_reset_calls_lifecycle(self):
        env = _DummyEnv(["agent0"])
        env.reset()
        self.assertTrue(env._reset_called)
        self.assertTrue(env._reset_model_called)
        self.assertTrue(env._render_called)

    def test_reset_returns_obs_info(self):
        env = _DummyEnv(["agent0"])
        obs, info = env.reset()
        self.assertIn("obs", obs)
        self.assertIsInstance(info, dict)

    def test_reset_with_seed(self):
        env = _DummyEnv(["agent0"])
        env.reset(seed=42)
        self.assertEqual(env.seed_value, 42)


class TestMixinAgentNum(unittest.TestCase):
    """agent_num property。"""

    def test_agent_num(self):
        env = _DummyEnv(["agent0", "agent1", "agent2"])
        self.assertEqual(env.agent_num, 3)
```

#### 3.1.3 验收标准

| 验收项 | 验证方式 |
|--------|---------|
| Mixin 文件存在 | `orca_gym/environment/orca_gym_env_mixin.py` |
| Mixin 无 `__init__` | `test_mixin_has_no_init` |
| 10 个方法存在 | `test_mixin_methods_exist` |
| 名称空间解析正确 | `TestMixinNamespace` 全部通过 |
| 空间生成正确 | `TestMixinSpaceGeneration` 全部通过 |
| reset 编排正确 | `TestMixinReset` 全部通过 |
| 现有代码未受影响 | `test_orca_gym_euler_env_skeleton.py` 仍全部通过（Env 仍继承 BaseEnv） |

**运行命令**：

```bash
<conda-base>/envs/orca/bin/python -m pytest tests/orca_gym/environment/test_orca_gym_env_mixin.py -v
<conda-base>/envs/orca/bin/python -m pytest tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py -v
```

---

### 阶段 2：切换 Env 继承链 + 删除补丁机制

**目标**：修改 `OrcaGymEulerEnv` 继承链，删除 `_BLOCKED_ATTRS`/`__getattr__`/`__setattr__`/`_SHIELDED_ATTRS`，重写 `__init__` 自主编排生命周期。

#### 3.2.1 修改 `orca_gym/environment/euler/orca_gym_euler_env.py`

**变更清单**：

| # | 变更项 | 操作 |
|---|--------|------|
| 1 | import 语句 | 删除 `from ..orca_gym_env import OrcaGymBaseEnv`，新增 `from ..orca_gym_env_mixin import OrcaGymEnvMixin` |
| 2 | 类定义 | `class OrcaGymEulerEnv(OrcaGymBaseEnv)` → `class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env)` |
| 3 | 类 docstring | 更新"禁止"段落：删除"不要访问 env._gym._studio"，新增"env.gym/env.stub/env.channel 不存在" |
| 4 | `_BLOCKED_ATTRS` | 删除整个 frozenset 定义 |
| 5 | `_SHIELDED_ATTRS` | 删除整个 frozenset 定义 |
| 6 | `__setattr__` | 删除整个方法 |
| 7 | `__getattr__` | 删除整个方法 |
| 8 | `__init__` | 重写：不调 `super().__init__()`，自主编排生命周期 |
| 9 | `__dir__` | 更新：基于 `OrcaGymEnvMixin` + `OrcaGymEulerEnv` 公共方法构建，不依赖 `super().__dir__()` 过滤 |
| 10 | `dt` property 注释 | 更新：删除"替代父类的 self.gym.opt.timestep"说明 |
| 11 | `data` property 注释 | 更新：删除"替代父类的 self.data（被 __setattr__ 屏蔽赋值）"说明 |
| 12 | `model` property 注释 | 更新：删除"替代父类的 self.model（被 __setattr__ 屏蔽赋值）"说明 |
| 13 | 模块 docstring | 更新：删除"父类和解"相关描述 |

**`__init__` 重写要点**：

```python
def __init__(
    self,
    frame_skip: int,
    orcagym_addr: str,
    agent_names: list[str],
    time_step: float,
    *,
    model_xml_path: str | None = None,
    skip_grpc_load: bool = False,
    render_mode: str = "human",
    sync_render: bool = False,
    **kwargs,
) -> None:
    # 1. 基础字段（Mixin 依赖）
    self._agent_names = agent_names
    self.frame_skip = frame_skip
    self.orcagym_addr = orcagym_addr
    self.seed = 0

    # 2. Env 自有字段
    self._skip_grpc_load = skip_grpc_load
    self._local_xml_path = model_xml_path
    self._render_mode = render_mode
    self._sync_render = sync_render
    self._studio_bridge = None
    self._time_step = time_step
    self._render_count = 0.0
    self._render_count_interval = 0.0
    self._render_time_step = 0.0
    self._render_interval = 1.0 / self.metadata.get("render_fps", 30)
    self._last_frame_index = -1

    # 3. 事件循环（原 super().__init__ 中的逻辑，现在自主处理）
    self.loop = asyncio.get_event_loop()

    # 4. 生命周期编排（原 super().__init__ 中的编排，现在自主调用）
    self.initialize_grpc()
    self.pause_simulation()
    self.set_time_step(time_step)
    self.model, self.data = self.initialize_simulation()  # 注意：model/data 走 property，赋值无效但保留以对齐原编排
    self.reset_simulation()
    self.init_qpos_qvel()
```

**`__init__` 注意事项**：

- `self.model, self.data = self.initialize_simulation()` 这一行：由于 `model`/`data` 是 property（无 setter），直接赋值会抛 `AttributeError`。**改为不赋值**，仅调用 `self.initialize_simulation()`：
  ```python
  self.initialize_simulation()  # 内部设置 _gym，model/data 通过 property 读取
  ```
- 原代码中 `self.loop = asyncio.get_event_loop()` 在 `super().__init__` 内部执行，现在需在 `__init__` 中显式设置
- 原代码中事件循环保护逻辑（Python 3.12 兼容）保留：
  ```python
  try:
      asyncio.get_event_loop()
  except RuntimeError:
      asyncio.set_event_loop(asyncio.new_event_loop())
  self.loop = asyncio.get_event_loop()
  ```

**`__dir__` 重写要点**：

```python
def __dir__(self) -> list[str]:
    """只列出公共 API，不含内部组件或引擎内部。

    基于 OrcaGymEnvMixin + OrcaGymEulerEnv 公共方法构建，
    显式排除 _gym/_stub/_channel/_studio_bridge 等。
    """
    # 收集 Mixin + Env + gym.Env 的公共属性
    result = set()
    for cls in type(self).__mro__:
        for name in cls.__dict__:
            if not name.startswith("_") or name in ("__init__",):
                continue
            # 保留公共方法（不含 _ 前缀）
            result.add(name)
    # 也收集实例属性中的公共字段
    for name in self.__dict__:
        if not name.startswith("_"):
            result.add(name)
    # 显式排除内部组件（虽然它们带 _ 前缀已被过滤，但 double check）
    excluded = {"_gym", "_stub", "_channel", "_studio_bridge",
                "_mjData", "_mjModel", "gym", "stub", "channel"}
    return sorted(result - excluded)
```

**简化版 `__dir__`（推荐）**：

```python
def __dir__(self) -> list[str]:
    """只列出公共 API，不含内部组件。"""
    result = set(super().__dir__())
    # super().__dir__() 会列出 _gym 等，需要过滤
    excluded = {"_gym", "_stub", "_channel", "_studio_bridge",
                "_mjData", "_mjModel", "gym", "stub", "channel",
                "_skip_grpc_load", "_local_xml_path", "_render_mode",
                "_sync_render", "_time_step", "_render_count",
                "_render_count_interval", "_render_time_step",
                "_render_interval", "_last_frame_index", "_agent_names"}
    return sorted(result - excluded)
```

> 注：`super().__dir__()` 在 MRO 下走 `OrcaGymEnvMixin.__dir__` → `gym.Env.__dir__` → `object.__dir__`，会返回实例 `__dict__` + 类 `__dict__` 的并集。由于 `gym.Env` 本身不实现 `__dir__`，最终走 `object.__dir__`，返回全部属性。需要过滤 `_` 前缀的内部字段。

#### 3.2.2 测试：更新 `test_orca_gym_euler_env_skeleton.py`

**删除的测试类**：

| 测试类 | 原因 |
|--------|------|
| `TestEnvK2ViolationPatterns` | 依赖 `_BLOCKED_ATTRS`/`__getattr__` 拦截，改为 ruff SLF001 静态检查 |
| `TestEnvK10ParentShielding` | K10 删除，`__setattr__` 屏蔽机制不再存在 |

**删除的测试方法**（在保留的类中）：

| 测试方法 | 所在类 | 原因 |
|---------|--------|------|
| `test_env_blocked_attrs_raise_guidance` | `TestEnvK2Isolation` | 依赖 `_BLOCKED_ATTRS` |
| `test_env_blocked_attrs_message_has_guidance` | `TestEnvK2Isolation` | 依赖 `__getattr__` 引导消息 |
| `test_env_no_internal_property` | `TestEnvK2Isolation` | 检查 `gym`/`stub`/`channel` 不在类属性，现由 K14 覆盖 |
| `test_env_all_mjdata_mjmodel_variants_blocked` | `TestEnvK2ViolationPatterns` | 整类删除 |
| `test_env_all_internal_component_variants_blocked` | 同上 | 整类删除 |
| `test_env_multilayer_tunnel_*` | 同上 | 整类删除（6 个方法） |
| `test_env_k8_euler_tunnel_blocked` | 同上 | 整类删除 |
| `test_env_k9_studio_tunnel_blocked` | 同上 | 整类删除 |
| `test_env_blocked_attrs_frozenset_complete` | 同上 | 整类删除 |
| `test_parent_*_assignment_shielded` | `TestEnvK10ParentShielding` | 整类删除（5 个方法） |
| `test_shielded_attrs_frozenset_complete` | 同上 | 整类删除 |

**新增的测试类**：

```python
class TestEnvK14Inheritance(unittest.TestCase):
    """K14: 继承链约束 — 直接继承 gym.Env + OrcaGymEnvMixin，不继承 OrcaGymBaseEnv。"""

    def test_env_inheritance_chain(self):
        """OrcaGymEulerEnv.__bases__ 含 gym.Env 和 OrcaGymEnvMixin，不含 OrcaGymBaseEnv。"""
        from orca_gym.environment.orca_gym_env_mixin import OrcaGymEnvMixin
        from orca_gym.environment.orca_gym_env import OrcaGymBaseEnv
        import gymnasium as gym

        bases = OrcaGymEulerEnv.__bases__
        self.assertIn(OrcaGymEnvMixin, bases)
        self.assertIn(gym.Env, bases)
        self.assertNotIn(OrcaGymBaseEnv, bases)

    def test_env_gym_attr_natural_attribute_error(self):
        """env.gym 抛 AttributeError（Python 原生，属性不存在）。"""
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            _ = env.gym

    def test_env_stub_attr_natural_attribute_error(self):
        """env.stub 抛 AttributeError。"""
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            _ = env.stub

    def test_env_channel_attr_natural_attribute_error(self):
        """env.channel 抛 AttributeError。"""
        env = _make_skeleton_env()
        with self.assertRaises(AttributeError):
            _ = env.channel

    def test_env_no_blocked_attrs_classvar(self):
        """Env 类不定义 _BLOCKED_ATTRS / _SHIELDED_ATTRS / __getattr__ / __setattr__。"""
        class_attrs = vars(OrcaGymEulerEnv)
        self.assertNotIn("_BLOCKED_ATTRS", class_attrs)
        self.assertNotIn("_SHIELDED_ATTRS", class_attrs)
        self.assertNotIn("__getattr__", class_attrs)
        self.assertNotIn("__setattr__", class_attrs)

    def test_env_mixin_methods_available(self):
        """Env 通过 Mixin 继承获得 body/joint/actuator/site/mocap/sensor 等方法。"""
        env = _make_skeleton_env()
        mixin_methods = [
            "body", "joint", "actuator", "site", "mocap", "sensor",
            "generate_action_space", "generate_observation_space",
            "set_seed_value", "_get_reset_info", "agent_num",
        ]
        for method in mixin_methods:
            with self.subTest(method=method):
                self.assertTrue(callable(getattr(env, method, None)),
                                f"Env 缺少 Mixin 方法 '{method}'")

    def test_env_body_namespace_works(self):
        """env.body('torso') 返回 'agent0_torso'（Mixin 方法真实工作）。"""
        env = _make_skeleton_env()
        result = env.body("torso")
        self.assertEqual(result, "agent0_torso")

    def test_env_agent_num_works(self):
        """env.agent_num 返回 1（Mixin property 真实工作）。"""
        env = _make_skeleton_env()
        self.assertEqual(env.agent_num, 1)
```

**修改的测试方法**：

| 测试方法 | 所在类 | 修改内容 |
|---------|--------|---------|
| `test_env_dir_only_exposes_public_api` | `TestEnvK2Isolation` | 保留，验证 `dir(env)` 不含 `gym`/`stub`/`channel`/`_gym`/`_studio_bridge` |
| `test_env_dir_contains_public_api` | `TestEnvK2Isolation` | 保留，新增验证 `body`/`joint`/`actuator` 等 Mixin 方法在 `dir(env)` 中 |
| `test_env_no_gym_private_access` | `TestEnvK4NoGymPrivateAccess` | 保留（ruff SLF001 互补的源码审查） |
| `test_dt_uses_sim_config` | `TestEnvK7PropertyDelegation` | 保留，更新注释（删除"替代父类"描述） |

**保留不变的测试类**：

| 测试类 | 原因 |
|--------|------|
| `TestEnvK1NamingConstraint` | K1 仍有效（`_gym`/`_stub`/`_channel` 带下划线） |
| `TestEnvK2Isolation`（部分） | `__dir__` 测试仍有效 |
| `TestEnvK4NoGymPrivateAccess` | K4 仍有效（Env 不穿墙访问 Gym 私有） |
| `TestEnvK6DataView` | K6 仍有效 |
| `TestEnvK7PropertyDelegation` | K7 仍有效 |
| `TestEnvK8NoEulerPrivate` | K8 仍有效 |
| `TestEnvK9StudioAccess` | K9 仍有效（`studio_bridge` 是方法，`env.studio` 不存在） |
| `TestEnvK11TypedReturn` | K11 仍有效 |
| `TestEnvK12Docstring` | K12 仍有效 |
| `TestEnvLifecycleAndStepping` | 生命周期测试仍有效（`__init__` 重写后行为不变） |

#### 3.2.3 验收标准

| 验收项 | 验证方式 |
|--------|---------|
| 继承链正确 | `test_env_inheritance_chain` |
| `env.gym` 天然 AttributeError | `test_env_gym_attr_natural_attribute_error` |
| 补丁机制已删除 | `test_env_no_blocked_attrs_classvar` |
| Mixin 方法可用 | `test_env_mixin_methods_available` + `test_env_body_namespace_works` |
| `__dir__` 正确 | `test_env_dir_only_exposes_public_api` + `test_env_dir_contains_public_api` |
| K4/K8/K9 源码审查 | `test_env_no_gym_private_access` + `test_do_simulation_no_euler_private_access` + `test_no_studio_property_access` |
| 生命周期真实工作 | `TestEnvLifecycleAndStepping` 全部通过 |
| property 委托正确 | `TestEnvK7PropertyDelegation` 全部通过 |

**运行命令**：

```bash
<conda-base>/envs/orca/bin/python -m pytest tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py -v
```

#### 3.2.4 设计决策说明：`gym.Env` 不写泛型参数

**问题背景**：原 `OrcaGymBaseEnv` 继承 `gym.Env[NDArray[np.float64], NDArray[np.float32]]`，阶段 2 切换后的 `OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env)` 未写泛型参数。需明确该决策的依据。

**调研结论**：`gym.Env[A, B]` 与 `gym.Env` 在运行时完全等价（`__class_getitem__` 返回原类），泛型参数仅服务于 mypy/pyright 静态类型检查。

| 项目 | 实际写法 | 是否写泛型 |
|------|---------|-----------|
| Gymnasium 官方文档示例 `GridWorldEnv` | `class GridWorldEnv(gym.Env)` | ❌ 不写 |
| SB3 自定义环境文档 `CustomEnv` | `class CustomEnv(gym.Env)` | ❌ 不写 |
| Gymnasium 经典环境 `MujocoEnv` | `class MujocoEnv(gym.Env)` | ❌ 不写 |
| SB3 测试用例 `IdentityEnv` | `class IdentityEnv(gym.Env, Generic[T])` | ✅ 写（测试场景） |
| 原 `OrcaGymBaseEnv` | `class OrcaGymBaseEnv(gym.Env[NDArray[np.float64], NDArray[np.float32]])` | ✅ 写 |

**原 BaseEnv 泛型的缺陷**：`OrcaGymBaseEnv` 作为抽象基类，将 `ObsType=np.float64`、`ActType=np.float32` 固化到基类签名，但子类（多智能体、机械臂、车辆）的 obs/act 类型未必都是 `NDArray[np.float64]/NDArray[np.float32]`。正确做法应让基类保持 `Generic[ObsType, ActType]`，由子类按需参数化。且 OrcaGym 项目从未配置 mypy/pyright 检查，泛型参数实际是死代码。

**决策**：`OrcaGymEulerEnv` 保持 `class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env)`，不写泛型参数。

**理由**：

1. **与官方/SB3 主流实践一致**：和 `GridWorldEnv`、`CustomEnv`、`MujocoEnv` 同款写法
2. **降低维护成本**：不写错误的泛型比写错的泛型好（原 BaseEnv 的固化类型就是错误示范）
3. **OrcaGymEulerEnv 是基类**：子类（SimpleEulerEnv 等）obs/act 类型由各自场景决定，不应固化
4. **项目未配置 mypy**：写了也是死代码，无实际校验

**未来路径**：若项目启用 mypy/pyright 检查，正确做法是让 Mixin/Env 保持泛型变量开放：

```python
from typing import Generic, TypeVar
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class OrcaGymEnvMixin(Generic[ObsType, ActType]): ...
class OrcaGymEulerEnv(
    OrcaGymEnvMixin[NDArray[np.float64], NDArray[np.float32]],
    gym.Env[NDArray[np.float64], NDArray[np.float32]],
): ...
```

但这是未来工程，当前阶段保持简单，不引入 TypeVar/Generic 复杂度。

---

### 阶段 3：端到端验证

**目标**：验证继承链切换后，已填充的功能（阶段二/三的 query_*/set_* 方法）和 SimpleEulerEnv 端到端流程不受影响。

#### 3.3.1 测试：SimpleEulerEnv 合规与功能

**运行现有测试**（不修改）：

```bash
<conda-base>/envs/orca/bin/python -m pytest tests/orca_gym/environment/euler/test_simple_euler_env_compliance.py -v
```

**验收标准**：

| 验收项 | 验证方式 |
|--------|---------|
| SimpleEulerEnv 不穿墙 | `TestSimpleEnvK3K5NoTunnelAccess` 全部通过 |
| reset_model 功能正确 | `TestSimpleEnvResetModelFunctional` 全部通过 |
| step 后 time 累计正确 | `test_step_works_after_reset_model` 通过 |

#### 3.3.2 测试：Gym 层测试不受影响

**运行现有测试**（不修改）：

```bash
<conda-base>/envs/orca/bin/python -m pytest tests/orca_gym/core/euler/ -v
```

**验收标准**：全部通过（Gym 层不涉及继承链变更）。

#### 3.3.3 测试：ruff SLF001 静态扫描

**配置 `pyproject.toml`**（若尚未配置）：

```toml
[tool.ruff.lint]
select = ["SLF001"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["SLF001"]
"**/__init__.py" = ["SLF001", "F401"]
```

**运行扫描**：

```bash
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca_gym/environment/euler/orca_gym_euler_env.py
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca_gym/core/euler/
```

**验收标准**：

| 验收项 | 验证方式 |
|--------|---------|
| Env 源码零报警 | `ruff check` 退出码 0 |
| Gym 源码零报警 | `ruff check` 退出码 0 |

> 注：`self._gym` 在 Env 内部访问是合法的（类内部访问），SLF001 不报警。`env._gym._sim` 等外部穿墙访问会报警，但 Env 源码中不出现此类访问。

#### 3.3.4 测试：Example 端到端运行

**实际 example 目录结构**（`OrcaPlayground/examples/euler/`）：

```
examples/euler/
├── 01_hello_euler/hello_euler.py      # 离线：随机动作驱动倒立摆，验证 API 契约
├── 02_online_render/online_render.py  # 在线：连接 OrcaStudio 实时渲染（需 gRPC 服务）
└── 03_rl_ppo/train_ppo.py             # 离线：SB3 PPO 训练倒立摆
```

**运行离线 example**（01 和 03，无需 OrcaStudio）：

```bash
# 01_hello_euler（离线模式，验证 API 契约）
cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python examples/euler/01_hello_euler/hello_euler.py --steps 200

# 03_rl_ppo（离线模式，Gymnasium 契约 + SB3 训练）
cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 20000
```

**验收标准**：

| 验收项 | 验证方式 |
|--------|---------|
| 01_hello_euler 运行无异常 | 脚本退出码 0，输出含「第 1 课验证通过」 |
| 01_hello_euler API 契约正确 | 输出含 nq/nv/nu、qpos.shape、sim_config.timestep |
| 03_rl_ppo 训练启动正常 | 脚本退出码 0，输出含 reward 训练曲线 |
| 03_rl_ppo Gymnasium 契约 | step 返回 5 元组，reset 返回 2 元组，无 AttributeError |

> 注：02_online_render 需要 OrcaStudio gRPC 服务，CI/沙盒环境无法运行，阶段 3 跳过。

---

## 4. 实施顺序与依赖

```
阶段 0: 基础设施先行（AGENTS.md + ruff）
   │   （配置 ruff SLF001、更新 AGENTS.md、建立扫描基线）
   ▼
阶段 1: 新建 OrcaGymEnvMixin
   │   （无破坏性，现有测试不受影响）
   ▼
阶段 2: 切换 Env 继承链 + 删除补丁机制
   │   （核心变更，Env 测试需同步更新，同步修复阶段 0 基线报警）
   ▼
阶段 3: 端到端验证
       （SimpleEulerEnv + Gym 层 + ruff 零报警 + example）
```

**关键约束**：

- 阶段 0 必须先于阶段 1-3 完成（ruff 与 AGENTS.md 是后续编码的静态检查基线和 AI 行为约束）
- 阶段 1 和阶段 2 不可并行（阶段 2 依赖阶段 1 的 Mixin 文件）
- 阶段 2 内部的代码变更和测试更新必须同步提交（避免中间状态测试失败）
- 阶段 2 须同步修复阶段 0 记录的基线报警（改为公共 API 委托）
- 阶段 3 必须在阶段 2 全部通过后执行，ruff SLF001 零报警为硬性验收项

---

## 5. 回滚策略

若阶段 2 或阶段 3 发现不可解决的问题：

1. **代码回滚**：`git revert` 阶段 2 的提交，恢复 `OrcaGymBaseEnv` 继承
2. **Mixin 保留**：阶段 1 的 `OrcaGymEnvMixin` 文件可保留（无破坏性），待问题解决后重新尝试阶段 2
3. **测试回滚**：`git revert` 阶段 2 的测试更新提交

**阶段 0 回滚**：`AGENTS.md`/`pyproject.toml` 的变更可独立 `git revert`，不影响现有代码运行（ruff 配置仅新增不删除既有依赖）。阶段 0 的两个测试文件可保留或删除。

**不可回滚的情况**：无。本方案不修改 `OrcaGymBaseEnv`/`OrcaGymLocalEnv`/Gym 层/子组件，仅变更 Env 继承链。

---

## 6. 完成标志

全部以下条件满足时，骨架变更完成：

| # | 条件 | 验证方式 |
|---|------|---------|
| 1 | 6 仓库配置文件含 ruff SLF001 配置 | 阶段 0 各仓库 `test_ruff_config_has_slf001` |
| 2 | 6 仓库 ruff 可执行 | 阶段 0 各仓库 `test_ruff_installed` |
| 3 | 6 仓库 AGENTS.md 含 API 隔离章节且无过时描述 | 阶段 0 各仓库 `test_agents_md.py` 全部通过 |
| 4 | OrcaFlow `warp_fork/` 已排除且 6 仓库现有代码 ruff 基线已记录 | 阶段 0 扫描输出 |
| 5 | `OrcaGymEnvMixin` 文件存在且测试通过 | 阶段 1 测试 |
| 6 | `OrcaGymEulerEnv` 继承 `OrcaGymEnvMixin, gym.Env` | `test_env_inheritance_chain` |
| 7 | `env.gym`/`env.stub`/`env.channel` 抛 `AttributeError` | `test_env_gym_attr_natural_attribute_error` 等 |
| 8 | `_BLOCKED_ATTRS`/`__getattr__`/`__setattr__`/`_SHIELDED_ATTRS` 已删除 | `test_env_no_blocked_attrs_classvar` |
| 9 | Mixin 方法在 Env 上可用 | `test_env_mixin_methods_available` |
| 10 | Env 骨架测试全部通过 | 阶段 2 测试 |
| 11 | SimpleEulerEnv 合规测试通过 | 阶段 3 测试 |
| 12 | Gym 层测试全部通过 | 阶段 3 测试 |
| 13 | OrcaGym ruff SLF001 零报警（含阶段 0 基线报警已修复） | 阶段 3 扫描 |
| 14 | Example 端到端运行正常 | 阶段 3 example |
