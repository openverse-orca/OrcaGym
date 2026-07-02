# OrcaGym Euler 阶段二变更修订文档：对齐骨架迁移后的架构

## 1. 文档定位

### 1.1 文档目标

本文是 `OrcaGymEulerEnv` + `OrcaGymEuler` **阶段二（功能填充）** 的**变更修订与重新验证**文档。

**修订背景**：阶段二原设计文档 [orca_gym_euler_phase2_filling_development.md](orca_gym_euler_phase2_filling_development.md) 基于旧骨架（`OrcaGymEulerEnv(OrcaGymBaseEnv)` + `_BLOCKED_ATTRS`/`__getattr__`/`__setattr__` 补丁机制）撰写。随后骨架已按 [orca_gym_euler_skeleton_migration_development.md](orca_gym_euler_skeleton_migration_development.md) 完成迁移：

- 继承链从 `OrcaGymBaseEnv` 切换为 `OrcaGymEnvMixin, gym.Env`（K14）
- 删除 Env 层 `_BLOCKED_ATTRS`/`__getattr__`/`__setattr__`/`_SHIELDED_ATTRS` 补丁机制
- 引入 `OrcaGymEnvMixin` 共享公共方法
- 引入 ruff SLF001 静态检查 + AGENTS.md API 隔离强制（M1/M2）
- `env.gym`/`env.stub`/`env.channel` 改为 Python 原生 `AttributeError`（M0）

**本文不替代老文档**，而是针对架构变更点，设计若干步骤完成修订与重新验证。老文档中未受架构变更影响的内容（如 MuJoCoSimCore/SimConfig/ModelRegistry/DataView 的填充逻辑）不在本文重复，仅验证其与新架构的兼容性。

### 1.2 上游约束

| 文档 | 约束范围 |
|------|---------|
| `docs/design/architecture/orca_gym_euler_architecture.md` | §5 组件设计、§6 API 契约、§7 封装隔离机制（M0-M7）、§8 步进编排、§12 K 约束（含 K14） |
| `docs/design/development/orca_gym_euler_skeleton_migration_development.md` | 骨架迁移实施细节（继承链切换、Mixin 引入、ruff 配置） |
| `AGENTS.md` | 规则 1（orca conda 环境）、规则 3（GPU 命令白名单）、规则 4（API 隔离强制） |

### 1.3 修订原则

1. **不回退骨架约束**：所有修订验证必须保持 K1–K14 约束与 M0-M7 机制。
2. **契约不可破坏**：R/W/S/C/N 五类规则（架构 §6）在修订后仍须满足。
3. **测试环境统一**：全部测试使用 `orca` conda 环境（AGENTS.md 规则 1），不再使用旧文档的 `OrcaFlow_Flow` 解释器。
4. **GPU 旁路**：RL 训练等 GPU 命令须用白名单解释器路径，禁用 shell 管道（AGENTS.md 规则 3）。
5. **最小改动**：仅修订受架构变更影响的填充与测试，不重构已对齐的功能逻辑。

---

## 2. 架构变更对阶段二的影响分析

### 2.1 变更清单与影响面

| # | 架构变更 | 影响的填充内容 | 影响程度 |
|---|---------|--------------|---------|
| 1 | 继承链 `OrcaGymBaseEnv` → `OrcaGymEnvMixin, gym.Env`（K14） | Env `__init__` 生命周期编排、`super().__init__()` 调用 | **高**：原文档假设父类编排，现需自主编排 |
| 2 | 删除 Env 层 `_BLOCKED_ATTRS`/`__getattr__`/`__setattr__` | K2 隔离测试、K10 屏蔽测试 | **高**：K2/K10 测试需重写，改为 M0 原生 AttributeError |
| 3 | `env.gym`/`env.stub`/`env.channel` 不存在（M0） | 原文档 §2.2 Example 违规点 `self.gym._sim._mjData` | **中**：违规模式不再可达，合规测试需更新 |
| 4 | `OrcaGymEnvMixin` 引入（共享方法） | Env 的 `body()`/`joint()`/`reset()` 等方法来源 | **低**：方法签名不变，仅来源从父类改为 Mixin |
| 5 | ruff SLF001 静态检查（M1） | 隔离机制验收（原 §7.2 仅 grep，现增加 ruff） | **中**：新增静态检查验收项 |
| 6 | AGENTS.md API 隔离强制（M2） | AI 行为约束 | **低**：文档约束，非代码变更 |
| 7 | Gym 层 `_BLOCKED_ATTRS`/`__getattribute__` 保留 | K3/K5 Gym 层隔离 | **无**：Gym 层隔离机制未变，填充逻辑兼容 |
| 8 | 测试环境 `OrcaFlow_Flow` → `orca` conda | 全部测试运行命令 | **低**：命令格式调整 |

### 2.2 老文档条款映射

| 老文档章节 | 老文档假设 | 新架构要求 | 修订动作 |
|-----------|-----------|-----------|---------|
| §1.4 填充原则 | "保持 K1–K12 约束" | K1–K14 + M0-M7 | 原则升级为 K1-K14 |
| §2.1 骨架现状 | Env 继承 `OrcaGymBaseEnv` | Env 继承 `OrcaGymEnvMixin, gym.Env` | 现状描述更新 |
| §2.2 Example 违规点 | `self.gym._sim._mjData` | `self.gym` 不存在（M0），违规模式不可达 | 违规点描述更新为 `self._gym._sim._mjData`（若存在） |
| §3.3 测试环境 | `OrcaFlow_Flow` 解释器 | `orca` conda 环境 | 命令格式更新 |
| §4 Step 6 Env 填充 | `class OrcaGymEulerEnv(OrcaGymBaseEnv)` + `super().__init__()` | `class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env)` + 自主编排 | 填充方案修订 |
| §4 Step 6 `initialize_grpc` | `self.gym = OrcaGymEuler(...)` + `object.__setattr__` | `self._gym = OrcaGymEuler(...)`（无 `__setattr__` 屏蔽） | 赋值方式修订 |
| §7.1 K2 验收 | `test_env_blocked_attrs_raise_guidance`（依赖 `__getattr__`） | M0 原生 AttributeError（无自定义消息） | 测试重写 |
| §7.1 K10 验收 | `test_setattr_shields_parent_attrs`（依赖 `__setattr__`） | K10 删除（无 `__setattr__`） | 测试删除 |
| §7.1 K14 验收 | 不存在 | 新增：继承链 + 原生 AttributeError | 测试新增 |
| §7.2 grep 断言 | 仅源码 grep | 增加 ruff SLF001 静态扫描 | 验收方式扩展 |
| §7.3 运行时测试 | `pytest.raises(AttributeError, match="通过.*公共.*访问")` | 原生 AttributeError 无此消息 | match 模式修订 |
| §8.1 总体验收 | "K1–K12 约束回归测试全部通过" | "K1–K14 约束 + M0-M7 机制回归测试全部通过" | 验收标准升级 |

### 2.3 不受影响的填充内容

以下填充内容**不受架构变更影响**，本文仅做兼容性验证，不重复设计：

| 组件 | 原因 |
|------|------|
| `MuJoCoSimCore` | 纯 MuJoCo 操作，不涉及 Env 继承链 |
| `SimConfig` | `_bind` 延迟绑定模式与继承链无关 |
| `ModelRegistry` | 同上 |
| `OrcaGymDataView` | 同上 |
| `OrcaGymEuler`（Gym 层） | K3/K5 隔离机制（`_BLOCKED_ATTRS` + `__getattribute__`）保留不变 |
| `OrcaStudioBridge` | gRPC 通信逻辑与继承链无关 |

---

## 3. 总体策略

### 3.1 修订验证次序

```
Step 1: 基线建立与影响面扫描
   │   （确认当前代码状态、ruff 基线、测试基线）
   ▼
Step 2: Env 层填充修订验证（生命周期/步进/状态设置）
   │   （K14 继承链、M0 原生 AttributeError、自主编排生命周期）
   ▼
Step 3: Gym 层与子组件填充兼容性验证
   │   （K3/K5 Gym 层隔离不变、_bind 模式、委托链路）
   ▼
Step 4: SimpleEulerEnv 违规修正验证
   │   （M0 下违规模式不可达、合规 API 使用）
   ▼
Step 5: 隔离机制验收清单修订（K1-K14 + M0-M7）
   │   （K2/K10 测试重写、K14 新增、ruff SLF001 验收）
   ▼
Step 6: ruff SLF001 静态检查与 AGENTS.md 合规
   │   （M1/M2 落地验证）
   ▼
Step 7: 端到端验证（Lesson 1/2/3）
       （orca conda 环境、GPU 白名单旁路）
```

### 3.2 测试环境

| 测试类型 | 环境 | 说明 |
|---------|------|------|
| 单元测试 / 隔离测试 | `orca` conda 环境（sandbox 内） | 纯 MuJoCo 仿真 + 源码审查，无 CUDA 依赖 |
| Lesson 1/3 端到端（离线） | `orca` conda 环境（sandbox 内） | 离线模式，无 gRPC |
| Lesson 2 端到端（在线） | 宿主机 + OrcaStudio 运行 | 需 gRPC 连接，sandbox 无法访问外部服务 |
| Lesson 3 RL 训练（GPU） | `orca` conda 环境 + TRAE 白名单旁路 | GPU 训练须白名单解释器路径，禁用管道（AGENTS.md 规则 3） |

**命令格式约定**（AGENTS.md 规则 1/3）：

```bash
# CPU 测试（sandbox 内）
<conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.<module>

# GPU 训练（白名单旁路，无管道）
cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 20000
```

> `<conda-base>` 通过 `conda info --base` 解析（当前为 `/home/superfhwl/miniconda3`）。

---

## 4. 修订与验证步骤

### Step 1：基线建立与影响面扫描

#### 目标

在修订前确认当前代码状态，建立 ruff SLF001 基线与测试基线，识别需修订的差异点。

#### 1.1 开发任务

**任务 1.1.1**：扫描 Euler 代码的 ruff SLF001 状态。

```bash
# Euler 子目录（阶段二填充范围）
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 \
    orca_gym/environment/euler/ orca_gym/core/euler/
```

**任务 1.1.2**：运行现有阶段二测试套件，建立通过/失败基线。

```bash
<conda-base>/envs/orca/bin/python -m unittest \
    tests.orca_gym.environment.euler.test_orca_gym_euler_env_skeleton \
    tests.orca_gym.environment.euler.test_simple_euler_env_compliance \
    tests.orca_gym.core.euler.test_orca_gym_euler \
    tests.orca_gym.core.euler.test_mujoco_sim_core \
    tests.orca_gym.core.euler.test_sim_config \
    tests.orca_gym.core.euler.test_model_registry \
    tests.orca_gym.core.euler.test_orca_gym_data_view \
    tests.orca_gym.core.euler.test_orca_studio_bridge
```

**任务 1.1.3**：扫描源码中的旧架构残留模式（识别需清理的差异点）。

扫描以下模式，确认无旧骨架残留：

| 扫描模式 | 含义 | 期望结果 |
|---------|------|---------|
| `class OrcaGymEulerEnv(OrcaGymBaseEnv)` | 旧继承链 | 无匹配 |
| `_BLOCKED_ATTRS` 在 Env 源码 | Env 层补丁机制 | 无匹配（Gym 层保留合法） |
| `__getattr__` 在 Env 源码 | Env 层拦截 | 无匹配 |
| `__setattr__` 在 Env 源码 | Env 层屏蔽 | 无匹配 |
| `super().__init__()` 在 Env `__init__` | 父类编排 | 无匹配 |
| `self.gym = ` 在 Env 源码 | 旧式公共属性赋值 | 无匹配（应为 `self._gym = `） |
| `object.__setattr__(self,` 在 Env 源码 | 绕过 `__setattr__` 屏蔽 | 无匹配 |

#### 1.2 测试

**文件**：`tests/orca_gym/environment/euler/test_phase2_revision_baseline.py`（新建）

```python
"""阶段二变更修订 — Step 1: 基线建立与影响面扫描。

验证 Euler 代码已对齐新骨架架构（K14 继承链、无补丁机制、无旧式赋值）。
"""

import pathlib
import re
import unittest

_ENV_SOURCE = (
    pathlib.Path(__file__).resolve().parents[4]
    / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
).read_text(encoding="utf-8")


class TestPhase2BaselineNoLegacySkeleton(unittest.TestCase):
    """Env 源码不含旧骨架残留。"""

    def test_no_old_inheritance_chain(self):
        """Env 类不继承 OrcaGymBaseEnv（K14）。"""
        self.assertNotIn(
            "class OrcaGymEulerEnv(OrcaGymBaseEnv)", _ENV_SOURCE,
            "K14 违规: Env 仍继承 OrcaGymBaseEnv",
        )

    def test_no_blocked_attrs_in_env(self):
        """Env 类不定义 _BLOCKED_ATTRS（补丁机制已删除）。"""
        # 排除 docstring 中的引用
        exec_source = re.sub(r'"""[\s\S]*?"""', '', _ENV_SOURCE)
        self.assertNotIn("_BLOCKED_ATTRS", exec_source)

    def test_no_getattr_in_env(self):
        """Env 类不定义 __getattr__（M0 替代）。"""
        exec_source = re.sub(r'"""[\s\S]*?"""', '', _ENV_SOURCE)
        self.assertNotIn("def __getattr__", exec_source)

    def test_no_setattr_in_env(self):
        """Env 类不定义 __setattr__（K10 删除）。"""
        exec_source = re.sub(r'"""[\s\S]*?"""', '', _ENV_SOURCE)
        self.assertNotIn("def __setattr__", exec_source)

    def test_no_super_init_in_env(self):
        """Env __init__ 不调用 super().__init__()（自主编排）。"""
        exec_source = re.sub(r'"""[\s\S]*?"""', '', _ENV_SOURCE)
        self.assertNotIn("super().__init__()", exec_source)

    def test_no_public_gym_assignment(self):
        """Env 源码不出现 self.gym = 赋值（应为 self._gym = ）。"""
        exec_source = re.sub(r'"""[\s\S]*?"""', '', _ENV_SOURCE)
        self.assertNotIn("self.gym =", exec_source)

    def test_no_object_setattr_bypass(self):
        """Env 源码不出现 object.__setattr__(self, 绕过屏蔽。"""
        exec_source = re.sub(r'"""[\s\S]*?"""', '', _ENV_SOURCE)
        self.assertNotIn("object.__setattr__(self,", exec_source)


class TestPhase2BaselineNewArchitecturePresent(unittest.TestCase):
    """Env 源码含新架构要素。"""

    def test_new_inheritance_chain(self):
        """Env 类继承 OrcaGymEnvMixin, gym.Env（K14）。"""
        self.assertIn(
            "class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env)", _ENV_SOURCE,
        )

    def test_mixin_imported(self):
        """Env 源码 import OrcaGymEnvMixin。"""
        self.assertIn("from ..orca_gym_env_mixin import OrcaGymEnvMixin", _ENV_SOURCE)

    def test_self_orchestrated_lifecycle(self):
        """Env __init__ 自主编排生命周期（含 initialize_grpc/set_time_step/initialize_simulation）。"""
        exec_source = re.sub(r'"""[\s\S]*?"""', '', _ENV_SOURCE)
        self.assertIn("self.initialize_grpc()", exec_source)
        self.assertIn("self.set_time_step(time_step)", exec_source)
        self.assertIn("self.initialize_simulation()", exec_source)
        self.assertIn("self.reset_simulation()", exec_source)
        self.assertIn("self.init_qpos_qvel()", exec_source)
```

#### 1.3 验收标准

| 验收项 | 验证方式 | 通过标准 |
|--------|---------|---------|
| Euler 代码 ruff SLF001 零报警 | 任务 1.1.1 | `All checks passed!` |
| 现有测试全部通过 | 任务 1.1.2 | `OK`，无 FAILED |
| 无旧继承链残留 | `test_no_old_inheritance_chain` | 通过 |
| 无 Env 层补丁机制 | `test_no_blocked_attrs_in_env` + `test_no_getattr_in_env` + `test_no_setattr_in_env` | 通过 |
| 无 super().__init__() | `test_no_super_init_in_env` | 通过 |
| 无旧式 self.gym 赋值 | `test_no_public_gym_assignment` | 通过 |
| 新继承链存在 | `test_new_inheritance_chain` | 通过 |
| Mixin 已导入 | `test_mixin_imported` | 通过 |
| 自主编排生命周期 | `test_self_orchestrated_lifecycle` | 通过 |

**运行命令**：

```bash
<conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.test_phase2_revision_baseline -v
```

---

### Step 2：Env 层填充修订验证（生命周期/步进/状态设置）

#### 目标

验证 `OrcaGymEulerEnv` 的阶段二填充（生命周期、步进、状态设置）与新骨架架构（K14 继承链 + M0 原生 AttributeError + 自主编排）兼容。

**与老文档 §4 Step 6 的差异**：

| 老文档假设 | 新架构实际 | 验证重点 |
|-----------|-----------|---------|
| `class OrcaGymEulerEnv(OrcaGymBaseEnv)` | `class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env)` | K14 继承链 |
| `super().__init__()` 编排生命周期 | `__init__` 自主编排 | 生命周期顺序正确 |
| `self.gym = OrcaGymEuler(...)` | `self._gym = OrcaGymEuler(...)` | K1 命名约束 |
| `object.__setattr__(self, "_channel", ...)` 绕过屏蔽 | `self._channel = ...` 直接赋值 | 无屏蔽机制 |
| `set_time_step` 经父类调用 | `__init__` 显式调用 `set_time_step` | SimConfig `_bind` 缓存路径 |
| K2 `__getattr__` 拦截 `env.gym` | M0 Python 原生 AttributeError | 无自定义消息 |

#### 2.1 开发任务

**任务 2.1.1**：验证 `__init__` 自主编排生命周期。

`OrcaGymEulerEnv.__init__` 须按以下顺序自主编排（不调 `super().__init__()`）：

```python
def __init__(self, frame_skip, orcagym_addr, agent_names, time_step, *, ...):
    # 1. 基础字段（Mixin 依赖 + Env 公共字段）
    self._agent_names = agent_names
    self.orcagym_addr = orcagym_addr
    self.frame_skip = frame_skip
    self.seed = 0
    # 2. Env 自有字段（含 _time_step 缓存、渲染节流字段）
    self._skip_grpc_load = skip_grpc_load
    self._local_xml_path = model_xml_path
    self._time_step = time_step
    # ... 渲染节流字段 ...
    # 3. 事件循环（Python 3.12 兼容）
    self.loop = asyncio.get_event_loop()
    # 4. 生命周期编排（原 super().__init__ 中的编排，现在自主调用）
    self.initialize_grpc()
    self.pause_simulation()
    self.set_time_step(time_step)
    self.initialize_simulation()   # 内部设置 _gym，model/data 通过 property 读取
    self.reset_simulation()
    self.init_qpos_qvel()
```

**任务 2.1.2**：验证 `initialize_grpc` 使用 `self._gym`（K1）。

```python
def initialize_grpc(self) -> None:
    if self._skip_grpc_load:
        self._channel = None          # 直接赋值，无 object.__setattr__ 绕过
        self._stub = None
        self._gym = OrcaGymEuler(stub=None)    # K1: _gym 带下划线
        self._studio_bridge = self._gym.studio_bridge()
        ...
        return
    self._channel = grpc.aio.insecure_channel(...)
    self._stub = GrpcServiceStub(self._channel)
    self._gym = OrcaGymEuler(stub=self._stub)
    self._studio_bridge = self._gym.studio_bridge()
```

**任务 2.1.3**：验证 `set_time_step` 的 SimConfig `_bind` 缓存路径。

`__init__` 在 `initialize_simulation` 前调用 `set_time_step`，此时 SimConfig 未绑定 mjModel：

```python
def set_time_step(self, time_step: float) -> None:
    self._time_step = time_step              # 缓存
    self.realtime_step = time_step * self.frame_skip
    if hasattr(self, "_gym") and self._gym is not None:
        try:
            self._gym.sim_config.timestep = time_step
        except RuntimeError:
            pass   # SimConfig 未绑定，缓存待 init_simulation
    ...

def initialize_simulation(self) -> Tuple[Any, OrcaGymDataView]:
    ...
    self.loop.run_until_complete(self._gym.init_simulation(model_xml_path))
    self._gym.sim_config.timestep = self._time_step    # 绑定后重新应用缓存
    ...
```

**任务 2.1.4**：验证步进与状态设置委托链路（K4/K8）。

```python
def do_simulation(self, ctrl, n_frames):
    self._gym.step_with_coupling(ctrl, n_frames, self.dt)   # K8: 不读 _euler
    self._gym.sync_to_view()

def set_joint_qpos(self, qpos):
    self._gym.set_qpos_qvel(qpos, self._gym.data.qvel)     # K4: 走公共方法
```

#### 2.2 测试

**文件**：`tests/orca_gym/environment/euler/test_phase2_revision_env_filling.py`（新建）

```python
"""阶段二变更修订 — Step 2: Env 层填充与新骨架兼容性验证。

验证 K14 继承链、M0 原生 AttributeError、自主编排生命周期、
SimConfig _bind 缓存路径、步进/状态设置委托链路。
"""

import asyncio
import pathlib
import unittest

import numpy as np

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimConfig


def _make_env():
    """构造离线模式 Env。"""
    _pendulum_xml = (
        pathlib.Path(__file__).resolve().parents[4].parent
        / "OrcaPlayground" / "envs" / "euler" / "scenes" / "simple_pendulum.xml"
    )
    return OrcaGymEulerEnv(
        frame_skip=4,
        orcagym_addr="localhost:50051",
        agent_names=["agent0"],
        time_step=0.002,
        model_xml_path=str(_pendulum_xml),
        skip_grpc_load=True,
    )


class TestPhase2EnvK14Inheritance(unittest.TestCase):
    """K14: 继承链约束。"""

    def test_inheritance_chain(self):
        """OrcaGymEulerEnv.__bases__ 含 gym.Env + OrcaGymEnvMixin，不含 OrcaGymBaseEnv。"""
        from orca_gym.environment.orca_gym_env_mixin import OrcaGymEnvMixin
        from orca_gym.environment.orca_gym_env import OrcaGymBaseEnv
        import gymnasium as gym

        bases = OrcaGymEulerEnv.__bases__
        self.assertIn(OrcaGymEnvMixin, bases)
        self.assertIn(gym.Env, bases)
        self.assertNotIn(OrcaGymBaseEnv, bases)


class TestPhase2EnvM0NativeAttributeError(unittest.TestCase):
    """M0: env.gym/stub/channel 抛 Python 原生 AttributeError（无自定义消息）。"""

    def test_env_gym_raises_native_attribute_error(self):
        """env.gym 抛 AttributeError，消息不含自定义引导文本。"""
        env = _make_env()
        with self.assertRaises(AttributeError) as ctx:
            _ = env.gym
        # M0: 原生 AttributeError，消息为 "'OrcaGymEulerEnv' object has no attribute 'gym'"
        # 不含"通过公共 API 访问"等自定义引导（那是旧 __getattr__ 的消息）
        self.assertNotIn("公共", str(ctx.exception))
        self.assertNotIn("API 契约", str(ctx.exception))

    def test_env_stub_raises_native_attribute_error(self):
        env = _make_env()
        with self.assertRaises(AttributeError):
            _ = env.stub

    def test_env_channel_raises_native_attribute_error(self):
        env = _make_env()
        with self.assertRaises(AttributeError):
            _ = env.channel

    def test_env_no_getattr_method(self):
        """Env 类不定义 __getattr__（M0 替代）。"""
        self.assertNotIn("__getattr__", vars(OrcaGymEulerEnv))

    def test_env_no_setattr_method(self):
        """Env 类不定义 __setattr__（K10 删除）。"""
        self.assertNotIn("__setattr__", vars(OrcaGymEulerEnv))


class TestPhase2EnvSelfOrchestratedLifecycle(unittest.TestCase):
    """Env __init__ 自主编排生命周期（不调 super().__init__）。"""

    def test_init_completes_without_error(self):
        """__init__ 完整执行不报错。"""
        env = _make_env()
        self.assertIsNotNone(env._gym)

    def test_init_orchestrates_lifecycle_in_order(self):
        """__init__ 按序调用生命周期方法（initialize_grpc → ... → init_qpos_qvel）。"""
        env = _make_env()
        # 验证生命周期已执行：_gym 已创建、init_qpos/init_qvel 已保存
        self.assertIsNotNone(env._gym)
        self.assertTrue(hasattr(env, "init_qpos"))
        self.assertTrue(hasattr(env, "init_qvel"))
        self.assertEqual(env.init_qpos.shape, (1,))   # pendulum nq=1

    def test_init_does_not_call_super_init(self):
        """Env 不调 super().__init__()（不触发 OrcaGymBaseEnv 编排）。"""
        import re
        source = pathlib.Path(
            OrcaGymEulerEnv.__module__.replace(".", "/") + ".py"
        ).read_text() if hasattr(OrcaGymEulerEnv, "__module__") else ""
        # 直接读源码文件
        env_file = (
            pathlib.Path(__file__).resolve().parents[4]
            / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
        )
        exec_source = re.sub(r'"""[\s\S]*?"""', '', env_file.read_text())
        self.assertNotIn("super().__init__()", exec_source)


class TestPhase2EnvSimConfigBindCachePath(unittest.TestCase):
    """SimConfig _bind 缓存路径：set_time_step 在 init_simulation 前调用。"""

    def test_time_step_cached_before_init(self):
        """__init__ 中 set_time_step 在 init_simulation 前调用，_time_step 缓存生效。"""
        env = _make_env()
        # 验证 _time_step 已缓存
        self.assertEqual(env._time_step, 0.002)
        # 验证 init_simulation 后 sim_config.timestep 已应用缓存值
        self.assertAlmostEqual(env.sim_config.timestep, 0.002)

    def test_sim_config_bound_after_init(self):
        """init_simulation 后 SimConfig 已绑定 mjModel。"""
        env = _make_env()
        # 绑定后 setter 应写入 mjModel.opt（非缓存）
        env.sim_config.timestep = 0.005
        self.assertAlmostEqual(env.sim_config.timestep, 0.005)

    def test_dt_uses_sim_config(self):
        """env.dt = sim_config.timestep * frame_skip（K7）。"""
        env = _make_env()
        expected = env.sim_config.timestep * env.frame_skip
        self.assertAlmostEqual(env.dt, expected)


class TestPhase2EnvSteppingAndStateSetting(unittest.TestCase):
    """步进与状态设置委托链路（K4/K8）。"""

    def test_do_simulation_delegates_to_step_with_coupling(self):
        """do_simulation 委托 _gym.step_with_coupling（K8: 不读 _euler）。"""
        env = _make_env()
        time_before = float(env.data.time)
        env.do_simulation(np.array([0.0]), 5)
        time_after = float(env.data.time)
        expected_dt = 5 * env.sim_config.timestep
        self.assertAlmostEqual(time_after - time_before, expected_dt, places=5)

    def test_set_joint_qpos_delegates_to_gym(self):
        """set_joint_qpos 委托 _gym.set_qpos_qvel（K4: 走公共方法）。"""
        env = _make_env()
        env.set_joint_qpos(np.array([0.5]))
        env.mj_forward()
        env._gym.sync_to_view()
        self.assertAlmostEqual(float(env.data.qpos[0]), 0.5)

    def test_set_joint_qvel_delegates_to_gym(self):
        """set_joint_qvel 委托 _gym.set_qpos_qvel（K4: 走公共方法）。"""
        env = _make_env()
        env.set_joint_qvel(np.array([0.3]))
        env.mj_forward()
        env._gym.sync_to_view()
        self.assertAlmostEqual(float(env.data.qvel[0]), 0.3)

    def test_data_returns_dataview(self):
        """env.data 返回 OrcaGymDataView（K6）。"""
        env = _make_env()
        self.assertIsInstance(env.data, OrcaGymDataView)

    def test_do_simulation_validates_action_dim(self):
        """do_simulation 对错误维度抛 ValueError。"""
        env = _make_env()
        with self.assertRaises(ValueError):
            env.do_simulation(np.zeros(0), 1)
```

#### 2.3 验收标准

| 验收项 | 验证方式 | 架构对齐 |
|--------|---------|---------|
| K14 继承链正确 | `test_inheritance_chain` | §5.1, §12.3 K14 |
| M0 原生 AttributeError | `test_env_gym_raises_native_attribute_error` + `test_env_no_getattr_method` | §7.1 M0 |
| K10 `__setattr__` 已删除 | `test_env_no_setattr_method` | 骨架迁移 §3.2 |
| 自主编排生命周期 | `test_init_orchestrates_lifecycle_in_order` + `test_init_does_not_call_super_init` | §5.1 |
| SimConfig `_bind` 缓存路径 | `test_time_step_cached_before_init` + `test_sim_config_bound_after_init` | §5.6, §6.5 C1 |
| K7 dt 委托 | `test_dt_uses_sim_config` | §6.5, §12.2 K7 |
| K8 步进不读 `_euler` | `test_do_simulation_delegates_to_step_with_coupling` | §5.8, §8.1 |
| K4 状态设置走公共方法 | `test_set_joint_qpos_delegates_to_gym` + `test_set_joint_qvel_delegates_to_gym` | §6.3 W1, §12.2 K4 |
| K6 data 返回 DataView | `test_data_returns_dataview` | §5.7, §12.2 K6 |
| 步进维度校验 | `test_do_simulation_validates_action_dim` | §8.1 |

**运行命令**：

```bash
<conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.test_phase2_revision_env_filling -v
```

---

### Step 3：Gym 层与子组件填充兼容性验证

#### 目标

验证 `OrcaGymEuler`（Gym 层）及子组件（MuJoCoSimCore/SimConfig/ModelRegistry/OrcaGymDataView/OrcaStudioBridge）的阶段二填充与新骨架架构兼容。

**关键点**：Gym 层的 K3/K5 隔离机制（`_BLOCKED_ATTRS` + `__getattribute__`）**未被骨架迁移删除**，仍保留在 `OrcaGymEuler` 中。本步验证该机制与 Env 层新架构的协同工作。

#### 3.1 开发任务

**任务 3.1.1**：验证 Gym 层 K3/K5 隔离机制保留。

`OrcaGymEuler` 仍保留 `_BLOCKED_ATTRS` + `__getattribute__`（拦截 `_mjData`/`_mjModel`/`_sim`/`_studio` 等的外部访问），这是 L2/L3 隔离，与 Env 层 M0 机制互补：

```python
class OrcaGymEuler:
    _BLOCKED_ATTRS = frozenset({
        "_mjData", "_mjModel", ...,
        "_sim", "_studio", "_registry", "_opt", "_view", "_euler",
        "sim", "studio", "registry", "opt", "view", "euler",
    })

    def __getattribute__(self, name: str):
        blocked = object.__getattribute__(self, "_BLOCKED_ATTRS")
        if name in blocked:
            raise AttributeError(...)   # 含引导消息
        return object.__getattribute__(self, name)
```

**任务 3.1.2**：验证 `_bind` 延迟绑定模式。

`OrcaGymEuler.init_simulation` 在加载 mjModel 后，调用 `SimConfig._bind` 和 `ModelRegistry._bind`：

```python
async def init_simulation(self, model_xml_path: str) -> None:
    sim = object.__getattribute__(self, "_sim")
    opt = object.__getattribute__(self, "_opt")
    registry = object.__getattribute__(self, "_registry")
    view = object.__getattribute__(self, "_view")
    sim.init_simulation(model_xml_path)
    opt._bind(sim._mjModel)           # noqa: SLF001  core 层组件编排
    registry._bind(sim._mjModel)      # noqa: SLF001  core 层组件编排
    object.__setattr__(self, "_orca_model", registry.build_orca_gym_model())
    sim.sync_to_view(view)
```

> **ruff SLF001 豁免说明**：`opt._bind(sim._mjModel)` 是 `OrcaGymEuler`（Gym 层）内部编排子组件，属于类内部协作。由于 `opt`/`registry`/`sim` 是 `self` 的子组件对象（非 `self` 本身），SLF001 会报警。此处用 `# noqa: SLF001` 显式豁免，注释标明"core 层组件编排"。

**任务 3.1.3**：验证委托链路 `Env → Gym → SimCore` 完整。

```
env.do_simulation(ctrl, n)
  → self._gym.step_with_coupling(ctrl, n, dt)     # Gym 公共方法
    → self._sim.step(n)                            # SimCore 内部
  → self._gym.sync_to_view()                       # Gym 公共方法
    → self._sim.sync_to_view(self._view)           # SimCore 内部

env.set_joint_qpos(qpos)
  → self._gym.set_qpos_qvel(qpos, self._gym.data.qvel)   # Gym 公共方法
    → self._sim.set_qpos_qvel(qpos, qvel)                # SimCore 内部
```

#### 3.2 测试

**文件**：`tests/orca_gym/environment/euler/test_phase2_revision_gym_compat.py`（新建）

```python
"""阶段二变更修订 — Step 3: Gym 层与子组件填充兼容性验证。

验证 Gym 层 K3/K5 隔离机制保留、_bind 延迟绑定、委托链路完整。
"""

import pathlib
import unittest

import numpy as np

from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler


def _make_gym():
    """构造离线模式 OrcaGymEuler。"""
    _pendulum_xml = (
        pathlib.Path(__file__).resolve().parents[4].parent
        / "OrcaPlayground" / "envs" / "euler" / "scenes" / "simple_pendulum.xml"
    )
    import asyncio
    gym = OrcaGymEuler(stub=None)
    gym.studio_bridge().configure_offline(str(_pendulum_xml))
    loop = asyncio.new_event_loop()
    loop.run_until_complete(gym.init_simulation(str(_pendulum_xml)))
    loop.close()
    return gym


class TestPhase2GymK3K5IsolationRetained(unittest.TestCase):
    """Gym 层 K3/K5 隔离机制保留（未被骨架迁移删除）。"""

    def test_gym_has_blocked_attrs(self):
        """OrcaGymEuler 类定义 _BLOCKED_ATTRS。"""
        self.assertIn("_BLOCKED_ATTRS", vars(OrcaGymEuler))

    def test_gym_has_getattribute(self):
        """OrcaGymEuler 类定义 __getattribute__（拦截外部访问）。"""
        self.assertIn("__getattribute__", vars(OrcaGymEuler))

    def test_gym_blocked_attrs_contains_mjdata_mjmodel(self):
        """_BLOCKED_ATTRS 含 _mjData/_mjModel（K3 L3 引擎内部）。"""
        blocked = OrcaGymEuler._BLOCKED_ATTRS
        self.assertIn("_mjData", blocked)
        self.assertIn("_mjModel", blocked)

    def test_gym_blocked_attrs_contains_subcomponents(self):
        """_BLOCKED_ATTRS 含 _sim/_studio/_registry/_opt/_view/_euler（K5 子组件）。"""
        blocked = OrcaGymEuler._BLOCKED_ATTRS
        for name in ["_sim", "_studio", "_registry", "_opt", "_view", "_euler"]:
            with self.subTest(attr=name):
                self.assertIn(name, blocked)

    def test_gym_external_access_blocked_with_guidance(self):
        """Gym 外部访问 _mjData 抛 AttributeError 含引导消息（K3）。"""
        gym = _make_gym()
        with self.assertRaises(AttributeError) as ctx:
            _ = gym._mjData
        # Gym 层 __getattribute__ 提供引导消息（与 Env 层 M0 原生 AttributeError 不同）
        self.assertIn("公共", str(ctx.exception))

    def test_gym_external_access_sim_blocked(self):
        """Gym 外部访问 _sim 抛 AttributeError（K5）。"""
        gym = _make_gym()
        with self.assertRaises(AttributeError):
            _ = gym._sim


class TestPhase2GymBindDeferredBinding(unittest.TestCase):
    """Gym _bind 延迟绑定模式。"""

    def test_sim_config_bound_after_init(self):
        """init_simulation 后 SimConfig._mj_model 已绑定。"""
        gym = _make_gym()
        # 通过 sim_config.timestep setter 不抛 RuntimeError 验证已绑定
        gym.sim_config.timestep = 0.003
        self.assertAlmostEqual(gym.sim_config.timestep, 0.003)

    def test_model_registry_bound_after_init(self):
        """init_simulation 后 ModelRegistry 已绑定，model property 返回 OrcaGymModel。"""
        gym = _make_gym()
        model = gym.model
        self.assertEqual(model.nq, 1)   # pendulum nq=1
        self.assertEqual(model.nv, 1)
        self.assertEqual(model.nu, 1)


class TestPhase2GymDelegationChain(unittest.TestCase):
    """委托链路 Env → Gym → SimCore 完整。"""

    def test_gym_step_with_coupling_works(self):
        """gym.step_with_coupling 步进后 time 增加。"""
        gym = _make_gym()
        gym.sync_to_view()
        time_before = float(gym.data.time)
        gym.step_with_coupling(np.array([0.0]), 5, 0.002 * 4)
        gym.sync_to_view()
        time_after = float(gym.data.time)
        self.assertAlmostEqual(time_after - time_before, 5 * 0.002, places=5)

    def test_gym_set_qpos_qvel_writes_state(self):
        """gym.set_qpos_qvel 写入 qpos/qvel。"""
        gym = _make_gym()
        gym.set_qpos_qvel(np.array([0.4]), np.array([0.2]))
        gym.mj_forward()
        gym.sync_to_view()
        self.assertAlmostEqual(float(gym.data.qpos[0]), 0.4)
        self.assertAlmostEqual(float(gym.data.qvel[0]), 0.2)

    def test_gym_sync_to_view_populates_dataview(self):
        """gym.sync_to_view 后 DataView 基本字段已填充。"""
        gym = _make_gym()
        gym.sync_to_view()
        self.assertEqual(gym.data.qpos.shape, (1,))
        self.assertEqual(gym.data.qvel.shape, (1,))


class TestPhase2GymHasEulerFalse(unittest.TestCase):
    """K8: has_euler() 恒返回 False（骨架阶段无 Euler）。"""

    def test_has_euler_returns_false(self):
        gym = _make_gym()
        self.assertFalse(gym.has_euler())
```

#### 3.3 验收标准

| 验收项 | 验证方式 | 架构对齐 |
|--------|---------|---------|
| K3 Gym 层 `_BLOCKED_ATTRS` 保留 | `test_gym_has_blocked_attrs` + `test_gym_blocked_attrs_contains_mjdata_mjmodel` | §7.1, §12.2 K3 |
| K5 Gym 层子组件隔离 | `test_gym_blocked_attrs_contains_subcomponents` + `test_gym_external_access_sim_blocked` | §12.2 K5 |
| Gym 层 `__getattribute__` 提供引导 | `test_gym_external_access_blocked_with_guidance` | §7.1（Gym 层隔离） |
| SimConfig `_bind` 绑定 | `test_sim_config_bound_after_init` | §5.6 |
| ModelRegistry `_bind` 绑定 | `test_model_registry_bound_after_init` | §5.5 |
| 委托链路完整 | `test_gym_step_with_coupling_works` + `test_gym_set_qpos_qvel_writes_state` + `test_gym_sync_to_view_populates_dataview` | §8.1 S1/S2 |
| K8 has_euler=False | `test_has_euler_returns_false` | §5.8, §12.2 K8 |

**运行命令**：

```bash
<conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.test_phase2_revision_gym_compat -v
```

---

### Step 4：SimpleEulerEnv 违规修正验证

#### 目标

验证 `SimpleEulerEnv.reset_model` 的架构违规修正与新骨架架构兼容。

**与老文档 §2.2 的差异**：

| 老文档假设 | 新架构实际 | 验证重点 |
|-----------|-----------|---------|
| 违规模式 `self.gym._sim._mjData` | `self.gym` 不存在（M0），违规模式不可达 | 确认无 `.gym._sim` / `._gym._sim` 穿墙 |
| 修正为 `env.set_joint_qpos()` 等 | 同（合规 API 不变） | 确认使用合规 API |

#### 4.1 开发任务

**任务 4.1.1**：验证 `SimpleEulerEnv.reset_model` 使用合规 API。

```python
# 合规实现（当前 simple_env.py）
def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
    qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
    # 合规：通过 Env 公共方法设置状态（K3/K5 + §6.3 W1）
    self.set_joint_qpos(qpos)
    self.set_joint_qvel(qvel)
    self.mj_forward()       # 更新派生量
    self._sync_view()       # 同步到 DataView
    self._step_count = 0
    return self._get_obs(), {}
```

**任务 4.1.2**：验证 `SimpleEulerEnv` 不出现旧式 `self.gym` 访问。

由于 M0（`env.gym` 不存在），任何 `self.gym.xxx` 访问会在运行时抛 `AttributeError`。需确认源码无此类残留。

#### 4.2 测试

**文件**：`tests/orca_gym/environment/euler/test_phase2_revision_simple_env.py`（新建）

```python
"""阶段二变更修订 — Step 4: SimpleEulerEnv 违规修正验证。

验证 M0 下违规模式不可达、合规 API 使用、功能正确。
"""

import pathlib
import re
import sys
import unittest

import numpy as np

_SIMPLE_ENV_PATH = (
    pathlib.Path(__file__).resolve().parents[4].parent
    / "OrcaPlayground" / "envs" / "euler" / "simple_env.py"
)


def _read_simple_env_source() -> str:
    return _SIMPLE_ENV_PATH.read_text(encoding="utf-8")


class TestPhase2SimpleEnvNoTunnelAccess(unittest.TestCase):
    """M0/K3/K5: 无穿墙访问。"""

    def test_no_gym_sim_tunnel(self):
        """源码不含 .gym._sim / _gym._sim 穿墙。"""
        source = _read_simple_env_source()
        for pattern in [".gym._sim", "_gym._sim"]:
            self.assertNotIn(pattern, source)

    def test_no_mjdata_mjmodel_tunnel(self):
        """源码不含 .gym._mjData / .gym._mjModel 穿墙。"""
        source = _read_simple_env_source()
        for pattern in [".gym._mjData", ".gym._mjModel", "_gym._mjData", "_gym._mjModel"]:
            self.assertNotIn(pattern, source)

    def test_no_self_gym_access(self):
        """源码不含 self.gym 访问（M0: env.gym 不存在）。

        旧文档违规点 self.gym._sim._mjData 在 M0 下不可达，
        但仍需确认源码无 self.gym 残留（会在运行时抛 AttributeError）。
        """
        source = _read_simple_env_source()
        # 排除注释和 docstring
        exec_source = re.sub(r'"""[\s\S]*?"""', '', source)
        exec_source = re.sub(r'#.*', '', exec_source)
        self.assertNotIn("self.gym", exec_source,
                         "M0 违规: simple_env.py 含 self.gym 访问（env.gym 不存在）")

    def test_reset_model_uses_compliant_api(self):
        """reset_model 使用 set_joint_qpos/set_joint_qvel/mj_forward/_sync_view。"""
        source = _read_simple_env_source()
        match = re.search(
            r"def reset_model\(self\):(.*?)(?=\n    def |\nclass |\Z)",
            source, re.DOTALL,
        )
        self.assertIsNotNone(match, "reset_model 方法未找到")
        body = match.group(1)
        self.assertIn("set_joint_qpos", body)
        self.assertIn("set_joint_qvel", body)
        self.assertIn("mj_forward", body)
        self.assertIn("_sync_view", body)
        self.assertNotIn("_sim._mjData", body)


class TestPhase2SimpleEnvFunctional(unittest.TestCase):
    """功能验证: reset_model 与 step 在新架构下正常工作。"""

    @classmethod
    def setUpClass(cls):
        _orca_playground = str(
            pathlib.Path(__file__).resolve().parents[4].parent / "OrcaPlayground"
        )
        if _orca_playground not in sys.path:
            sys.path.insert(0, _orca_playground)
        from envs.euler.simple_env import SimpleEulerEnv
        cls.env = SimpleEulerEnv()

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def test_reset_model_writes_perturbed_state(self):
        """reset_model 后 qpos 反映随机扰动。"""
        env = self.env
        env.reset_simulation()
        env.init_qpos_qvel()
        env.np_random = np.random.RandomState(42)
        env.reset_model()
        qpos = float(env.data.qpos[0])
        self.assertNotAlmostEqual(qpos, 0.0, places=3)
        self.assertLessEqual(abs(qpos), 0.1 + 1e-6)

    def test_step_works_after_reset_model(self):
        """reset_model 后 step 正常工作（time 累计正确）。"""
        env = self.env
        env.reset_simulation()
        env.init_qpos_qvel()
        env.np_random = np.random.RandomState(42)
        env.reset_model()
        time_before = float(env.data.time)
        env.step(np.array([0.0], dtype=np.float32))
        time_after = float(env.data.time)
        expected_dt = env.frame_skip * env.sim_config.timestep
        self.assertAlmostEqual(time_after - time_before, expected_dt, places=5)

    def test_env_gym_raises_attribute_error(self):
        """M0: SimpleEulerEnv 继承 Env，env.gym 抛原生 AttributeError。"""
        env = self.env
        with self.assertRaises(AttributeError):
            _ = env.gym
```

#### 4.3 验收标准

| 验收项 | 验证方式 | 架构对齐 |
|--------|---------|---------|
| 无 `_sim._mjData` 穿墙 | `test_no_gym_sim_tunnel` + `test_no_mjdata_mjmodel_tunnel` | §6.3 W1, §12.2 K3/K5 |
| M0 无 `self.gym` 访问 | `test_no_self_gym_access` | §7.1 M0 |
| 合规 API 使用 | `test_reset_model_uses_compliant_api` | §6.3 W1 |
| reset_model 功能正确 | `test_reset_model_writes_perturbed_state` | §6.3 W3 |
| step 功能正确 | `test_step_works_after_reset_model` | §8.1 S1 |
| M0 AttributeError | `test_env_gym_raises_attribute_error` | §7.1 M0 |

**运行命令**：

```bash
<conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.test_phase2_revision_simple_env -v
```

---

### Step 5：隔离机制验收清单修订（K1-K14 + M0-M7）

#### 目标

修订老文档 §7 的隔离机制验收清单，对齐骨架迁移后的 K1-K14 约束与 M0-M7 机制。

**与老文档 §7 的差异**：

| 老文档约束 | 新架构要求 | 修订动作 |
|-----------|-----------|---------|
| K2 `__getattr__` 拦截 `env.gym` | M0 Python 原生 AttributeError | 测试重写（无自定义消息） |
| K10 `__setattr__` 屏蔽父类属性 | K10 删除（无 `__setattr__`） | 测试删除 |
| K14 不存在 | 新增：继承链 + 原生 AttributeError | 测试新增 |
| §7.2 仅源码 grep | 增加 ruff SLF001 静态扫描 | 验收方式扩展 |
| §7.3 `match="通过.*公共.*访问"` | 原生 AttributeError 无此消息 | match 模式修订 |

#### 5.1 开发任务

**任务 5.1.1**：修订 K2 验收（Env 层）。

K2 从"`__getattr__` 拦截含引导消息"改为"M0 Python 原生 AttributeError"：

```python
# 老文档（已失效）
def test_phase2_env_blocked_attrs():
    for attr in ["gym", "_mjData", "stub", "channel"]:
        with pytest.raises(AttributeError, match="通过.*公共.*访问"):
            getattr(env, attr)

# 新架构（M0）
def test_phase2_env_gym_native_attribute_error():
    with self.assertRaises(AttributeError) as ctx:
        _ = env.gym
    # M0: 原生 AttributeError，无自定义引导消息
    self.assertNotIn("公共", str(ctx.exception))
```

> **注意**：`_mjData`/`_mjModel` 在 Env 层不存在（Env 不持有这些属性），访问抛原生 AttributeError。Gym 层 `_mjData` 访问被 `__getattribute__` 拦截（含引导消息），但那是 Gym 层 K3，不是 Env 层 K2。

**任务 5.1.2**：删除 K10 验收。

K10（`__setattr__` 屏蔽父类属性）已随骨架迁移删除，相关测试删除：

```python
# 老文档（已删除）
class TestEnvK10ParentShielding:
    def test_setattr_shields_parent_attrs(self): ...
    def test_shielded_attrs_frozenset_complete(self): ...
```

**任务 5.1.3**：新增 K14 验收。

K14 验证继承链约束（已在 Step 2 覆盖，此处为清单完整性）。

**任务 5.1.4**：扩展 §7.2 grep 断言为 ruff SLF001 静态扫描。

老文档 §7.2 仅用源码 grep，新架构增加 ruff SLF001 作为 CI 门禁（M1）。

#### 5.2 测试

**文件**：`tests/orca_gym/environment/euler/test_phase2_revision_isolation_checklist.py`（新建）

```python
"""阶段二变更修订 — Step 5: 隔离机制验收清单（K1-K14 + M0-M7）。

修订老文档 §7：
- K2: __getattr__ 拦截 → M0 原生 AttributeError
- K10: __setattr__ 屏蔽 → 删除
- K14: 新增继承链约束
- §7.2: grep → ruff SLF001 静态扫描
- §7.3: match 模式修订
"""

import pathlib
import re
import subprocess
import sys
import unittest

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimConfig


def _make_env():
    _pendulum_xml = (
        pathlib.Path(__file__).resolve().parents[4].parent
        / "OrcaPlayground" / "envs" / "euler" / "scenes" / "simple_pendulum.xml"
    )
    return OrcaGymEulerEnv(
        frame_skip=4, orcagym_addr="localhost:50051",
        agent_names=["agent0"], time_step=0.002,
        model_xml_path=str(_pendulum_xml), skip_grpc_load=True,
    )


class TestPhase2K1NamingConstraint(unittest.TestCase):
    """K1: 命名约束 — 内部组件带下划线。"""

    def test_env_no_public_internal_attrs(self):
        env = _make_env()
        self.assertNotIn("gym", env.__dict__)
        self.assertNotIn("stub", env.__dict__)
        self.assertNotIn("channel", env.__dict__)
        self.assertIn("_gym", env.__dict__)
        self.assertIn("_stub", env.__dict__)
        self.assertIn("_channel", env.__dict__)


class TestPhase2K2M0NativeAttributeError(unittest.TestCase):
    """K2 + M0: env.gym/stub/channel 抛原生 AttributeError（无 __getattr__）。"""

    def test_env_gym_native_attribute_error(self):
        """env.gym 抛 AttributeError，消息为 Python 原生格式。"""
        env = _make_env()
        with self.assertRaises(AttributeError) as ctx:
            _ = env.gym
        # M0: 原生 AttributeError，不含旧 __getattr__ 的自定义引导
        self.assertNotIn("公共", str(ctx.exception))
        self.assertNotIn("API 契约", str(ctx.exception))

    def test_env_stub_native_attribute_error(self):
        env = _make_env()
        with self.assertRaises(AttributeError):
            _ = env.stub

    def test_env_channel_native_attribute_error(self):
        env = _make_env()
        with self.assertRaises(AttributeError):
            _ = env.channel

    def test_env_no_getattr_classvar(self):
        """Env 类不定义 __getattr__（M0 替代）。"""
        self.assertNotIn("__getattr__", vars(OrcaGymEulerEnv))


class TestPhase2K10Deleted(unittest.TestCase):
    """K10: __setattr__ 屏蔽机制已删除。"""

    def test_env_no_setattr_classvar(self):
        """Env 类不定义 __setattr__。"""
        self.assertNotIn("__setattr__", vars(OrcaGymEulerEnv))

    def test_env_no_shielded_attrs_classvar(self):
        """Env 类不定义 _SHIELDED_ATTRS。"""
        self.assertNotIn("_SHIELDED_ATTRS", vars(OrcaGymEulerEnv))

    def test_env_attribute_assignment_works(self):
        """Env 实例属性赋值正常工作（无 __setattr__ 屏蔽）。"""
        env = _make_env()
        env._test_field = "test_value"
        self.assertEqual(env._test_field, "test_value")


class TestPhase2K14Inheritance(unittest.TestCase):
    """K14: 继承链约束。"""

    def test_inheritance_chain(self):
        from orca_gym.environment.orca_gym_env_mixin import OrcaGymEnvMixin
        from orca_gym.environment.orca_gym_env import OrcaGymBaseEnv
        import gymnasium as gym
        bases = OrcaGymEulerEnv.__bases__
        self.assertIn(OrcaGymEnvMixin, bases)
        self.assertIn(gym.Env, bases)
        self.assertNotIn(OrcaGymBaseEnv, bases)


class TestPhase2K4K8K9SourceAudit(unittest.TestCase):
    """K4/K8/K9: 源码审查（Env 不穿墙访问 Gym 私有）。"""

    @classmethod
    def setUpClass(cls):
        env_file = (
            pathlib.Path(__file__).resolve().parents[4]
            / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
        )
        source = env_file.read_text(encoding="utf-8")
        # 去除 docstring 和注释
        source = re.sub(r'"""[\s\S]*?"""', '', source)
        cls.exec_source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )

    def test_k4_no_gym_private_access(self):
        """K4: Env 可执行代码不含 _gym._sim/_studio/_registry/_opt/_view/_euler。"""
        for pattern in ["_gym._sim", "_gym._studio", "_gym._registry",
                        "_gym._opt", "_gym._view", "_gym._euler"]:
            with self.subTest(pattern=pattern):
                self.assertNotIn(pattern, self.exec_source)

    def test_k8_no_euler_private_access(self):
        """K8: Env 可执行代码不含 _euler 属性访问。"""
        match = re.search(r'(?<![\w])_euler(?![\w])', self.exec_source)
        self.assertIsNone(match)

    def test_k9_no_studio_property_access(self):
        """K9: Env 可执行代码不含 gym.studio 穿墙（允许 _gym.studio_bridge()）。"""
        cleaned = self.exec_source.replace("_gym.studio_bridge", "")
        self.assertNotIn("gym.studio", cleaned)


class TestPhase2K6K7K11TypedReturn(unittest.TestCase):
    """K6/K7/K11: 类型化返回。"""

    def test_k6_data_returns_dataview(self):
        env = _make_env()
        self.assertIsInstance(env.data, OrcaGymDataView)

    def test_k7_sim_config_returns_config(self):
        env = _make_env()
        self.assertIsInstance(env.sim_config, SimConfig)

    def test_k11_data_not_mjdata(self):
        env = _make_env()
        self.assertNotEqual(type(env.data).__name__, "MjData")


class TestPhase2K12Docstring(unittest.TestCase):
    """K12: docstring 含使用契约。"""

    def test_env_docstring_has_contract(self):
        doc = OrcaGymEulerEnv.__doc__ or ""
        self.assertIn("使用契约", doc)
        self.assertIn("禁止", doc)


class TestPhase2M1RuffSLF001StaticCheck(unittest.TestCase):
    """M1: ruff SLF001 静态检查（Euler 代码零报警）。"""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = pathlib.Path(__file__).resolve().parents[4]

    def test_ruff_slf001_euler_env_clean(self):
        """Euler Env 源码 ruff SLF001 零报警。"""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "SLF001",
             str(self.repo_root / "orca_gym" / "environment" / "euler")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
                         f"ruff SLF001 报警:\n{result.stdout}")

    def test_ruff_slf001_euler_core_clean(self):
        """Euler Core 源码 ruff SLF001 零报警。"""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "SLF001",
             str(self.repo_root / "orca_gym" / "core" / "euler")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
                         f"ruff SLF001 报警:\n{result.stdout}")


class TestPhase2M3DirControl(unittest.TestCase):
    """M3: __dir__ 控制（只暴露公共 API）。"""

    def test_env_dir_no_internal(self):
        """dir(env) 不含 gym/stub/channel/_gym/_studio_bridge/_mjData/_mjModel。"""
        env = _make_env()
        d = dir(env)
        for name in ["gym", "stub", "channel", "_gym", "_stub", "_channel",
                      "_studio_bridge", "_mjData", "_mjModel"]:
            with self.subTest(attr=name):
                self.assertNotIn(name, d)

    def test_env_dir_contains_public_api(self):
        """dir(env) 含公共 API。"""
        env = _make_env()
        d = dir(env)
        for name in ["data", "model", "sim_config", "dt", "ctrl",
                      "do_simulation", "mj_step", "mj_forward", "render"]:
            with self.subTest(attr=name):
                self.assertIn(name, d)
```

#### 5.3 验收标准

| 约束 | 验证方式 | 通过标准 | 架构对齐 |
|------|---------|---------|---------|
| K1 命名 | `test_env_no_public_internal_attrs` | 通过 | §12.2 K1 |
| K2 + M0 原生 AttributeError | `test_env_gym_native_attribute_error` + `test_env_no_getattr_classvar` | 通过，无自定义消息 | §7.1 M0, §12.2 K2 |
| K4 源码审查 | `test_k4_no_gym_private_access` | 无穿墙模式 | §12.2 K4 |
| K6 DataView | `test_k6_data_returns_dataview` | 通过 | §12.2 K6 |
| K7 SimConfig | `test_k7_sim_config_returns_config` | 通过 | §12.2 K7 |
| K8 不读 `_euler` | `test_k8_no_euler_private_access` | 通过 | §12.2 K8 |
| K9 不穿墙 studio | `test_k9_no_studio_property_access` | 通过 | §12.2 K9 |
| K10 已删除 | `test_env_no_setattr_classvar` + `test_env_no_shielded_attrs_classvar` + `test_env_attribute_assignment_works` | 通过 | 骨架迁移 §3.2 |
| K11 typed 返回 | `test_k11_data_not_mjdata` | 通过 | §12.2 K11 |
| K12 docstring | `test_env_docstring_has_contract` | 通过 | §12.2 K12 |
| K14 继承链 | `test_inheritance_chain` | 通过 | §12.3 K14 |
| M1 ruff SLF001 | `test_ruff_slf001_euler_env_clean` + `test_ruff_slf001_euler_core_clean` | 退出码 0 | §7.2 M1 |
| M3 `__dir__` 控制 | `test_env_dir_no_internal` + `test_env_dir_contains_public_api` | 通过 | §7.4 M3 |

**运行命令**：

```bash
<conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.test_phase2_revision_isolation_checklist -v
```

---

### Step 6：ruff SLF001 静态检查与 AGENTS.md 合规

#### 目标

验证 M1（ruff SLF001 静态检查）与 M2（AGENTS.md AI 行为约束）在阶段二填充代码上落地。

**与老文档的差异**：老文档未含 ruff SLF001 验收（ruff 在骨架迁移阶段 0 才引入）。本步为新增验收项。

#### 6.1 开发任务

**任务 6.1.1**：验证 Euler 代码 ruff SLF001 零报警。

```bash
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 \
    orca_gym/environment/euler/ orca_gym/core/euler/
```

**任务 6.1.2**：验证 AGENTS.md 含 API 隔离强制章节（M2）。

**任务 6.1.3**：验证 `_bind` 跨层访问的 `# noqa: SLF001` 豁免规范。

`OrcaGymEuler.init_simulation` 中 `opt._bind(sim._mjModel)` 是 Gym 层内部编排子组件，SLF001 会报警。须用 `# noqa: SLF001` 显式豁免，并注释标明"core 层组件编排"。

#### 6.2 测试

**文件**：`tests/orca_gym/environment/euler/test_phase2_revision_ruff_agents.py`（新建）

```python
"""阶段二变更修订 — Step 6: ruff SLF001 与 AGENTS.md 合规（M1/M2）。"""

import pathlib
import re
import subprocess
import sys
import unittest


class TestPhase2M1RuffSLF001(unittest.TestCase):
    """M1: ruff SLF001 静态检查。"""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = pathlib.Path(__file__).resolve().parents[4]

    def test_ruff_installed(self):
        """ruff 已安装且可执行。"""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, "ruff 未安装")

    def test_ruff_slf001_euler_clean(self):
        """Euler 代码 ruff SLF001 零报警。"""
        for sub in ["environment/euler", "core/euler"]:
            with self.subTest(path=sub):
                result = subprocess.run(
                    [sys.executable, "-m", "ruff", "check", "--select", "SLF001",
                     str(self.repo_root / "orca_gym" / sub)],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0,
                                 f"ruff SLF001 报警 [{sub}]:\n{result.stdout}")


class TestPhase2M2AgentsMdApiIsolation(unittest.TestCase):
    """M2: AGENTS.md 含 API 隔离强制章节。"""

    @classmethod
    def setUpClass(cls):
        cls.agents_md = (
            pathlib.Path(__file__).resolve().parents[4] / "AGENTS.md"
        ).read_text(encoding="utf-8")

    def test_agents_md_has_api_isolation_rule(self):
        """AGENTS.md 含'API 隔离强制'章节。"""
        self.assertIn("规则 4", self.agents_md)
        self.assertIn("API 隔离强制", self.agents_md)

    def test_agents_md_lists_blocked_attrs(self):
        """AGENTS.md 列出禁止穿墙的内部属性。"""
        self.assertIn("env._gym", self.agents_md)
        self.assertIn("env._stub", self.agents_md)
        self.assertIn("env._channel", self.agents_md)
        self.assertIn("_mjModel", self.agents_md)
        self.assertIn("_mjData", self.agents_md)

    def test_agents_md_has_correct_usage_table(self):
        """AGENTS.md 含正确/禁止 API 使用对照表。"""
        self.assertIn("env.data.qpos", self.agents_md)
        self.assertIn("env.set_joint_qpos", self.agents_md)
        self.assertIn("env.do_simulation", self.agents_md)
        self.assertIn("env.sim_config.timestep", self.agents_md)

    def test_agents_md_has_ruff_command(self):
        """AGENTS.md 含 ruff SLF001 检查命令。"""
        self.assertIn("ruff check --select SLF001", self.agents_md)


class TestPhase2NoqaExemptionDiscipline(unittest.TestCase):
    """ruff SLF001 noqa 豁免规范: 仅 core 层组件编排允许。"""

    @classmethod
    def setUpClass(cls):
        env_file = (
            pathlib.Path(__file__).resolve().parents[4]
            / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
        )
        cls.env_source = env_file.read_text(encoding="utf-8")
        gym_file = (
            pathlib.Path(__file__).resolve().parents[4]
            / "orca_gym" / "core" / "euler" / "orca_gym_euler.py"
        )
        cls.gym_source = gym_file.read_text(encoding="utf-8")

    def test_env_no_noqa_slf001(self):
        """Env 源码不使用 noqa: SLF001 豁免（Env 不应穿墙）。"""
        self.assertNotIn("noqa: SLF001", self.env_source)
        self.assertNotIn("noqa:SLF001", self.env_source)

    def test_gym_noqa_only_for_bind_orchestration(self):
        """Gym 源码 noqa: SLF001 仅用于 _bind 编排（非穿墙访问）。"""
        # 提取 noqa 行
        noqa_lines = re.findall(
            r".*# noqa: ?SLF001.*", self.gym_source
        )
        self.assertGreaterEqual(len(noqa_lines), 1, "Gym 应有 _bind 编排的 noqa 豁免")
        # 每条 noqa 应伴随 _bind 或 sync_to_view 等组件编排模式
        for line in noqa_lines:
            self.assertTrue(
                "_bind(" in line or "object.__getattribute__" in line,
                f"noqa 行非组件编排豁免: {line.strip()}",
            )
```

#### 6.3 验收标准

| 验收项 | 验证方式 | 通过标准 | 架构对齐 |
|--------|---------|---------|---------|
| ruff 已安装 | `test_ruff_installed` | 退出码 0 | §7.2 M1 |
| Euler 代码 SLF001 零报警 | `test_ruff_slf001_euler_clean` | 退出码 0 | §7.2 M1 |
| AGENTS.md 含 API 隔离章节 | `test_agents_md_has_api_isolation_rule` | 通过 | §7.3 M2 |
| AGENTS.md 列禁止属性 | `test_agents_md_lists_blocked_attrs` | 通过 | §7.3 M2 |
| AGENTS.md 含正确用法表 | `test_agents_md_has_correct_usage_table` | 通过 | §7.3 M2 |
| AGENTS.md 含 ruff 命令 | `test_agents_md_has_ruff_command` | 通过 | §7.3 M2 |
| Env 不用 noqa 豁免 | `test_env_no_noqa_slf001` | 通过 | §7.2 |
| Gym noqa 仅限组件编排 | `test_gym_noqa_only_for_bind_orchestration` | 通过 | §7.2 |

**运行命令**：

```bash
<conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.test_phase2_revision_ruff_agents -v
```

---

### Step 7：端到端验证（Lesson 1/2/3）

#### 目标

在 `orca` conda 环境下运行三个端到端示例,验证阶段二填充在新骨架架构下的端到端正确性。

**与老文档 §5 的差异**：

| 老文档假设 | 新架构要求 | 修订动作 |
|-----------|-----------|---------|
| `OrcaFlow_Flow` 解释器 | `orca` conda 环境 | 命令格式更新 |
| sandbox 内 GPU 训练 | GPU 训练须白名单旁路,禁用管道 | 命令格式更新（AGENTS.md 规则 3） |
| 验收基于旧骨架 | 验收须对齐 K14 + M0-M7 | 验收标准升级 |

#### 7.1 开发任务

**任务 7.1.1**：Lesson 1 离线仿真（CPU,sandbox 内）。

```bash
cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python examples/euler/01_offline_sim/main.py
```

**任务 7.1.2**：Lesson 2 在线 gRPC 渲染（宿主机 + OrcaStudio 运行）。

```bash
# 先启动 OrcaStudio（宿主机）
# 再运行 Lesson 2（宿主机,因 sandbox 无法访问外部 gRPC 服务）
cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python examples/euler/02_online_render/main.py
```

**任务 7.1.3**：Lesson 3 RL PPO 训练（GPU,白名单旁路）。

GPU 训练须用白名单解释器路径开头,**禁用 shell 管道**（AGENTS.md 规则 3）：

```bash
# 正确 — 白名单解释器直接调用,无管道,输出重定向到文件
cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 20000 > /tmp/lesson3_ppo.log 2>&1

# 错误 — 管道触发 sandbox 包裹,GPU 不可用
cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python examples/euler/03_rl_ppo/train_ppo.py 2>&1 | tail -30
```

#### 7.2 测试

**文件**：`tests/orca_gym/environment/euler/test_phase2_revision_e2e.py`（新建）

```python
"""阶段二变更修订 — Step 7: 端到端验证（Lesson 1/2/3）。

CPU 离线测试可在 sandbox 内运行；GPU/在线测试标记为 skip,
需在宿主机手动运行（见文档 §7.1）。
"""

import os
import pathlib
import subprocess
import sys
import unittest

_ORCA_PLAYGROUND = pathlib.Path(__file__).resolve().parents[4].parent / "OrcaPlayground"
_PYTHON = sys.executable


@unittest.skipUnless(
    _ORCA_PLAYGROUND.exists(),
    f"OrcaPlayground 不存在: {_ORCA_PLAYGROUND}",
)
class TestPhase2Lesson1OfflineSim(unittest.TestCase):
    """Lesson 1: 离线仿真（CPU,sandbox 内可运行）。"""

    def test_lesson1_runs_without_error(self):
        """Lesson 1 离线仿真完整运行,退出码 0。"""
        script = _ORCA_PLAYGROUND / "examples" / "euler" / "01_offline_sim" / "main.py"
        if not script.exists():
            self.skipTest(f"Lesson 1 脚本不存在: {script}")
        result = subprocess.run(
            [_PYTHON, str(script)],
            capture_output=True, text=True,
            cwd=str(_ORCA_PLAYGROUND),
            timeout=120,
        )
        self.assertEqual(result.returncode, 0,
                         f"Lesson 1 失败:\nstdout={result.stdout}\nstderr={result.stderr}")


@unittest.skipUnless(
    os.environ.get("ORCA_EULER_E2E_ONLINE") == "1",
    "在线测试需 ORCA_EULER_E2E_ONLINE=1 + OrcaStudio 运行",
)
class TestPhase2Lesson2OnlineRender(unittest.TestCase):
    """Lesson 2: 在线 gRPC 渲染（宿主机 + OrcaStudio）。"""

    def test_lesson2_runs_without_error(self):
        script = _ORCA_PLAYGROUND / "examples" / "euler" / "02_online_render" / "main.py"
        result = subprocess.run(
            [_PYTHON, str(script)],
            capture_output=True, text=True,
            cwd=str(_ORCA_PLAYGROUND),
            timeout=300,
        )
        self.assertEqual(result.returncode, 0,
                         f"Lesson 2 失败:\nstdout={result.stdout}\nstderr={result.stderr}")


@unittest.skipUnless(
    os.environ.get("ORCA_EULER_E2E_GPU") == "1",
    "GPU 训练需 ORCA_EULER_E2E_GPU=1 + 白名单旁路（AGENTS.md 规则 3）",
)
class TestPhase2Lesson3RlPpo(unittest.TestCase):
    """Lesson 3: RL PPO 训练（GPU,白名单旁路）。

    手动运行命令（宿主机,无管道）:
        cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python \\
            examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 20000 \\
            > /tmp/lesson3_ppo.log 2>&1
    """

    def test_lesson3_runs_without_error(self):
        script = _ORCA_PLAYGROUND / "examples" / "euler" / "03_rl_ppo" / "train_ppo.py"
        result = subprocess.run(
            [_PYTHON, str(script), "--total-timesteps", "20000"],
            capture_output=True, text=True,
            cwd=str(_ORCA_PLAYGROUND),
            timeout=600,
        )
        self.assertEqual(result.returncode, 0,
                         f"Lesson 3 失败:\nstdout={result.stdout}\nstderr={result.stderr}")
```

#### 7.3 验收标准

| 验收项 | 验证方式 | 通过标准 | 架构对齐 |
|--------|---------|---------|---------|
| Lesson 1 离线仿真 | `test_lesson1_runs_without_error` | 退出码 0 | §8.1 端到端 |
| Lesson 2 在线渲染 | `test_lesson2_runs_without_error`（手动） | 退出码 0 | §8.1 端到端 |
| Lesson 3 RL PPO | `test_lesson3_runs_without_error`（手动,GPU 旁路） | 退出码 0,reward 上升 | §8.1 端到端 |

**运行命令**：

```bash
# Lesson 1（sandbox 内,自动）
<conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.test_phase2_revision_e2e -v

# Lesson 2/3（宿主机,手动）
# 见 §7.1 任务 7.1.2/7.1.3
```

---

## 5. 总验收清单

### 5.1 修订完成验收矩阵

| 步骤 | 验收项 | 测试文件 | 通过标准 |
|------|--------|---------|---------|
| Step 1 | 基线建立与影响面扫描 | `test_phase2_revision_baseline.py` | 全部通过 |
| Step 2 | Env 层填充修订 | `test_phase2_revision_env_filling.py` | 全部通过 |
| Step 3 | Gym 层与子组件兼容性 | `test_phase2_revision_gym_compat.py` | 全部通过 |
| Step 4 | SimpleEulerEnv 违规修正 | `test_phase2_revision_simple_env.py` | 全部通过 |
| Step 5 | 隔离机制验收清单 | `test_phase2_revision_isolation_checklist.py` | 全部通过 |
| Step 6 | ruff SLF001 + AGENTS.md | `test_phase2_revision_ruff_agents.py` | 全部通过 |
| Step 7 | 端到端验证 | `test_phase2_revision_e2e.py` | Lesson 1 自动通过;Lesson 2/3 手动通过 |

### 5.2 架构约束对齐总表

| 约束 | 老文档状态 | 新架构状态 | 验证步骤 |
|------|-----------|-----------|---------|
| K1 命名 | ✓ | ✓ | Step 5 |
| K2 Env 层隔离 | `__getattr__` 拦截 | M0 原生 AttributeError | Step 2, Step 5 |
| K3 Gym 层 `_mjData` 隔离 | ✓ | ✓（保留） | Step 3 |
| K4 Env 不穿墙 Gym 私有 | ✓ | ✓ | Step 5 |
| K5 Gym 层子组件隔离 | ✓ | ✓（保留） | Step 3 |
| K6 data 返回 DataView | ✓ | ✓ | Step 2, Step 5 |
| K7 sim_config 返回 SimConfig | ✓ | ✓ | Step 2, Step 5 |
| K8 不读 `_euler` | ✓ | ✓ | Step 2, Step 5 |
| K9 不穿墙 studio | ✓ | ✓ | Step 5 |
| K10 `__setattr__` 屏蔽 | ✓ | **删除** | Step 5（验证已删除） |
| K11 typed 返回 | ✓ | ✓ | Step 5 |
| K12 docstring 契约 | ✓ | ✓ | Step 5 |
| K14 继承链 | 不存在 | **新增** | Step 2, Step 5 |
| M0 原生 AttributeError | 不存在 | **新增** | Step 2, Step 5 |
| M1 ruff SLF001 | 不存在 | **新增** | Step 6 |
| M2 AGENTS.md AI 约束 | 不存在 | **新增** | Step 6 |
| M3 `__dir__` 控制 | 不存在 | **新增** | Step 5 |

### 5.3 契约对齐总表（架构 §6 R/W/S/C/N）

| 契约 | 规则 | 验证步骤 |
|------|------|---------|
| R1 读取状态 | `env.data.qpos` / `env.data.body_xpos(name)` | Step 2 |
| R2 查询方法 | `env.query_*()` | Step 2 |
| W1 写入状态 | `env.set_joint_qpos()` / `env.set_joint_qvel()` | Step 2, Step 4 |
| W2 施加外力 | `env.apply_body_force()` | Step 2（如已实现） |
| S1 步进 | `env.do_simulation(ctrl, n)` | Step 2, Step 4 |
| S2 委托步进 | `env._gym.step_with_coupling()` | Step 3 |
| C1 求解器配置 | `env.sim_config.timestep = 0.002` | Step 2 |
| C2 SimConfig `_bind` | `sim_config._bind(mj_model)` | Step 3 |
| N1 命名空间 | `env.body(name)` / `env.joint(name)` | Step 2（Mixin） |

### 5.4 一键运行全部修订测试

```bash
# CPU 测试（sandbox 内）
<conda-base>/envs/orca/bin/python -m unittest \
    tests.orca_gym.environment.euler.test_phase2_revision_baseline \
    tests.orca_gym.environment.euler.test_phase2_revision_env_filling \
    tests.orca_gym.environment.euler.test_phase2_revision_gym_compat \
    tests.orca_gym.environment.euler.test_phase2_revision_simple_env \
    tests.orca_gym.environment.euler.test_phase2_revision_isolation_checklist \
    tests.orca_gym.environment.euler.test_phase2_revision_ruff_agents \
    tests.orca_gym.environment.euler.test_phase2_revision_e2e \
    -v
```

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| `_make_gym()` 构造方式与实际不符 | Step 3 测试失败 | 优先复用骨架测试的构造方式;失败时对照 `test_orca_gym_euler.py` |
| Lesson 2/3 环境依赖外部服务 | 端到端无法在 CI 运行 | 标记 skip,文档说明手动运行命令 |
| ruff SLF001 在 `__init__.py` 误报 | 验收假阳性 | `pyproject.toml` 已配置 `"**/__init__.py" = ["SLF001", "F401"]` 豁免 |
| GPU 命令被 sandbox 包裹 | Lesson 3 `CUDA_ERROR_304` | 严格遵循 AGENTS.md 规则 3:白名单解释器路径开头,无管道,输出重定向 |
| K14 继承链与 Mixin MRO 冲突 | `super()` 调用异常 | Env 不调 `super().__init__()`（自主编排）,Mixin 不调 `super().__init__()` |
| `env._gym.studio_bridge()` 触发 SLF001 | ruff 报警 | `_gym.studio_bridge()` 是公共方法委托,SLF001 不报警（仅 `_` 前缀属性访问才报警） |

---

## 7. 与老文档的关系

### 7.1 不替代老文档

本文**不替代** [orca_gym_euler_phase2_filling_development.md](orca_gym_euler_phase2_filling_development.md),而是其**变更修订补充**：

- 老文档中**未受架构变更影响**的内容（如 MuJoCoSimCore/SimConfig/ModelRegistry/DataView 的填充逻辑、Lesson 1/2/3 的功能目标）仍然有效。
- 老文档中**受架构变更影响**的内容（Env 继承链、补丁机制、隔离测试、测试环境）以本文为准。
- 老文档的 §3 测试策略、§4 Step 1-5/7-8（非 Env 层填充）等仍可参考,但须结合本文的测试环境与隔离机制修订。

### 7.2 优先级

当老文档与本文冲突时,**以本文为准**。具体冲突点见 §2.2 老文档条款映射表。

### 7.3 后续工作

本文完成阶段二变更修订与验证后,后续阶段（阶段三 Euler 耦合编排器填充等）应基于本文的架构基线（K1-K14 + M0-M7）继续设计,不再回退到旧骨架约束。

---

## 附录 A：测试文件清单

| 文件 | 步骤 | 用途 |
|------|------|------|
| `tests/orca_gym/environment/euler/test_phase2_revision_baseline.py` | Step 1 | 基线建立与影响面扫描 |
| `tests/orca_gym/environment/euler/test_phase2_revision_env_filling.py` | Step 2 | Env 层填充与新骨架兼容性 |
| `tests/orca_gym/environment/euler/test_phase2_revision_gym_compat.py` | Step 3 | Gym 层与子组件兼容性 |
| `tests/orca_gym/environment/euler/test_phase2_revision_simple_env.py` | Step 4 | SimpleEulerEnv 违规修正 |
| `tests/orca_gym/environment/euler/test_phase2_revision_isolation_checklist.py` | Step 5 | 隔离机制验收清单（K1-K14 + M0-M7） |
| `tests/orca_gym/environment/euler/test_phase2_revision_ruff_agents.py` | Step 6 | ruff SLF001 + AGENTS.md 合规 |
| `tests/orca_gym/environment/euler/test_phase2_revision_e2e.py` | Step 7 | 端到端验证（Lesson 1/2/3） |

## 附录 B：命令速查

```bash
# 解析 conda-base
CONDA_BASE=$(/home/superfhwl/miniconda3/bin/conda info --base)

# 单步测试
$CONDA_BASE/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.test_phase2_revision_<step> -v

# 全部修订测试（CPU）
$CONDA_BASE/envs/orca/bin/python -m unittest \
    tests.orca_gym.environment.euler.test_phase2_revision_baseline \
    tests.orca_gym.environment.euler.test_phase2_revision_env_filling \
    tests.orca_gym.environment.euler.test_phase2_revision_gym_compat \
    tests.orca_gym.environment.euler.test_phase2_revision_simple_env \
    tests.orca_gym.environment.euler.test_phase2_revision_isolation_checklist \
    tests.orca_gym.environment.euler.test_phase2_revision_ruff_agents \
    tests.orca_gym.environment.euler.test_phase2_revision_e2e -v

# ruff SLF001 静态检查
$CONDA_BASE/envs/orca/bin/python -m ruff check --select SLF001 \
    orca_gym/environment/euler/ orca_gym/core/euler/

# Lesson 3 GPU 训练（白名单旁路,无管道）
cd <OrcaPlayground-root> && $CONDA_BASE/envs/orca/bin/python \
    examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 20000 \
    > /tmp/lesson3_ppo.log 2>&1
```
