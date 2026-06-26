# OrcaGym Euler 骨架变更实施方案：直接继承 gym.Env + OrcaGymEnvMixin

## 1. 文档定位

本文是 `OrcaGymEulerEnv` 骨架变更的实施指导，对应架构文档 §4.1/§5.1/§5.9/§11.1/§12 的最新规约。

**变更目标**：将 `OrcaGymEulerEnv` 的继承链从 `OrcaGymBaseEnv` 切换为 `OrcaGymEnvMixin, gym.Env`，删除 `_BLOCKED_ATTRS`/`__getattr__`/`__setattr__`/`_SHIELDED_ATTRS` 补丁机制，通过 `OrcaGymEnvMixin` 共享公共方法。

**上游约束**：架构文档 `docs/design/architecture/orca_gym_euler_architecture.md`（§5.9 OrcaGymEnvMixin 定义、§7.1 M0 Python 原生属性不存在、§12.3 K14 继承链约束）。

**前提条件**：阶段二功能填充已完成（生命周期、步进、状态设置、渲染方法均已真实实现），本方案不修改这些已填充的功能逻辑，仅变更继承结构和隔离机制。

---

## 2. 变更范围

### 2.1 新增文件

| 文件 | 内容 |
|------|------|
| `orca_gym/environment/orca_gym_env_mixin.py` | `OrcaGymEnvMixin` 类，迁移自 `OrcaGymBaseEnv` 的 10 个公共方法 |
| `tests/orca_gym/environment/test_orca_gym_env_mixin.py` | Mixin 单元测试 |

### 2.2 修改文件

| 文件 | 变更内容 |
|------|---------|
| `orca_gym/environment/euler/orca_gym_euler_env.py` | 继承链切换、删除补丁机制、重写 `__init__`、更新 `__dir__` |
| `tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py` | 删除 K2/K10 旧测试，新增 K14 测试 |

### 2.3 不修改的文件

| 文件 | 原因 |
|------|------|
| `orca_gym/environment/orca_gym_env.py`（`OrcaGymBaseEnv`） | 原有继承体系不动（架构 §4.3），Local 体系仍使用 |
| `orca_gym/core/euler/*.py` | Gym 层及子组件不变更 |
| `tests/orca_gym/environment/euler/test_simple_euler_env_compliance.py` | SimpleEulerEnv 合规测试不依赖继承链 |
| `tests/orca_gym/core/euler/*.py` | Gym 层测试不变更 |

---

## 3. 实施阶段

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

**运行三个 example**（若环境允许）：

```bash
# 01_hello_euler（离线模式）
<conda-base>/envs/orca/bin/python OrcaPlayground/examples/euler/01_hello_euler.py

# 03_rl_ppo（离线模式，Gymnasium 契约）
<conda-base>/envs/orca/bin/python OrcaPlayground/examples/euler/03_rl_ppo.py
```

**验收标准**：

| 验收项 | 验证方式 |
|--------|---------|
| 01_hello_euler 运行无异常 | 脚本退出码 0 |
| 03_rl_ppo 训练启动正常 | 脚本退出码 0（或 Ctrl+C 中断后无 AttributeError） |

> 注：02_online_render 需要 OrcaStudio gRPC 服务，若环境不可用可跳过。

---

## 4. 实施顺序与依赖

```
阶段 1: 新建 OrcaGymEnvMixin
   │   （无破坏性，现有测试不受影响）
   ▼
阶段 2: 切换 Env 继承链 + 删除补丁机制
   │   （核心变更，Env 测试需同步更新）
   ▼
阶段 3: 端到端验证
       （SimpleEulerEnv + Gym 层 + ruff + example）
```

**关键约束**：

- 阶段 1 和阶段 2 不可并行（阶段 2 依赖阶段 1 的 Mixin 文件）
- 阶段 2 内部的代码变更和测试更新必须同步提交（避免中间状态测试失败）
- 阶段 3 必须在阶段 2 全部通过后执行

---

## 5. 回滚策略

若阶段 2 或阶段 3 发现不可解决的问题：

1. **代码回滚**：`git revert` 阶段 2 的提交，恢复 `OrcaGymBaseEnv` 继承
2. **Mixin 保留**：阶段 1 的 `OrcaGymEnvMixin` 文件可保留（无破坏性），待问题解决后重新尝试阶段 2
3. **测试回滚**：`git revert` 阶段 2 的测试更新提交

**不可回滚的情况**：无。本方案不修改 `OrcaGymBaseEnv`/`OrcaGymLocalEnv`/Gym 层/子组件，仅变更 Env 继承链。

---

## 6. 完成标志

全部以下条件满足时，骨架变更完成：

| # | 条件 | 验证方式 |
|---|------|---------|
| 1 | `OrcaGymEnvMixin` 文件存在且测试通过 | 阶段 1 测试 |
| 2 | `OrcaGymEulerEnv` 继承 `OrcaGymEnvMixin, gym.Env` | `test_env_inheritance_chain` |
| 3 | `env.gym`/`env.stub`/`env.channel` 抛 `AttributeError` | `test_env_gym_attr_natural_attribute_error` 等 |
| 4 | `_BLOCKED_ATTRS`/`__getattr__`/`__setattr__`/`_SHIELDED_ATTRS` 已删除 | `test_env_no_blocked_attrs_classvar` |
| 5 | Mixin 方法在 Env 上可用 | `test_env_mixin_methods_available` |
| 6 | Env 骨架测试全部通过 | 阶段 2 测试 |
| 7 | SimpleEulerEnv 合规测试通过 | 阶段 3 测试 |
| 8 | Gym 层测试全部通过 | 阶段 3 测试 |
| 9 | ruff SLF001 零报警 | 阶段 3 扫描 |
| 10 | Example 端到端运行正常 | 阶段 3 example |
