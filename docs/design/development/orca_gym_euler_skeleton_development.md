# OrcaGym Euler 骨架开发设计文档

## 1. 文档定位

### 1.1 文档目标

本文是 `OrcaGymEulerEnv` + `OrcaGymEuler` **骨架阶段（P1–P3）** 的开发设计文档，对应架构文档 §12「骨架设计规约」。

骨架阶段的目标：**严格按照架构文档设计，将框架搭起来，满足全部架构约束（K1–K12），但不含业务功能。** 后续 P4 阶段在骨架中填充 64 个 `query_*/set_*` 方法与 Studio 交互方法。

> 架构文档：`docs/design/architecture/orca_gym_euler_architecture.md`（§12 为骨架规约，是本文的唯一上游约束）

### 1.2 骨架先行策略

采用**骨架先行、逐步填充**策略：

1. **骨架阶段（P1–P3，本文范围）**：实现满足全部架构约束的最小可运行骨架。骨架的方法体可以是空实现、`raise NotImplementedError`、或仅返回占位值——但**类结构、命名、隔离机制、委托路径、父类和解必须一次到位**。
2. **填充阶段（P4，不在本文范围）**：在骨架中填入 64 个 `query_*/set_*` 方法与 Studio 交互方法的真实逻辑。

**风险**（架构 §12.1）：若骨架本身不满足约束，后续填充阶段会持续受 `OrcaGymLocalEnv`/`OrcaGymLocal` 老架构影响，逐步滑回上帝类 + 封装泄漏的老路。骨架是承重墙，必须一次到位。

### 1.3 骨架的"无功能"边界

骨架阶段明确**不实现**的功能：

| 不实现的内容 | 骨架中的形态 |
|-------------|------------|
| 64 个 `query_*/set_*` 方法 | 不创建（P4 填充） |
| `apply_body_force`/`clear_*` 力应用方法 | 不创建（P4 填充） |
| Studio 交互 12 方法（视频/帧/内容文件） | 不创建（P4 填充） |
| MuJoCo 实际仿真逻辑（`mj_step` 真实调用） | 空实现或占位 |
| gRPC 实际通信 | 空实现或占位 |
| Euler 耦合逻辑 | `has_euler()` 恒返回 `False`，`_euler` 恒为 `None` |

骨架阶段**必须实现**的内容：

| 必须实现的内容 | 对应约束 |
|-------------|---------|
| 全部类定义与签名（§12.4） | K1–K12 |
| `_BLOCKED_ATTRS` + `__getattr__` + `__dir__`（Env 层 + Gym 层） | K2/K3 |
| `__setattr__` 父类屏蔽（方案 A） | K10 |
| 带下划线的内部属性命名 | K1 |
| 公共 property 委托（data/model/sim_config/dt/ctrl） | K6/K7 |
| Studio 桥接方法（非 property） | K9 |
| `has_euler()` + `step_with_coupling()` 封装 | K8 |
| 类 docstring 契约 | K12 |

---

## 2. 总体策略

### 2.1 自底向上构建

骨架按依赖关系自底向上构建，确保每一步只依赖已完成的下层：

```
P1: 叶子组件（无内部依赖）
    ├── SimConfig           （独立）
    └── OrcaGymDataView     （独立）

P2: 仿真核心组件 + Gym 组合
    ├── ModelRegistry       （依赖 OrcaGymModel/OrcaGymData）
    ├── MuJoCoSimCore       （依赖 DataView, SimConfig）
    ├── OrcaStudioBridge    （依赖 grpc stub）
    └── OrcaGymEuler        （组合上述全部 + 隔离机制 K3/K5/K8/K9）

P3: Env Facade + 父类和解
    └── OrcaGymEulerEnv     （组合 OrcaGymEuler + 隔离机制 K1/K2/K4/K6/K7/K10/K11/K12）
```

### 2.2 每步交付物

每个开发步骤交付三样东西：

1. **源码**：类定义、方法签名、隔离机制实现（方法体可空/占位）
2. **单元测试**：验证本步骤涉及的架构约束（不测功能，只测约束落实）
3. **验收清单**：本步骤通过的 K 约束编号

### 2.3 测试原则

骨架阶段的单元测试**重点验证架构约束的落实**，而非功能正确性：

- ✅ 测 `_BLOCKED_ATTRS` 是否拦截了禁用属性
- ✅ 测 `__dir__` 是否只暴露公共 API
- ✅ 测 `__setattr__` 是否正确屏蔽父类赋值
- ✅ 测 property 返回类型是否正确
- ✅ 测源码中不含穿墙访问模式（grep 断言）
- ❌ 不测 `mj_step` 后 qpos 是否正确更新（骨架不实现真实仿真）

---

## 3. 目录结构与文件规划

### 3.1 源码目录

```
orca_gym/
  core/
    euler/
      __init__.py
      sim_config.py               # SimConfig
      orca_gym_data_view.py       # OrcaGymDataView
      model_registry.py           # ModelRegistry
      mujoco_sim_core.py          # MuJoCoSimCore
      orca_studio_bridge.py       # OrcaStudioBridge
      orca_gym_euler.py           # OrcaGymEuler
  environment/
    euler/
      __init__.py
      orca_gym_euler_env.py       # OrcaGymEulerEnv
```

### 3.2 测试目录

```
tests/
  orca_gym/
    __init__.py                       (已存在)
    core/
      __init__.py
      euler/
        __init__.py
        test_sim_config.py            # P1-Step1
        test_orca_gym_data_view.py    # P1-Step2
        test_model_registry.py        # P2-Step1
        test_mujoco_sim_core.py       # P2-Step2
        test_orca_studio_bridge.py    # P2-Step3
        test_orca_gym_euler.py        # P2-Step4 (Gym 骨架验收)
    environment/
      __init__.py
      euler/
        __init__.py
        test_orca_gym_euler_env_skeleton.py  # P3 (Env 骨架验收)
```

测试通过 `tests/run_tests.py --component core/euler` 或 `--component environment/euler` 运行（见 [run_tests.py](../../../../tests/run_tests.py) 组件发现机制）。

---

## 4. 阶段一（P1）：叶子组件骨架

### P1-Step1：SimConfig

#### 目标

创建 `SimConfig` 骨架，提供 typed 的 MuJoCo 求解器参数读写接口（架构 §5.6）。

骨架阶段 `SimConfig` 不持有真实 `mjModel`，property 的 getter/setter 可操作内部占位字段或 `raise NotImplementedError`。但**签名和 typed 接口必须完整**。

#### 开发任务

**文件**：`orca_gym/core/euler/sim_config.py`

```python
class SimConfig:
    """MuJoCo 求解器参数配置。

    替代直接访问 _mjModel.opt.*。
    修改在下次 mj_step 时生效。

    使用契约:
        读取: ts = env.sim_config.timestep
        写入: env.sim_config.timestep = 0.002
        批量: env.sim_config.load_from_dict({"integrator": 0, "iterations": 100})

    禁止:
        不要通过 env._gym._sim._mjModel.opt.* 绕道访问。
    """

    def __init__(self, mj_model=None) -> None: ...

    # --- 骨架包含的 property（架构 §12.2）---
    @property
    def timestep(self) -> float: ...
    @timestep.setter
    def timestep(self, value: float) -> None: ...

    @property
    def integrator(self) -> int: ...
    @integrator.setter
    def integrator(self, value: int) -> None: ...

    @property
    def iterations(self) -> int: ...
    @iterations.setter
    def iterations(self, value: int) -> None: ...

    @property
    def gravity(self) -> np.ndarray: ...
    @gravity.setter
    def gravity(self, value: np.ndarray) -> None: ...

    def load_from_dict(self, config: dict) -> None: ...
    def to_dict(self) -> dict: ...
```

#### 单元测试设计

**文件**：`tests/orca_gym/core/euler/test_sim_config.py`

| 测试用例 | 验证内容 |
|---------|---------|
| `test_sim_config_constructable` | `SimConfig()` 可无参构造（骨架不依赖真实 mjModel） |
| `test_sim_config_has_timestep_property` | `timestep` 是 property，有 getter 和 setter |
| `test_sim_config_has_integrator_property` | `integrator` 是 property |
| `test_sim_config_has_iterations_property` | `iterations` 是 property |
| `test_sim_config_has_gravity_property` | `gravity` 是 property |
| `test_sim_config_has_load_from_dict` | `load_from_dict` 方法存在且可调用 |
| `test_sim_config_has_to_dict` | `to_dict` 方法存在且可调用 |
| `test_sim_config_docstring_has_contract` | docstring 含「使用契约」和「禁止」关键词 |

#### 验收标准

- [x] `SimConfig` 类可独立构造，不依赖 mujoco 运行时
- [x] 4 个 property（timestep/integrator/iterations/gravity）签名完整
- [x] `load_from_dict`/`to_dict` 方法存在
- [x] docstring 含使用契约

---

### P1-Step2：OrcaGymDataView

#### 目标

创建 `OrcaGymDataView` 骨架，提供 MuJoCo 状态的完整只读视图（架构 §5.7, §7.4）。

骨架阶段 DataView 不持有真实数据，字段可初始化为 `None` 或空数组。但**字段定义、方法签名、`__getattr__` 兜底机制必须完整**。

#### 开发任务

**文件**：`orca_gym/core/euler/orca_gym_data_view.py`

```python
class OrcaGymDataView:
    """MuJoCo 状态的完整只读视图。

    替代直接访问 _mjData。所有字段在 update_data() 后保证一致。
    用户永远不需要访问 _mjData。

    使用契约:
        读取状态:   env.data.qpos / env.data.body_xpos("link1")
        写入状态:   env.set_joint_qpos() / env.apply_body_force()

    禁止:
        不要通过 env._gym._sim._mjData 绕道访问。
        缺少字段时，扩展本类，不要绕道。
    """

    def __init__(self) -> None:
        # --- 基本状态（架构 §5.7，原 OrcaGymData 已有）---
        self.qpos: np.ndarray = np.array([])
        self.qvel: np.ndarray = np.array([])
        self.qacc: np.ndarray = np.array([])
        self.qfrc_bias: np.ndarray = np.array([])
        self.time: float = 0.0

        # --- 扩展字段（架构 §5.7，覆盖用户绕道访问的字段）---
        self.xfrc_applied: np.ndarray = np.array([])
        self.actuator_force: np.ndarray = np.array([])
        self.contact: list = []

    # --- body 查询方法 ---
    def body_xpos(self, body_name: str) -> np.ndarray: ...
    def body_xquat(self, body_name: str) -> np.ndarray: ...
    def body_xmat(self, body_name: str) -> np.ndarray: ...
    def body_cvel(self, body_name: str) -> np.ndarray: ...
    def body_subtree_mass(self, body_name: str) -> float: ...

    # --- site 查询方法 ---
    def site_xpos(self, site_name: str) -> np.ndarray: ...
    def site_xmat(self, site_name: str) -> np.ndarray: ...

    # --- M3: __getattr__ 兜底（架构 §7.4）---
    def __getattr__(self, name: str): ...
```

#### 单元测试设计

**文件**：`tests/orca_gym/core/euler/test_orca_gym_data_view.py`

| 测试用例 | 验证内容 |
|---------|---------|
| `test_data_view_constructable` | `OrcaGymDataView()` 可无参构造 |
| `test_data_view_has_basic_fields` | 实例有 `qpos`/`qvel`/`qacc`/`qfrc_bias`/`time` 五个基本字段 |
| `test_data_view_has_extended_fields` | 实例有 `xfrc_applied`/`actuator_force`/`contact` 扩展字段 |
| `test_data_view_has_body_query_methods` | `body_xpos`/`body_xquat`/`body_xmat`/`body_cvel`/`body_subtree_mass` 方法存在 |
| `test_data_view_has_site_query_methods` | `site_xpos`/`site_xmat` 方法存在 |
| `test_data_view_getattr_guidance` | 访问不存在的字段（如 `cvel`）抛 `AttributeError`，消息含引导文本（列出可用字段/方法） |
| `test_data_view_docstring_has_contract` | docstring 含「使用契约」和「禁止」关键词 |

#### 验收标准

- [x] 5 个基本字段 + 3 个扩展字段定义完整
- [x] 5 个 body 查询方法 + 2 个 site 查询方法签名完整
- [x] `__getattr__` 兜底：缺字段时抛 `AttributeError` 且消息含引导
- [x] docstring 含使用契约

---

## 5. 阶段二（P2）：仿真核心组件 + Gym 组合

### P2-Step1：ModelRegistry

#### 目标

创建 `ModelRegistry` 骨架，负责构建 `OrcaGymModel`/`OrcaGymData` 并提供扩展查询（架构 §5.5）。

骨架阶段 `ModelRegistry` 不执行真实模型构建，方法体可 `raise NotImplementedError`。但**签名和类结构必须完整**。

#### 开发任务

**文件**：`orca_gym/core/euler/model_registry.py`

```python
class ModelRegistry:
    """模型注册与结构查询。

    构建 OrcaGymModel/OrcaGymData，提供 body/equality 等模型信息查询。
    """

    def __init__(self, mj_model=None) -> None: ...

    def build_orca_gym_model(self): ...   # -> OrcaGymModel
    def build_orca_gym_data(self): ...    # -> OrcaGymData（注意：不是 DataView）
    def body_subtree_mass(self, body_name: str) -> float: ...
    def equality_data_width(self) -> int: ...
    def equality_object_ids(self, eq_idx: int) -> tuple[int, int]: ...
```

#### 单元测试设计

**文件**：`tests/orca_gym/core/euler/test_model_registry.py`

| 测试用例 | 验证内容 |
|---------|---------|
| `test_registry_constructable` | `ModelRegistry()` 可无参构造 |
| `test_registry_has_build_methods` | `build_orca_gym_model`/`build_orca_gym_data` 方法存在 |
| `test_registry_has_query_methods` | `body_subtree_mass`/`equality_data_width`/`equality_object_ids` 方法存在 |

#### 验收标准

- [x] 类可独立构造
- [x] 2 个 build 方法 + 3 个查询方法签名完整

---

### P2-Step2：MuJoCoSimCore

#### 目标

创建 `MuJoCoSimCore` 骨架，持有 `_mjModel`/`_mjData`（架构 §5.3）。

这是 `_mjModel`/`_mjData` **唯一**的存放位置。骨架阶段不执行真实 MuJoCo 操作，但**属性定义和方法签名必须完整**，且 `_mjModel`/`_mjData` 不能作为公共属性被外部访问（由 Gym 层隔离机制保证）。

#### 开发任务

**文件**：`orca_gym/core/euler/mujoco_sim_core.py`

```python
class MuJoCoSimCore:
    """MuJoCo 仿真核心，持有 _mjModel/_mjData。

    _mjModel/_mjData 只存在于此类内部，不对外暴露。
    通过 sync_to_view() 将状态同步到 OrcaGymDataView。

    禁止:
        外部不应直接访问本类的 _mjModel/_mjData。
        读取状态 → env.data（OrcaGymDataView）
    """

    def __init__(self) -> None:
        self._mjModel = None    # mujoco.MjModel | None
        self._mjData = None     # mujoco.MjData | None

    def init_simulation(self, model_xml_path: str) -> None: ...
    def step(self, nstep: int) -> None: ...
    def forward(self) -> None: ...
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...
    def sync_to_view(self, view: OrcaGymDataView) -> None: ...
    def apply_body_force(self, body_id: int, force: np.ndarray, torque: np.ndarray) -> None: ...
    def clear_body_force(self, body_id: int) -> None: ...
    def clear_all_forces(self) -> None: ...

    @property
    def nq(self) -> int: ...
    @property
    def nv(self) -> int: ...
    @property
    def nu(self) -> int: ...
```

#### 单元测试设计

**文件**：`tests/orca_gym/core/euler/test_mujoco_sim_core.py`

| 测试用例 | 验证内容 |
|---------|---------|
| `test_sim_core_constructable` | `MuJoCoSimCore()` 可无参构造 |
| `test_sim_core_has_mj_model_data_private` | 实例有 `_mjModel`/`_mjData` 属性（私有命名） |
| `test_sim_core_has_lifecycle_methods` | `init_simulation`/`step`/`forward`/`set_ctrl`/`sync_to_view` 方法存在 |
| `test_sim_core_has_force_methods` | `apply_body_force`/`clear_body_force`/`clear_all_forces` 方法存在 |
| `test_sim_core_has_nq_nv_nu_properties` | `nq`/`nv`/`nu` property 存在 |
| `test_sim_core_docstring_forbids_external_access` | docstring 含「禁止」和 `_mjModel`/`_mjData` 关键词 |

#### 验收标准

- [x] `_mjModel`/`_mjData` 为私有属性（带下划线）
- [x] 生命周期方法 + 力应用方法 + nq/nv/nu property 签名完整
- [x] docstring 明确禁止外部访问 `_mjModel`/`_mjData`

---

### P2-Step3：OrcaStudioBridge

#### 目标

创建 `OrcaStudioBridge` 骨架，处理与 OrcaStudio 的 gRPC 交互（架构 §5.4）。

骨架阶段不执行真实 gRPC 通信，方法体可 `raise NotImplementedError`。但**签名和依赖反转设计必须完整**（不持有 `_mjData`）。

#### 开发任务

**文件**：`orca_gym/core/euler/orca_studio_bridge.py`

```python
class OrcaStudioBridge:
    """OrcaStudio gRPC 集成桥接。

    依赖反转：不持有 _mjData，通过接收数据参数实现解耦。
    只负责通信和场景同步，不碰 mj_step。

    禁止:
        不要通过本类访问 MuJoCo 内部数据结构。
    """

    def __init__(self, stub=None) -> None: ...

    # --- 骨架最小集（架构 §12.2）---
    async def render(self, qpos: np.ndarray, sim_time: float) -> None: ...
    async def load_model_xml(self) -> str: ...
    async def pause_simulation(self) -> None: ...
    def configure_offline(self, xml_path: str, assets_dir: str | None = None) -> None: ...
    def set_timestep_remote(self, timestep: float) -> None: ...
    async def get_body_manipulation_anchored(self) -> tuple: ...
    async def get_body_manipulation_movement(self) -> dict: ...
```

#### 单元测试设计

**文件**：`tests/orca_gym/core/euler/test_orca_studio_bridge.py`

| 测试用例 | 验证内容 |
|---------|---------|
| `test_bridge_constructable` | `OrcaStudioBridge()` 可无参构造（stub=None） |
| `test_bridge_has_render` | `render` 方法存在 |
| `test_bridge_has_load_model_xml` | `load_model_xml` 方法存在 |
| `test_bridge_has_pause_simulation` | `pause_simulation` 方法存在 |
| `test_bridge_has_configure_offline` | `configure_offline` 方法存在 |
| `test_bridge_has_set_timestep_remote` | `set_timestep_remote` 方法存在 |
| `test_bridge_has_body_manipulation_methods` | `get_body_manipulation_anchored`/`get_body_manipulation_movement` 方法存在 |
| `test_bridge_no_mjdata_attribute` | 实例 `__dict__` 不含 `_mjData`/`_mjModel`/`mjData`/`mjModel` |
| `test_bridge_docstring_mentions_decoupling` | docstring 含「依赖反转」或「解耦」关键词 |

#### 验收标准

- [x] 类可无参构造（stub=None）
- [x] 7 个骨架方法签名完整
- [x] **不持有 `_mjData`/`_mjModel`**（依赖反转）
- [x] docstring 说明依赖反转设计

---

### P2-Step4：OrcaGymEuler 组合 + 隔离机制

#### 目标

创建 `OrcaGymEuler` 骨架，组合上述子组件，实现隔离机制（K3/K5/K8/K9）。

这是骨架阶段的核心交付物之一。`OrcaGymEuler` 是仿真核心 Facade，**不暴露子组件对象**（K5），**不暴露 `_mjModel`/`_mjData`**（K3），通过公共方法委托子组件。

#### 开发任务

**文件**：`orca_gym/core/euler/orca_gym_euler.py`

```python
class OrcaGymEuler:
    """仿真核心 Facade，组合子组件，不暴露 _mjModel/_mjData，不暴露子组件对象。

    ┌─────────────────────────────────────────────────────────────┐
    │  API 契约：用户不应直接访问 _mjData / _mjModel / 任何子组件。│
    │  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView）        │
    │  写入外力 → 使用 env.apply_body_force()                     │
    │  配置求解器 → 使用 env.sim_config                           │
    │  缺少功能时 → 扩展 OrcaGymEulerEnv 公共方法                 │
    └─────────────────────────────────────────────────────────────┘
    """

    # K3/K5: 隔离机制
    _BLOCKED_ATTRS = frozenset({
        # L3 引擎内部
        "_mjData", "_mjModel", "mj_data", "mj_model",
        "_mj_data", "_mj_model", "mjData", "mjModel",
        # K5: 子组件对象也不对外暴露
        "_sim", "_studio", "_registry", "_opt", "_view", "_euler",
        "sim", "studio", "registry", "opt", "view", "euler",
    })

    def __init__(self, stub=None) -> None:
        # 内部组件（全部带下划线，不在 __dir__ 暴露）
        self._sim = MuJoCoSimCore()
        self._studio = OrcaStudioBridge(stub=stub)
        self._registry = ModelRegistry()
        self._opt = SimConfig()
        self._view = OrcaGymDataView()
        self._euler = None    # EulerOrchestrator | None（骨架阶段恒为 None）

    # --- K3/K5: 隔离机制 ---
    def __getattr__(self, name: str): ...   # 拦截 _BLOCKED_ATTRS，返回引导性错误
    def __dir__(self) -> list[str]: ...     # 只列公共 API，不含子组件对象

    # --- 生命周期 ---
    async def init_simulation(self, model_xml_path: str) -> None: ...
    async def load_model_xml(self) -> str: ...

    # --- 仿真控制（委托 _sim）---
    def mj_step(self, nstep: int) -> None: ...
    def mj_forward(self) -> None: ...
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...

    # --- 状态同步 ---
    def sync_to_view(self) -> None: ...

    # --- K5: 状态访问（返回 typed 对象，不返回子组件引用）---
    @property
    def data(self) -> OrcaGymDataView: ...
    @property
    def model(self): ...    # -> OrcaGymModel
    @property
    def sim_config(self) -> SimConfig: ...

    # --- K9: Studio 桥接访问（方法而非 property）---
    def studio_bridge(self) -> OrcaStudioBridge: ...
    # 禁止: 不提供 @property def studio(self)

    # --- Studio 委托（骨架最小集）---
    async def render(self) -> None: ...
    async def pause_simulation(self) -> None: ...

    # --- K8: 步进耦合查询（供 do_simulation 使用，不暴露 _euler）---
    def has_euler(self) -> bool: ...           # 骨架阶段恒返回 False
    def step_with_coupling(self, ctrl: np.ndarray, n_frames: int, dt: float) -> None: ...
    # 禁止: do_simulation 内写 if self._gym._euler is not None
```

#### 单元测试设计

**文件**：`tests/orca_gym/core/euler/test_orca_gym_euler.py`

| 测试用例 | 验证内容 | 对应 K 约束 |
|---------|---------|-----------|
| `test_gym_constructable` | `OrcaGymEuler()` 可无参构造 | — |
| `test_gym_has_private_components` | 实例有 `_sim`/`_studio`/`_registry`/`_opt`/`_view`/`_euler`（全部带下划线） | K5 |
| `test_gym_blocked_attrs_include_components` | 访问 `gym._sim`/`gym._studio`/`gym._opt`/`gym._view`/`gym._euler`/`gym._mjData`/`gym._mjModel` 抛 `AttributeError` | K3/K5 |
| `test_gym_blocked_attrs_message_has_guidance` | `AttributeError` 消息含引导文本（`env.data`/`env.apply_body_force`/`env.sim_config`） | K3 |
| `test_gym_no_internal_property` | 源码 grep 不到 `@property\n    def studio`/`@property\n    def sim`/`@property\n    def opt`/`@property\n    def view`/`@property\n    def euler` | K5 |
| `test_gym_dir_only_exposes_public_api` | `dir(gym)` 不含 `_sim`/`_studio`/`_registry`/`_opt`/`_view`/`_euler`/`_mjData`/`_mjModel`/`sim`/`studio`/`opt` | K3 |
| `test_gym_dir_contains_public_methods` | `dir(gym)` 含 `data`/`model`/`sim_config`/`mj_step`/`mj_forward`/`set_ctrl`/`sync_to_view`/`studio_bridge`/`render`/`pause_simulation`/`has_euler`/`step_with_coupling` | K3 |
| `test_gym_has_euler_returns_false` | `has_euler()` 返回 `False`（骨架阶段无 Euler） | K8 |
| `test_gym_step_with_coupling_callable` | `step_with_coupling` 方法存在且可调用 | K8 |
| `test_gym_studio_bridge_is_method_not_property` | `studio_bridge` 是方法（callable），`gym.studio` 抛 `AttributeError` | K9 |
| `test_gym_data_returns_view` | `gym.data` 返回 `OrcaGymDataView` 实例 | K6 |
| `test_gym_sim_config_returns_config` | `gym.sim_config` 返回 `SimConfig` 实例 | — |
| `test_gym_docstring_has_contract` | docstring 含「API 契约」和「禁止」关键词 | K12 |

##### 违规访问拦截测试（对照架构 §6.2/§6.3/§6.5/§7.6）

以下测试对照架构文档明确列举的违规访问模式，验证 `_BLOCKED_ATTRS` 全部变体、多层穿墙路径、K8/K9 违规模式均被拦截：

| 测试用例 | 验证内容 | 对应 K 约束 |
|---------|---------|-----------|
| `test_all_mjdata_mjmodel_variants_blocked` | `_mjData`/`_mjModel` 全部 8 个变体（`mj_data`/`mj_model`/`_mj_data`/`_mj_model`/`mjData`/`mjModel`）都被拦截 | K3 |
| `test_all_component_variants_blocked` | 子组件全部带/不带下划线变体（`_sim`/`sim`/`_studio`/`studio` 等 12 个）都被拦截 | K5 |
| `test_multilayer_tunnel_mjdata_blocked` | 多层穿墙 `gym._sim._mjData` 在第一层 `gym._sim` 即被拦截（架构 §6.2 R1） | K3/K5 |
| `test_multilayer_tunnel_mjmodel_blocked` | 多层穿墙 `gym._sim._mjModel` 在第一层即被拦截（架构 §6.5 C1） | K3/K5 |
| `test_multilayer_tunnel_xfrc_blocked` | 多层穿墙 `gym._sim._mjData.xfrc_applied` 在第一层即被拦截（架构 §6.3 W2） | K3/K5 |
| `test_k8_euler_private_access_blocked` | `gym._euler` 访问被拦截，引导消息含 `has_euler`/`step_with_coupling`（架构 §8.2） | K8 |
| `test_k9_studio_property_access_blocked` | `gym.studio` 访问被拦截，引导消息含 `studio_bridge`（架构 §7.1 M2） | K9 |
| `test_k5_sim_access_guided_to_step_methods` | `gym._sim` 访问被拦截，引导消息含 `mj_step`/`mj_forward`/`do_simulation` | K5 |
| `test_k5_opt_access_guided_to_sim_config` | `gym._opt` 访问被拦截，引导消息含 `sim_config` | K5 |
| `test_k5_view_access_guided_to_data` | `gym._view` 访问被拦截，引导消息含 `env.data` | K5 |
| `test_blocked_attrs_frozenset_complete` | `_BLOCKED_ATTRS` 是 frozenset 且包含全部 20 个拦截名（8 引擎内部 + 12 子组件） | K3/K5 |

#### 验收标准

- [x] K3：`_BLOCKED_ATTRS` 含 `_mjData`/`_mjModel` + 子组件名，`__getattribute__` 拦截并返回引导
- [x] K3：`_BLOCKED_ATTRS` 全部 8 个变体（`mj_data`/`_mj_data`/`mjData` 等）都被拦截
- [x] K5：不提供 `studio`/`sim`/`opt`/`view`/`euler` 的 public property
- [x] K5：`__dir__` 不列出任何子组件对象或引擎内部
- [x] K3/K5：多层穿墙路径（`gym._sim._mjData`/`gym._sim._mjModel`/`gym._sim._mjData.xfrc_applied`）在第一层即被拦截
- [x] K8：`has_euler()` 返回 `False`，`step_with_coupling()` 存在，`gym._euler` 访问被拦截并引导
- [x] K9：`studio_bridge()` 是方法，`gym.studio` 抛 `AttributeError` 并引导用 `studio_bridge()`
- [x] K5：引导消息针对不同违规类型给出精准引导（`_sim`→步进方法、`_opt`→`sim_config`、`_view`→`env.data`、`_euler`→`has_euler`、`_studio`→`studio_bridge`）
- [x] K12：docstring 含 API 契约框

---

## 6. 阶段三（P3）：Env Facade + 父类和解

### P3-Step1：OrcaGymEulerEnv 完整骨架

#### 目标

创建 `OrcaGymEulerEnv` 骨架，继承 `OrcaGymBaseEnv`，实现 Env 层隔离机制（K1/K2/K4/K6/K7/K8/K9/K10/K11/K12）。

这是骨架阶段**最关键**的交付物。Env 层必须：
1. 持有 `_gym`/`_stub`/`_channel`/`_studio_bridge`（带下划线）—— K1
2. 实现 `_BLOCKED_ATTRS` + `__getattr__` + `__dir__` —— K2
3. 通过 `__setattr__` 屏蔽父类的 `self.gym`/`self.model`/`self.data` 赋值 —— K10（方案 A）
4. 所有仿真控制通过 `self._gym` 公共方法委托，不触私有 —— K4/K8
5. `env.data` 返回 `OrcaGymDataView` —— K6
6. Studio 交互通过 `self._studio_bridge`，不通过 `gym.studio` —— K9

#### 开发任务

**文件**：`orca_gym/environment/euler/orca_gym_euler_env.py`

```python
class OrcaGymEulerEnv(OrcaGymBaseEnv):
    """OrcaGym Euler 环境 Facade。

    使用契约:
        读取状态:   env.data.qpos / env.data.body_xpos(name) / env.query_*()
        写入状态:   env.set_joint_qpos() / env.apply_body_force()
        仿真步进:   env.do_simulation(ctrl, n_frames)
        求解器配置: env.sim_config.timestep = 0.002

    禁止:
        不要访问 env._gym._sim._mjData 或任何内部 MuJoCo 对象。
        不要访问 env._gym._studio / env._studio_bridge（内部组件）。
        缺少功能时，扩展本类的公共方法。
    """

    # K2: Env 层隔离机制（与 Gym 层对称）
    _BLOCKED_ATTRS = frozenset({
        # L3 引擎内部
        "_mjData", "_mjModel", "mj_data", "mj_model",
        "_mj_data", "_mj_model", "mjData", "mjModel",
        # L2 内部组件（含父类残留的公共名）
        "gym", "stub", "channel",
    })

    # K10: 父类契约屏蔽字段（方案 A）
    _SHIELDED_ATTRS = frozenset({"gym", "stub", "channel", "model", "data"})

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
        # 注意: __setattr__ 是类级方法，定义即生效，无需额外注册
        self._skip_grpc_load = skip_grpc_load
        self._local_xml_path = model_xml_path
        self._render_mode = render_mode
        # ... 其他 Env 自有字段 ...

        # 调用父类 __init__（会触发 initialize_grpc / initialize_simulation 等）
        # 父类中的 self.gym=X / self.model=Y / self.data=Z 被 __setattr__ 屏蔽
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

    # --- K10: __setattr__ 屏蔽父类赋值（方案 A，架构 §12.5）---
    def __setattr__(self, name, value):
        if name == "gym":
            object.__setattr__(self, "_gym", value)   # 转发到 _gym
            return
        if name == "stub":
            object.__setattr__(self, "_stub", value)
            return
        if name == "channel":
            object.__setattr__(self, "_channel", value)
            return
        if name == "model":
            return   # 忽略：model 始终通过 @property 从 self._gym.model 取
        if name == "data":
            return   # 忽略：data 始终通过 @property 从 self._gym.data 取
        super().__setattr__(name, value)

    # --- K2: 隔离机制 ---
    def __getattr__(self, name: str): ...   # 拦截 _BLOCKED_ATTRS，返回引导性错误
    def __dir__(self) -> list[str]: ...     # 只列 L1 公共 API

    # --- 生命周期（实现 OrcaGymBaseEnv 抽象方法）---
    def initialize_grpc(self) -> None: ...
    def initialize_simulation(self) -> tuple: ...   # -> (OrcaGymModel, OrcaGymDataView)，K6
    def reset_simulation(self) -> None: ...
    def init_qpos_qvel(self) -> None: ...
    def set_time_step(self, time_step: float) -> None: ...
    def pause_simulation(self) -> None: ...
    def close(self) -> None: ...

    # --- 仿真控制（K4/K8: 全部委托 self._gym 公共方法，不触私有）---
    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None: ...
    def mj_step(self, nstep: int) -> None: ...
    def mj_forward(self) -> None: ...
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...

    # --- K6/K7: 状态访问（通过 Gym 公共属性，不触私有）---
    @property
    def data(self) -> OrcaGymDataView: ...
    @property
    def model(self): ...    # -> OrcaGymModel
    @property
    def sim_config(self) -> SimConfig: ...
    @property
    def dt(self) -> float: ...
    @property
    def ctrl(self) -> np.ndarray: ...
    @ctrl.setter
    def ctrl(self, value: np.ndarray) -> None: ...

    # --- 渲染（K9: Studio 交互通过 self._studio_bridge）---
    def render(self): ...   # -> NDArray | None
    def do_body_manipulation(self) -> None: ...   # 占位，P4 填充

    # --- Gymnasium 接口（子类实现）---
    def step(self, action) -> tuple: ...
    def reset_model(self) -> tuple[dict, dict]: ...
    def _get_obs(self) -> dict: ...
```

#### 关键实现要点

**K4/K8 合规的 `do_simulation` 实现**（架构 §12.4.3）：

```python
# ✅ K4 合规：do_simulation 只走 Gym 公共方法
def do_simulation(self, ctrl, n_frames):
    if np.array(ctrl).shape != (self.model.nu,):
        raise ValueError(...)
    # K8: 不写 if self._gym._euler is not None
    self._gym.step_with_coupling(ctrl, n_frames, self.dt)
    self._gym.sync_to_view()
```

**K9 合规的 `initialize_grpc` 实现**（架构 §12.4.3）：

```python
# ✅ K9 合规：Studio 交互通过自持 _studio_bridge
def initialize_grpc(self):
    if self._skip_grpc_load:
        self._channel = None
        self._stub = None
        self._gym = OrcaGymEuler(stub=None)
        self._studio_bridge = self._gym.studio_bridge()   # 取一次引用
        if self._local_xml_path:
            self._studio_bridge.configure_offline(...)    # 后续用自持引用
        return
    self._channel = grpc.aio.insecure_channel(...)
    self._stub = GrpcServiceStub(self._channel)
    self._gym = OrcaGymEuler(stub=self._stub)
    self._studio_bridge = self._gym.studio_bridge()
```

**K7 合规的属性委托**（架构 §12.4.3）：

```python
# ✅ K7 合规：属性通过 Gym 公共属性
@property
def data(self) -> OrcaGymDataView:
    return self._gym.data

@property
def sim_config(self) -> SimConfig:
    return self._gym.sim_config

@property
def dt(self) -> float:
    return self._gym.sim_config.timestep * self.frame_skip
```

#### 单元测试设计

**文件**：`tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py`

##### K1: 命名约束

| 测试用例 | 验证内容 |
|---------|---------|
| `test_env_no_public_internal_attrs` | Env 实例 `__dict__` 不含 `gym`/`stub`/`channel`（含 `_gym`/`_stub`/`_channel`） |

##### K2: Env 层隔离机制

| 测试用例 | 验证内容 |
|---------|---------|
| `test_env_blocked_attrs_raise_guidance` | 访问 `env.gym`/`env.stub`/`env._mjData`/`env._mjModel`/`env.mjData`/`env.mjModel` 抛 `AttributeError`，消息含引导文本（指明用 `env.data`/`env.sim_config` 替代） |
| `test_env_dir_only_exposes_public_api` | `dir(env)` 不含 `gym`/`stub`/`channel`/`_gym`/`_studio_bridge`/`_mjData`/`_mjModel` |
| `test_env_dir_contains_public_api` | `dir(env)` 含 `data`/`model`/`sim_config`/`dt`/`ctrl`/`do_simulation`/`mj_step`/`mj_forward`/`set_ctrl`/`render` |

##### K2 违规访问拦截测试（对照架构 §6.2/§6.3/§6.5/§7.6）

以下测试验证 Env 层 `_BLOCKED_ATTRS` 全部变体、三层穿墙路径、K4/K8/K9 违规模式均被拦截。Env 层是用户直接接触的入口，穿墙路径比 Gym 层多一层（`env._gym._sim._mjData`）：

| 测试用例 | 验证内容 | 对应 K 约束 |
|---------|---------|-----------|
| `test_env_all_mjdata_mjmodel_variants_blocked` | Env `_BLOCKED_ATTRS` 中 `_mjData`/`_mjModel` 全部 8 个变体都被拦截 | K2 |
| `test_env_all_internal_component_variants_blocked` | `gym`/`stub`/`channel` 及其带下划线变体都被拦截 | K1/K2 |
| `test_env_multilayer_tunnel_mjdata_blocked` | 三层穿墙 `env._gym._sim._mjData` 在第一层 `env._gym` 即被拦截（架构 §6.2 R1） | K2/K4 |
| `test_env_multilayer_tunnel_mjmodel_opt_blocked` | 三层穿墙 `env._gym._sim._mjModel.opt` 在第一层即被拦截（架构 §6.5 C1） | K2/K4 |
| `test_env_multilayer_tunnel_xfrc_blocked` | 三层穿墙 `env._gym._sim._mjData.xfrc_applied` 在第一层即被拦截（架构 §6.3 W2） | K2/K4 |
| `test_env_k8_euler_tunnel_blocked` | 四层穿墙 `env._gym._euler` 在第一层 `env._gym` 即被拦截（架构 §8.2） | K2/K8 |
| `test_env_k9_studio_tunnel_blocked` | 穿墙 `env._gym.studio` 在第一层 `env._gym` 即被拦截（架构 §7.1 M2） | K2/K9 |
| `test_env_blocked_attrs_frozenset_complete` | Env `_BLOCKED_ATTRS` 是 frozenset 且包含全部拦截名 | K2 |

##### K4: 不穿墙访问 Gym 私有

| 测试用例 | 验证内容 |
|---------|---------|
| `test_env_no_gym_private_access` | Env 源码（`orca_gym_euler_env.py`）grep 不到 `_gym._sim`/`_gym._studio`/`_gym._registry`/`_gym._opt`/`_gym._view`/`_gym._euler` |

##### K6: data 返回 DataView

| 测试用例 | 验证内容 |
|---------|---------|
| `test_data_property_returns_view` | `env.data` 是 `OrcaGymDataView` 实例，非 `OrcaGymData` |
| `test_initialize_simulation_returns_view` | `initialize_simulation()` 返回的第二个元素是 `OrcaGymDataView` |

##### K8: do_simulation 不读 _euler

| 测试用例 | 验证内容 |
|---------|---------|
| `test_do_simulation_no_euler_private_access` | `do_simulation` 源码 grep 不到 `_euler` |

##### K9: Studio 访问合规

| 测试用例 | 验证内容 |
|---------|---------|
| `test_no_studio_property_access` | Env 源码 grep 不到 `gym.studio`（允许 `_studio_bridge` 和 `_gym.studio_bridge()`） |

##### K10: 父类契约屏蔽

| 测试用例 | 验证内容 |
|---------|---------|
| `test_parent_assignment_shielded` | 父类 `self.gym = X` 后 `env.gym` 仍抛 `AttributeError`，`env._gym` 是 X |
| `test_parent_model_assignment_shielded` | 父类 `self.model = M` 后 `env.model` 走 property（从 `_gym.model` 取），不接受父类赋值 |
| `test_parent_data_assignment_shielded` | 父类 `self.data = D` 后 `env.data` 走 property（从 `_gym.data` 取），不接受父类赋值 |
| `test_stub_channel_shielded` | 父类 `self.stub = S` 后 `env.stub` 抛 `AttributeError`，`env._stub` 是 S |

##### K11: typed 返回

| 测试用例 | 验证内容 |
|---------|---------|
| `test_public_methods_return_typed` | `env.data` 返回 `OrcaGymDataView`，`env.sim_config` 返回 `SimConfig`（不返回 `mujoco.MjData`/`mujoco.MjModel`） |

##### K12: docstring 契约

| 测试用例 | 验证内容 |
|---------|---------|
| `test_env_docstring_has_contract` | Env 类 docstring 含「使用契约」和「禁止」关键词 |

##### K1/K4/K9 源码审查测试（grep 断言）

以下测试通过读取源码文件并进行字符串匹配来验证约束：

```python
import pathlib

ENV_SOURCE = pathlib.Path("orca_gym/environment/euler/orca_gym_euler_env.py").read_text()

def test_env_no_gym_private_access():
    """K4: Env 源码不含 _gym._sim / _gym._studio 等穿墙访问。"""
    forbidden_patterns = [
        "_gym._sim", "_gym._studio", "_gym._registry",
        "_gym._opt", "_gym._view", "_gym._euler",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in ENV_SOURCE, (
            f"K4 违规: Env 源码包含穿墙访问 '{pattern}'"
        )

def test_no_studio_property_access():
    """K9: Env 源码不含 gym.studio（允许 _studio_bridge 和 _gym.studio_bridge()）。"""
    assert "gym.studio." not in ENV_SOURCE.replace("_gym.studio_bridge", ""), (
        "K9 违规: Env 源码通过 gym.studio 访问 Studio"
    )
    # gym.studio_bridge() 是允许的（方法调用）

def test_do_simulation_no_euler_private_access():
    """K8: do_simulation 源码不含 _euler。"""
    # 提取 do_simulation 方法体（简单实现：检查整个文件中 _euler 出现次数为 0）
    assert "_euler" not in ENV_SOURCE, (
        "K8 违规: Env 源码不应出现 _euler（耦合查询通过 has_euler/step_with_coupling）"
    )
```

#### 验收标准

- [x] K1：`__dict__` 含 `_gym`/`_stub`/`_channel`，不含 `gym`/`stub`/`channel`（`TestEnvK1NamingConstraint.test_env_no_public_internal_attrs`）
- [x] K2：`_BLOCKED_ATTRS` 拦截 + `__dir__` 只暴露公共 API（`TestEnvK2Isolation.test_env_blocked_attrs_raise_guidance` / `test_env_dir_only_exposes_public_api` / `test_env_dir_contains_public_api`）
- [x] K2：`_BLOCKED_ATTRS` 全部 8 个 `_mjData`/`_mjModel` 变体都被拦截（`TestEnvK2ViolationPatterns.test_env_all_mjdata_mjmodel_variants_blocked`）
- [x] K2/K4：三层穿墙路径（`env._gym._sim._mjData`/`env._gym._sim._mjModel.opt`/`env._gym._sim._mjData.xfrc_applied`）在第一层 `env._gym` 即被拦截（`TestEnvK2ViolationPatterns.test_env_multilayer_tunnel_*`；`env._gym._sim` 在 OrcaGymEuler `__getattr__` 层拦截）
- [x] K4：源码 grep 不到 `_gym._sim`/`_gym._studio`/`_gym._registry`/`_gym._opt`/`_gym._view`/`_gym._euler`（`TestEnvK4NoGymPrivateAccess.test_env_no_gym_private_access`，AST 去除 docstring 后检查）
- [x] K6：`env.data` 类型为 `OrcaGymDataView`（`TestEnvK6DataView.test_data_property_returns_view`）
- [x] K7：`model`/`sim_config`/`dt` 通过 Gym 公共属性委托（`TestEnvK7PropertyDelegation.test_data_delegates_to_gym` / `test_sim_config_delegates_to_gym` / `test_dt_uses_sim_config`）
- [x] K8：`do_simulation` 源码 grep 不到 `_euler`，`env._gym._euler` 穿墙在第一层即被拦截（`TestEnvK8NoEulerPrivate.test_do_simulation_no_euler_private_access` + `TestEnvK2ViolationPatterns.test_env_k8_euler_tunnel_blocked`，词边界正则忽略 `orca_gym_euler` 模块名）
- [x] K9：源码 grep 不到 `gym.studio`（允许 `_studio_bridge` 和 `_gym.studio_bridge()`），`env._gym.studio` 穿墙在第一层即被拦截（`TestEnvK9StudioAccess.test_no_studio_property_access` + `TestEnvK2ViolationPatterns.test_env_k9_studio_tunnel_blocked`）
- [x] K10：`__setattr__` 屏蔽父类的 `gym`/`stub`/`channel`/`model`/`data` 赋值（`TestEnvK10ParentShielding.test_parent_*_assignment_shielded`）
- [x] K11：公共方法返回 typed 对象（`TestEnvK11TypedReturn.test_data_returns_view_not_mjdata` / `test_sim_config_returns_config`）
- [x] K12：docstring 含使用契约（`TestEnvK12Docstring.test_env_docstring_has_contract`）

---

## 7. 骨架验收门槛（硬性门槛）

### 7.1 验收测试总览

骨架阶段全部测试通过方可进入 P4 填充。测试分两个组件目录运行：

```bash
# Gym 层测试（P1 + P2）
<conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler

# Env 层测试（P3）
<conda-base>/envs/orca/bin/python tests/run_tests.py --component environment/euler

# 全量运行
<conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler --component environment/euler
```

### 7.2 K1–K12 约束验收矩阵

| K 约束 | 描述 | 验收测试 | 测试文件 |
|--------|------|---------|---------|
| K1 | Env 持有 `_gym`/`_stub`/`_channel`（带下划线） | `test_env_no_public_internal_attrs` | `test_orca_gym_euler_env_skeleton.py` |
| K2 | Env 实现 `_BLOCKED_ATTRS` + `__getattr__` + `__dir__` | `test_env_blocked_attrs_raise_guidance`, `test_env_dir_only_exposes_public_api` | `test_orca_gym_euler_env_skeleton.py` |
| K3 | Gym 实现 `_BLOCKED_ATTRS` + `__getattr__` + `__dir__` | `test_gym_blocked_attrs_*`, `test_gym_dir_*` | `test_orca_gym_euler.py` |
| K4 | Env 不直接访问 Gym 私有 | `test_env_no_gym_private_access` | `test_orca_gym_euler_env_skeleton.py` |
| K5 | Gym 不通过 public property 暴露子组件 | `test_gym_no_internal_property`, `test_gym_blocked_attrs_include_components` | `test_orca_gym_euler.py` |
| K6 | `env.data` 类型为 `OrcaGymDataView` | `test_data_property_returns_view`, `test_gym_data_returns_view` | `test_orca_gym_euler_env_skeleton.py`, `test_orca_gym_euler.py` |
| K7 | `env.model`/`sim_config`/`dt` 通过 Gym 公共委托 | 源码审查 + `test_env_dir_contains_public_api` | `test_orca_gym_euler_env_skeleton.py` |
| K8 | `do_simulation` 不读 `_euler`，通过公共方法 | `test_do_simulation_no_euler_private_access`, `test_gym_has_euler_and_step_with_coupling` | `test_orca_gym_euler_env_skeleton.py`, `test_orca_gym_euler.py` |
| K9 | Studio 通过 `_studio_bridge` 或方法，不通过 `gym.studio` | `test_no_studio_property_access`, `test_gym_studio_bridge_is_method_not_property` | `test_orca_gym_euler_env_skeleton.py`, `test_orca_gym_euler.py` |
| K10 | 父类赋值被 `__setattr__` 屏蔽 | `test_parent_assignment_shielded`, `test_parent_model_assignment_shielded`, `test_parent_data_assignment_shielded`, `test_stub_channel_shielded` | `test_orca_gym_euler_env_skeleton.py` |
| K11 | 公共方法返回 typed 对象 | `test_public_methods_return_typed` | `test_orca_gym_euler_env_skeleton.py` |
| K12 | Env/Gym/DataView docstring 含契约 | `test_env_docstring_has_contract`, `test_gym_docstring_has_contract`, `test_data_view_docstring_has_contract` | 各组件测试文件 |

### 7.3 骨架验收清单

进入 P4 填充阶段前，以下全部勾选：

- [x] P1-Step1: SimConfig 骨架 + 测试通过
- [x] P1-Step2: OrcaGymDataView 骨架 + 测试通过
- [x] P2-Step1: ModelRegistry 骨架 + 测试通过
- [x] P2-Step2: MuJoCoSimCore 骨架 + 测试通过
- [x] P2-Step3: OrcaStudioBridge 骨架 + 测试通过
- [x] P2-Step4: OrcaGymEuler 骨架 + 测试通过（K3/K5/K8/K9）
- [x] P3-Step1: OrcaGymEulerEnv 骨架 + 测试通过（K1/K2/K4/K6/K7/K8/K9/K10/K11/K12）
- [x] `--component core/euler` 全量通过（80 测试）
- [x] `--component environment/euler` 全量通过（43 测试）
- [x] 源码审查：无穿墙访问、无 public property 暴露子组件、无 `_mjData`/`_mjModel` 公共暴露

---

## 8. 骨架与 P4 填充的边界

### 8.1 P4 填充阶段不得破坏的不变性

骨架完成后，P4 填充阶段**不得破坏**以下不变性（架构 §12.7）：

| 不变性 | 骨架阶段的体现 | P4 填充约束 |
|--------|-------------|------------|
| `_mjModel`/`_mjData` 永不作为公共属性暴露 | K1/K2/K3 | 新增方法不得返回 `_mjData`/`_mjModel` |
| `env.data` 类型恒为 `OrcaGymDataView` | K6 | 新增 `query_*` 可读 DataView，不得改 `data` 返回类型 |
| Env 不穿墙访问 Gym 私有 | K4/K8 | 新增 `set_*`/`apply_body_force` 通过 Gym 公共方法委托 |
| Studio 桥接不作为公共 property | K5/K9 | 新增 Studio 交互方法委托 `self._studio_bridge`，不复活 `gym.studio` |
| `__dir__` 与公共 API 同步 | K2 | 每新增一个公共方法，同步加入 Env/Gym 的 `__dir__`，并补 `test_dir_matches_public_api` |

### 8.2 P4 填充的回归要求

P4 填充每批方法后，必须重跑 §7 的全部骨架验收测试，确保骨架约束未被破坏。

```bash
# P4 每批填充后的回归命令
<conda-base>/envs/orca/bin/python tests/run_tests.py --component core/euler --component environment/euler
```

---

## 9. 开发顺序速查

```
P1-Step1  SimConfig              → test_sim_config.py
P1-Step2  OrcaGymDataView        → test_orca_gym_data_view.py
P2-Step1  ModelRegistry          → test_model_registry.py
P2-Step2  MuJoCoSimCore          → test_mujoco_sim_core.py
P2-Step3  OrcaStudioBridge       → test_orca_studio_bridge.py
P2-Step4  OrcaGymEuler           → test_orca_gym_euler.py          (K3/K5/K8/K9)
P3-Step1  OrcaGymEulerEnv        → test_orca_gym_euler_env_skeleton.py  (K1/K2/K4/K6/K7/K8/K9/K10/K11/K12)
          ──────────────────────
          骨架验收门槛（§7）       → 全部 K1–K12 通过 → 进入 P4
```
