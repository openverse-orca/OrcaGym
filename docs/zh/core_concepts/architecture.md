# ⚙️ OrcaGym Euler 架构

> 📌 **前置阅读**：如果你是第一次接触 OrcaGym 架构，建议先阅读 [架构总览](architecture-overview.md) 了解整体分层与 API 边界，再回到本文查看组件设计细节。

## 1. 为什么需要新架构

### 1.1 现有体系的问题

`OrcaGymLocalEnv` + `OrcaGymLocal` 作为当前主用路径，存在三类结构性缺陷：

| 缺陷类型 | 表现 | 后果 |
|---------|------|------|
| **上帝类** | `OrcaGymLocal` 单类承担仿真核心、Studio 集成、模型注册、求解器配置、物体操作等所有职责 | 难以维护、难以扩展、职责耦合 |
| **API 不完备** | `OrcaGymData` 只缓存 5 个字段（qpos/qvel/qacc/qfrc_bias/time），缺 xfrc_applied/cvel/contact 等 | 你不得不绕道 `gym._mjData` 直接访问 |
| **封装泄漏** | `_mjModel`/`_mjData` 作为公共属性暴露，`self.gym` 既是内部组件又作外部库句柄 | 83 处直接访问，封装形同虚设 |

### 1.2 用户代码现状

对 OrcaPlayground 和 OrcaManipulation 两个仓库的分析显示：

- **17 个直接子类** + **4 个间接子类**继承自 `OrcaGymLocalEnv`
- **83 处**直接访问 `gym._mjData` / `gym._mjModel`
- 典型绕道场景：外力注入（`xfrc_applied`）、求解器配置（`opt.*`）、body 属性查询（`body_subtreemass`/`cvel`）、等式约束结构访问（`eq_data`）

你绕道不是因为不懂封装，而是**封装没有覆盖你的需求**。

### 1.3 新架构做了什么

`OrcaGymEulerEnv` + `OrcaGymEuler` 采用 **Facade + 职责内聚分解**的设计，替代 `OrcaGymLocalEnv` + `OrcaGymLocal` 的上帝类模式：

- **完备的公共 API**：覆盖所有合法的 MuJoCo 操作需求，你不再需要绕道访问内部数据
- **封装隔离**：`_mjModel`/`_mjData` 不对外暴露，从机制上引导你和 AI 走正确路径
- **平滑迁移**：约 70% API 零改动，25% 机械替换，5% 设计调整

---

## 2. 核心设计原则

### 2.1 六大原则

| 原则 | 含义 | 对比旧体系 |
|------|------|-----------|
| **P1 完备性** | 公共 API 覆盖所有合法的 MuJoCo 操作需求 | 大量缺口迫使绕道 |
| **P2 不暴露引擎内部** | `_mjModel`/`_mjData` 不作为公共属性暴露 | 直接暴露 |
| **P3 状态一致性** | 任何写操作后，`self.data` 保证一致；任何读操作都走 `self.data` 或显式查询 | `self.data` 与 `_mjData` 双轨制 |
| **P4 力应用可追踪** | 外力注入通过显式方法，未来 Euler 耦合器可感知 | `xfrc_applied` 直接写，无感知 |
| **P5 职责内聚** | 按职责内聚划分模块，一组方法因同一原因变化、共享同一组数据 | 上帝类 |
| **P6 框架无状态、业务自编排** | 框架只提供无状态原语（单次原子读写），多步编排流程由你的业务代码自行组合实现并自管状态 | 框架混杂原语与编排，易误用 |

#### P6 补充说明

框架公共 API 都是**无状态原语**——单次调用完成单一数据读写，不依赖前后调用顺序，不持有跨调用的快照或绑定标记。例如：

- `equality_find_slot_by_body` — 按 body 查槽位
- `equality_constraint(slot)` — 读单槽位
- `equality_update(slot, **fields)` — 写单槽位
- `set_mocap_pos_and_quat` — 写 mocap 位姿

如果你的业务需要"绑定/释放/抓取"等多步编排流程，请自行组合这些原语实现，业务状态由你自己管理。这比框架代管更易审查、更不易误用。

### 2.2 设计模式

| 模式 | 应用位置 | 解决的问题 |
|------|---------|-----------|
| **Facade** | `OrcaGymEulerEnv` / `OrcaGymEuler` | 组合多个子组件，提供统一 API，避免上帝类 |
| **组合优于继承** | Env 持有 Gym，Gym 持有子组件 | 避免继承链腐化，职责可独立演进 |
| **策略模式** | `EulerOrchestrator`（可选） | 通过 None 检查切换有无 Euler 的策略 |
| **依赖反转** | `OrcaStudioBridge` 不持有 mjData，通过接收数据参数实现解耦 | Studio 集成与仿真核心解耦 |
| **只读视图** | `OrcaGymDataView` | 提供完整状态读取，禁止写入 |

---

## 3. 架构总览

### 3.1 整体结构

```
gym.Env
  └── OrcaGymEulerEnv (新)                    (Facade + 契约执行者，直接继承 gym.Env)
        │   ↑ OrcaGymEnvMixin（名称空间、动作/观测空间生成、reset 编排）
        │
        │   组合（非继承）
        ├── _gym: OrcaGymEuler           (仿真核心 Facade)
        │     ├── _sim: MuJoCoSimCore    # 持有 _mjModel/_mjData（不对外暴露）
        │     ├── _studio: OrcaStudioBridge  # gRPC 集成
        │     ├── _registry: ModelRegistry  # 模型信息
        │     ├── _opt: SimConfig        # 求解器配置（typed）
        │     └── _euler: EulerOrchestrator | None  # Euler 耦合（占位，后续设计）
        │
        │   公共 API（你面向的接口）
        ├── .data → OrcaGymDataView      # 完整状态视图
        ├── .model → OrcaGymModel        # 模型结构（原样复用）
        ├── .sim_config → SimConfig      # 求解器配置
        ├── .ctrl → np.ndarray           # 控制数组
        │
        ├── 仿真控制
        ├── 状态查询
        ├── 状态设置
        ├── 名称空间
        └── Studio 交互
```

### 3.2 与旧体系的对比

| 维度 | OrcaGymLocalEnv + OrcaGymLocal | OrcaGymEulerEnv + OrcaGymEuler |
|------|-------------------------------|-------------------------------|
| 类结构 | 上帝类，单类承担所有职责 | Facade + 职责内聚分解 |
| `_mjModel`/`_mjData` | 公共属性，83 处直接访问 | 内部组件，多层隔离 |
| `OrcaGymData` | 5 字段缓存，不完整 | `OrcaGymDataView` 完整只读视图 |
| 求解器配置 | 无接口，绕道 `opt.*` | `SimConfig` typed 配置 |
| 外力注入 | 直接写 `xfrc_applied` | `apply_body_force()` 显式方法 |
| 继承体系 | 继承自腐化的 `OrcaGymBase` 链 | 直接继承 `gym.Env` + `OrcaGymEnvMixin` |

### 3.3 与旧体系的共存

原有 `OrcaGymBase` → Remote / Local / Warp 继承体系不动，现有系统继续运行不受影响。`OrcaGymEulerEnv` 是独立的新类，与 `OrcaGymLocal` 长期共存。迁移路径上，`OrcaGymLocal` 最终将被废弃，届时原有体系直接删除即可。

---

## 4. 组件说明

### 4.1 OrcaGymEulerEnv — 环境 Facade

作为 Gymnasium `Env` 的实现，组合 `OrcaGymEuler` 仿真核心，向你暴露统一 API。继承结构：`OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env)`。

```python
class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env):
    """OrcaGym Euler 双引擎环境。

    使用契约:
        读取状态:   env.data.qpos / env.data.body_xpos(name) / env.query_*()
        写入状态:   env.set_joint_qpos() / env.apply_body_force()
        仿真步进:   env.do_simulation(ctrl, n_frames)
        求解器配置: env.sim_config.timestep = 0.002

    禁止:
        不要访问 env._gym._sim._mjData 或任何内部 MuJoCo 对象。
        env.gym/env.stub/env.channel 不存在，直接继承 gym.Env 不创建这些属性。
        缺少功能时，扩展本类的公共方法。
    """
```

**关键属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `data` | `OrcaGymDataView` | 完整状态只读视图，替代 `_mjData` 读取 |
| `model` | `OrcaGymModel` | 模型结构信息（原样复用） |
| `sim_config` | `SimConfig` | 求解器参数配置，替代 `opt.*` 直接访问 |
| `ctrl` | `np.ndarray` | 控制数组 |
| `frame_skip` | `int` | 每次 `step()` 的物理步进数 |
| `dt` | `float` | 单步物理时间 |

### 4.2 OrcaGymEuler — 仿真核心 Facade

组合仿真子组件，向 `OrcaGymEulerEnv` 提供仿真操作接口。持有 `MuJoCoSimCore`、`OrcaStudioBridge`、`ModelRegistry`、`SimConfig`、`EulerOrchestrator`。**不暴露** `_mjModel`/`_mjData`，依赖 `_` 前缀约定 + ruff SLF001 静态检查，通过 `__dir__` 控制 IDE 自动补全只显示公共 API。

```python
class OrcaGymEuler:
    """双引擎编排核心。

    ┌─────────────────────────────────────────────────────────────┐
    │  API 契约：你不应直接访问 _mjData / _mjModel。              │
    │  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView）        │
    │  写入外力 → 使用 env.apply_body_force()                     │
    │  配置求解器 → 使用 env.sim_config                           │
    │  缺少功能时 → 扩展 OrcaGymEulerEnv 公共方法                 │
    └─────────────────────────────────────────────────────────────┘
    """
```

### 4.3 MuJoCoSimCore — 仿真核心

持有 `_mjModel`/`_mjData`，执行 `mj_step`/`mj_forward`/`set_ctrl` 等纯 MuJoCo 操作。`_mjModel`/`_mjData` 只存在于此类内部，不对外暴露。

```python
class MuJoCoSimCore:
    def __init__(self):
        self._mjModel: mujoco.MjModel | None = None
        self._mjData: mujoco.MjData | None = None

    def init_simulation(self, model_xml_path: str) -> None: ...
    def step(self, nstep: int) -> None: ...
    def forward(self) -> None: ...
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...
    def sync_to_view(self, view: OrcaGymDataView) -> None: ...
    def apply_body_force(self, body_id: int, force: np.ndarray, torque: np.ndarray) -> None: ...
```

### 4.4 OrcaStudioBridge — Studio 集成

处理与 OrcaStudio 的 gRPC 交互，包括渲染、视频保存、物体操作等。**依赖反转**设计：不持有 `_mjData`，通过接收数据参数实现解耦；不碰 `mj_step`，只负责通信和场景同步。

```python
class OrcaStudioBridge:
    def __init__(self, stub: GrpcServiceStub | None): ...
    async def render(self, qpos: np.ndarray, sim_time: float) -> None: ...
    async def load_model_xml(self) -> str: ...
    async def begin_save_video(self, path: str, mode: CaptureMode) -> None: ...
    async def stop_save_video(self) -> None: ...
    async def get_current_frame(self) -> int: ...
    async def get_body_manipulation_anchored(self) -> tuple: ...
    async def get_body_manipulation_movement(self) -> dict: ...
```

### 4.5 ModelRegistry — 模型注册

构建 `OrcaGymModel`/`OrcaGymData`，提供 `query_all_*` 等模型信息查询。`OrcaGymModel` 是成功抽象，原样复用，并扩展了缺失的模型结构查询。

```python
class ModelRegistry:
    def __init__(self, mj_model: mujoco.MjModel): ...
    def build_orca_gym_model(self) -> OrcaGymModel: ...
    def build_orca_gym_data(self) -> OrcaGymData: ...
    def body_subtree_mass(self, body_name: str) -> float: ...
    def equality_data_width(self) -> int: ...
    def equality_object_ids(self, eq_idx: int) -> tuple[int, int]: ...
```

### 4.6 SimConfig — 求解器配置

提供 typed 的 MuJoCo 求解器参数读写接口，替代 `_mjModel.opt.*` 直接访问。覆盖所有用户可访问的 `opt` 字段，修改在下次 `mj_step` 时生效。

```python
class SimConfig:
    """MuJoCo 求解器参数配置。替代直接访问 _mjModel.opt.*。修改在下次 mj_step 时生效。"""

    @property
    def timestep(self) -> float: ...
    @timestep.setter
    def timestep(self, value: float): ...

    @property
    def integrator(self) -> int: ...
    @integrator.setter
    def integrator(self, value: int): ...

    @property
    def iterations(self) -> int: ...
    @iterations.setter
    def iterations(self, value: int): ...

    @property
    def gravity(self) -> np.ndarray: ...
    @gravity.setter
    def gravity(self, value: np.ndarray): ...

    # ... 覆盖 opt 的所有用户可访问字段 ...

    def load_from_dict(self, config: dict) -> None: ...
```

**迁移映射**：

| 旧代码 | 新代码 |
|--------|--------|
| `gym._mjModel.opt.timestep = 0.002` | `env.sim_config.timestep = 0.002` |
| `gym._mjModel.opt.iterations = 100` | `env.sim_config.iterations = 100` |
| `gym._mjModel.opt.integrator = 0` | `env.sim_config.integrator = 0` |
| 30 行 `opt.*` 设置 | `env.sim_config.load_from_dict({...})` |

### 4.7 OrcaGymDataView — 完整状态视图

提供 MuJoCo 状态的完整只读视图，替代直接访问 `_mjData`。覆盖所有你需要读取的字段（不仅是原 `OrcaGymData` 的 5 个）。通过方法提供 body/site/geom 属性查询，你按名称访问，不需要知道 id。

```python
class OrcaGymDataView:
    """MuJoCo 状态的完整只读视图。

    替代直接访问 _mjData。所有字段在 update_data() 后保证一致。
    你永远不需要访问 _mjData。

    如果需要此视图未提供的字段，请在 OrcaGymDataView 中扩展，
    不要通过 env._gym._sim._mjData 绕道访问。
    """

    # --- 基本状态 ---
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    qfrc_bias: np.ndarray
    time: float

    # --- 扩展字段 ---
    xfrc_applied: np.ndarray       # 只读视图（写入用 apply_body_force）
    actuator_force: np.ndarray     # 执行器力
    contact: list                  # 接触列表

    def body_xpos(self, body_name: str) -> np.ndarray: ...
    def body_xquat(self, body_name: str) -> np.ndarray: ...
    def body_xmat(self, body_name: str) -> np.ndarray: ...
    def body_cvel(self, body_name: str) -> np.ndarray: ...
    def body_subtree_mass(self, body_name: str) -> float: ...

    def site_xpos(self, site_name: str) -> np.ndarray: ...
    def site_xmat(self, site_name: str) -> np.ndarray: ...
```

**迁移映射**：

| 旧代码 | 新 API |
|--------|--------|
| `gym._mjData.qpos` | `env.data.qpos` |
| `gym._mjData.body(id).xpos` | `env.data.body_xpos(name)` |
| `gym._mjData.cvel[id]` | `env.data.body_cvel(name)` |
| `gym._mjData.xpos[body_id, 2]` | `env.data.body_xpos(name)[2]` |
| `gym._mjData.time` | `env.data.time` |

### 4.8 EulerOrchestrator — Euler 耦合（占位）

编排 Euler 非刚体求解器与 MuJoCo 刚体求解器的耦合步进。当前为占位组件，`OrcaGymEuler` 的 `_euler` 字段为 `None`，`OrcaGymEulerEnv` 表现为纯 MuJoCo 环境。具体设计后续单独文档论述。

```python
class EulerOrchestrator:
    """Euler 非刚体求解器编排（占位，后续设计）。"""

    def euler_step(self, dt: float) -> None:
        """Euler 非刚体求解器步进。"""
        raise NotImplementedError("Euler 耦合编排待后续设计")

    def notify_external_force(self, body_name: str, force: np.ndarray, torque: np.ndarray) -> None:
        """通知 Euler 有外力注入（用于耦合一致性）。"""
        raise NotImplementedError
```

### 4.9 OrcaGymEnvMixin — 环境公共方法混入

抽取 `OrcaGymLocalEnv`/`OrcaGymBaseEnv` 中与仿真引擎无关的公共方法，供 `OrcaGymEulerEnv` 和 `OrcaGymLocalEnv` 共享。

```python
class OrcaGymEnvMixin:
    """OrcaGym 环境公共方法 Mixin。

    提供名称空间解析、动作/观测空间生成、reset 编排等方法。
    不定义 __init__，不持有状态，子类自行初始化 _agent_names 等字段。
    """

    # --- 名称空间解析（自动添加 agent 前缀）---
    def body(self, name: str, agent_id: int = None) -> str: ...
    def joint(self, name: str, agent_id: int = None) -> str: ...
    def actuator(self, name: str, agent_id: int = None) -> str: ...
    def site(self, name: str, agent_id: int = None) -> str: ...
    def mocap(self, name: str, agent_id: int = None) -> str: ...
    def sensor(self, name: str, agent_id: int = None) -> str: ...

    # --- 空间生成 ---
    def generate_action_space(self, bounds: np.ndarray) -> Space: ...
    def generate_observation_space(self, obs: Union[Dict, np.ndarray]) -> Space: ...

    # --- reset 编排 ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None): ...
    def set_seed_value(self, seed: int = None) -> list: ...
    def _get_reset_info(self) -> Dict[str, float]: ...

    # --- 辅助 ---
    def _name_with_agent0(self, name: str) -> str: ...
    def _name_with_agent(self, agent_id: int, name: str) -> str: ...
    @property
    def agent_num(self) -> int: ...
```

**使用方式**：

```python
# Euler 体系
class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env):
    def __init__(self, ...):
        self._agent_names = agent_names
        self._gym = OrcaGymEuler(...)
        # Mixin 方法可直接使用
        ...

# Local 体系（可选重构，不强制）
class OrcaGymLocalEnv(OrcaGymEnvMixin, gym.Env):
    ...
```

---

## 5. API 使用契约

### 5.1 契约层级

| 层级 | 含义 | 违反后果 |
|------|------|---------|
| **L1 公共 API** | `__dir__` 暴露的方法和属性，你应该使用 | 正常工作 |
| **L2 内部组件** | `_gym`/`_sim`/`_studio` 等，你不应访问 | ruff SLF001 报警 |
| **L3 引擎内部** | `_mjModel`/`_mjData`，你绝不应访问 | ruff SLF001 报警 |

### 5.2 状态读取

**规则**：所有状态读取通过 `env.data`（`OrcaGymDataView`）或 `env.query_*()` 方法。`env.data` 在 `do_simulation()` 返回后、`mj_forward()` 返回后保证一致。`env.data` 是只读视图，写入操作必须通过显式方法。

```python
# ✅ 正确
qpos = env.data.qpos
body_pos = env.data.body_xpos("link1")

# ❌ 错误
qpos = env._gym._sim._mjData.qpos  # ruff SLF001 报警
```

### 5.3 状态写入

**规则**：所有状态写入通过显式方法，不直接操作 MuJoCo 数据结构。外力注入通过 `apply_body_force()`，不直接写 `xfrc_applied`。写入操作后若需立即读取一致状态，须调用 `mj_forward()`。

```python
# ✅ 正确
env.set_joint_qpos({"joint1": np.array([0.5])})
env.apply_body_force("link1", force, torque)
env.mj_forward()

# ❌ 错误
env._gym._sim._mjData.xfrc_applied[body_id, :3] = force  # ruff SLF001 报警
```

### 5.4 仿真步进

| 方法 | 职责 | Euler 耦合 | 适用场景 |
|------|------|-----------|---------|
| `do_simulation(ctrl, n)` | 标准步进 | 有（未来） | 大多数 Env 的 `step()` |
| `mj_step(n)` | 纯 MuJoCo 步进 | 无 | 需要精细控制时序的高级用户 |
| `mj_forward()` | 前向计算 | 无 | 状态设置后更新派生量 |

两种使用模式：

```python
# 模式 A（推荐，含 Euler 耦合）
env.do_simulation(ctrl, self.frame_skip)

# 模式 B（纯 MuJoCo，无耦合）
for _ in range(self.frame_skip):
    env.set_ctrl(torques)
    env.mj_step(1)
    env._update_data()
```

> 模式 B 当前与 OrcaGymLocalEnv 行为一致。若未来需要 Euler 耦合，模式 B 用户必须改用模式 A。

### 5.5 求解器配置

所有 `opt.*` 参数通过 `env.sim_config` 读写，配置修改在下次 `mj_step` 时生效。

```python
# ✅ 正确
env.sim_config.timestep = 0.002
env.sim_config.iterations = 100
env.sim_config.load_from_dict({"integrator": 0, "iterations": 100})

# ❌ 错误
env._gym._sim._mjModel.opt.timestep = 0.002  # ruff SLF001 报警
```

### 5.6 名称空间

所有名称通过 `env.joint()`/`env.body()`/`env.site()`/`env.actuator()`/`env.sensor()` 解析，自动添加 agent 前缀。这部分 API 与 OrcaGymLocalEnv 完全一致，零改动迁移。

```python
joint_name = env.joint("joint1")  # → "agent_name/joint1"
body_name = env.body("object")
```

### 5.7 完整公共 API 清单

| 类别 | API |
|------|-----|
| **状态读取** | `data`（OrcaGymDataView）, `model`（OrcaGymModel）, `ctrl`, `frame_skip`, `dt`, `realtime_step` |
| **仿真控制** | `do_simulation(ctrl, n)`, `mj_step(n)`, `mj_forward()` |
| **状态查询** | `query_joint_qpos/qvel/qacc/offsets/lengths()`, `query_site_pos_and_quat/mat/xvalp_xvalr()`, `query_actuator_torques()`, `query_sensor_data()`, `query_contact_simple()`, `get_body_xpos_xmat_xquat()` |
| **状态设置** | `set_joint_qpos/qvel()`, `set_mocap_pos_and_quat()`, `set_geom_friction()`, `apply_body_force()`, `clear_body_force()`, `clear_all_forces()` |
| **等式约束原语（无状态，L1）** | `equality_find_slot_by_body(body_name)`, `equality_constraint(slot)`, `equality_update(slot, **fields, forward=True)` |
| **求解器配置** | `sim_config`（SimConfig） |
| **名称空间** | `joint()`, `body()`, `site()`, `actuator()`, `sensor()` |
| **Studio 交互** | `render()`, `begin_save_video()`, `stop_save_video()`, `get_current_frame()`, `get_frame_png()` |
| **生命周期** | `initialize_simulation()`, `initialize_grpc()`, `pause_simulation()`, `close()` |

> **Studio UI 抓取为内部 API**：原 `anchor_actor()` / `release_body_anchored()` / `do_body_manipulation()` 按 P6 原则改为 `_` 前缀内部方法，由 `render()` 内部驱动，不进入公共 API。程序化体操作应使用等式约束无状态原语自行实现。

---

## 6. 封装与隔离

### 6.1 机制总览

本架构通过多层引导让"正确方式"成为阻力最小的路径。`OrcaGymEulerEnv` 直接继承 `gym.Env`，不创建 `gym`/`stub`/`channel` 属性，Python 原生即拒绝访问；其余内部对象依赖 `_` 前缀约定 + ruff SLF001 静态检查 + AGENTS.md AI 行为约束：

| 机制 | 实现 | 效果 |
|------|------|------|
| **Python 原生属性不存在** | `OrcaGymEulerEnv` 直接继承 `gym.Env`，`__init__` 中只赋值 `_gym`/`_stub`/`_channel` | `env.gym`/`env.stub`/`env.channel` 抛 `AttributeError` |
| **ruff SLF001 静态检查** | 配置 `ruff check --select SLF001`，扫描外部访问 `_` 前缀属性的代码 | 提交前/CI 阶段检测穿墙访问 |
| **AGENTS.md AI 约束** | 每个自研仓库根目录配置 `AGENTS.md`，明文禁止 AI 使用 `_` 前缀属性 | 从输入端约束 AI 代码生成行为 |
| **`__dir__` 控制** | Env/Gym/DataView 实现 `__dir__`，只暴露公共 API | IDE 自动补全引导正确路径 |
| **DataView 兜底** | `OrcaGymDataView.__getattr__` 缺字段时引导扩展 | 缺功能时引导扩展而非绕道 |
| **类型标注** | 公共方法返回 typed 对象，不返回 `mujoco.MjData` | AI 代码生成走正确路径 |
| **docstring 契约** | 类文档明确列出正确用法和禁止事项 | 阅读 API 即知契约 |
| **路径深度** | `_mjData` 在 `env._gym._sim._mjData` 三层之下 | 天然屏障 |

### 6.2 隔离效果对比

| 场景 | 触发机制 | 你/AI 看到的 |
|------|---------|-------------|
| AI 生成 `env._mjData.qpos` | ruff SLF001 | 提交前报警：用 `env.data.qpos` |
| AI 生成 `env._gym._mjData` | ruff SLF001 | 提交前报警：用 `env.data` |
| AI 生成 `env._mjModel.opt.iterations` | ruff SLF001 | 提交前报警：用 `env.sim_config.iterations` |
| AI 在 IDE 中补全 `env.` | `__dir__` 控制 | 只看到公共 API |
| AI 阅读 class docstring | 类型标注 + 契约 | 知道正确用法和禁止事项 |
| AI 未执行 ruff 直接提交 | AGENTS.md 约束 + CI 门禁 | CI 拒绝合并 |

### 6.3 与旧体系的隔离强度对比

| 系统 | 绕道路径 | 层数 | 内部组件是否可见 | 静态检查 |
|------|---------|------|----------------|---------|
| OrcaGymLocalEnv | `env.gym._mjData` | 2 | `gym` 是公共属性 | 无 |
| OrcaGymEulerEnv | `env._gym._sim._mjData` | 3 | `__dir__` 不列出 | ruff SLF001 报警 |

---

## 7. 步进编排

### 7.1 `do_simulation` 内部流程

```python
def do_simulation(self, ctrl: np.ndarray, n_frames: int):
    """标准仿真步进（含 Euler 耦合）。

    契约:
    - 设置控制输入 → 步进 n_frames 次 → 同步状态
    - 若 Euler 激活，每步刚体解算后插入 Euler 非刚体耦合
    - 步进完成后 self.data 保证一致
    """
    self._gym.set_ctrl(ctrl)
    if self._gym._euler is not None:
        for _ in range(n_frames):
            self._gym.mj_step(1)
            self._gym.euler_step(self._dt)
    else:
        self._gym.mj_step(n_frames)
    self._update_data()
```

### 7.2 两种使用模式

**模式 A（委托式，推荐）**：

```python
def step(self, action):
    torque = self._compute_torque(action)
    self.do_simulation(torque, self.frame_skip)
    obs = self._get_obs()
    return obs, reward, terminated, truncated, info
```

**模式 B（手动循环式）**：

```python
def step(self, action):
    for _ in range(self.frame_skip):
        torque = self._compute_torque(action)
        self.set_ctrl(torque)
        self.mj_step(nstep=1)
        self._update_data()
    obs = self._get_obs()
    return obs, reward, terminated, truncated, info
```

**契约**：模式 B 当前与 OrcaGymLocalEnv 行为一致（纯 MuJoCo）。若未来需要 Euler 耦合，模式 B 用户必须改用模式 A。

---

## 8. 迁移指南

### 8.1 迁移代价

| API 类别 | 兼容难度 | 说明 |
|---------|---------|------|
| 生命周期与属性 | 低 | `model`/`data`/`ctrl`/`frame_skip` 等原样提供 |
| 仿真步进（模式 A） | 低 | `do_simulation` 内部委托，签名一致 |
| 仿真步进（模式 B） | 中 | `mj_step(1)` 行为需注意无 Euler 耦合 |
| 状态查询 | 低 | `query_*` 方法原样复制 |
| 状态设置 | 低 | `set_*` 方法原样复制 + 新增 `apply_body_force` |
| 名称空间解析 | 低 | `joint()`/`body()`/`site()` 等原样提供 |
| `_mjData`/`_mjModel` 直接访问 | **低** | 有正式 API 替代，机械替换 |
| Studio 交互 | 低 | gRPC 逻辑原样复制 |

**总体**：约 70% 零改动，25% 机械替换，5% 设计调整。

### 8.2 83 处直接访问的替代方案

#### 读取类（→ OrcaGymDataView）

| 旧代码 | 新 API |
|--------|--------|
| `gym._mjData.qpos` | `env.data.qpos` |
| `gym._mjData.qvel` | `env.data.qvel` |
| `gym._mjData.body(id).xpos` | `env.data.body_xpos(name)` |
| `gym._mjData.cvel[id]` | `env.data.body_cvel(name)` |
| `gym._mjData.xpos[body_id, 2]` | `env.data.body_xpos(name)[2]` |
| `gym._mjData.time` | `env.data.time` |

#### 写入类（→ 显式方法）

| 旧代码 | 新 API |
|--------|--------|
| `gym._mjData.xfrc_applied[id, :3] = f` | `env.apply_body_force(name, f, tau)` |
| `gym._mjData.xfrc_applied[id].fill(0)` | `env.clear_body_force(name)` |
| `gym._mjData.eq_active[gi] = bool` | `env.set_equality_active(idx, active)` |

#### 配置类（→ SimConfig）

| 旧代码 | 新 API |
|--------|--------|
| `gym._mjModel.opt.timestep = 0.002` | `env.sim_config.timestep = 0.002` |
| `gym._mjModel.opt.iterations = 100` | `env.sim_config.iterations = 100` |
| `gym._mjModel.opt.integrator = 0` | `env.sim_config.integrator = 0` |
| `gym._mjModel.opt.gravity = ...` | `env.sim_config.gravity = ...` |
| 30 行 `opt.*` 设置 | `env.sim_config.load_from_dict({...})` |

#### 模型结构类（→ OrcaGymModel 扩展）

| 旧代码 | 新 API |
|--------|--------|
| `gym._mjModel.body_subtreemass[id]` | `env.model.body_subtree_mass(name)` |
| `gym._mjModel.eq_data.shape[1]` | `env.model.equality_data_width()` |
| `gym._mjModel.eq_obj1id[gi]` | `env.model.equality_object_ids(idx)` |
| `gym._mjModel.joint(i).name` | `env.model.joint_name_by_id(i)` |
| `gym._mjModel.njnt` | `env.model.njnt` |

### 8.3 迁移分类示例

**第一类：零改动（业务逻辑保持）**

关节查询、状态设置、名称空间、渲染、步进——API 签名完全一致。

**第二类：机械替换（`_mjData` → 正式 API）**

```python
# 迁移前
self.gym._mjData.xfrc_applied[body_id, :3] = force

# 迁移后
self.apply_body_force(body_name, force, torque)
```

**第三类：设计调整（少数情况）**

```python
# 迁移前：手动循环步进
for _ in range(self.frame_skip):
    self.set_ctrl(torques)
    self.mj_step(nstep=1)
    self.gym.update_data()

# 迁移后：若需 Euler 耦合，改用 do_simulation
self.do_simulation(torques, self.frame_skip)
```

---

## 9. 设计决策

### 9.1 为何直接继承 gym.Env + OrcaGymEnvMixin

`OrcaGymEulerEnv` 直接继承 `gym.Env`，通过 `OrcaGymEnvMixin` 共享公共方法，不继承 `OrcaGymBaseEnv` 也不继承 `OrcaGymLocalEnv`。

**理由**：
- `OrcaGymLocalEnv` 是上帝类，继承会继承所有职责耦合
- `OrcaGymLocal` 的 `_mjModel`/`_mjData` 暴露设计与 P2 原则冲突
- `OrcaGymBaseEnv` 的 `self.gym`/`self.model`/`self.data` 赋值与 Euler 体系直接冲突，需要 `__setattr__` 屏蔽等补丁机制
- 直接继承 `gym.Env` 后，`env.gym`/`env.stub`/`env.channel` 天然不存在（Python 原生 AttributeError），无需任何拦截机制
- `OrcaGymEnvMixin` 抽取名称空间、空间生成、reset 编排等与引擎无关的方法，避免代码重复

### 9.2 为何放弃 MuJoCoAdapter

不提供 `MuJoCoAdapter`（受控的 MuJoCo 句柄适配器）。

**理由**：原需求来自 robosuite 控制器等外部库需要直接操作 MuJoCo 对象，决定不再支持 robosuite 组件后该需求消失。放弃后设计更简洁——没有"逃生舱"，所有需求都通过扩展 Env/Gym 的公共方法解决。

### 9.3 为何 `_mjData`/`_mjModel` 放在 MuJoCoSimCore

`_mjModel`/`_mjData` 只存在于 `MuJoCoSimCore` 内部，`OrcaGymEuler` 和 `OrcaGymEulerEnv` 不持有引用。

**理由**：
- 增加绕道路径深度（`env._gym._sim._mjData` 三层）
- 职责内聚：MuJoCo 原生操作集中在 `MuJoCoSimCore`
- `OrcaGymEuler` 作为 Facade 只协调，不直接操作引擎数据

### 9.4 为何保留 `self.gym` 概念但重命名为 `_gym`

`OrcaGymEulerEnv` 持有 `_gym: OrcaGymEuler`，但不作为公共属性暴露。

**理由**：
- 保留分层结构（Env → Gym → SimCore）便于职责划分
- `_gym` 不在 `__dir__` 中列出，AI 难以发现
- 你通过 Env 的公共方法间接使用 Gym，不直接接触

---

## 10. 总结

本文档的核心要点：

1. **Facade + 职责内聚分解**替代上帝类，组件按职责划分为 `MuJoCoSimCore`/`OrcaStudioBridge`/`ModelRegistry`/`SimConfig`/`EulerOrchestrator`
2. **直接继承 `gym.Env` + `OrcaGymEnvMixin`**：不继承 `OrcaGymBaseEnv`，公共方法通过 Mixin 共享
3. **完备的公共 API 契约**覆盖所有合法 MuJoCo 操作需求，消除绕道理由
4. **多层封装隔离**（ruff SLF001 + AGENTS.md + Python 原生属性不存在 + `__dir__` + DataView 兜底 + 类型标注 + docstring）引导你和 AI 走正确路径
5. **步进编排契约**明确 `do_simulation`（含耦合）与 `mj_step`（纯 MuJoCo）的语义区分
6. **迁移策略**约 70% 零改动、25% 机械替换、5% 设计调整
