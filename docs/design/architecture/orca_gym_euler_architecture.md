# OrcaGym Euler 架构：新一代 Gym 组件设计

## 1. 文档定位

### 1.1 文档目标

本文论述 OrcaGym 体系中新一代 Gym 组件 `OrcaGymEulerEnv` + `OrcaGymEuler` 的架构设计，为未来的开发和演进指明方向。本文聚焦于**如何设计一个更优的 OrcaGym 组件作为未来主路径**，暂不涉及 MuJoCo + Euler 多求解器耦合与编排的具体实现（该部分后续单独设计）。

### 1.2 核心论点

> **`OrcaGymEulerEnv` + `OrcaGymEuler` 采用 Facade + 职责内聚分解的设计，替代 `OrcaGymLocalEnv` + `OrcaGymLocal` 的上帝类模式。通过完备的公共 API 契约和多层封装隔离机制，引导用户和 AI 走正确路径，避免直接访问 MuJoCo 内部数据结构。**

### 1.3 适用范围

| 范围 | 说明 |
|------|------|
| **本文涵盖** | OrcaGymEulerEnv / OrcaGymEuler 的组件设计、API 契约、封装隔离机制、迁移策略 |
| **本文不涵盖** | Euler 非刚体求解器与 MuJoCo 的耦合编排（EulerOrchestrator 仅作为占位组件出现） |

---

## 2. 设计动机：为何重构 OrcaGym 组件

### 2.1 现有体系的问题

`OrcaGymLocalEnv` + `OrcaGymLocal` 作为当前主用路径，存在三类结构性缺陷：

| 缺陷类型 | 表现 | 后果 |
|---------|------|------|
| **上帝类** | `OrcaGymLocal` 单类承担仿真核心、Studio 集成、模型注册、求解器配置、物体操作等所有职责 | 难以维护、难以扩展、职责耦合 |
| **API 不完备** | `OrcaGymData` 只缓存 5 个字段（qpos/qvel/qacc/qfrc_bias/time），缺 xfrc_applied/cvel/contact 等 | 用户被迫绕道 `gym._mjData` 直接访问 |
| **封装泄漏** | `_mjModel`/`_mjData` 作为公共属性暴露，`self.gym` 既是内部组件又作外部库句柄 | 83 处直接访问，封装形同虚设 |

### 2.2 用户代码现状

对 OrcaPlayground 和 OrcaManipulation 两个仓库的分析显示：

- **17 个直接子类** + **4 个间接子类**继承自 `OrcaGymLocalEnv`
- **83 处**直接访问 `gym._mjData` / `gym._mjModel`
- 典型绕道场景：外力注入（`xfrc_applied`）、求解器配置（`opt.*`）、body 属性查询（`body_subtreemass`/`cvel`）、等式约束结构访问（`eq_data`）

用户绕道不是因为不懂封装，而是**封装没有覆盖他们的需求**。

### 2.3 设计目标

| 目标 | 含义 |
|------|------|
| **G1 干净的新代码** | 新开发的代码可以干净、高效、遵照契约地使用 Euler 体系 |
| **G2 平滑的老代码迁移** | 业务逻辑（MuJoCo 编程概念、数据处理逻辑）保持原使用方法；明显不合理的使用需改为新接口 |
| **G3 封装隔离** | 用户难以绕过 API 直接裸访问 MuJoCo 数据，避免未来 MuJoCo 与 Euler 配合出现问题 |

---

## 3. 核心设计原则

### 3.1 五大原则

| 原则 | 含义 | 对比 OrcaGymLocalEnv |
|------|------|---------------------|
| **P1 完备性** | 公共 API 覆盖所有合法的 MuJoCo 操作需求，用户无理由绕道 | 大量缺口迫使绕道 |
| **P2 不暴露引擎内部** | `_mjModel`/`_mjData` 不作为公共属性暴露，只存在于 `MuJoCoSimCore` 内部 | 直接暴露 |
| **P3 状态一致性契约** | 任何写操作后，`self.data` 保证一致；任何读操作都走 `self.data` 或显式查询 | `self.data` 与 `_mjData` 双轨制 |
| **P4 力应用可追踪** | 外力注入通过显式方法，未来 Euler 耦合器可感知 | `xfrc_applied` 直接写，无感知 |
| **P5 职责内聚** | 按职责内聚划分模块，一组方法因同一原因变化、共享同一组数据 | 上帝类 |

### 3.2 设计模式选型

| 模式 | 应用位置 | 解决的问题 |
|------|---------|-----------|
| **Facade** | `OrcaGymEulerEnv` / `OrcaGymEuler` | 组合多个子组件，提供统一 API，避免上帝类 |
| **组合优于继承** | Env 持有 Gym，Gym 持有子组件 | 避免继承链腐化，职责可独立演进 |
| **策略模式** | `EulerOrchestrator`（可选） | 通过 None 检查切换有无 Euler 的策略 |
| **依赖反转** | `OrcaStudioBridge` 不持有 mjData，通过接收数据参数实现解耦 | Studio 集成与仿真核心解耦 |
| **只读视图** | `OrcaGymDataView` | 提供完整状态读取，禁止写入 |

---

## 4. 架构总览

### 4.1 整体结构

```
gym.Env
  └── OrcaGymBaseEnv                          (保留，提供 step/reset 框架)
        └── OrcaGymEulerEnv (新)              (Facade + 契约执行者)
              │
              │   组合（非继承）
              ├── _gym: OrcaGymEuler           (仿真核心 Facade)
              │     ├── _sim: MuJoCoSimCore    # 持有 _mjModel/_mjData（不对外暴露）
              │     ├── _studio: OrcaStudioBridge  # gRPC 集成
              │     ├── _registry: ModelRegistry  # 模型信息
              │     ├── _opt: SimConfig        # 求解器配置（typed）
              │     └── _euler: EulerOrchestrator | None  # Euler 耦合（占位，后续设计）
              │
              │   公共 API（用户面向）
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

### 4.2 与 OrcaGymLocalEnv 体系的对比

| 维度 | OrcaGymLocalEnv + OrcaGymLocal | OrcaGymEulerEnv + OrcaGymEuler |
|------|-------------------------------|-------------------------------|
| 类结构 | 上帝类，单类承担所有职责 | Facade + 职责内聚分解 |
| `_mjModel`/`_mjData` | 公共属性，83 处直接访问 | 内部组件，多层隔离 |
| `OrcaGymData` | 5 字段缓存，不完整 | `OrcaGymDataView` 完整只读视图 |
| 求解器配置 | 无接口，绕道 `opt.*` | `SimConfig` typed 配置 |
| 外力注入 | 直接写 `xfrc_applied` | `apply_body_force()` 显式方法 |
| 继承体系 | 继承自腐化的 `OrcaGymBase` 链 | 独立类，不继承 `OrcaGymLocal` |

### 4.3 与 OrcaGymLocalEnv 体系的共存策略

> **原有 `OrcaGymBase` → Remote / Local / Warp 继承体系不动**，让原有系统继续运行不受影响。`OrcaGymEulerEnv` 是独立的新类，不继承 `OrcaGymLocalEnv`，与 `OrcaGymLocal` 长期共存。

迁移路径：`OrcaGymLocal` 像 `Remote` 一样最终将被废弃，届时原有体系直接删除即可。

---

## 5. 组件设计

### 5.1 OrcaGymEulerEnv — 环境 Facade

**职责**：作为 Gymnasium `Env` 的实现，组合 `OrcaGymEuler` 仿真核心，向用户代码暴露统一 API。

**设计契约**：

```python
class OrcaGymEulerEnv(OrcaGymBaseEnv):
    """OrcaGym Euler 双引擎环境。

    使用契约:
        读取状态:   env.data.qpos / env.data.body_xpos(name) / env.query_*()
        写入状态:   env.set_joint_qpos() / env.apply_body_force()
        仿真步进:   env.do_simulation(ctrl, n_frames)
        求解器配置: env.sim_config.timestep = 0.002

    禁止:
        不要访问 env._gym._sim._mjData 或任何内部 MuJoCo 对象。
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

**关键方法**：见第 6 章 API 契约。

### 5.2 OrcaGymEuler — 仿真核心 Facade

**职责**：组合仿真子组件，向 `OrcaGymEulerEnv` 提供仿真操作接口。

**设计要点**：

- 持有 `MuJoCoSimCore`、`OrcaStudioBridge`、`ModelRegistry`、`SimConfig`、`EulerOrchestrator`
- **不暴露** `_mjModel`/`_mjData`，通过 `__getattr__` 拦截引导性错误
- 通过 `__dir__` 控制可见性，IDE 自动补全只显示公共 API

```python
class OrcaGymEuler:
    """双引擎编排核心。

    ┌─────────────────────────────────────────────────────────────┐
    │  API 契约：用户不应直接访问 _mjData / _mjModel。             │
    │  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView）        │
    │  写入外力 → 使用 env.apply_body_force()                     │
    │  配置求解器 → 使用 env.sim_config                           │
    │  缺少功能时 → 扩展 OrcaGymEulerEnv 公共方法                 │
    └─────────────────────────────────────────────────────────────┘
    """
```

### 5.3 MuJoCoSimCore — 仿真核心

**职责**：持有 `_mjModel`/`_mjData`，执行 `mj_step`/`mj_forward`/`set_ctrl` 等纯 MuJoCo 操作。

**设计要点**：

- `_mjModel`/`_mjData` 只存在于此类内部
- 不对外暴露这两个属性
- 提供步进、前向、控制设置等原子操作
- `sync_to_view()` 方法将内部状态同步到 `OrcaGymDataView`

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

### 5.4 OrcaStudioBridge — Studio 集成

**职责**：处理与 OrcaStudio 的 gRPC 交互，包括渲染、视频保存、物体操作等。

**设计要点**：

- **依赖反转**：不持有 `_mjData`，通过接收数据参数实现解耦
- 不碰 `mj_step`，只负责通信和场景同步

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

### 5.5 ModelRegistry — 模型注册

**职责**：构建 `OrcaGymModel`/`OrcaGymData`，提供 `query_all_*` 等模型信息查询。

**设计要点**：

- `OrcaGymModel` 是成功抽象，原样复用
- 扩展缺失的模型结构查询（如 `body_subtree_mass`、`equality_data_width`）

```python
class ModelRegistry:
    def __init__(self, mj_model: mujoco.MjModel): ...
    def build_orca_gym_model(self) -> OrcaGymModel: ...
    def build_orca_gym_data(self) -> OrcaGymData: ...
    def body_subtree_mass(self, body_name: str) -> float: ...
    def equality_data_width(self) -> int: ...
    def equality_object_ids(self, eq_idx: int) -> tuple[int, int]: ...
```

### 5.6 SimConfig — 求解器配置

**职责**：提供 typed 的 MuJoCo 求解器参数读写接口，替代 `_mjModel.opt.*` 直接访问。

**设计要点**：

- 覆盖用户可访问的所有 `opt` 字段
- 修改在下次 `mj_step` 时生效
- 支持 `load_from_dict()` 批量设置

```python
class SimConfig:
    """MuJoCo 求解器参数配置。

    替代直接访问 _mjModel.opt.*。
    修改在下次 mj_step 时生效。
    """

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

### 5.7 OrcaGymDataView — 完整状态视图

**职责**：提供 MuJoCo 状态的完整只读视图，替代直接访问 `_mjData`。

**设计要点**：

- 覆盖所有用户需要读取的字段（不仅是原 `OrcaGymData` 的 5 个）
- 通过方法提供 body/site/geom 属性查询，用户按名称访问，不需要知道 id
- `__getattr__` 兜底，缺字段时引导扩展

```python
class OrcaGymDataView:
    """MuJoCo 状态的完整只读视图。

    替代直接访问 _mjData。所有字段在 update_data() 后保证一致。
    用户永远不需要访问 _mjData。

    如果需要此视图未提供的字段，请在 OrcaGymDataView 中扩展，
    不要通过 env._gym._sim._mjData 绕道访问。
    """

    # --- 基本状态（原 OrcaGymData 已有）---
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    qfrc_bias: np.ndarray
    time: float

    # --- 扩展：覆盖用户绕道访问的字段 ---
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

### 5.8 EulerOrchestrator — Euler 耦合（占位）

**职责**：编排 Euler 非刚体求解器与 MuJoCo 刚体求解器的耦合步进。

**当前状态**：占位组件，具体设计后续单独文档论述。当前阶段 `OrcaGymEuler` 的 `_euler` 字段为 `None`，`OrcaGymEulerEnv` 表现为纯 MuJoCo 环境。

```python
class EulerOrchestrator:
    """Euler 非刚体求解器编排（占位，后续设计）。

    当前阶段不实现具体耦合逻辑。
    """

    def euler_step(self, dt: float) -> None:
        """Euler 非刚体求解器步进。"""
        raise NotImplementedError("Euler 耦合编排待后续设计")

    def notify_external_force(self, body_name: str, force: np.ndarray, torque: np.ndarray) -> None:
        """通知 Euler 有外力注入（用于耦合一致性）。"""
        raise NotImplementedError
```

---

## 6. API 契约

### 6.1 契约层级

| 层级 | 含义 | 违反后果 |
|------|------|---------|
| **L1 公共 API** | `__dir__` 暴露的方法和属性，用户应使用 | 正常工作 |
| **L2 内部组件** | `_gym`/`_sim`/`_studio` 等，用户不应访问 | `__getattr__` 拦截，引导性错误 |
| **L3 引擎内部** | `_mjModel`/`_mjData`，用户绝不应访问 | `__getattr__` 拦截，引导性错误 |

### 6.2 状态读取契约

**规则 R1**：所有状态读取通过 `env.data`（`OrcaGymDataView`）或 `env.query_*()` 方法。

**规则 R2**：`env.data` 在以下时机保证一致：
- `do_simulation()` 返回后
- `mj_forward()` 返回后
- `set_joint_qpos()`/`set_joint_qvel()`/`set_mocap_pos_and_quat()` 后调用 `mj_forward()` 后

**规则 R3**：`env.data` 是只读视图，写入操作必须通过显式方法。

```python
# 正确
qpos = env.data.qpos
body_pos = env.data.body_xpos("link1")

# 错误（违反 R1）
qpos = env._gym._sim._mjData.qpos  # __getattr__ 拦截
```

### 6.3 状态写入契约

**规则 W1**：所有状态写入通过显式方法，不直接操作 MuJoCo 数据结构。

**规则 W2**：外力注入通过 `apply_body_force()`，不直接写 `xfrc_applied`。

**规则 W3**：写入操作后若需立即读取一致状态，须调用 `mj_forward()`。

```python
# 正确
env.set_joint_qpos({"joint1": np.array([0.5])})
env.apply_body_force("link1", force, torque)
env.mj_forward()

# 错误（违反 W1/W2）
env._gym._sim._mjData.xfrc_applied[body_id, :3] = force  # __getattr__ 拦截
```

### 6.4 仿真步进契约

**规则 S1**：`do_simulation(ctrl, n_frames)` 是标准步进入口，含 Euler 耦合（未来）。

**规则 S2**：`mj_step(n)` 是纯 MuJoCo 步进，不含 Euler 耦合。

**规则 S3**：两种步进模式必须兼容：
- 模式 A（委托式）：`env.do_simulation(ctrl, self.frame_skip)`
- 模式 B（手动循环）：`env.set_ctrl(torques); env.mj_step(1); env._update_data()`

**规则 S4**：模式 B 用户若需 Euler 耦合，必须改用 `do_simulation()`。

```python
# 模式 A（推荐，含 Euler 耦合）
env.do_simulation(ctrl, self.frame_skip)

# 模式 B（纯 MuJoCo，无耦合）
for _ in range(self.frame_skip):
    env.set_ctrl(torques)
    env.mj_step(1)
    env._update_data()
```

### 6.5 求解器配置契约

**规则 C1**：所有 `opt.*` 参数通过 `env.sim_config` 读写。

**规则 C2**：配置修改在下次 `mj_step` 时生效。

```python
# 正确
env.sim_config.timestep = 0.002
env.sim_config.iterations = 100
env.sim_config.load_from_dict({"integrator": 0, "iterations": 100})

# 错误（违反 C1）
env._gym._sim._mjModel.opt.timestep = 0.002  # __getattr__ 拦截
```

### 6.6 名称空间契约

**规则 N1**：所有名称通过 `env.joint()`/`env.body()`/`env.site()`/`env.actuator()`/`env.sensor()` 解析，自动添加 agent 前缀。

```python
# 正确
joint_name = env.joint("joint1")  # → "agent_name/joint1"
body_name = env.body("object")

# 这部分 API 与 OrcaGymLocalEnv 完全一致，零改动迁移
```

### 6.7 完整公共 API 清单

| 类别 | API |
|------|-----|
| **状态读取** | `data`（OrcaGymDataView）, `model`（OrcaGymModel）, `ctrl`, `frame_skip`, `dt`, `realtime_step` |
| **仿真控制** | `do_simulation(ctrl, n)`, `mj_step(n)`, `mj_forward()` |
| **状态查询** | `query_joint_qpos/qvel/qacc/offsets/lengths()`, `query_site_pos_and_quat/mat/xvalp_xvalr()`, `query_actuator_torques()`, `query_sensor_data()`, `query_contact_simple()`, `get_body_xpos_xmat_xquat()` |
| **状态设置** | `set_joint_qpos/qvel()`, `set_mocap_pos_and_quat()`, `update_equality_constraints()`, `set_geom_friction()`, `apply_body_force()`, `clear_body_force()`, `clear_all_forces()` |
| **求解器配置** | `sim_config`（SimConfig） |
| **名称空间** | `joint()`, `body()`, `site()`, `actuator()`, `sensor()` |
| **Studio 交互** | `render()`, `begin_save_video()`, `stop_save_video()`, `get_current_frame()`, `get_frame_png()`, `anchor_actor()`, `release_body_anchored()` |
| **生命周期** | `initialize_simulation()`, `initialize_grpc()`, `pause_simulation()`, `close()` |

---

## 7. 封装与隔离机制

### 7.1 机制总览

Python 无法真正阻止属性访问，但通过多层引导让"正确方式"成为阻力最小的路径：

| 机制 | 实现 | 效果 |
|------|------|------|
| **M1 `__getattr__` 拦截** | Env/Gym 层不存储 `_mjData`/`_mjModel`，访问时触发引导性错误 | AI/用户立即知道正确替代方案 |
| **M2 `__dir__` 控制** | 只暴露公共 API，不列出 `_gym`/`_sim` | IDE 自动补全引导正确路径 |
| **M3 DataView 兜底** | `OrcaGymDataView.__getattr__` 缺字段时引导扩展 | 缺功能时引导扩展而非绕道 |
| **M4 类型标注** | 公共方法返回 typed 对象，不返回 `mujoco.MjData` | AI 代码生成走正确路径 |
| **M5 docstring 契约** | 类文档明确列出正确用法和禁止事项 | 阅读 API 即知契约 |
| **M6 路径深度** | `_mjData` 在 `env._gym._sim._mjData` 三层之下 | 天然屏障，AI 难以猜到 |

### 7.2 `__getattr__` 拦截实现

```python
class OrcaGymEuler:
    _BLOCKED_ATTRS = frozenset({
        "_mjData", "_mjModel", "mj_data", "mj_model",
        "_mj_data", "_mj_model", "mjData", "mjModel",
    })

    def __getattr__(self, name: str):
        if name in self._BLOCKED_ATTRS:
            raise AttributeError(
                f"'{type(self).__name__}' 不直接暴露 '{name}'。\n"
                f"  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView），如 env.data.qpos\n"
                f"  写入外力 → 使用 env.apply_body_force(body_name, force, torque)\n"
                f"  配置求解器 → 使用 env.sim_config\n"
                f"  查询 body 属性 → 使用 env.data.body_xpos(name) 等\n"
                f"  如果以上 API 都不满足需求，请在 OrcaGymEulerEnv 中扩展新方法，"
                f"不要直接访问内部 MuJoCo 对象。"
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

### 7.3 `__dir__` 控制实现

```python
class OrcaGymEuler:
    def __dir__(self):
        return [
            # 状态
            "data", "model", "sim_config", "ctrl",
            # 仿真控制
            "do_simulation", "mj_step", "mj_forward",
            # 状态查询
            "query_joint_qpos", "query_joint_qvel", "query_site_pos_and_quat",
            # 状态设置
            "set_joint_qpos", "set_mocap_pos_and_quat", "apply_body_force",
            # ... 完整公共 API 列表 ...
        ]
```

### 7.4 OrcaGymDataView 兜底

```python
class OrcaGymDataView:
    def __getattr__(self, name: str):
        raise AttributeError(
            f"'OrcaGymDataView' 没有字段 '{name}'。\n"
            f"  当前可用字段: {list(self.__dict__.keys())}\n"
            f"  当前可用方法: body_xpos, body_xquat, body_cvel, site_xpos, ...\n"
            f"  如果需要 '{name}'，请在 OrcaGymDataView 中添加该字段或方法。"
        )
```

### 7.5 机制组合效果

| 场景 | 触发机制 | AI/用户看到的 |
|------|---------|-------------|
| AI 生成 `env._mjData.qpos` | `__getattr__` 拦截 | 引导性错误：用 `env.data.qpos` |
| AI 生成 `env._gym._mjData` | Gym 层 `__getattr__` 拦截 | 引导性错误：用 `env.data` |
| AI 生成 `env.data.xfrc_applied` | DataView `__getattr__` | 引导：用 `env.apply_body_force()` |
| AI 生成 `env._mjModel.opt.iterations` | `__getattr__` 拦截 | 引导：用 `env.sim_config.iterations` |
| AI 在 IDE 中补全 `env.` | `__dir__` 控制 | 只看到公共 API |
| AI 阅读 class docstring | 类型标注 + 契约 | 知道正确用法和禁止事项 |

### 7.6 与 OrcaGymLocalEnv 的隔离强度对比

| 系统 | 绕道路径 | 层数 | 内部组件是否可见 |
|------|---------|------|----------------|
| OrcaGymLocalEnv | `env.gym._mjData` | 2 | `gym` 是公共属性 |
| OrcaGymEulerEnv | `env._gym._sim._mjData` | 3 | `__dir__` 不列出，`__getattr__` 拦截 |

---

## 8. 步进编排契约

### 8.1 步进方法职责

| 方法 | 职责 | Euler 耦合 | 适用场景 |
|------|------|-----------|---------|
| `do_simulation(ctrl, n)` | 标准步进 | 有（未来） | 大多数 Env 的 `step()` |
| `mj_step(n)` | 纯 MuJoCo 步进 | 无 | 需要精细控制时序的高级用户 |
| `mj_forward()` | 前向计算 | 无 | 状态设置后更新派生量 |

### 8.2 `do_simulation` 内部编排

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

### 8.3 两种用户步进模式

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

## 9. 迁移策略

### 9.1 迁移代价评估

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
| 外围组件类型标注 | 中 | 需引入 Protocol 或更新类型标注 |

**总体**：约 70% 零改动，25% 机械替换，5% 设计调整。

### 9.2 83 处直接访问的替代方案清单

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

### 9.3 外围组件类型标注迁移

OrcaManipulation 的控制器、任务、场景管理器等组件以 `env: OrcaGymLocalEnv` 为参数类型。迁移策略：引入 Protocol 平滑过渡。

```python
# orca_gym/environment/protocols.py
from typing import Protocol

class OrcaGymEnvProtocol(Protocol):
    """OrcaGym 环境协议，OrcaGymLocalEnv 和 OrcaGymEulerEnv 都满足。"""
    model: OrcaGymModel
    data: OrcaGymDataView
    ctrl: np.ndarray
    def do_simulation(self, ctrl, n_frames): ...
    def mj_step(self, nstep): ...
    def mj_forward(self): ...
    def set_ctrl(self, ctrl): ...
    def query_joint_qpos(self, names): ...
    # ... 其余公共方法 ...
```

OrcaManipulation 的类型标注从 `env: OrcaGymLocalEnv` 改为 `env: OrcaGymEnvProtocol`，一次修改即可兼容两个 Env 类。

### 9.4 迁移分类

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

## 10. 演进路线

### 10.1 阶段划分

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| **阶段 1：骨架实现** | OrcaGymEulerEnv + OrcaGymEuler 骨架，纯 MuJoCo 行为，与 OrcaGymLocalEnv API 对齐 | 可运行的纯 MuJoCo 环境 |
| **阶段 2：API 完备** | OrcaGymDataView、SimConfig、apply_body_force 等完整实现，覆盖 83 处绕道场景 | 用户可零绕道使用 |
| **阶段 3：封装隔离** | `__getattr__`/`__dir__`/DataView 兜底机制实现 | AI/用户被引导走正确路径 |
| **阶段 4：迁移验证** | 选取 OrcaPlayground 中代表性 Env 迁移到 OrcaGymEulerEnv，验证兼容性 | 迁移指南 + 验证报告 |
| **阶段 5：Euler 耦合** | EulerOrchestrator 实现，MuJoCo + Euler 多求解器耦合编排 | 双引擎环境（后续单独设计） |

### 10.2 废弃路线

```
当前: OrcaGymLocalEnv (主用) + OrcaGymRemoteEnv (废弃) + OrcaGymWarpEnv (废弃)
阶段 4 后: OrcaGymEulerEnv (新主用) + OrcaGymLocalEnv (维护模式)
未来: OrcaGymEulerEnv (唯一主用) + OrcaGymLocalEnv (删除)
```

### 10.3 不变性约束

在整个演进过程中，以下不变性必须保持：

| 不变性 | 含义 |
|--------|------|
| **API 兼容性** | 阶段 1 建立的公共 API 签名在后续阶段不破坏性变更 |
| **封装隔离** | `_mjModel`/`_mjData` 永远不作为公共属性暴露 |
| **状态一致性** | `env.data` 在契约规定的时机保证一致 |
| **步进语义** | `do_simulation` 含耦合，`mj_step` 不含耦合，语义不反转 |

---

## 11. 设计决策记录

### 11.1 为何不继承 OrcaGymLocalEnv

**决策**：`OrcaGymEulerEnv` 继承 `OrcaGymBaseEnv`，不继承 `OrcaGymLocalEnv`。

**理由**：
- `OrcaGymLocalEnv` 是上帝类，继承会继承所有职责耦合
- `OrcaGymLocal` 的 `_mjModel`/`_mjData` 暴露设计与 P2 原则冲突
- 独立类可以自由设计封装机制，不受父类约束

### 11.2 为何放弃 MuJoCoAdapter

**决策**：不提供 `MuJoCoAdapter`（受控的 MuJoCo 句柄适配器）。

**理由**：
- 原需求来自 robosuite 控制器等外部库需要直接操作 MuJoCo 对象
- 决定不再支持 robosuite 组件，该需求消失
- 放弃后设计更简洁——没有"逃生舱"，所有需求都通过扩展 Env/Gym 的公共方法解决

### 11.3 为何 `_mjData`/`_mjModel` 放在 MuJoCoSimCore 而非 OrcaGymEuler

**决策**：`_mjModel`/`_mjData` 只存在于 `MuJoCoSimCore` 内部，`OrcaGymEuler` 和 `OrcaGymEulerEnv` 不持有引用。

**理由**：
- 增加绕道路径深度（`env._gym._sim._mjData` 三层）
- 职责内聚：MuJoCo 原生操作集中在 `MuJoCoSimCore`
- `OrcaGymEuler` 作为 Facade 只协调，不直接操作引擎数据

### 11.4 为何保留 `self.gym` 概念但重命名为 `_gym`

**决策**：`OrcaGymEulerEnv` 持有 `_gym: OrcaGymEuler`，但不作为公共属性暴露。

**理由**：
- 保留分层结构（Env → Gym → SimCore）便于职责划分
- `_gym` 不在 `__dir__` 中列出，AI 难以发现
- 用户通过 Env 的公共方法间接使用 Gym，不直接接触

---

## 12. 总结

本文论述了 `OrcaGymEulerEnv` + `OrcaGymEuler` 的架构设计，核心要点：

1. **Facade + 职责内聚分解**替代上帝类，组件按职责划分为 `MuJoCoSimCore`/`OrcaStudioBridge`/`ModelRegistry`/`SimConfig`/`EulerOrchestrator`
2. **完备的公共 API 契约**（P1 完备性）覆盖所有合法 MuJoCo 操作需求，消除用户绕道理由
3. **多层封装隔离机制**（`__getattr__` 拦截 + `__dir__` 控制 + DataView 兜底 + 类型标注 + docstring 契约 + 路径深度）引导 AI 和用户走正确路径
4. **步进编排契约**明确 `do_simulation`（含耦合）与 `mj_step`（纯 MuJoCo）的语义区分
5. **迁移策略**约 70% 零改动、25% 机械替换、5% 设计调整，外围组件通过 Protocol 平滑过渡

本架构为未来 OrcaGym 主路径奠定基础，Euler 耦合编排将在后续文档中单独设计。
