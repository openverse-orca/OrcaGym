SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2026 The OrcaGym Contributors

# OrcaGym 编排 Euler 的集成架构

## 1. 文档定位

### 1.1 文档目标

本文从 **OrcaGym 的角度**论述 Euler 与 OrcaGym 的关系，以及 OrcaGym 如何在自身体系下完成集成，实现编排 Euler、与 MuJoCo 配合完成物理仿真。

本文回答以下核心问题：

1. **角色定位**：在 OrcaGym 体系中，Euler 是什么？MuJoCo 是什么？OrcaGymEuler / OrcaGymEulerEnv 承担什么职责？
2. **编排契约**：OrcaGymEuler 如何编排 MuJoCo 和 Euler 的协同？同步周期如何组织？数据如何在两个引擎之间流动？
3. **数据交换 API**：OrcaGym 通过哪些 API 读写 Euler 的状态？力回流走哪条通道？
4. **封装隔离**：编排逻辑如何在不破坏 `_mjData` / `_mjModel` 封装隔离的前提下完成跨引擎同步？
5. **渲染协作**：Euler 与 OrcaStudio 的渲染协作（占位，待 Euler 渲染文档完成后补充）

### 1.2 与其他文档的关系

| 文档 | 关系 |
|------|------|
| [orca_gym_euler_architecture.md](orca_gym_euler_architecture.md) | 已实现的 OrcaGymEuler / OrcaGymEulerEnv 组件设计文档。本文聚焦于其**编排能力**（Euler 耦合部分），是其编排能力的补充 |
| [overview.md](overview.md) | OrcaGym 分层架构概览。本文细化 Euler 后端的编排机制 |
| Euler 侧 [euler_in_orcagym_integration.md](../../../../OrcaEuler/Docs/Design/architecture/euler_in_orcagym_integration.md) | 从 Euler 视角论述同样的集成关系。两篇文档是同一架构的两个视角，应保持一致 |

### 1.3 核心论点

OrcaGym 与 Euler 的集成关系由以下决策定义：

> **决策 1：OrcaGym 是仿真框架，Euler 是被编排的独立物理引擎。**
> OrcaGymEuler 是双引擎编排核心，持有 MuJoCoSimCore（MuJoCo 运行时）和 Euler Runtime（Model/State/Control/Solver）。OrcaGym 负责在调 `Euler.Solver.step()` 前后做跨引擎数据同步，Euler 不感知 OrcaGym。

> **决策 2：跨引擎编排在 OrcaGym 侧，不在 Euler 侧。**
> MuJoCo 和 Euler 是并列的独立物理引擎。Euler 不封装 MuJoCo，不感知 MuJoCo 的存在。两个引擎的协同（刚体状态传递、力回流）由 OrcaGymEuler 负责。

> **决策 3：数据交换基于 Euler 的标准北向接口。**
> OrcaGym 通过 Euler 的 `Model` / `State` / `Control` 三对象 + `Solver.step()` 契约与 Euler 交换数据。不通过 `_mjData` 代理、不通过专用适配器。Euler 的 `State.body_q` / `body_qd` 是输入（OrcaGym 写），`State.body_f` 是输出（Euler 写）。

> **决策 4：力回流走 MuJoCo 的 `xfrc_applied` 通道。**
> Euler 的 `State.body_f`（COM 世界系 wrench）→ MuJoCo 的 `xfrc_applied`（COM 世界系 wrench），语义一致、零变换、与 Newton 对齐。不走 `qfrc_applied`（广义力通道）。

---

## 2. 角色与定位

### 2.1 OrcaGym 体系中的引擎角色

```
┌────────────────────────────────────────────────────────────────────────┐
│  OrcaGymEulerEnv（gym.Env 接口，用户面向）                             │
│  - step() / reset() / render()                                         │
│  - do_simulation(ctrl, n_frames) → 委托 OrcaGymEuler                   │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  OrcaGymEuler（双引擎编排核心，Facade）                                │
│  - 持有 MuJoCoSimCore（_sim）和 Euler Runtime（_euler）                │
│  - step_with_coupling(ctrl, n_frames, dt)                              │
│  - 跨引擎同步：_sync_mujoco_to_euler_state / _sync_euler_force_to_mujoco│
└──────────┬───────────────────────────────────┬─────────────────────────┘
           │                                   │
           ▼                                   ▼
┌─────────────────────────────┐   ┌─────────────────────────────────────┐
│  MuJoCoSimCore              │   │  Euler Runtime                      │
│  - _mjModel / _mjData       │   │  - Model / State / Control / Solver │
│  - step / forward / set_ctrl│   │  - State.body_q / body_qd / body_f  │
│  - sync_to_view             │   │  - Solver.step(state_in, state_out, │
│  - query_* (公共查询 API)   │   │             control, contacts, dt)  │
└─────────────────────────────┘   └─────────────────────────────────────┘
        │                                          │
        ▼                                          ▼
   MuJoCo 引擎（CPU）                         Euler 引擎（GPU）
   刚体动力学                                 非刚体求解（SPH/PBD/MPM）
```

### 2.2 组件职责

| 组件 | 职责 | 持有的内部对象 |
|------|------|---------------|
| **OrcaGymEulerEnv** | 用户面向的 gym.Env 接口，编排生命周期、步进、渲染 | `_gym`（OrcaGymEuler）、`_studio_bridge` |
| **OrcaGymEuler** | 双引擎编排核心 Facade，组合 MuJoCoSimCore 和 Euler Runtime | `_sim`（MuJoCoSimCore）、`_euler`（Euler Runtime 占位）、`_view`、`_opt`、`_registry`、`_studio` |
| **MuJoCoSimCore** | MuJoCo 仿真核心，持有 `_mjModel` / `_mjData`，提供 step/forward/query_* | `_mjModel`、`_mjData` |
| **Euler Runtime** | Euler 的 Model/State/Control/Solver，由 OrcaGymEuler 持有引用 | Euler 内部对象 |

### 2.3 已实现 vs. 待实现

| 能力 | 状态 | 位置 |
|------|------|------|
| OrcaGymEuler / OrcaGymEulerEnv 组件骨架 | ✅ 已实现 | [orca_gym_euler.py](../../../../OrcaGym/orca_gym/core/euler/orca_gym_euler.py)、[orca_gym_euler_env.py](../../../../OrcaGym/orca_gym/environment/euler/orca_gym_euler_env.py) |
| MuJoCoSimCore（纯 MuJoCo 步进、查询、同步） | ✅ 已实现 | [mujoco_sim_core.py](../../../../OrcaGym/orca_gym/core/euler/mujoco_sim_core.py) |
| `step_with_coupling`（当前等价于纯 MuJoCo 步进） | ✅ 骨架已实现 | [orca_gym_euler.py:step_with_coupling](../../../../OrcaGym/orca_gym/core/euler/orca_gym_euler.py) |
| Euler Runtime 持有与编排 | ⏳ 待实现 | `_euler` 当前为 None |
| 跨引擎同步（`_sync_mujoco_to_euler_state` / `_sync_euler_force_to_mujoco`） | ⏳ 待实现 | 本文定义契约 |
| body 索引映射（基于 `body_label`） | ⏳ 待实现 | 依赖 Euler 实现 `Model.body_label` |

---

## 3. 编排契约

### 3.1 单步编排

OrcaGymEuler 的 `step_with_coupling` 是双引擎编排的核心入口。当前骨架实现等价于纯 MuJoCo 步进（`_euler` 为 None），未来 Euler 耦合实现后扩展为四阶段编排：

```python
def step_with_coupling(self, ctrl: np.ndarray, n_frames: int, dt: float) -> None:
    """带 Euler 耦合的步进。

    当前骨架：_euler 为 None，等价于纯 MuJoCo 步进。
    未来扩展：四阶段编排（MuJoCo 解算 → 状态同步 → Euler 解算 → 力回流）。
    """
    if not self.has_euler():
        # 骨架阶段：纯 MuJoCo 步进
        sim = object.__getattribute__(self, "_sim")
        sim.set_ctrl(ctrl)
        sim.step(n_frames)
        return

    # 未来实现：四阶段编排
    self._step_with_euler_coupling(ctrl, n_frames, dt)
```

### 3.2 四阶段编排（未来实现）

```python
def _step_with_euler_coupling(self, ctrl, n_frames, dt):
    sim = object.__getattribute__(self, "_sim")
    euler = object.__getattribute__(self, "_euler")

    # 阶段 1: MuJoCo 刚体解算
    sim.set_ctrl(ctrl)
    sim.step(n_frames)  # mj_step × n_frames

    # 阶段 2: MuJoCo 刚体状态 → Euler State（OrcaGymEuler 负责同步）
    self._sync_mujoco_to_euler_state()

    # 阶段 3: Euler 非刚体解算（标准 step 接口）
    for _ in range(euler.steps_per_cycle):
        euler.state_0.clear_forces()
        euler.solver.step(
            euler.state_0, euler.state_1,
            euler.control, None, euler.dt
        )
        euler.state_0, euler.state_1 = euler.state_1, euler.state_0

    # 阶段 4: Euler 力回流 → MuJoCo（OrcaGymEuler 负责同步）
    self._sync_euler_force_to_mujoco()
```

**关键点**：
1. **Euler 只看到标准 `step` 接口**：不感知 MuJoCo 的存在
2. **同步职责在 OrcaGymEuler**：阶段 2 和阶段 4 的跨引擎同步由 OrcaGymEuler 负责
3. **Euler 的 `State.body_q` / `body_qd` 由 OrcaGymEuler 写入**：Euler 只读
4. **Euler 的 `State.body_f` 由 Euler 写入**：OrcaGymEuler 读取并回流到 MuJoCo

### 3.3 同步周期（SyncCycle）

OrcaGymEuler 与 Euler 的耦合按"同步周期"（Sync Cycle）事务级同步，而非每个物理步同步。当前阶段实现**等齿比场景**：

```
T_cycle = rigid_steps_per_cycle × rigid_dt
        = euler_steps_per_cycle × euler_dt
```

- **一个同步周期内**，MuJoCo 连续执行 `rigid_steps_per_cycle` 个物理步，无 Euler 同步
- **一个同步周期内**，Euler 连续执行 `euler_steps_per_cycle` 个物理步，无 MuJoCo 同步
- **同步周期边界处**，执行阶段 2（MuJoCo→Euler）和阶段 4（Euler→MuJoCo）的双向同步

**当前实现**：`do_simulation(ctrl, n_frames)` 一次调用就是一个同步周期，`n_frames` = `rigid_steps_per_cycle`。

**未来扩展**：M:N 齿比耦合（`rigid_steps × dt_rigid = euler_steps × dt_euler`）由 Euler 内部 `CouplingOrchestrator` 与 OrcaGymEuler 协商。

### 3.4 与 Euler 内部 CouplingOrchestrator 的分层

```
┌──────────────────────────────────────────────────────────────────────┐
│  跨引擎编排（OrcaGymEuler 负责）                                      │
│  - MuJoCo ↔ Euler 的数据同步                                          │
│  - step_with_coupling 的四阶段编排                                    │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Euler 内部多求解器调度（CouplingOrchestrator 负责）                  │
│  - Euler 内部非刚体求解器之间的耦合（如 SPH + PBD）                   │
│  - 不涉及与 MuJoCo 的跨引擎编排                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**分层原则**：
- **跨引擎编排**（Euler vs. MuJoCo）→ OrcaGymEuler 负责
- **Euler 内部多求解器调度**（SPH vs. PBD）→ Euler 的 `CouplingOrchestrator` 负责

OrcaGymEuler 调用 `Euler.Solver.step()` 时，Euler 内部可能由 `CouplingOrchestrator` 编排多个非刚体求解器，但这对 OrcaGymEuler 透明——OrcaGymEuler 只看到一次 `step` 调用。

---

## 4. 数据交换机制

### 4.1 数据流总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                        OrcaGymEuler                                  │
│                                                                      │
│   ┌─────────────┐         ┌──────────────────────┐                  │
│   │ MuJoCoSimCore│         │   Euler Runtime      │                  │
│   │  _mjData     │         │  Model / State       │                  │
│   │  _mjModel    │         │  Control / Solver    │                  │
│   └──────┬──────┘         └──────────┬───────────┘                  │
│          │                           │                              │
│          │  阶段 1: mj_step          │  阶段 3: euler_solver.step    │
│          │  (刚体解算)               │  (非刚体解算)                 │
│          │                           │                              │
│          ▼                           ▼                              │
│   ┌──────────────────────────────────────────────────┐              │
│   │  阶段 2: _sync_mujoco_to_euler_state()           │              │
│   │  _mjData.xpos/xquat/cvel → State.body_q/body_qd  │              │
│   └──────────────────────────────────────────────────┘              │
│                                                                      │
│   ┌──────────────────────────────────────────────────┐              │
│   │  阶段 4: _sync_euler_force_to_mujoco()           │              │
│   │  State.body_f → _mjData.xfrc_applied             │              │
│   └──────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 阶段 2：MuJoCo 刚体状态 → Euler State

**方向**：CPU（`_mjData`）→ GPU（Euler `State.body_q` / `body_qd`）

**语义映射**：

| MuJoCo 字段 | Euler 字段 | 转换 |
|-------------|-----------|------|
| `xpos[body]`（3D 位置，世界系） | `State.body_q[i].p`（transform 平移分量） | 直接拷贝 |
| `xquat[body]`（4D 四元数，世界系，**(w,x,y,z)** 顺序） | `State.body_q[i].q`（transform 旋转分量，**(x,y,z,w)** 顺序） | **需顺序转换**：`xyzw = (q[1], q[2], q[3], q[0])` |
| `cvel[body]`（6D 速度，COM 线速度 + 世界系角速度） | `State.body_qd[i]`（spatial_vector `[v_linear, w_angular]`） | 直接拷贝（语义一致） |

**四元数顺序差异**（关键）：
- MuJoCo 采用 `(w, x, y, z)` 顺序（MuJoCo 官方约定）
- Euler / Flow / Warp 采用 `(x, y, z, w)` 顺序（Warp `types.py` 的 `"xyzw"` 索引约定，Euler 测试代码注释确认）
- 两者**不一致**，同步时必须显式转换

**转换位置**：在 OrcaGymEuler 的 `_sync_mujoco_to_euler_state` 内完成（core 层跨引擎同步层）。Euler 和 MuJoCo 各自保持原生四元数约定，互不感知。这与 Newton 的 `quat_wxyz_to_xyzb` / `quat_xyzw_to_wxyz` 转换函数（[newton kernels.py:211-225](file:///home/superfhwl/repo/newton/newton/_src/solvers/mujoco/kernels.py#L211-L225)）方案完全对齐。

**索引映射**：通过 `body_label` 建立 `mj_body_id ↔ euler_body_idx` 映射，不依赖解析顺序。

**实现路径**（当前为模式 A，CPU→GPU 搬运，NumPy 向量化）：

```python
def _sync_mujoco_to_euler_state(self):
    """MuJoCo 刚体状态 → Euler State（OrcaGymEuler 负责同步）。

    性能设计：NumPy 向量化，避免 Python for 循环。
    200 body 场景下单次同步 < 100 μs（详见 §4.6 性能设计）。
    """
    sim = object.__getattribute__(self, "_sim")
    euler = object.__getattribute__(self, "_euler")

    # core 层跨组件访问（合法，见 §5.2）
    mj_data = sim._mjData  # noqa: SLF001  core 层跨组件访问

    # 预构建的索引数组（初始化时一次性构建，运行时复用）
    mj_bodies = self._mj_bodies_arr        # np.ndarray (nbody_active,)
    euler_idxs = self._euler_idxs_arr      # np.ndarray (nbody_active,)

    # 复用预分配的 NumPy 缓冲区（避免每帧分配）
    body_q_np = self._body_q_buf           # (body_count, 7)
    body_qd_np = self._body_qd_buf         # (body_count, 6)

    # 向量化拷贝 + 四元数顺序转换（MuJoCo wxyz → Euler xyzw）
    body_q_np[euler_idxs, :3] = mj_data.xpos[mj_bodies]
    xquat_wxyz = mj_data.xquat[mj_bodies]                  # (N, 4) wxyz
    body_q_np[euler_idxs, 3:] = xquat_wxyz[:, [1, 2, 3, 0]]  # 向量化重排为 xyzw
    body_qd_np[euler_idxs, :] = mj_data.cvel[mj_bodies]

    # CPU → GPU 上传
    euler.state_0.body_q.assign(flow.from_numpy(body_q_np))
    euler.state_0.body_qd.assign(flow.from_numpy(body_qd_np))
```

**向量化关键点**：
1. `_mj_bodies_arr` / `_euler_idxs_arr` 在 `_build_body_index_map` 时一次性构建为 NumPy 数组，运行时复用
2. `body_q_buf` / `body_qd_buf` 在初始化时预分配，避免每帧 `np.zeros` 的分配开销
3. 四元数顺序转换用 `xquat[:, [1, 2, 3, 0]]` 一次向量化重排，替代逐 body 循环
4. 索引映射的 dict 在运行时不参与热路径，热路径只用 NumPy 数组 fancy indexing

**封装隔离说明**：OrcaGymEuler 在 core 层，访问 `MuJoCoSimCore._mjData` 属于 core 层组件编排，不违反用户侧的封装隔离（用户侧通过 `env.data` / `env.query_*` 访问，不触 `_mjData`）。

**未来模式 B**（MuJoCoFlow 路径）：当刚体由 MuJoCoFlow 在 GPU 上解算时，通过 GPU 内存句柄共享实现零拷贝。模式 A → 模式 B 的重构代价可接受，因为同步逻辑封装在 `_sync_mujoco_to_euler_state` 内。

### 4.3 阶段 4：Euler 力回流 → MuJoCo

**方向**：GPU（Euler `State.body_f`）→ CPU（`_mjData.xfrc_applied`）

**关键决策**：Euler 的 `State.body_f` → MuJoCo 的 `xfrc_applied`（笛卡尔力通道），不走 `qfrc_applied`（广义力通道）。

**理由**（参考 Newton 的 `SolverMuJoCo` 实现）：

1. **语义一致**：
   - Euler `body_f` = 6D wrench，布局 `[force(3), torque(3)]`，作用点 = body COM，世界系
   - MuJoCo `xfrc_applied` = 6D wrench，布局 `[force(3), torque(3)]`，作用点 = `xipos`（body COM 世界坐标），世界系
   - 两者语义完全一致，可 1:1 直接拷贝，无需任何坐标变换

2. **正确性有保障**：
   - MuJoCo 内部通过 `mj_applyFT` 自动处理 Jacobian 转置映射（`J'·xfrc_applied`）和作用点变换
   - 不需要 OrcaGymEuler 计算 Jacobian 或坐标变换

3. **与 Newton 对齐**：
   - Newton 的 `apply_mjc_body_f_kernel` 正是此方案：`body_f` → `xfrc_applied` 1:1 拷贝

**为什么不用 `qfrc_applied`**：
- `qfrc_applied` 是关节空间的广义力，与 `body_f`（笛卡尔 wrench）不在同一空间
- 若走 `qfrc_applied`，OrcaGymEuler 需要自己计算 `J'·body_f`（Jacobian 转置），复杂且易错
- 让 MuJoCo 内部统一处理 `J'` 变换和作用点变换，是最稳健的方案

**实现路径**：

```python
def _sync_euler_force_to_mujoco(self):
    """Euler body_f → MuJoCo xfrc_applied（力回流）。"""
    sim = object.__getattribute__(self, "_sim")
    euler = object.__getattribute__(self, "_euler")

    # GPU → CPU 下载
    body_f_np = euler.state_0.body_f.numpy()  # (body_count, 6)

    # 预构建的索引数组（与阶段 2 复用）
    mj_bodies = self._mj_bodies_arr
    euler_idxs = self._euler_idxs_arr

    # 清零 xfrc_applied（MuJoCo 不会自动清零，必须显式清）
    mj_data = sim._mjData  # noqa: SLF001  core 层跨组件访问
    mj_data.xfrc_applied[:] = 0.0

    # 向量化 1:1 拷贝（force+torque 整行拷贝）
    mj_data.xfrc_applied[mj_bodies] = body_f_np[euler_idxs]
```

**清零时机**：每个 `mj_step` 前 OrcaGymEuler 必须显式清零 `xfrc_applied`，否则力会累积。这与 Newton 的 `_apply_mjc_control` 行为一致。

### 4.4 body 索引映射

OrcaGymEuler 在初始化时建立 `body_index_map: dict[int, int]`（`mj_body_id → euler_body_idx`）：

```python
def _build_body_index_map(self):
    """通过 body_label 建立 MuJoCo ↔ Euler body 索引映射。

    同时构建 NumPy 索引数组和预分配缓冲区，供热路径向量化同步使用。
    """
    sim = object.__getattribute__(self, "_sim")
    euler = object.__getattribute__(self, "_euler")

    mj_model = sim._mjModel  # noqa: SLF001
    self._body_index_map = {}  # dict 形式，供调试/查询

    # Euler label → idx 反查表
    euler_label_to_idx = {
        label: idx
        for idx, label in enumerate(euler.model.body_label)
    }

    # 遍历 MuJoCo body（跳过 world id=0），构建映射
    mj_body_list = []
    euler_idx_list = []
    for mj_body in range(1, mj_model.nbody):
        mj_name = mj_model.body(mj_body).name
        euler_idx = euler_label_to_idx.get(mj_name)
        if euler_idx is not None:
            self._body_index_map[mj_body] = euler_idx
            mj_body_list.append(mj_body)
            euler_idx_list.append(euler_idx)
        else:
            # 警告：MuJoCo body 在 Euler Model 中无对应
            ...

    # 构建 NumPy 索引数组（热路径使用，避免 Python for 循环）
    self._mj_bodies_arr = np.asarray(mj_body_list, dtype=np.int32)
    self._euler_idxs_arr = np.asarray(euler_idx_list, dtype=np.int32)

    # 预分配 NumPy 缓冲区（避免每帧 np.zeros 分配开销）
    body_count = euler.model.body_count
    self._body_q_buf = np.zeros((body_count, 7), dtype=np.float32)
    self._body_qd_buf = np.zeros((body_count, 6), dtype=np.float32)
```

**依赖**：Euler 实现 `Model.body_label`（与 Newton 对齐）。

### 4.5 Body→Shape 映射的注意事项

`body_label` 索引映射**仅在 body 层面**保证 MuJoCo ↔ Euler 对齐。shape（geom）层面不需要独立索引映射，因为：

1. **Euler 的 Body→Shape 设计与 MuJoCo 的 Body→Geom 同构**：body 是自由度载体，shape/geom 无自由度跟随 body。Euler 的 `shape_body[shape_idx]` 指向所属 body 的 `euler_body_idx`，与 MuJoCo geom 隶属 body 的关系一致。
2. **shape 在 `add_mjcf()` 解析时建立对应关系**：Euler 的 `ModelBuilder.add_mjcf()` 遍历 MJCF 的每个 body，把其下 geom 逐个转为 Euler `add_shape_*` 调用，`shape_body` 直接指向该 body 的 euler 索引。
3. **shape_transform 语义一致**：MuJoCo 的 `geom pos/quat`（body 局部系）对应 Euler 的 `shape_transform`（body 局部系），`add_mjcf()` 转换时需注意四元数顺序（MuJoCo wxyz → Euler xyzw）。

**静态 shape**：MuJoCo world body (id=0) 下的 geom 转为 Euler 的 `add_shape(..., body=-1, is_static=True)`，其 `pos/quat` 直接作为 `shape_transform` 传入。

**碰撞检测的演进**：Euler 的 `shape_*` 字段设计不限于粒子碰撞。当前阶段 shape 作为流体粒子的碰撞边界（SDF 采样），未来 `CollisionPipeline` 集成后将支持刚体-刚体碰撞和流固耦合接触。`shape_flags` 的 `COLLIDE_SHAPES` / `COLLIDE_PARTICLES` 两个独立位已为这两个阶段预留。详细映射关系见 Euler 侧文档 [euler_in_orcagym_integration.md §6.3](../../../../OrcaEuler/Docs/Design/architecture/euler_in_orcagym_integration.md)。

### 4.6 性能设计

本节针对模式 A（CPU↔GPU 搬运）进行性能设计，验证同步开销在 100Hz 同步频率下可接受。

#### 4.6.1 同步范围明确

**关键前提**：每周期同步**只搬运 body 级状态**，不涉及 shape/geom 级数据。

| 数据 | 生命周期 | 是否参与每周期同步 |
|------|---------|------------------|
| `body_q` / `body_qd`（body 位姿/速度） | 动态 | ✅ 阶段 2 同步 |
| `body_f`（body 受力） | 动态 | ✅ 阶段 4 同步 |
| `shape_transform` / `shape_type` / `shape_scale` 等 | 静态 | ❌ `add_mjcf()` 时一次性构建 |
| `body_mass` / `body_inertia` / `body_com` | 静态 | ❌ `add_mjcf()` 时一次性构建 |

**结论**：1000 geom 不参与每周期同步，同步开销只与 body 数量相关。

#### 4.6.2 典型场景负载分析

**场景假设**：200 body / 1000 geom / 100Hz 同步频率（周期 10ms）

| 阶段 | 方向 | 数据量 | 单次计算量 |
|------|------|--------|-----------|
| 阶段 2（MuJoCo→Euler） | CPU→GPU | 200 × (7+6) = 2600 floats = **10.4 KB** | 200 次四元数 wxyz→xyzw 重排 + 数组拷贝 |
| 阶段 4（Euler→MuJoCo） | GPU→CPU | 200 × 6 = 1200 floats = **4.8 KB** | 200 次 1:1 拷贝 + xfrc 整体清零 |
| **单次同步合计** | 双向 | **15.2 KB** | 见 §4.6.3 耗费预估 |

**100Hz 带宽需求**：15.2 KB × 100 Hz = **1.52 MB/s**

#### 4.6.3 CPU 侧耗时预估

| 方案 | 写法 | 单次耗时 | 占 10ms 周期 |
|------|------|---------|-------------|
| 方案 1（不推荐） | 纯 Python for 循环 | 0.6 - 1.0 ms | 6 - 10% |
| **方案 2（采用）** | **NumPy 向量化 + 预分配缓冲区** | **30 - 90 μs** | **0.4 - 1%** |

**方案 2 的关键优化**（已在 §4.2 / §4.3 / §4.4 伪代码中体现）：

1. **预构建 NumPy 索引数组**：`_mj_bodies_arr` / `_euler_idxs_arr` 在 `_build_body_index_map` 一次性构建，热路径用 fancy indexing，避免 Python dict 查找 + for 循环
2. **预分配 NumPy 缓冲区**：`_body_q_buf` / `_body_qd_buf` 在初始化时分配，避免每帧 `np.zeros` 的分配开销
3. **向量化四元数重排**：`xquat[:, [1, 2, 3, 0]]` 一次重排，替代逐 body 循环
4. **向量化拷贝**：`mj_data.xfrc_applied[mj_bodies] = body_f_np[euler_idxs]` 整行拷贝

#### 4.6.4 传输带宽预估

| 项 | 计算 | 值 |
|----|------|-----|
| 单次同步数据量 | — | 15.2 KB |
| 100Hz 带宽需求 | 15.2 KB × 100 | **1.52 MB/s** |
| PCIe 4.0 实际带宽 | — | ~20-25 GB/s |
| 带宽占用率 | 1.52 / 25000 | **0.006%** |

**结论**：传输带宽**完全不是瓶颈**，占用率可忽略。

#### 4.6.5 总耗时占比

| 阶段 | 方案 2 耗时 |
|------|-----------|
| CPU 计算（向量化） | 30 - 90 μs |
| CPU→GPU 传输（10.4 KB） | ~5 μs |
| GPU→CPU 传输（4.8 KB） | ~3 μs |
| **同步总耗时** | **40 - 100 μs** |
| **占 10ms 周期比例** | **0.4 - 1%** |

#### 4.6.6 模式 A → 模式 B 的演进预期

模式 A 的同步开销已可忽略（< 1%），模式 B（MuJoCoFlow GPU 零拷贝）的优化收益主要在：

1. **消除 CPU↔GPU 传输**：省去 ~8 μs/次的传输开销
2. **消除 CPU 侧 NumPy 拷贝**：省去 30-90 μs/次的向量化计算
3. **解锁更高同步频率**：模式 A 在 1000Hz 同步时占 4-10%，模式 B 可支持 1000Hz+ 而无显著开销

**演进代价**：模式 A → 模式 B 重构集中在 `_sync_mujoco_to_euler_state` / `_sync_euler_force_to_mujoco` 两个方法内，外部接口不变。

---

## 5. 封装隔离合规性

### 5.1 用户侧隔离

用户通过 `OrcaGymEulerEnv` 访问，不触内部对象：

| 操作 | 正确（公共 API） | 禁止（穿墙） |
|------|-----------------|-------------|
| 读取状态 | `env.data.qpos` / `env.data.body_xpos(name)` / `env.query_*()` | `env._gym._sim._mjData.qpos` |
| 写入外力 | `env.apply_body_force()` | `env._gym._sim._mjData.xfrc_applied[...] = ...` |
| 步进 | `env.do_simulation(ctrl, n_frames)` / `env.step()` | `env._gym._sim._mjData.step()` |
| 求解器配置 | `env.sim_config.timestep = 0.002` | `env._gym._sim._mjModel.opt.timestep = 0.002` |
| Euler 耦合查询 | `env.has_euler()`（未来扩展） | `env._gym._euler` |

OrcaGymEuler 的 `__getattribute__` 拦截 `_BLOCKED_ATTRS`（含 `_mjData` / `_mjModel` / `_sim` / `_euler` 等），外部访问直接抛 `AttributeError`。

### 5.2 Core 层跨组件访问

OrcaGymEuler 在 core 层编排 MuJoCoSimCore 和 Euler Runtime，**需要**访问 `MuJoCoSimCore._mjData` 来做跨引擎同步。这属于 core 层组件编排，不违反用户侧隔离：

```python
# OrcaGymEuler 内部（core 层）
sim = object.__getattribute__(self, "_sim")
mj_data = sim._mjData  # noqa: SLF001  core 层跨组件访问，合法
```

**合规原则**：
- 用户侧（`OrcaGymEulerEnv` 及其子类）→ 通过公共 API 访问，不触 `_mjData` / `_euler`
- Core 层（`OrcaGymEuler`）→ 可通过 `object.__getattribute__` 绕过自身拦截，访问 `MuJoCoSimCore._mjData` 做跨引擎同步

### 5.3 ruff SLF001 合规

跨引擎同步代码中的 `sim._mjData` 访问需标注 `# noqa: SLF001`，并附注释说明是 core 层组件编排：

```python
mj_data = sim._mjData  # noqa: SLF001  core 层跨组件访问：跨引擎同步
```

提交前必须执行：
```bash
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca_gym/
```

---

## 6. Euler Runtime 在 OrcaGymEuler 中的持有方式

### 6.1 当前骨架

```python
class OrcaGymEuler:
    def __init__(self, stub=None) -> None:
        self._sim = MuJoCoSimCore()
        self._euler = None    # Euler Runtime | None（骨架阶段恒为 None）
```

`has_euler()` 返回 False，`step_with_coupling` 等价于纯 MuJoCo 步进。

### 6.2 未来扩展

Euler Runtime 作为一个独立的持有对象（不是直接散落在 OrcaGymEuler 上的字段），封装 Model/State/Control/Solver：

```python
@dataclass
class EulerRuntime:
    """OrcaGymEuler 持有的 Euler 运行时引用。"""
    model: EulerModel
    state_0: EulerState
    state_1: EulerState
    control: EulerControl
    solver: EulerSolverBase
    dt: float
    steps_per_cycle: int

class OrcaGymEuler:
    def __init__(self, stub=None) -> None:
        self._sim = MuJoCoSimCore()
        self._euler: EulerRuntime | None = None  # 未来填充
```

**设计原则**：
- `_euler` 仍然被 `_BLOCKED_ATTRS` 拦截，用户不直接访问
- OrcaGymEuler 通过 `has_euler()` 查询、`step_with_coupling` 编排
- 用户若需要查询 Euler 状态（如粒子位置用于观测），通过 OrcaGymEulerEnv 扩展公共方法委托（如 `env.query_euler_particle_positions()`）

---

## 7. 渲染协作（占位）

> **待补充**：Euler 与 OrcaStudio 的 GPU 零拷贝渲染同步由独立的渲染架构文档论述。本节在 Euler 渲染文档完成后补充：
>
> 1. Euler 粒子（流体）渲染的 GPU 内存句柄共享机制
> 2. Euler 可变形体代理 Mesh 顶点动画的渲染同步
> 3. OrcaGymEuler 在 `render()` 中如何编排 MuJoCo 渲染（通过 OrcaStudio Bridge）和 Euler 渲染（直接 GPU 同步）

---

## 8. 总结

OrcaGym 编排 Euler 的集成遵循以下原则：

1. **角色定位**：OrcaGym 是仿真框架，Euler 是被编排的独立物理引擎，MuJoCo 是并列的独立物理引擎
2. **编排核心**：OrcaGymEuler 是双引擎编排核心，持有 MuJoCoSimCore 和 Euler Runtime
3. **四阶段编排**：MuJoCo 解算 → 状态同步 → Euler 解算 → 力回流
4. **数据交换基于标准接口**：OrcaGymEuler 读写 Euler 的 `State.body_q` / `body_qd` / `body_f`，Euler 不感知数据来源
5. **力回流走 `xfrc_applied`**：Euler `body_f` → MuJoCo `xfrc_applied`，语义一致、零变换、与 Newton 对齐
6. **body_label 作为索引映射基础**：不依赖解析顺序，稳健可靠
7. **同步周期事务级同步**：等齿比场景先行，M:N 齿比后续扩展
8. **封装隔离**：用户侧通过公共 API 访问，core 层可跨组件访问做同步
9. **分层编排**：跨引擎编排在 OrcaGymEuler，Euler 内部多求解器调度在 CouplingOrchestrator
