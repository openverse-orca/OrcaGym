# OrcaGym Euler 锚点与等式约束 API 重构开发方案

## 1. 背景与目标

### 1.1 问题陈述

当前 `OrcaGymEulerEnv` 中的体操作能力存在两类职责混叠在一个类里：

1. **Studio UI 抓取专用逻辑**：`do_body_manipulation` / `anchor_actor` / `release_body_anchored` / `update_anchor_equality_constraints`，这组方法在内部耦合了 UI 抓取的特定假设（固定使用 `ActorManipulator_Anchor`、跟随 `manip_state["mocap_pose"]`、释放时恢复 XML 原始 obj id 模拟 dummy body 机制）。
2. **用户程序化操作需求的底层能力**：09 课等示例需要在用户代码中通过自备 mocap body + equality 把机器人某个 body 程序化拖动到目标位姿。

二者目前共用同一个 `update_anchor_equality_constraints(actor_name, anchor_type)` 入口，参数 `actor_name`/`anchor_type` 已暗含 UI 抓取语义；09 课被迫借用此入口并改写 `_anchor_mocap_name`，结果出现了：

- mocap body 与 equality 槽位错配（设置 A mocap 的位姿，绑定 B mocap 的约束）
- 绑定瞬间未将 mocap 对齐到 actor 当前位姿，下一帧约束把 actor 猛拉到 mocap 初始位置
- UI 抓取流程的状态字段（`_anchored_actor` / `_anchor_original_eq`）被用户代码污染，UI 再次介入时行为不可预测

### 1.2 重构目标

本方案的核心分层原则：**框架只提供无状态原语，业务编排（含状态）归业务所有者实现。**

| 目标 | 含义 |
|------|------|
| **G1 框架层只开放无状态 equality 原语** | 在 Env 层提供**单次原子**的 equality 读/写原语（find_slot / read_constraint / update_constraint / modify_objects / set_mocap / mj_forward），不跨调用持有状态、不编排多步业务流程 |
| **G2 业务编排归业务所有者** | bind_mocap / release 这类"多步编排 + 快照状态"的业务模式不在框架公共 API 中暴露；UI 抓取的编排内联进 `_anchor_*` 内部方法组，程序化操作（如 09 课）由消费者自行实现 |
| **G3 UI 抓取能力标识为内部 API** | `do_body_manipulation` / `anchor_actor` / `release_body_anchored` 等 UI 专用方法改名为 `_` 前缀，docstring 标注"内部 API，AI/用户不应调用"，保留在同一 Env 类中（无需继承新子类） |
| **G4 状态字段隔离** | UI 抓取状态字段（`_anchored_actor` / `_anchor_original_eq` 等）保持 `_` 前缀私有；程序化操作的快照状态由消费者自管，不存放在 Env |
| **G5 遵从架构契约** | 不暴露 `_mjModel`/`_mjData`；所有读写走 Env→Gym→SimCore/Registry 三层委托；公共原语遵循 L1 公共 API 规范，UI 抓取方法遵循 L2 内部 API 规范，受 `__dir__` + ruff SLF001 约束 |

### 1.3 分层原则详述

**无状态原语（框架公共 API）的判定标准**：
- 单次调用完成单一数据读写，不依赖前后调用的顺序状态
- 不持有跨调用的快照、绑定标记等业务状态
- 例：`equality_find_slot_by_body`（按 body 查槽位）、`equality_constraint(slot)`（读单槽位）、`equality_update(slot, **fields)`（写单槽位 + mj_forward）、`set_mocap_pos_and_quat`（写 mocap 位姿）

**业务编排（消费者实现）的判定标准**：
- 编排多个原语完成有语义意图的流程（如"绑定"/"释放"/"抓取"）
- 持有跨调用的业务状态（如快照、绑定标记、当前锚定对象）
- 例：bind_mocap（find_slot + read_constraint + align_mocap + update_constraint）、release（read_current + restore_from_snapshot）

> `equality_snapshot()`（批量读全部约束）技术上是无状态读，但仅服务于 save/restore 业务模式，且只是 `equality_constraint(i)` 的循环便利。按分层原则不纳入公共 API，消费者需要时自行循环调用 `equality_constraint`。

### 1.4 非目标

- **Env 层公共 API 收敛**：删除 `OrcaGymEulerEnv.update_equality_constraints` / `OrcaGymEulerEnv.modify_equality_objects` 公共方法（功能被 `equality_update` 覆盖）。`SimCore.update_equality_constraints` 作为 `equality_update` 的底层实现保留，不进入 Env 公共 API。
- **`SimCore.modify_equality_objects`**：Env 层不再委托，SimCore 层保留方法定义但本方案不再使用（无内部调用方）。后续可单独清理。
- 不改动 `ModelRegistry.equality_constraint` / `equality_object_ids` / `n_equality` 的读取语义（已正确返回完整字段）
- 不重构 `OrcaStudioBridge.get_body_manipulation_state` 的 gRPC 协议
- 不引入子类继承体系（`OrcaGymEulerInteractiveEnv`），避免所有 UI 抓取用户被迫继承新类
- 不引入新的架构组件（如新的 `AnchorManager` 子组件）—— 在现有组件内职责内聚即可
- **不动 Local/Remote 体系**：`orca_gym_local_env.py` / `orca_gym_remote_env.py` / `orca_gym_local.py` / `orca_gym_warp.py` / `orca_gym_remote.py` 等非 Euler 体系的 `update_equality_constraints` / `modify_equality_objects` 不处理，与 Euler 体系无关
- **不动 OrcaPlayground 调用方**：`envs/fluid/sim_env.py` / `envs/franka_rl/franka_gym_env.py` 等基于老框架的调用方不处理

---

## 2. 架构约束对齐

本方案严格遵守 `docs/design/architecture/orca_gym_euler_architecture.md` 的以下约束：

### 2.1 组件划分（架构 §5）

- **无状态 equality 原语**是 `OrcaGymEulerEnv` 的**公共方法**（L1），委托到 `OrcaGymEuler` 的对应方法，再委托到 `MuJoCoSimCore` / `ModelRegistry`。
- **UI 抓取业务编排**保留在 `OrcaGymEulerEnv` 内，作为**内部方法**（L2，`_` 前缀），不单独划分子类。业务编排内联在 `_anchor_*` 方法组中，通过组合公共原语实现。

### 2.2 API 契约（架构 §6）

| 契约 | 本方案对齐方式 |
|------|---------------|
| **R1 状态读取** | equality 读取通过 `env.equality_find_slot_by_body()` / `env.equality_constraint()` 公共原语，不穿墙 `_mjModel.eq_*` |
| **W1 状态写入** | equality 写入通过 `env.equality_update()` 公共原语（含 active/solref/solimp 字段 + 可选 forward），不穿墙 `_mjModel.eq_*`；底层 `SimCore.update_equality_constraints` 作为 `equality_update` 实现细节，不进入 Env 公共 API |
| **W3 写后一致** | `equality_update` 内部调用 `mj_forward()`，保证 `env.data` 一致 |
| **C1 求解器配置** | 不涉及 |
| **N1 名称空间** | 公共原语接收 body **名称**（非 id），内部通过 `self.model.body_name2id` 解析 |
| **L1/L2/L3 层级** | 无状态原语为公共方法（L1），进入 `__dir__`；UI 抓取业务编排方法为 `_` 前缀内部方法（L2），不进入 `__dir__` |

### 2.3 封装隔离（架构 §7）

| 机制 | 本方案对齐方式 |
|------|---------------|
| **M1 ruff SLF001** | 公共原语无 `_` 前缀；UI 抓取业务编排方法为 `_` 前缀内部方法，SLF001 对外部访问报警 |
| **M3 `__dir__`** | Env `__dir__` 只暴露无状态原语；UI 抓取方法不进入 `__dir__` |
| **M6 docstring** | 公共原语 docstring 列出原子语义；UI 抓取方法 docstring 标注"内部 API，AI/用户不应调用，由 `render()` 内部驱动" |

### 2.4 隔离强度说明

UI 抓取内部 API 的隔离不依赖 AGENTS.md 显式条目（避免规则文档臃肿），而由以下三重机制保障：

1. **`_` 前缀约定**：社区约定 + ruff SLF001 静态检查，外部访问 `env._anchor_actor` 等触发报警
2. **`__dir__` 控制**：UI 抓取方法不进入公共 API 列表，IDE 自动补全与 `dir(env)` 不可见
3. **docstring `.. warning::`**：方法文档显式标注"内部 API，AI/用户不应调用"，AI 代码生成时遵守

### 2.5 冲突处理

本方案不触及架构文档定义的契约边界：

- 不暴露 `_mjModel`/`_mjData`
- 不引入架构文档未定义的新组件
- 不引入新的子类继承体系
- 不修改步进编排契约（`do_simulation` / `mj_step` 不变）
- 不修改 `OrcaGymDataView` 的字段集合（equality 数据是 model 级，非 data 级）

无需暂停提交用户决策。

---

## 3. 设计方案

### 3.1 总体结构

```
gym.Env
  └── OrcaGymEulerEnv (单一 Env 类)
        │
        │   无状态 equality 原语（L1 公共 API，无 _ 前缀，进入 __dir__）：
        │     - equality_find_slot_by_body(body_name) -> int      # 按名查槽位
        │     - equality_constraint(slot) -> dict                  # 读单槽位完整数据
        │     - equality_update(slot, **fields, forward=True)      # 写单槽位字段 + 可选 mj_forward
        │     - set_mocap_pos_and_quat(dict)                       # 写 mocap 位姿（已有）
        │     - mj_forward() / get_body_xpos_xmat_xquat(names)     # 已有公共原语
        │   （Env 层不再暴露 update_equality_constraints / modify_equality_objects；
        │    SimCore.update_equality_constraints 作为 equality_update 底层实现保留）
        │
        │   UI 抓取业务编排（L2 内部 API，_ 前缀，不进入 __dir__）：
        │     - _do_body_manipulation()           # UI 状态机驱动，由 render() 调用
        │     - _anchor_actor(actor_name, type)   # 编排：find+snapshot+align+update（绑定）
        │     - _release_body_anchored()          # 编排：find+restore（释放）
        │   （原 update_anchor_equality_constraints 删除，编排逻辑内联到 _anchor_*）
        │
        │   状态字段（_ 前缀私有，UI 抓取业务状态）：
        │     - _anchor_mocap_name = "ActorManipulator_Anchor"（固定，UI 抓取专用）
        │     - _anchored_actor / _anchor_type / _anchor_original_eq
        │
        │   render() 内部调用 _do_body_manipulation()
```

**关键分层约束**：
- 公共原语**不持有任何跨调用状态**，每次调用即时读写。
- UI 抓取业务编排**内联组合公共原语**完成多步流程，业务状态存放在 `_anchor_*` 字段。
- 程序化操作（09 课等）**不调用** `_anchor_*` 内部方法，而是自行组合公共原语实现自己的业务编排与状态管理（参见 §4）。

### 3.2 公共无状态 equality 原语（L1）

在 `OrcaGymEulerEnv` 中新增/暴露以下公共方法。所有方法**单次原子读写**，不持有跨调用状态。

#### 3.2.1 `equality_find_slot_by_body(body_name) -> int`

查找含指定 body 的等式约束槽位索引。

```python
def equality_find_slot_by_body(self, body_name: str) -> int:
    """查找含指定 body 的等式约束槽位索引。

    遍历所有等式约束，返回第一个 obj1_id 或 obj2_id 等于该 body id 的槽位。
    未找到返回 -1。

    Args:
        body_name: body 名称（已含 agent 前缀）。

    .. note::
        本原语不做名称空间解析（对齐架构 §6.6 N1 的分工）。
        调用方应先用 ``env.body("pelvis")`` 解析出带 agent 前缀的完整名称，
        再传入本方法。示例::

            slot = env.equality_find_slot_by_body(env.body("pelvis"))
    """
    body_id = self.model.body_name2id(body_name)
    for i in range(self._gym.n_equality()):
        obj1, obj2 = self._gym.equality_object_ids(i)
        if obj1 == body_id or obj2 == body_id:
            return i
    return -1
```

#### 3.2.2 `equality_constraint(slot) -> dict`

读取单个等式约束完整数据（委托 Gym）。

```python
def equality_constraint(self, slot: int) -> dict:
    """读取单个等式约束完整数据。

    返回 type/obj1_id/obj2_id/active/solref/solimp/data。
    单次原子读，不持有状态。消费者需批量读取时自行循环调用本方法。
    """
    return self._gym.equality_constraint(slot)
```

> 注：原方案的 `equality_snapshot()`（批量读全部约束）不再纳入公共 API。它只是 `equality_constraint(i)` 的循环便利，且仅服务于 save/restore 业务模式。消费者需要时自行循环。

#### 3.2.3 `equality_update(slot, **fields, forward=True) -> None`

更新指定槽位的等式约束字段（单次原子写 + 可选 mj_forward）。

```python
def equality_update(
    self,
    slot: int,
    *,
    eq_type: int | None = None,
    obj1_name: str | None = None,
    obj2_name: str | None = None,
    data: np.ndarray | None = None,
    active: bool | None = None,
    solref: np.ndarray | None = None,
    solimp: np.ndarray | None = None,
    forward: bool = True,
) -> None:
    """更新指定槽位的等式约束字段。

    只修改显式传入的字段，未传入的字段保留原值。按当前 (obj1_id, obj2_id)
    匹配槽位写入。

    Args:
        slot: 等式约束槽位索引。
        eq_type: mjtEq 类型常量（可选）。
        obj1_name: 新的 obj1 body 名称（可选，内部解析为 id）。
        obj2_name: 新的 obj2 body 名称（可选，内部解析为 id）。
        data: 约束数据 np.ndarray（可选，形状 (mjNEQDATA,)）。
        active: 是否激活（可选，写入 eq_active0）。
        solref: 求解器参考参数 (2,)（可选，写入 eq_solref）。
        solimp: 求解器 impedance 参数 (3,)（可选，写入 eq_solimp）。
        forward: 是否在写入后调用 mj_forward()。默认 True，保证 env.data
            一致。若设为 False，调用方需自行调用 env.mj_forward() 才能读取
            一致的状态——这是高级用法，仅用于批量写入多个槽位时避免重复
            forward 的性能优化场景（典型如：连续 N 次 equality_update(..., forward=False)
            后，末尾调用一次 env.mj_forward()）。

    .. warning::
        ``forward=False`` 时写入已生效于 _mjModel，但 ``env.data``（OrcaGymDataView）
        未同步。此时若读取 ``env.data.body_xpos`` 等派生量将得到旧值，可能误导
        后续决策。仅在确认不读取派生量、或调用方将立即补 mj_forward() 时使用。

    .. note::
        本原语不做名称空间解析（对齐架构 §6.6 N1 的分工）。
        ``obj1_name`` / ``obj2_name`` 应为已含 agent 前缀的完整名称，
        调用方应先用 ``env.body("pelvis")`` 解析后再传入。示例::

            env.equality_update(
                slot,
                obj1_name=env.body("pelvis"),
                obj2_name=env.mocap("TestMocapAnchor"),
            )
    """
    eq = self.equality_constraint(slot)
    new_type = eq_type if eq_type is not None else eq["type"]
    new_obj1_id = (
        self.model.body_name2id(obj1_name) if obj1_name is not None else eq["obj1_id"]
    )
    new_obj2_id = (
        self.model.body_name2id(obj2_name) if obj2_name is not None else eq["obj2_id"]
    )
    new_data = data if data is not None else eq["data"]
    # 通过底层 SimCore.update_equality_constraints 写入 type/obj/data
    # （SimCore 层方法，作为 equality_update 的实现细节，不进入 Env 公共 API）
    self._gym.update_equality_constraints([{
        "type": new_type,
        "obj1_id": eq["obj1_id"],      # 用于匹配（当前值）
        "obj2_id": eq["obj2_id"],      # 用于匹配（当前值）
        "new_obj1_id": new_obj1_id,
        "new_obj2_id": new_obj2_id,
        "data": new_data,
    }])
    # active / solref / solimp 直接写 _mjModel.eq_*（无匹配语义，按 slot 索引）
    # 通过 SimCore 暴露的 typed 写入器委托，避免 Env 穿墙 _mjModel
    if active is not None:
        self._gym.set_equality_active(slot, active)
    if solref is not None:
        self._gym.set_equality_solref(slot, solref)
    if solimp is not None:
        self._gym.set_equality_solimp(slot, solimp)
    if forward:
        self.mj_forward()
```

> **实现注记**：`active` / `solref` / `solimp` 三字段无"匹配写入"语义（按 slot 索引直接写），与 type/obj/data 走 `update_equality_constraints` 的匹配写入不同。因此 `SimCore` 需新增 `set_equality_active` / `set_equality_solref` / `set_equality_solimp` 三个 typed 写入器（一行赋值级别），`OrcaGymEuler` 委托透传。这三个 SimCore 方法不进入 Env 公共 API，仅作为 `equality_update` 的实现细节。

> **`modify_equality_objects` / `update_equality_constraints` 不再在 Env 层暴露**：原 `OrcaGymEulerEnv.modify_equality_objects`（按索引批量改 obj id）功能被 `equality_update(slot, obj1_name=..., obj2_name=...)` 覆盖；原 `OrcaGymEulerEnv.update_equality_constraints`（按内容匹配批量全量写）是 `equality_update` 的底层实现。两者从 Env 公共 API 删除，`SimCore.update_equality_constraints` 保留作为 `equality_update` 底层实现。

### 3.3 UI 抓取业务编排（`_` 前缀，L2 内部 API）

将原 `OrcaGymEulerEnv` 的 UI 抓取方法改名为 `_` 前缀，docstring 标注内部 API。**不迁移到子类**，保留在同一 Env 类中。

**关键约束**：UI 抓取内部方法的约束操作**完全基于 §3.2 的公共原语**（`equality_find_slot_by_body` / `equality_constraint` / `equality_update` / `set_mocap_pos_and_quat` / `mj_forward`）组合实现，不直接调用 `self._gym.equality_*` 或 `self._gym.mj_forward`。绑定/释放的多步编排逻辑**内联**在 `_anchor_actor` / `_release_body_anchored` 中，不单独抽出 `equality_bind_mocap` / `equality_release` 公共方法。

#### 3.3.1 方法改名映射

| 原方法名 | 新方法名 | 说明 |
|---|---|---|
| `do_body_manipulation` | `_do_body_manipulation` | UI 状态机驱动，由 `render()` 内部调用 |
| `anchor_actor` | `_anchor_actor` | 锚定：内联编排 find+snapshot+align+update |
| `release_body_anchored` | `_release_body_anchored` | 释放：内联编排 find+restore |
| `update_anchor_equality_constraints` | **删除** | 编排逻辑内联到 `_anchor_actor` |
| `anchored_actor` property | `_anchored_actor_name` property（`_` 前缀） | UI 抓取内部状态查询 |

#### 3.3.2 `_anchor_actor` 实现（内联编排，作为消费者参考模式）

```python
def _anchor_actor(self, actor_name: str, anchor_type: str = "weld") -> None:
    """【内部 API】UI 抓取专用：锚定 actor body。

    .. warning::
        此方法是 Studio UI 抓取的内部实现，由 ``_do_body_manipulation`` 调用。
        AI 和用户代码**不应直接调用**此方法。
        程序化体操作请**仿照本方法的编排模式**，使用公共原语
        (``equality_find_slot_by_body`` / ``equality_constraint`` /
        ``equality_update`` / ``set_mocap_pos_and_quat``) 自行实现绑定流程。

    使用 Studio 系统自带的 ActorManipulator_Anchor mocap body，
    对齐 OrcaGymLocalEnv 的 anchor_actor 语义。

    编排流程（由公共无状态原语组合，业务状态存于 _anchor_* 字段）：
        1. equality_find_slot_by_body 查找含 mocap 的槽位
        2. equality_constraint 保存原始快照到 _anchor_original_eq
        3. get_body_xpos_xmat_xquat 读 actor 当前位姿
        4. set_mocap_pos_and_quat 把 mocap 对齐到 actor 位姿（避免下一帧拉扯）
        5. equality_update 把槽位另一端改为 actor，eq_type 改为目标类型

    Args:
        actor_name: 被锚定的 body 名称。
        anchor_type: 锚点类型 "weld"/"connect"。
    """
    import mujoco

    # 1. 查找含 UI 抓取专用 mocap 的槽位
    slot = self.equality_find_slot_by_body(self._anchor_mocap_name)
    if slot == -1:
        raise ValueError(
            f"模型中无含 {self._anchor_mocap_name} 的 equality 槽位，"
            f"请检查关卡 XML"
        )
    # 2. 保存原始约束快照（释放时恢复）
    self._anchor_original_eq = self.equality_constraint(slot)
    # 3. 对齐 mocap 位姿到 actor 当前位姿（避免下一帧拉扯）
    mocap_id = self.model.body_name2id(self._anchor_mocap_name)
    actor_pose = self.get_body_xpos_xmat_xquat([actor_name])[actor_name]
    self.set_mocap_pos_and_quat({
        self._anchor_mocap_name: {
            "pos": actor_pose["xpos"],
            "quat": actor_pose["xquat"],
        }
    })
    # 4. 确定改 obj1 还是 obj2（mocap 一端保持，另一端改为 actor）
    if self._anchor_original_eq["obj1_id"] == mocap_id:
        new_obj1_name = self._anchor_mocap_name
        new_obj2_name = actor_name
    else:
        new_obj1_name = actor_name
        new_obj2_name = self._anchor_mocap_name
    # 5. 映射 eq_type 字符串到常量并写入
    type_map = {
        "weld": mujoco.mjtEq.mjEQ_WELD,
        "connect": mujoco.mjtEq.mjEQ_CONNECT,
        "ball": mujoco.mjtEq.mjEQ_CONNECT,
    }
    mujoco_eq_type = type_map.get(anchor_type, mujoco.mjtEq.mjEQ_CONNECT)
    self.equality_update(
        slot,
        eq_type=mujoco_eq_type,
        obj1_name=new_obj1_name,
        obj2_name=new_obj2_name,
    )
    self._anchored_actor = actor_name
    self._anchor_type = anchor_type
```

#### 3.3.3 `_release_body_anchored` 实现（内联编排）

```python
def _release_body_anchored(self) -> None:
    """【内部 API】UI 抓取专用：释放锚定的 actor。

    .. warning::
        此方法是 Studio UI 抓取的内部实现，由 ``_do_body_manipulation`` 调用。
        AI 和用户代码**不应直接调用**此方法。
        程序化体操作请**仿照本方法的编排模式**，使用公共原语
        (``equality_find_slot_by_body`` / ``equality_update``) 自行实现释放流程。

    通过恢复 XML 原始 obj id 实现，对齐 Local 的 dummy body 机制。

    编排流程（由公共无状态原语组合）：
        1. equality_find_slot_by_body 查找当前绑定槽位
        2. equality_update 用 _anchor_original_eq 快照恢复原始 obj id / type / data
    """
    if self._anchored_actor is None:
        return
    if self._anchor_original_eq is not None:
        slot = self.equality_find_slot_by_body(self._anchored_actor)
        if slot != -1:
            # 用快照恢复原始约束（obj id / type / data）
            self.equality_update(
                slot,
                eq_type=self._anchor_original_eq["type"],
                obj1_name=self.model.body_id2name(self._anchor_original_eq["obj1_id"]),
                obj2_name=self.model.body_id2name(self._anchor_original_eq["obj2_id"]),
                data=self._anchor_original_eq["data"],
            )
    self._anchored_actor = None
    self._anchor_type = None
    self._anchor_original_eq = None
```

#### 3.3.4 `_do_body_manipulation` 实现

```python
def _do_body_manipulation(self) -> None:
    """【内部 API】Studio UI 抓取状态机。

    .. warning::
        此方法是 Studio UI 抓取的内部实现，由 ``render()`` 内部调用。
        AI 和用户代码**不应直接调用**此方法。

    流程：
    1. 读取 Studio body manipulation 状态
    2. 若 Studio 无锚定且本地已锚定：_release_body_anchored
    3. 若 Studio 有锚定且本地未锚定：_anchor_actor
    4. 已锚定时同步 mocap 到 UI 拖拽位姿（走 set_mocap_pos_and_quat 公共原语）

    注意：步骤 4 的 mj_forward 走 Env 层公共方法 self.mj_forward()，
    不直接调用 self._gym.mj_forward()。
    """
    if self._skip_grpc_load:
        return
    # get_body_manipulation_state 是 Studio Bridge 的 gRPC 状态查询，
    # 属于 UI 抓取特有输入源（非 equality 通用能力），Env 内部委托 self._gym
    manip_state = self.loop.run_until_complete(
        self._gym.get_body_manipulation_state()
    )
    actor_name = manip_state["actor_name"]
    anchor_type = manip_state["anchor_type"]
    if actor_name is None:
        if self._anchored_actor is not None:
            self._release_body_anchored()
        return
    if self._anchored_actor is None:
        self._anchor_actor(actor_name, anchor_type or "weld")
    if self._anchored_actor is not None and manip_state.get("mocap_pose"):
        self.set_mocap_pos_and_quat(
            {self._anchor_mocap_name: manip_state["mocap_pose"]}
        )
        self.mj_forward()  # Env 层公共方法，不穿墙 self._gym
```

#### 3.3.5 `render()` 内部调用

`render()` 方法中原本调用 `self.do_body_manipulation()`，改为调用 `self._do_body_manipulation()`：

```python
def render(self):
    # ... 渲染逻辑 ...
    if self._sync_render:
        self._render_count += self._render_count_interval
        if self._render_count >= 1.0:
            self.loop.run_until_complete(self._gym.render())
            self._do_body_manipulation()  # 改名
            self._render_count -= 1.0
    else:
        time_diff = time.perf_counter() - self._render_time_step
        if time_diff > self._render_interval:
            self._render_time_step = time.perf_counter()
            self.loop.run_until_complete(self._gym.render())
            self._do_body_manipulation()  # 改名
    return None
```

### 3.4 Env 状态字段保留与隔离

UI 抓取业务状态字段保留在 `OrcaGymEulerEnv`，均为 `_` 前缀私有：

```python
class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env):
    def __init__(self, ...):
        # UI 抓取专用业务状态（_ 前缀，不暴露）
        self._anchor_mocap_name: str = "ActorManipulator_Anchor"  # 固定
        self._anchored_actor: str | None = None
        self._anchor_type: str | None = None
        self._anchor_original_eq: dict | None = None
```

**关键约束**：
- 公共原语（§3.2）**不读写**任何 `_anchor_*` 字段，保持无状态。
- UI 抓取业务编排（§3.3）的状态存于 `_anchor_*` 字段，与公共原语解耦。
- 程序化操作（09 课等）**不在 Env 存放业务状态**，由消费者自管快照（参见 §4）。

### 3.5 `__dir__` 与可见性

```python
def __dir__(self):
    return [
        # ... 已有公共 API ...
        # 无状态 equality 原语（L1 公共）
        "equality_find_slot_by_body",
        "equality_constraint",
        "equality_update",
        # 已有公共原语
        "set_mocap_pos_and_quat",
        "mj_forward",
        "get_body_xpos_xmat_xquat",
    ]
    # 注意：_do_body_manipulation / _anchor_actor / _release_body_anchored
    # 不在 __dir__ 中，ruff SLF001 对外部访问报警
    # 注意：equality_bind_mocap / equality_release / equality_snapshot
    # 不再作为公共 API（业务编排归消费者实现）
    # 注意：update_equality_constraints / modify_equality_objects
    # 不再在 Env 层暴露（SimCore.update_equality_constraints 作为 equality_update
    # 底层实现保留；modify_equality_objects 功能被 equality_update 覆盖）
```

---

## 4. 09 课改造方案（消费者自行编排）

09 课**不调用** `_anchor_*` 内部方法，而是**仿照 `_anchor_actor` 的编排模式**，使用公共无状态原语自行实现绑定/移动/释放的业务编排，并自管业务状态。

### 4.1 设计要点

- **业务状态自管**：09 课 env 自持 `_bound_slot` / `_original_eq_snapshot`，不存放在 `OrcaGymEulerEnv`。
- **绑定前检查**：绑定入口检查 `_bound_slot is not None`（是否已绑定），避免重复绑定覆盖快照（修复原 bug）。
- **编排模式镜像 `_anchor_actor`**：find_slot → save_snapshot → align_mocap → update_constraint。

### 4.2 绑定编排（消费者实现）

```python
def _bind_mocap_to_pelvis(self, pelvis_short_name: str) -> None:
    """程序化绑定：把自备 mocap 绑定到 pelvis（仿照 _anchor_actor 编排模式）。

    使用公共无状态原语组合，业务状态自管。

    Args:
        pelvis_short_name: 不含 agent 前缀的 body 短名（如 "pelvis"），
                           内部用 env.body() 解析为完整名。
    """
    import mujoco

    # 幂等保护：已绑定时不重复绑定，避免覆盖快照
    if self._bound_slot is not None:
        return

    # 名称空间解析（架构 §6.6 N1：调用方负责解析前缀）
    pelvis_name = self.body(pelvis_short_name)              # "agent0/pelvis"
    mocap_name = self.mocap(self._mocap_short_name)         # "agent0/TestMocapAnchor"

    # 1. 查找含自备 mocap 的槽位（公共原语）
    slot = self.equality_find_slot_by_body(mocap_name)
    if slot == -1:
        raise ValueError(
            f"模型中无含 {mocap_name} 的 equality 槽位，"
            f"请检查关卡 XML"
        )
    # 2. 保存原始约束快照（消费者自管业务状态）
    self._original_eq_snapshot = self.equality_constraint(slot)
    self._bound_slot = slot
    # 3. 对齐 mocap 位姿到 pelvis 当前位姿（避免下一帧拉扯）
    mocap_id = self.model.body_name2id(mocap_name)
    pelvis_pose = self.get_body_xpos_xmat_xquat([pelvis_name])[pelvis_name]
    self.set_mocap_pos_and_quat({
        mocap_name: {
            "pos": pelvis_pose["xpos"],
            "quat": pelvis_pose["xquat"],
        }
    })
    # 4. 确定改 obj1 还是 obj2（mocap 一端保持，另一端改为 pelvis）
    if self._original_eq_snapshot["obj1_id"] == mocap_id:
        new_obj1_name = mocap_name
        new_obj2_name = pelvis_name
    else:
        new_obj1_name = pelvis_name
        new_obj2_name = mocap_name
    # 5. 写入约束（公共原语，内部 mj_forward）
    self.equality_update(
        slot,
        eq_type=mujoco.mjtEq.mjEQ_WELD,
        obj1_name=new_obj1_name,
        obj2_name=new_obj2_name,
    )
```

### 4.3 释放编排（消费者实现）

```python
def _release_mocap(self) -> None:
    """程序化释放：从快照恢复原始约束（仿照 _release_body_anchored 编排模式）。"""
    if self._bound_slot is None or self._original_eq_snapshot is None:
        return
    # 用快照恢复原始约束（公共原语，内部 mj_forward）
    self.equality_update(
        self._bound_slot,
        eq_type=self._original_eq_snapshot["type"],
        obj1_name=self.model.body_id2name(self._original_eq_snapshot["obj1_id"]),
        obj2_name=self.model.body_id2name(self._original_eq_snapshot["obj2_id"]),
        data=self._original_eq_snapshot["data"],
    )
    self._bound_slot = None
    self._original_eq_snapshot = None
```

### 4.4 移动阶段

```python
# 名称空间解析（与绑定编排一致，调用方负责解析前缀）
mocap_name = self.mocap(self._mocap_short_name)

for step in range(_PHASE_STEPS):
    # 控制策略由消费者决定：零力矩 / 低增益 PD / 行走策略
    ctrl = np.zeros(self.model.nu)
    self.do_simulation(ctrl, self.frame_skip)
    # 周期性移动 mocap（公共原语）
    progress = (step + 1) / _PHASE_STEPS
    target_pos = start_pos + delta * progress
    self.set_mocap_pos_and_quat({
        mocap_name: {
            "pos": target_pos.tolist(),
            "quat": [1, 0, 0, 0],
        }
    })
```

### 4.5 关键修正

| 原问题 | 修正方式 |
|---|---|
| 绑定瞬间未对齐 mocap | 消费者编排中显式 `set_mocap_pos_and_quat` 对齐 |
| UI 抓取状态被污染 | 09 课不使用 `_anchor_*` 字段，自管 `_bound_slot` / `_original_eq_snapshot` |
| 重复绑定覆盖快照（bug） | 绑定入口 `if self._bound_slot is not None: return` 幂等保护 |
| PD 控制器对抗约束 | 控制策略由消费者代码决定，框架不强制 |
| `equality_release` 误用 | 不再有该公共方法；消费者用 `equality_update` + 快照恢复，配对关系在消费者代码内显式可见 |

---

## 5. 实施步骤

### 阶段 1：新增公共无状态 equality 原语

1. 在 `OrcaGymEulerEnv` 新增 §3.2 的公共原语：`equality_find_slot_by_body` / `equality_constraint` / `equality_update`（含 active/solref/solimp 字段 + forward 开关）
2. 在 `MuJoCoSimCore` 新增 `set_equality_active` / `set_equality_solref` / `set_equality_solimp` 三个 typed 写入器（一行赋值级别），`OrcaGymEuler` 委托透传
3. **删除** `OrcaGymEulerEnv.update_equality_constraints` / `OrcaGymEulerEnv.modify_equality_objects` 公共方法（功能被 `equality_update` 覆盖；`SimCore.update_equality_constraints` 保留作为 `equality_update` 底层实现）
4. 更新 `__dir__` 暴露三个公共原语，移除 `update_equality_constraints` / `modify_equality_objects`
5. 单元测试：验证 `equality_find_slot_by_body` / `equality_constraint` / `equality_update` 行为正确（单次原子读写，无跨调用状态；forward=True/False 两种模式都验证）

### 阶段 2：UI 抓取方法改名与编排内联

1. 将 `do_body_manipulation` / `anchor_actor` / `release_body_anchored` 改名为 `_` 前缀
2. 重写 `_anchor_actor` / `_release_body_anchored`：把 bind/release 编排逻辑**内联**，仅调用公共原语（`equality_find_slot_by_body` / `equality_constraint` / `equality_update` / `set_mocap_pos_and_quat`）
3. 删除 `update_anchor_equality_constraints`（编排逻辑已内联到 `_anchor_actor`）
4. **删除**原公共方法 `equality_bind_mocap` / `equality_release` / `equality_snapshot`（若已存在），编排归消费者
5. **改造 `equality_release` 内部实现**（若 `equality_release` 已内联进 `_release_body_anchored` 则跳过）：将原直接调用 `self._gym.update_equality_constraints(...)` 的代码改为基于 `equality_update`（id→name 反查 + equality_update），不再穿墙 `_gym.update_equality_constraints`
6. `render()` 中调用点改为 `self._do_body_manipulation()`
7. `anchored_actor` property 改名 `_anchored_actor_name`（`_` 前缀）
8. docstring 追加 `.. warning::` 内部 API 标注，并提示"程序化操作请仿照本方法编排模式使用公共原语"

### 阶段 3：09 课改造（消费者自管编排）

1. 09 课 env 继承 `OrcaGymEulerEnv`（无需新子类）
2. 实现 `_bind_mocap_to_pelvis` / `_release_mocap`（§4.2 / §4.3），仿照 `_anchor_actor` 编排模式，使用公共原语
3. 绑定入口加 `if self._bound_slot is not None: return` 幂等保护
4. 移动/释放改用 §4.3 / §4.4 的自管编排
5. 移除诊断打印
6. 验证 bound_up / bound_forward / bound_left 三个方向都能正确拖动，选 4 释放后机器人能恢复自主行走

### 阶段 4：回归验证

1. UI 抓取流程（`render()` 驱动）功能不变
2. 09 课三个方向位移判定通过
3. 选 4 释放后约束恢复为 mocap↔box，机器人恢复自主行走
4. ruff SLF001 零报警
5. 现有测试全部通过

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 现有代码调用 `env.anchor_actor` / `env.release_body_anchored` 失效 | 全仓库搜索调用点，改为 `_` 前缀内部方法或公共原语组合 |
| 现有代码调用 `env.equality_bind_mocap` / `env.equality_release` 失效 | 全仓库搜索；调用方改为仿照 `_anchor_actor` 编排模式用公共原语自行实现 |
| 现有代码调用 `env.update_equality_constraints` / `env.modify_equality_objects` 失效 | 全仓库搜索 Euler 体系调用方；调用方改为 `equality_update(slot, ...)` 单槽位写入。Local/Remote 体系不动（与 Euler 无关），OrcaPlayground 调用方基于老框架不动 |
| `equality_update` 内部 `mj_forward` 增加开销 | 默认 forward=True 可接受（约束写入是低频操作）；批量场景用 forward=False + 末尾一次 mj_forward 优化 |
| `forward=False` 误用导致状态不一致 | docstring `.. warning::` 显式说明风险；单元测试验证 forward=False 后 env.data 派生量为旧值，补 mj_forward 后一致 |
| 09 课零力矩导致机器人摔倒 | 控制策略由消费者代码决定，框架不强制；可改用低增益 PD 维持姿态 |
| 消费者编排错误破坏约束 | 编排逻辑在消费者代码内显式可见，配对关系（save/restore）由消费者自管，比框架代管更易审查 |
| UI 抓取方法 `_` 前缀后 AI 仍尝试调用 | ruff SLF001 报警 + `__dir__` 隐藏 + docstring `.. warning::` 三重约束 |

---

## 7. 验收标准

| 编号 | 标准 | 验证方式 |
|------|------|---------|
| V1 | `OrcaGymEulerEnv` 不再包含无前缀的 `anchor_actor` / `release_body_anchored` / `do_body_manipulation` / `update_anchor_equality_constraints` | 源码审查 |
| V2 | `OrcaGymEulerEnv` 不再包含公共方法 `equality_bind_mocap` / `equality_release` / `equality_snapshot`（业务编排归消费者） | 源码审查 |
| V3 | `OrcaGymEulerEnv.__dir__` 包含无状态原语 `equality_find_slot_by_body` / `equality_constraint` / `equality_update`，**不包含** `update_equality_constraints` / `modify_equality_objects` | `dir(env)` 检查 |
| V3.1 | `equality_update` 签名支持 `eq_type` / `obj1_name` / `obj2_name` / `data` / `active` / `solref` / `solimp` / `forward` 参数 | 源码审查 |
| V3.2 | `equality_update(forward=False)` 后 `env.data` 派生量为旧值，补 `env.mj_forward()` 后一致 | 单元测试 |
| V3.3 | `MuJoCoSimCore` 含 `set_equality_active` / `set_equality_solref` / `set_equality_solimp` typed 写入器，`OrcaGymEuler` 委托透传 | 源码审查 |
| V3.4 | `OrcaGymEulerEnv` 不再含 `update_equality_constraints` / `modify_equality_objects` 公共方法（`SimCore.update_equality_constraints` 保留作为底层实现） | 源码审查 |
| V4 | `OrcaGymEulerEnv` 包含 `_anchor_actor` / `_release_body_anchored` / `_do_body_manipulation`（`_` 前缀） | 源码审查 |
| V4.1 | UI 抓取内部方法的约束操作完全基于公共原语（`equality_find_slot_by_body`/`equality_constraint`/`equality_update`/`set_mocap_pos_and_quat`/`mj_forward`），不直接调用 `self._gym.equality_*` 或 `self._gym.mj_forward` | 源码审查 |
| V5 | UI 抓取方法 docstring 含 `.. warning::` 内部 API 标注，并提示程序化操作仿照编排模式 | 源码审查 |
| V6 | 09 课自行实现 `_bind_mocap_to_pelvis` / `_release_mocap`，使用公共原语组合，不访问 `_anchor_*` 字段 | 09 课源码审查 |
| V6.1 | 09 课绑定入口有 `if self._bound_slot is not None: return` 幂等保护 | 09 课源码审查 |
| V7 | 09 课 bound_up / bound_forward / bound_left 位移判定通过 | 运行 09 课 |
| V7.1 | 09 课选 4 释放后约束恢复为 mocap↔box，机器人恢复自主行走 | 运行 09 课 |
| V8 | UI 抓取流程功能不变（`render()` 驱动） | 手动 UI 锁定/拖拽/释放测试 |
| V9 | ruff SLF001 零报警 | `ruff check --select SLF001 orca_gym/` |
| V10 | 现有单元测试通过 | `pytest tests/orca_gym/core/euler/` |
