# OrcaGym Euler 锚点与等式约束 API 重构开发方案

## 1. 背景与目标

### 1.1 问题陈述

当前 `OrcaGymEulerEnv` 中的体操作能力存在两类职责混叠在一个类里：

1. **Studio UI 抓取专用逻辑**：`do_body_manipulation` / `anchor_actor` / `release_body_anchored` / `update_anchor_equality_constraints`，这组方法在内部耦合了 UI 抓取的特定假设（固定使用 `ActorManipulator_Anchor`、跟随 `manip_state["mocap_pose"]`、释放时恢复 XML 原始 obj id 模拟 dummy body 机制）。
2. **用户程序化操作需求的通用能力**：09 课等示例需要在用户代码中通过自备 mocap body + equality 把机器人某个 body 程序化拖动到目标位姿。

二者目前共用同一个 `update_anchor_equality_constraints(actor_name, anchor_type)` 入口，参数 `actor_name`/`anchor_type` 已暗含 UI 抓取语义；09 课被迫借用此入口并改写 `_anchor_mocap_name`，结果出现了：

- mocap body 与 equality 槽位错配（设置 A mocap 的位姿，绑定 B mocap 的约束）
- 绑定瞬间未将 mocap 对齐到 actor 当前位姿，下一帧约束把 actor 猛拉到 mocap 初始位置
- UI 抓取流程的状态字段（`_anchored_actor` / `_anchor_original_eq`）被用户代码污染，UI 再次介入时行为不可预测

### 1.2 重构目标

| 目标 | 含义 |
|------|------|
| **G1 通用 equality 能力下沉为公共 API** | 在 Env 层提供不依赖 UI 抓取语义的 equality 读/写/mocap 联动方法，供用户代码（lesson 9、OrcaManipulation 等）直接使用 |
| **G2 UI 抓取能力标识为内部 API** | `do_body_manipulation` / `anchor_actor` / `release_body_anchored` 等 UI 专用方法改名为 `_` 前缀，docstring 标注"内部 API，AI/用户不应调用"，保留在同一 Env 类中（无需继承新子类） |
| **G3 状态字段隔离** | UI 抓取状态字段（`_anchored_actor` 等）保持 `_` 前缀私有；通用 equality 操作不依赖这些字段，调用方自管快照 |
| **G4 遵从架构契约** | 不暴露 `_mjModel`/`_mjData`；所有读写走 Env→Gym→SimCore/Registry 三层委托；通用方法遵循 L1 公共 API 规范，UI 抓取方法遵循 L2 内部 API 规范，受 `__dir__` + ruff SLF001 约束 |

### 1.3 非目标

- 不改动 `MuJoCoSimCore.update_equality_constraints` / `modify_equality_objects` 的底层写入语义（已正确实现按 `(obj1_id, obj2_id)` 匹配槽位写入）
- 不改动 `ModelRegistry.equality_constraint` / `equality_object_ids` / `n_equality` 的读取语义（已正确返回完整字段）
- 不重构 `OrcaStudioBridge.get_body_manipulation_state` 的 gRPC 协议
- 不引入子类继承体系（`OrcaGymEulerInteractiveEnv`），避免所有 UI 抓取用户被迫继承新类
- 不引入新的架构组件（如新的 `AnchorManager` 子组件）—— 在现有组件内职责内聚即可

---

## 2. 架构约束对齐

本方案严格遵守 `docs/design/architecture/orca_gym_euler_architecture.md` 的以下约束：

### 2.1 组件划分（架构 §5）

- 通用 equality 能力是 `OrcaGymEulerEnv` 的**公共方法**（L1），委托到 `OrcaGymEuler` 的对应方法，再委托到 `MuJoCoSimCore` / `ModelRegistry`。
- UI 抓取能力保留在 `OrcaGymEulerEnv` 内，作为**内部方法**（L2，`_` 前缀），不单独划分子类。

### 2.2 API 契约（架构 §6）

| 契约 | 本方案对齐方式 |
|------|---------------|
| **R1 状态读取** | 通用 equality 读取通过 `env.equality_*()` 公共方法，不穿墙 `_mjModel.eq_*` |
| **W1 状态写入** | 通用 equality 写入通过 `env.equality_*()` 公共方法，不穿墙 `_mjModel.eq_*` |
| **W3 写后一致** | 通用 equality 写入方法内部调用 `mj_forward()`，保证 `env.data` 一致 |
| **C1 求解器配置** | 不涉及 |
| **N1 名称空间** | 通用 equality 方法接收 body **名称**（非 id），内部通过 `self.model.body_name2id` 解析 |
| **L1/L2/L3 层级** | UI 抓取方法为 `_` 前缀内部方法（L2），不进入 `__dir__`；通用 equality 方法为公共方法（L1），进入 `__dir__` |

### 2.3 封装隔离（架构 §7）

| 机制 | 本方案对齐方式 |
|------|---------------|
| **M1 ruff SLF001** | 通用 equality 方法为公共方法（无 `_` 前缀），UI 抓取方法为 `_` 前缀内部方法，SLF001 对外部访问 UI 抓取方法报警 |
| **M3 `__dir__`** | Env `__dir__` 只暴露通用 equality 方法；UI 抓取方法不进入 `__dir__` |
| **M6 docstring** | 通用方法 docstring 列出正确用法；UI 抓取方法 docstring 标注"内部 API，AI/用户不应调用，由 `render()` 内部驱动" |

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
        │   通用 equality 方法（L1 公共 API，无 _ 前缀，进入 __dir__）：
        │     - equality_snapshot()
        │     - equality_find_slot_by_body(body_name)
        │     - equality_constraint(slot)
        │     - equality_update(slot, **fields)
        │     - equality_bind_mocap(mocap_name, body_name, eq_type)
        │     - equality_release(slot, original_snapshot)
        │     - mocap_body_names / set_mocap_pos_and_quat（已有）
        │
        │   UI 抓取方法（L2 内部 API，_ 前缀，不进入 __dir__）：
        │     - _do_body_manipulation()
        │     - _anchor_actor(actor_name, anchor_type)
        │     - _release_body_anchored()
        │   （原 update_anchor_equality_constraints 删除，逻辑被 equality_bind_mocap 取代）
        │
        │   状态字段（_ 前缀私有）：
        │     - _anchor_mocap_name = "ActorManipulator_Anchor"（固定，UI 抓取专用）
        │     - _anchored_actor / _anchor_type / _anchor_original_eq
        │
        │   render() 内部调用 _do_body_manipulation()
```

### 3.2 基础 Env 通用 equality API

在 `OrcaGymEulerEnv` 中新增以下公共方法。所有方法**不持有任何 Env 级缓存状态**，每次调用即时读写，调用方自行管理快照与恢复。

#### 3.2.1 `equality_snapshot() -> list[dict]`

读取所有等式约束的完整数据快照。

```python
def equality_snapshot(self) -> list[dict]:
    """读取所有等式约束的完整数据快照。

    返回 list[dict]，每项含 type/obj1_id/obj2_id/active/solref/solimp/data。
    用于保存约束原始状态，便于后续恢复。

    替代直接访问 _mjModel.eq_*。
    """
    return [
        self._gym.equality_constraint(i)
        for i in range(self._gym.n_equality())
    ]
```

#### 3.2.2 `equality_find_slot_by_body(body_name) -> int`

查找含指定 body 的等式约束槽位索引。

```python
def equality_find_slot_by_body(self, body_name: str) -> int:
    """查找含指定 body 的等式约束槽位索引。

    遍历所有等式约束，返回第一个 obj1_id 或 obj2_id 等于该 body id 的槽位。
    未找到返回 -1。

    Args:
        body_name: body 名称（已含 agent 前缀）。
    """
    body_id = self.model.body_name2id(body_name)
    for i in range(self._gym.n_equality()):
        obj1, obj2 = self._gym.equality_object_ids(i)
        if obj1 == body_id or obj2 == body_id:
            return i
    return -1
```

#### 3.2.3 `equality_update(slot, **fields) -> None`

更新指定槽位的等式约束字段。

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
) -> None:
    """更新指定槽位的等式约束字段。

    只修改显式传入的字段，未传入的字段保留原值。

    Args:
        slot: 等式约束槽位索引。
        eq_type: mjtEq 类型常量（可选）。
        obj1_name: 新的 obj1 body 名称（可选，内部解析为 id）。
        obj2_name: 新的 obj2 body 名称（可选，内部解析为 id）。
        data: 约束数据 np.ndarray（可选，形状 (mjNEQDATA,)）。
        active: 激活状态（可选）。

    写入后自动调用 mj_forward()，保证 env.data 一致。
    """
    # 读取当前槽位完整数据，合并传入字段，回写
    eq = self._gym.equality_constraint(slot)
    new_type = eq_type if eq_type is not None else eq["type"]
    new_obj1_id = (
        self.model.body_name2id(obj1_name) if obj1_name is not None else eq["obj1_id"]
    )
    new_obj2_id = (
        self.model.body_name2id(obj2_name) if obj2_name is not None else eq["obj2_id"]
    )
    new_data = data if data is not None else eq["data"]
    # 用 (obj1_id, obj2_id) 匹配槽位写入
    self._gym.update_equality_constraints([{
        "type": new_type,
        "obj1_id": eq["obj1_id"],      # 用于匹配（当前值）
        "obj2_id": eq["obj2_id"],      # 用于匹配（当前值）
        "new_obj1_id": new_obj1_id,
        "new_obj2_id": new_obj2_id,
        "data": new_data,
    }])
    self._gym.mj_forward()
```

#### 3.2.4 `equality_bind_mocap(mocap_name, body_name, eq_type) -> int`

把 mocap body 与指定 body 通过 equality 绑定，返回绑定的槽位索引。

```python
def equality_bind_mocap(
    self,
    mocap_name: str,
    body_name: str,
    eq_type: str = "weld",
) -> int:
    """把 mocap body 与指定 body 通过 equality 绑定。

    语义：
    1. 查找含 mocap_name 的 equality 槽位
    2. 将 mocap 位姿对齐到 body 当前位姿（避免下一帧拉扯）
    3. 把槽位的另一端 obj 改为 body，eq_type 改为指定类型
    4. 保留 XML 原始 eq_data/solref/solimp（MuJoCo 编译器推导值）

    Args:
        mocap_name: mocap body 名称（已含 agent 前缀）。
        body_name: 被绑定的 body 名称（已含 agent 前缀）。
        eq_type: "weld" 或 "connect"。

    Returns:
        绑定的等式约束槽位索引。

    Raises:
        ValueError: 模型无 equality 槽位，或未找到含 mocap_name 的槽位。
    """
    import mujoco

    # 1. 查找含 mocap 的槽位
    slot = self.equality_find_slot_by_body(mocap_name)
    if slot == -1:
        raise ValueError(
            f"未在等式约束中找到含 mocap body '{mocap_name}' 的槽位，"
            f"请在 XML 中预定义 <equality><weld body1='{mocap_name}' "
            f"body2='...'/></equality>"
        )
    # 2. 读取槽位完整数据
    eq = self._gym.equality_constraint(slot)
    mocap_id = self.model.body_name2id(mocap_name)
    body_id = self.model.body_name2id(body_name)
    # 3. 对齐 mocap 位姿到 body 当前位姿（避免下一帧拉扯）
    body_pose = self.get_body_xpos_xmat_xquat([body_name])[body_name]
    self.set_mocap_pos_and_quat({
        mocap_name: {
            "pos": body_pose["xpos"],
            "quat": body_pose["xquat"],
        }
    })
    # 4. 确定改 obj1 还是 obj2
    if eq["obj1_id"] == mocap_id:
        new_obj1_name = mocap_name
        new_obj2_name = body_name
    else:
        new_obj1_name = body_name
        new_obj2_name = mocap_name
    # 5. 映射 eq_type 字符串到常量
    type_map = {
        "weld": mujoco.mjtEq.mjEQ_WELD,
        "connect": mujoco.mjtEq.mjEQ_CONNECT,
        "ball": mujoco.mjtEq.mjEQ_CONNECT,
    }
    mujoco_eq_type = type_map.get(eq_type, mujoco.mjtEq.mjEQ_CONNECT)
    # 6. 写入（保留原始 data/solref/solimp）
    self.equality_update(
        slot,
        eq_type=mujoco_eq_type,
        obj1_name=new_obj1_name,
        obj2_name=new_obj2_name,
    )
    return slot
```

#### 3.2.5 `equality_release(slot, original_snapshot) -> None`

从快照恢复指定槽位的原始约束。

```python
def equality_release(
    self,
    slot: int,
    original_snapshot: dict,
) -> None:
    """从快照恢复指定槽位的等式约束。

    用于释放绑定：把槽位的 obj id / eq_type / data 恢复到绑定前的原始值。

    Args:
        slot: 等式约束槽位索引。
        original_snapshot: equality_snapshot()[slot] 或 equality_constraint(slot)
                          返回的 dict，含 type/obj1_id/obj2_id/data 等字段。
    """
    # 读取当前槽位（用于匹配）
    cur_eq = self._gym.equality_constraint(slot)
    # 用当前 obj id 匹配，写入原始值
    self._gym.update_equality_constraints([{
        "type": original_snapshot["type"],
        "obj1_id": cur_eq["obj1_id"],             # 用于匹配（当前值）
        "obj2_id": cur_eq["obj2_id"],             # 用于匹配（当前值）
        "new_obj1_id": original_snapshot["obj1_id"],  # 恢复原始
        "new_obj2_id": original_snapshot["obj2_id"],  # 恢复原始
        "data": original_snapshot["data"],
    }])
    self._gym.mj_forward()
```

#### 3.2.6 `equality_constraint(slot) -> dict`（已有，Env 层暴露）

`self._gym.equality_constraint(slot)` 已存在。在 Env 层新增同名公共方法委托：

```python
def equality_constraint(self, slot: int) -> dict:
    """读取单个等式约束完整数据（委托 Gym）。

    返回 type/obj1_id/obj2_id/active/solref/solimp/data。
    """
    return self._gym.equality_constraint(slot)
```

### 3.3 UI 抓取方法（`_` 前缀，L2 内部 API）

将原 `OrcaGymEulerEnv` 的 UI 抓取方法改名为 `_` 前缀，docstring 标注内部 API。**不迁移到子类**，保留在同一 Env 类中。

**关键约束**：UI 抓取内部方法**完全基于通用 equality API 实现**，不直接调用 `self._gym.equality_*` / `self._gym.mj_forward` 等底层方法。所有约束操作通过 §3.2 的公共方法完成，mocap 操作通过 `set_mocap_pos_and_quat`，前向求解通过 `mj_forward`（Env 层公共方法）。

#### 3.3.1 方法改名映射

| 原方法名 | 新方法名 | 说明 |
|---|---|---|
| `do_body_manipulation` | `_do_body_manipulation` | UI 状态机驱动，由 `render()` 内部调用 |
| `anchor_actor` | `_anchor_actor` | 锚定：调 `equality_bind_mocap` |
| `release_body_anchored` | `_release_body_anchored` | 释放：调 `equality_release` |
| `update_anchor_equality_constraints` | **删除** | 逻辑已被 `equality_bind_mocap` 取代 |
| `anchored_actor` property | `_anchored_actor_name` property（`_` 前缀） | UI 抓取内部状态查询 |

#### 3.3.2 `_anchor_actor` 实现

```python
def _anchor_actor(self, actor_name: str, anchor_type: str = "weld") -> None:
    """【内部 API】UI 抓取专用：锚定 actor body。

    .. warning::
        此方法是 Studio UI 抓取的内部实现，由 `_do_body_manipulation` 调用。
        AI 和用户代码**不应直接调用**此方法。
        程序化体操作请使用公共 API `equality_bind_mocap()`。

    使用 Studio 系统自带的 ActorManipulator_Anchor mocap body，
    对齐 OrcaGymLocalEnv 的 anchor_actor 语义。

    实现完全基于通用 equality API：
    - equality_find_slot_by_body 查找槽位
    - equality_constraint 保存原始快照
    - equality_bind_mocap 完成绑定（含 mocap 对齐）

    Args:
        actor_name: 被锚定的 body 名称。
        anchor_type: 锚点类型 "weld"/"connect"。
    """
    # 1. 查找 UI 抓取专用 mocap 的 equality 槽位
    slot = self.equality_find_slot_by_body(self._anchor_mocap_name)
    if slot == -1:
        raise ValueError(
            f"模型中无含 {self._anchor_mocap_name} 的 equality 槽位，"
            f"请检查关卡 XML"
        )
    # 2. 保存原始约束快照（释放时恢复，走通用 equality_constraint 公共方法）
    self._anchor_original_eq = self.equality_constraint(slot)
    # 3. 绑定（通用 API 内部完成 mocap 对齐 + 约束写入 + mj_forward）
    self.equality_bind_mocap(
        self._anchor_mocap_name, actor_name, anchor_type
    )
    self._anchored_actor = actor_name
    self._anchor_type = anchor_type
```

#### 3.3.3 `_release_body_anchored` 实现

```python
def _release_body_anchored(self) -> None:
    """【内部 API】UI 抓取专用：释放锚定的 actor。

    .. warning::
        此方法是 Studio UI 抓取的内部实现，由 `_do_body_manipulation` 调用。
        AI 和用户代码**不应直接调用**此方法。
        程序化体操作请使用公共 API `equality_release()`。

    通过恢复 XML 原始 obj id 实现，对齐 Local 的 dummy body 机制。

    实现完全基于通用 equality API：
    - equality_find_slot_by_body 查找当前绑定槽位
    - equality_release 从快照恢复原始约束
    """
    if self._anchored_actor is None:
        return
    if self._anchor_original_eq is not None:
        slot = self.equality_find_slot_by_body(self._anchored_actor)
        if slot != -1:
            self.equality_release(slot, self._anchor_original_eq)
    self._anchored_actor = None
    self._anchor_type = None
    self._anchor_original_eq = None
```

#### 3.3.4 `_do_body_manipulation` 实现

```python
def _do_body_manipulation(self) -> None:
    """【内部 API】Studio UI 抓取状态机。

    .. warning::
        此方法是 Studio UI 抓取的内部实现，由 `render()` 内部调用。
        AI 和用户代码**不应直接调用**此方法。

    流程：
    1. 读取 Studio body manipulation 状态
    2. 若 Studio 无锚定且本地已锚定：_release_body_anchored
    3. 若 Studio 有锚定且本地未锚定：_anchor_actor
    4. 已锚定时同步 mocap 到 UI 拖拽位姿（走 set_mocap_pos_and_quat 公共方法）

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

UI 抓取状态字段保留在 `OrcaGymEulerEnv`，均为 `_` 前缀私有：

```python
class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env):
    def __init__(self, ...):
        # UI 抓取专用状态（_ 前缀，不暴露）
        self._anchor_mocap_name: str = "ActorManipulator_Anchor"  # 固定
        self._anchored_actor: str | None = None
        self._anchor_type: str | None = None
        self._anchor_original_eq: dict | None = None
```

**关键约束**：通用 equality 方法（§3.2）**不读写**这些 `_anchor_*` 字段，调用方自管快照。这样 UI 抓取状态与程序化操作完全解耦。

### 3.5 `__dir__` 与可见性

```python
def __dir__(self):
    return [
        # ... 已有公共 API ...
        # 通用 equality API（新增，L1 公共）
        "equality_snapshot", "equality_find_slot_by_body",
        "equality_constraint", "equality_update",
        "equality_bind_mocap", "equality_release",
    ]
    # 注意：_do_body_manipulation / _anchor_actor / _release_body_anchored
    # 不在 __dir__ 中，ruff SLF001 对外部访问报警
```

---

## 4. 09 课改造方案

09 课改用通用 equality API，不再借用 UI 抓取入口。

### 4.1 绑定阶段

```python
# 绑定前保存原始约束快照
slot = self.equality_find_slot_by_body(self._mocap_body_name)
self._original_eq_snapshot = self.equality_constraint(slot)

# 绑定（通用 API 内部完成 mocap 对齐 + 约束写入）
self.equality_bind_mocap(
    self._mocap_body_name, pelvis_name, eq_type="weld"
)
self._bound_slot = slot
```

### 4.2 移动阶段

```python
for step in range(_PHASE_STEPS):
    # 零力矩或低增益 PD（让约束能拖动）
    ctrl = np.zeros(self.model.nu)
    self.do_simulation(ctrl, self.frame_skip)
    # 周期性移动 mocap
    progress = (step + 1) / _PHASE_STEPS
    target_pos = start_pos + delta * progress
    self.set_mocap_pos_and_quat({
        self._mocap_body_name: {
            "pos": target_pos.tolist(),
            "quat": [1, 0, 0, 0],
        }
    })
```

### 4.3 释放阶段

```python
# 从快照恢复原始约束
self.equality_release(self._bound_slot, self._original_eq_snapshot)
```

### 4.4 关键修正

| 原问题 | 修正 |
|---|---|
| 绑定瞬间未对齐 mocap | `equality_bind_mocap` 内部自动对齐 |
| UI 抓取状态被污染 | 09 课不使用 `_anchor_*` 字段，自管 `_original_eq_snapshot` |
| PD 控制器对抗约束 | 绑定阶段使用零力矩或低增益 PD（由用户代码决定，API 不强制） |

---

## 5. 实施步骤

### 阶段 1：新增通用 equality API

1. 在 `OrcaGymEulerEnv` 新增 §3.2 的 6 个公共方法
2. 更新 `__dir__` 暴露这些方法
3. 单元测试：验证 `equality_snapshot` / `equality_find_slot_by_body` / `equality_update` / `equality_bind_mocap` / `equality_release` 行为正确

### 阶段 2：UI 抓取方法改名与重写

1. 将 `do_body_manipulation` / `anchor_actor` / `release_body_anchored` 改名为 `_` 前缀
2. 重写 `_anchor_actor` / `_release_body_anchored` 使用通用 API（`equality_bind_mocap` / `equality_release`）
3. 删除 `update_anchor_equality_constraints`（逻辑已被 `equality_bind_mocap` 取代）
4. `render()` 中调用点改为 `self._do_body_manipulation()`
5. `anchored_actor` property 改名 `_anchored_actor_name`（`_` 前缀）
6. docstring 追加 `.. warning::` 内部 API 标注

### 阶段 3：09 课改造

1. 09 课 env 继承 `OrcaGymEulerEnv`（无需新子类）
2. 绑定/移动/释放改用 §4 的通用 API
3. 移除诊断打印
4. 验证 bound_up / bound_forward / bound_left 三个方向都能正确拖动

### 阶段 4：回归验证

1. UI 抓取流程（`render()` 驱动）功能不变
2. 09 课三个方向位移判定通过
3. ruff SLF001 零报警
4. 现有测试全部通过

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 现有代码调用 `env.anchor_actor` / `env.release_body_anchored` 失效 | 全仓库搜索调用点，改为 `_` 前缀内部方法或通用 API |
| `equality_bind_mocap` 内部 `mj_forward` 增加开销 | 可接受，绑定是低频操作；若需批量操作再优化 |
| 09 课零力矩导致机器人摔倒 | 由用户代码决定控制策略，API 不强制；可改用低增益 PD 维持姿态 |
| 通用 API 暴露后用户误用破坏约束 | docstring 明确语义；`equality_release` 需传入快照，强制配对使用 |
| UI 抓取方法 `_` 前缀后 AI 仍尝试调用 | ruff SLF001 报警 + `__dir__` 隐藏 + docstring `.. warning::` 三重约束（不依赖 AGENTS.md 显式条目，避免规则文档臃肿） |

---

## 7. 验收标准

| 编号 | 标准 | 验证方式 |
|------|------|---------|
| V1 | `OrcaGymEulerEnv` 不再包含无前缀的 `anchor_actor` / `release_body_anchored` / `do_body_manipulation` / `update_anchor_equality_constraints` | 源码审查 |
| V2 | `OrcaGymEulerEnv.__dir__` 包含 6 个通用 equality 方法 | `dir(env)` 检查 |
| V3 | `OrcaGymEulerEnv` 包含 `_anchor_actor` / `_release_body_anchored` / `_do_body_manipulation`（`_` 前缀） | 源码审查 |
| V3.1 | UI 抓取内部方法的约束操作完全基于通用 equality API（`equality_bind_mocap`/`equality_release`/`equality_constraint`/`equality_find_slot_by_body`），不直接调用 `self._gym.equality_*` 或 `self._gym.mj_forward` | 源码审查 |
| V4 | UI 抓取方法 docstring 含 `.. warning::` 内部 API 标注 | 源码审查 |
| V5 | 09 课使用 `equality_bind_mocap` / `equality_release`，不访问 `_anchor_*` 字段 | 09 课源码审查 |
| V6 | 09 课 bound_up / bound_forward / bound_left 位移判定通过 | 运行 09 课 |
| V7 | UI 抓取流程功能不变（`render()` 驱动） | 手动 UI 锁定/拖拽/释放测试 |
| V8 | ruff SLF001 零报警 | `ruff check --select SLF001 orca_gym/` |
| V9 | 现有单元测试通过 | `pytest tests/orca_gym/core/euler/` |
