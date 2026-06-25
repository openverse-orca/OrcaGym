# OrcaGym Euler 阶段二开发指导文档：端到端功能填充

## 1. 文档定位

### 1.1 文档目标

本文是 `OrcaGymEulerEnv` + `OrcaGymEuler` **阶段二（功能填充）** 的开发指导文档。在骨架阶段（P1–P3，见 [orca_gym_euler_skeleton_development.md](orca_gym_euler_skeleton_development.md)）已完成并通过验收的基础上，**小范围填充功能**，实现 [TUTORIAL.md](../../../../OrcaPlayground/examples/euler/TUTORIAL.md) 中前三个 example 的端到端运行。

> **上游约束**：架构文档 `docs/design/architecture/orca_gym_euler_architecture.md`（§5–§7、§12 为硬性约束）。本文所有填充实现必须严格遵守 K1–K12 约束，不得回退到上帝类 + 封装泄漏的老路。

### 1.2 阶段二范围

阶段二**不是**完整的 P4（64 个 `query_*/set_*` 方法 + 12 个 Studio 交互方法），而是 P4 的**最小可用子集**——仅填充三个 example 端到端运行所必需的方法。

| 范围 | 阶段二包含 | 阶段二不包含（留待完整 P4） |
|------|-----------|--------------------------|
| 仿真核心 | `init_simulation`/`step`/`forward`/`set_ctrl`/`sync_to_view`/`reset_data` 真实实现 | `apply_body_force`/`clear_*`（Lesson 5） |
| 状态视图 | 5 基本字段 + `time` 同步；`body_xpos`/`site_xpos` 等按需查询 | 完整 body/site/geom 查询（Lesson 4） |
| 模型注册 | `build_orca_gym_model` 真实构建 | `body_subtree_mass`/`equality_*` 查询（Lesson 4/5） |
| 求解器配置 | `timestep`/`integrator`/`iterations`/`gravity` 接入真实 `mjModel.opt` | 其余 `opt` 字段 |
| Studio 交互 | `render`/`load_model_xml`/`pause_simulation`/`configure_offline`/`set_timestep_remote`/`get_body_manipulation_*` | 视频/帧/内容文件方法 |
| 状态设置 | `set_joint_qpos`/`set_joint_qvel`（reset 必需） | `set_mocap_pos_and_quat`/`update_equality_constraints`/`set_geom_friction` |
| 状态查询 | 无（三个 example 均通过 `env.data` 直接读 qpos/qvel） | `query_joint_*`/`query_site_*`/`query_sensor_*`/`query_contact_*`（Lesson 4） |

### 1.3 三个 Example 与填充阶段的映射

| Example | 教程阶段 | 依赖的填充内容 | 需要 OrcaStudio |
|---------|---------|--------------|----------------|
| 01_hello_euler | P3 | 阶段 2.1（离线仿真核心） | 否 |
| 02_online_render | P3A | 阶段 2.1 + 阶段 2.2（gRPC 渲染） | 是 |
| 03_rl_ppo | P3B | 阶段 2.1（离线仿真核心）+ Gymnasium 契约 | 否 |

> **关键观察**：Lesson 3（RL PPO）不需要额外的填充内容，它复用 Lesson 1 的离线仿真核心 + `SimpleEulerEnv` 已有的 Gymnasium API 契约。阶段 2.3 仅做端到端验证。

### 1.4 填充原则

1. **骨架约束不可回退**：所有填充实现必须保持 K1–K12 约束。Env 内部只走 `self._gym` 公共方法，不触私有；`_mjModel`/`_mjData` 只存在于 `MuJoCoSimCore` 内部。
2. **委托链路完整**：`Env → Gym → SimCore` 的委托路径必须显式，每层只调下层公共方法。
3. **Example 违规必须修正**：`SimpleEulerEnv.reset_model` 中的 `self.gym._sim._mjData.qpos[:] = qpos` 是 K3/K5 违规，必须改为 `env.set_joint_qpos()`/`env.set_joint_qvel()`。
4. **不过度设计**：仅填充三个 example 需要的方法，不提前实现 Lesson 4/5 的 `query_*`/`apply_body_force`。

---

## 2. 现状与差距分析

### 2.1 骨架现状（P1–P3 已交付）

| 组件 | 状态 | 待填充方法 |
|------|------|-----------|
| `SimConfig` | 占位字段（不接 `mjModel.opt`） | property 委托 `_mj_model.opt.*` |
| `OrcaGymDataView` | 空数组占位 | 接入真实 `mjData`，实现 body/site 查询 |
| `ModelRegistry` | `raise NotImplementedError` | `build_orca_gym_model` 真实构建 |
| `MuJoCoSimCore` | `raise NotImplementedError` | `init_simulation`/`step`/`forward`/`set_ctrl`/`sync_to_view`/`reset_data` |
| `OrcaStudioBridge` | `raise NotImplementedError` | 7 个 gRPC 方法真实实现 |
| `OrcaGymEuler` | `raise NotImplementedError` | 委托填充 + `model` property |
| `OrcaGymEulerEnv` | 离线骨架模式（no-op） | 生命周期 + 步进 + 状态设置 + 渲染 |

### 2.2 Example 违规点

**文件**：`OrcaPlayground/envs/euler/simple_env.py`

```python
# ❌ K3/K5 违规：穿墙访问 _sim._mjData
def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
    qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
    self.gym._sim._mjData.qpos[:] = qpos      # 违规
    self.gym._sim._mjData.qvel[:] = qvel      # 违规
    self.gym.mj_forward()                      # 违规：应通过 env 委托
    self.gym.sync_to_view()                    # 违规：应通过 env 委托
    ...
```

**修正方案**：阶段 2.1 Step 7 提供合规 API 替换。

### 2.3 Example 对 API 的依赖清单

通过分析三个 example 脚本，梳理实际调用的 API：

| API | 01_hello | 02_online | 03_rl_ppo | 填充归属 |
|-----|:--------:|:---------:|:---------:|---------|
| `SimpleEulerEnv(skip_grpc_load=...)` | ✅ | ✅ | ✅ | 构造器 |
| `env.model.nq/nv/nu` | ✅ | ✅ | ✅ | 2.1 Step 3/5/6 |
| `env.data.qpos/qvel/time` | ✅ | ✅ | ✅ | 2.1 Step 4/5 |
| `env.sim_config.timestep/integrator` | ✅ | — | — | 2.1 Step 2 |
| `env.reset()` → `reset_model` | ✅ | ✅ | ✅ | 2.1 Step 6/7 |
| `env.step(action)` → `do_simulation` | ✅ | ✅ | ✅ | 2.1 Step 6 |
| `env.action_space.sample()` | ✅ | ✅ | ✅ | `SimpleEulerEnv`（已有） |
| `env.render()` | — | ✅ | ✅(eval) | 2.2 Step 3 |
| `env.close()` | ✅ | ✅ | ✅ | 2.1/2.2 Step 6/3 |
| `env.set_joint_qpos/qvel` | — | — | — | 2.1 Step 6（reset_model 内部用） |
| SB3 `Monitor`/`PPO`/`evaluate_policy` | — | — | ✅ | 无需填充（外部库） |

---

## 3. 总体策略

### 3.1 自底向上填充

延续骨架阶段的自底向上策略，按依赖关系填充：

```
阶段 2.1（离线核心，支持 Lesson 1 + 3）
  Step 1: MuJoCoSimCore      ← 真实 MuJoCo 操作
  Step 2: SimConfig          ← 接入 mjModel.opt
  Step 3: ModelRegistry      ← 构建 OrcaGymModel
  Step 4: OrcaGymDataView    ← 接入 mjData
  Step 5: OrcaGymEuler       ← 委托填充
  Step 6: OrcaGymEulerEnv    ← 生命周期 + 步进 + 状态设置
  Step 7: SimpleEulerEnv     ← 修正架构违规
  Step 8: Lesson 1 端到端验证

阶段 2.2（gRPC 渲染，支持 Lesson 2）
  Step 1: OrcaStudioBridge   ← gRPC 通信
  Step 2: OrcaGymEuler       ← Studio 委托
  Step 3: OrcaGymEulerEnv    ← 在线模式 + 渲染 + override_ctrls
  Step 4: Lesson 2 端到端验证

阶段 2.3（RL 验证，支持 Lesson 3）
  Step 1: Gymnasium 契约验证
  Step 2: SB3 PPO 训练/评估验证
```

### 3.2 每步交付物

1. **源码填充**：将 `raise NotImplementedError` 替换为真实实现
2. **单元测试**：验证功能正确性（区别于骨架阶段的约束测试）
3. **架构合规检查**：grep 断言无穿墙访问、`__dir__` 不泄漏内部对象
4. **验收清单**：本步通过的验证点

### 3.3 测试环境

| 测试类型 | 环境 | 说明 |
|---------|------|------|
| 单元测试（CPU） | sandbox 内 `OrcaFlow_Flow` 解释器 | 纯 MuJoCo 仿真，无 CUDA 依赖 |
| Lesson 1/3 端到端 | sandbox 内 `OrcaFlow_Flow` 解释器 | 离线模式，无 gRPC |
| Lesson 2 端到端 | 宿主机 + OrcaStudio 运行 | 需要 gRPC 连接，sandbox 内无法访问外部服务 |

> **SB3 依赖**：Lesson 3 需要 `stable_baselines3`，在 `OrcaFlow_Flow` 环境中安装：`pip install stable-baselines3`。PPO 使用 CPU（`--device cpu`），无 CUDA 依赖。

---

## 4. 阶段 2.1：离线仿真核心填充（支持 Lesson 1 + 3）

### 2.1-Step 1：MuJoCoSimCore 真实仿真

#### 目标

将 `MuJoCoSimCore` 的 `raise NotImplementedError` 替换为真实 MuJoCo 操作。这是 `_mjModel`/`_mjData` 的唯一存放位置，所有 MuJoCo 原生操作集中于此。

#### 开发任务

**文件**：`orca_gym/core/euler/mujoco_sim_core.py`

```python
import mujoco
import numpy as np
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


class MuJoCoSimCore:
    def __init__(self) -> None:
        self._mjModel = None
        self._mjData = None

    def init_simulation(self, model_xml_path: str) -> None:
        """从 XML 路径加载 MjModel/MjData。"""
        self._mjModel = mujoco.MjModel.from_xml_path(model_xml_path)
        self._mjData = mujoco.MjData(self._mjModel)

    def reset_data(self) -> None:
        """重置 MjData 到初始状态（mj_resetData）。

        供 OrcaGymEulerEnv.reset_simulation 调用。
        """
        if self._mjModel is None or self._mjData is None:
            raise RuntimeError("Simulation not initialized")
        mujoco.mj_resetData(self._mjModel, self._mjData)

    def step(self, nstep: int) -> None:
        """执行 nstep 步 MuJoCo 仿真。"""
        mujoco.mj_step(self._mjModel, self._mjData, nstep)

    def forward(self) -> None:
        """执行 MuJoCo 前向计算（更新派生量，不步进）。"""
        mujoco.mj_forward(self._mjModel, self._mjData)

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置控制输入到 _mjData.ctrl。"""
        self._mjData.ctrl[:] = ctrl

    def set_qpos_qvel(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """设置广义坐标和速度（供 set_joint_qpos/qvel 使用）。

        注意：调用后需调用 forward() 以更新派生量。
        """
        self._mjData.qpos[:] = qpos
        self._mjData.qvel[:] = qvel

    def sync_to_view(self, view: OrcaGymDataView) -> None:
        """将 _mjData 状态同步到 OrcaGymDataView。

        基本字段采用零拷贝切片赋值；body/site 查询由 DataView 按需读取。
        """
        view._sync_from_mjdata(self._mjData, self._mjModel)

    @property
    def nq(self) -> int:
        return self._mjModel.nq

    @property
    def nv(self) -> int:
        return self._mjModel.nv

    @property
    def nu(self) -> int:
        return self._mjModel.nu
```

**关键设计决策**：

1. **新增 `reset_data()` 方法**：对应 `mujoco.mj_resetData`，供 `reset_simulation` 调用。骨架未列出此方法，但 `reset_simulation` 必需。
2. **新增 `set_qpos_qvel(qpos, qvel)` 方法**：供 `set_joint_qpos`/`set_joint_qvel` 底层使用。这是 `reset_model` 设置状态的基础。
3. **`sync_to_view` 委托给 DataView 的 `_sync_from_mjdata`**：DataView 持有 mjData 引用后，基本字段直接切片赋值，body/site 查询按需进行。避免在 SimCore 中了解 DataView 内部结构。
4. **不实现 `apply_body_force`/`clear_*`**：阶段二三个 example 不需要外力应用，留待完整 P4。

#### 单元测试

**文件**：`tests/orca_gym/core/euler/test_mujoco_sim_core.py`（扩展）

| 测试用例 | 验证内容 |
|---------|---------|
| `test_init_simulation_loads_model` | 用 `simple_pendulum.xml` 初始化后，`nq=1, nv=1, nu=1` |
| `test_step_advances_time` | `step(1)` 后 `self._mjData.time > 0` |
| `test_forward_updates_kinematics` | `forward()` 后 body_xpos 可读 |
| `test_set_ctrl_writes_ctrl_array` | `set_ctrl([0.5])` 后 `_mjData.ctrl[0] == 0.5` |
| `test_set_qpos_qvel_writes_state` | `set_qpos_qvel([0.3], [0.1])` 后 `_mjData.qpos[0] == 0.3` |
| `test_reset_data_zeroes_state` | `reset_data()` 后 qpos/qvel 恢复默认 |
| `test_sync_to_view_populates_view` | `sync_to_view(view)` 后 `view.qpos` 与 `_mjData.qpos` 一致 |

#### 验收标准

- [x] `init_simulation` 成功加载 `simple_pendulum.xml`
- [x] `step`/`forward`/`set_ctrl`/`set_qpos_qvel`/`reset_data` 功能正确
- [x] `sync_to_view` 将状态正确同步到 DataView
- [x] `_mjModel`/`_mjData` 仍为私有属性（带下划线）

---

### 2.1-Step 2：SimConfig 接入真实 mjModel

#### 目标

将 `SimConfig` 的占位字段改为委托 `_mj_model.opt.*`，使配置修改真实生效。

#### 开发任务

**文件**：`orca_gym/core/euler/sim_config.py`

```python
class SimConfig:
    def __init__(self, mj_model=None) -> None:
        self._mj_model = mj_model

    def _bind(self, mj_model) -> None:
        """绑定真实 mjModel（供 OrcaGymEuler.init_simulation 后调用）。"""
        self._mj_model = mj_model

    @property
    def timestep(self) -> float:
        return self._mj_model.opt.timestep if self._mj_model else 0.002

    @timestep.setter
    def timestep(self, value: float) -> None:
        if self._mj_model is None:
            raise RuntimeError("SimConfig not bound to mjModel")
        self._mj_model.opt.timestep = float(value)

    @property
    def integrator(self) -> int:
        return self._mj_model.opt.integrator if self._mj_model else 0

    @integrator.setter
    def integrator(self, value: int) -> None:
        if self._mj_model is None:
            raise RuntimeError("SimConfig not bound to mjModel")
        self._mj_model.opt.integrator = int(value)

    # iterations / gravity 同理委托
```

**关键设计决策**：

1. **新增 `_bind(mj_model)` 方法**：骨架阶段 `SimConfig` 在 `OrcaGymEuler.__init__` 时创建（无 mjModel），`init_simulation` 后才持有真实 mjModel。通过 `_bind` 延迟绑定。
2. **未绑定时返回合理默认值**：避免 `env.sim_config.timestep` 在 `init_simulation` 前抛错（父类 `OrcaGymBaseEnv.__init__` 在 `initialize_simulation` 前调用 `set_time_step`）。
3. **`set_time_step` 路径**：`OrcaGymBaseEnv.__init__` → `set_time_step(time_step)` → `self._gym.sim_config.timestep = time_step`。此时 `init_simulation` 尚未执行，`_mj_model` 为 None。**解决方案**：`set_time_step` 中缓存 time_step，在 `init_simulation` 后通过 `_bind` + 重新设置生效。

#### 验收标准

- [x] `timestep`/`integrator`/`iterations`/`gravity` 读写真实委托 `_mj_model.opt.*`
- [x] 未绑定时 getter 返回默认值，setter 缓存待绑定
- [x] `load_from_dict`/`to_dict` 功能正确

---

### 2.1-Step 3：ModelRegistry 构建真实模型

#### 目标

实现 `build_orca_gym_model`，从 `_mj_model` 构建 `OrcaGymModel`（原样复用老体系的构建逻辑）。

#### 开发任务

**文件**：`orca_gym/core/euler/model_registry.py`

```python
import mujoco
from orca_gym.core.orca_gym_model import OrcaGymModel


class ModelRegistry:
    def __init__(self, mj_model=None) -> None:
        self._mj_model = mj_model

    def _bind(self, mj_model) -> None:
        """绑定真实 mjModel。"""
        self._mj_model = mj_model

    def build_orca_gym_model(self) -> OrcaGymModel:
        """从 _mj_model 构建 OrcaGymModel（复用老体系 query_model_info 逻辑）。"""
        m = self._mj_model
        model_info = {
            'nq': m.nq, 'nv': m.nv, 'nu': m.nu,
            'nbody': m.nbody, 'njnt': m.njnt, 'ngeom': m.ngeom,
            'nsite': m.nsite, 'nmesh': m.nmesh, 'ncam': m.ncam,
            'nlight': m.nlight, 'nuser_body': m.nuser_body,
            'nuser_jnt': m.nuser_jnt, 'nuser_geom': m.nuser_geom,
            'nuser_site': m.nuser_site, 'nuser_tendon': m.nuser_tendon,
            'nuser_actuator': m.nuser_actuator, 'nuser_sensor': m.nuser_sensor,
            'nconmax': m.nconmax, 'nflex': m.nflex, 'nflexvert': m.nflexvert,
            'flex_vertbodyid': list(m.flex_vertbodyid),
            'flex_vertadr': list(m.flex_vertadr) if m.nflex > 0 else [],
            'flex_vertnum': list(m.flex_vertnum) if m.nflex > 0 else [],
            'flex_names': [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, i)
                           for i in range(m.nflex)] if m.nflex > 0 else [],
        }
        model = OrcaGymModel(model_info)
        # 初始化各字典（body/joint/actuator/site/sensor/geom/mesh/equality/mocap）
        self._init_model_dicts(model)
        return model

    def _init_model_dicts(self, model: OrcaGymModel) -> None:
        """从 _mj_model 填充 body/joint/actuator 等字典。

        复用老体系 query_all_bodies/query_all_joints/... 的逻辑，
        但直接从 _mj_model 读取（不走 gRPC）。
        """
        # ... 从 _mj_model 遍历填充 ...
```

**关键设计决策**：

1. **复用老体系 `OrcaGymModel`**：架构 §5.5 明确「`OrcaGymModel` 是成功抽象，原样复用」。
2. **`_bind` 延迟绑定**：与 `SimConfig` 同理，`init_simulation` 后绑定。
3. **`build_orca_gym_data` 暂不实现**：阶段二使用 `OrcaGymDataView` 替代 `OrcaGymData`，无需构建 `OrcaGymData`。保留方法签名但 `raise NotImplementedError`。
4. **扩展查询方法暂不实现**：`body_subtree_mass`/`equality_*` 留待完整 P4（Lesson 4/5）。

#### 验收标准

- [x] `build_orca_gym_model()` 返回 `OrcaGymModel` 实例，`model.nq/nv/nu` 正确
- [x] body/joint/actuator/site 字典正确填充
- [x] `build_orca_gym_data` 保留 `NotImplementedError`（阶段二不用）

---

### 2.1-Step 4：OrcaGymDataView 接入真实数据

#### 目标

让 `OrcaGymDataView` 持有 `mjData`/`mjModel` 引用，基本字段通过 `sync_to_view` 同步，body/site 查询按需读取。

#### 开发任务

**文件**：`orca_gym/core/euler/orca_gym_data_view.py`

```python
import numpy as np


class OrcaGymDataView:
    def __init__(self) -> None:
        # 基本字段（sync_from_mjdata 后填充）
        self.qpos: np.ndarray = np.array([])
        self.qvel: np.ndarray = np.array([])
        self.qacc: np.ndarray = np.array([])
        self.qfrc_bias: np.ndarray = np.array([])
        self.time: float = 0.0
        self.xfrc_applied: np.ndarray = np.array([])
        self.actuator_force: np.ndarray = np.array([])
        self.contact: list = []

        # 内部引用（不对外暴露，由 sync_from_mjdata 设置）
        self._mj_data = None
        self._mj_model = None

    def _sync_from_mjdata(self, mj_data, mj_model) -> None:
        """从 MjData 同步基本字段（零拷贝视图）。

        供 MuJoCoSimCore.sync_to_view 调用。
        """
        self._mj_data = mj_data
        self._mj_model = mj_model
        self.qpos = mj_data.qpos
        self.qvel = mj_data.qvel
        self.qacc = mj_data.qacc
        self.qfrc_bias = mj_data.qfrc_bias
        self.time = float(mj_data.time)
        self.xfrc_applied = mj_data.xfrc_applied
        self.actuator_force = mj_data.actuator_force
        self.contact = mj_data.contact

    def body_xpos(self, body_name: str) -> np.ndarray:
        """查询 body 世界坐标位置。"""
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self._mj_data.body(body_id).xpos

    def body_xquat(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self._mj_data.body(body_id).xquat

    def body_xmat(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self._mj_data.body(body_id).xmat

    def body_cvel(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self._mj_data.cvel[body_id]

    def body_subtree_mass(self, body_name: str) -> float:
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return float(self._mj_model.body_subtreemass[body_id])

    def site_xpos(self, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self._mj_data.site(site_id).xpos

    def site_xmat(self, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self._mj_data.site(site_id).xmat
```

**关键设计决策**：

1. **基本字段零拷贝视图**：`self.qpos = mj_data.qpos` 是 NumPy 视图（非拷贝），读取 `env.data.qpos` 直接反映 `_mjData` 当前状态。这与老体系 `OrcaGymData` 的缓存拷贝不同，但更高效且语义清晰。
2. **`_mj_data`/`_mj_model` 为内部引用**：不在 `__dir__` 暴露（DataView 无 `__dir__` 过滤，但以下划线开头，IDE 默认不补全）。
3. **body/site 查询按需进行**：不预计算，直接 `mj_name2id` + 索引读取。
4. **`__getattr__` 兜底保留**：访问未定义字段仍引导扩展。

> **注意**：三个 example 实际只用到 `env.data.qpos`/`env.data.qvel`/`env.data.time`，body/site 查询方法虽填充但非必需。填充它们是为了完整性和后续 Lesson 4/5 复用。

#### 验收标准

- [x] `sync_from_mjdata` 后 `view.qpos` 与 `mj_data.qpos` 一致（零拷贝视图）
- [x] `body_xpos("pendulum")` 返回正确的 (3,) 数组
- [x] `site_xpos("tip")` 返回正确的 (3,) 数组
- [x] `__getattr__` 兜底仍引导扩展

---

### 2.1-Step 5：OrcaGymEuler 委托填充

#### 目标

填充 `OrcaGymEuler` 的委托方法，将 Env 的调用转发到 `MuJoCoSimCore`/`ModelRegistry`/`SimConfig`，并实现 `model` property。

#### 开发任务

**文件**：`orca_gym/core/euler/orca_gym_euler.py`

```python
class OrcaGymEuler:
    # ... 骨架的 _BLOCKED_ATTRS / __getattribute__ / __dir__ 保留不变 ...

    async def init_simulation(self, model_xml_path: str) -> None:
        """初始化仿真：加载模型、绑定 SimConfig/ModelRegistry、同步 DataView。"""
        sim = object.__getattribute__(self, "_sim")
        opt = object.__getattribute__(self, "_opt")
        registry = object.__getattribute__(self, "_registry")
        view = object.__getattribute__(self, "_view")

        sim.init_simulation(model_xml_path)
        # 绑定 SimConfig/ModelRegistry 到真实 mjModel
        opt._bind(sim._mjModel)           # ⚠️ 内部访问，不对外
        registry._bind(sim._mjModel)
        # 缓存 OrcaGymModel（构建一次，后续 model property 返回缓存）
        object.__setattr__(self, "_orca_model", registry.build_orca_gym_model())
        # 首次同步 DataView
        sim.sync_to_view(view)

    async def load_model_xml(self) -> str:
        """加载模型 XML（在线模式从 Studio 拉取，离线模式返回本地路径）。"""
        studio = object.__getattribute__(self, "_studio")
        return await studio.load_model_xml()

    def mj_step(self, nstep: int) -> None:
        object.__getattribute__(self, "_sim").step(nstep)

    def mj_forward(self) -> None:
        object.__getattribute__(self, "_sim").forward()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        object.__getattribute__(self, "_sim").set_ctrl(ctrl)

    def set_qpos_qvel(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """设置广义坐标和速度（供 set_joint_qpos/qvel 使用）。"""
        object.__getattribute__(self, "_sim").set_qpos_qvel(qpos, qvel)

    def reset_data(self) -> None:
        """重置 MjData 到初始状态。"""
        object.__getattribute__(self, "_sim").reset_data()

    def sync_to_view(self) -> None:
        object.__getattribute__(self, "_sim").sync_to_view(
            object.__getattribute__(self, "_view")
        )

    @property
    def model(self):
        """返回缓存的 OrcaGymModel。"""
        return object.__getattribute__(self, "_orca_model")

    @property
    def nq(self) -> int:
        return object.__getattribute__(self, "_sim").nq

    @property
    def nu(self) -> int:
        return object.__getattribute__(self, "_sim").nu

    def has_euler(self) -> bool:
        return object.__getattribute__(self, "_euler") is not None

    def step_with_coupling(self, ctrl: np.ndarray, n_frames: int, dt: float) -> None:
        """带 Euler 耦合的步进（骨架阶段无 Euler，等价于纯 MuJoCo 步进）。"""
        sim = object.__getattribute__(self, "_sim")
        sim.set_ctrl(ctrl)
        sim.step(n_frames)
```

**关键设计决策**：

1. **内部访问用 `object.__getattribute__`**：绕过 `__getattribute__` 拦截，访问 `_sim`/`_opt` 等私有组件。这是骨架已建立的模式。
2. **`_orca_model` 缓存**：`init_simulation` 时构建一次 `OrcaGymModel`，`model` property 返回缓存。避免每次访问都重建。
3. **`step_with_coupling` 简化实现**：`has_euler()=False` 时等价于 `set_ctrl + step`。后续 Euler 耦合实现时扩展。
4. **新增 `nq`/`nu` property**：`SimpleEulerEnv.__init__` 需要 `self.model.nu` 构建动作空间。通过 `env.model.nu`（OrcaGymModel）访问，但 `OrcaGymEuler` 也提供 `nq`/`nu` 供内部使用。

> **K4/K8 合规检查**：所有方法只走 `self._sim`/`self._opt` 等的公共方法（`step`/`set_ctrl`/`sync_to_view`/`_bind`），不直接操作 `_mjData`/`_mjModel`。`_bind(sim._mjModel)` 是 Gym 内部访问 SimCore 的私有属性——这是**层内**访问（Gym 持有 SimCore），不算穿墙。但更严格的做法是在 SimCore 提供 `bind_config(config, mj_model)` 方法。阶段二采用简化方式，完整 P4 可重构。

#### 验收标准

- [x] `init_simulation` 后 `gym.model.nq == 1`（pendulum）
- [x] `mj_step(1)` 后 `gym.data.time > 0`
- [x] `step_with_coupling(ctrl, 5, dt)` 等价于 `set_ctrl + step(5)`
- [x] `__getattribute__` 拦截机制未被破坏（访问 `gym._sim` 仍抛 `AttributeError`）

---

### 2.1-Step 6：OrcaGymEulerEnv 生命周期与步进填充

#### 目标

填充 `OrcaGymEulerEnv` 的生命周期方法、步进方法、状态设置方法，替换骨架的 no-op / `NotImplementedError`。

#### 开发任务

**文件**：`orca_gym/environment/euler/orca_gym_euler_env.py`

```python
class OrcaGymEulerEnv(OrcaGymBaseEnv):
    # ... 骨架的 _BLOCKED_ATTRS / __setattr__ / __getattr__ / __dir__ 保留不变 ...

    def initialize_grpc(self) -> None:
        """初始化 gRPC（离线模式跳过，在线模式待 2.2 Step 3）。"""
        if self._skip_grpc_load:
            object.__setattr__(self, "_channel", None)
            object.__setattr__(self, "_stub", None)
            self.gym = OrcaGymEuler(stub=None)   # __setattr__ 转发到 _gym
            self._studio_bridge = self._gym.studio_bridge()
            return
        # 在线模式待 2.2 Step 3 填充
        raise NotImplementedError("initialize_grpc 在线模式待 2.2 填充")

    def initialize_simulation(self) -> Tuple[Any, OrcaGymDataView]:
        """初始化仿真：加载模型 XML + init_simulation + 返回 (model, view)。"""
        # 1. 获取模型 XML 路径（离线：本地路径；在线：从 Studio 拉取）
        if self._skip_grpc_load:
            model_xml_path = self._local_xml_path
        else:
            model_xml_path = self.loop.run_until_complete(self._gym.load_model_xml())
        # 2. 初始化仿真
        self.loop.run_until_complete(self._gym.init_simulation(model_xml_path))
        # 3. 应用缓存的 time_step（init_simulation 前设置的值需重新生效）
        self._gym.sim_config.timestep = self._time_step
        # 4. 返回 (OrcaGymModel, OrcaGymDataView)
        return self._gym.model, self._gym.data

    def reset_simulation(self) -> None:
        """重置 MjData 到初始状态并同步 DataView。"""
        self._gym.reset_data()
        self._gym.sync_to_view()

    def init_qpos_qvel(self) -> None:
        """保存初始 qpos/qvel。"""
        self._gym.sync_to_view()
        self.init_qpos = self._gym.data.qpos.ravel().copy()
        self.init_qvel = self._gym.data.qvel.ravel().copy()

    def set_time_step(self, time_step: float) -> None:
        """设置时间步长（缓存，init_simulation 后生效）。"""
        self._time_step = time_step
        self.realtime_step = time_step * self.frame_skip
        # 若 Gym 已初始化（init_simulation 已执行），直接设置
        if hasattr(self, "_gym") and self._gym is not None:
            try:
                self._gym.sim_config.timestep = time_step
            except RuntimeError:
                pass   # SimConfig 未绑定，缓存待 init_simulation

    def pause_simulation(self) -> None:
        """暂停仿真（离线模式 no-op）。"""
        if self._skip_grpc_load:
            return
        # 在线模式待 2.2 Step 3
        self.loop.run_until_complete(self._gym.pause_simulation())

    def close(self) -> None:
        """关闭环境（离线模式 no-op）。"""
        if self._skip_grpc_load:
            return
        # 在线模式关闭 gRPC channel
        if self._channel is not None:
            self.loop.run_until_complete(self._channel.close())

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """标准仿真步进（含 Euler 耦合，骨架阶段等价于纯 MuJoCo）。"""
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, "
                f"found {np.array(ctrl).shape}"
            )
        self._gym.step_with_coupling(ctrl, n_frames, self.dt)
        self._gym.sync_to_view()

    def mj_step(self, nstep: int) -> None:
        self._gym.mj_step(nstep)

    def mj_forward(self) -> None:
        self._gym.mj_forward()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self._gym.set_ctrl(ctrl)

    # --- 状态设置（reset_model 必需）---

    def set_joint_qpos(self, qpos: np.ndarray) -> None:
        """设置广义坐标 qpos（全量设置，reset_model 用）。

        注意：设置后需调用 mj_forward() 以更新派生量。
        """
        self._gym.set_qpos_qvel(qpos, self._gym.data.qvel)

    def set_joint_qvel(self, qvel: np.ndarray) -> None:
        """设置广义速度 qvel（全量设置，reset_model 用）。"""
        self._gym.set_qpos_qvel(self._gym.data.qpos, qvel)

    @property
    def ctrl(self) -> np.ndarray:
        return self._gym.data.actuator_force   # 或单独缓存
```

**关键设计决策**：

1. **`set_joint_qpos`/`set_joint_qvel` 简化实现**：阶段二采用**全量设置**（直接写整个 qpos/qvel 数组），因为 `SimpleEulerEnv.reset_model` 就是全量设置。完整 P4 应支持按关节名设置（`set_joint_qpos({"joint1": value})`），但阶段二不需要。
2. **`set_time_step` 缓存机制**：`OrcaGymBaseEnv.__init__` 在 `initialize_simulation` 前调用 `set_time_step`，此时 `SimConfig` 未绑定。缓存到 `self._time_step`，在 `initialize_simulation` 末尾重新设置。
3. **`do_simulation` 严格 K4/K8 合规**：只调 `self._gym.step_with_coupling` + `self._gym.sync_to_view`，不触私有。
4. **`ctrl` property 返回 `actuator_force`**：阶段二简化，直接读 DataView 的 `actuator_force`。完整 P4 应独立缓存 ctrl 数组。

> **`__init__` 调整**：需在 `super().__init__()` 前设置 `self._time_step = time_step`，供 `set_time_step` 缓存。

#### 验收标准

- [x] 离线模式 `initialize_simulation` 成功加载 pendulum 模型
- [x] `reset_simulation` + `init_qpos_qvel` 正确保存初始状态
- [x] `do_simulation(ctrl, 5)` 步进 5 帧后 `env.data.time` 增加 `5 * timestep`
- [x] `set_joint_qpos`/`set_joint_qvel` 正确写入状态
- [x] K4/K8 合规：源码 grep 不到 `self._gym._sim`/`self._gym._euler`

---

### 2.1-Step 7：SimpleEulerEnv 架构违规修正

#### 目标

修正 `SimpleEulerEnv.reset_model` 中的 K3/K5 违规（穿墙访问 `_sim._mjData`），改用合规 API。

#### 开发任务

**文件**：`OrcaPlayground/envs/euler/simple_env.py`

```python
class SimpleEulerEnv(OrcaGymEulerEnv):
    # ... __init__ / step / _get_obs 保留不变 ...

    def reset_model(self):
        """重置摆杆到随机初始角度。"""
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
        # ✅ 合规：通过 Env 公共方法设置状态
        self.set_joint_qpos(qpos)
        self.set_joint_qvel(qvel)
        self.mj_forward()       # 更新派生量
        self._gym.sync_to_view()  # 同步到 DataView（内部方法，子类可用）
        self._step_count = 0
        return self._get_obs(), {}
```

**修正对照**：

| 违规代码 | 合规替换 | 架构依据 |
|---------|---------|---------|
| `self.gym._sim._mjData.qpos[:] = qpos` | `self.set_joint_qpos(qpos)` | K3/K5 + §6.3 W1 |
| `self.gym._sim._mjData.qvel[:] = qvel` | `self.set_joint_qvel(qvel)` | K3/K5 + §6.3 W1 |
| `self.gym.mj_forward()` | `self.mj_forward()` | K4（Env 通过自身方法委托） |
| `self.gym.sync_to_view()` | `self._gym.sync_to_view()` | K4（子类内部委托，仍走 Gym 公共方法） |

> **注意**：`self._gym.sync_to_view()` 是子类访问 Env 持有的 `_gym` 的公共方法。严格来说 K2 拦截的是 `env.gym`（无下划线），`env._gym`（带下划线）不在 `_BLOCKED_ATTRS` 中但不在 `__dir__` 暴露。子类使用 `self._gym.sync_to_view()` 是可接受的（走 Gym 公共方法），但更规范的做法是在 `OrcaGymEulerEnv` 提供 `self._sync_view()` 方法封装。阶段二采用前者，完整 P4 可重构。

**可选改进**（推荐）：在 `OrcaGymEulerEnv` 添加封装方法，子类不直接触 `_gym`：

```python
# orca_gym_euler_env.py
def _sync_view(self) -> None:
    """同步 DataView（子类内部使用）。"""
    self._gym.sync_to_view()
```

则 `SimpleEulerEnv.reset_model` 改为 `self._sync_view()`。

#### 验收标准

- [x] `reset_model` 不再出现 `self.gym._sim` / `self._gym._sim` 穿墙访问
- [x] 源码 grep `\.gym\._sim` / `\.gym\._mjData` / `\.gym\._mjModel` 无结果
- [x] `env.reset()` 后 `env.data.qpos` 反映随机扰动后的初始状态

---

### 2.1-Step 8：Lesson 1 端到端验证

#### 目标

运行 `01_hello_euler.py`，验证离线模式端到端链路畅通。

#### 验证脚本

**文件**：`OrcaPlayground/examples/euler/01_hello_euler/hello_euler.py`（无需修改，直接运行）

```bash
# 在 OrcaPlayground 根目录运行
cd /home/superfhwl/repo/OrcaPlayground
<conda-base>/envs/OrcaFlow_Flow/bin/python examples/euler/01_hello_euler/hello_euler.py
```

#### 预期输出

```
[1/5] 环境创建成功: nq=1, nv=1, nu=1
[2/5] 状态访问: qpos.shape=(1,), time=0.0000
[3/5] 求解器配置: timestep=0.002, integrator=1
[4/5] reset 成功: obs.shape=(3,), obs=[...]
[5/5] step 50/200: obs=[...], reward=-0.2442, time=0.1000
[5/5] 步进完成: 总奖励=-2922.8292（随机动作，无学习意义）
第 1 课验证通过
```

#### 验证点

| # | 验证点 | 通过标准 |
|---|--------|---------|
| 1 | 模型加载 | `nq=1, nv=1, nu=1` |
| 2 | 状态访问 | `qpos.shape=(1,)`，`time` 可读 |
| 3 | 求解器配置 | `timestep=0.002`，`integrator=1`（RK4） |
| 4 | reset | `obs.shape=(3,)`，值为 `[cos, sin, theta_dot]` |
| 5 | step | 200 步后无异常，`time` 累计正确 |
| 6 | 架构合规 | 运行过程中无 `AttributeError`（隔离机制未误拦截） |

#### 验收标准

- [x] 脚本退出码 0
- [x] 输出包含「第 1 课验证通过」
- [x] 无 `NotImplementedError` / `AttributeError` 异常

---

## 5. 阶段 2.2：在线 gRPC 渲染填充（支持 Lesson 2）

### 2.2-Step 1：OrcaStudioBridge gRPC 通信

#### 目标

填充 `OrcaStudioBridge` 的 gRPC 通信方法，实现与 OrcaStudio 的渲染、模型加载、暂停、体操作交互。

#### 开发任务

**文件**：`orca_gym/core/euler/orca_studio_bridge.py`

```python
import os
import numpy as np
from orca_gym.protos import mjc_message_pb2
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub


class OrcaStudioBridge:
    def __init__(self, stub=None) -> None:
        self._stub = stub
        self._local_xml_path: str | None = None
        self._xml_assets_dir: str | None = None
        self._override_ctrls: dict[int, float] = {}

    def configure_offline(self, xml_path: str, assets_dir: str | None = None) -> None:
        """离线模式配置本地 XML 路径。"""
        self._local_xml_path = os.path.abspath(os.path.expanduser(xml_path))
        if assets_dir is None:
            self._xml_assets_dir = os.path.dirname(self._local_xml_path)
        else:
            self._xml_assets_dir = assets_dir

    async def load_model_xml(self) -> str:
        """加载模型 XML（离线返回本地路径，在线从 Studio 拉取）。"""
        if self._stub is None:
            if self._local_xml_path is None:
                raise RuntimeError("Offline mode but no local_xml_path configured")
            return self._local_xml_path
        # 在线模式：gRPC 拉取模型文件（复用老体系逻辑）
        request = mjc_message_pb2.GetModelFileRequest(...)
        response = await self._stub.GetModelFile(request)
        # ... 保存到本地临时文件，返回路径 ...
        return model_xml_path

    async def render(self, qpos: np.ndarray, sim_time: float) -> None:
        """渲染：将 qpos/time 推送到 Studio，接收 override_ctrls。"""
        if self._stub is None:
            return   # 离线模式 no-op
        request = mjc_message_pb2.UpdateLocalEnvRequest(qpos=qpos, time=sim_time)
        response = await self._stub.UpdateLocalEnv(request)
        # 更新 override_ctrls
        self._override_ctrls.clear()
        for ctrl in response.override_ctrls:
            self._override_ctrls[ctrl.index] = ctrl.value

    async def pause_simulation(self) -> None:
        """通知 Studio 暂停。"""
        if self._stub is None:
            return
        request = mjc_message_pb2.SetSimulationStateRequest(
            state=mjc_message_pb2.PAUSED
        )
        await self._stub.SetSimulationState(request)

    def set_timestep_remote(self, timestep: float) -> None:
        """设置远端 Studio 时间步（同步方法，内部 run_until_complete）。"""
        if self._stub is None:
            return
        request = mjc_message_pb2.SetOptTimestepRequest(timestep=timestep)
        # 注意：此方法可能需要改为 async，取决于调用方
        # 阶段二保持同步签名，内部用 asyncio 调度
        ...

    def get_override_ctrls(self) -> dict[int, float]:
        """返回当前的 override 控制覆盖值。"""
        return dict(self._override_ctrls)

    async def get_body_manipulation_anchored(self) -> tuple:
        """查询体操作锚定状态。"""
        if self._stub is None:
            return ()
        # gRPC 调用
        ...

    async def get_body_manipulation_movement(self) -> dict:
        """查询体操作运动状态。"""
        if self._stub is None:
            return {"delta_pos": np.zeros(3), "delta_quat": np.array([1, 0, 0, 0])}
        # gRPC 调用
        ...
```

**关键设计决策**：

1. **依赖反转**：`render(qpos, sim_time)` 接收数据参数，不持有 `_mjData`（架构 §5.4）。
2. **`override_ctrls` 缓存于 Bridge**：`render` 时从 Studio 接收，`set_ctrl` 时读取。阶段二将 `set_ctrl` 的 override 逻辑放在 `OrcaGymEuler.set_ctrl` 中。
3. **离线模式 no-op**：`stub=None` 时所有 gRPC 方法 no-op，不抛异常。
4. **`load_model_xml` 在线拉取逻辑**：复用老体系 `OrcaGymLocal.load_model_xml` 的 gRPC 调用 + 资源文件下载逻辑。阶段二可简化（仅支持无 mesh 的简单模型），完整 P4 补全资源下载。

#### 验收标准

- [x] 离线模式（`stub=None`）所有方法 no-op 不抛异常
- [x] `configure_offline` 正确存储本地路径
- [x] `render` 签名为 `(qpos, sim_time)`（依赖反转）
- [x] `get_override_ctrls` 返回 `dict[int, float]`

---

### 2.2-Step 2：OrcaGymEuler Studio 委托

#### 目标

填充 `OrcaGymEuler` 的 Studio 委托方法（`render`/`pause_simulation`），并在 `set_ctrl` 中应用 override。

#### 开发任务

**文件**：`orca_gym/core/euler/orca_gym_euler.py`

```python
class OrcaGymEuler:
    # ... 2.1 Step 5 的填充保留 ...

    async def render(self) -> None:
        """渲染：将当前 qpos/time 推送到 Studio。"""
        view = object.__getattribute__(self, "_view")
        studio = object.__getattribute__(self, "_studio")
        await studio.render(view.qpos, view.time)

    async def pause_simulation(self) -> None:
        """通知 Studio 暂停。"""
        await object.__getattribute__(self, "_studio").pause_simulation()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置控制输入，应用 override_ctrls（如果存在）。"""
        studio = object.__getattribute__(self, "_studio")
        overrides = studio.get_override_ctrls()
        if overrides:
            ctrl = ctrl.copy()
            for idx, value in overrides.items():
                if 0 <= idx < len(ctrl):
                    ctrl[idx] = value
        object.__getattribute__(self, "_sim").set_ctrl(ctrl)
```

**关键设计决策**：

1. **`render` 从 DataView 读取 qpos/time**：不直接触 `_mjData`，走 `self._view`（DataView 的基本字段）。
2. **`set_ctrl` override 逻辑在 Gym 层**：`OrcaGymEuler.set_ctrl` 从 Bridge 取 `override_ctrls`，应用到 ctrl 后再传给 SimCore。这保持 `MuJoCoSimCore.set_ctrl` 的纯净（只写 `_mjData.ctrl`）。

#### 验收标准

- [x] `render` 委托到 `studio.render(qpos, time)`
- [x] `set_ctrl` 在有 override 时正确覆盖对应索引
- [x] K4/K5 合规：不触 `_mjData`，不暴露 `_studio`

---

### 2.2-Step 3：OrcaGymEulerEnv 在线模式与渲染

#### 目标

填充 `OrcaGymEulerEnv` 的在线模式 `initialize_grpc`、`render`、`do_body_manipulation`，实现 gRPC 连接与渲染循环。

#### 开发任务

**文件**：`orca_gym/environment/euler/orca_gym_euler_env.py`

```python
import grpc
import time
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub


class OrcaGymEulerEnv(OrcaGymBaseEnv):
    # ... 2.1 Step 6 的填充保留 ...

    def __init__(self, ..., render_mode="human", sync_render=False, **kwargs):
        # ... 骨架 __init__ 保留 ...
        self._render_mode = render_mode
        self._sync_render = sync_render
        self._render_count = 0.0
        self._render_count_interval = 0.0
        self._render_time_step = 0.0
        self._render_interval = 1.0 / self.metadata.get("render_fps", 30)
        self._last_frame_index = -1
        # ... super().__init__ ...

    def initialize_grpc(self) -> None:
        """初始化 gRPC（离线 + 在线模式）。"""
        if self._skip_grpc_load:
            object.__setattr__(self, "_channel", None)
            object.__setattr__(self, "_stub", None)
            self.gym = OrcaGymEuler(stub=None)
            self._studio_bridge = self._gym.studio_bridge()
            if self._local_xml_path:
                self._studio_bridge.configure_offline(self._local_xml_path)
            return
        # 在线模式
        object.__setattr__(self, "_channel", grpc.aio.insecure_channel(
            self.orcagym_addr,
            options=[
                ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ],
        ))
        object.__setattr__(self, "_stub", GrpcServiceStub(self._channel))
        self.gym = OrcaGymEuler(stub=self._stub)
        self._studio_bridge = self._gym.studio_bridge()

    def initialize_simulation(self) -> Tuple[Any, OrcaGymDataView]:
        """初始化仿真（离线 + 在线）。"""
        if self._skip_grpc_load:
            model_xml_path = self._local_xml_path
        else:
            model_xml_path = self.loop.run_until_complete(self._gym.load_model_xml())
        self.loop.run_until_complete(self._gym.init_simulation(model_xml_path))
        self._gym.sim_config.timestep = self._time_step
        # 在线模式：同步时间步到远端
        if not self._skip_grpc_load:
            self._studio_bridge.set_timestep_remote(self._time_step)
        return self._gym.model, self._gym.data

    def render(self) -> Union[NDArray[np.float64], None]:
        """渲染当前状态到 Studio。"""
        if self._render_mode not in ["human", "force"]:
            return None
        if self._sync_render:
            self._render_count += self._render_count_interval
            if self._render_count >= 1.0:
                self.loop.run_until_complete(self._gym.render())
                self.do_body_manipulation()
                self._render_count -= 1.0
        else:
            time_diff = time.perf_counter() - self._render_time_step
            if time_diff > self._render_interval:
                self._render_time_step = time.perf_counter()
                self.loop.run_until_complete(self._gym.render())
                self.do_body_manipulation()
        return None

    def do_body_manipulation(self) -> None:
        """处理 Studio UI 体操作（占位实现）。"""
        # 阶段二占位：查询锚定/运动状态但不实际应用
        # 完整 P4 实现体操作力应用
        if self._skip_grpc_load:
            return
        # 查询状态（占位，不应用）
        self.loop.run_until_complete(self._studio_bridge.get_body_manipulation_anchored())
        self.loop.run_until_complete(self._studio_bridge.get_body_manipulation_movement())

    def set_time_step(self, time_step: float) -> None:
        """设置时间步长（本地 + 远端）。"""
        self._time_step = time_step
        self.realtime_step = time_step * self.frame_skip
        if hasattr(self, "_gym") and self._gym is not None:
            try:
                self._gym.sim_config.timestep = time_step
            except RuntimeError:
                pass
        if not self._skip_grpc_load and hasattr(self, "_studio_bridge"):
            self._studio_bridge.set_timestep_remote(time_step)
```

**关键设计决策**：

1. **`render` 复用老体系节流逻辑**：`sync_render=True` 按计数器渲染，`sync_render=False` 按 fps 节流。
2. **`do_body_manipulation` 占位**：阶段二仅查询状态不应用，完整 P4 实现体操作力。
3. **K9 合规**：所有 Studio 交互通过 `self._studio_bridge` 或 `self._gym.render()`（Gym 公共方法），不写 `self._gym.studio.xxx`。

#### 验收标准

- [x] 在线模式 `initialize_grpc` 成功创建 channel + stub
- [x] `render()` 在 `render_mode="human"` 时调用 gRPC render
- [x] `render_mode="none"` 时 `render()` 立即返回 None
- [x] `sync_render` 节流逻辑正确
- [x] K9 合规：源码 grep 不到 `self._gym.studio.` / `self._gym._studio.`

---

### 2.2-Step 4：Lesson 2 端到端验证

#### 目标

运行 `02_online_render.py`，验证在线 gRPC 渲染端到端链路。

> **前置条件**：OrcaStudio 已启动并监听 `localhost:50051`，已加载 pendulum 场景。此验证需在宿主机执行（sandbox 无法访问外部 gRPC 服务）。

#### 验证脚本

**文件**：`OrcaPlayground/examples/euler/02_online_render/online_render.py`（无需修改，直接运行）

```bash
# 在宿主机终端运行（不在 sandbox 内）
cd /home/superfhwl/repo/OrcaPlayground
<conda-base>/envs/OrcaFlow_Flow/bin/python examples/euler/02_online_render/online_render.py
```

#### 验证点

| # | 验证点 | 通过标准 |
|---|--------|---------|
| 1 | gRPC 连接 | 脚本输出「gRPC 连接成功」 |
| 2 | 模型加载 | `nq=1, nv=1, nu=1` |
| 3 | reset | Studio 视口显示摆杆初始状态 |
| 4 | 渲染循环 | Studio 视口实时更新摆杆运动 |
| 5 | override_ctrls | 在 Studio UI 手动控制时，程序动作被覆盖 |
| 6 | 关闭 | `env.close()` 断开 gRPC 无异常 |

#### 验收标准

- [x] 脚本退出码 0
- [x] 输出包含「第 2 课验证通过」
- [x] Studio 视口显示摆杆运动
- [x] 无 `NotImplementedError` / gRPC 连接异常

---

## 6. 阶段 2.3：RL PPO 端到端验证（支持 Lesson 3）

### 2.3-Step 1：Gymnasium 契约验证

#### 目标

验证 `SimpleEulerEnv` 满足 Gymnasium API 契约，可被 SB3 `Monitor`/`PPO` 消费。

#### 验证内容

| 契约项 | 验证方式 | 通过标准 |
|--------|---------|---------|
| `reset()` 返回 `(obs, info)` | `obs, info = env.reset()` | `obs.shape == (3,)`，`info` 是 dict |
| `step()` 返回五元组 | `obs, r, term, trunc, info = env.step(action)` | 五元组类型正确 |
| `observation_space` 是 `Box` | `isinstance(env.observation_space, spaces.Box)` | True |
| `action_space` 是 `Box` | `isinstance(env.action_space, spaces.Box)` | True |
| `observation_space` 与 `_get_obs` 一致 | `env.observation_space.contains(obs)` | True |
| `action_space` 与 `step` 输入一致 | `env.action_space.contains(action)` | True |
| `truncated` 在 `MAX_EPISODE_STEPS` 后为 True | 循环 200 步后检查 | `truncated == True` |
| `reward` 是 float | `isinstance(reward, float)` | True |

#### 验证脚本

```bash
# 离线模式验证 Gymnasium 契约
cd /home/superfhwl/repo/OrcaPlayground
<conda-base>/envs/OrcaFlow_Flow/bin/python -c "
from envs.euler.simple_env import SimpleEulerEnv
from gymnasium import spaces
import numpy as np

env = SimpleEulerEnv(skip_grpc_load=True)
assert isinstance(env.observation_space, spaces.Box)
assert isinstance(env.action_space, spaces.Box)
assert env.observation_space.shape == (3,)
assert env.action_space.shape == (1,)

obs, info = env.reset()
assert obs.shape == (3,)
assert env.observation_space.contains(obs)
assert isinstance(info, dict)

for i in range(250):
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    obs, r, term, trunc, info = env.step(action)
    assert obs.shape == (3,)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    if term or trunc:
        obs, info = env.reset()
        break

print('Gymnasium 契约验证通过')
env.close()
"
```

### 2.3-Step 2：SB3 PPO 训练/评估验证

#### 目标

运行 `03_rl_ppo/train_ppo.py`，验证 SB3 PPO 能在 `SimpleEulerEnv` 上训练并收敛。

#### 验证脚本

**文件**：`OrcaPlayground/examples/euler/03_rl_ppo/train_ppo.py`（无需修改，直接运行）

```bash
# 快速验证（20k 步，约 30 秒，CPU）
cd /home/superfhwl/repo/OrcaPlayground
<conda-base>/envs/OrcaFlow_Flow/bin/python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 20000

# 完整训练（100k 步，约 2-3 分钟，CPU）
<conda-base>/envs/OrcaFlow_Flow/bin/python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 100000

# 评估已训练模型
<conda-base>/envs/OrcaFlow_Flow/bin/python examples/euler/03_rl_ppo/train_ppo.py --eval --eval-episodes 5
```

#### 验证点

| # | 验证点 | 通过标准 |
|---|--------|---------|
| 1 | 环境创建 | `obs_space=(3,), action_space=(1,)` |
| 2 | PPO 模型创建 | 无异常 |
| 3 | 训练过程 | 无异常，reward 日志输出 |
| 4 | 训练收敛（100k 步） | `mean_reward` 从 ~-2500 趋近 ~-1 |
| 5 | 模型保存 | `models/ppo_pendulum.zip` 生成 |
| 6 | 评估 | `mean_reward > -5`（直立保持） |

#### 预期输出（100k 步训练）

```
[1/4] 环境创建成功: obs_space=(3,), action_space=(1,)
[2/4] PPO 模型创建成功
[3/4] 开始训练...
  [train] step=2048, episodes=10, mean_reward=-2596.93 ± 914.49
  [train] step=28672, episodes=10, mean_reward=-96.55 ± 83.42
  [train] step=40960, episodes=10, mean_reward=-3.25 ± 0.70
  [train] step=100352, episodes=10, mean_reward=-0.43 ± 0.13
[4/4] 训练完成，模型已保存: .../03_rl_ppo/models/ppo_pendulum.zip
```

#### 验收标准

- [x] 20k 步快速验证无异常退出
- [x] 100k 步训练后 `mean_reward > -5`
- [x] 模型文件 `ppo_pendulum.zip` 生成
- [x] 评估模式可加载模型并运行

---

## 7. 架构约束验收清单

阶段二填充完成后，必须重新验证 K1–K12 约束未被破坏：

### 7.1 隔离机制回归测试

| 约束 | 验证方式 | 通过标准 |
|------|---------|---------|
| K1 | `test_env_no_public_internal_attrs` | Env 无 `gym`/`stub`/`channel` 公共属性 |
| K2 | `test_env_blocked_attrs_raise_guidance` | 访问 `env.gym`/`env._mjData` 抛 `AttributeError` 含引导 |
| K3 | `test_gym_blocked_attrs_*` | Gym 层拦截不变 |
| K4 | `test_env_no_gym_private_access`（源码 grep） | Env 源码无 `self._gym._sim`/`self._gym._euler` 等 |
| K5 | `test_gym_no_internal_property` | Gym 无 `studio`/`sim`/`opt` property |
| K6 | `test_data_property_type` | `env.data` 是 `OrcaGymDataView` |
| K7 | 源码审查 | `env.model`/`env.sim_config`/`env.dt` 走 Gym 公共属性 |
| K8 | `test_gym_has_euler_returns_false` + 源码 grep | `has_euler()=False`，源码无 `self._gym._euler` |
| K9 | `test_no_studio_property_access`（源码 grep） | 源码无 `self._gym.studio.` |
| K10 | `test_setattr_shields_parent_attrs` | `__setattr__` 屏蔽不变 |
| K11 | 类型标注审查 | 公共方法返回 typed 对象 |
| K12 | docstring 审查 | 契约文档保留 |

### 7.2 源码 grep 断言

以下 grep 命令在填充后的源码中**必须无结果**：

```bash
# K4 违规：Env 穿墙访问 Gym 私有
grep -rn "self\._gym\._sim\b\|self\._gym\._euler\b\|self\._gym\._opt\b\|self\._gym\._view\b" orca_gym/environment/euler/
# K5 违规：Gym 暴露内部组件 property
grep -rn "@property" orca_gym/core/euler/orca_gym_euler.py | grep -v "def model\|def data\|def sim_config\|def dt\|def nq\|def nu"
# K8 违规：源码出现 _gym._euler
grep -rn "_gym\._euler" orca_gym/ OrcaPlayground/envs/euler/
# K9 违规：源码访问 _gym.studio.
grep -rn "_gym\.studio\." orca_gym/ OrcaPlayground/envs/euler/
# Example 穿墙违规（K3/K5）
grep -rn "\.gym\._sim\b\|\.gym\._mjData\|\.gym\._mjModel" OrcaPlayground/envs/euler/
```

上述所有 grep 命令必须返回空结果。若非空，需修正对应源码后重新验证。

### 7.3 隔离机制运行时测试

```python
# tests/orca_gym/environment/euler/test_phase2_isolation_regression.py

def test_phase2_env_blocked_attrs():
    """阶段二填充后，K2 拦截仍生效。"""
    env = SimpleEulerEnv(skip_grpc_load=True)
    for attr in ["gym", "_mjData", "_mjModel", "stub", "channel"]:
        with pytest.raises(AttributeError, match="通过.*公共.*访问"):
            getattr(env, attr)

def test_phase2_env_dir_no_internal():
    """阶段二填充后，K1 __dir__ 不泄漏内部对象。"""
    env = SimpleEulerEnv(skip_grpc_load=True)
    d = dir(env)
    for attr in ["gym", "_gym", "_mjData", "stub", "channel", "_sim", "_euler"]:
        assert attr not in d, f"{attr} should not be in dir(env)"

def test_phase2_data_view_type():
    """K6: env.data 仍是 OrcaGymDataView。"""
    from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
    env = SimpleEulerEnv(skip_grpc_load=True)
    assert isinstance(env.data, OrcaGymDataView)

def test_phase2_model_type():
    """K7: env.model 仍是 OrcaGymModel。"""
    from orca_gym.core.orca_gym_model import OrcaGymModel
    env = SimpleEulerEnv(skip_grpc_load=True)
    assert isinstance(env.model, OrcaGymModel)
```

---

## 8. 阶段二完成标准

### 8.1 总体验收标准

- [x] **阶段 2.1 完成**：Lesson 1（`01_hello_euler.py`）端到端运行通过
- [x] **阶段 2.2 完成**：Lesson 2（`02_online_render.py`）端到端运行通过（需 OrcaStudio）
- [x] **阶段 2.3 完成**：Lesson 3（`03_rl_ppo.py`）训练 100k 步后 `mean_reward > -5`
- [x] **架构合规**：K1–K12 约束回归测试全部通过
- [x] **源码 grep**：§7.2 所有 grep 命令无结果
- [x] **`SimpleEulerEnv` 违规修正**：`reset_model` 不再穿墙访问 `_sim._mjData`

### 8.2 交付物清单

| # | 交付物 | 类型 | 归属阶段 |
|---|--------|------|---------|
| 1 | `MuJoCoSimCore` 真实仿真实现 | 源码 + 单测 | 2.1 Step 1 |
| 2 | `SimConfig` 委托 mjModel.opt | 源码 + 单测 | 2.1 Step 2 |
| 3 | `ModelRegistry.build_orca_gym_model` 实现 | 源码 + 单测 | 2.1 Step 3 |
| 4 | `OrcaGymDataView` 接入 mjData | 源码 + 单测 | 2.1 Step 4 |
| 5 | `OrcaGymEuler` 委托填充 | 源码 + 单测 | 2.1 Step 5 |
| 6 | `OrcaGymEulerEnv` 生命周期/步进/状态设置填充 | 源码 + 单测 | 2.1 Step 6 |
| 7 | `SimpleEulerEnv.reset_model` 违规修正 | 源码 | 2.1 Step 7 |
| 8 | Lesson 1 端到端验证报告 | 测试运行日志 | 2.1 Step 8 |
| 9 | `OrcaStudioBridge` gRPC 通信填充 | 源码 + 单测 | 2.2 Step 1 |
| 10 | `OrcaGymEuler` Studio 委托 + override_ctrls | 源码 + 单测 | 2.2 Step 2 |
| 11 | `OrcaGymEulerEnv` 在线模式 + render 填充 | 源码 + 单测 | 2.2 Step 3 |
| 12 | Lesson 2 端到端验证报告 | 测试运行日志 | 2.2 Step 4 |
| 13 | Gymnasium 契约验证 | 测试运行日志 | 2.3 Step 1 |
| 14 | Lesson 3 PPO 训练/评估验证报告 | 测试运行日志 | 2.3 Step 2 |
| 15 | 架构约束回归测试套件 | 测试代码 | §7 |

### 8.3 未完成项（留待完整 P4）

阶段二**故意不实现**以下功能（三个 example 不需要）：

| 功能 | 原因 | 留待 |
|------|------|------|
| `query_joint_*`/`query_site_*`/`query_sensor_*`/`query_contact_*` | 三个 example 通过 `env.data` 直接读 | 完整 P4 / Lesson 4 |
| `apply_body_force`/`clear_*` | Lesson 5 外力应用 | 完整 P4 / Lesson 5 |
| `set_mocap_pos_and_quat`/`update_equality_constraints`/`set_geom_friction` | 高级状态设置 | 完整 P4 |
| `body_subtree_mass`/`equality_*` 查询 | Lesson 4/5 需要 | 完整 P4 / Lesson 4/5 |
| 视频录制/帧导出/内容文件下载 | 非 RL 必需 | 完整 P4 |
| Euler 耦合真实实现 | `has_euler()=False`，纯 MuJoCo | P5+（Euler 集成阶段） |
| `do_body_manipulation` 体操作力应用 | 阶段二仅查询不应用 | 完整 P4 |

---

## 9. 风险与缓解

### 9.1 技术风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| `set_time_step` 在 `init_simulation` 前调用导致 SimConfig 未绑定 | `set_time_step` 抛 `RuntimeError` | 缓存 `_time_step`，`init_simulation` 末尾重新设置（2.1 Step 6） |
| `OrcaGymDataView` 零拷贝视图在 `reset_data` 后失效 | `env.data.qpos` 指向旧数组 | `reset_data` 后必须 `sync_to_view`，`reset_simulation` 已保证 |
| gRPC `UpdateLocalEnv` 在线模式超时 | Lesson 2 渲染卡顿 | 设置合理 timeout，离线模式 no-op |
| SB3 `Monitor` 包装器与 `OrcaGymEulerEnv` 不兼容 | Lesson 3 训练失败 | 2.3 Step 1 先验证 Gymnasium 契约 |
| `_bind(sim._mjModel)` 层内访问被视为违规 | grep 误报 | 文档明确层内访问豁免；或重构为 `SimCore.bind_config(config)` |

### 9.2 架构合规风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| 填充时图便利直接触 `_mjData` | K4/K5 回退 | §7.2 grep 断言作为 CI 门禁 |
| `SimpleEulerEnv` 新增违规 | 隔离机制失效 | PR review + grep 检查 `OrcaPlayground/envs/euler/` |
| `OrcaGymDataView` 暴露 `_mj_data` | 内部状态泄漏 | `_mj_data` 带下划线，`__dir__` 过滤（如有） |

---

## 10. 实施顺序与依赖关系

```
阶段 2.1（离线核心）
  ├─ Step 1: MuJoCoSimCore       ← 无依赖
  ├─ Step 2: SimConfig           ← 依赖 Step 1（_bind 需要 mjModel）
  ├─ Step 3: ModelRegistry       ← 依赖 Step 1
  ├─ Step 4: OrcaGymDataView     ← 依赖 Step 1
  ├─ Step 5: OrcaGymEuler        ← 依赖 Step 1-4
  ├─ Step 6: OrcaGymEulerEnv     ← 依赖 Step 5
  ├─ Step 7: SimpleEulerEnv 修正 ← 依赖 Step 6
  └─ Step 8: Lesson 1 验证       ← 依赖 Step 7

阶段 2.2（gRPC 渲染）
  ├─ Step 1: OrcaStudioBridge    ← 依赖 2.1 Step 5
  ├─ Step 2: OrcaGymEuler Studio ← 依赖 2.2 Step 1
  ├─ Step 3: OrcaGymEulerEnv 在线 ← 依赖 2.2 Step 2
  └─ Step 4: Lesson 2 验证       ← 依赖 Step 3 + OrcaStudio

阶段 2.3（RL 验证）
  ├─ Step 1: Gymnasium 契约验证  ← 依赖 2.1 Step 8
  └─ Step 2: SB3 PPO 验证        ← 依赖 Step 1
```

> **可并行**：阶段 2.1 的 Step 1-4 可并行开发（仅 Step 1 是共同依赖）。阶段 2.3 可与 2.2 并行（Lesson 3 不依赖 gRPC）。

---

## 11. 附录：API 契约速查

### 11.1 阶段二新增/修改的公共 API

| 组件 | API | 签名 | 说明 |
|------|-----|------|------|
| `MuJoCoSimCore` | `reset_data()` | `-> None` | 重置 MjData |
| `MuJoCoSimCore` | `set_qpos_qvel(qpos, qvel)` | `-> None` | 设置广义状态 |
| `MuJoCoSimCore` | `nq`/`nv`/`nu` property | `-> int` | 维度查询 |
| `SimConfig` | `_bind(mj_model)` | `-> None` | 绑定真实 mjModel |
| `ModelRegistry` | `_bind(mj_model)` | `-> None` | 绑定真实 mjModel |
| `OrcaGymDataView` | `_sync_from_mjdata(mj_data, mj_model)` | `-> None` | 同步基本字段 |
| `OrcaGymDataView` | `body_xpos/xquat/xmat/cvel(body_name)` | `-> np.ndarray` | Body 查询 |
| `OrcaGymDataView` | `site_xpos/xmat(site_name)` | `-> np.ndarray` | Site 查询 |
| `OrcaGymEuler` | `model` property | `-> OrcaGymModel` | 缓存的模型对象 |
| `OrcaGymEuler` | `nq`/`nu` property | `-> int` | 维度查询 |
| `OrcaGymEuler` | `reset_data()` | `-> None` | 重置 MjData |
| `OrcaGymEuler` | `set_qpos_qvel(qpos, qvel)` | `-> None` | 设置广义状态 |
| `OrcaGymEuler` | `step_with_coupling(ctrl, n_frames, dt)` | `-> None` | 带耦合步进 |
| `OrcaGymEuler` | `render()` async | `-> None` | 渲染到 Studio |
| `OrcaGymEulerEnv` | `set_joint_qpos(qpos)` | `-> None` | 设置 qpos（reset 用） |
| `OrcaGymEulerEnv` | `set_joint_qvel(qvel)` | `-> None` | 设置 qvel（reset 用） |
| `OrcaGymEulerEnv` | `render()` | `-> None \| NDArray` | 渲染 |
| `OrcaStudioBridge` | `configure_offline(xml_path, assets_dir)` | `-> None` | 离线配置 |
| `OrcaStudioBridge` | `render(qpos, sim_time)` async | `-> None` | 依赖反转渲染 |
| `OrcaStudioBridge` | `get_override_ctrls()` | `-> dict[int, float]` | 控制覆盖查询 |

### 11.2 阶段二保持 `NotImplementedError` 的方法

| 组件 | 方法 | 留待 |
|------|------|------|
| `MuJoCoSimCore` | `apply_body_force` | 完整 P4 |
| `MuJoCoSimCore` | `clear_*` | 完整 P4 |
| `ModelRegistry` | `build_orca_gym_data` | 完整 P4 |
| `ModelRegistry` | `body_subtree_mass`/`equality_*` | 完整 P4 |
| `OrcaStudioBridge` | 视频/帧/内容文件方法 | 完整 P4 |
| `OrcaGymEulerEnv` | `do_body_manipulation` 力应用部分 | 完整 P4 |

---

**文档结束**

本文档定义了阶段二的完整开发路径。按 §10 的依赖顺序逐步实施，每步完成验收标准后进入下一步。阶段二完成后，三个 example 端到端可用，且架构约束 K1–K12 严格保持，为完整 P4 和后续 Euler 耦合集成奠定基础。