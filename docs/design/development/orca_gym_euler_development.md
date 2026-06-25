# OrcaGymEulerEnv + OrcaGymEuler 开发设计文档

## 1. 文档定位

### 1.1 文档目标

本文是 [orca_gym_euler_architecture.md](../architecture/orca_gym_euler_architecture.md) 的配套开发设计文档，将架构设计分解为可独立验证的开发阶段，每个阶段交付可运行的代码和单元测试。

### 1.2 开发原则

| 原则 | 含义 |
|------|------|
| **阶段独立验证** | 每个阶段完成后可独立运行单元测试验证，不依赖后续阶段 |
| **尽早联调** | 前期基础设施阶段完成后立即开始与 OrcaStudio 端到端联调 |
| **契约先行** | 每个阶段明确该阶段需满足的 API 契约（参见架构文档第 6 章） |
| **测试驱动** | 单元测试与实现同步开发，测试放在指定目录 |

### 1.3 测试目录约定

| 测试对象 | 目录 |
|---------|------|
| `OrcaGymEuler`（core 层） | `tests/orca_gym/core/euler/` |
| `OrcaGymEulerEnv`（environment 层） | `tests/orca_gym/environment/euler/` |

### 1.4 阶段总览

| 阶段 | 名称 | 核心交付 | 联调能力 |
|------|------|---------|---------|
| **P1** | 基础设施骨架 | MuJoCoSimCore + OrcaGymEuler 骨架 + 封装隔离机制 | 无 |
| **P2** | 状态视图与配置 | OrcaGymDataView + SimConfig + ModelRegistry | 无 |
| **P3** | Studio 集成与端到端联调 | OrcaStudioBridge + OrcaGymEulerEnv 骨架 + `run_simple.py` | **可联调** |
| **P4** | API 完备化 | query_*/set_*/apply_body_force 等完整 API | 可联调 |
| **P5** | 典型用户模式 E2E 验证 | OrcaPlayground 4 个典型模式 example + Protocol | 可联调 |

**E2E 验证策略**：不构造 `OrcaGymEulerEnv` 桩，采用外置 OrcaPlayground example 驱动，这是最终用户的典型用法，是真正的 E2E 验证。P3 的 `run_simple.py` 提供最早联调入口，P5 扩展到 4 个典型用户模式全覆盖。

---

## 2. 阶段 P1：基础设施骨架

### 2.1 目标

搭建 `MuJoCoSimCore` + `OrcaGymEuler` 骨架，实现封装隔离机制（`__getattr__`/`__dir__`），验证纯 MuJoCo 仿真能力（加载模型、step、forward）。

### 2.2 交付物

#### 2.2.1 `orca_gym/core/euler/mujoco_sim_core.py`

```python
class MuJoCoSimCore:
    """MuJoCo 仿真核心，持有 _mjModel/_mjData，不对外暴露。"""

    def __init__(self): ...
    def init_simulation(self, model_xml_path: str) -> None: ...
    def step(self, nstep: int) -> None: ...
    def forward(self) -> None: ...
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...
    def apply_body_force(self, body_id: int, force: np.ndarray, torque: np.ndarray) -> None: ...
    def clear_body_force(self, body_id: int) -> None: ...
    def clear_all_forces(self) -> None: ...
    def sync_to_view(self, view: OrcaGymDataView) -> None: ...  # P2 实现
```

#### 2.2.2 `orca_gym/core/euler/orca_gym_euler.py`

```python
class OrcaGymEuler:
    """仿真核心 Facade，组合子组件，不暴露 _mjModel/_mjData。"""

    _BLOCKED_ATTRS = frozenset({
        "_mjData", "_mjModel", "mj_data", "mj_model",
        "_mj_data", "_mj_model", "mjData", "mjModel",
    })

    def __init__(self, stub: GrpcServiceStub | None = None): ...
    def __getattr__(self, name: str): ...  # 拦截引导
    def __dir__(self): ...  # 控制可见性

    # 委托到 _sim
    def init_simulation(self, model_xml_path: str) -> None: ...
    def mj_step(self, nstep: int) -> None: ...
    def mj_forward(self) -> None: ...
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...
```

#### 2.2.3 封装隔离机制

- `OrcaGymEuler.__getattr__` 拦截 `_mjData`/`_mjModel` 等属性，返回引导性错误
- `OrcaGymEuler.__dir__` 只暴露公共 API

### 2.3 单元测试

#### `tests/orca_gym/core/euler/test_mujoco_sim_core.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_init_simulation_loads_model` | 加载简单 MJCF，`_mjModel`/`_mjData` 非 None |
| `test_step_advances_time` | `step(1)` 后 `_mjData.time` 增加 timestep |
| `test_forward_updates_derived` | `forward()` 后 `qacc` 非零 |
| `test_set_ctrl_sets_actuator` | `set_ctrl` 后 `_mjData.ctrl` 与输入一致 |
| `test_apply_body_force_writes_xfrc` | `apply_body_force` 后 `xfrc_applied` 对应位置非零 |
| `test_clear_body_force_zeros_xfrc` | `clear_body_force` 后对应位置为零 |
| `test_clear_all_forces_zeros_all` | `clear_all_forces` 后 `xfrc_applied` 全零 |

#### `tests/orca_gym/core/euler/test_orca_gym_euler.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_blocked_attrs_raise_guidance_error` | 访问 `_mjData`/`_mjModel` 抛出 `AttributeError` 且消息含引导文本 |
| `test_dir_only_exposes_public_api` | `dir(gym)` 不含 `_mjData`/`_mjModel`/`_sim` |
| `test_init_simulation_delegates_to_sim_core` | `gym.init_simulation(path)` 后 sim_core 已加载 |
| `test_mj_step_delegates` | `gym.mj_step(1)` 后 time 推进 |
| `test_mj_forward_delegates` | `gym.mj_forward()` 不报错 |
| `test_set_ctrl_delegates` | `gym.set_ctrl(ctrl)` 后 sim_core 的 ctrl 一致 |

### 2.4 验收标准

- [ ] 所有 P1 单元测试通过
- [ ] `MuJoCoSimCore` 能加载 MJCF 并执行 `mj_step`/`mj_forward`
- [ ] `OrcaGymEuler` 访问 `_mjData` 抛出引导性错误
- [ ] `dir(OrcaGymEuler())` 不含内部组件

---

## 3. 阶段 P2：状态视图与配置

### 3.1 目标

实现 `OrcaGymDataView`（完整状态只读视图）和 `SimConfig`（求解器配置），扩展 `ModelRegistry`，覆盖架构文档第 9.2 节列出的 83 处绕道场景的替代 API。

### 3.2 交付物

#### 3.2.1 `orca_gym/core/euler/orca_gym_data_view.py`

```python
class OrcaGymDataView:
    """MuJoCo 状态的完整只读视图，替代直接访问 _mjData。"""

    # 基本状态
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    qfrc_bias: np.ndarray
    time: float

    # 扩展字段
    xfrc_applied: np.ndarray       # 只读视图
    actuator_force: np.ndarray
    contact: list

    # body 属性查询
    def body_xpos(self, body_name: str) -> np.ndarray: ...
    def body_xquat(self, body_name: str) -> np.ndarray: ...
    def body_xmat(self, body_name: str) -> np.ndarray: ...
    def body_cvel(self, body_name: str) -> np.ndarray: ...
    def body_subtree_mass(self, body_name: str) -> float: ...

    # site 属性查询
    def site_xpos(self, site_name: str) -> np.ndarray: ...
    def site_xmat(self, site_name: str) -> np.ndarray: ...

    def __getattr__(self, name: str): ...  # 兜底引导
```

#### 3.2.2 `orca_gym/core/euler/sim_config.py`

```python
class SimConfig:
    """MuJoCo 求解器参数配置，替代 _mjModel.opt.* 直接访问。"""

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
    def to_dict(self) -> dict: ...
```

#### 3.2.3 `orca_gym/core/euler/model_registry.py`

```python
class ModelRegistry:
    """模型信息注册，构建 OrcaGymModel，提供扩展查询。"""

    def __init__(self, mj_model: mujoco.MjModel): ...
    def build_orca_gym_model(self) -> OrcaGymModel: ...
    def build_orca_gym_data(self) -> OrcaGymData: ...

    # 扩展查询（替代 _mjModel 直接访问）
    def body_subtree_mass(self, body_name: str) -> float: ...
    def equality_data_width(self) -> int: ...
    def equality_object_ids(self, eq_idx: int) -> tuple[int, int]: ...
    def joint_name_by_id(self, joint_id: int) -> str: ...
```

#### 3.2.4 `MuJoCoSimCore.sync_to_view` 实现

```python
class MuJoCoSimCore:
    def sync_to_view(self, view: OrcaGymDataView) -> None:
        """将 _mjData 状态同步到 DataView。"""
        view._qpos = self._mjData.qpos.copy()
        view._qvel = self._mjData.qvel.copy()
        view._qacc = self._mjData.qacc.copy()
        view._qfrc_bias = self._mjData.qfrc_bias.copy()
        view._time = self._mjData.time
        view._xfrc_applied = self._mjData.xfrc_applied  # 只读视图，不 copy
        view._actuator_force = self._mjData.actuator_force.copy()
        # ... 其他字段
```

### 3.3 单元测试

#### `tests/orca_gym/core/euler/test_orca_gym_data_view.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_qpos_qvel_qacc_consistent_after_sync` | sync 后 DataView 字段与 `_mjData` 一致 |
| `test_body_xpos_by_name` | `body_xpos("world")` 返回正确位置 |
| `test_body_cvel_by_name` | `body_cvel(body_name)` 返回正确速度 |
| `test_body_subtree_mass_by_name` | `body_subtree_mass(body_name)` 返回正确质量 |
| `test_site_xpos_by_name` | `site_xpos(site_name)` 返回正确位置 |
| `test_xfrc_applied_is_read_only_view` | `xfrc_applied` 是 `_mjData.xfrc_applied` 的视图 |
| `test_missing_field_raises_guidance` | 访问不存在的字段抛出引导性错误 |
| `test_time_field` | `time` 字段与 `_mjData.time` 一致 |

#### `tests/orca_gym/core/euler/test_sim_config.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_timestep_get_set` | 读写 `timestep` 反映到 `_mjModel.opt.timestep` |
| `test_integrator_get_set` | 读写 `integrator` 反映到 `_mjModel.opt.integrator` |
| `test_iterations_get_set` | 读写 `iterations` 反映到 `_mjModel.opt.iterations` |
| `test_gravity_get_set` | 读写 `gravity` 反映到 `_mjModel.opt.gravity` |
| `test_load_from_dict` | `load_from_dict({...})` 批量设置多个参数 |
| `test_to_dict` | `to_dict()` 返回所有参数的字典 |
| `test_all_opt_fields_covered` | 遍历 `_mjModel.opt` 所有字段，确认 SimConfig 都有对应属性 |

#### `tests/orca_gym/core/euler/test_model_registry.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_build_orca_gym_model` | 构建的 `OrcaGymModel` 与原 `OrcaGymLocal` 一致 |
| `test_body_subtree_mass` | `body_subtree_mass(name)` 与 `_mjModel.body_subtreemass[id]` 一致 |
| `test_equality_data_width` | `equality_data_width()` 与 `_mjModel.eq_data.shape[1]` 一致 |
| `test_equality_object_ids` | `equality_object_ids(idx)` 与 `_mjModel.eq_obj1id/eq_obj2id` 一致 |
| `test_joint_name_by_id` | `joint_name_by_id(i)` 与 `mujoco.mj_id2name` 一致 |

### 3.4 验收标准

- [ ] 所有 P2 单元测试通过
- [ ] `OrcaGymDataView` 覆盖原 `OrcaGymData` 的 5 个字段 + 扩展字段
- [ ] `SimConfig` 覆盖 `_mjModel.opt` 的所有用户可访问字段
- [ ] `ModelRegistry` 扩展查询与 `_mjModel` 直接访问结果一致
- [ ] DataView 缺字段时抛出引导性错误

---

## 4. 阶段 P3：Studio 集成与端到端联调

### 4.1 目标

实现 `OrcaStudioBridge`，完成 `OrcaGymEulerEnv` 骨架，**启动与 OrcaStudio 的端到端联调**，验证模型加载、渲染、步进、状态同步的完整链路。

### 4.2 交付物

#### 4.2.1 `orca_gym/core/euler/orca_studio_bridge.py`

```python
class OrcaStudioBridge:
    """OrcaStudio gRPC 集成，依赖反转，不持有 _mjData。"""

    def __init__(self, stub: GrpcServiceStub | None): ...
    async def load_model_xml(self) -> str: ...
    async def render(self, qpos: np.ndarray, sim_time: float) -> None: ...
    async def pause_simulation(self) -> None: ...
    async def begin_save_video(self, path: str, mode: CaptureMode) -> None: ...
    async def stop_save_video(self) -> None: ...
    async def get_current_frame(self) -> int: ...
    async def get_camera_time_stamp(self, last_frame: int) -> dict: ...
    async def get_frame_png(self, image_path: str) -> None: ...
    async def get_body_manipulation_anchored(self) -> tuple: ...
    async def get_body_manipulation_movement(self) -> dict: ...
    async def load_content_file(self, ...) -> str: ...
```

**设计要点**：
- `render(qpos, sim_time)` 接收数据参数，不直接访问 `_mjData`（依赖反转）
- 所有方法委托 gRPC stub，与 `OrcaGymLocal` 的对应方法逻辑一致

#### 4.2.2 `orca_gym/environment/orca_gym_euler_env.py`（骨架）

```python
class OrcaGymEulerEnv(OrcaGymBaseEnv):
    """OrcaGym Euler 环境 Facade。"""

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        **kwargs,
    ): ...

    # 生命周期
    def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]: ...
    def initialize_grpc(self): ...
    def pause_simulation(self): ...
    def close(self): ...

    # 仿真控制
    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None: ...
    def mj_step(self, nstep: int): ...
    def mj_forward(self): ...
    def set_ctrl(self, ctrl): ...

    # 状态访问
    @property
    def data(self) -> OrcaGymDataView: ...
    @property
    def model(self) -> OrcaGymModel: ...
    @property
    def sim_config(self) -> SimConfig: ...

    # 渲染
    def render(self): ...

    # 名称空间（从 OrcaGymBaseEnv 继承或重写）
    def joint(self, name: str) -> str: ...
    def body(self, name: str) -> str: ...
    def site(self, name: str) -> str: ...
    def actuator(self, name: str) -> str: ...
    def sensor(self, name: str) -> str: ...
```

#### 4.2.3 `OrcaGymEuler` 完善子组件组合

```python
class OrcaGymEuler:
    def __init__(self, stub: GrpcServiceStub | None = None):
        self._sim = MuJoCoSimCore()
        self._studio = OrcaStudioBridge(stub)
        self._registry: ModelRegistry | None = None
        self._opt: SimConfig | None = None
        self._euler: EulerOrchestrator | None = None  # 占位

    async def init_simulation(self, model_xml_path: str) -> None:
        await self._sim.init_simulation(model_xml_path)
        self._registry = ModelRegistry(self._sim._mjModel)
        self._opt = SimConfig(self._sim._mjModel)

    async def load_model_xml(self) -> str:
        return await self._studio.load_model_xml()

    async def render(self, qpos: np.ndarray, sim_time: float) -> None:
        await self._studio.render(qpos, sim_time)
```

### 4.3 单元测试

#### `tests/orca_gym/core/euler/test_orca_studio_bridge.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_init_with_none_stub` | `stub=None` 时不报错，方法可跳过 |
| `test_load_model_xml_calls_stub` | mock stub，验证 gRPC 调用 |
| `test_render_passes_qpos_and_time` | mock stub，验证 `render(qpos, time)` 传参 |
| `test_pause_simulation_calls_stub` | mock stub，验证调用 |
| `test_begin_save_video_calls_stub` | mock stub，验证调用 |

#### `tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_init_creates_gym` | `__init__` 后 `_gym` 非 None |
| `test_data_property_returns_view` | `env.data` 返回 `OrcaGymDataView` |
| `test_model_property_returns_model` | `env.model` 返回 `OrcaGymModel` |
| `test_sim_config_property_returns_config` | `env.sim_config` 返回 `SimConfig` |
| `test_do_simulation_advances_time` | `do_simulation(ctrl, 1)` 后 `env.data.time` 增加 |
| `test_mj_step_advances_time` | `mj_step(1)` 后 time 推进 |
| `test_mj_forward_no_error` | `mj_forward()` 不报错 |
| `test_set_ctrl_sets_control` | `set_ctrl(ctrl)` 后内部 ctrl 一致 |
| `test_joint_namespace_resolution` | `env.joint("j1")` 返回带 agent 前缀的名称 |

### 4.4 端到端联调验证（OrcaPlayground 真实 example 驱动）

**核心原则**：端到端验证不构造 `OrcaGymEulerEnv` 桩，而是采用外置的 OrcaPlayground example 来驱动，这是最终用户的典型用法，是真正的 E2E 验证。

#### 4.4.1 OrcaPlayground 集成结构

在 OrcaPlayground 项目中创建 `euler` 目录，P3 阶段先落地最小可联调的 example：

```
OrcaPlayground/
├── envs/
│   └── euler/                          # Euler Env 子类目录
│       ├── __init__.py
│       └── simple_env.py               # P3：最小联调 Env
├── examples/
│   └── euler/                          # Euler example 入口脚本目录
│       ├── __init__.py
│       └── run_simple.py               # P3：最小联调入口
```

#### 4.4.2 `envs/euler/simple_env.py`（P3 最小联调 Env）

对应 D12Env 的"简单委托式"开发模式，验证最基本的 API 契约：

```python
import numpy as np
from orca_gym.environment.orca_gym_euler_env import OrcaGymEulerEnv


class SimpleEulerEnv(OrcaGymEulerEnv):
    """最小联调 Env，验证基本 API 契约（对应 D12Env 模式）。"""

    def __init__(self, frame_skip, orcagym_addr, agent_names, time_step, **kwargs):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

    def reset_model(self) -> tuple[dict, dict]:
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        self.mj_forward()
        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        # 使用正式 API（env.data），不访问 gym._mjData
        return {
            "qpos": np.array(self.data.qpos, dtype=np.float32),
            "qvel": np.array(self.data.qvel, dtype=np.float32),
            "ctrl": np.array(self.ctrl, dtype=np.float32),
        }
```

#### 4.4.3 `examples/euler/run_simple.py`（P3 最小联调入口）

```python
"""SimpleEulerEnv 端到端联调入口。

用法:
  python run_simple.py --orcagym_addr <ip:port> --scene <scene_xml>
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from envs.euler.simple_env import SimpleEulerEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orcagym_addr", required=True)
    parser.add_argument("--scene", required=True, help="场景 XML 路径")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    env = SimpleEulerEnv(
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
        agent_names=["agent0"],
        time_step=0.002,
        model_xml_path=args.scene,
    )

    for ep in range(args.episodes):
        obs, info = env.reset()
        for step in range(100):
            ctrl = np.zeros(env.nu, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(ctrl)
            env.render()
            if terminated or truncated:
                break

    env.close()


if __name__ == "__main__":
    main()
```

#### 4.4.4 P3 联调验证清单

通过 `run_simple.py` 驱动，验证完整链路：

- [ ] OrcaStudio 启动，加载测试场景 XML
- [ ] `SimpleEulerEnv` 连接 gRPC，加载模型
- [ ] `env.data.qpos`/`env.data.qvel` 读取正确（验证 DataView）
- [ ] `env.do_simulation` 步进，Studio 视口实时更新
- [ ] `env.sim_config.timestep` 修改生效（验证 SimConfig）
- [ ] `env.render()` 触发 Studio 渲染
- [ ] `env.data` 访问不触发 `_mjData` 拦截（验证封装隔离不误伤）
- [ ] 100 步循环无异常

### 4.5 验收标准

- [ ] 所有 P3 单元测试通过
- [ ] `examples/euler/run_simple.py` 可连接 OrcaStudio 完成完整循环
- [ ] `OrcaGymEulerEnv` 可加载模型、步进、渲染
- [ ] `env.data`/`env.model`/`env.sim_config` 可正常访问
- [ ] Studio 视口显示与仿真状态同步
- [ ] `env.data` 访问不误触发封装隔离拦截

---

## 5. 阶段 P4：API 完备化

### 5.1 目标

实现架构文档第 6.7 节列出的完整公共 API，覆盖所有 `query_*`/`set_*`/`apply_body_force` 等方法，使用户可以零绕道使用。

### 5.2 交付物

#### 5.2.1 状态查询方法（`OrcaGymEulerEnv` + `OrcaGymEuler`）

从 `OrcaGymLocal` 迁移以下方法，签名保持一致：

| 方法 | 来源 |
|------|------|
| `query_joint_qpos(qjoint_names)` | `OrcaGymLocal:2190` |
| `query_joint_qvel(joint_names)` | `OrcaGymLocal:2215` |
| `query_joint_qacc(joint_names)` | `OrcaGymLocal:2239` |
| `query_joint_offsets(joint_names)` | `OrcaGymLocal:1714` |
| `query_joint_lengths(joint_names)` | `OrcaGymLocal:1745` |
| `query_joint_dofadrs(joint_names)` | `OrcaGymLocal:2742` |
| `query_site_pos_and_quat(site_names)` | `OrcaGymLocal:2303`（重写为返回 quat） |
| `query_site_pos_and_mat(site_names)` | `OrcaGymLocal:2303` |
| `query_site_size(site_names)` | `OrcaGymLocal:2327` |
| `query_site_xvalp_xvalr(site_names)` | `OrcaGymLocal:957`（Env 层） |
| `query_site_pos_and_quat_B(site_names, base_body_list)` | `OrcaGymLocal:882`（Env 层） |
| `query_sensor_data(sensor_names)` | `OrcaGymLocal:1836` |
| `query_actuator_torques(actuator_names)` | `OrcaGymLocal:2690` |
| `query_contact_simple()` | `OrcaGymLocal:2556` |
| `query_contact_force(contact_ids)` | `OrcaGymLocal:2636` |
| `get_body_xpos_xmat_xquat(body_name_list)` | `OrcaGymLocal:668`（Env 层） |
| `get_body_xpos_xmat_xquat_xvel(body_name_list)` | `OrcaGymLocal:721`（Env 层） |
| `query_velocity_body_B(ee_body, base_body)` | `OrcaGymLocal:2765` |
| `query_position_body_B(ee_body, base_body)` | `OrcaGymLocal:2811` |
| `query_orientation_body_B(ee_body, base_body)` | `OrcaGymLocal:2849` |
| `query_joint_axes_B(joint_names, base_body)` | `OrcaGymLocal:2888` |
| `query_robot_velocity_odom(...)` | `OrcaGymLocal:2926` |
| `query_robot_position_odom(...)` | `OrcaGymLocal:2968` |
| `query_robot_orientation_odom(...)` | `OrcaGymLocal:3003` |
| `jnt_qposadr(joint_name)` | `OrcaGymLocal:2283` |
| `jnt_dofadr(joint_name)` | `OrcaGymLocal:2303` |

#### 5.2.2 状态设置方法

| 方法 | 来源 |
|------|------|
| `set_joint_qpos(joint_qpos)` | `OrcaGymLocal:2351` |
| `set_joint_qvel(joint_qvel)` | `OrcaGymLocal:2371` |
| `set_mocap_pos_and_quat(mocap_pos_and_quat_dict)` | `OrcaGymLocal:2519`（Env 层） |
| `update_equality_constraints(eq_list)` | `OrcaGymLocal:2451` |
| `modify_equality_objects(...)` | `OrcaGymLocal:2419` |
| `set_geom_friction(geom_friction_dict)` | `OrcaGymLocal:2592` |
| `add_extra_weight(weight_load_dict)` | `OrcaGymLocal:2611` |
| `set_actuator_trnid(actuator_id, trnid)` | `OrcaGymLocal:1261` |
| `disable_actuator(actuator_groups)` | `OrcaGymLocal:1280` |

#### 5.2.3 新增力应用方法

| 方法 | 说明 |
|------|------|
| `apply_body_force(body_name, force, torque)` | 替代 `xfrc_applied` 直接写入 |
| `clear_body_force(body_name)` | 清除单个 body 外力 |
| `clear_all_forces()` | 清除所有外力 |
| `mj_apply_force_at_site(site_name, force, torque)` | 从 `OrcaGymLocal:2132` 迁移 |
| `mj_clear_xfrc_applied_for_site(site_name)` | 从 `OrcaGymLocal:2168` 迁移 |

#### 5.2.4 Studio 交互方法（Env 层）

| 方法 | 来源 |
|------|------|
| `begin_save_video(file_path, capture_mode)` | `OrcaGymLocalEnv:154` |
| `stop_save_video()` | `OrcaGymLocalEnv:160` |
| `get_current_frame()` | `OrcaGymLocalEnv:182` |
| `get_next_frame()` | `OrcaGymLocalEnv:166` |
| `get_camera_time_stamp(last_frame_index)` | `OrcaGymLocalEnv:188` |
| `get_frame_png(image_path)` | `OrcaGymLocalEnv:194` |
| `get_body_manipulation_anchored()` | `OrcaGymLocalEnv:148` |
| `get_body_manipulation_movement()` | `OrcaGymLocalEnv:200` |
| `anchor_actor(actor_name, anchor_type)` | `OrcaGymLocalEnv:347` |
| `release_body_anchored()` | `OrcaGymLocalEnv:329` |
| `do_body_manipulation()` | `OrcaGymLocalEnv:282` |
| `update_anchor_equality_constraints(...)` | `OrcaGymLocalEnv:417` |

#### 5.2.5 其他方法

| 方法 | 来源 |
|------|------|
| `mj_jacBody(jacp, jacr, body_id)` | `OrcaGymLocal:2083` |
| `mj_jacSite(jacp, jacr, site_id)` | `OrcaGymLocal:2108` |
| `mj_jac_site(site_names)` | `OrcaGymLocal:2391` |
| `mj_inverse()` | `OrcaGymLocal:2037` |
| `mj_fullM()` | `OrcaGymLocal:2057` |
| `get_cfrc_ext()` | `OrcaGymLocal:2668` |
| `get_goal_bounding_box(geom_name)` | `OrcaGymLocalEnv:1078` |
| `load_content_file(...)` | `OrcaGymLocalEnv:1121` |
| `set_time_step(time_step)` | `OrcaGymLocalEnv:586` |
| `update_data()` | `OrcaGymLocalEnv:595` |
| `reset_simulation()` | `OrcaGymLocalEnv:615` |
| `init_qpos_qvel()` | `OrcaGymLocalEnv:635` |

### 5.3 单元测试

#### `tests/orca_gym/environment/euler/test_query_methods.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_query_joint_qpos` | 查询结果与 `_mjData.qpos` 对应位置一致 |
| `test_query_joint_qvel` | 查询结果与 `_mjData.qvel` 对应位置一致 |
| `test_query_joint_qacc` | 查询结果与 `_mjData.qacc` 对应位置一致 |
| `test_query_joint_offsets` | 偏移与 `mujoco.mj_jnt_qposadr` 一致 |
| `test_query_joint_lengths` | 长度与关节类型对应 |
| `test_query_site_pos_and_quat` | 返回的 xpos/xquat 与 `_mjData.site_xpos` 一致 |
| `test_query_site_pos_and_mat` | 返回的 xpos/xmat 与 `_mjData.site_xmat` 一致 |
| `test_query_site_xvalp_xvalr` | 返回的线速度/角速度正确 |
| `test_query_sensor_data` | 传感器数据与 `_mjData.sensordata` 一致 |
| `test_query_actuator_torques` | 执行器力矩与 `_mjData.actuator_force` 一致 |
| `test_query_contact_simple` | 接触列表非空（有接触场景） |
| `test_get_body_xpos_xmat_xquat` | 返回值与 `_mjData.xpos/xmat/xquat` 一致 |
| `test_query_velocity_body_B` | B 系速度计算正确 |
| `test_query_position_body_B` | B 系位置计算正确 |
| `test_query_robot_position_odom` | 里程计位置计算正确 |

#### `tests/orca_gym/environment/euler/test_set_methods.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_set_joint_qpos` | 设置后 `_mjData.qpos` 对应位置更新 |
| `test_set_joint_qvel` | 设置后 `_mjData.qvel` 对应位置更新 |
| `test_set_mocap_pos_and_quat` | 设置后 `_mjData.mocap_pos/mocap_quat` 更新 |
| `test_update_equality_constraints` | 设置后 `_mjData.eq_*` 更新 |
| `test_set_geom_friction` | 设置后 `_mjModel.geom_friction` 更新 |
| `test_apply_body_force` | 设置后 `_mjData.xfrc_applied` 对应位置非零 |
| `test_clear_body_force` | 清除后对应位置为零 |
| `test_clear_all_forces` | 清除后全部为零 |
| `test_mj_apply_force_at_site` | 通过 site 施加力后 `xfrc_applied` 更新 |

#### `tests/orca_gym/environment/euler/test_studio_methods.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_begin_stop_save_video` | mock bridge，验证调用链 |
| `test_get_current_frame` | mock bridge，返回帧号 |
| `test_anchor_actor` | mock bridge，验证锚点设置 |
| `test_release_body_anchored` | mock bridge，验证释放 |
| `test_do_body_manipulation` | mock bridge，验证操作流程 |

#### `tests/orca_gym/environment/euler/test_other_methods.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_mj_jacBody` | 雅可比矩阵形状和值正确 |
| `test_mj_jacSite` | 雅可比矩阵形状和值正确 |
| `test_mj_inverse` | 逆动力学计算不报错 |
| `test_set_time_step` | 设置后 `sim_config.timestep` 更新 |
| `test_update_data` | 调用后 `env.data` 与内部一致 |
| `test_reset_simulation` | 重置后状态回到初始 |

### 5.4 验收标准

- [ ] 所有 P4 单元测试通过
- [ ] 架构文档第 6.7 节列出的所有公共 API 已实现
- [ ] 所有 `query_*` 方法返回值与 `OrcaGymLocal` 对应方法一致
- [ ] 所有 `set_*` 方法生效后 `_mjData`/`_mjModel` 正确更新
- [ ] `apply_body_force`/`clear_body_force` 正确操作 `xfrc_applied`

---

## 6. 阶段 P5：典型用户模式 E2E 验证

### 6.1 目标

在 OrcaPlayground 的 `envs/euler` 和 `examples/euler` 目录下，为每种典型用户开发模式设计一个 `OrcaGymEulerEnv` 子类和对应的命令行入口脚本，作为真正的 E2E 验证。同时引入 `OrcaGymEnvProtocol` 平滑外围组件迁移。

**核心原则**：P5 不再是"迁移原 Env"，而是"用 Euler 体系重新实现典型用户模式"，每个 example 都是最终用户的真实用法，是真正的 E2E 验证。

### 6.2 交付物

#### 6.2.1 `orca_gym/environment/protocols.py`

```python
from typing import Protocol

class OrcaGymEnvProtocol(Protocol):
    """OrcaGym 环境协议，OrcaGymLocalEnv 和 OrcaGymEulerEnv 都满足。"""
    model: OrcaGymModel
    data: OrcaGymDataView
    ctrl: np.ndarray
    frame_skip: int
    dt: float

    def do_simulation(self, ctrl, n_frames) -> None: ...
    def mj_step(self, nstep) -> None: ...
    def mj_forward(self) -> None: ...
    def set_ctrl(self, ctrl) -> None: ...
    def query_joint_qpos(self, names) -> dict: ...
    def query_joint_qvel(self, names) -> dict: ...
    def query_site_pos_and_quat(self, names) -> dict: ...
    def set_joint_qpos(self, qpos_dict) -> None: ...
    def set_mocap_pos_and_quat(self, mocap_dict) -> None: ...
    def apply_body_force(self, body_name, force, torque) -> None: ...
    # ... 其余公共方法 ...
```

#### 6.2.2 OrcaPlayground 典型用户模式 E2E 验证

在 P3 的 `simple_env.py` 基础上，P5 扩展覆盖所有典型用户开发模式。每个模式对应一个 `OrcaGymEulerEnv` 子类 + 一个命令行入口脚本：

```
OrcaPlayground/
├── envs/
│   └── euler/
│       ├── __init__.py
│       ├── simple_env.py           # P3：简单委托式（对应 D12Env）
│       ├── loop_env.py             # P5：手动循环式 + query_*（对应 G1Env）
│       ├── force_env.py            # P5：apply_body_force + equality（对应 fluid SimEnv）
│       └── multi_agent_env.py      # P5：多 agent + 接触查询 + 传感器（对应 LeggedSimEnv）
├── examples/
│   └── euler/
│       ├── __init__.py
│       ├── run_simple.py           # P3：简单委托式入口
│       ├── run_loop.py             # P5：手动循环式入口
│       ├── run_force.py            # P5：力应用入口
│       └── run_multi_agent.py      # P5：多 agent 入口
```

##### 模式 A：简单委托式（`simple_env.py`，P3 已完成）

对应 D12Env 模式：Env 仅做 `reset_model`/`_get_obs`，步进完全委托给基类 `do_simulation`。

验证点：
- `env.data.qpos`/`env.data.qvel` 读取
- `env.do_simulation` 步进
- `env.render()` 渲染

##### 模式 B：手动循环式（`loop_env.py`，P5）

对应 G1Env 模式：Env 在 `step` 中手动调用 `mj_step` + `query_*` 构建观测。

```python
class LoopEulerEnv(OrcaGymEulerEnv):
    """手动循环式 Env，验证 query_* API（对应 G1Env 模式）。"""

    def _get_obs(self) -> dict:
        joint_names = [...]
        return {
            "joint_qpos": self.query_joint_qpos(joint_names),
            "joint_qvel": self.query_joint_qvel(joint_names),
            "body_xpos": self.get_body_xpos_xmat_xquat(["pelvis"]),
            "sensor_data": self.query_sensor_data(["imu_gyro"]),
        }

    def step(self, action):
        self.set_ctrl(action)
        for _ in range(self.frame_skip):
            self.mj_step(1)
            self.update_data()
        return self._get_obs(), 0.0, False, False, {}
```

验证点：
- `query_joint_qpos`/`query_joint_qvel`
- `get_body_xpos_xmat_xquat`
- `query_sensor_data`
- 手动 `mj_step` + `update_data` 循环

##### 模式 C：力应用式（`force_env.py`，P5）

对应 fluid SimEnv 模式：Env 通过 `apply_body_force` 施加外力，操作 equality 约束。

```python
class ForceEulerEnv(OrcaGymEulerEnv):
    """力应用式 Env，验证 apply_body_force + equality（对应 fluid SimEnv 模式）。"""

    def apply_fluid_force(self, body_name: str, force: np.ndarray, torque: np.ndarray):
        # 使用正式 API，替代 self.gym._mjData.xfrc_applied[...] = ...
        self.apply_body_force(body_name, force, torque)

    def clear_forces(self):
        self.clear_all_forces()

    def setup_constraint(self, eq_list):
        self.update_equality_constraints(eq_list)
```

验证点：
- `apply_body_force` 替代 `xfrc_applied` 直接写入
- `clear_all_forces`
- `update_equality_constraints`
- 力应用后步进结果正确

##### 模式 D：多 Agent 复杂式（`multi_agent_env.py`，P5）

对应 LeggedSimEnv 模式：多 agent、接触查询、传感器、里程计。

```python
class MultiAgentEulerEnv(OrcaGymEulerEnv):
    """多 Agent 复杂式 Env，验证接触/传感器/里程计（对应 LeggedSimEnv 模式）。"""

    def _get_obs(self) -> dict:
        return {
            "joint_qpos": self.query_joint_qpos(self._joint_names),
            "contact": self.query_contact_simple(),
            "contact_force": self.query_contact_force(self._contact_ids),
            "actuator_torque": self.query_actuator_torques(self._actuator_names),
            "robot_pos_odom": self.query_robot_position_odom(
                self._base_body, self._init_pos, self._init_quat
            ),
        }
```

验证点：
- `query_contact_simple`/`query_contact_force`
- `query_actuator_torques`
- `query_robot_position_odom`
- 多 agent 名称空间解析

#### 6.2.3 命令行入口脚本

每个模式对应一个 `run_*.py`，结构参考 P3 的 `run_simple.py`：

| 入口脚本 | 驱动 Env | 验证模式 |
|---------|---------|---------|
| `run_simple.py` | `SimpleEulerEnv` | 简单委托式（P3） |
| `run_loop.py` | `LoopEulerEnv` | 手动循环式 + query_* |
| `run_force.py` | `ForceEulerEnv` | apply_body_force + equality |
| `run_multi_agent.py` | `MultiAgentEulerEnv` | 多 agent + 接触/传感器/里程计 |

#### 6.2.4 迁移指南文档

`docs/design/development/migration_guide.md`（P5 末尾撰写），包含：
- 83 处绕道的完整替代方案清单（架构文档第 9.2 节）
- 4 个典型用户模式的 Euler 实现示例（即 `envs/euler/` 下的 4 个 Env）
- 从原 `OrcaGymLocalEnv` 子类迁移到 `OrcaGymEulerEnv` 子类的步骤
- 常见陷阱和注意事项

### 6.3 单元测试

#### `tests/orca_gym/environment/euler/test_protocol_compliance.py`

| 测试用例 | 验证点 |
|---------|--------|
| `test_euler_env_satisfies_protocol` | `OrcaGymEulerEnv` 实例满足 `OrcaGymEnvProtocol` |
| `test_local_env_satisfies_protocol` | `OrcaGymLocalEnv` 实例满足 `OrcaGymEnvProtocol` |
| `test_protocol_method_signatures` | 协议方法签名与两个 Env 一致 |

#### `tests/orca_gym/environment/euler/test_euler_envs_api.py`

针对 `envs/euler/` 下的 4 个 Env，在无 Studio 环境下用本地 MJCF 验证 API 调用正确性：

| 测试用例 | 验证点 |
|---------|--------|
| `test_simple_env_obs_uses_data_view` | `SimpleEulerEnv._get_obs` 使用 `env.data` 而非 `gym._mjData` |
| `test_loop_env_query_methods` | `LoopEulerEnv` 的 query_* 返回正确 |
| `test_force_env_apply_body_force` | `ForceEulerEnv.apply_body_force` 生效 |
| `test_force_env_clear_forces` | `ForceEulerEnv.clear_forces` 清零 |
| `test_multi_agent_env_contacts` | `MultiAgentEulerEnv` 接触查询正常 |
| `test_multi_agent_env_odom` | `MultiAgentEulerEnv` 里程计计算正确 |
| `test_all_envs_no_mjdata_access` | 4 个 Env 源码无 `_mjData`/`_mjModel` 访问 |

### 6.4 E2E 验证清单（需 OrcaStudio 环境）

通过 4 个 `run_*.py` 入口脚本驱动，验证完整链路：

- [ ] `run_simple.py` 完成 100 步循环，Studio 视口更新
- [ ] `run_loop.py` 手动循环步进，query_* 返回正确
- [ ] `run_force.py` 施加外力后物体运动符合预期
- [ ] `run_multi_agent.py` 多 agent 接触/传感器/里程计正常
- [ ] 所有 4 个 Env 源码无 `_mjData`/`_mjModel` 直接访问

### 6.5 验收标准

- [ ] `OrcaGymEnvProtocol` 定义完成，两个 Env 都满足
- [ ] `envs/euler/` 下 4 个 EulerEnv 子类实现完成
- [ ] `examples/euler/` 下 4 个入口脚本可连接 OrcaStudio 完成完整循环
- [ ] 4 个 Env 源码无任何 `_mjData`/`_mjModel` 直接访问残留
- [ ] 迁移指南文档完成

---

## 7. 阶段依赖与并行性

### 7.1 依赖关系

```
P1 (基础设施骨架)
 │
 ├── P2 (状态视图与配置)     ← 依赖 P1 的 MuJoCoSimCore
 │
 ├── P3 (Studio 集成)        ← 依赖 P1 的 OrcaGymEuler 骨架
 │     │                     P2 可并行，P3 只需 P1
 │     │
 │     └── P4 (API 完备化)   ← 依赖 P2 的 DataView/SimConfig + P3 的 Env 骨架
 │           │
 │           └── P5 (迁移验证) ← 依赖 P4 完整 API
```

### 7.2 并行机会

| 并行组合 | 说明 |
|---------|------|
| P2 ∥ P3 | P2 实现 DataView/SimConfig，P3 实现 StudioBridge/Env 骨架，两者只依赖 P1 |
| P4 内部 | query_*/set_*/studio 方法可分人并行实现 |

### 7.3 关键里程碑

| 里程碑 | 阶段 | 标志 |
|--------|------|------|
| **M1 封装隔离验证** | P1 完成 | `_mjData` 访问被拦截 |
| **M2 API 完备验证** | P2 完成 | DataView/SimConfig 覆盖所有绕道场景 |
| **M3 端到端联调** | P3 完成 | `examples/euler/run_simple.py` 驱动 Studio 同步 |
| **M4 完整 API** | P4 完成 | 所有公共 API 可用 |
| **M5 典型模式 E2E** | P5 完成 | 4 个 `run_*.py` 入口脚本全部通过 |

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| `OrcaGymLocal` 部分方法逻辑复杂，迁移引入 bug | P4 延期 | 逐方法迁移 + 单元测试对照原实现 |
| `OrcaGymDataView` 字段覆盖不全 | 用户仍需绕道 | P2 测试遍历 `_mjData` 所有字段，确认覆盖 |
| `SimConfig` 遗漏 `opt` 字段 | 用户无法配置某些参数 | P2 测试遍历 `_mjModel.opt` 所有字段 |
| 端到端联调环境不稳定 | P3 延期 | 单元测试 mock stub，联调测试标记为可选 |
| 迁移后行为不一致 | P5 返工 | 数值对比测试，`np.allclose` 验证 |
| `__getattr__` 拦截影响性能 | 步进变慢 | `__getattr__` 只在属性不存在时触发，正常访问无开销 |

---

## 9. 测试策略

### 9.1 测试分层

| 层级 | 范围 | 工具 | 位置 |
|------|------|------|------|
| **单元测试** | 单个类/方法 | `unittest` + `unittest.mock` | `tests/orca_gym/{core,environment}/euler/` |
| **集成测试** | 多组件协作 | `unittest` | `tests/orca_gym/environment/euler/` |
| **端到端测试** | Env + Studio | `unittest`（需 Studio 环境） | `tests/orca_gym/environment/euler/` |
| **迁移验证** | 迁移后 Env 行为 | `unittest` + 数值对比 | `tests/orca_gym/environment/euler/` |

### 9.2 测试夹具

#### 共享 MJCF 测试模型

`tests/orca_gym/environment/euler/fixtures/simple_scene.xml`：

```xml
<!-- 简单场景：1 个自由关节 + 1 个铰链关节 + 1 个 site + 1 个传感器 -->
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <freejoint name="free_joint"/>
      <geom name="g1" type="box" size="0.1 0.1 0.1" mass="1"/>
      <site name="s1" pos="0 0 0.1" size="0.01"/>
      <body name="link2" pos="0 0 0.2">
        <joint name="hinge_joint" type="hinge" axis="0 1 0"/>
        <geom name="g2" type="sphere" size="0.05" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <framepos obj="link1" name="link1_pos"/>
  </sensor>
  <actuator>
    <motor joint="hinge_joint" name="hinge_motor" gear="1"/>
  </actuator>
</mujoco>
```

#### 测试基类

```python
# tests/orca_gym/environment/euler/fixtures/base.py
class OrcaGymEulerTestBase(unittest.TestCase):
    """OrcaGymEuler 测试基类，提供共享夹具。"""

    FIXTURE_XML = "simple_scene.xml"

    def setUp(self):
        xml_path = os.path.join(os.path.dirname(__file__), self.FIXTURE_XML)
        self.gym = OrcaGymEuler(stub=None)
        self.gym.init_simulation(xml_path)

    def tearDown(self):
        self.gym.close()
```

### 9.3 Studio 联调测试标记

Studio 联调测试需要真实 gRPC 环境，标记为可选：

```python
@unittest.skipUnless(
    os.environ.get("ORCA_STUDIO_ADDR"),
    "需 OrcaStudio 环境，设置 ORCA_STUDIO_ADDR 启用"
)
class TestE2EStudioIntegration(unittest.TestCase):
    ...
```

### 9.4 E2E 验证策略（OrcaPlayground example 驱动）

**核心原则**：E2E 验证不构造 `OrcaGymEulerEnv` 桩，而是采用外置的 OrcaPlayground example 来驱动，这是最终用户的典型用法，是真正的 E2E 验证。

| 验证方式 | 适用阶段 | 说明 |
|---------|---------|------|
| 单元测试（mock stub） | P1-P4 | 验证单个类/方法，无需 Studio |
| OrcaPlayground example 驱动 | P3, P5 | 真实用户用法，需 Studio 环境 |

P3 的 `examples/euler/run_simple.py` 和 P5 的 4 个 `run_*.py` 入口脚本构成完整的 E2E 验证矩阵，覆盖所有典型用户开发模式。

---

## 10. 文件清单

### 10.1 OrcaGym 源代码文件

| 阶段 | 文件路径 | 说明 |
|------|---------|------|
| P1 | `orca_gym/core/euler/__init__.py` | 模块初始化 |
| P1 | `orca_gym/core/euler/mujoco_sim_core.py` | MuJoCo 仿真核心 |
| P1 | `orca_gym/core/euler/orca_gym_euler.py` | 仿真核心 Facade |
| P2 | `orca_gym/core/euler/orca_gym_data_view.py` | 状态只读视图 |
| P2 | `orca_gym/core/euler/sim_config.py` | 求解器配置 |
| P2 | `orca_gym/core/euler/model_registry.py` | 模型注册 |
| P3 | `orca_gym/core/euler/orca_studio_bridge.py` | Studio 集成 |
| P3 | `orca_gym/environment/orca_gym_euler_env.py` | 环境 Facade |
| P4 | （完善上述文件） | API 完备化 |
| P5 | `orca_gym/environment/protocols.py` | 环境协议 |

### 10.2 OrcaGym 测试文件

| 阶段 | 文件路径 | 说明 |
|------|---------|------|
| P1 | `tests/orca_gym/core/euler/__init__.py` | 模块初始化 |
| P1 | `tests/orca_gym/core/euler/test_mujoco_sim_core.py` | SimCore 测试 |
| P1 | `tests/orca_gym/core/euler/test_orca_gym_euler.py` | Gym Facade 测试 |
| P2 | `tests/orca_gym/core/euler/test_orca_gym_data_view.py` | DataView 测试 |
| P2 | `tests/orca_gym/core/euler/test_sim_config.py` | SimConfig 测试 |
| P2 | `tests/orca_gym/core/euler/test_model_registry.py` | ModelRegistry 测试 |
| P3 | `tests/orca_gym/core/euler/test_orca_studio_bridge.py` | StudioBridge 测试 |
| P3 | `tests/orca_gym/environment/euler/__init__.py` | 模块初始化 |
| P3 | `tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py` | Env 骨架测试 |
| P3 | `tests/orca_gym/environment/euler/fixtures/simple_scene.xml` | 测试模型 |
| P3 | `tests/orca_gym/environment/euler/fixtures/base.py` | 测试基类 |
| P4 | `tests/orca_gym/environment/euler/test_query_methods.py` | 查询方法测试 |
| P4 | `tests/orca_gym/environment/euler/test_set_methods.py` | 设置方法测试 |
| P4 | `tests/orca_gym/environment/euler/test_studio_methods.py` | Studio 方法测试 |
| P4 | `tests/orca_gym/environment/euler/test_other_methods.py` | 其他方法测试 |
| P5 | `tests/orca_gym/environment/euler/test_protocol_compliance.py` | 协议合规测试 |
| P5 | `tests/orca_gym/environment/euler/test_euler_envs_api.py` | Euler Env API 测试 |

### 10.3 OrcaPlayground E2E 验证文件

| 阶段 | 文件路径 | 说明 |
|------|---------|------|
| P3 | `OrcaPlayground/envs/euler/__init__.py` | Env 子类模块初始化 |
| P3 | `OrcaPlayground/envs/euler/simple_env.py` | 简单委托式 Env（对应 D12Env） |
| P3 | `OrcaPlayground/examples/euler/__init__.py` | example 模块初始化 |
| P3 | `OrcaPlayground/examples/euler/run_simple.py` | 简单委托式入口 |
| P5 | `OrcaPlayground/envs/euler/loop_env.py` | 手动循环式 Env（对应 G1Env） |
| P5 | `OrcaPlayground/envs/euler/force_env.py` | 力应用式 Env（对应 fluid SimEnv） |
| P5 | `OrcaPlayground/envs/euler/multi_agent_env.py` | 多 Agent 复杂式 Env（对应 LeggedSimEnv） |
| P5 | `OrcaPlayground/examples/euler/run_loop.py` | 手动循环式入口 |
| P5 | `OrcaPlayground/examples/euler/run_force.py` | 力应用入口 |
| P5 | `OrcaPlayground/examples/euler/run_multi_agent.py` | 多 Agent 入口 |

---

## 11. 总结

本开发设计文档将 `OrcaGymEulerEnv` + `OrcaGymEuler` 的开发分解为 5 个阶段：

| 阶段 | 核心交付 | 联调能力 | 关键里程碑 |
|------|---------|---------|-----------|
| **P1** | 基础设施骨架 + 封装隔离 | 无 | M1 封装隔离验证 |
| **P2** | DataView + SimConfig + ModelRegistry | 无 | M2 API 完备验证 |
| **P3** | StudioBridge + Env 骨架 + `run_simple.py` | **可联调** | M3 端到端联调 |
| **P4** | 完整公共 API | 可联调 | M4 完整 API |
| **P5** | 4 个典型模式 E2E + Protocol | 可联调 | M5 典型模式 E2E |

**关键设计**：
- P1 完成后即可验证封装隔离机制（`__getattr__`/`__dir__`）
- P2 与 P3 可并行，P3 完成后即可通过 `examples/euler/run_simple.py` 开始端到端联调
- P4 按 API 类别分批实现，每批配套单元测试
- P5 通过 OrcaPlayground 的 4 个典型用户模式 example 驱动真正的 E2E 验证，引入 Protocol 平滑过渡

**E2E 验证原则**：
- 不构造 `OrcaGymEulerEnv` 桩，采用外置 OrcaPlayground example 驱动
- 每个 example 对应一种典型用户开发模式，是最终用户的真实用法
- P3 的 `run_simple.py` 提供最早联调入口，P5 扩展到 4 个模式全覆盖

每个阶段都有独立的验收标准和单元测试，确保渐进式交付和持续验证。
