# OrcaGym Euler 阶段三开发指导文档：剩余实现对齐与离线单元测试

## 1. 文档定位

### 1.1 文档目标

本文是 `OrcaGymEulerEnv` + `OrcaGymEuler` **阶段三（剩余实现对齐 + 离线单元测试）** 的开发指导文档。在阶段一（骨架）、阶段二（最小功能填充，支持 Lesson 1–3 端到端运行）已完成并通过验收的基础上，**分阶段补齐剩余的 MuJoCo 能力，对齐 `OrcaGymLocalEnv` 的公共 API**，并通过**离线加载真实 G1 XML** 的单元测试验证功能正确性。

> **上游约束**：架构文档 `docs/design/architecture/orca_gym_euler_architecture.md`（§5–§7、§10–§12 为硬性约束）。本文所有填充实现必须严格遵守 K1–K12 约束，不得回退到上帝类 + 封装泄漏的老路。

### 1.2 阶段三范围

阶段三**不涉及** Euler 非刚体求解器耦合（`EulerOrchestrator` 占位，后续单独设计），也**不涉及在线端到端验证**（归阶段四），聚焦于：

1. **MuJoCo 能力对齐**：将 `OrcaGymLocalEnv` 暴露的 MuJoCo 操作能力（查询、设置、力应用、约束、雅可比、Studio 交互）填充到 `OrcaGymEulerEnv`，使老代码可零绕道迁移。
2. **废弃用法剔除**：架构文档明确废弃的用法（见 §2.2）不在阶段三实现，迁移时改用新接口。
3. **离线单元测试**：通过离线加载 `g1_29dof_camera.xml` 获取真实 MuJoCo 数据，对每个新增方法编写 CPU 可跑的单元测试，验证功能正确性（维度/数值/一致性）。在线端到端验证见阶段四文档 `orca_gym_euler_phase4_online_validation_development.md`。

### 1.3 阶段三与阶段二的边界

| 维度 | 阶段二（已完成） | 阶段三（本文） |
|------|----------------|--------------|
| 仿真核心 | `init/step/forward/set_ctrl/set_qpos_qvel/reset_data/sync_to_view` | `apply_body_force`/`clear_*`/`mj_jac*`/`mj_apply_force_at_site` |
| 状态视图 | 5 基本字段 + body/site 按需查询（7 方法） | 完整 body/site/geom/sensor/contact 查询 + 批量接口 |
| 模型注册 | `build_orca_gym_model` | `body_subtree_mass`/`equality_*` 扩展查询 |
| 求解器配置 | `timestep/integrator/iterations/gravity` | 其余 `opt` 字段（按需） |
| 状态查询 | 无（三个 example 走 `env.data` 直读） | 全部 `query_*`/`get_body_*`/`jnt_*` 方法 |
| 状态设置 | `set_joint_qpos/qvel` | `set_mocap_pos_and_quat`/`set_geom_friction`/`update_equality_constraints` |
| Studio 交互 | `render`/`load_model_xml`/`pause`/`get_body_manipulation_*`（占位） | `anchor_actor`/`release_body_anchored`/视频/帧/内容文件 |
| 单元测试 | Lesson 1–3（离线为主） | 离线加载 G1 XML 的真实数据单元测试（Lesson 4–8 在线验证归阶段四） |

---

## 2. 范围与废弃清单

### 2.1 需要对齐的 API 清单（来自 OrcaGymLocalEnv）

通过对比 `OrcaGymLocalEnv` 公共方法与 `OrcaGymEulerEnv` 现状，梳理阶段三需填充的 API：

#### 2.1.1 状态查询（Query）

| API | OrcaGymLocalEnv 实现 | 阶段三归属 |
|-----|---------------------|-----------|
| `query_joint_qpos(qvel/qacc)(joint_names)` | 读 `_mjData.qpos/qvel/qacc` 切片 | 3.1 |
| `query_joint_offsets/lengths/dofadrs(joint_names)` | 读 `_mjModel.jnt_qposadr/dofadr` | 3.1 |
| `jnt_qposadr/jnt_dofadr(joint_name)` | 读 `_mjModel.jnt_qposadr/dofadr` | 3.1 |
| `get_body_xpos_xmat_xquat(body_name_list)` | 读 `_mjData.xpos/xmat/xquat` | 3.1 |
| `get_body_xpos_xmat_xquat_xvel(body_name_list)` | `mj_jacBody @ qvel` | 3.1/3.3 |
| `query_site_pos_and_mat/quat/size(site_names)` | 读 `_mjData.site_xpos/xmat` + `_mjModel.site_size` | 3.1 |
| `query_site_xvalp_xvalr(site_names)` | `mj_jacSite @ qvel` | 3.1/3.3 |
| `query_site_pos_and_quat_B/_xvalp_xvalr_B(...)` | site 查询 + 基座坐标系变换 | 3.1 |
| `query_sensor_data(sensor_names)` | 读 `_mjData.sensordata` 切片 | 3.1 |
| `query_actuator_torques(actuator_names)` | 读 `_mjData.actuator_force` 切片 | 3.1 |
| `query_contact_simple()` | 遍历 `_mjData.contact` | 3.1 |
| `query_contact_force(contact_ids)` | `mj_contactForce` | 3.1 |
| `get_cfrc_ext()` | 读 `_mjData.cfrc_ext` | 3.1 |
| `query_velocity/position/orientation_body_B(...)` | body 查询 + 基座变换 | 3.1 |
| `query_joint_axes_B(joint_names, base_body)` | 读 `_mjModel.jnt_axis` + 基座变换 | 3.1 |
| `query_robot_velocity/position/orientation_odom(...)` | body 位姿 + 累积里程计 | 3.1 |
| `get_goal_bounding_box(geom_name)` | 读 `_mjModel.geom_size` | 3.1 |

#### 2.1.2 状态设置与力应用（Set / Force）

| API | OrcaGymLocalEnv 实现 | 阶段三归属 |
|-----|---------------------|-----------|
| `apply_body_force(body_name, force, torque)` | 写 `_mjData.xfrc_applied`（老版穿墙） | 3.2 |
| `clear_body_force(body_name)` / `clear_all_forces()` | 清零 `xfrc_applied` | 3.2 |
| `mj_apply_force_at_site(site_name, force, torque)` | `mj_applyForce` 等价 | 3.2 |
| `mj_clear_xfrc_applied_for_site(site_name)` | 清 site 关联 body 的 xfrc | 3.2 |
| `set_mocap_pos_and_quat(mocap_dict)` | 写 `_mjData.mocap_pos/quat` + 远端同步 | 3.2 |
| `set_geom_friction(geom_friction_dict)` | 写 `_mjModel.geom_friction` | 3.2 |
| `add_extra_weight(weight_load_dict)` | 修改 `_mjModel.body_mass/inertia` | 3.2 |

#### 2.1.3 约束操作（Equality）

| API | OrcaGymLocalEnv 实现 | 阶段三归属 |
|-----|---------------------|-----------|
| `update_equality_constraints(eq_list)` | 写 `_mjModel.eq_*` | 3.5 |
| `modify_equality_objects(...)` | 改 `eq_obj1id/eq_obj2id` | 3.5 |
| `update_anchor_equality_constraints(actor_name, anchor_type)` | 锚点约束更新 | 3.5 |

#### 2.1.4 雅可比与高级 MuJoCo（Jacobian）

| API | OrcaGymLocalEnv 实现 | 阶段三归属 |
|-----|---------------------|-----------|
| `mj_jacBody(jacp, jacr, body_id)` | `mujoco.mj_jacBody` | 3.3 |
| `mj_jacSite(jacp, jacr, site_name)` | `mujoco.mj_jacSite` | 3.3 |
| `mj_jac_site(site_names)`（批量） | 循环 `mj_jacSite` | 3.3 |

#### 2.1.5 Studio 在线交互（Online）

| API | OrcaGymLocalEnv 实现 | 阶段三归属 |
|-----|---------------------|-----------|
| `do_body_manipulation()`（完整实现） | 锚定 + mocap + 等式约束 | 3.4 |
| `anchor_actor(actor_name, anchor_type)` | mocap + equality 联动 | 3.4/3.5 |
| `release_body_anchored()` | 清锚点约束 | 3.4/3.5 |
| `begin_save_video(file_path, capture_mode)` | gRPC `BeginSaveVideo` | 3.4 |
| `stop_save_video()` | gRPC `StopSaveVideo` | 3.4 |
| `get_current_frame()` / `get_next_frame()` | gRPC `GetCurrentFrame` | 3.4 |
| `get_camera_time_stamp(last_frame)` | gRPC `GetCameraTimeStamp` | 3.4 |
| `get_frame_png(image_path)` | gRPC `GetFramePng` | 3.4 |

### 2.2 废弃清单（不在阶段三实现）

架构文档（§10.2、§11.2、§12.5）明确废弃的用法，阶段三**不实现**，迁移时改用新接口：

| 废弃项 | 原因 | 替代方案 |
|--------|------|---------|
| `MuJoCoAdapter`（受控 MuJoCo 句柄适配器） | §11.2 决定不再支持 robosuite，无"逃生舱" | 扩展 `OrcaGymEulerEnv` 公共方法 |
| `env._mjData` / `env._mjModel` 直接访问 | §5.3/P2 封装隔离 | `env.data`（DataView）/ `env.model` / `env.query_*()` |
| `gym._mjData.xfrc_applied` 直接写 | §5.7/P4 力应用可追踪 | `env.apply_body_force()` |
| `gym._mjModel.opt.*` 直接访问 | §5.6/SimConfig | `env.sim_config.*` |
| `update_data()` 公共同步方法 | 老体系 gRPC 双轨制，新体系本地零拷贝 | 内部 `sync_to_view()`，公共 API 无需手动同步 |
| `load_initial_frame()` gRPC 方法 | 老体系从远端加载初始帧 | 本地 `reset_data()`（`mj_resetData`） |
| `update_data_external(...)` 远端数据注入 | 老体系 Remote 模式专用 | 新体系本地仿真，无远端注入 |
| `query_opt_config()` / `set_opt_config()` gRPC 方法 | 老体系远端 opt 读写 | `env.sim_config`（本地 `mjModel.opt`） |
| `query_model_info()` / `query_all_*()` gRPC 方法 | 老体系远端模型查询 | `ModelRegistry.build_orca_gym_model()`（本地构建） |
| `set_actuator_trnid(...)` 运行时改执行器传动 | 极少使用，且破坏模型一致性 | 不实现，需在 XML 定义 |
| `mj_inverse()` / `mj_fullM()` | 极少使用（逆动力学/质量矩阵） | 按需在 `MuJoCoSimCore` 扩展，不在阶段三主路径 |

> **迁移原则**：老代码中出现的废弃用法，迁移时一律改用新接口。若新接口缺失，**扩展 `OrcaGymEulerEnv` 公共方法**，不回退到穿墙访问。

---

## 3. 现状与差距分析

### 3.1 组件现状（阶段二交付）

| 组件 | 已实现 | 待填充（阶段三） |
|------|--------|----------------|
| `MuJoCoSimCore` | `init/step/forward/set_ctrl/set_qpos_qvel/reset_data/sync_to_view` | `apply_body_force`/`clear_*`/`mj_jac*`/`mj_apply_force_at_site`/查询方法 |
| `OrcaGymDataView` | 5 基本字段 + `xfrc_applied/actuator_force/contact` + 7 个 body/site 查询 | geom 查询、批量接口、`cfrc_ext`/`cvel` 等扩展字段 |
| `ModelRegistry` | `build_orca_gym_model` | `body_subtree_mass`/`equality_*` 扩展查询 |
| `SimConfig` | `timestep/integrator/iterations/gravity` | 按需扩展其余 `opt` 字段 |
| `OrcaStudioBridge` | `render/load_model_xml/pause/configure_offline/set_timestep_remote/get_body_manipulation_*` | 视频/帧/内容文件方法 |
| `OrcaGymEuler` | 委托步进/同步/Studio 基础 | 委托查询/设置/力应用/雅可比/约束 |
| `OrcaGymEulerEnv` | 生命周期/步进/`set_joint_qpos/qvel`/render(占位 body manipulation) | 全部 `query_*`/`set_*`/`apply_*`/约束/Studio 完整 |

### 3.2 委托链路设计

阶段三所有新方法遵循统一委托链路，保持 K1–K12 约束：

```
OrcaGymEulerEnv（公共 API）
    │  仅通过 self._gym 公共方法委托（K4）
    ▼
OrcaGymEuler（Facade 委托）
    │  通过 object.__getattribute__ 内部访问子组件（K3/K5）
    ▼
MuJoCoSimCore / ModelRegistry / OrcaStudioBridge（实现层）
    │  MuJoCoSimCore 持有 _mjModel/_mjData，所有 MuJoCo 原生操作集中于此（P2）
    ▼
mujoco.MjModel / mujoco.MjData（引擎内部，L3）
```

**关键约束**：
- `OrcaGymEulerEnv` 只调 `self._gym.<公共方法>`，不触 `_gym._sim/_studio`（K4）
- `OrcaGymEuler` 内部用 `object.__getattribute__(self, "_sim")` 绕过自身拦截（K3/K5）
- 查询方法返回 typed 对象（np.ndarray / dict / tuple），不返回 `MjData`/`MjModel`（K11）

### 3.3 子步骤分层与 K 约束映射

阶段三将每个子阶段拆分为**独立的子步骤**，每个子步骤聚焦单一组件层，配备专属的架构遵从性测试。各层对应的 K 约束如下：

| 子步骤所属层 | 核心 K 约束 | 架构遵从性测试重点 |
|-------------|-----------|-------------------|
| `MuJoCoSimCore`（实现层） | P2、K11 | 原生操作集中于此；返回 typed 对象，不泄漏 `MjData`/`MjModel` |
| `OrcaGymDataView`（视图层） | K6 | `env.data` 为 DataView；新增字段是零拷贝视图非拷贝 |
| `ModelRegistry`（模型层） | K11 | 返回 typed 对象；不暴露 `_mj_model` |
| `OrcaGymEuler`（Facade 层） | K3、K5、K11 | 委托用 `object.__getattribute__`；`__dir__` 不泄漏子组件；不新增 property |
| `OrcaGymEulerEnv`（公共 API 层） | K1、K2、K4、K11、K12 | 用 `self._gym` 委托；不触私有；`__dir__` 合规；docstring 完整 |
| `OrcaStudioBridge`（Studio 层） | K9 | Studio 交互走 `_studio_bridge`，不走 `gym.studio` |

---

## 4. 总体策略

### 4.1 自底向上 + 子步骤独立验收

延续阶段二策略，按依赖关系与职责内聚分 5 个子阶段，每个子阶段进一步拆分为**独立可验收的子步骤**。每个子步骤完成源码填充后，**立即编写该子步骤专属的架构遵从性测试 + 功能单元测试**，通过后方可进入下一子步骤：

```
阶段 3.1（状态查询 API）—— 8 个子步骤
  3.1.1  MuJoCoSimCore    ← 关节查询方法（query_joint_qpos/qvel/qacc/offsets/lengths/dofadrs, jnt_qposadr/dofadr）
  3.1.2  MuJoCoSimCore    ← Body/Site 查询方法（query_body_xpos_xmat_xquat/xvel, query_site_pos_and_mat/size）
  3.1.3  MuJoCoSimCore    ← 传感器/执行器/接触/Geom 查询方法（query_sensor_data/actuator_torques/contact_simple/force, get_cfrc_ext, get_goal_bounding_box）
  3.1.4  OrcaGymDataView  ← 扩展查询字段（cfrc_ext 字段, geom_xpos/xmat/size）
  3.1.5  ModelRegistry    ← 扩展查询（body_subtree_mass, equality_data_width, equality_object_ids）
  3.1.6  OrcaGymEuler     ← 查询委托链路（全部 query_* 委托 + sensor_info 拼装）
  3.1.7  OrcaGymEulerEnv  ← 公共查询 API（委托 _gym）
  3.1.8  OrcaGymEulerEnv  ← 基座坐标系变换方法（纯 NumPy：query_*_B / *_odom / query_joint_axes_B）

阶段 3.2（状态设置与外力应用）—— 5 个子步骤
  3.2.1  MuJoCoSimCore    ← 力应用方法（apply_body_force/clear_body_force/clear_all_forces/mj_apply_force_at_site/mj_clear_xfrc_applied_for_site）
  3.2.2  MuJoCoSimCore    ← 状态设置方法（set_mocap_pos_and_quat/set_geom_friction/add_extra_weight）
  3.2.3  OrcaStudioBridge ← mocap 远端同步（set_mocap_pos_and_quat async gRPC）
  3.2.4  OrcaGymEuler/Env ← 力应用与设置委托（Env 层 body_name→body_id 解析 + 委托）
  3.2.5  OrcaGymDataView  ← xfrc_applied 只读保护验证（DataView 返回只读视图）

阶段 3.3（雅可比与高级 MuJoCo）—— 3 个子步骤
  3.3.1  MuJoCoSimCore    ← mj_jacBody/mj_jacSite（单点雅可比）
  3.3.2  MuJoCoSimCore    ← mj_jac_site 批量（site_names → {jacp, jacr}）
  3.3.3  OrcaGymEuler/Env ← 雅可比委托（Env 层 site_name→site_id 解析 + 委托）

阶段 3.4（Studio 在线交互完整实现）—— 4 个子步骤
  3.4.1  OrcaStudioBridge ← 视频录制方法（begin_save_video/stop_save_video）
  3.4.2  OrcaStudioBridge ← 帧捕获方法（get_current_frame/get_next_frame/get_camera_time_stamp/get_frame_png）
  3.4.3  OrcaStudioBridge ← 内容文件方法（load_content_file）
  3.4.4  OrcaGymEuler/Env ← Studio 委托 + do_body_manipulation 完整实现（依赖 3.5 约束方法）

阶段 3.5（等式约束与完整体操作）—— 6 个子步骤
  3.5.1  MuJoCoSimCore    ← 等式约束方法（update_equality_constraints/modify_equality_objects）
  3.5.2  ModelRegistry    ← equality 查询（equality_data_width/equality_object_ids——若 3.1.5 未覆盖则在此补齐）
  3.5.3  OrcaGymEuler/Env ← 约束委托（update_equality_constraints/modify_equality_objects/update_anchor_equality_constraints）
  3.5.4  OrcaGymEulerEnv  ← anchor_actor 实现（mocap + equality 联动，走合规 API）
  3.5.5  OrcaGymEulerEnv  ← release_body_anchored 实现（清锚点约束 + mocap 复位）
  3.5.6  OrcaGymEulerEnv  ← do_body_manipulation 完整实现（锚定 + mocap 移动 + 释放编排）
```

### 4.2 每个子步骤交付物

每个子步骤必须独立交付以下 4 项，**全部通过后方可进入下一子步骤**：

1. **源码填充**：将 `raise NotImplementedError` 替换为真实实现（仅涉及该子步骤声明的文件/方法）
2. **架构遵从性测试**（该子步骤专属）：针对该子步骤涉及的组件层，编写 K 约束断言测试（grep 断言 + 运行时类型断言 + `__dir__` 合规断言），**独立于功能测试**
3. **功能单元测试**（该子步骤专属）：CPU 可跑的单元测试，离线加载 `g1_29dof_camera.xml` 获取真实 MuJoCo 数据，验证功能正确性（维度/数值/一致性）
4. **子步骤验收**：逐条勾选验收清单（源码 + 架构测试 + 功能测试 + K 约束保持）

### 4.3 测试环境

| 测试类型 | 环境 | 说明 |
|---------|------|------|
| 离线单元测试（CPU） | sandbox 内 `OrcaFlow_Flow` 解释器 | 纯 MuJoCo 仿真，离线加载 G1 XML，无需 OrcaStudio |
| 在线端到端 Example | 宿主机 + OrcaStudio | 归阶段四，见 `orca_gym_euler_phase4_online_validation_development.md` |

> **离线单元测试数据来源**：测试通过 `mujoco.MjModel.from_xml_path('OrcaPlayground/envs/euler/robots/g1_29dof_camera.xml')` 加载真实 G1 模型，获取真实的关节/body/site/sensor 数据，而非 mock 数据。这样单元测试能验证真实模型下的功能正确性（维度、数值范围、一致性），是阶段三的核心验证手段。

### 4.4 架构遵从性测试通用规范

每个子步骤的架构遵从性测试遵循统一规范，按组件层选择对应断言组合：

**A. grep 断言（源码静态检查）**

```bash
# K4: Env 源码中不应出现 self._gym._sim / self._gym._studio / self._gym._mjData 等
grep -rn "self\._gym\._\(sim\|studio\|registry\|opt\|view\|euler\|mjData\|mjModel\)" \
    orca_gym/environment/euler/orca_gym_euler_env.py
# 期望：无匹配（exit code 1）

# K3: Gym 委托方法应用 object.__getattribute__ 访问子组件
grep -n "object\.__getattribute__" orca_gym/core/euler/orca_gym_euler.py
# 期望：新增委托方法均有匹配

# K5: Gym 不新增 _sim/_studio property
grep -n "@property" orca_gym/core/euler/orca_gym_euler.py | grep "_\(sim\|studio\|registry\)"
# 期望：无匹配
```

**B. 运行时类型断言（动态检查）**

```python
# K11: 公共方法返回 typed 对象，不返回 MjData/MjModel
import mujoco
result = env.query_joint_qpos(["left_hip_pitch"])
assert isinstance(result, dict)
assert isinstance(result["left_hip_pitch"], np.ndarray)
assert not isinstance(result, (mujoco.MjData, mujoco.MjModel))

# K6: env.data 类型为 OrcaGymDataView
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
assert isinstance(env.data, OrcaGymDataView)
```

**C. `__dir__` 合规断言**

```python
# K2/K3: __dir__ 不泄漏内部对象
env_dir = set(dir(env))
for forbidden in ["_sim", "_studio", "_registry", "_mjData", "_mjModel"]:
    assert forbidden not in env_dir, f"env.__dir__ 泄漏 {forbidden}"

gym_dir = set(dir(env._gym))  # 内部测试可访问
for forbidden in ["_sim", "_studio", "_mjData", "_mjModel"]:
    assert forbidden not in gym_dir, f"gym.__dir__ 泄漏 {forbidden}"
```

---

## 5. 阶段 3.1：状态查询 API

### 5.1 目标

填充全部 `query_*` / `get_body_*` / `jnt_*` 状态查询方法，使用户可零绕道读取 MuJoCo 状态，替代老体系的 `gym.query_*` gRPC 调用。新体系全部从本地 `_mjData`/`_mjModel` 读取（`MuJoCoSimCore` 持有），不经 gRPC。

本阶段拆分为 **8 个独立子步骤**（3.1.1–3.1.8），按"实现层 → 视图层 → 模型层 → Facade 层 → 公共 API 层"自底向上推进，每个子步骤独立验收。

---

### 5.2 子步骤 3.1.1：MuJoCoSimCore 关节查询方法

**涉及文件**：`orca_gym/core/euler/mujoco_sim_core.py`

**实现内容**：新增关节查询方法（全部从 `_mjData`/`_mjModel` 直接读取，返回 typed 对象）：

```python
class MuJoCoSimCore:
    # --- 关节查询 ---
    def query_joint_qpos(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qpos（按关节类型切片）。"""
    def query_joint_qvel(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qvel。"""
    def query_joint_qacc(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qacc。"""
    def query_joint_offsets(self, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """查询关节 qpos/qvel/qacc 偏移量。"""
    def query_joint_lengths(self, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """查询关节 qpos/qvel/qacc 长度。"""
    def query_joint_dofadrs(self, joint_names: list[str]) -> dict[str, int]:
        """查询关节 dof 起始地址。"""
    def jnt_qposadr(self, joint_name: str) -> int:
        """查询关节 qpos 起始地址。"""
    def jnt_dofadr(self, joint_name: str) -> int:
        """查询关节 qvel 起始地址。"""
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_simcore_joint_query_returns_ndarray` | `query_joint_qpos/qvel/qacc` 返回 `dict[str, np.ndarray]`，不返回 `MjData`/`MjModel` | K11 |
| `test_simcore_joint_query_no_mjdata_leak` | grep 断言 `mujoco_sim_core.py` 关节查询方法不 `return self._mjData` 或 `self._mjModel` | P2/K11 |
| `test_simcore_jnt_qposadr_returns_int` | `jnt_qposadr`/`jnt_dofadr` 返回 `int`（非 numpy 标量泄漏） | K11 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_query_joint_qpos_returns_correct_slice` | `query_joint_qpos(["left_hip_pitch"])` 返回值与 `_mjData.qpos[adr:adr+len]` 一致 |
| `test_query_joint_qvel_matches_dof_slice` | qvel 切片正确 |
| `test_query_joint_offsets_lengths_consistent` | offsets + lengths 与 jnt_qposadr/dofadr 一致 |
| `test_jnt_qposadr_returns_correct_adr` | `jnt_qposadr("left_hip_pitch")` 与 `mjModel.jnt_qposadr` 一致 |

**子步骤验收**：
- [x] 8 个关节查询方法实现完成，`raise NotImplementedError` 已替换
- [x] 架构遵从性测试通过（K11 typed 返回 + P2 不泄漏 MjData）
- [x] 功能单元测试通过（加载 G1 XML 验证切片数值正确）
- [x] `mujoco_sim_core.py` 无 `return self._mjData` / `return self._mjModel`

---

### 5.3 子步骤 3.1.2：MuJoCoSimCore Body/Site 查询方法

**涉及文件**：`orca_gym/core/euler/mujoco_sim_core.py`

**实现内容**：

```python
class MuJoCoSimCore:
    # --- Body 查询 ---
    def query_body_xpos_xmat_xquat(self, body_name_list: list[str]) -> dict[str, dict]:
        """查询 body 位姿（xpos/xmat/xquat）。"""
    def query_body_xpos_xmat_xquat_xvel(self, body_name_list: list[str]) -> dict[str, dict]:
        """查询 body 位姿 + 世界系线速度（mj_jacBody @ qvel）。"""

    # --- Site 查询 ---
    def query_site_pos_and_mat(self, site_names: list[str]) -> dict[str, dict]:
        """查询 site xpos/xmat。"""
    def query_site_size(self, site_names: list[str]) -> dict[str, np.ndarray]:
        """查询 site 尺寸。"""
```

> **注意**：`query_body_xpos_xmat_xquat_xvel` 依赖 `mj_jacBody`（3.3 实现）。本子步骤先实现接口骨架，内部调用 `mj_jacBody`，若 3.3 未完成则临时 `raise NotImplementedError`，3.3 完成后移除。

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_simcore_body_query_returns_dict_of_dict` | 返回 `dict[str, dict]`，内层 dict 含 `xpos`/`xmat`/`xquat` 键，值为 `np.ndarray` | K11 |
| `test_simcore_site_query_returns_dict` | `query_site_pos_and_mat`/`query_site_size` 返回 typed dict | K11 |
| `test_simcore_body_query_no_mjdata_leak` | grep 断言 body/site 查询方法不直接返回 `MjData`/`MjModel` | P2/K11 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_query_body_xpos_shape` | `xpos` 形状 `(3,)`，`xmat` 形状 `(3,3)`，`xquat` 形状 `(4,)` |
| `test_query_body_xpos_matches_mjdata` | 返回值与 `_mjData.xpos[body_id]` 一致 |
| `test_query_site_pos_and_mat_matches_mjdata` | 与 `_mjData.site_xpos`/`site_xmat` 一致 |
| `test_query_site_size_matches_model` | 与 `_mjModel.site_size` 一致 |

**子步骤验收**：
- [x] 4 个 Body/Site 查询方法实现完成
- [x] 架构遵从性测试通过（K11 + P2）
- [x] 功能单元测试通过（数值与 `_mjData`/`_mjModel` 一致）
- [x] `query_body_xpos_xmat_xquat_xvel` 的 `mj_jacBody` 依赖标注（3.3 解除）

---

### 5.4 子步骤 3.1.3：MuJoCoSimCore 传感器/执行器/接触/Geom 查询方法

**涉及文件**：`orca_gym/core/euler/mujoco_sim_core.py`

**实现内容**：

```python
class MuJoCoSimCore:
    # --- 传感器 / 执行器 / 接触查询 ---
    def query_sensor_data(self, sensor_names: list[str], sensor_info: dict) -> dict[str, np.ndarray]:
        """查询传感器数据（按 adr/dim 切片 sensordata）。"""
    def query_actuator_torques(self, actuator_names: list[str]) -> dict[str, np.ndarray]:
        """查询执行器力矩（actuator_force 切片）。"""
    def query_contact_simple(self) -> list[dict]:
        """查询简单接触信息（遍历 contact 列表）。"""
    def query_contact_force(self, contact_ids: list[int]) -> dict[int, np.ndarray]:
        """查询接触力（mj_contactForce）。"""
    def get_cfrc_ext(self) -> np.ndarray:
        """查询外部约束力（cfrc_ext）。"""

    # --- Geom 查询 ---
    def get_goal_bounding_box(self, geom_name: str) -> np.ndarray:
        """查询 geom 尺寸（bounding box）。"""
```

**关键设计决策**：`query_sensor_data` 需 `sensor_info` 参数（adr/dim 来自 `OrcaGymModel`，由 `OrcaGymEuler` 委托时传入）。`query_contact_simple` 返回结构化 dict 列表（`geom1`/`geom2`/`dist`/`pos`/`frame`）。

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_simcore_sensor_query_returns_dict_ndarray` | 返回 `dict[str, np.ndarray]` | K11 |
| `test_simcore_contact_simple_returns_list_of_dict` | 返回 `list[dict]`，dict 含 `geom1`/`geom2`/`dist` 等键 | K11 |
| `test_simcore_contact_force_returns_dict_int_ndarray` | 返回 `dict[int, np.ndarray]` | K11 |
| `test_simcore_get_cfrc_ext_returns_ndarray` | 返回 `np.ndarray`，非 `MjData` | K11/P2 |
| `test_simcore_geom_query_no_mjmodel_leak` | grep 断言 `get_goal_bounding_box` 不返回 `self._mjModel` | P2/K11 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_query_sensor_data_matches_sensordata` | 传感器数据与 `_mjData.sensordata` 切片一致 |
| `test_query_actuator_torques_matches_force` | 与 `_mjData.actuator_force` 切片一致 |
| `test_query_contact_simple_returns_list` | 接触列表结构正确（ncon 个元素） |
| `test_query_contact_force_via_mj_contactForce` | 接触力与 `mujoco.mj_contactForce` 一致 |
| `test_get_cfrc_ext_shape` | 形状 `(nbody, 6)` |
| `test_get_goal_bounding_box_matches_geom_size` | 与 `_mjModel.geom_size` 一致 |

**子步骤验收**：
- [x] 6 个传感器/执行器/接触/Geom 查询方法实现完成
- [x] 架构遵从性测试通过（K11 typed 返回 + P2 不泄漏）
- [x] 功能单元测试通过（数值一致）
- [x] `query_sensor_data` 接受 `sensor_info` 参数，SimCore 不持有 `OrcaGymModel`

---

### 5.5 子步骤 3.1.4：OrcaGymDataView 扩展查询字段

**涉及文件**：`orca_gym/core/euler/orca_gym_data_view.py`

**实现内容**：

```python
class OrcaGymDataView:
    # 新增扩展字段（零拷贝视图）
    cfrc_ext: np.ndarray       # 外部约束力
    # cvel 已有 body_cvel() 方法，无需新增字段

    # 新增 geom 查询
    def geom_xpos(self, geom_name: str) -> np.ndarray: ...
    def geom_xmat(self, geom_name: str) -> np.ndarray: ...
    def geom_size(self, geom_name: str) -> np.ndarray: ...
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_dataview_cfrc_ext_is_view_not_copy` | `cfrc_ext.base` 非None（零拷贝视图），修改 `_mjData.cfrc_ext` 后 DataView 同步 | K6 |
| `test_dataview_geom_query_returns_ndarray` | `geom_xpos`/`geom_xmat`/`geom_size` 返回 `np.ndarray` | K6/K11 |
| `test_dataview_no_mjdata_leak` | grep 断言 DataView 不 `return self._mjData` | K6/P2 |
| `test_env_data_is_dataview` | `isinstance(env.data, OrcaGymDataView)` | K6 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_cfrc_ext_matches_mjdata` | `env.data.cfrc_ext` 与 `_mjData.cfrc_ext` 数值一致 |
| `test_geom_xpos_matches_mjdata` | 与 `_mjData.geom_xpos` 一致 |
| `test_geom_size_matches_model` | 与 `_mjModel.geom_size` 一致 |

**子步骤验收**：
- [x] `cfrc_ext` 字段 + 3 个 geom 查询方法实现完成
- [x] 架构遵从性测试通过（K6 DataView 零拷贝视图 + 不泄漏 MjData）
- [x] 功能单元测试通过（数值一致）
- [x] `cfrc_ext` 为零拷贝视图（`base` 非None）

---

### 5.6 子步骤 3.1.5：ModelRegistry 扩展查询

**涉及文件**：`orca_gym/core/euler/model_registry.py`

**实现内容**：将 `raise NotImplementedError` 替换为真实实现：

```python
class ModelRegistry:
    def body_subtree_mass(self, body_name: str) -> float:
        """查询 body 子树总质量（读 _mj_model.body_subtreemass）。"""
    def equality_data_width(self) -> int:
        """查询等式约束数据宽度（_mj_model.eq_data.shape[1]）。"""
    def equality_object_ids(self, eq_idx: int) -> tuple[int, int]:
        """查询等式约束关联的两个对象 id。"""
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_registry_body_subtree_mass_returns_float` | 返回 `float`（Python 标量），非 numpy 泄漏 | K11 |
| `test_registry_equality_returns_typed` | `equality_data_width` 返回 `int`，`equality_object_ids` 返回 `tuple[int, int]` | K11 |
| `test_registry_no_mjmodel_leak` | grep 断言方法不 `return self._mj_model` | P2/K11 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_body_subtree_mass_positive` | `body_subtree_mass("pelvis")` 返回正标量 |
| `test_body_subtree_mass_matches_mujoco` | 与 `_mjModel.body_subtreemass[body_id]` 一致 |
| `test_equality_data_width_matches_model` | 与 `_mjModel.eq_data.shape[1]` 一致 |
| `test_equality_object_ids_matches_model` | 与 `eq_obj1id`/`eq_obj2id` 一致 |

**子步骤验收**：
- [x] 3 个 ModelRegistry 扩展查询方法实现完成
- [x] 架构遵从性测试通过（K11 typed 返回 + P2 不泄漏 `_mj_model`）
- [x] 功能单元测试通过（数值与 MuJoCo 一致）
- [x] `raise NotImplementedError` 已替换

---

### 5.7 子步骤 3.1.6：OrcaGymEuler 查询委托链路

**涉及文件**：`orca_gym/core/euler/orca_gym_euler.py`

**实现内容**：新增委托方法（内部用 `object.__getattribute__` 访问子组件）：

```python
class OrcaGymEuler:
    # --- 查询委托 ---
    def query_joint_qpos(self, joint_names):
        return object.__getattribute__(self, "_sim").query_joint_qpos(joint_names)
    def query_joint_qvel(self, joint_names): ...
    # ... 其余 query_* 委托同理 ...
    def query_sensor_data(self, sensor_names):
        sim = object.__getattribute__(self, "_sim")
        model = object.__getattribute__(self, "_orca_model")
        sensor_info = {name: model.get_sensor(name) for name in sensor_names}
        return sim.query_sensor_data(sensor_names, sensor_info)
    def body_subtree_mass(self, body_name):
        return object.__getattribute__(self, "_registry").body_subtree_mass(body_name)
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_gym_query_delegates_use_getattribute` | grep 断言新增委托方法均用 `object.__getattribute__(self, "_sim"/"_registry"/"_orca_model")` | K3 |
| `test_gym_dir_no_internal_leak` | `dir(gym)` 不含 `_sim`/`_studio`/`_registry`/`_mjData`/`_mjModel` | K2/K3 |
| `test_gym_no_new_property` | grep 断言 `@property` 装饰器不新增 `_sim`/`_studio`/`_registry` | K5 |
| `test_gym_query_returns_typed` | `gym.query_joint_qpos(...)` 返回 `dict[str, np.ndarray]`，非 `MjData` | K11 |
| `test_gym_sensor_delegates_assemble_sensor_info` | `query_sensor_data` 从 `_orca_model` 拼装 sensor_info 传入 SimCore | K3 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_gym_query_joint_qpos_delegates_to_simcore` | Gym 委托结果与 SimCore 直接调用结果一致 |
| `test_gym_query_sensor_data_assembles_sensor_info` | sensor_info 正确拼装，传感器数据正确 |
| `test_gym_body_subtree_mass_delegates_to_registry` | 委托结果与 Registry 直接调用一致 |

**子步骤验收**：
- [x] 全部查询委托方法实现完成（关节/Body/Site/传感器/执行器/接触/Geom + body_subtree_mass）
- [x] 架构遵从性测试通过（K3 `object.__getattribute__` + K2 `__dir__` + K5 无 property + K11 typed 返回）
- [x] 功能单元测试通过（委托链路结果与底层一致）
- [x] grep 断言：委托方法均用 `object.__getattribute__`，不直接 `self._sim`

---

### 5.8 子步骤 3.1.7：OrcaGymEulerEnv 公共查询 API

**涉及文件**：`orca_gym/environment/euler/orca_gym_euler_env.py`

**实现内容**：新增公共查询方法（全部委托 `self._gym`）：

```python
class OrcaGymEulerEnv:
    # --- 关节查询（委托 _gym）---
    def query_joint_qpos(self, joint_names): return self._gym.query_joint_qpos(joint_names)
    def query_joint_qvel(self, joint_names): return self._gym.query_joint_qvel(joint_names)
    def query_joint_qacc(self, joint_names): return self._gym.query_joint_qacc(joint_names)
    def query_joint_offsets(self, joint_names): return self._gym.query_joint_offsets(joint_names)
    def query_joint_lengths(self, joint_names): return self._gym.query_joint_lengths(joint_names)
    def query_joint_dofadrs(self, joint_names): return self._gym.query_joint_dofadrs(joint_names)
    def jnt_qposadr(self, joint_name): return self._gym.jnt_qposadr(joint_name)
    def jnt_dofadr(self, joint_name): return self._gym.jnt_dofadr(joint_name)
    # --- Body/Site/传感器/执行器/接触/Geom 查询 ---
    def get_body_xpos_xmat_xquat(self, body_name_list): return self._gym.query_body_xpos_xmat_xquat(body_name_list)
    def get_body_xpos_xmat_xquat_xvel(self, body_name_list): return self._gym.query_body_xpos_xmat_xquat_xvel(body_name_list)
    def query_site_pos_and_mat(self, site_names): return self._gym.query_site_pos_and_mat(site_names)
    def query_site_size(self, site_names): return self._gym.query_site_size(site_names)
    def query_sensor_data(self, sensor_names): return self._gym.query_sensor_data(sensor_names)
    def query_actuator_torques(self, actuator_names): return self._gym.query_actuator_torques(actuator_names)
    def query_contact_simple(self): return self._gym.query_contact_simple()
    def query_contact_force(self, contact_ids): return self._gym.query_contact_force(contact_ids)
    def get_cfrc_ext(self): return self._gym.get_cfrc_ext()
    def get_goal_bounding_box(self, geom_name): ...
    def body_subtree_mass(self, body_name): return self._gym.body_subtree_mass(body_name)
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_env_query_no_gym_private_access` | grep 断言 `orca_gym_euler_env.py` 查询方法不触 `self._gym._sim`/`_studio`/`_registry`/`_mjData`/`_mjModel` | K4 |
| `test_env_query_uses_self_gym_only` | grep 断言查询方法均用 `self._gym.<公共方法>` 委托，不 `self._sim`/`self._studio` | K1/K4 |
| `test_env_dir_includes_new_query_methods` | `dir(env)` 含 `query_joint_qpos` 等新方法 | K2 |
| `test_env_dir_no_internal_leak` | `dir(env)` 不含 `_sim`/`_studio`/`_registry`/`_mjData`/`_mjModel` | K2 |
| `test_env_query_returns_typed` | 公共查询方法返回 ndarray/dict/tuple，非 MjData/MjModel | K11 |
| `test_env_query_docstrings_present` | 新增方法有 docstring（含用法与禁止说明） | K12 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_env_query_joint_qpos_returns_correct_slice` | `env.query_joint_qpos(["left_hip_pitch"])` 返回正确切片 |
| `test_env_get_body_xpos_xmat_xquat_flat_shape` | 返回扁平数组，形状 `(3*N,)`/`(9*N,)`/`(4*N,)` |
| `test_env_query_sensor_data_matches_sensordata` | 与 `_mjData.sensordata` 一致 |
| `test_env_query_contact_simple_returns_list` | 接触列表结构正确 |
| `test_env_body_subtree_mass_positive` | 返回正标量 |

**子步骤验收**：
- [x] 全部公共查询 API 实现完成（委托 `self._gym`）
- [x] 架构遵从性测试通过（K1/K2/K4/K11/K12）
- [x] 功能单元测试通过（加载 G1 XML 验证查询数值正确）
- [x] grep 断言：`orca_gym_euler_env.py` 无 `self._gym._xxx` 穿墙访问

---

### 5.9 子步骤 3.1.8：OrcaGymEulerEnv 基座坐标系变换方法

**涉及文件**：`orca_gym/environment/euler/orca_gym_euler_env.py`

**实现内容**：基座坐标系变换方法（纯 NumPy，在 Env 层实现，不下沉到 SimCore）：

```python
class OrcaGymEulerEnv:
    # --- 基座坐标系变换方法（纯 NumPy，在 Env 层实现）---
    def query_site_pos_and_quat_B(self, site_names, base_body_list): ...
    def query_site_xvalp_xvalr(self, site_names): ...
    def query_site_xvalp_xvalr_B(self, site_names, base_body_list): ...
    def query_velocity_body_B(self, ee_body, base_body): ...
    def query_position_body_B(self, ee_body, base_body): ...
    def query_orientation_body_B(self, ee_body, base_body): ...
    def query_joint_axes_B(self, joint_names, base_body): ...
    def query_robot_velocity_odom(self, base_body, initial_base_pos, initial_base_quat): ...
    def query_robot_position_odom(self, base_body, initial_base_pos, initial_base_quat): ...
    def query_robot_orientation_odom(self, base_body, initial_base_pos, initial_base_quat): ...
```

**关键设计决策**：
1. **基座坐标系变换方法放 Env 层**：依赖 `scipy.spatial.transform.Rotation`，是纯 NumPy 计算（基于 body/site 查询结果），不下沉到 SimCore（保持 SimCore 只做 MuJoCo 原生操作）。复用 `orca_gym.utils.rotations`（`mat2quat`/`quat2mat` 等）。
2. **签名与 OrcaGymLocalEnv 完全一致**：确保老代码零改动迁移（仅 `gym.` → `env.`）。

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_env_base_transform_no_gym_private` | grep 断言 `*_B`/`*_odom` 方法不触 `self._gym._sim`/`_mjData`，仅调 `self._gym.query_*` 公共方法 | K4 |
| `test_env_base_transform_returns_typed` | 返回 ndarray/dict，非 MjData/MjModel | K11 |
| `test_env_base_transform_no_simcore_dependency` | grep 断言 Env 层变换方法不 import `MuJoCoSimCore` 或直接访问 `_mjData` | K4/P2 |
| `test_env_base_transform_docstrings_present` | 新增方法有 docstring | K12 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_query_site_pos_and_quat_B_relative` | 基座坐标系变换正确（与世界系差一个基座变换） |
| `test_query_velocity_body_B_consistency` | 末端在基座系下速度 = 基座逆变换 ⊗ 世界系速度 |
| `test_query_robot_position_odom_accumulates` | 里程计累积正确（多步后位置 = 初始 + 累积位移） |
| `test_query_joint_axes_B_transformed` | 关节轴在基座系下正确变换 |

**子步骤验收**：
- [x] 全部基座坐标系变换方法实现完成（纯 NumPy，Env 层）
- [x] 架构遵从性测试通过（K4 不触私有 + K11 typed 返回 + K12 docstring）
- [x] 功能单元测试通过（基座变换数值正确）
- [x] grep 断言：变换方法不 import `MuJoCoSimCore`，不直接访问 `_mjData`

---

## 6. 阶段 3.2：状态设置与外力应用 API

### 6.1 目标

填充力应用与状态设置方法，替代老体系的 `xfrc_applied` 直接写、`set_mocap_pos_and_quat` 等。力应用通过显式方法（P4 可追踪），mocap 设置同步到远端 Studio。

本阶段拆分为 **5 个独立子步骤**（3.2.1–3.2.5），先实现层后委托层，最后验证 DataView 只读保护。

---

### 6.2 子步骤 3.2.1：MuJoCoSimCore 力应用方法

**涉及文件**：`orca_gym/core/euler/mujoco_sim_core.py`

**实现内容**：

```python
class MuJoCoSimCore:
    def apply_body_force(self, body_id: int, force: np.ndarray, torque: np.ndarray) -> None:
        """对 body 施加外力/力矩（写 xfrc_applied[body_id, :6]）。"""
        f = np.asarray(force, dtype=np.float64).reshape(3)
        tau = np.asarray(torque, dtype=np.float64).reshape(3)
        self._mjData.xfrc_applied[body_id, :3] = f
        self._mjData.xfrc_applied[body_id, 3:6] = tau

    def clear_body_force(self, body_id: int) -> None:
        """清除 body 外力（xfrc_applied[body_id, :6] = 0）。"""
        self._mjData.xfrc_applied[body_id, :6] = 0.0

    def clear_all_forces(self) -> None:
        """清除所有外力（xfrc_applied[:] = 0）。"""
        self._mjData.xfrc_applied[:] = 0.0

    def mj_apply_force_at_site(self, site_id: int, force: np.ndarray, torque: np.ndarray) -> None:
        """在 site 处施加力（等价 mujoco.mj_applyForce，写 xfrc_applied[site.bodyid]）。"""

    def mj_clear_xfrc_applied_for_site(self, site_id: int) -> None:
        """清除 site 关联 body 的 xfrc。"""
        body_id = self._mjModel.site_bodyid[site_id]
        self.clear_body_force(body_id)
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_simcore_force_methods_write_xfrc_only` | grep 断言力应用方法只写 `xfrc_applied`，不返回 `MjData`/`MjModel` | P2/K11 |
| `test_simcore_force_methods_return_none` | `apply_body_force`/`clear_*` 返回 `None`（写操作无返回值） | K11 |
| `test_simcore_force_no_mjdata_leak` | grep 断言不 `return self._mjData` | P2/K11 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_apply_body_force_writes_xfrc` | 施力后 `_mjData.xfrc_applied[body_id, :3]` 等于 force |
| `test_clear_body_force_zeroes_xfrc` | 清力后 `xfrc_applied[body_id, :6]` 为 0 |
| `test_clear_all_forces_zeroes_all` | 清全部后 `xfrc_applied[:]` 为 0 |
| `test_mj_apply_force_at_site_writes_body_xfrc` | site 施力后关联 body 的 xfrc 写入 |
| `test_mj_clear_xfrc_for_site_clears_body` | 清 site xfrc 后关联 body 的 xfrc 清零 |

**子步骤验收**：
- [x] 5 个力应用方法实现完成
- [x] 架构遵从性测试通过（P2 只写 xfrc + K11 返回 None）
- [x] 功能单元测试通过（xfrc 写入/清零数值正确）
- [x] grep 断言：力应用方法不 `return self._mjData`

---

### 6.3 子步骤 3.2.2：MuJoCoSimCore 状态设置方法

**涉及文件**：`orca_gym/core/euler/mujoco_sim_core.py`

**实现内容**：

```python
class MuJoCoSimCore:
    def set_mocap_pos_and_quat(self, mocap_dict: dict[str, dict]) -> None:
        """设置 mocap body 位置/四元数（写 mocap_pos/mocap_quat）。"""
        for body_name, pose in mocap_dict.items():
            body_id = mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mocap_id = self._mjModel.body_mocapid[body_id]
            if mocap_id >= 0:
                self._mjData.mocap_pos[mocap_id] = pose["pos"]
                self._mjData.mocap_quat[mocap_id] = pose["quat"]

    def set_geom_friction(self, geom_friction_dict: dict[str, np.ndarray]) -> None:
        """设置 geom 摩擦系数（写 geom_friction）。"""
        for geom_name, friction in geom_friction_dict.items():
            geom_id = mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self._mjModel.geom_friction[geom_id] = friction

    def add_extra_weight(self, weight_load_dict: dict) -> None:
        """为 body 添加额外重量（修改 body_mass/body_inertia）。"""
        for body_name, weight in weight_load_dict.items():
            body_id = self._mjModel.body(body_name).id
            self._mjModel.body_mass[body_id] += weight
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_simcore_set_methods_return_none` | 3 个设置方法返回 `None`（写操作无返回值） | K11 |
| `test_simcore_set_methods_no_mjdata_leak` | grep 断言不 `return self._mjData`/`self._mjModel` | P2/K11 |
| `test_simcore_set_geom_friction_writes_model` | grep 断言写 `_mjModel.geom_friction`（模型字段，非 data） | P2 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_set_mocap_pos_and_quat_writes_mocap` | mocap_pos/quat 正确写入 |
| `test_set_geom_friction_persists` | geom_friction 修改持久化 |
| `test_add_extra_weight_increases_mass` | 添加重量后 body_mass 增加 |

**子步骤验收**：
- [x] 3 个状态设置方法实现完成
- [x] 架构遵从性测试通过（K11 返回 None + P2 不泄漏）
- [x] 功能单元测试通过（写入数值正确）
- [x] `set_geom_friction`/`add_extra_weight` 写 `_mjModel`（非 `_mjData`）

---

### 6.4 子步骤 3.2.3：OrcaStudioBridge mocap 远端同步

**涉及文件**：`orca_gym/core/euler/orca_studio_bridge.py`

**实现内容**：

```python
class OrcaStudioBridge:
    async def set_mocap_pos_and_quat(self, mocap_data: dict, send_remote: bool = False) -> None:
        """设置 mocap 位姿并同步到远端 Studio（依赖反转：接收 mocap_data）。"""
        if self._stub is None:
            return
        if send_remote:
            request = mjc_message_pb2.SetMocapPosAndQuatRequest(...)
            await self._stub.SetMocapPosAndQuat(request)
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_bridge_mocap_offline_noop` | 离线模式（`_stub is None`）不抛错，直接 return | K9 |
| `test_bridge_mocap_no_mjdata_dependency` | grep 断言 Bridge 不 import `MjData`/`MjModel`，仅操作 gRPC stub | K9/P2 |
| `test_bridge_mocap_async_signature` | 方法为 `async def`，返回 `None` | K9 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_set_mocap_remote_offline_returns_none` | 离线模式返回 None 不抛错 |
| `test_set_mocap_remote_online_calls_stub` | 在线模式（mock stub）调用 `SetMocapPosAndQuat` |

**子步骤验收**：
- [x] `set_mocap_pos_and_quat` async 方法实现完成
- [x] 架构遵从性测试通过（K9 走 bridge + 离线 no-op）
- [x] 功能单元测试通过（离线 no-op + 在线委托 stub）
- [x] grep 断言：Bridge 不 import `MjData`/`MjModel`

---

### 6.5 子步骤 3.2.4：OrcaGymEuler/Env 力应用与设置委托

**涉及文件**：`orca_gym/core/euler/orca_gym_euler.py`、`orca_gym/environment/euler/orca_gym_euler_env.py`

**实现内容**：力应用方法在 `OrcaGymEulerEnv` 按 body_name 解析 body_id 后委托，`set_mocap_pos_and_quat` 在线模式同步远端：

```python
class OrcaGymEulerEnv:
    def apply_body_force(self, body_name: str, force: np.ndarray, torque: np.ndarray) -> None:
        body_id = self.model.body_name2id(body_name)
        self._gym.apply_body_force(body_id, force, torque)

    def clear_body_force(self, body_name: str) -> None:
        body_id = self.model.body_name2id(body_name)
        self._gym.clear_body_force(body_id)

    def clear_all_forces(self) -> None:
        self._gym.clear_all_forces()

    def mj_apply_force_at_site(self, site_name: str, force, torque) -> None:
        site_id = self.model.site_name2id(site_name)
        self._gym.mj_apply_force_at_site(site_id, force, torque)

    def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict: dict) -> None:
        self._gym.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)
        send_remote = self._render_mode == "human" and not getattr(self, "_is_subenv", False)
        self.loop.run_until_complete(
            self._gym.set_mocap_pos_and_quat_remote(mocap_pos_and_quat_dict, send_remote))

    def set_geom_friction(self, geom_friction_dict: dict) -> None:
        self._gym.set_geom_friction(geom_friction_dict)

    def add_extra_weight(self, weight_load_dict: dict) -> None:
        self._gym.add_extra_weight(weight_load_dict)
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_env_force_no_gym_private_access` | grep 断言力应用/设置方法不触 `self._gym._sim`/`_mjData`/`_mjModel` | K4 |
| `test_env_force_uses_self_gym_and_model` | grep 断言走 `self._gym.<方法>` + `self.model.body_name2id`，不 `self._sim` | K1/K4 |
| `test_env_mocap_uses_studio_bridge` | grep 断言 `set_mocap_pos_and_quat` 远端同步走 `self._gym`/`self._studio_bridge`，不走 `gym.studio` | K9 |
| `test_env_force_dir_includes_new_methods` | `dir(env)` 含 `apply_body_force`/`clear_all_forces` 等 | K2 |
| `test_env_force_docstrings_present` | 新增方法有 docstring | K12 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_env_apply_body_force_writes_xfrc` | 施力后 `env.data.xfrc_applied[body_id, :3]` 等于 force |
| `test_env_clear_body_force_zeroes_xfrc` | 清力后 `env.data.xfrc_applied[body_id, :6]` 为 0 |
| `test_env_set_mocap_pos_and_quat_writes_mocap` | mocap_pos/quat 正确写入 |
| `test_env_set_geom_friction_persists` | geom_friction 修改持久化 |
| `test_env_apply_force_pelvis_z_changes` | 施力后 pelvis z 位置变化（步进验证） |

**子步骤验收**：
- [x] Gym + Env 力应用与设置委托方法实现完成
- [x] 架构遵从性测试通过（K1/K2/K4/K9/K12）
- [x] 功能单元测试通过（施力后 xfrc/pelvis z 变化验证）
- [x] grep 断言：`orca_gym_euler_env.py` 力应用方法无 `self._gym._xxx`

---

### 6.6 子步骤 3.2.5：OrcaGymDataView xfrc_applied 只读保护验证

**涉及文件**：`orca_gym/core/euler/orca_gym_data_view.py`（验证现有保护，可能需加固）

**实现内容**：确认 `env.data.xfrc_applied` 为只读视图，直接写应引导报错或无效果（P4 力应用可追踪：用户必须走 `env.apply_body_force()`）。

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_dataview_xfrc_is_view_not_copy` | `env.data.xfrc_applied.base` 非None（零拷贝视图），读到的值与 `_mjData` 同步 | K6 |
| `test_dataview_xfrc_direct_write_blocked_or_warns` | 直接写 `env.data.xfrc_applied[...] = ...` 应报错或 warning，引导走 `apply_body_force` | P4/K6 |
| `test_env_data_is_dataview_after_force` | 施力后 `isinstance(env.data, OrcaGymDataView)` 仍为 True | K6 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_apply_force_blocked_via_data_view` | `env.data.xfrc_applied` 只读，直接写应引导报错 |
| `test_xfrc_readable_after_apply_force` | `env.apply_body_force()` 后 `env.data.xfrc_applied` 可读到正确的力 |

**子步骤验收**：
- [x] DataView `xfrc_applied` 只读保护验证完成（直接写被阻断或 warning）
- [x] 架构遵从性测试通过（K6 DataView + P4 力应用可追踪）
- [x] 功能单元测试通过（只读保护 + 施力后可读）
- [x] 用户无法绕过 `apply_body_force()` 直接写 `xfrc_applied`

---

## 7. 阶段 3.3：雅可比与高级 MuJoCo 操作

### 7.1 目标

填充 `mj_jacBody`/`mj_jacSite`/`mj_jac_site` 批量方法，支持逆运动学、速度控制等高级算法。这些是 Franka RL、Legged Gym 等复杂场景的关键依赖。

本阶段拆分为 **3 个独立子步骤**（3.3.1–3.3.3），先实现层后委托层。

---

### 7.2 子步骤 3.3.1：MuJoCoSimCore mj_jacBody/mj_jacSite

**涉及文件**：`orca_gym/core/euler/mujoco_sim_core.py`

**实现内容**：

```python
class MuJoCoSimCore:
    def mj_jacBody(self, jacp: np.ndarray, jacr: np.ndarray, body_id: int) -> None:
        """计算 body 雅可比（mujoco.mj_jacBody，原地填充 jacp/jacr）。"""
        mujoco.mj_jacBody(self._mjModel, self._mjData, jacp, jacr, body_id)

    def mj_jacSite(self, jacp: np.ndarray, jacr: np.ndarray, site_id: int) -> None:
        """计算 site 雅可比（mujoco.mj_jacSite）。"""
        mujoco.mj_jacSite(self._mjModel, self._mjData, jacp, jacr, site_id)
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_simcore_jac_methods_return_none` | `mj_jacBody`/`mj_jacSite` 返回 `None`（原地填充） | K11 |
| `test_simcore_jac_no_mjdata_leak` | grep 断言不 `return self._mjData`/`self._mjModel` | P2/K11 |
| `test_simcore_jac_uses_mujoco_native` | grep 断言调用 `mujoco.mj_jacBody`/`mj_jacSite`（原生操作集中 SimCore） | P2 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_mj_jacBody_correct_shape` | jacp/jacr 形状 `(3, nv)` |
| `test_mj_jacSite_correct_shape` | jacp/jacr 形状 `(3, nv)` |
| `test_mj_jacBody_velocity_consistency` | `jacp @ qvel` 与 `body_cvel` 线速度数值一致 |

**子步骤验收**：
- [ ] `mj_jacBody`/`mj_jacSite` 实现完成
- [ ] 架构遵从性测试通过（K11 返回 None + P2 原生操作集中）
- [ ] 功能单元测试通过（形状 + jacp@qvel 一致性）
- [ ] 解除 3.1.2 `query_body_xpos_xmat_xquat_xvel` 的 `mj_jacBody` 依赖标注

---

### 7.3 子步骤 3.3.2：MuJoCoSimCore mj_jac_site 批量

**涉及文件**：`orca_gym/core/euler/mujoco_sim_core.py`

**实现内容**：

```python
class MuJoCoSimCore:
    def mj_jac_site(self, site_names: list[str]) -> dict[str, dict]:
        """批量计算 site 雅可比，返回 {site_name: {jacp, jacr}}。"""
        nv = self._mjModel.nv
        result = {}
        for site_name in site_names:
            site_id = mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_SITE, site_name)
            jacp = np.zeros((3, nv), dtype=np.float64)
            jacr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacSite(self._mjModel, self._mjData, jacp, jacr, site_id)
            result[site_name] = {"jacp": jacp, "jacr": jacr}
        return result
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_simcore_jac_site_returns_dict_of_dict` | 返回 `dict[str, dict]`，内层含 `jacp`/`jacr` 键，值为 `np.ndarray` | K11 |
| `test_simcore_jac_site_no_mjdata_leak` | grep 断言不 `return self._mjData`/`self._mjModel` | P2/K11 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_mj_jac_site_batch_returns_all` | 批量返回所有 site 的雅可比 |
| `test_mj_jac_site_velocity_consistency` | `jacp @ qvel` 与 site 速度一致 |
| `test_mj_jac_site_keys_match_input` | 返回 dict 键与输入 site_names 一致 |

**子步骤验收**：
- [ ] `mj_jac_site` 批量方法实现完成
- [ ] 架构遵从性测试通过（K11 typed 返回 + P2）
- [ ] 功能单元测试通过（批量返回 + 速度一致性）
- [ ] 解除 3.1.8 `query_site_xvalp_xvalr` 的 `mj_jac_site` 依赖标注

---

### 7.4 子步骤 3.3.3：OrcaGymEuler/Env 雅可比委托

**涉及文件**：`orca_gym/core/euler/orca_gym_euler.py`、`orca_gym/environment/euler/orca_gym_euler_env.py`

**实现内容**：

```python
class OrcaGymEulerEnv:
    def mj_jacBody(self, jacp, jacr, body_id):
        self._gym.mj_jacBody(jacp, jacr, body_id)

    def mj_jacSite(self, jacp, jacr, site_name):
        site_id = self.model.site_name2id(site_name)
        self._gym.mj_jacSite(jacp, jacr, site_id)

    def mj_jac_site(self, site_names):
        return self._gym.mj_jac_site(site_names)
```

> **注意**：`mj_jacBody`/`mj_jacSite` 接收 body_id/site_id（与老体系签名一致），用户通过 `self.model.body_name2id()` 获取 id。`query_site_xvalp_xvalr` 在 3.1.8 已实现，内部调用 `mj_jac_site`。

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_env_jac_no_gym_private_access` | grep 断言雅可比方法不触 `self._gym._sim`/`_mjData`/`_mjModel` | K4 |
| `test_env_jac_uses_self_gym_and_model` | grep 断言走 `self._gym.mj_jac*` + `self.model.site_name2id` | K1/K4 |
| `test_env_jac_dir_includes_methods` | `dir(env)` 含 `mj_jacBody`/`mj_jacSite`/`mj_jac_site` | K2 |
| `test_env_jac_returns_typed` | `mj_jac_site` 返回 `dict[str, dict]`，`mj_jacBody`/`mj_jacSite` 返回 None | K11 |
| `test_env_jac_docstrings_present` | 新增方法有 docstring | K12 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_env_mj_jacBody_delegates_correctly` | Env 委托结果与 SimCore 直接调用一致 |
| `test_env_mj_jac_site_delegates_correctly` | Env 批量委托结果与 SimCore 一致 |
| `test_env_mj_jacSite_resolves_site_name` | site_name → site_id 解析正确 |

**子步骤验收**：
- [ ] Gym + Env 雅可比委托方法实现完成
- [ ] 架构遵从性测试通过（K1/K2/K4/K11/K12）
- [ ] 功能单元测试通过（委托结果一致 + site_name 解析）
- [ ] grep 断言：`orca_gym_euler_env.py` 雅可比方法无 `self._gym._xxx`

---

## 8. 阶段 3.4：Studio 在线交互完整实现

### 8.1 目标

填充视频录制、帧捕获、内容文件等 Studio 交互方法，完善在线模式能力。这些是 OrcaPlayground 中可视化、数据采集场景的依赖。

本阶段拆分为 **4 个独立子步骤**（3.4.1–3.4.4），先 Bridge 层后委托层，最后集成 `do_body_manipulation`（依赖 3.5 约束方法）。

---

### 8.2 子步骤 3.4.1：OrcaStudioBridge 视频录制方法

**涉及文件**：`orca_gym/core/euler/orca_studio_bridge.py`

**实现内容**：

```python
class OrcaStudioBridge:
    async def begin_save_video(self, file_path: str, capture_mode) -> None:
        """开始录制视频（gRPC BeginSaveVideo）。"""
    async def stop_save_video(self) -> None:
        """停止录制视频（gRPC StopSaveVideo）。"""
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_bridge_video_offline_noop` | 离线模式（`_stub is None`）不抛错 | K9 |
| `test_bridge_video_no_mjdata_dependency` | grep 断言 Bridge 不 import `MjData`/`MjModel` | K9/P2 |
| `test_bridge_video_async_signature` | 方法为 `async def`，返回 `None` | K9 |
| `test_env_video_uses_studio_bridge` | grep 断言 Env 视频方法走 `self._studio_bridge`/`self._gym`，不走 `gym.studio` | K9 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_begin_save_video_offline_noop` | 离线模式 no-op 不抛错 |
| `test_stop_save_video_offline_noop` | 离线模式 no-op 不抛错 |
| `test_begin_stop_save_video_online_calls_stub` | 在线模式（mock stub）调用 gRPC |

**子步骤验收**：
- [ ] `begin_save_video`/`stop_save_video` async 方法实现完成
- [ ] 架构遵从性测试通过（K9 走 bridge + 离线 no-op + 不依赖 MjData）
- [ ] 功能单元测试通过（离线 no-op + 在线委托 stub）
- [ ] grep 断言：Bridge 不 import `MjData`/`MjModel`

---

### 8.3 子步骤 3.4.2：OrcaStudioBridge 帧捕获方法

**涉及文件**：`orca_gym/core/euler/orca_studio_bridge.py`

**实现内容**：

```python
class OrcaStudioBridge:
    async def get_current_frame(self) -> int:
        """获取当前帧索引（gRPC GetCurrentFrame）。"""
    async def get_camera_time_stamp(self, last_frame: int) -> dict:
        """获取相机时间戳（gRPC GetCameraTimeStamp）。"""
    async def get_frame_png(self, image_path: str):
        """获取帧 PNG（gRPC GetFramePng）。"""
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_bridge_frame_offline_returns_default` | 离线模式返回默认值（-1/空 dict）不抛错 | K9 |
| `test_bridge_frame_no_mjdata_dependency` | grep 断言不 import `MjData`/`MjModel` | K9/P2 |
| `test_bridge_frame_returns_typed` | `get_current_frame` 返回 `int`，`get_camera_time_stamp` 返回 `dict` | K11 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_get_current_frame_offline_returns_neg1` | 离线模式返回 -1 |
| `test_get_camera_time_stamp_offline_returns_empty` | 离线模式返回空 dict |
| `test_get_frame_png_offline_noop` | 离线模式 no-op |
| `test_get_current_frame_online_calls_stub` | 在线模式委托 stub |

**子步骤验收**：
- [ ] 3 个帧捕获 async 方法实现完成
- [ ] 架构遵从性测试通过（K9 走 bridge + K11 typed 返回）
- [ ] 功能单元测试通过（离线默认值 + 在线委托 stub）
- [ ] grep 断言：Bridge 不 import `MjData`/`MjModel`

---

### 8.4 子步骤 3.4.3：OrcaStudioBridge 内容文件方法

**涉及文件**：`orca_gym/core/euler/orca_studio_bridge.py`

**实现内容**：

```python
class OrcaStudioBridge:
    async def load_content_file(self, content_file_name, remote_file_dir="",
                                local_file_dir="", temp_file_path=None):
        """加载内容文件（gRPC LoadContentFile）。"""
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_bridge_content_file_offline_noop` | 离线模式 no-op 不抛错 | K9 |
| `test_bridge_content_file_no_mjdata_dependency` | grep 断言不 import `MjData`/`MjModel` | K9/P2 |
| `test_bridge_content_file_async_signature` | 方法为 `async def` | K9 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_load_content_file_offline_noop` | 离线模式 no-op |
| `test_load_content_file_online_calls_stub` | 在线模式委托 stub |

**子步骤验收**：
- [ ] `load_content_file` async 方法实现完成
- [ ] 架构遵从性测试通过（K9 走 bridge + 离线 no-op）
- [ ] 功能单元测试通过（离线 no-op + 在线委托 stub）
- [ ] grep 断言：Bridge 不 import `MjData`/`MjModel`

---

### 8.5 子步骤 3.4.4：OrcaGymEuler/Env Studio 委托 + do_body_manipulation

**涉及文件**：`orca_gym/core/euler/orca_gym_euler.py`、`orca_gym/environment/euler/orca_gym_euler_env.py`

**实现内容**：Studio 委托方法 + `do_body_manipulation` 完整实现（依赖 3.5 约束方法）：

```python
class OrcaGymEulerEnv:
    def begin_save_video(self, file_path, capture_mode=CaptureMode.ASYNC):
        self.loop.run_until_complete(self._gym.begin_save_video(file_path, capture_mode))
    def stop_save_video(self):
        self.loop.run_until_complete(self._gym.stop_save_video())
    def get_current_frame(self) -> int:
        return self.loop.run_until_complete(self._gym.get_current_frame())
    def get_next_frame(self) -> int:
        """带轮询的获取下一帧（复用老体系逻辑）。"""
    def get_camera_time_stamp(self, last_frame_index) -> dict:
        return self.loop.run_until_complete(self._gym.get_camera_time_stamp(last_frame_index))
    def get_frame_png(self, image_path):
        return self.loop.run_until_complete(self._gym.get_frame_png(image_path))
```

> **`do_body_manipulation` 完整实现**：阶段二的 `do_body_manipulation` 仅查询状态不应用，阶段 3.4.4 配合 3.5 的约束方法实现完整体操作（锚定 + mocap + 等式约束）。完整实现见 §9.7（子步骤 3.5.6）。

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_env_studio_no_gym_private_access` | grep 断言 Studio 方法不触 `self._gym._sim`/`_studio`/`_mjData` | K4 |
| `test_env_studio_uses_studio_bridge` | grep 断言视频/帧方法走 `self._gym`/`self._studio_bridge`，不走 `gym.studio` | K9 |
| `test_env_studio_dir_includes_methods` | `dir(env)` 含 `begin_save_video`/`get_current_frame` 等 | K2 |
| `test_env_studio_returns_typed` | `get_current_frame` 返回 `int`，`get_camera_time_stamp` 返回 `dict` | K11 |
| `test_env_studio_docstrings_present` | 新增方法有 docstring | K12 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_env_begin_stop_save_video_offline_noop` | 离线模式 no-op |
| `test_env_get_current_frame_offline_returns_neg1` | 离线模式返回 -1 |
| `test_env_video_methods_delegate_to_bridge` | 在线模式委托到 bridge（mock stub） |

**子步骤验收**：
- [ ] Gym + Env Studio 委托方法实现完成
- [ ] 架构遵从性测试通过（K2/K4/K9/K11/K12）
- [ ] 功能单元测试通过（离线 no-op + 在线委托链路）
- [ ] grep 断言：Studio 方法走 `self._studio_bridge`/`self._gym`，不走 `gym.studio`

---

## 9. 阶段 3.5：等式约束与完整体操作

### 9.1 目标

填充等式约束操作方法，实现完整的 Studio UI 体操作（`anchor_actor`/`release_body_anchored`/`do_body_manipulation`），支持 mocap body 拖拽。这是 Franka RL（mocap + weld constraint 位置控制）等场景的关键依赖。

本阶段拆分为 **6 个独立子步骤**（3.5.1–3.5.6），先约束实现层后委托层，最后分步实现体操作的三个动作（锚定/释放/编排）。

---

### 9.2 子步骤 3.5.1：MuJoCoSimCore 等式约束方法

**涉及文件**：`orca_gym/core/euler/mujoco_sim_core.py`

**实现内容**：

```python
class MuJoCoSimCore:
    def update_equality_constraints(self, constraint_list: list[dict]) -> None:
        """更新等式约束（写 _mjModel.eq_*）。"""
        for i, eq in enumerate(constraint_list):
            self._mjModel.eq_type[i] = eq["eq_type"]
            self._mjModel.eq_obj1id[i] = eq["obj1_id"]
            self._mjModel.eq_obj2id[i] = eq["obj2_id"]
            self._mjModel.eq_active[i] = eq.get("active", 1)
            # eq_solref/eq_solimp/eq_data 按需更新

    def modify_equality_objects(self, old_obj1_id, old_obj2_id, new_obj1_id, new_obj2_id) -> None:
        """修改等式约束关联的对象 id。"""
        for i in range(self._mjModel.neq):
            if (self._mjModel.eq_obj1id[i] == old_obj1_id
                    and self._mjModel.eq_obj2id[i] == old_obj2_id):
                self._mjModel.eq_obj1id[i] = new_obj1_id
                self._mjModel.eq_obj2id[i] = new_obj2_id
                break
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_simcore_eq_methods_return_none` | 2 个约束方法返回 `None`（写操作） | K11 |
| `test_simcore_eq_methods_write_model_only` | grep 断言写 `_mjModel.eq_*`（模型字段），不写 `_mjData` | P2 |
| `test_simcore_eq_no_mjdata_leak` | grep 断言不 `return self._mjData`/`self._mjModel` | P2/K11 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_update_equality_constraints_writes_eq` | 约束更新后 `_mjModel.eq_obj1id` 正确 |
| `test_modify_equality_objects_swaps_ids` | 对象 id 交换正确 |
| `test_update_equality_constraints_active_flag` | `eq_active` 标志正确写入 |
| `test_modify_eq_rebind_box_to_pelvis` | `modify_equality_objects` 将 weld 的 obj2id 从 box 改为 pelvis 后，`eq_obj2id` 等于 pelvis body_id |
| `test_eq_active_disable_breaks_coupling` | `eq_active=0` 停用 weld 后，mocap 移动不再驱动绑定 body（box/pelvis） |

**子步骤验收**：
- [ ] `update_equality_constraints`/`modify_equality_objects` 实现完成
- [ ] 架构遵从性测试通过（K11 返回 None + P2 写 model）
- [ ] 功能单元测试通过（eq 字段写入正确 + 重绑定 + 停用解耦）
- [ ] grep 断言：约束方法写 `_mjModel.eq_*`，不 `return self._mjModel`

---

### 9.3 子步骤 3.5.2：ModelRegistry equality 查询补齐

**涉及文件**：`orca_gym/core/euler/model_registry.py`

**实现内容**：确认/补齐 `equality_data_width`/`equality_object_ids`（若 3.1.5 已覆盖则此子步骤为验证加固）。

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_registry_eq_returns_typed` | `equality_data_width` 返回 `int`，`equality_object_ids` 返回 `tuple[int, int]` | K11 |
| `test_registry_eq_no_mjmodel_leak` | grep 断言不 `return self._mj_model` | P2/K11 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_equality_data_width_matches_model` | 与 `_mjModel.eq_data.shape[1]` 一致 |
| `test_equality_object_ids_matches_model` | 与 `eq_obj1id`/`eq_obj2id` 一致 |

**子步骤验收**：
- [ ] equality 查询方法确认/补齐完成
- [ ] 架构遵从性测试通过（K11 + P2）
- [ ] 功能单元测试通过（数值与 MuJoCo 一致）

---

### 9.4 子步骤 3.5.3：OrcaGymEuler/Env 约束委托

**涉及文件**：`orca_gym/core/euler/orca_gym_euler.py`、`orca_gym/environment/euler/orca_gym_euler_env.py`

**实现内容**：

```python
class OrcaGymEulerEnv:
    def update_equality_constraints(self, eq_list):
        self._gym.update_equality_constraints(eq_list)
    def modify_equality_objects(self, old_obj1_id, old_obj2_id, new_obj1_id, new_obj2_id):
        self._gym.modify_equality_objects(old_obj1_id, old_obj2_id, new_obj1_id, new_obj2_id)
    def update_anchor_equality_constraints(self, actor_name, anchor_type):
        """更新锚点等式约束（复用老体系逻辑，走 update_equality_constraints）。"""
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_env_eq_no_gym_private_access` | grep 断言约束方法不触 `self._gym._sim`/`_mjModel` | K4 |
| `test_env_eq_uses_self_gym` | grep 断言走 `self._gym.update_equality_constraints` | K1/K4 |
| `test_env_eq_dir_includes_methods` | `dir(env)` 含 `update_equality_constraints`/`modify_equality_objects` | K2 |
| `test_env_eq_returns_none` | 约束方法返回 `None`（写操作） | K11 |
| `test_env_eq_docstrings_present` | 新增方法有 docstring | K12 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_env_update_equality_constraints_delegates` | Env 委托结果与 SimCore 一致 |
| `test_env_modify_equality_objects_delegates` | 委托结果正确 |

**子步骤验收**：
- [ ] Gym + Env 约束委托方法实现完成
- [ ] 架构遵从性测试通过（K1/K2/K4/K11/K12）
- [ ] 功能单元测试通过（委托链路正确）
- [ ] grep 断言：约束方法无 `self._gym._xxx`

---

### 9.5 子步骤 3.5.4：OrcaGymEulerEnv anchor_actor 实现

**涉及文件**：`orca_gym/environment/euler/orca_gym_euler_env.py`

**实现内容**：实现 `anchor_actor`（mocap + equality 联动，走合规 API）：

```python
class OrcaGymEulerEnv:
    def __init__(self, ...):
        # 锚点 body 初始化（与老体系一致）
        self._body_anchored = None
        self._is_flex_vertex_anchored = False
        self._anchor_body_name = "ActorManipulator_Anchor"
        self._anchor_dummy_body_name = "ActorManipulator_dummy"
        body_names = self.model.get_body_names()
        if (self._anchor_body_name in body_names
                and self._anchor_dummy_body_name in body_names):
            self._anchor_body_id = self.model.body_name2id(self._anchor_body_name)
            self._anchor_dummy_body_id = self.model.body_name2id(self._anchor_dummy_body_name)
        else:
            self._anchor_body_id = None
            self._anchor_dummy_body_id = None

    def anchor_actor(self, actor_name, anchor_type):
        """锚定 actor（mocap + equality 联动）。"""
        ...  # 复用老体系逻辑，走 set_mocap_pos_and_quat + update_anchor_equality_constraints
```

**关键设计决策**：复用老体系逻辑，走合规 API。`resolve_flex_body_name` 复用 `OrcaGymModel.resolve_flex_body_name()`。`get_eq_type`/`AnchorType` 从 `orca_gym.core.orca_gym_local` 导入或迁移到公共 utils。

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_env_anchor_no_gym_private_access` | grep 断言 `anchor_actor` 不触 `self._gym._sim`/`_mjData`/`_mjModel` | K4 |
| `test_env_anchor_uses_compliant_api` | grep 断言走 `self.set_mocap_pos_and_quat`/`self.update_equality_constraints`，不穿墙 | K4/P4 |
| `test_env_anchor_no_direct_xfrc_write` | grep 断言 `anchor_actor` 不直接写 `_mjData.xfrc_applied` | P4/K4 |
| `test_env_anchor_dir_includes_method` | `dir(env)` 含 `anchor_actor` | K2 |
| `test_env_anchor_docstrings_present` | 有 docstring | K12 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_anchor_actor_no_anchor_body_noop` | 无锚点 body 时 no-op |
| `test_anchor_actor_sets_body_anchored` | 锚定后 `_body_anchored` 非 None |
| `test_anchor_actor_calls_update_equality` | 锚定后调用 `update_equality_constraints` |

**子步骤验收**：
- [ ] `anchor_actor` 实现完成（走合规 API，不穿墙）
- [ ] 架构遵从性测试通过（K2/K4/K12/P4 不穿墙）
- [ ] 功能单元测试通过（无锚点 body no-op + 锚定状态正确）
- [ ] grep 断言：`anchor_actor` 不直接访问 `_mjData`/`_mjModel`

---

### 9.6 子步骤 3.5.5：OrcaGymEulerEnv release_body_anchored 实现

**涉及文件**：`orca_gym/environment/euler/orca_gym_euler_env.py`

**实现内容**：实现 `release_body_anchored`（释放锚定，走合规 API）：

```python
class OrcaGymEulerEnv:
    def release_body_anchored(self):
        """释放当前锚定的 body（停用等式约束 + 清状态）。"""
        if self._body_anchored is None:
            return
        # 走 update_equality_constraints 停用约束
        self.update_anchor_equality_constraints(actor_name=self._body_anchored,
                                                anchor_type=AnchorType.RELEASE)
        self._body_anchored = None
        self._is_flex_vertex_anchored = False
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_env_release_no_gym_private_access` | grep 断言不触 `self._gym._sim`/`_mjData`/`_mjModel` | K4 |
| `test_env_release_uses_compliant_api` | grep 断言走 `self.update_anchor_equality_constraints` | K4/P4 |
| `test_env_release_no_direct_eq_write` | grep 断言不直接写 `_mjModel.eq_active` | P4/K4 |
| `test_env_release_dir_includes_method` | `dir(env)` 含 `release_body_anchored` | K2 |
| `test_env_release_docstrings_present` | 有 docstring | K12 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_release_body_anchored_no_anchored_noop` | 无锚定时 no-op |
| `test_release_body_anchored_clears_state` | 释放后 `_body_anchored` 为 None |
| `test_release_body_anchored_calls_update_eq` | 释放时调用 `update_anchor_equality_constraints` |

**子步骤验收**：
- [ ] `release_body_anchored` 实现完成（走合规 API，不穿墙）
- [ ] 架构遵从性测试通过（K2/K4/K12/P4 不穿墙）
- [ ] 功能单元测试通过（无锚定 no-op + 释放状态正确）
- [ ] grep 断言：不直接写 `_mjModel.eq_*`

---

### 9.7 子步骤 3.5.6：OrcaGymEulerEnv do_body_manipulation 完整编排

**涉及文件**：`orca_gym/environment/euler/orca_gym_euler_env.py`

**实现内容**：实现 `do_body_manipulation` 完整编排（查询 + 锚定 + mocap 同步，三步组合）：

```python
class OrcaGymEulerEnv:
    def do_body_manipulation(self, body_name, manipulation_data):
        """Studio UI 体操作完整编排：查询 → 锚定 → mocap 同步。"""
        # 步骤 1：查询状态（阶段二已实现）
        body_id = self.model.body_name2id(body_name)
        if self._anchor_body_id is None:
            return  # 无锚点 body 时 no-op
        # 步骤 2：首次操作锚定
        if self._body_anchored is None:
            self.anchor_actor(body_name, AnchorType.WELD)
        elif self._body_anchored != body_name:
            self.release_body_anchored()
            self.anchor_actor(body_name, AnchorType.WELD)
        # 步骤 3：mocap 同步（走 set_mocap_pos_and_quat）
        mocap_data = {self._anchor_body_name: {
            "pos": manipulation_data["pos"],
            "quat": manipulation_data["quat"],
        }}
        self.set_mocap_pos_and_quat(mocap_data)
```

**架构遵从性测试**（专属）：

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_env_do_body_manipulation_no_gym_private_access` | grep 断言不触 `self._gym._sim`/`_mjData`/`_mjModel` | K4 |
| `test_env_do_body_manipulation_uses_compliant_api` | grep 断言走 `self.anchor_actor`/`self.set_mocap_pos_and_quat` | K4/P4 |
| `test_env_do_body_manipulation_no_direct_mocap_write` | grep 断言不直接写 `_mjData.mocap_pos`/`mocap_quat` | P4/K4 |
| `test_env_do_body_manipulation_dir_includes_method` | `dir(env)` 含 `do_body_manipulation` | K2 |
| `test_env_do_body_manipulation_docstrings_present` | 有 docstring | K12 |
| `test_env_do_body_manipulation_delegates_to_anchor_and_mocap` | grep 断言走 `anchor_actor`/`release_body_anchored`/`set_mocap_pos_and_quat` | K1/K4 |

**功能单元测试**（专属）：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_do_body_manipulation_no_anchor_body_noop` | 无锚点 body 时 no-op |
| `test_do_body_manipulation_first_call_anchors` | 首次调用触发 `anchor_actor` |
| `test_do_body_manipulation_switch_body_releases_and_anchors` | 切换 body 时释放再锚定 |
| `test_do_body_manipulation_syncs_mocap` | 调用后 mocap_pos/quat 正确更新 |
| `test_do_body_manipulation_pelvis_moves_with_mocap` | mocap 拖动后关联 body 位置变化（步进验证） |
| `test_do_body_manipulation_drives_box_via_weld` | 默认 weld 绑定 anchor↔box，移动 mocap 后 box xpos ≈ mocap_pos（atol=0.05，100 步收敛） |
| `test_do_body_manipulation_rebind_to_g1_pelvis` | `modify_equality_objects` 重绑 obj2id→pelvis 后，mocap 驱动 G1 pelvis 位移 > 0.05m（200 步） |

**子步骤验收**：
- [ ] `do_body_manipulation` 完整编排实现（查询 → 锚定 → mocap 同步）
- [ ] 架构遵从性测试通过（K1/K2/K4/K12/P4 不穿墙）
- [ ] 功能单元测试通过（首次锚定 + 切换释放 + mocap 同步 + 步进验证 + 重绑定驱动 G1）
- [ ] grep 断言：`do_body_manipulation` 不直接写 `_mjData.mocap_*`/`xfrc_*`

---

## 10. 跨子步骤一致性验证

### 10.1 目标

在每个子步骤独立验收后，验证跨子步骤、跨层的一致性。这些测试在所有子步骤完成后运行，确保整体架构遵从性不被破坏。

### 10.2 全局架构遵从性回归套件

| 测试用例 | 验证内容 | K 约束 |
|---------|---------|--------|
| `test_global_env_dir_no_private_leak` | `dir(env)` 不含 `_mjData`/`_mjModel`/`_sim`/`_studio` 属性 | K2/P2 |
| `test_global_env_data_is_dataview` | `isinstance(env.data, OrcaGymDataView)` 始终为 True | K6 |
| `test_global_no_mjdata_in_public_returns` | 所有公共方法返回值无 `mujoco.MjData`/`mujoco.MjModel` 类型 | K11/P2 |
| `test_global_gym_no_mjdata_in_public_returns` | `OrcaGymEuler` 所有公共方法返回值无 `MjData`/`MjModel` | K11 |
| `test_global_bridge_no_mjdata_import` | `orca_studio_bridge.py` 不 import `MjData`/`MjModel` | K9/P2 |
| `test_global_simcore_no_mjdata_return` | `mujoco_sim_core.py` 不含 `return self._mjData`/`return self._mjModel` | P2 |
| `test_global_env_methods_docstring_coverage` | 新增公共方法 docstring 覆盖率 ≥ 90% | K12 |
| `test_global_no_direct_xfrc_write_in_env` | `orca_gym_euler_env.py` 不含 `_mjData.xfrc_applied[` 直接写 | P4 |
| `test_global_no_direct_mocap_write_in_env` | `orca_gym_euler_env.py` 不含 `_mjData.mocap_pos[`/`mocap_quat[` 直接写 | P4 |
| `test_global_no_direct_eq_write_in_env` | `orca_gym_euler_env.py` 不含 `_mjModel.eq_` 直接写 | P4 |

### 10.3 端到端链路验证

| 测试用例 | 验证内容 |
|---------|---------|
| `test_e2e_query_and_force` | 查询 body 状态 → 施力 → 步进 → 状态变化（数值一致） |
| `test_e2e_jac_and_velocity_control` | 雅可比计算 → 速度控制 → 步进 → 位置变化 |
| `test_e2e_anchor_and_manipulate` | 锚定 → do_body_manipulation → 步进 → body 跟随 mocap |
| `test_e2e_offline_all_methods_no_crash` | 离线模式调用所有新增公共方法不抛错（no-op 或返回默认值） |
| `test_e2e_data_view_consistency_after_all_ops` | 所有操作后 `env.data` 仍为 DataView，读到的字段与 `_mjData` 同步 |
| `test_e2e_mocap_drive_box_via_weld` | mocap_pos 写入 → weld 约束驱动 box → 步进后 box xpos ≈ mocap_pos（atol=0.05） |
| `test_e2e_eq_disable_box_free` | `eq_active=0` 停用 weld → mocap 移动后 box 不跟随（自由落体） |
| `test_e2e_modify_eq_rebind_to_g1_pelvis` | `modify_equality_objects` 将 weld 的 obj2id 从 box 改为 pelvis → mocap 驱动 G1 pelvis 位移 > 0.05m |

### 10.4 子步骤依赖解除验证

每个子步骤的"依赖解除"标注（如"解除 3.1.x 的 `mj_jacBody` 依赖标注"）需在对应子步骤完成后验证：

| 测试用例 | 验证内容 |
|---------|---------|
| `test_dependency_312_jacbody_resolved` | 3.1.2 完成后，`query_body_xpos_xmat_xquat_xvel` 不再标注 `mj_jacBody` 依赖 |
| `test_dependency_318_jacsite_resolved` | 3.1.8 完成后，`query_site_xvalp_xvalr` 不再标注 `mj_jac_site` 依赖 |
| `test_dependency_354_anchor_uses_351_eq` | 3.5.4 `anchor_actor` 走 3.5.1 的 `update_equality_constraints` |
| `test_dependency_356_manipulation_uses_354_355` | 3.5.6 `do_body_manipulation` 走 3.5.4/3.5.5 + 3.2.4 的 `set_mocap_pos_and_quat` |

---

## 11. 回归测试矩阵

### 11.1 测试分层

每个子步骤的测试分为三层，按金字塔分布：

| 层级 | 占比 | 内容 | 工具 |
|------|------|------|------|
| 架构遵从性测试 | ~40% | grep 断言 + `dir()` 检查 + 类型断言 | `unittest` + `grep`（静态） |
| 功能单元测试 | ~50% | 加载 G1 XML，验证数值/形状/一致性 | `unittest` + `mujoco` + `numpy` |
| 端到端链路测试 | ~10% | 跨子步骤组合，步进验证 | `unittest` + `mujoco.mj_step` |

### 11.2 子步骤测试矩阵

| 子步骤 | 架构测试数 | 功能测试数 | 总计 |
|--------|----------|----------|------|
| 3.1.1–3.1.8（状态查询） | 24 | 32 | 56 |
| 3.2.1–3.2.5（力应用与设置） | 18 | 16 | 34 |
| 3.3.1–3.3.3（雅可比） | 10 | 9 | 19 |
| 3.4.1–3.4.4（Studio 交互） | 14 | 12 | 26 |
| 3.5.1–3.5.6（约束与体操作） | 28 | 18 | 46 |
| §10 全局回归 | 10 | 5 | 15 |
| **合计** | **104** | **92** | **196** |

### 11.3 测试文件组织

```
tests/euler/
├── test_arch_compliance/          # 架构遵从性测试（grep + dir + 类型）
│   ├── test_simcore_arch.py       # MuJoCoSimCore 架构测试
│   ├── test_dataview_arch.py      # OrcaGymDataView 架构测试
│   ├── test_registry_arch.py      # ModelRegistry 架构测试
│   ├── test_gym_arch.py           # OrcaGymEuler 架构测试
│   ├── test_env_arch.py           # OrcaGymEulerEnv 架构测试
│   └── test_bridge_arch.py        # OrcaStudioBridge 架构测试
├── test_functional/               # 功能单元测试（G1 XML 加载）
│   ├── test_simcore_query.py      # 状态查询功能测试
│   ├── test_simcore_force.py      # 力应用功能测试
│   ├── test_simcore_jac.py        # 雅可比功能测试
│   ├── test_env_delegation.py     # Env 委托功能测试
│   └── test_body_manipulation.py  # 体操作功能测试
├── test_e2e/                      # 端到端链路测试
│   └── test_offline_e2e.py        # 离线全链路测试
└── conftest.py                    # G1 XML fixture
```

### 11.4 架构遵从性测试代码模板

```python
"""架构遵从性测试模板 - 每个子步骤套用此模板。"""
import unittest
import re
from pathlib import Path
import numpy as np
import mujoco


class TestSubStepArchCompliance(unittest.TestCase):
    """子步骤 X.Y.Z 架构遵从性测试。"""

    @classmethod
    def setUpClass(cls):
        cls.env = _make_offline_env()  # 加载 g1_29dof_camera.xml
        cls.env_data_view_path = Path("orca_gym/core/euler/orca_gym_data_view.py")
        cls.simcore_path = Path("orca_gym/core/euler/mujoco_sim_core.py")
        cls.env_path = Path("orca_gym/environment/euler/orca_gym_euler_env.py")

    def test_k2_env_dir_no_private_leak(self):
        """K2: dir(env) 不含 _mjData/_mjModel/_sim/_studio。"""
        forbidden = {"_mjData", "_mjModel", "_sim", "_studio"}
        leaked = forbidden & set(dir(self.env))
        self.assertFalse(leaked, f"dir(env) 泄漏内部属性: {leaked}")

    def test_k11_public_returns_no_mjdata(self):
        """K11: 公共方法返回 typed 对象，不返回 MjData/MjModel。"""
        result = self.env.query_joint_qpos(["left_hip_pitch"])
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["left_hip_pitch"], np.ndarray)
        self.assertNotIsInstance(result, (mujoco.MjData, mujoco.MjModel))

    def test_p2_simcore_no_mjdata_return(self):
        """P2: mujoco_sim_core.py 不含 return self._mjData/_mjModel。"""
        src = self.simcore_path.read_text()
        self.assertNotIn("return self._mjData", src)
        self.assertNotIn("return self._mjModel", src)

    def test_k4_env_no_gym_private_access(self):
        """K4: env 方法不触 self._gym._xxx。"""
        src = self.env_path.read_text()
        # 提取新增方法体，断言不触 _gym._sim/_mjData/_mjModel
        forbidden_patterns = [
            r"self\._gym\._sim",
            r"self\._gym\._mjData",
            r"self\._gym\._mjModel",
        ]
        for pat in forbidden_patterns:
            matches = re.findall(pat, src)
            self.assertEqual(matches, [], f"env 穿墙访问 gym 私有: {pat}")


if __name__ == "__main__":
    unittest.main()
```

---

## 12. 依赖关系与排期

### 12.1 子步骤依赖图

```
3.1.1 (SimCore 关节查询) ─┐
3.1.2 (SimCore Body/Site) ─┤
3.1.3 (DataView 暴露)     ─┼─→ 3.1.5 (Registry 元信息) ─→ 3.1.6 (Gym 委托) ─→ 3.1.7 (Env 委托)
3.1.4 (DataView 加固)     ─┘                                                    │
                                                                                ↓
3.1.8 (基座坐标变换) ←───────────────────────────────────────────── 3.3.1 (SimCore jacBody)
                                       │                                        │
                                       ↓                                        ↓
                                 3.3.2 (SimCore jac_site) ──→ 3.3.3 (Env 雅可比委托)
                                                                              │
3.2.1 (SimCore 力应用) ─→ 3.2.2 (SimCore 状态设置) ─→ 3.2.3 (Bridge mocap)    │
                                       │                            │          │
                                       ↓                            ↓          │
                       3.2.4 (Env 力应用委托) ←───────────────────┘          │
                                       │                                     │
                                       ↓                                     ↓
                       3.2.5 (DataView xfrc 保护)              3.4.1-3.4.3 (Bridge Studio)
                                                                   │
                                                                   ↓
                                              3.4.4 (Env Studio 委托) ←── 3.5.1 (SimCore 约束)
                                                                   │              │
                                                                   ↓              ↓
                                              3.5.2 (Registry eq) ─→ 3.5.3 (Env 约束委托)
                                                                   │
                                                                   ↓
                                              3.5.4 (anchor_actor) ─→ 3.5.5 (release) ─→ 3.5.6 (do_body_manipulation)
                                                                                                   │
                                                                                                   ↓
                                                                                          §10 全局回归
```

### 12.2 推荐排期（按依赖拓扑）

| 批次 | 子步骤 | 可并行 | 前置依赖 |
|------|--------|--------|----------|
| 1 | 3.1.1, 3.1.2, 3.1.3, 3.1.4 | ✅（4 个 SimCore/DataView 子步骤独立） | 无 |
| 2 | 3.1.5 | ❌ | 3.1.1-3.1.4 |
| 3 | 3.1.6 | ❌ | 3.1.5 |
| 4 | 3.1.7 | ❌ | 3.1.6 |
| 5 | 3.3.1, 3.2.1, 3.2.2, 3.2.3, 3.4.1, 3.4.2, 3.4.3 | ✅（不同层独立） | 3.1.7 |
| 6 | 3.1.8, 3.3.2, 3.2.4, 3.4.4, 3.5.1 | ✅ | 批次 5 对应子步骤 |
| 7 | 3.3.3, 3.2.5, 3.5.2 | ✅ | 批次 6 对应子步骤 |
| 8 | 3.5.3 | ❌ | 3.5.1, 3.5.2 |
| 9 | 3.5.4 | ❌ | 3.5.3 |
| 10 | 3.5.5 | ❌ | 3.5.4 |
| 11 | 3.5.6 | ❌ | 3.5.4, 3.5.5, 3.2.4 |
| 12 | §10 全局回归 | ❌ | 全部子步骤 |

### 12.3 关键路径

关键路径（最长依赖链）：
```
3.1.1 → 3.1.5 → 3.1.6 → 3.1.7 → 3.5.1 → 3.5.3 → 3.5.4 → 3.5.5 → 3.5.6 → §10
```
共 10 个串行节点，决定整体排期下限。

---

## 13. 风险与回退

### 13.1 子步骤级回退策略

每个子步骤独立提交（一个 commit），失败时仅回退该子步骤，不影响其他已完成子步骤：

| 风险 | 影响 | 回退方案 |
|------|------|----------|
| 子步骤功能测试失败 | 该子步骤不达标 | 回退该子步骤 commit，其他子步骤不受影响 |
| 架构遵从性测试失败 | 架构被破坏 | 立即回退，修复后重提 |
| 子步骤依赖标注错误（如 3.1.8 依赖 3.3.1 未发现） | 3.1.8 无法独立完成 | 将 3.1.8 移至 3.3.1 之后，更新依赖图 |
| `do_body_manipulation` 编排逻辑复杂 | 3.5.6 难以一次完成 | 进一步拆分 3.5.6 为 3.5.6a/3.5.6b/3.5.6c |

### 13.2 架构遵从性"红线"

以下情况视为架构被破坏，必须立即回退，不得"先合入再修"：

1. `dir(env)` 出现 `_mjData`/`_mjModel`/`_sim`/`_studio` 属性
2. 公共方法返回 `mujoco.MjData`/`mujoco.MjModel` 类型
3. `orca_gym_euler_env.py` 直接写 `_mjData.xfrc_applied`/`mocap_pos`/`_mjModel.eq_*`
4. `orca_studio_bridge.py` import `MjData`/`MjModel`
5. `mujoco_sim_core.py` 出现 `return self._mjData`/`return self._mjModel`

### 13.3 测试数据风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| `g1_29dof_camera.xml` 路径变更 | 所有功能测试 fixture 失效 | fixture 路径集中管理在 `conftest.py`，一处修改全局生效 |
| ~~G1 XML 无 mocap body~~ | ~~3.5.4-3.5.6 无法测试~~ | **已解决**：`g1_29dof_camera.xml` 已内置 mocap body `ActorManipulator_Anchor`（`mocap="true"`）+ box `manipulation_box`（free joint）+ weld 等式约束 `anchor_box_weld`（绑定 anchor↔box） |
| ~~G1 XML 无等式约束~~ | ~~3.5.1-3.5.3 无法测试~~ | **已解决**：同上，XML 已含 `<equality><weld name="anchor_box_weld" .../></equality>`，测试可通过 `modify_equality_objects` 重绑 obj2id 到 G1 任意 body（如 pelvis）验证 mocap 驱动 |

> **测试 fixture 说明**（G1 XML 内置测试对象）：
>
> | 对象 | XML 名称 | 类型 | 用途 |
> |------|---------|------|------|
> | 测试 box | `manipulation_box` | body + free joint + box geom | 被驱动对象，验证 weld 约束效果 |
> | mocap anchor | `ActorManipulator_Anchor` | body (`mocap="true"`) + sphere geom | 驱动源，通过 `set_mocap_pos_and_quat` 写入位姿 |
> | weld 约束 | `anchor_box_weld` | equality (weld, active=true) | 绑定 anchor↔box，可通过 `modify_equality_objects` 重绑 obj2id 到 G1 body |
>
> **body id 参考**（mujoco 3.7.0 实测）：`manipulation_box`=32、`ActorManipulator_Anchor`=33、`pelvis`=1；`nmocap=1`、`neq=1`。测试中应通过 `mj_name2id` 解析，不硬编码 id。

---

## 14. 附录

### 14.1 子步骤验收检查清单模板

每个子步骤完成后，填写以下检查清单：

```markdown
## 子步骤 X.Y.Z 验收清单

**子步骤名称**：___
**涉及文件**：___
**完成日期**：___
**负责人**：___

### 实现完成度
- [ ] 实现内容已按文档完成
- [ ] `raise NotImplementedError` 已替换为真实实现
- [ ] 新增方法有 docstring（K12）

### 架构遵从性测试
- [ ] 架构测试已编写（套用 §11.4 模板）
- [ ] 架构测试全部通过
- [ ] grep 断言全部通过（无穿墙、无泄漏）

### 功能单元测试
- [ ] 功能测试已编写（加载 G1 XML）
- [ ] 功能测试全部通过（数值/形状/一致性）
- [ ] 离线模式 no-op 验证通过（若适用）

### 依赖管理
- [ ] 前置依赖子步骤已完成
- [ ] 本子步骤的依赖解除标注已更新（如"解除 3.1.x 的依赖"）
- [ ] 后续依赖本子步骤的子步骤可正常启动

### 文档同步
- [ ] 本子步骤的 API 已添加到公共方法清单
- [ ] 若有设计决策变更，已更新本文档
```

### 14.2 架构遵从性 grep 断言速查

| 文件 | 禁止出现 | K 约束 |
|------|---------|--------|
| `mujoco_sim_core.py` | `return self._mjData`、`return self._mjModel` | P2 |
| `orca_gym_data_view.py` | `return self._mj_model`、`return self._mj_data` | P2 |
| `model_registry.py` | `return self._mj_model` | P2 |
| `orca_gym_euler.py` | `return self._mjData`、`return self._mjModel`、`self._sim.` | P2/K4 |
| `orca_gym_euler_env.py` | `self._gym._sim`、`self._gym._mjData`、`self._gym._mjModel`、`gym.studio`、`_mjData.xfrc_applied[`、`_mjData.mocap_pos[`、`_mjModel.eq_` | K4/P4/K9 |
| `orca_studio_bridge.py` | `import mujoco`（仅 MjData/MjModel 部分）、`MjData`、`MjModel` | K9/P2 |

### 14.3 子步骤总览表

| 阶段 | 子步骤数 | 子步骤范围 | 核心目标 |
|------|---------|-----------|---------|
| 3.1 状态查询 | 8 | 3.1.1–3.1.8 | MuJoCoSimCore/DataView/Registry/Gym/Env 状态查询 + 基座坐标变换 |
| 3.2 力应用与设置 | 5 | 3.2.1–3.2.5 | SimCore 力应用/状态设置 + Bridge mocap + Env 委托 + DataView 保护 |
| 3.3 雅可比 | 3 | 3.3.1–3.3.3 | SimCore jacBody/jacSite/jac_site + Env 委托 |
| 3.4 Studio 交互 | 4 | 3.4.1–3.4.4 | Bridge 视频/帧/内容文件 + Env 委托 |
| 3.5 约束与体操作 | 6 | 3.5.1–3.5.6 | SimCore 等式约束 + Env 委托 + anchor/release/manipulation 编排 |
| §10 全局回归 | 1 | 10.1–10.4 | 跨子步骤一致性 + 端到端链路 + 依赖解除 |
| **合计** | **27** | | |
