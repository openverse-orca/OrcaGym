# OrcaGym API Reference (Markdown)

> 本文件是 **单文件 API Reference**，用于开发者快速查接口，也便于 AI 检索与引用。  
> 文档范围以 `orca_gym` 里 **相对稳定的 core + local env** 为主；远程相关内容本文件暂不展开。

## 0. 文档导航（索引/细节/手册）

本仓库的 API 文档分为三层，本文件属于“摘要 + 关键链路（Recipes）”。需要全量符号索引、完整签名与 docstring 时，跳转到下列文档：

- **API Flat（平铺索引）**：按模块平铺列出符号清单（用于快速定位名字）  
  - `../doc/api_flat/INDEX.md`（相对路径：[`../doc/api_flat/INDEX.md`](../doc/api_flat/INDEX.md)）
- **API Detail（细节参考）**：每个符号包含 **签名 + docstring（原样收录）**（用于工具无法读代码时的“离线参考”）  
  - `../doc/api_detail/INDEX.md`（相对路径：[`../doc/api_detail/INDEX.md`](../doc/api_detail/INDEX.md)）
- **API Manual（手册版）**：更偏“浏览体验”的手册式排版（用于人工阅读）  
  - `../doc/api_manual/INDEX.md`（相对路径：[`../doc/api_manual/INDEX.md`](../doc/api_manual/INDEX.md)）

## 目录

- [0. 文档导航（索引/细节/手册）](#0-文档导航索引细节手册)
- [1. 总览](#1-总览)
- [2. 关键概念与术语](#2-关键概念与术语)
- [3. 顶层公共入口 (`orca_gym`)](#3-顶层公共入口-orca_gym)
- [4. Core API](#4-core-api)
  - [4.1 `OrcaGymBase` (gRPC 基础封装)](#41-orcagymbase-grpc-基础封装)
  - [4.2 `OrcaGymModel` (静态模型信息)](#42-orcagymmodel-静态模型信息)
  - [4.3 `OrcaGymData` (动态仿真状态)](#43-orcagymdata-动态仿真状态)
  - [4.4 `OrcaGymOptConfig` (MuJoCo opt 配置)](#44-orcagymoptconfig-mujoco-opt-配置)
  - [4.5 `OrcaGymLocal` (本地 MuJoCo backend)](#45-orcagymlocal-本地-mujoco-backend)
- [5. Environment API](#5-environment-api)
  - [5.1 `OrcaGymBaseEnv` (Gymnasium 基类)](#51-orcagymbaseenv-gymnasium-基类)
  - [5.2 `OrcaGymLocalEnv` (本地环境实现)](#52-orcagymlocalenv-本地环境实现)
- [6. 典型调用链（Recipes）](#6-典型调用链recipes)
  - [6.1 环境初始化与 reset 链](#61-环境初始化与-reset-链)
  - [6.2 step 链：action → ctrl → 物理步进 → data 刷新](#62-step-链action--ctrl--物理步进--data-刷新)
  - [6.3 修改状态链：set_qpos/qvel 后为什么要 mj_forward](#63-修改状态链set_qposqvel-后为什么要-mj_forward)
  - [6.4 渲染链：什么时候会触发 UI/交互控制](#64-渲染链什么时候会触发-ui交互控制)
  - [6.5 抓取/锚定链：mocap + 等式约束（WELD/CONNECT）](#65-抓取锚定链mocap--等式约束weldconnect)
  - [6.6 观测构建链：从 data/model/sensor 拼观测](#66-观测构建链从-datamodelsensor-拼观测)
  - [6.7 接触/外力链：contact → contact_force/cfrc_ext](#67-接触外力链contact--contact_forcecfrc_ext)
  - [6.8 常见“数据不同步”问题自检表](#68-常见数据不同步问题自检表)

---

## 1. 总览

OrcaGym 的核心形态是：**Gymnasium 环境（Python）** 驱动 **MuJoCo 仿真**，并通过 **gRPC** 与 OrcaSim 服务端通信。

最常见的对象关系：

- `env: OrcaGymLocalEnv`：Gymnasium 环境对象，训练代码调用 `reset/step/render`
- `env.gym: OrcaGymLocal`：本地 backend（MuJoCo 模型、数据、查询/控制接口）
- `env.model: OrcaGymModel`：模型静态信息（body/joint/actuator/site/sensor/等式约束等）
- `env.data: OrcaGymData`：仿真动态状态（qpos/qvel/qacc/time/qfrc_bias 等）
- `env.gym.opt: OrcaGymOptConfig`：MuJoCo `opt` 配置快照

---

## 2. 关键概念与术语

- **qpos / qvel / qacc**：关节位置 / 速度 / 加速度（MuJoCo 的广义坐标体系）。
- **执行器（Actuator）**：由动作（action/ctrl）驱动，施加力/扭矩/位置控制等。
- **动作空间（Action Space）**：智能体输出的控制向量允许范围（通常由 actuator `CtrlRange` 构造）。
- **观测空间（Observation Space）**：智能体能看到的状态集合（numpy array 或 dict of arrays）。
- **等式约束（Equality Constraint）**：MuJoCo 里强制两个对象满足某种“等式关系”的约束（如 WELD/CONNECT）。
- **Mocap Body**：可通过直接设置位姿来“操控/牵引”的特殊 body，常配合等式约束实现抓取/拖拽。
- **frame_skip**：Gym `step()` 对应的 MuJoCo 物理步数（一个策略动作会推进多个物理步）。

### 2.1 数组维度约定（AI/开发者查阅友好）

在 OrcaGym 里，很多 API 直接使用 MuJoCo 的维度定义：

- `model.nq`：`qpos` 的长度
- `model.nv`：`qvel/qacc` 的长度（自由度数）
- `model.nu`：`ctrl/action` 的长度（执行器数）

常见 shape：

- `qpos`: `(nq,)`
- `qvel`: `(nv,)`
- `qacc`: `(nv,)`
- `qfrc_bias`: `(nv,)`
- `ctrl`: `(nu,)`
- `xpos`: `(N*3,)`（N 个 body 的 xyz 拼接并 flatten）
- `xmat`: `(N*9,)`（N 个 3x3 旋转矩阵按行展开并 flatten）
- `xquat`: `(N*4,)`

---

## 3. 顶层公共入口 (`orca_gym`)

文件：`../orca_gym/__init__.py`

公开导出（`__all__`）：

- `OrcaGymBase`
- `OrcaGymModel`
- `OrcaGymData`
- `OrcaGymOptConfig`
- `OrcaGymLocal`
- `mjc_message_pb2` / `mjc_message_pb2_grpc`（protobuf 自动生成接口，通常仅框架内部用）

---

## 4. Core API

### 4.1 `OrcaGymBase` (gRPC 基础封装)

文件：`../orca_gym/core/orca_gym.py`

用途：

- 作为 local/remote backend 的基类，封装最基础的 gRPC 调用与 model/opt/data 指针。

常用属性：

- `stub`：gRPC stub
- `model`：`OrcaGymModel`
- `opt`：`OrcaGymOptConfig`
- `data`：`OrcaGymData`

常用方法（节选）：

- `pause_simulation()`：将仿真切到 PAUSED（OrcaGym 采用“被动模式”，由 Gym step 驱动）
- `set_qpos(qpos)` / `set_qvel(qvel)`：设置状态
- `mj_forward()` / `mj_inverse()` / `mj_step(nstep)`：MuJoCo 核心计算/步进（远端调用版本）

#### 4.1.1 方法字典（更细）

- `__init__(stub)`
  - **目的**：保存 `stub`，并初始化 `model/opt/data` 指针（由子类填充）
- `pause_simulation()`
  - **目的**：设置服务端仿真状态为 `PAUSED`
  - **注意**：OrcaGym 采用被动模式，通常在 Env 初始化时调用
- `set_qpos(qpos)` / `set_qvel(qvel)`
  - **qpos/qvel**：应满足上面的维度约定
- `mj_step(nstep)`
  - **nstep**：MuJoCo 物理步数
  - **注意**：这是“远端调用”版本；本地 backend 会直接 `mujoco.mj_step`

---

### 4.2 `OrcaGymModel` (静态模型信息)

文件：`../orca_gym/core/orca_gym_model.py`

用途：

- 保存静态模型信息：body/joint/geom/site/actuator/sensor/等式约束/mocap 映射等。
- 提供“name ↔ id”的映射，以及常见属性查询（例如 actuator ctrl range）。

常见用法（动作空间生成）：

- 从 `model.get_actuator_ctrlrange()` 得到边界，交给环境生成 `action_space`。

关键词：

- **等式约束**：`init_eq_list` / `get_eq_list`
- **执行器**：`init_actuator_dict` / `actuator_name2id` / `get_actuator_ctrlrange`

#### 4.2.1 常见字段（概念上，非完整列表）

- `nq/nv/nu`：维度信息（影响 `qpos/qvel/ctrl` 的长度）
- `neq`：等式约束数量
- `*_dict`：body/joint/actuator/site/sensor 等字典（名称为 key）

#### 4.2.2 常用方法（按用途分组）

- **初始化（由 backend 在 init 时填充）**
  - `init_eq_list(eq_list)`
  - `init_mocap_dict(mocap_dict)`
  - `init_actuator_dict(actuator_dict)`
  - `init_body_dict(body_dict)`
  - `init_joint_dict(joint_dict)`
  - `init_geom_dict(geom_dict)`
  - `init_site_dict(site_dict)`
  - `init_sensor_dict(sensor_dict)`

- **执行器（Action/ctrl 相关）**
  - `actuator_name2id(actuator_name)`：名称→执行器 id
  - `get_actuator_ctrlrange()`：返回 `(nu, 2)` 的边界数组（用于动作空间）

- **Body / Joint / Site / Sensor 名称查询**
  - `get_body_names()`：返回 body 名称集合
  - `body_name2id(body_name)`：名称→body id
  - 以及 joint/site/sensor 的类似接口（用于从名字定位索引/地址）

---

### 4.3 `OrcaGymData` (动态仿真状态)

文件：`../orca_gym/core/orca_gym_data.py`

用途：

- 保存仿真动态状态（本地缓存），由 `env.gym.update_data()` 同步更新。

核心字段：

- `qpos` / `qvel` / `qacc`
- `qfrc_bias`：重力/科氏力等被动力
- `time`

核心方法：

- `update_qpos_qvel_qacc(qpos, qvel, qacc)`
- `update_qfrc_bias(qfrc_bias)`

#### 4.3.1 使用建议

- **读状态**：建议使用 `copy()`，避免引用被后续 `update_data()` 覆盖
- **刷新时机**：在 `mj_step/do_simulation` 后调用 `gym.update_data()`，再读取 `env.data`

---

### 4.4 `OrcaGymOptConfig` (MuJoCo opt 配置)

文件：`../orca_gym/core/orca_gym_opt_config.py`

用途：

- MuJoCo `opt` 的配置快照（timestep/solver/iterations/contact 参数等）。
- 常用于解释 `env.dt`、控制频率、稳定性/性能权衡。

常见字段（节选）：

- `timestep`
- `solver`, `iterations`
- `gravity`
- `o_margin`, `o_solref`, `o_solimp`, `o_friction`

#### 4.4.1 与环境时间的关系

- `env.dt = env.gym.opt.timestep * env.frame_skip`
- 策略侧控制频率可按 `control_hz = 1 / env.dt` 计算。

---

### 4.5 `OrcaGymLocal` (本地 MuJoCo backend)

文件：`../orca_gym/core/orca_gym_local.py`

用途：

- 管理本地 MuJoCo 的 `MjModel/MjData`，并通过一层封装提供：
  - 模型加载与资源下载（XML/mesh/hfield）
  - 物理步进与数据同步
  - 执行器控制
  - body/joint/site/sensor 查询
  - 接触力、外力、雅可比、质量矩阵等动力学能力
  - mocap 与等式约束相关操作

#### 4.5.1 关键辅助定义

- `get_qpos_size(joint_type)`：不同关节类型在 `qpos` 里占用元素数
- `get_dof_size(joint_type)`：不同关节类型自由度（`qvel` 维度）
- `AnchorType`：`NONE / WELD / BALL`
- `CaptureMode`：`ASYNC / SYNC`
- `get_eq_type(anchor_type)`：将 `AnchorType` 映射为 MuJoCo 等式约束类型

#### 4.5.2 初始化与模型加载

- `load_model_xml()`：从服务端获取 XML，并下载依赖资源
- `init_simulation(model_xml_path)`：创建 `MjModel/MjData`，并初始化 `opt/model/data`

#### 4.5.3 仿真控制与同步

- `set_ctrl(ctrl)`：设置控制输入（支持 UI 侧 override）
- `mj_step(nstep)`：推进物理步
- `mj_forward()`：前向更新（常在修改状态后调用）
- `update_data()`：把 `_mjData` 同步到 `self.data`（`OrcaGymData`）
- `load_initial_frame()`：重置到初始状态

#### 4.5.4 常用查询（按类别）

- **模型结构查询（初始化时也会调用）**
  - `query_model_info()`
  - `query_all_bodies()` / `query_all_joints()` / `query_all_actuators()` / `query_all_sites()` / `query_all_sensors()`
  - `query_all_equality_constraints()` / `query_all_mocap_bodies()`

- **位姿/传感器**
  - `query_body_xpos_xmat_xquat(body_name_list)`
  - `query_sensor_data(sensor_names)`
  - `query_joint_qpos(joint_names)` / `query_joint_qvel(joint_names)` / `query_joint_qacc(joint_names)`

- **动力学/控制**
  - `mj_fullM()`：质量矩阵
  - `mj_jacBody(jacp, jacr, body_id)` / `mj_jacSite(jacp, jacr, site_id)` / `mj_jac_site(site_names)`
  - `query_actuator_torques(actuator_names)`

- **接触**
  - `query_contact_simple()`
  - `query_contact_force(contact_ids)`
  - `get_cfrc_ext()`

- **抓取/操控（mocap + 等式约束）**
  - `modify_equality_objects(old_obj1_id, old_obj2_id, new_obj1_id, new_obj2_id)`
  - `update_equality_constraints(constraint_list)`
  - `set_mocap_pos_and_quat(mocap_data, send_remote=False)`

#### 4.5.5 方法字典（建议作为“查接口”主入口）

按主题分组如下：

##### A) 资源/模型加载与缓存

- `async load_model_xml()`
  - **目的**：向服务端请求 XML 文件名/内容，并确保 mesh/hfield 等资源在本地缓存目录存在
  - **副作用**：会在 `~/.orcagym/tmp` 写入缓存文件（通过文件锁避免多进程冲突）

- `async init_simulation(model_xml_path)`
  - **目的**：创建 `mujoco.MjModel/MjData`，并构造 `self.opt/self.model/self.data`
  - **副作用**：会调用 `query_*` 系列初始化 model 信息，并 `update_data()` 同步初始状态

- `xml_file_dir`（property）
  - **目的**：本地缓存目录（默认 `~/.orcagym/tmp`）

##### B) opt / timestep

- `set_time_step(time_step)`
  - **目的**：更新 `_timestep` 并写入 `mjModel.opt.timestep`

- `set_opt_config()`
  - **目的**：把 `self.opt` 的各字段写回本地 `mjModel.opt`（常在加载后统一配置）

- `query_opt_config()`
  - **返回**：dict（可用于构造 `OrcaGymOptConfig` 或打印/对比）

##### C) 物理步进与数据同步（最常用调用链）

推荐调用链（核心规律）：

- **控制**：`set_ctrl(ctrl)`
- **推进**：`mj_step(nstep)`
- **同步**：`update_data()` → 之后读 `self.data.*`

对应方法：

- `set_ctrl(ctrl)`
  - **ctrl**：长度 `nu` 的向量
  - **注意**：如果 UI/渲染侧返回了 `override_ctrls`，会覆盖部分维度

- `mj_step(nstep)`
  - **nstep**：物理步数（在 env 层通常等于 `frame_skip` 或 `decimation`）

- `update_data()`
  - **目的**：将 `_mjData.{qpos,qvel,qacc,qfrc_bias,time}` 同步到 `self.data`
  - **注意**：若仅修改 `_mjData` 而未调用该方法，`self.data` 仍可能保持旧值

##### D) 状态设置（qpos/qvel）与一致性

- `set_joint_qpos(joint_qpos: dict)`
- `set_joint_qvel(joint_qvel: dict)`
  - **注意**：修改完状态通常需要 `mj_forward()` 再读取位姿/传感器

- `mj_forward()`
  - **目的**：刷新运动学/传感器等派生量（避免 NaN/不一致）

- `load_initial_frame()`
  - **目的**：reset 到初始状态（MuJoCo `mj_resetData`）

##### E) 常用查询（观测/奖励/控制常用）

- `query_joint_qpos(joint_names)` / `query_joint_qvel(joint_names)` / `query_joint_qacc(joint_names)`
  - **返回**：`{joint_name: np.ndarray}`（不同关节类型长度不同）

- `query_body_xpos_xmat_xquat(body_name_list)`
  - **返回**：`{body_name: {'Pos','Mat','Quat'}}`

- `query_sensor_data(sensor_names)`
  - **返回**：`{sensor_name: np.ndarray}`

##### F) 动力学常用接口（高级用户）

- `mj_fullM()`
  - **返回**：质量矩阵 `M`（形状 `(nv, nv)`）

- `mj_jacBody(jacp, jacr, body_id)` / `mj_jacSite(jacp, jacr, site_id)`
  - **输入**：预分配数组 `jacp/jacr`（形状 `(3, nv)`）
  - **目的**：避免在循环中频繁分配内存

- `query_actuator_torques(actuator_names)`
  - **返回**：`{actuator_name: torque_vector}`（6 维向量，具体填充与关节类型相关）

##### G) 接触与外力（debug/奖励常用）

- `query_contact_simple()`
  - **返回**：当前接触点列表（几何体 id 对）

- `query_contact_force(contact_ids)`
  - **返回**：`{contact_id: force6d}`（`[fx, fy, fz, mx, my, mz]`）

- `get_cfrc_ext()`
  - **返回**：每个 body 的外部约束力（形状 `(nbody, 6)`）

##### H) 抓取/操控（mocap + 等式约束）

OrcaGym 常见抓取套路：

1. 找到 anchor/mocap body（模型中通常名为 `ActorManipulator_Anchor`）
2. 更新 mocap 位姿：`set_mocap_pos_and_quat(...)`
3. 更新等式约束：`modify_equality_objects(...)` + `update_equality_constraints(...)`

对应方法：

- `set_mocap_pos_and_quat(mocap_data, send_remote=False)`
- `modify_equality_objects(old_obj1_id, old_obj2_id, new_obj1_id, new_obj2_id)`
- `update_equality_constraints(constraint_list)`

---

## 5. Environment API

### 5.1 `OrcaGymBaseEnv` (Gymnasium 基类)

文件：`../orca_gym/environment/orca_gym_env.py`

用途：

- 所有 OrcaGym 环境的共同基类，提供：
  - gRPC 初始化（`initialize_grpc`）
  - reset/seed 通用流程
  - `action_space`/`observation_space` 的生成工具
  - agent name 前缀工具（`body/joint/actuator/site/mocap/sensor`）

需要子类实现的方法（抽象契约）：

- `step(action)`
- `reset_model()`
- `initialize_simulation()`
- `_step_orca_sim_simulation(ctrl, n_frames)`
- `render()`

常用工具方法：

- `generate_action_space(bounds)`
- `generate_observation_space(obs)`

#### 5.1.1 BaseEnv 的“你一定会踩的点”

- `reset()`：会调用 `reset_simulation()` → `reset_model()` → `render()`
- `dt`（property）：返回 `gym.opt.timestep * frame_skip`
- 命名辅助：`body/joint/actuator/site/mocap/sensor(name, agent_id=None)` 会自动加 agent 前缀（多 agent 时很重要）

---

### 5.2 `OrcaGymLocalEnv` (本地环境实现)

文件：`../orca_gym/environment/orca_gym_local_env.py`

用途：

- 绑定 `OrcaGymLocal` 作为 `self.gym`，提供本地 gRPC 环境能力。

关键流程：

- `initialize_grpc()`：创建 `grpc.aio` channel + stub，并构造 `OrcaGymLocal`
- `initialize_simulation()`：加载模型 XML 并 init backend
- `do_simulation(ctrl, n_frames)`：环境 `step()` 的核心（设置 ctrl、步进、更新 data）

常用接口（节选）：

- `set_ctrl(ctrl)` / `mj_step(nstep)` / `mj_forward()`
- `get_body_xpos_xmat_xquat(body_name_list)`
- `query_site_pos_and_quat(site_names)`
- `set_mocap_pos_and_quat(mocap_pos_and_quat_dict)`

#### 5.2.1 方法字典（环境视角）

##### A) 初始化与关闭

- `initialize_grpc()`
  - **目的**：创建 `grpc.aio` channel → `GrpcServiceStub` → `OrcaGymLocal`

- `initialize_simulation() -> (model, data)`
  - **目的**：通过 `gym.load_model_xml()` 和 `gym.init_simulation()` 初始化 backend

- `close()`
  - **目的**：关闭 gRPC channel

##### B) step 的核心：`do_simulation`

- `do_simulation(ctrl, n_frames) -> None`
  - **目的**：对外提供“设置控制 + 推进 n_frames + 同步数据”的原子操作
  - **校验**：会检查 `ctrl` 维度是否为 `(nu,)`（不匹配会抛 `ValueError`）
  - **副作用**：内部会调用 `self.gym.update_data()`，使 `env.data` 变为最新

##### C) 渲染与交互（锚点操控依赖渲染触发）

- `render()`
  - **注意**：部分“场景交互/锚点操作”逻辑在渲染时触发（见 `do_body_manipulation()`）

- `anchor_actor(actor_name, anchor_type)` / `release_body_anchored()`
  - **目的**：把某个 actor 通过等式约束与 anchor 连接或释放

##### D) 查询与工具方法（Env 转发到 backend）

- `set_ctrl` / `mj_step` / `mj_forward`
- `get_body_xpos_xmat_xquat`
- `query_site_pos_and_quat` / `query_site_size`
- `query_contact_force` / `get_cfrc_ext`
- `query_velocity_body_B` / `query_robot_*_odom`（机器人学常用“基座系/里程计系”）

---

## 6. 典型调用链（Recipes）

本节是“开发者真正怎么把这些 API 串起来”的最小闭环，重点解决：

- **什么时候数据是新的？什么时候必须 refresh？**
- **reset/step/render 真实会发生什么？**
- **抓取/交互到底是哪几步？**

### 6.1 环境初始化与 reset 链

典型顺序（以 `OrcaGymLocalEnv` 为例）：

```
OrcaGymLocalEnv.__init__
  -> OrcaGymBaseEnv.__init__
      -> initialize_grpc()
      -> pause_simulation()
      -> set_time_step()
      -> initialize_simulation()   # load_model_xml + init_simulation
      -> reset_simulation()
      -> init_qpos_qvel()
```

随后训练代码通常调用：

```
env.reset()
  -> reset_simulation()
  -> reset_model()         # 子类实现：设置初始关节/随机化等
  -> render()
```

参考：

- `orca_gym/environment/orca_gym_env.py`：`OrcaGymBaseEnv.__init__ / reset`
- `orca_gym/environment/orca_gym_local_env.py`：`initialize_grpc / initialize_simulation`

最小代码片段（直接用 Gymnasium 初始化 + reset）：

```python
import gymnasium as gym

# 使用已注册的 env_id（例如任务环境 ID）
env = gym.make(
    "YOUR_ENV_ID",
    frame_skip=5,
    orcagym_addr="localhost:50051",
    agent_names=["agent0"],
    time_step=0.001,
    render_mode="none",
)

obs, info = env.reset()
print("obs keys/type:", type(obs), getattr(obs, "keys", lambda: None)())
```

### 6.2 step 链：action → ctrl → 物理步进 → data 刷新

最常见的“正确链路”是：

```
action (policy 输出)
  -> env.set_ctrl(ctrl) 或 env.do_simulation(ctrl, n_frames)
  -> (内部) gym.mj_step(nstep)
  -> (内部) gym.update_data()
  -> 读取 env.data / env.gym.data 生成 obs、reward
```

核心原则：

- **只要推进了物理（`mj_step/do_simulation`），就要同步一次 `update_data()`**，再读 `data`。
- `OrcaGymLocalEnv.do_simulation()` 已经在末尾调用了 `self.gym.update_data()`，所以使用它最不容易“读到旧状态”。

参考：

- `orca_gym/environment/orca_gym_local_env.py`：`do_simulation`

最小代码片段（标准 Gym step loop）：

```python
import numpy as np
import gymnasium as gym

env = gym.make(
    "YOUR_ENV_ID",
    frame_skip=5,
    orcagym_addr="localhost:50051",
    agent_names=["agent0"],
    time_step=0.001,
    render_mode="none",
)
obs, info = env.reset()

for _ in range(100):
    # 最稳妥：从 action_space 采样，保证 shape/范围正确
    action = env.action_space.sample().astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### 6.3 修改状态链：set_qpos/qvel 后为什么要 mj_forward

当直接修改状态（例如 reset、mocap 更新、关节设定）后：

```
set_joint_qpos / set_joint_qvel / set_mocap_pos_and_quat
  -> mj_forward()
  -> (可选) update_data()  # 若后续需要从 env.data 读取
```

原因：

- MuJoCo 有很多“派生量”（site/body 位姿、传感器、接触等）需要 `mj_forward` 才会一致。
- 只改 `qpos/qvel` 不 forward，后续查询位姿/传感器很容易出现 **NaN/旧值/不一致**。

参考：

- `orca_gym/environment/orca_gym_local_env.py`：初始化时会 `mj_forward()`（避免 NaN）
- `orca_gym/core/orca_gym_local.py`：`mj_forward / update_data`

最小代码片段（直接改关节位置 → forward → 读取位姿）：

```python
import numpy as np

# 前置：已有 env（OrcaGymLocalEnv），并需将部分关节设置到目标位置
# joint_qpos = {"hip_joint": np.array([0.1], dtype=np.float64), ...}
joint_qpos = {}

env.set_joint_qpos(joint_qpos)
env.mj_forward()     # 关键：刷新派生量（body/site 位姿、传感器等）
env.update_data()    # 如果后续从 env.data 读（可选但常用）

# 之后再读 env.data / 查询位姿才更可靠
# print(env.data.qpos.copy())
```

### 6.4 渲染链：什么时候会触发 UI/交互控制

本地环境常见现象：**一些交互（例如 anchor 操作、UI 覆盖控制）在渲染时才会被处理**。

典型链路：

```
env.render()
  -> gym.render()                    # 发送 qpos/time 到服务端渲染
  -> (返回) override_ctrls           # UI 侧可能覆盖部分执行器控制
  -> env.do_body_manipulation()      # 一些交互逻辑在这里处理
```

因此，当依赖“场景交互”但 `render_mode` 非 human/force，或渲染频率过低时，可能出现：

- UI 操作没生效
- override 控制没进入 `set_ctrl`

参考：

- `orca_gym/core/orca_gym_local.py`：`render/update_local_env/override_ctrls`
- `orca_gym/environment/orca_gym_local_env.py`：`render/do_body_manipulation`

最小代码片段（human 渲染 + 让 UI 覆盖控制生效）：

```python
import time
import gymnasium as gym

env = gym.make(
    "YOUR_ENV_ID",
    frame_skip=5,
    orcagym_addr="localhost:50051",
    agent_names=["agent0"],
    time_step=0.001,
    render_mode="human",
)
env.reset()

# 在 human 模式下周期 render，UI 侧的交互/override 才会被处理并回传
for _ in range(200):
    env.render()
    time.sleep(env.dt)
```

### 6.5 抓取/锚定链：mocap + 等式约束（WELD/CONNECT）

OrcaGym 的抓取/锚定通常是“两件事”的组合：

- **mocap**：驱动锚点（anchor）的位姿
- **等式约束**：把“锚点”和“目标物体”用 WELD/CONNECT 连接起来

典型步骤（概念链路）：

```
1) 识别目标 actor / anchor body
2) 更新 mocap 位姿：set_mocap_pos_and_quat(...)
3) 选择约束类型：AnchorType.WELD / AnchorType.BALL
4) 更新约束对象与参数：
   - modify_equality_objects(...)
   - update_equality_constraints(eq_list)
5) 必要时 mj_forward() / update_data()
```

注意点：

- **WELD**：固定位置+姿态；**CONNECT(BALL)**：固定位置、允许旋转（更像球关节）
- 多数情况下“锚点交互”在 render 过程中被触发（见 6.4）

参考：

- `orca_gym/core/orca_gym_local.py`：`AnchorType/get_eq_type/modify_equality_objects/update_equality_constraints/set_mocap_pos_and_quat`
- `orca_gym/environment/orca_gym_local_env.py`：`anchor_actor/release_body_anchored/update_anchor_equality_constraints`

可复制伪代码（依赖模型里存在 anchor body/eq/mocap 定义；通常名字类似 `ActorManipulator_Anchor`）：

```python
import numpy as np
from orca_gym.core.orca_gym_local import AnchorType

# 1) 选择目标 actor（物体 body 名称），以及锚点类型
actor_name = "SomeObjectBodyName"
anchor_type = AnchorType.WELD  # 或 AnchorType.BALL

# 2) 让环境层完成“锚定逻辑”（会处理 eq_list、obj id 等转换）
env.anchor_actor(actor_name, anchor_type)

# 3) 通过 mocap 移动锚点（驱动物体跟随）
# 注意：mocap 名称需与模型一致；多数场景会在 env 内部封装好 anchor body 名称
env.set_mocap_pos_and_quat({
    env._anchor_body_name: {  # 这是 OrcaGymLocalEnv 里的默认锚点名（模型里需存在）
        "pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
        "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }
})

# 4) 刷新（避免派生量不一致）
env.mj_forward()
env.update_data()
```

### 6.6 观测构建链：从 data/model/sensor 拼观测

自定义环境时，常见的观测来源包括：

- `env.data.qpos/qvel/qacc/qfrc_bias/time`
- `env.model` 的结构信息（关节列表、索引/adr、执行器范围等）
- `env.query_sensor_data(...)` 或 backend 的 `query_sensor_data(...)`

建议的构建流程：

```
do_simulation(...) / update_data()
  -> 读取 data（qpos/qvel...）
  -> (可选) 读取传感器：query_sensor_data(...)
  -> 拼成 obs（np.ndarray 或 Dict[str, np.ndarray]）
  -> 第一次 reset 时用 obs 推断 observation_space（generate_observation_space）
```

参考：

- `orca_gym/environment/orca_gym_env.py`：`generate_observation_space`

最小代码片段（从 data 拼一个 numpy 观测）：

```python
import numpy as np

# 假设 env 已经过 do_simulation/step 且 data 已更新
qpos = env.data.qpos.copy()
qvel = env.data.qvel.copy()

# 最简观测：拼接 qpos + qvel（实际项目通常会做归一化/裁剪）
obs = np.concatenate([qpos, qvel]).astype(np.float32)
```

### 6.7 接触/外力链：contact → contact_force/cfrc_ext

调试/奖励中常见链路：

```
update_data()
  -> query_contact_simple()           # 先拿有哪些接触
  -> query_contact_force(contact_ids) # 再拿接触力
  -> get_cfrc_ext()                   # 或直接看每个 body 的外部约束力
```

注意点：

- `query_contact_force` 需要接触 id；通常先用 `query_contact_simple` 枚举
- `cfrc_ext` 维度是 `(nbody, 6)`，用于快速看“哪个 body 受力异常”

参考：

- `orca_gym/core/orca_gym_local.py`：`query_contact_simple/query_contact_force/get_cfrc_ext`
- `orca_gym/environment/orca_gym_local_env.py`：对应的转发接口

最小代码片段（列出接触对并打印接触力）：

```python
# 确保已 do_simulation/step 且 data 已更新
contacts = env.query_contact_simple()
contact_ids = [c["ID"] for c in contacts]

if contact_ids:
    forces = env.query_contact_force(contact_ids)
    first_id = contact_ids[0]
    print("first contact force6d:", forces[first_id])

# 或者直接读每个 body 的外力（用于快速定位异常）
cfrc_ext = env.get_cfrc_ext()
print("cfrc_ext shape:", cfrc_ext.shape)
```

### 6.8 常见“数据不同步”问题自检表

若出现“读到旧状态 / 观测跳变 / 位姿不对”，优先检查：

- 是否在 `mj_step/do_simulation` 后 **调用了 `update_data()`**？
- 是否在修改 `qpos/qvel/mocap` 后 **调用了 `mj_forward()`**？
- 读取 `env.data.qpos` 时是否遗漏 `copy()`，导致后续被覆盖？
- 是否在多线程/多进程环境中并发读写同一 env（不推荐）？

---
