# OrcaGym API Manual: `orca_gym/core`

> **📖 这是什么文档？**  
> 这是 `orca_gym/core` 模块的完整 API 参考手册，采用“索引 + 详情”的版式设计，便于快速查找和深入学习。

## 📚 文档说明

### 文档特点

- **索引优先**：每个模块和类都提供索引表格，方便快速浏览和定位
- **详情展开**：点击或展开详情部分，查看完整的方法签名、参数说明和使用示例
- **面向本地环境**：本手册主要覆盖本地环境实现，远程环境相关内容已省略
- **仅公开接口**：只列出 public 符号（不以下划线开头），聚焦实际可用的 API

### 如何使用本手册

1. **快速查找**：使用下方的模块索引表格，找到你需要的模块
2. **浏览类列表**：进入模块后，先看“Classes（索引）”表格，了解有哪些类
3. **查看方法**：每个类都有“方法索引”表格，快速了解可用方法
4. **深入阅读**：展开“方法详情”部分，查看完整的签名、参数说明和使用示例

### 相关文档

- **快速概览**：查看 [`API_REFERENCE.md`](../API_REFERENCE.md) 了解整体架构和典型调用链
- **详细参考**：查看 [`api_detail/core.md`](../api_detail/core.md) 获取自动生成的完整 API 签名列表
- **Environment 模块**：查看 [`api_manual/environment.md`](environment.md) 了解环境层接口

---

## 📦 Modules（索引）

快速浏览所有模块，点击模块名跳转到详细内容：

| Module | 说明 |
| --- | --- |
| [`orca_gym/core/orca_gym_local.py`](#orca_gymcoreorca_gym_localpy) | **本地 MuJoCo Backend**：本地 MuJoCo 仿真引擎的核心实现（最常用） |
| [`orca_gym/core/orca_gym_model.py`](#orca_gymcoreorca_gym_modelpy) | **模型信息**：静态模型信息封装，提供 body/joint/actuator 等查询接口 |
| [`orca_gym/core/orca_gym_data.py`](#orca_gymcoreorca_gym_datapy) | **仿真数据**：动态仿真状态封装，包含 qpos/qvel/qacc 等状态 |
| [`orca_gym/core/orca_gym_opt_config.py`](#orca_gymcoreorca_gym_opt_configpy) | **优化配置**：MuJoCo 仿真器优化参数配置（timestep/solver 等） |
| [`orca_gym/core/orca_gym.py`](#orca_gymcoreorca_gympy) | **基础封装**：gRPC 基础封装，提供远程调用的基类接口 |

---

## `orca_gym/core/orca_gym.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymBase` | OrcaGymBase class |

### Classes（详情）

#### `class OrcaGymBase`

> OrcaGymBase class

<details>
<summary>Class docstring</summary>

OrcaGymBase class

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymBase.pause_simulation` | `async def pause_simulation(self)` | _No docstring._ |  |
| `OrcaGymBase.print_opt_config` | `def print_opt_config(self)` | _No docstring._ |  |
| `OrcaGymBase.print_model_info` | `def print_model_info(self, model_info)` | _No docstring._ |  |
| `OrcaGymBase.set_qpos` | `async def set_qpos(self, qpos)` | _No docstring._ |  |
| `OrcaGymBase.mj_forward` | `async def mj_forward(self)` | _No docstring._ |  |
| `OrcaGymBase.mj_inverse` | `async def mj_inverse(self)` | _No docstring._ |  |
| `OrcaGymBase.mj_step` | `async def mj_step(self, nstep)` | _No docstring._ |  |
| `OrcaGymBase.set_qvel` | `async def set_qvel(self, qvel)` | _No docstring._ |  |

##### 方法详情

###### `OrcaGymBase.pause_simulation`

Signature:

```python
async def pause_simulation(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymBase.print_opt_config`

Signature:

```python
def print_opt_config(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymBase.print_model_info`

Signature:

```python
def print_model_info(self, model_info)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymBase.set_qpos`

Signature:

```python
async def set_qpos(self, qpos)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymBase.mj_forward`

Signature:

```python
async def mj_forward(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymBase.mj_inverse`

Signature:

```python
async def mj_inverse(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymBase.mj_step`

Signature:

```python
async def mj_step(self, nstep)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymBase.set_qvel`

Signature:

```python
async def set_qvel(self, qvel)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>

---

## `orca_gym/core/orca_gym_data.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymData` | _No docstring._ |

### Classes（详情）

#### `class OrcaGymData`

<details>
<summary>Class docstring</summary>

_No docstring._

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymData.update_qpos_qvel_qacc` | `def update_qpos_qvel_qacc(self, qpos, qvel, qacc)` | 更新关节位置、速度和加速度数据 |  |
| `OrcaGymData.update_qfrc_bias` | `def update_qfrc_bias(self, qfrc_bias)` | 更新关节偏置力数据 |  |

##### 方法详情

###### `OrcaGymData.update_qpos_qvel_qacc`

Signature:

```python
def update_qpos_qvel_qacc(self, qpos, qvel, qacc)
```

<details>
<summary>Docstring</summary>

更新关节位置、速度和加速度数据

通常在从服务器同步数据后调用，用于更新本地数据副本。
这些数据用于构建观测空间和计算奖励。

术语说明:
    - 观测空间 (Observation Space): 强化学习中智能体能够观察到的状态信息集合
    - 奖励 (Reward): 强化学习中用于评估动作好坏的标量信号

使用示例:
    ```python
    # 从服务器获取最新状态后更新
    self.gym.update_data()  # 从服务器同步
    self.data.update_qpos_qvel_qacc(
        self.gym.data.qpos,
        self.gym.data.qvel,
        self.gym.data.qacc
    )
    ```

</details>


###### `OrcaGymData.update_qfrc_bias`

Signature:

```python
def update_qfrc_bias(self, qfrc_bias)
```

<details>
<summary>Docstring</summary>

更新关节偏置力数据

术语说明:
    - 偏置力 (Bias Force): 包括重力、科里奥利力、离心力等被动力
    - 科里奥利力 (Coriolis Force): 由于物体在旋转参考系中运动产生的惯性力
    - 动力学计算: 根据力和力矩计算物体的加速度和运动状态

使用示例:
    ```python
    # 更新偏置力（通常由服务器计算）
    self.data.update_qfrc_bias(self.gym.data.qfrc_bias)
    ```

</details>

---

## `orca_gym/core/orca_gym_local.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `AnchorType` | 锚点类型枚举 |
| `CaptureMode` | 视频捕获模式枚举 |
| `OrcaGymLocal` | OrcaGym 本地仿真接口 |

### Functions（索引）

| Function | 签名 | 摘要 |
| --- | --- | --- |
| `get_qpos_size` | `def get_qpos_size(joint_type)` | 获取关节在 qpos 数组中的元素数量 |
| `get_dof_size` | `def get_dof_size(joint_type)` | 获取关节的自由度数量（在 qvel 数组中的元素数量） |
| `get_eq_type` | `def get_eq_type(anchor_type)` | 根据锚点类型获取对应的等式约束类型 |

### Classes（详情）

#### `class AnchorType`

> 锚点类型枚举

<details>
<summary>Class docstring</summary>

锚点类型枚举

术语说明:
    - 锚点 (Anchor): 用于物体操作的虚拟连接点
    - NONE: 无锚定，释放物体
    - WELD: 焊接锚定，完全固定位置和姿态
    - BALL: 球关节锚定，固定位置但允许旋转

使用示例:
    ```python
    # 使用焊接锚定抓取物体
    self.anchor_actor("object_name", AnchorType.WELD)
    
    # 使用球关节锚定
    self.anchor_actor("object_name", AnchorType.BALL)
    
    # 释放物体
    self.release_body_anchored()  # 内部使用 AnchorType.NONE

</details>


#### `class CaptureMode`

> 视频捕获模式枚举

<details>
<summary>Class docstring</summary>

视频捕获模式枚举

术语说明:
    - 同步模式 (SYNC): 每个相机帧都与仿真步进对齐，性能较低但帧对齐
    - 异步模式 (ASYNC): 相机帧独立捕获，性能较高但可能不完全对齐

使用示例:
    ```python
    # 开始保存视频，使用异步模式（默认，性能更好）
    env.begin_save_video("output.mp4", CaptureMode.ASYNC)
    
    # 使用同步模式（帧对齐，但性能较低）
    env.begin_save_video("output.mp4", CaptureMode.SYNC)
    ```

</details>


#### `class OrcaGymLocal`

> OrcaGym 本地仿真接口

<details>
<summary>Class docstring</summary>

OrcaGym 本地仿真接口

负责与本地 MuJoCo 仿真器的交互，包括模型加载、仿真控制、状态查询等。
这是 OrcaGymLocalEnv 的核心通信对象，通过 gRPC 与 OrcaSim 服务器通信。

核心功能:
    1. 模型管理: 加载 XML 模型、初始化 MuJoCo 模型和数据
    2. 仿真控制: 步进、重置、前向计算等
    3. 状态查询: 查询 body、joint、site、sensor 等状态
    4. 物体操作: 通过 mocap body 和等式约束操作物体
    5. 文件管理: 下载和缓存模型文件、mesh 等资源

架构说明:
    - 继承自 OrcaGymBase，提供基础的 gRPC 通信能力
    - 维护本地 MuJoCo 模型和数据 (_mjModel, _mjData)
    - 提供封装后的模型和数据对象 (self.model, self.data)
    - 支持控制覆盖 (override_ctrls)，允许外部控制特定执行器

使用示例:
    ```python
    # 在环境初始化时创建
    self.gym = OrcaGymLocal(self.stub)
    
    # 初始化仿真
    model_xml_path = await self.gym.load_model_xml()
    await self.gym.init_simulation(model_xml_path)
    
    # 访问模型和数据
    body_names = list(self.gym.model.get_body_names())
    qpos = self.gym.data.qpos
    ```

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymLocal.load_model_xml` | `async def load_model_xml(self)` | 从服务器加载模型 XML 文件 |  |
| `OrcaGymLocal.init_simulation` | `async def init_simulation(self, model_xml_path)` | 初始化 MuJoCo 仿真 |  |
| `OrcaGymLocal.render` | `async def render(self)` | 渲染当前仿真状态到 OrcaSim 服务器 |  |
| `OrcaGymLocal.update_local_env` | `async def update_local_env(self, qpos, time)` | 更新本地环境状态到服务器，并接收控制覆盖值 |  |
| `OrcaGymLocal.load_content_file` | `async def load_content_file(self, content_file_name, remote_file_dir='', local_file_dir='', temp_file_path=None)` | _No docstring._ |  |
| `OrcaGymLocal.process_xml_node` | `async def process_xml_node(self, node)` | _No docstring._ |  |
| `OrcaGymLocal.begin_save_video` | `async def begin_save_video(self, file_path, capture_mode=CaptureMode.ASYNC)` | _No docstring._ |  |
| `OrcaGymLocal.stop_save_video` | `async def stop_save_video(self)` | _No docstring._ |  |
| `OrcaGymLocal.get_current_frame` | `async def get_current_frame(self) -> int` | _No docstring._ |  |
| `OrcaGymLocal.get_camera_time_stamp` | `async def get_camera_time_stamp(self, last_frame) -> dict` | _No docstring._ |  |
| `OrcaGymLocal.get_frame_png` | `async def get_frame_png(self, image_path)` | _No docstring._ |  |
| `OrcaGymLocal.xml_file_dir` | `def xml_file_dir(self)` | _No docstring._ | `@property` |
| `OrcaGymLocal.process_xml_file` | `async def process_xml_file(self, file_path)` | _No docstring._ |  |
| `OrcaGymLocal.load_local_env` | `async def load_local_env(self)` | _No docstring._ |  |
| `OrcaGymLocal.get_body_manipulation_anchored` | `async def get_body_manipulation_anchored(self)` | _No docstring._ |  |
| `OrcaGymLocal.get_body_manipulation_movement` | `async def get_body_manipulation_movement(self)` | _No docstring._ |  |
| `OrcaGymLocal.set_time_step` | `def set_time_step(self, time_step)` | 设置仿真时间步长（本地和远程） |  |
| `OrcaGymLocal.set_opt_timestep` | `def set_opt_timestep(self, timestep)` | 设置本地 MuJoCo 模型的时间步长 |  |
| `OrcaGymLocal.set_timestep_remote` | `async def set_timestep_remote(self, timestep)` | 异步设置远程服务器的时间步长 |  |
| `OrcaGymLocal.set_opt_config` | `def set_opt_config(self)` | _No docstring._ |  |
| `OrcaGymLocal.query_opt_config` | `def query_opt_config(self)` | _No docstring._ |  |
| `OrcaGymLocal.query_model_info` | `def query_model_info(self)` | 查询模型基本信息（维度参数） |  |
| `OrcaGymLocal.query_all_equality_constraints` | `def query_all_equality_constraints(self)` | 查询所有等式约束 |  |
| `OrcaGymLocal.query_all_mocap_bodies` | `def query_all_mocap_bodies(self)` | _No docstring._ |  |
| `OrcaGymLocal.query_all_actuators` | `def query_all_actuators(self)` | 查询所有执行器信息 |  |
| `OrcaGymLocal.get_goal_bounding_box` | `def get_goal_bounding_box(self, goal_body_name)` | 计算目标物体（goal_body_name）在世界坐标系下的轴对齐包围盒。 |  |
| `OrcaGymLocal.set_actuator_trnid` | `def set_actuator_trnid(self, actuator_id, trnid)` | _No docstring._ |  |
| `OrcaGymLocal.disable_actuator` | `def disable_actuator(self, actuator_groups)` | _No docstring._ |  |
| `OrcaGymLocal.query_all_bodies` | `def query_all_bodies(self)` | 查询所有 body 信息 |  |
| `OrcaGymLocal.query_all_joints` | `def query_all_joints(self)` | 查询所有关节信息 |  |
| `OrcaGymLocal.query_all_geoms` | `def query_all_geoms(self)` | 查询所有几何体信息 |  |
| `OrcaGymLocal.query_all_sites` | `def query_all_sites(self)` | 查询所有 site 信息 |  |
| `OrcaGymLocal.query_all_sensors` | `def query_all_sensors(self)` | 查询所有传感器信息 |  |
| `OrcaGymLocal.update_data` | `def update_data(self)` | 从 MuJoCo 数据更新到封装的 data 对象 |  |
| `OrcaGymLocal.update_data_external` | `def update_data_external(self, qpos, qvel, qacc, qfrc_bias, time)` | Cooperate with the external environment. |  |
| `OrcaGymLocal.query_qfrc_bias` | `def query_qfrc_bias(self)` | _No docstring._ |  |
| `OrcaGymLocal.load_initial_frame` | `def load_initial_frame(self)` | 重置仿真数据到初始状态 |  |
| `OrcaGymLocal.query_joint_offsets` | `def query_joint_offsets(self, joint_names)` | _No docstring._ |  |
| `OrcaGymLocal.query_joint_lengths` | `def query_joint_lengths(self, joint_names)` | _No docstring._ |  |
| `OrcaGymLocal.query_body_xpos_xmat_xquat` | `def query_body_xpos_xmat_xquat(self, body_name_list)` | 查询 body 的位姿（位置、旋转矩阵、四元数） |  |
| `OrcaGymLocal.query_sensor_data` | `def query_sensor_data(self, sensor_names)` | 查询传感器数据 |  |
| `OrcaGymLocal.set_ctrl` | `def set_ctrl(self, ctrl)` | 设置控制输入，应用控制覆盖（如果存在） |  |
| `OrcaGymLocal.mj_step` | `def mj_step(self, nstep)` | 执行 MuJoCo 仿真步进 |  |
| `OrcaGymLocal.mj_forward` | `def mj_forward(self)` | 执行 MuJoCo 前向计算 |  |
| `OrcaGymLocal.mj_inverse` | `def mj_inverse(self)` | 执行 MuJoCo 逆动力学计算 |  |
| `OrcaGymLocal.mj_fullM` | `def mj_fullM(self)` | 计算完整的质量矩阵 |  |
| `OrcaGymLocal.mj_jacBody` | `def mj_jacBody(self, jacp, jacr, body_id)` | 计算 body 的雅可比矩阵 |  |
| `OrcaGymLocal.mj_jacSite` | `def mj_jacSite(self, jacp, jacr, site_id)` | 计算 site 的雅可比矩阵 |  |
| `OrcaGymLocal.query_joint_qpos` | `def query_joint_qpos(self, joint_names)` | _No docstring._ |  |
| `OrcaGymLocal.query_joint_qvel` | `def query_joint_qvel(self, joint_names)` | _No docstring._ |  |
| `OrcaGymLocal.query_joint_qacc` | `def query_joint_qacc(self, joint_names)` | _No docstring._ |  |
| `OrcaGymLocal.jnt_qposadr` | `def jnt_qposadr(self, joint_name)` | _No docstring._ |  |
| `OrcaGymLocal.jnt_dofadr` | `def jnt_dofadr(self, joint_name)` | _No docstring._ |  |
| `OrcaGymLocal.query_site_pos_and_mat` | `def query_site_pos_and_mat(self, site_names)` | _No docstring._ |  |
| `OrcaGymLocal.query_site_size` | `def query_site_size(self, site_names)` | _No docstring._ |  |
| `OrcaGymLocal.set_joint_qpos` | `def set_joint_qpos(self, joint_qpos)` | _No docstring._ |  |
| `OrcaGymLocal.set_joint_qvel` | `def set_joint_qvel(self, joint_qvel)` | _No docstring._ |  |
| `OrcaGymLocal.mj_jac_site` | `def mj_jac_site(self, site_names)` | _No docstring._ |  |
| `OrcaGymLocal.modify_equality_objects` | `def modify_equality_objects(self, old_obj1_id, old_obj2_id, new_obj1_id, new_obj2_id)` | 修改等式约束的目标对象 |  |
| `OrcaGymLocal.update_equality_constraints` | `def update_equality_constraints(self, constraint_list)` | 更新等式约束的参数 |  |
| `OrcaGymLocal.set_mocap_pos_and_quat` | `async def set_mocap_pos_and_quat(self, mocap_data, send_remote=False)` | 设置 mocap body 的位置和四元数 |  |
| `OrcaGymLocal.query_contact_simple` | `def query_contact_simple(self)` | 查询简单接触信息 |  |
| `OrcaGymLocal.set_geom_friction` | `def set_geom_friction(self, geom_friction_dict)` | _No docstring._ |  |
| `OrcaGymLocal.add_extra_weight` | `def add_extra_weight(self, random_weight_dict)` | _No docstring._ |  |
| `OrcaGymLocal.query_contact_force` | `def query_contact_force(self, contact_ids)` | 查询接触力 |  |
| `OrcaGymLocal.get_cfrc_ext` | `def get_cfrc_ext(self)` | 获取外部约束力 |  |
| `OrcaGymLocal.query_actuator_torques` | `def query_actuator_torques(self, actuator_names)` | 查询执行器扭矩 |  |
| `OrcaGymLocal.query_joint_dofadrs` | `def query_joint_dofadrs(self, joint_names)` | _No docstring._ |  |
| `OrcaGymLocal.query_velocity_body_B` | `def query_velocity_body_B(self, ee_body, base_body)` | 查询 body 相对于基座 body 的速度（基座坐标系） |  |
| `OrcaGymLocal.query_position_body_B` | `def query_position_body_B(self, ee_body, base_body)` | 查询 body 相对于基座 body 的位置（基座坐标系） |  |
| `OrcaGymLocal.query_orientation_body_B` | `def query_orientation_body_B(self, ee_body, base_body)` | 查询 body 相对于基座 body 的姿态（基座坐标系） |  |
| `OrcaGymLocal.query_joint_axes_B` | `def query_joint_axes_B(self, joint_names, base_body)` | _No docstring._ |  |
| `OrcaGymLocal.query_robot_velocity_odom` | `def query_robot_velocity_odom(self, base_body, initial_base_pos, initial_base_quat)` | 查询机器人在里程计坐标系中的速度 |  |
| `OrcaGymLocal.query_robot_position_odom` | `def query_robot_position_odom(self, base_body, initial_base_pos, initial_base_quat)` | 查询机器人在里程计坐标系中的位置 |  |
| `OrcaGymLocal.query_robot_orientation_odom` | `def query_robot_orientation_odom(self, base_body, initial_base_pos, initial_base_quat)` | 查询机器人在里程计坐标系中的姿态 |  |

##### 方法详情

###### `OrcaGymLocal.load_model_xml`

Signature:

```python
async def load_model_xml(self)
```

<details>
<summary>Docstring</summary>

从服务器加载模型 XML 文件

从 OrcaSim 服务器获取模型 XML 文件，下载依赖的资源文件（mesh、hfield等），
并返回本地文件路径。

Returns:
    model_xml_path: 模型 XML 文件的本地路径

使用示例:
    ```python
    # 在环境初始化时调用
    model_xml_path = await self.gym.load_model_xml()
    # 返回: "/path/to/model.xml"
    ```

</details>


###### `OrcaGymLocal.init_simulation`

Signature:

```python
async def init_simulation(self, model_xml_path)
```

<details>
<summary>Docstring</summary>

初始化 MuJoCo 仿真

从 XML 文件加载 MuJoCo 模型，创建数据对象，初始化所有模型信息容器
（body、joint、actuator、site、sensor 等），并创建封装的 model 和 data 对象。

术语说明:
    - MjModel: MuJoCo 模型对象，包含所有静态模型信息（几何、质量、约束等）
    - MjData: MuJoCo 数据对象，包含所有动态状态（位置、速度、力等）
    - 模型信息容器: OrcaGymModel 和 OrcaGymData，提供更友好的访问接口

Args:
    model_xml_path: 模型 XML 文件的路径

使用示例:
    ```python
    # 在环境初始化时调用
    model_xml_path = await self.gym.load_model_xml()
    await self.gym.init_simulation(model_xml_path)
    
    # 之后可以访问模型和数据
    self.model = self.gym.model
    self.data = self.gym.data
    ```

</details>


###### `OrcaGymLocal.render`

Signature:

```python
async def render(self)
```

<details>
<summary>Docstring</summary>

渲染当前仿真状态到 OrcaSim 服务器

将当前的关节位置和仿真时间发送到服务器，用于可视化。
同时接收服务器返回的控制覆盖值（如果用户在界面中手动控制）。

使用示例:
    ```python
    # 在环境的 render 方法中调用
    await self.gym.render()
    # 服务器会更新可视化，并可能返回控制覆盖值
    ```

</details>


###### `OrcaGymLocal.update_local_env`

Signature:

```python
async def update_local_env(self, qpos, time)
```

<details>
<summary>Docstring</summary>

更新本地环境状态到服务器，并接收控制覆盖值

将当前状态发送到服务器用于渲染，同时接收用户通过界面手动控制的值。
这些覆盖值会在下次 set_ctrl 时应用。

术语说明:
    - 控制覆盖 (Control Override): 外部（如用户界面）覆盖执行器的控制值
    - 用于实现手动控制、遥操作等功能

Args:
    qpos: 当前关节位置数组
    time: 当前仿真时间

使用示例:
    ```python
    # 在 render 中自动调用
    await self.gym.update_local_env(self.data.qpos, self._mjData.time)
    # 如果用户在界面中控制，override_ctrls 会被更新
    ```

</details>


###### `OrcaGymLocal.load_content_file`

Signature:

```python
async def load_content_file(self, content_file_name, remote_file_dir='', local_file_dir='', temp_file_path=None)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.process_xml_node`

Signature:

```python
async def process_xml_node(self, node)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.begin_save_video`

Signature:

```python
async def begin_save_video(self, file_path, capture_mode=CaptureMode.ASYNC)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.stop_save_video`

Signature:

```python
async def stop_save_video(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.get_current_frame`

Signature:

```python
async def get_current_frame(self) -> int
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.get_camera_time_stamp`

Signature:

```python
async def get_camera_time_stamp(self, last_frame) -> dict
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.get_frame_png`

Signature:

```python
async def get_frame_png(self, image_path)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.xml_file_dir` (property)

Signature:

```python
def xml_file_dir(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.process_xml_file`

Signature:

```python
async def process_xml_file(self, file_path)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.load_local_env`

Signature:

```python
async def load_local_env(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.get_body_manipulation_anchored`

Signature:

```python
async def get_body_manipulation_anchored(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.get_body_manipulation_movement`

Signature:

```python
async def get_body_manipulation_movement(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.set_time_step`

Signature:

```python
def set_time_step(self, time_step)
```

<details>
<summary>Docstring</summary>

设置仿真时间步长（本地和远程）

同时更新本地 MuJoCo 模型的时间步长和远程服务器的时间步长。

Args:
    time_step: 时间步长（秒），通常为 0.001-0.01

使用示例:
    ```python
    # 在环境初始化时设置
    self.gym.set_time_step(0.001)  # 1000 Hz
    await self.gym.set_timestep_remote(0.001)  # 同步到服务器
    ```

</details>


###### `OrcaGymLocal.set_opt_timestep`

Signature:

```python
def set_opt_timestep(self, timestep)
```

<details>
<summary>Docstring</summary>

设置本地 MuJoCo 模型的时间步长

术语说明:
    - opt.timestep: MuJoCo 优化选项中的时间步长参数
    - 影响: 时间步长越小，仿真越精确但计算越慢

Args:
    timestep: 时间步长（秒）

使用示例:
    ```python
    # 在模型加载后设置
    if self._mjModel is not None:
        self.set_opt_timestep(0.001)
    ```

</details>


###### `OrcaGymLocal.set_timestep_remote`

Signature:

```python
async def set_timestep_remote(self, timestep)
```

<details>
<summary>Docstring</summary>

异步设置远程服务器的时间步长

将时间步长同步到 OrcaSim 服务器，确保本地和远程一致。

Args:
    timestep: 时间步长（秒）

Returns:
    response: gRPC 响应对象

使用示例:
    ```python
    # 在设置时间步长时同步到服务器
    await self.gym.set_timestep_remote(0.001)
    ```

</details>


###### `OrcaGymLocal.set_opt_config`

Signature:

```python
def set_opt_config(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_opt_config`

Signature:

```python
def query_opt_config(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_model_info`

Signature:

```python
def query_model_info(self)
```

<details>
<summary>Docstring</summary>

查询模型基本信息（维度参数）

返回模型的各种维度信息，用于初始化 OrcaGymModel 对象。

术语说明:
    - nq: 位置坐标数量 (generalized coordinates)
    - nv: 速度坐标数量 (degrees of freedom)
    - nu: 执行器数量 (actuators)
    - nbody: body 数量
    - njnt: 关节数量
    - ngeom: 几何体数量
    - nsite: site 数量
    - nconmax: 最大接触数量

Returns:
    model_info: 包含所有维度信息的字典

使用示例:
    ```python
    # 在初始化仿真时调用
    model_info = self.query_model_info()
    self.model = OrcaGymModel(model_info)
    # 之后可以通过 self.model.nq, self.model.nv 等访问维度
    ```

</details>


###### `OrcaGymLocal.query_all_equality_constraints`

Signature:

```python
def query_all_equality_constraints(self)
```

<details>
<summary>Docstring</summary>

查询所有等式约束

返回模型中所有等式约束的信息，用于物体操作等功能。

术语说明:
    - 等式约束: 详见 orca_gym/core/orca_gym_model.py 中的说明
    - obj1_id, obj2_id: 被约束的两个 body 的 ID
    - eq_type: 约束类型（WELD、CONNECT 等）
    - eq_solref, eq_solimp: 约束求解器参数

Returns:
    equality_constraints: 等式约束列表，每个元素包含约束的详细信息

使用示例:
    ```python
    # 在初始化仿真时调用
    eq_list = self.query_all_equality_constraints()
    self.model.init_eq_list(eq_list)
    # 之后可以通过 self.model.get_eq_list() 访问
    ```

</details>


###### `OrcaGymLocal.query_all_mocap_bodies`

Signature:

```python
def query_all_mocap_bodies(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_all_actuators`

Signature:

```python
def query_all_actuators(self)
```

<details>
<summary>Docstring</summary>

查询所有执行器信息

返回所有执行器的详细信息，包括名称、关联关节、控制范围、齿轮比等。

术语说明:
    - 执行器: 详见 orca_gym/core/orca_gym_model.py 中的说明
    - GearRatio: 齿轮比，执行器输出与关节输入的比例
    - CtrlRange: 控制范围，执行器可接受的最小和最大控制值
    - TrnType: 传输类型，执行器驱动的对象类型（关节、肌腱、site等）

Returns:
    actuator_dict: 执行器字典，键为执行器名称，值为执行器信息

使用示例:
    ```python
    # 在初始化仿真时调用
    actuator_dict = self.query_all_actuators()
    self.model.init_actuator_dict(actuator_dict)
    # 之后可以通过 self.model.get_actuator_dict() 访问
    ```

</details>


###### `OrcaGymLocal.get_goal_bounding_box`

Signature:

```python
def get_goal_bounding_box(self, goal_body_name)
```

<details>
<summary>Docstring</summary>

计算目标物体（goal_body_name）在世界坐标系下的轴对齐包围盒。
支持 BOX、SPHERE 类型，BOX 会考虑 geom 的旋转。

</details>


###### `OrcaGymLocal.set_actuator_trnid`

Signature:

```python
def set_actuator_trnid(self, actuator_id, trnid)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.disable_actuator`

Signature:

```python
def disable_actuator(self, actuator_groups)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_all_bodies`

Signature:

```python
def query_all_bodies(self)
```

<details>
<summary>Docstring</summary>

查询所有 body 信息

返回所有 body 的详细信息，包括位置、姿态、质量、惯性等。

术语说明:
    - Body: 详见 orca_gym/core/orca_gym_model.py 中的说明
    - Mass: body 的质量
    - Inertia: body 的惯性张量
    - ParentID: 父 body 的 ID，形成运动链

Returns:
    body_dict: body 字典，键为 body 名称，值为 body 信息

使用示例:
    ```python
    # 在初始化仿真时调用
    body_dict = self.query_all_bodies()
    self.model.init_body_dict(body_dict)
    # 之后可以通过 self.model.get_body_dict() 访问
    ```

</details>


###### `OrcaGymLocal.query_all_joints`

Signature:

```python
def query_all_joints(self)
```

<details>
<summary>Docstring</summary>

查询所有关节信息

返回所有关节的详细信息，包括类型、范围、位置、轴等。

术语说明:
    - 关节: 详见 orca_gym/core/orca_gym_model.py 中的说明
    - Range: 关节的运动范围 [min, max]
    - Axis: 关节的旋转或滑动轴
    - Stiffness: 关节刚度
    - Damping: 关节阻尼

Returns:
    joint_dict: 关节字典，键为关节名称，值为关节信息

使用示例:
    ```python
    # 在初始化仿真时调用
    joint_dict = self.query_all_joints()
    self.model.init_joint_dict(joint_dict)
    # 之后可以通过 self.model.get_joint_dict() 访问
    ```

</details>


###### `OrcaGymLocal.query_all_geoms`

Signature:

```python
def query_all_geoms(self)
```

<details>
<summary>Docstring</summary>

查询所有几何体信息

返回所有几何体的详细信息，包括类型、大小、摩擦、位置等。

术语说明:
    - 几何体 (Geom): 用于碰撞检测的几何形状
    - 类型: BOX、SPHERE、CAPSULE、MESH 等
    - Friction: 摩擦系数 [滑动, 扭转, 滚动]
    - Size: 几何体尺寸，不同类型有不同含义

Returns:
    geom_dict: 几何体字典，键为几何体名称，值为几何体信息

使用示例:
    ```python
    # 在初始化仿真时调用
    geom_dict = self.query_all_geoms()
    self.model.init_geom_dict(geom_dict)
    # 之后可以通过 self.model.get_geom_dict() 访问
    ```

</details>


###### `OrcaGymLocal.query_all_sites`

Signature:

```python
def query_all_sites(self)
```

<details>
<summary>Docstring</summary>

查询所有 site 信息

返回所有 site 的详细信息，包括位置、姿态等。

术语说明:
    - Site: 详见 orca_gym/core/orca_gym_model.py 中的说明
    - 用途: 标记末端执行器、目标点等关键位置

Returns:
    site_dict: site 字典，键为 site 名称，值为 site 信息

使用示例:
    ```python
    # 在初始化仿真时调用
    site_dict = self.query_all_sites()
    self.model.init_site_dict(site_dict)
    # 之后可以通过 self.model.get_site_dict() 访问
    ```

</details>


###### `OrcaGymLocal.query_all_sensors`

Signature:

```python
def query_all_sensors(self)
```

<details>
<summary>Docstring</summary>

查询所有传感器信息

返回所有传感器的详细信息，包括类型、维度、地址等。

术语说明:
    - 传感器: 详见 orca_gym/core/orca_gym_model.py 中的说明
    - Type: 传感器类型（加速度计、陀螺仪、触觉等）
    - Dim: 传感器输出维度
    - Adr: 传感器数据在 sensordata 数组中的地址

Returns:
    sensor_dict: 传感器字典，键为传感器名称，值为传感器信息

使用示例:
    ```python
    # 在初始化仿真时调用
    sensor_dict = self.query_all_sensors()
    self.model.init_sensor_dict(sensor_dict)
    # 之后可以通过 self.model.gen_sensor_dict() 访问
    ```

</details>


###### `OrcaGymLocal.update_data`

Signature:

```python
def update_data(self)
```

<details>
<summary>Docstring</summary>

从 MuJoCo 数据更新到封装的 data 对象

将 _mjData 中的最新状态（qpos、qvel、qacc、qfrc_bias、time）同步到
封装的 OrcaGymData 对象中，供环境使用。

术语说明:
    - 缓存: 使用 _qpos_cache 等数组作为中间缓存，避免频繁分配内存
    - qfrc_bias: 偏置力，包括重力、科里奥利力等被动力

使用示例:
    ```python
    # 在仿真步进后调用，同步最新状态
    self.gym.mj_step(nstep)
    self.gym.update_data()  # 同步状态到 self.data
    
    # 之后可以安全访问
    current_qpos = self.data.qpos.copy()
    current_qvel = self.data.qvel.copy()
    ```

</details>


###### `OrcaGymLocal.update_data_external`

Signature:

```python
def update_data_external(self, qpos, qvel, qacc, qfrc_bias, time)
```

<details>
<summary>Docstring</summary>

Cooperate with the external environment.
Update the data for rendering in orcagym environment.

</details>


###### `OrcaGymLocal.query_qfrc_bias`

Signature:

```python
def query_qfrc_bias(self)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.load_initial_frame`

Signature:

```python
def load_initial_frame(self)
```

<details>
<summary>Docstring</summary>

重置仿真数据到初始状态

将 MuJoCo 数据重置为初始状态，包括位置、速度、加速度等。
相当于将仿真恢复到初始时刻。

术语说明:
    - mj_resetData: MuJoCo 的重置函数，将所有动态数据重置为初始值

使用示例:
    ```python
    # 在重置仿真时调用
    self.gym.load_initial_frame()
    self.gym.update_data()  # 同步到封装的 data 对象
    ```

</details>


###### `OrcaGymLocal.query_joint_offsets`

Signature:

```python
def query_joint_offsets(self, joint_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_joint_lengths`

Signature:

```python
def query_joint_lengths(self, joint_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_body_xpos_xmat_xquat`

Signature:

```python
def query_body_xpos_xmat_xquat(self, body_name_list)
```

<details>
<summary>Docstring</summary>

查询 body 的位姿（位置、旋转矩阵、四元数）

从 MuJoCo 数据中直接查询 body 的位姿信息。
这是底层查询方法，被 OrcaGymLocalEnv 封装后使用。

术语说明:
    - xpos: body 在世界坐标系中的位置
    - xmat: body 在世界坐标系中的旋转矩阵（3x3，按行展开为9个元素）
    - xquat: body 在世界坐标系中的四元数 [w, x, y, z]

Args:
    body_name_list: body 名称列表

Returns:
    body_pos_mat_quat_list: 字典，键为 body 名称，值为包含 'Pos'、'Mat'、'Quat' 的字典

使用示例:
    ```python
    # 在环境的方法中调用
    body_dict = self.gym.query_body_xpos_xmat_xquat(["base_link"])
    base_pos = body_dict["base_link"]["Pos"]  # [x, y, z]
    base_mat = body_dict["base_link"]["Mat"]  # 9个元素，3x3矩阵按行展开
    base_quat = body_dict["base_link"]["Quat"]  # [w, x, y, z]
    ```

</details>


###### `OrcaGymLocal.query_sensor_data`

Signature:

```python
def query_sensor_data(self, sensor_names)
```

<details>
<summary>Docstring</summary>

查询传感器数据

从 MuJoCo 的 sensordata 数组中读取指定传感器的当前值。

术语说明:
    - sensordata: MuJoCo 中存储所有传感器数据的数组
    - Adr: 传感器数据在数组中的起始地址
    - Dim: 传感器输出维度

Args:
    sensor_names: 传感器名称列表

Returns:
    sensor_data_dict: 字典，键为传感器名称，值为传感器数据数组

使用示例:
    ```python
    # 查询 IMU 传感器数据
    sensor_data = self.gym.query_sensor_data(["imu_accelerometer", "imu_gyro"])
    accel = sensor_data["imu_accelerometer"]  # 加速度数据
    gyro = sensor_data["imu_gyro"]  # 角速度数据
    ```

</details>


###### `OrcaGymLocal.set_ctrl`

Signature:

```python
def set_ctrl(self, ctrl)
```

<details>
<summary>Docstring</summary>

设置控制输入，应用控制覆盖（如果存在）

设置执行器控制值，如果存在控制覆盖（来自用户界面手动控制），
则覆盖对应执行器的值。

术语说明:
    - 控制覆盖: 外部（如用户界面）覆盖执行器的控制值，用于遥操作

Args:
    ctrl: 控制输入数组，形状 (nu,)

使用示例:
    ```python
    # 在仿真步进前设置控制
    self.gym.set_ctrl(action)  # action 形状: (nu,)
    # 如果用户在界面中手动控制，对应执行器的值会被覆盖
    ```

</details>


###### `OrcaGymLocal.mj_step`

Signature:

```python
def mj_step(self, nstep)
```

<details>
<summary>Docstring</summary>

执行 MuJoCo 仿真步进

执行 nstep 次物理仿真步进，每次步进的时间为 timestep。
在调用前需要先设置控制输入 (set_ctrl)。

术语说明:
    - mj_step: MuJoCo 的核心步进函数，执行一次完整的物理仿真
    - 包括: 前向计算、约束求解、积分等步骤

Args:
    nstep: 步进次数，通常为 1 或 frame_skip

使用示例:
    ```python
    # 在 do_simulation 中调用
    self.gym.set_ctrl(ctrl)
    self.gym.mj_step(nstep=5)  # 步进 5 次
    ```

</details>


###### `OrcaGymLocal.mj_forward`

Signature:

```python
def mj_forward(self)
```

<details>
<summary>Docstring</summary>

执行 MuJoCo 前向计算

更新所有动力学相关状态，包括位置、速度、加速度、力等。
在设置关节状态、mocap 位置等操作后需要调用，确保状态一致。

术语说明:
    - 前向计算: 根据当前状态和输入计算下一时刻的状态
    - 包括: 正向运动学、动力学、约束等计算

使用示例:
    ```python
    # 在设置状态后调用
    self.gym.set_joint_qpos(qpos)
    self.gym.mj_forward()  # 更新所有相关状态
    ```

</details>


###### `OrcaGymLocal.mj_inverse`

Signature:

```python
def mj_inverse(self)
```

<details>
<summary>Docstring</summary>

执行 MuJoCo 逆动力学计算

根据给定的加速度计算所需的力和力矩。
用于计算实现特定运动所需的控制输入。

术语说明:
    - 逆动力学: 根据期望的加速度计算所需的力和力矩
    - 用途: 用于计算实现特定运动所需的控制输入

使用示例:
    ```python
    # 计算实现期望加速度所需的力
    self.gym.mj_inverse()
    required_force = self._mjData.qfrc_actuator
    ```

</details>


###### `OrcaGymLocal.mj_fullM`

Signature:

```python
def mj_fullM(self)
```

<details>
<summary>Docstring</summary>

计算完整的质量矩阵

返回系统的完整质量矩阵 M，形状 (nv, nv)，用于动力学计算。

术语说明:
    - 质量矩阵 (Mass Matrix): 描述系统惯性的矩阵，用于动力学方程
    - 形状: (nv, nv)，其中 nv 是系统的自由度数量
    - 用途: 用于逆动力学、力控制等算法

Returns:
    mass_matrix: 质量矩阵，形状 (nv, nv)

使用示例:
    ```python
    # 计算质量矩阵用于逆动力学
    M = self.gym.mj_fullM()  # 形状: (nv, nv)
    # 用于计算: tau = M @ qacc + C + G
    ```

</details>


###### `OrcaGymLocal.mj_jacBody`

Signature:

```python
def mj_jacBody(self, jacp, jacr, body_id)
```

<details>
<summary>Docstring</summary>

计算 body 的雅可比矩阵

术语说明:
    - 雅可比矩阵: 详见 orca_gym/environment/orca_gym_local_env.py 中的说明
    - jacp: 位置雅可比，形状 (3, nv)
    - jacr: 旋转雅可比，形状 (3, nv)

Args:
    jacp: 输出数组，用于存储位置雅可比，形状 (3, nv)
    jacr: 输出数组，用于存储旋转雅可比，形状 (3, nv)
    body_id: body 的 ID

使用示例:
    ```python
    # 计算末端执行器的雅可比
    jacp = np.zeros((3, self.model.nv))
    jacr = np.zeros((3, self.model.nv))
    body_id = self.model.body_name2id("end_effector")
    self.gym.mj_jacBody(jacp, jacr, body_id)
    ```

</details>


###### `OrcaGymLocal.mj_jacSite`

Signature:

```python
def mj_jacSite(self, jacp, jacr, site_id)
```

<details>
<summary>Docstring</summary>

计算 site 的雅可比矩阵

术语说明:
    - 雅可比矩阵: 详见 orca_gym/environment/orca_gym_local_env.py 中的说明
    - Site: 标记点，详见 orca_gym/core/orca_gym_model.py 中的说明

Args:
    jacp: 输出数组，用于存储位置雅可比，形状 (3, nv)
    jacr: 输出数组，用于存储旋转雅可比，形状 (3, nv)
    site_id: site 的 ID

使用示例:
    ```python
    # 计算 site 的雅可比矩阵
    site_id = self._mjModel.site("end_effector").id
    jacp = np.zeros((3, self.model.nv))
    jacr = np.zeros((3, self.model.nv))
    self.gym.mj_jacSite(jacp, jacr, site_id)
    ```

</details>


###### `OrcaGymLocal.query_joint_qpos`

Signature:

```python
def query_joint_qpos(self, joint_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_joint_qvel`

Signature:

```python
def query_joint_qvel(self, joint_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_joint_qacc`

Signature:

```python
def query_joint_qacc(self, joint_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.jnt_qposadr`

Signature:

```python
def jnt_qposadr(self, joint_name)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.jnt_dofadr`

Signature:

```python
def jnt_dofadr(self, joint_name)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_site_pos_and_mat`

Signature:

```python
def query_site_pos_and_mat(self, site_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_site_size`

Signature:

```python
def query_site_size(self, site_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.set_joint_qpos`

Signature:

```python
def set_joint_qpos(self, joint_qpos)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.set_joint_qvel`

Signature:

```python
def set_joint_qvel(self, joint_qvel)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.mj_jac_site`

Signature:

```python
def mj_jac_site(self, site_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.modify_equality_objects`

Signature:

```python
def modify_equality_objects(self, old_obj1_id, old_obj2_id, new_obj1_id, new_obj2_id)
```

<details>
<summary>Docstring</summary>

修改等式约束的目标对象

将等式约束从一个 body 对转移到另一个 body 对，用于物体操作。

术语说明:
    - 等式约束对象: 被约束的两个 body 的 ID
    - 用途: 在抓取物体时，将约束从锚点-虚拟体转移到锚点-真实物体

Args:
    old_obj1_id, old_obj2_id: 原约束的两个 body ID
    new_obj1_id, new_obj2_id: 新约束的两个 body ID

使用示例:
    ```python
    # 修改约束以连接物体
    self.gym.modify_equality_objects(
        old_obj1_id=old_obj1_id,
        old_obj2_id=old_obj2_id,
        new_obj1_id=eq["obj1_id"],
        new_obj2_id=eq["obj2_id"]
    )
    ```

</details>


###### `OrcaGymLocal.update_equality_constraints`

Signature:

```python
def update_equality_constraints(self, constraint_list)
```

<details>
<summary>Docstring</summary>

更新等式约束的参数

更新约束的类型和数据，用于改变约束的刚度和行为。

术语说明:
    - eq_data: 约束数据，包含约束的具体参数
    - eq_type: 约束类型（WELD、CONNECT 等）

Args:
    constraint_list: 约束列表，每个元素包含 obj1_id、obj2_id、eq_data、eq_type

使用示例:
    ```python
    # 更新约束列表
    eq_list = self.model.get_eq_list()
    # 修改约束参数...
    self.gym.update_equality_constraints(eq_list)
    ```

</details>


###### `OrcaGymLocal.set_mocap_pos_and_quat`

Signature:

```python
async def set_mocap_pos_and_quat(self, mocap_data, send_remote=False)
```

<details>
<summary>Docstring</summary>

设置 mocap body 的位置和四元数

设置 mocap body 的位姿，用于物体操作。如果 send_remote=True，同时同步到服务器。

术语说明:
    - Mocap Body: 详见 orca_gym/core/orca_gym_model.py 中的说明
    - body_mocapid: body 对应的 mocap ID，-1 表示不是 mocap body

Args:
    mocap_data: 字典，键为 mocap body 名称，值为包含 'pos' 和 'quat' 的字典
    send_remote: 是否同步到远程服务器（用于渲染）

使用示例:
    ```python
    # 设置锚点位置
    await self.gym.set_mocap_pos_and_quat({
        "ActorManipulator_Anchor": {
            "pos": np.array([0.5, 0.0, 0.8]),
            "quat": np.array([1.0, 0.0, 0.0, 0.0])
        }
    }, send_remote=True)
    ```

</details>


###### `OrcaGymLocal.query_contact_simple`

Signature:

```python
def query_contact_simple(self)
```

<details>
<summary>Docstring</summary>

查询简单接触信息

返回当前所有接触点的基本信息，包括接触的几何体对。

术语说明:
    - 接触 (Contact): 两个几何体之间的碰撞或接触
    - Geom1, Geom2: 接触的两个几何体的 ID
    - Dim: 接触的维度，通常为 3（点接触）或 6（面接触）

Returns:
    contacts: 接触信息列表，每个元素包含接触 ID、维度、几何体 ID 等

使用示例:
    ```python
    # 查询当前所有接触
    contacts = self.gym.query_contact_simple()
    for contact in contacts:
        print(f"Contact between geom {contact['Geom1']} and {contact['Geom2']}")
    ```

</details>


###### `OrcaGymLocal.set_geom_friction`

Signature:

```python
def set_geom_friction(self, geom_friction_dict)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.add_extra_weight`

Signature:

```python
def add_extra_weight(self, random_weight_dict)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_contact_force`

Signature:

```python
def query_contact_force(self, contact_ids)
```

<details>
<summary>Docstring</summary>

查询接触力

计算指定接触点的接触力，包括线性力和力矩。

术语说明:
    - 接触力: 两个物体接触时产生的力和力矩
    - 返回值: 6维向量，前3个为线性力 [fx, fy, fz]，后3个为力矩 [mx, my, mz]

Args:
    contact_ids: 接触点 ID 列表

Returns:
    contact_force_dict: 字典，键为接触 ID，值为6维力向量

使用示例:
    ```python
    # 查询接触力
    contact_ids = [0, 1, 2]  # 接触点 ID
    forces = self.gym.query_contact_force(contact_ids)
    force_0 = forces[0]  # [fx, fy, fz, mx, my, mz]
    ```

</details>


###### `OrcaGymLocal.get_cfrc_ext`

Signature:

```python
def get_cfrc_ext(self)
```

<details>
<summary>Docstring</summary>

获取外部约束力

返回所有 body 受到的外部约束力，包括接触力、等式约束力等。

术语说明:
    - cfrc_ext: 外部约束力，形状 (nbody, 6)，每行为 [fx, fy, fz, mx, my, mz]
    - 用途: 用于分析物体受力、计算奖励等

Returns:
    cfrc_ext: 外部约束力数组，形状 (nbody, 6)

使用示例:
    ```python
    # 获取所有 body 的外部约束力
    cfrc_ext = self.gym.get_cfrc_ext()
    base_force = cfrc_ext[base_body_id]  # 基座的受力
    ```

</details>


###### `OrcaGymLocal.query_actuator_torques`

Signature:

```python
def query_actuator_torques(self, actuator_names)
```

<details>
<summary>Docstring</summary>

查询执行器扭矩

计算执行器产生的实际扭矩，考虑齿轮比等因素。

术语说明:
    - 执行器扭矩: 执行器实际输出的扭矩
    - 齿轮比 (Gear Ratio): 执行器输出与关节输入的比例
    - actuator_force: MuJoCo 中执行器的原始力/扭矩值

Args:
    actuator_names: 执行器名称列表

Returns:
    actuator_torques: 字典，键为执行器名称，值为6维扭矩向量

使用示例:
    ```python
    # 查询执行器扭矩
    torques = self.gym.query_actuator_torques(["joint1_actuator", "joint2_actuator"])
    torque_1 = torques["joint1_actuator"]  # 6维向量
    ```

</details>


###### `OrcaGymLocal.query_joint_dofadrs`

Signature:

```python
def query_joint_dofadrs(self, joint_names)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_velocity_body_B`

Signature:

```python
def query_velocity_body_B(self, ee_body, base_body)
```

<details>
<summary>Docstring</summary>

查询 body 相对于基座 body 的速度（基座坐标系）

计算末端执行器相对于基座的线速度和角速度，在基座坐标系中表示。

术语说明:
    - 基座坐标系: 以机器人基座为原点的局部坐标系
    - 线速度: 物体在空间中的移动速度
    - 角速度: 物体绕轴旋转的速度

Args:
    ee_body: 末端执行器 body 名称
    base_body: 基座 body 名称

Returns:
    combined_vel: 6维速度向量，前3个为线速度，后3个为角速度（基座坐标系）

使用示例:
    ```python
    # 查询末端执行器相对于基座的速度
    vel_B = self.gym.query_velocity_body_B("end_effector", "base_link")
    linear_vel = vel_B[:3]  # 线速度
    angular_vel = vel_B[3:]  # 角速度
    ```

</details>


###### `OrcaGymLocal.query_position_body_B`

Signature:

```python
def query_position_body_B(self, ee_body, base_body)
```

<details>
<summary>Docstring</summary>

查询 body 相对于基座 body 的位置（基座坐标系）

计算末端执行器相对于基座的位置，在基座坐标系中表示。

术语说明:
    - 相对位置: 相对于基座的位置，而不是世界坐标系
    - 基座坐标系: 以机器人基座为原点的局部坐标系

Args:
    ee_body: 末端执行器 body 名称
    base_body: 基座 body 名称

Returns:
    relative_pos: 相对位置数组 [x, y, z]（基座坐标系）

使用示例:
    ```python
    # 查询末端执行器相对于基座的位置
    pos_B = self.gym.query_position_body_B("end_effector", "base_link")
    # 返回: [x, y, z]，相对于基座的位置
    ```

</details>


###### `OrcaGymLocal.query_orientation_body_B`

Signature:

```python
def query_orientation_body_B(self, ee_body, base_body)
```

<details>
<summary>Docstring</summary>

查询 body 相对于基座 body 的姿态（基座坐标系）

计算末端执行器相对于基座的姿态（四元数），在基座坐标系中表示。

术语说明:
    - 相对姿态: 相对于基座的旋转，而不是世界坐标系
    - 四元数: [x, y, z, w] 格式（SciPy 格式）

Args:
    ee_body: 末端执行器 body 名称
    base_body: 基座 body 名称

Returns:
    relative_quat: 相对四元数 [x, y, z, w]（基座坐标系）

使用示例:
    ```python
    # 查询末端执行器相对于基座的姿态
    quat_B = self.gym.query_orientation_body_B("end_effector", "base_link")
    # 返回: [x, y, z, w]，相对于基座的姿态
    ```

</details>


###### `OrcaGymLocal.query_joint_axes_B`

Signature:

```python
def query_joint_axes_B(self, joint_names, base_body)
```

<details>
<summary>Docstring</summary>

_No docstring._

</details>


###### `OrcaGymLocal.query_robot_velocity_odom`

Signature:

```python
def query_robot_velocity_odom(self, base_body, initial_base_pos, initial_base_quat)
```

<details>
<summary>Docstring</summary>

查询机器人在里程计坐标系中的速度

计算机器人基座相对于初始位置的速度，在初始姿态的坐标系中表示。

术语说明:
    - 里程计 (Odometry): 基于初始位置的相对位置和速度估计
    - 初始姿态坐标系: 以机器人初始姿态为参考的坐标系
    - 用途: 用于移动机器人的定位和导航

Args:
    base_body: 基座 body 名称
    initial_base_pos: 初始基座位置 [x, y, z]
    initial_base_quat: 初始基座四元数 [w, x, y, z]

Returns:
    linear_vel_odom: 线速度 [vx, vy, vz]（里程计坐标系）
    angular_vel_odom: 角速度 [wx, wy, wz]（里程计坐标系）

使用示例:
    ```python
    # 查询机器人速度（相对于初始位置）
    linear, angular = self.gym.query_robot_velocity_odom(
        "base_link", initial_pos, initial_quat
    )
    ```

</details>


###### `OrcaGymLocal.query_robot_position_odom`

Signature:

```python
def query_robot_position_odom(self, base_body, initial_base_pos, initial_base_quat)
```

<details>
<summary>Docstring</summary>

查询机器人在里程计坐标系中的位置

计算机器人基座相对于初始位置的位置，在初始姿态的坐标系中表示。

术语说明:
    - 里程计: 详见 query_robot_velocity_odom 的说明

Args:
    base_body: 基座 body 名称
    initial_base_pos: 初始基座位置 [x, y, z]
    initial_base_quat: 初始基座四元数 [w, x, y, z]

Returns:
    pos_odom: 位置 [x, y, z]（里程计坐标系）

使用示例:
    ```python
    # 查询机器人位置（相对于初始位置）
    pos_odom = self.gym.query_robot_position_odom(
        "base_link", initial_pos, initial_quat
    )
    ```

</details>


###### `OrcaGymLocal.query_robot_orientation_odom`

Signature:

```python
def query_robot_orientation_odom(self, base_body, initial_base_pos, initial_base_quat)
```

<details>
<summary>Docstring</summary>

查询机器人在里程计坐标系中的姿态

计算机器人基座相对于初始姿态的旋转，在初始姿态的坐标系中表示。

术语说明:
    - 里程计: 详见 query_robot_velocity_odom 的说明

Args:
    base_body: 基座 body 名称
    initial_base_pos: 初始基座位置 [x, y, z]（未使用，为接口一致性保留）
    initial_base_quat: 初始基座四元数 [w, x, y, z]

Returns:
    quat_odom: 四元数 [x, y, z, w]（里程计坐标系，SciPy 格式）

使用示例:
    ```python
    # 查询机器人姿态（相对于初始姿态）
    quat_odom = self.gym.query_robot_orientation_odom(
        "base_link", initial_pos, initial_quat
    )
    ```

</details>


### Functions（详情）

#### `get_qpos_size`

Signature:

```python
def get_qpos_size(joint_type)
```

<details>
<summary>Docstring</summary>

获取关节在 qpos 数组中的元素数量

术语说明:
    - qpos (关节位置): 关节的广义坐标，不同关节类型占用不同数量的元素
    - FREE 关节: 7个元素 (3个位置 + 4个四元数)
    - BALL 关节: 4个元素 (四元数)
    - SLIDE/HINGE 关节: 1个元素 (单个标量)

使用示例:
    ```python
    # 查询关节在 qpos 中的长度
    joint_type = self._mjModel.jnt_type[joint_id]
    qpos_size = get_qpos_size(joint_type)  # 返回 1, 3, 4 或 7
    ```

</details>


#### `get_dof_size`

Signature:

```python
def get_dof_size(joint_type)
```

<details>
<summary>Docstring</summary>

获取关节的自由度数量（在 qvel 数组中的元素数量）

术语说明:
    - DOF (自由度): 关节允许的运动维度
    - qvel (关节速度): 关节的广义速度，对应 qpos 的导数
    - FREE 关节: 6个自由度 (3个线性 + 3个旋转)
    - BALL 关节: 3个自由度 (3个旋转)
    - SLIDE/HINGE 关节: 1个自由度 (单个标量)

使用示例:
    ```python
    # 查询关节的自由度数量
    joint_type = self._mjModel.jnt_type[joint_id]
    dof_size = get_dof_size(joint_type)  # 返回 1, 3 或 6
    ```

</details>


#### `get_eq_type`

Signature:

```python
def get_eq_type(anchor_type)
```

<details>
<summary>Docstring</summary>

根据锚点类型获取对应的等式约束类型

术语说明:
    - 等式约束类型: MuJoCo 中用于连接两个 body 的约束类型
    - mjEQ_WELD: 焊接约束，完全固定位置和姿态
    - mjEQ_CONNECT: 连接约束，球关节连接，固定位置但允许旋转

Args:
    anchor_type (AnchorType): 锚点类型

Returns:
    mujoco.mjtEq: 对应的等式约束类型

使用示例:
    ```python
    # 在更新等式约束时使用
    eq_type = get_eq_type(AnchorType.WELD)  # 返回 mjEQ_WELD
    eq["eq_type"] = eq_type
    ```

</details>

---

## `orca_gym/core/orca_gym_model.py`

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymModel` | _No docstring._ |

### Classes（详情）

#### `class OrcaGymModel`

<details>
<summary>Class docstring</summary>

_No docstring._

</details>


##### 方法索引

| Method | 签名 | 摘要 | Decorators |
| --- | --- | --- | --- |
| `OrcaGymModel.init_model_info` | `def init_model_info(self, model_info)` | 初始化模型基本信息（维度参数） |  |
| `OrcaGymModel.init_eq_list` | `def init_eq_list(self, eq_list)` | 初始化等式约束列表 |  |
| `OrcaGymModel.get_eq_list` | `def get_eq_list(self)` | 获取等式约束列表 |  |
| `OrcaGymModel.init_mocap_dict` | `def init_mocap_dict(self, mocap_dict)` | 初始化 mocap body 字典 |  |
| `OrcaGymModel.init_actuator_dict` | `def init_actuator_dict(self, actuator_dict)` | 初始化执行器字典，建立名称和ID的映射关系 |  |
| `OrcaGymModel.get_actuator_dict` | `def get_actuator_dict(self)` | 获取所有执行器字典 |  |
| `OrcaGymModel.get_actuator_byid` | `def get_actuator_byid(self, id)` | 根据ID获取执行器信息 |  |
| `OrcaGymModel.get_actuator_byname` | `def get_actuator_byname(self, name)` | 根据名称获取执行器信息 |  |
| `OrcaGymModel.actuator_name2id` | `def actuator_name2id(self, actuator_name)` | 执行器名称转ID |  |
| `OrcaGymModel.actuator_id2name` | `def actuator_id2name(self, actuator_id)` | 执行器ID转名称 |  |
| `OrcaGymModel.init_body_dict` | `def init_body_dict(self, body_dict)` | 初始化 body 字典，建立名称和ID的映射关系 |  |
| `OrcaGymModel.get_body_dict` | `def get_body_dict(self)` | 获取所有 body 字典 |  |
| `OrcaGymModel.get_body_byid` | `def get_body_byid(self, id)` | 根据ID获取 body 信息 |  |
| `OrcaGymModel.get_body_byname` | `def get_body_byname(self, name)` | 根据名称获取 body 信息 |  |
| `OrcaGymModel.body_name2id` | `def body_name2id(self, body_name)` | Body 名称转ID |  |
| `OrcaGymModel.body_id2name` | `def body_id2name(self, body_id)` | Body ID转名称 |  |
| `OrcaGymModel.init_joint_dict` | `def init_joint_dict(self, joint_dict)` | 初始化关节字典，建立名称和ID的映射关系 |  |
| `OrcaGymModel.get_joint_dict` | `def get_joint_dict(self)` | 获取所有关节字典 |  |
| `OrcaGymModel.get_joint_byid` | `def get_joint_byid(self, id)` | 根据ID获取关节信息 |  |
| `OrcaGymModel.get_joint_byname` | `def get_joint_byname(self, name)` | 根据名称获取关节信息 |  |
| `OrcaGymModel.joint_name2id` | `def joint_name2id(self, joint_name)` | 关节名称转ID |  |
| `OrcaGymModel.joint_id2name` | `def joint_id2name(self, joint_id)` | 关节ID转名称 |  |
| `OrcaGymModel.init_geom_dict` | `def init_geom_dict(self, geom_dict)` | 初始化几何体字典，建立名称和ID的映射关系 |  |
| `OrcaGymModel.get_geom_dict` | `def get_geom_dict(self)` | 获取所有几何体字典 |  |
| `OrcaGymModel.get_geom_byid` | `def get_geom_byid(self, id)` | 根据ID获取几何体信息 |  |
| `OrcaGymModel.get_geom_byname` | `def get_geom_byname(self, name)` | 根据名称获取几何体信息 |  |
| `OrcaGymModel.geom_name2id` | `def geom_name2id(self, geom_name)` | 几何体名称转ID |  |
| `OrcaGymModel.geom_id2name` | `def geom_id2name(self, geom_id)` | 几何体ID转名称 |  |
| `OrcaGymModel.get_body_names` | `def get_body_names(self)` | 获取所有 body 名称列表 |  |
| `OrcaGymModel.get_geom_body_name` | `def get_geom_body_name(self, geom_id)` | 根据几何体ID获取其所属的 body 名称 |  |
| `OrcaGymModel.get_geom_body_id` | `def get_geom_body_id(self, geom_id)` | 根据几何体ID获取其所属的 body ID |  |
| `OrcaGymModel.get_actuator_ctrlrange` | `def get_actuator_ctrlrange(self)` | 获取所有执行器的控制范围（用于定义动作空间） |  |
| `OrcaGymModel.get_joint_qposrange` | `def get_joint_qposrange(self, joint_names)` | 获取指定关节的位置范围 |  |
| `OrcaGymModel.init_site_dict` | `def init_site_dict(self, site_dict)` | 初始化 site 字典 |  |
| `OrcaGymModel.get_site_dict` | `def get_site_dict(self)` | 获取所有 site 字典 |  |
| `OrcaGymModel.get_site` | `def get_site(self, name_or_id)` | 根据名称或ID获取 site 信息 |  |
| `OrcaGymModel.site_name2id` | `def site_name2id(self, site_name)` | Site 名称转ID |  |
| `OrcaGymModel.site_id2name` | `def site_id2name(self, site_id)` | Site ID转名称 |  |
| `OrcaGymModel.init_sensor_dict` | `def init_sensor_dict(self, sensor_dict)` | 初始化传感器字典，识别传感器类型 |  |
| `OrcaGymModel.gen_sensor_dict` | `def gen_sensor_dict(self)` | 获取所有传感器字典 |  |
| `OrcaGymModel.get_sensor` | `def get_sensor(self, name_or_id)` | 根据名称或ID获取传感器信息 |  |
| `OrcaGymModel.sensor_name2id` | `def sensor_name2id(self, sensor_name)` | 传感器名称转ID |  |
| `OrcaGymModel.sensor_id2name` | `def sensor_id2name(self, sensor_id)` | 传感器ID转名称 |  |

##### 方法详情

###### `OrcaGymModel.init_model_info`

Signature:

```python
def init_model_info(self, model_info)
```

<details>
<summary>Docstring</summary>

初始化模型基本信息（维度参数）

</details>


###### `OrcaGymModel.init_eq_list`

Signature:

```python
def init_eq_list(self, eq_list)
```

<details>
<summary>Docstring</summary>

初始化等式约束列表

术语说明:
    - 等式约束 (Equality Constraint): 在 MuJoCo 中用于连接两个 body 的约束关系
    - 常见类型: CONNECT (球关节连接)、WELD (焊接固定)、JOINT (关节耦合) 等
    - 用途: 实现抓取、固定物体等操作，通过约束将两个 body 连接在一起

使用示例:
    ```python
    # 获取等式约束列表用于物体操作
    eq_list = self.model.get_eq_list()
    # 修改约束以连接物体
    eq["obj2_id"] = self.model.body_name2id(actor_name)
    ```

</details>


###### `OrcaGymModel.get_eq_list`

Signature:

```python
def get_eq_list(self)
```

<details>
<summary>Docstring</summary>

获取等式约束列表

术语说明:
    - 等式约束: 用于连接两个 body 的约束关系，详见 init_eq_list 的说明

使用示例:
    ```python
    # 获取约束列表用于修改
    eq_list = self.model.get_eq_list()
    for eq in eq_list:
        if eq["obj1_id"] == self._anchor_body_id:
            # 修改约束目标
            eq["obj2_id"] = self.model.body_name2id(actor_name)
    ```

</details>


###### `OrcaGymModel.init_mocap_dict`

Signature:

```python
def init_mocap_dict(self, mocap_dict)
```

<details>
<summary>Docstring</summary>

初始化 mocap body 字典

术语说明:
    - Mocap Body (Motion Capture Body): 虚拟的、可自由移动的 body，不受物理约束
    - 用途: 用于物体操作，通过等式约束将 mocap body 与真实物体连接，移动 mocap body 即可控制物体
    - 常见应用: 抓取、拖拽、移动物体等操作

使用示例:
    ```python
    # 设置 mocap body 位置用于物体操作
    self.set_mocap_pos_and_quat({
        "ActorManipulator_Anchor": {
            "pos": np.array([0.5, 0.0, 0.8]),
            "quat": np.array([1.0, 0.0, 0.0, 0.0])
        }
    })
    ```

</details>


###### `OrcaGymModel.init_actuator_dict`

Signature:

```python
def init_actuator_dict(self, actuator_dict)
```

<details>
<summary>Docstring</summary>

初始化执行器字典，建立名称和ID的映射关系

术语说明:
    - 执行器 (Actuator): 机器人的驱动元件，如电机、液压缸等，用于产生力和力矩
    - 控制输入: 发送给执行器的命令值，通常对应期望的扭矩、位置或速度
    - nu: 执行器数量，等于动作空间的维度

使用示例:
    ```python
    # 执行器在模型加载时自动初始化
    # 可以通过以下方式访问:
    actuator_dict = self.model.get_actuator_dict()
    actuator_id = self.model.actuator_name2id("joint1_actuator")
    ```

</details>


###### `OrcaGymModel.get_actuator_dict`

Signature:

```python
def get_actuator_dict(self)
```

<details>
<summary>Docstring</summary>

获取所有执行器字典

</details>


###### `OrcaGymModel.get_actuator_byid`

Signature:

```python
def get_actuator_byid(self, id)
```

<details>
<summary>Docstring</summary>

根据ID获取执行器信息

</details>


###### `OrcaGymModel.get_actuator_byname`

Signature:

```python
def get_actuator_byname(self, name)
```

<details>
<summary>Docstring</summary>

根据名称获取执行器信息

</details>


###### `OrcaGymModel.actuator_name2id`

Signature:

```python
def actuator_name2id(self, actuator_name)
```

<details>
<summary>Docstring</summary>

执行器名称转ID

将执行器名称转换为对应的 ID，用于设置控制输入。

使用示例:
    ```python
    # 获取执行器 ID 列表用于控制
    self._arm_actuator_id = [
        self.model.actuator_name2id(actuator_name) 
        for actuator_name in self._arm_moto_names
    ]
    ```

</details>


###### `OrcaGymModel.actuator_id2name`

Signature:

```python
def actuator_id2name(self, actuator_id)
```

<details>
<summary>Docstring</summary>

执行器ID转名称

</details>


###### `OrcaGymModel.init_body_dict`

Signature:

```python
def init_body_dict(self, body_dict)
```

<details>
<summary>Docstring</summary>

初始化 body 字典，建立名称和ID的映射关系

术语说明:
    - Body: MuJoCo 中的刚体，是物理仿真的基本单元
    - 每个 body 有质量、惯性、位置、姿态等属性
    - Body 之间通过关节 (Joint) 连接，形成运动链

使用示例:
    ```python
    # Body 在模型加载时自动初始化
    # 可以通过以下方式访问:
    body_names = list(self.model.get_body_names())
    body_id = self.model.body_name2id("base_link")
    ```

</details>


###### `OrcaGymModel.get_body_dict`

Signature:

```python
def get_body_dict(self)
```

<details>
<summary>Docstring</summary>

获取所有 body 字典

</details>


###### `OrcaGymModel.get_body_byid`

Signature:

```python
def get_body_byid(self, id)
```

<details>
<summary>Docstring</summary>

根据ID获取 body 信息

</details>


###### `OrcaGymModel.get_body_byname`

Signature:

```python
def get_body_byname(self, name)
```

<details>
<summary>Docstring</summary>

根据名称获取 body 信息

</details>


###### `OrcaGymModel.body_name2id`

Signature:

```python
def body_name2id(self, body_name)
```

<details>
<summary>Docstring</summary>

Body 名称转ID

将 body 名称转换为对应的 ID，用于需要 ID 的底层操作。

使用示例:
    ```python
    # 在更新等式约束时使用
    body_id = self.model.body_name2id(actor_name)
    eq["obj2_id"] = body_id
    ```

</details>


###### `OrcaGymModel.body_id2name`

Signature:

```python
def body_id2name(self, body_id)
```

<details>
<summary>Docstring</summary>

Body ID转名称

</details>


###### `OrcaGymModel.init_joint_dict`

Signature:

```python
def init_joint_dict(self, joint_dict)
```

<details>
<summary>Docstring</summary>

初始化关节字典，建立名称和ID的映射关系

术语说明:
    - 关节 (Joint): 连接两个 body 的约束，定义它们之间的相对运动
    - 关节类型: 旋转关节 (revolute)、滑动关节 (prismatic)、自由关节 (free) 等
    - 关节自由度: 关节允许的运动维度，旋转关节1个，滑动关节1个，自由关节6个

使用示例:
    ```python
    # 关节在模型加载时自动初始化
    # 可以通过以下方式访问:
    joint_dict = self.model.get_joint_dict()
    joint_id = self.model.joint_name2id("joint1")
    ```

</details>


###### `OrcaGymModel.get_joint_dict`

Signature:

```python
def get_joint_dict(self)
```

<details>
<summary>Docstring</summary>

获取所有关节字典

</details>


###### `OrcaGymModel.get_joint_byid`

Signature:

```python
def get_joint_byid(self, id)
```

<details>
<summary>Docstring</summary>

根据ID获取关节信息

</details>


###### `OrcaGymModel.get_joint_byname`

Signature:

```python
def get_joint_byname(self, name)
```

<details>
<summary>Docstring</summary>

根据名称获取关节信息

</details>


###### `OrcaGymModel.joint_name2id`

Signature:

```python
def joint_name2id(self, joint_name)
```

<details>
<summary>Docstring</summary>

关节名称转ID

</details>


###### `OrcaGymModel.joint_id2name`

Signature:

```python
def joint_id2name(self, joint_id)
```

<details>
<summary>Docstring</summary>

关节ID转名称

</details>


###### `OrcaGymModel.init_geom_dict`

Signature:

```python
def init_geom_dict(self, geom_dict)
```

<details>
<summary>Docstring</summary>

初始化几何体字典，建立名称和ID的映射关系

</details>


###### `OrcaGymModel.get_geom_dict`

Signature:

```python
def get_geom_dict(self)
```

<details>
<summary>Docstring</summary>

获取所有几何体字典

</details>


###### `OrcaGymModel.get_geom_byid`

Signature:

```python
def get_geom_byid(self, id)
```

<details>
<summary>Docstring</summary>

根据ID获取几何体信息

</details>


###### `OrcaGymModel.get_geom_byname`

Signature:

```python
def get_geom_byname(self, name)
```

<details>
<summary>Docstring</summary>

根据名称获取几何体信息

</details>


###### `OrcaGymModel.geom_name2id`

Signature:

```python
def geom_name2id(self, geom_name)
```

<details>
<summary>Docstring</summary>

几何体名称转ID

</details>


###### `OrcaGymModel.geom_id2name`

Signature:

```python
def geom_id2name(self, geom_id)
```

<details>
<summary>Docstring</summary>

几何体ID转名称

</details>


###### `OrcaGymModel.get_body_names`

Signature:

```python
def get_body_names(self)
```

<details>
<summary>Docstring</summary>

获取所有 body 名称列表

返回可迭代的 body 名称集合，用于查找特定 body 或遍历所有 body。

使用示例:
    ```python
    # 查找包含特定关键词的 body
    all_bodies = self.model.get_body_names()
    for body in all_bodies:
        if "base" in body.lower() and "link" in body.lower():
            self.base_body_name = body
            break
    ```

使用示例:
    ```python
    # 遍历所有 body 进行查询
    for body_name in self.model.get_body_names():
        pos, _, quat = self.get_body_xpos_xmat_xquat([body_name])
    ```

</details>


###### `OrcaGymModel.get_geom_body_name`

Signature:

```python
def get_geom_body_name(self, geom_id)
```

<details>
<summary>Docstring</summary>

根据几何体ID获取其所属的 body 名称

</details>


###### `OrcaGymModel.get_geom_body_id`

Signature:

```python
def get_geom_body_id(self, geom_id)
```

<details>
<summary>Docstring</summary>

根据几何体ID获取其所属的 body ID

</details>


###### `OrcaGymModel.get_actuator_ctrlrange`

Signature:

```python
def get_actuator_ctrlrange(self)
```

<details>
<summary>Docstring</summary>

获取所有执行器的控制范围（用于定义动作空间）

返回形状为 (nu, 2) 的数组，每行包含 [min, max] 控制范围。
常用于在环境初始化时定义 action_space。

术语说明:
    - 动作空间 (Action Space): 强化学习中智能体可以执行的所有动作的集合
    - 控制范围: 执行器能够接受的最小和最大控制值，超出范围会被截断
    - nu: 执行器数量，等于动作空间的维度

使用示例:
    ```python
    # 获取执行器控制范围并定义动作空间
    all_actuator_ctrlrange = self.model.get_actuator_ctrlrange()
    # ctrlrange 形状: (nu, 2)，每行为 [min, max]
    self.action_space = self.generate_action_space(all_actuator_ctrlrange)
    ```

</details>


###### `OrcaGymModel.get_joint_qposrange`

Signature:

```python
def get_joint_qposrange(self, joint_names)
```

<details>
<summary>Docstring</summary>

获取指定关节的位置范围

</details>


###### `OrcaGymModel.init_site_dict`

Signature:

```python
def init_site_dict(self, site_dict)
```

<details>
<summary>Docstring</summary>

初始化 site 字典

术语说明:
    - Site: MuJoCo 中的标记点，用于标记特定位置（如末端执行器、目标点）
    - Site 不参与物理仿真，仅用于查询位置和姿态
    - 常用于: 查询末端执行器位姿、定义目标位置、计算距离等

使用示例:
    ```python
    # Site 在模型加载时自动初始化
    # 可以通过以下方式查询:
    site_pos, site_quat = self.query_site_pos_and_quat(["end_effector"])
    ```

</details>


###### `OrcaGymModel.get_site_dict`

Signature:

```python
def get_site_dict(self)
```

<details>
<summary>Docstring</summary>

获取所有 site 字典

</details>


###### `OrcaGymModel.get_site`

Signature:

```python
def get_site(self, name_or_id)
```

<details>
<summary>Docstring</summary>

根据名称或ID获取 site 信息

</details>


###### `OrcaGymModel.site_name2id`

Signature:

```python
def site_name2id(self, site_name)
```

<details>
<summary>Docstring</summary>

Site 名称转ID

</details>


###### `OrcaGymModel.site_id2name`

Signature:

```python
def site_id2name(self, site_id)
```

<details>
<summary>Docstring</summary>

Site ID转名称

</details>


###### `OrcaGymModel.init_sensor_dict`

Signature:

```python
def init_sensor_dict(self, sensor_dict)
```

<details>
<summary>Docstring</summary>

初始化传感器字典，识别传感器类型

术语说明:
    - 传感器 (Sensor): 用于测量物理量的虚拟设备
    - 常见类型:
        - accelerometer: 加速度计，测量线性加速度
        - gyro: 陀螺仪，测量角速度
        - touch: 触觉传感器，测量接触力
        - velocimeter: 速度计，测量线性速度
        - framequat: 框架四元数，测量姿态

使用示例:
    ```python
    # 传感器在模型加载时自动初始化
    # 可以通过以下方式查询:
    sensor_data = self.query_sensor_data(["imu_accelerometer", "imu_gyro"])
    ```

</details>


###### `OrcaGymModel.gen_sensor_dict`

Signature:

```python
def gen_sensor_dict(self)
```

<details>
<summary>Docstring</summary>

获取所有传感器字典

</details>


###### `OrcaGymModel.get_sensor`

Signature:

```python
def get_sensor(self, name_or_id)
```

<details>
<summary>Docstring</summary>

根据名称或ID获取传感器信息

</details>


###### `OrcaGymModel.sensor_name2id`

Signature:

```python
def sensor_name2id(self, sensor_name)
```

<details>
<summary>Docstring</summary>

传感器名称转ID

</details>


###### `OrcaGymModel.sensor_id2name`

Signature:

```python
def sensor_id2name(self, sensor_id)
```

<details>
<summary>Docstring</summary>

传感器ID转名称

</details>

---

## `orca_gym/core/orca_gym_opt_config.py`

> OrcaGymOptConfig - MuJoCo 仿真器优化配置

<details>
<summary>Module docstring</summary>

OrcaGymOptConfig - MuJoCo 仿真器优化配置

本模块提供 MuJoCo 仿真器优化参数的封装类，用于配置物理仿真器的各种参数。
这些参数影响仿真的精度、稳定性和性能。

使用场景:
    - 在环境初始化时从服务器获取配置
    - 通过 env.gym.opt 访问配置对象
    - 调整物理仿真精度和性能平衡

典型用法:
    ```python
    # 配置通过 OrcaGymLocal 的初始化自动获取
    env = OrcaGymLocalEnv(...)
    # 访问配置
    timestep = env.gym.opt.timestep
    gravity = env.gym.opt.gravity
    solver = env.gym.opt.solver
    ```

</details>


### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymOptConfig` | MuJoCo 仿真器优化配置容器 |

### Classes（详情）

#### `class OrcaGymOptConfig`

> MuJoCo 仿真器优化配置容器

<details>
<summary>Class docstring</summary>

MuJoCo 仿真器优化配置容器

存储 MuJoCo 仿真器的所有配置参数，包括时间步长、求解器、积分器、
碰撞检测、物理参数等。这些参数影响仿真的精度、稳定性和性能。

配置参数分类:
    1. 时间相关: timestep, apirate
    2. 求解器相关: solver, iterations, tolerance
    3. 物理参数: gravity, density, viscosity, wind, magnetic
    4. 接触参数: o_margin, o_solref, o_solimp, o_friction
    5. 积分器: integrator, impratio
    6. 碰撞检测: ccd_tolerance, ccd_iterations
    7. 其他: jacobian, cone, disableflags, enableflags

使用示例:
    ```python
    # 访问时间步长
    dt = self.gym.opt.timestep * self.frame_skip
    
    # 访问重力
    gravity = self.gym.opt.gravity  # [x, y, z]
    ```

</details>
