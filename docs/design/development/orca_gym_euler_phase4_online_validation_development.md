# OrcaGym Euler 阶段四开发指导文档：端到端在线验证

## 1. 文档定位

### 1.1 文档目标

本文是 `OrcaGymEulerEnv` **阶段四（端到端在线验证）** 的开发指导文档。在阶段三（功能填充 + 离线单元测试）已完成并通过验收的基础上，通过在线模式端到端 example 验证各子阶段功能在真实 OrcaStudio 场景中的运行效果。

阶段四的核心目标：

1. **在线 Example**：为阶段三每个子阶段设计在线模式端到端 example，基于 G1 人形机器人场景
2. **在线判定逻辑**：example 脚本中直接嵌入验证逻辑，运行时实时读写 env 数据，判定是否符合设计
3. **双重验证**：同时支持**用户人工观察**（Studio 视口）+ **数值读写判定**（脚本 assert）两种验收方式

> **上游约束**：架构文档 `docs/design/architecture/orca_gym_euler_architecture.md`（§5–§7、§10–§12 为硬性约束）。本文所有 example 的 API 调用必须严格遵守 K1–K12 约束。

### 1.2 与阶段三的关系

| 维度 | 阶段三（已完成） | 阶段四（本文） |
|------|----------------|--------------|
| 目的 | 功能填充 + 功能正确性 | 真实运行效果验证 |
| 测试方式 | 离线单元测试（CPU 加载 XML） | 在线端到端 example（宿主机 + OrcaStudio） |
| 环境 | sandbox 内 `OrcaFlow_Flow`，离线模式 | 宿主机 + OrcaStudio + G1 关卡 |
| 数据来源 | 离线加载 `g1_29dof_camera.xml` 的真实 MuJoCo 数据 | 在线 gRPC 连接 Studio，真实场景运行数据 |
| 判定方式 | unittest assert（数值判定） | 人工观察 + 数值判定（脚本内嵌） |

阶段三的离线单元测试验证**功能正确性**（API 调用链路、数值计算）；阶段四的在线 example 验证**真实运行效果**（含 Studio 交互、gRPC 通信、摄像头流、ONNX 策略推理等在线特性）。

### 1.3 前置条件

- 阶段三全部子阶段（3.1–3.5）验收通过
- OrcaStudio 运行中，已加载含 G1 的关卡
- G1 Euler 专用资源已就位（`OrcaPlayground/envs/euler/robots/`）
- 宿主机 CUDA 环境（ONNX 策略推理如需 GPU）

---

## 2. 端到端 Example 设计（基于 G1 人形机器人）

阶段四所有 Lesson 4–8 **统一采用 G1 人形机器人场景**，基于 Euler 专用模型 `OrcaPlayground/envs/euler/robots/g1_29dof_camera.xml`（在原 G1 模型基础上于头部加装摄像头传感器），在线模式（`skip_grpc_load=False`）。G1 是一个完整的可 locomotion 行走的 29 自由度人形机器人，关节/执行器/传感器齐全，能充分覆盖阶段三各子阶段的验证需求。

> **独立资源原则**：Euler 例子使用独立的模型、mesh、ONNX 策略、配置文件，全部存放在 `OrcaPlayground/envs/euler/robots/` 下，**不与原 `examples/g1/` 共享代码和数据**，避免原例子变更影响 Euler 验证。

### 2.0 G1 场景统一约定

#### 2.0.1 资产准备

| 项 | 说明 |
|----|------|
| 资产包 | **OrcaPlaygroundAssets** 中的 `g1_29dof_old_usda`（源文件 `robots/g1/`），拖入 OrcaStudio 场景 |
| 是否手动拖入布局 | **是**，运行前先把 G1 摆进场景（参考 `examples/g1/README.md` 的拖入说明） |
| Euler 专用模型 XML | `OrcaPlayground/envs/euler/robots/g1_29dof_camera.xml`（29 自由度 + free base + 头部摄像头） |
| Euler 专用资源目录 | `OrcaPlayground/envs/euler/robots/`（含 `meshes/`、`models/`、`config/`，从原 G1 例子拷贝独立存档） |
| 场景要求 | 场景中需要且只能有 1 台完整匹配的 G1；机器人实例名不固定，脚本自动扫描绑定 |
| 摄像头传感器 | **已内置**：`g1_29dof_camera.xml` 在头部 `camera_head` body 内定义了 `<camera name="camera_head" user="7070 7071" fovy="75"/>`。`user="7070 7071"` 声明 RGBD 相机的 **color 流端口 7070 / depth 流端口 7071**（Studio 加载场景时识别该字段并启动对应 WebSocket 流服务，相机即被使能），机器人拖入场景后即可使用全部 camera 传感器功能，**无需额外配置**。两套采集 API 见 §2.0.5 |

#### 2.0.2 G1 机器人结构概览

**关节（30 个）**：

| 部位 | 关节数 | 关节名（后缀） |
|------|--------|--------------|
| free base | 1 | `floating_base_joint`（pelvis，type=free） |
| 腿部 ×2 | 12 | `{left,right}_hip_pitch/roll/yaw_joint`、`{left,right}_knee_joint`、`{left,right}_ankle_pitch/roll_joint` |
| 腰部 | 3 | `waist_yaw/roll/pitch_joint` |
| 手臂 ×2 | 14 | `{left,right}_shoulder_pitch/roll/yaw_joint`、`{left,right}_elbow_joint`、`{left,right}_wrist_roll/pitch/yaw_joint` |

**执行器（29 个 motor）**：对应 29 个旋转关节，ctrlrange 按部位分档（腿 ±88/±139/±50，腰 ±88/±50，臂 ±25/±5）。

**传感器**：
- `jointpos`（29 个）：各关节位置
- `jointvel`（29 个）：各关节速度
- `jointactuatorfrc`（29 个）：各关节力矩
- `imu_quat`（framequat, site=imu）：IMU 四元数
- `imu_gyro`（gyro, site=imu）：IMU 角速度
- `imu_acc`（accelerometer, site=imu）：IMU 加速度
- `frame_pos`（framepos, site=imu）：imu site 位置
- `frame_vel`（framelinvel, site=imu）：imu site 速度

**摄像头传感器（Euler 专用 XML 新增）**：
- body `camera_head`（位于 torso_link 上方，pos=`0.00 0 0.4`，euler=`0.0 0 -1.570`）
- site `camera_head_site`（size=0.01, group=3）
- camera `camera_head`（`user="7070 7071"` 声明 RGBD 相机 **color 流端口 7070 / depth 流端口 7071**；`fovy="75"`）
- 拖入 OrcaStudio 场景后，Studio 识别 `user` 字段，在对应端口启动 WebSocket 视频流服务，相机即被使能；脚本通过端口连接即可取流（见 §2.0.5）

**site**：`imu`（位于 torso_link，pos=`-0.03959 -0.00224 0.13792`）、`camera_head_site`（位于 camera_head body）

**关键 body**：
- `pelvis`（基底，free joint，世界系位姿代表整机位姿）
- `torso_link`（躯干，含 imu site 与 camera_head body，质心 9.6kg）
- `camera_head`（头部摄像头载体，含 camera 传感器）
- `{left,right}_ankle_roll_link`（脚底，含 4 个接触球 geom）
- `{left,right}_rubber_hand`（手末端，含 mesh）

**接触**：脚底 ankle_roll_link 有 4 个 `size=0.005` 球 geom，与地面产生接触。

**XML 内置测试对象**（用于 Lesson 5 的 `set_mocap_pos_and_quat` 与 Lesson 8 的 equality/mocap 驱动验证）：

| 对象 | XML 名称 | 类型 | 用途 |
|------|---------|------|------|
| 测试 box | `manipulation_box` | body + free joint + box geom（pos=`0.5 0 0.3`，size=0.05，mass=0.5） | 被驱动对象，验证 weld 约束效果 |
| mocap anchor | `ActorManipulator_Anchor` | body (`mocap="true"`，pos=`0.5 0 0.3`) + sphere geom（size=0.02，contype=0/conaffinity=0） | 驱动源，通过 `set_mocap_pos_and_quat` 写入位姿 |
| weld 约束 | `anchor_box_weld` | `<equality><weld body1="ActorManipulator_Anchor" body2="manipulation_box" active="true"/></equality>` | 绑定 anchor↔box，可通过 `modify_equality_objects` 重绑 obj2id 到 G1 任意 body（如 pelvis）验证 mocap 驱动 G1 动力学 |

> **设计要点**：这三个测试对象是独立于 G1 本体的辅助 fixture，不干扰 G1 的 29 自由度行走控制。`anchor_box_weld` 的 `body2` 默认指向 `manipulation_box`，在线测试中可通过 `modify_equality_objects(eq_name="anchor_box_weld", obj2id=pelvis_id)` 将约束重绑到 G1 pelvis，验证 mocap 驱动 G1 动力学（不增删 equality 节点，仅改 obj1id/obj2id）。

#### 2.0.3 统一运行配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `orcagym_addr` | `127.0.0.1:50051` | OrcaStudio gRPC 地址 |
| `skip_grpc_load` | `False` | 在线模式 |
| `time_step` | `0.001` | 物理步长 1ms（与 G1 example 一致） |
| `frame_skip` | `20` | 控制频率 50Hz（与 G1 example 一致） |
| `render_mode` | `human` | 渲染到 Studio 视口 |
| agent_name 解析 | 脚本扫描场景自动识别 | 参考 `run_g1_sim.py` 的 `resolve_g1_scene_agent_name` |

#### 2.0.4 场景扫描与 agent_name 解析

所有 Lesson 复用 `envs/common/model_scanner.py` 的扫描机制（参考 `run_g1_sim.py`），通过关节/执行器/传感器后缀模板匹配场景中的 G1 实例：

```python
G1_JOINT_SUFFIXES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", ... , "floating_base_joint",
]  # 30 个
G1_ACTUATOR_SUFFIXES = ["left_hip_pitch", ..., "right_wrist_yaw"]  # 29 个
G1_SENSOR_SUFFIXES = ["imu_quat", "imu_gyro"]  # 最小匹配集

template = build_suffix_template("G1", G1_JOINT_SUFFIXES, G1_ACTUATOR_SUFFIXES, G1_SENSOR_SUFFIXES)
agent_name = scan_scene_for_template(orcagym_addr, time_step, template)
```

> **注意**：各 Lesson 的 Env 子类继承 `OrcaGymEulerEnv`，在 `initialize_simulation` 中加载 `envs/euler/robots/g1_29dof_camera.xml`，并通过扫描得到的 agent_name 前缀拼接完整关节/body 名称（如 `f"{agent_name}/torso_link"`）。

#### 2.0.5 摄像头传感器 API 激活与采集流程

G1 拖入 OrcaStudio 场景后，Studio 读取 `<camera user="7070 7071">` 中的端口字段，在 color 端口（7070）/ depth 端口（7071）启动 WebSocket 视频流服务，`camera_head` 即被使能，**无需额外激活调用**。

采集数据有**两套 API**，按用途选择：

**套 A：gRPC 录制/抓帧（参考 `orca_gym/core/orca_gym_local.py`、`examples/d12/act/run_d12_act.py`、`OrcaManipulation/src/dataStorage/abstract_data_storage.py`）**——适合离线录制 mp4、单帧抓 PNG、帧索引/时间戳查询。

```python
# 1. 开始录制视频（gRPC BeginSaveMp4File）
#    path 是**目录**（不是文件名）：Studio 在该目录下生成 mp4 文件
#    参考 OrcaManipulation：os.makedirs(video_dir, exist_ok=True) 后传目录给 env.begin_save_video
import os
video_dir = "/tmp/g1_walk_video"
os.makedirs(video_dir, exist_ok=True)
env.begin_save_video(video_dir, capture_mode=CaptureMode.ASYNC)

# 2. 轮询帧索引（gRPC GetCurrentFrameIndex）
#    get_current_frame() 返回 -1 表示摄像头未使能
#    get_next_frame() 带轮询等待下一帧（内部 sleep realtime_step，最多等 10 次）
frame_idx = env.get_next_frame()

# 3. 抓帧 PNG 到目录（gRPC GetCameraFramePNG）—— 视频截帧
#    返回 {camera_name: {"pos":..., "quat":...}} dict，PNG 写入 {image_path}/color/{camera_name}_color_0.png
camera_info = env.get_frame_png("/tmp/g1_frames")
available_cams = list(camera_info.keys())  # ["camera_head"]

# 4. 查询相机时间戳（gRPC GetTimeStamp）
timestamps = env.get_camera_time_stamp(last_frame_index=frame_idx)

# 5. 停止录制（gRPC StopSaveMp4File）—— mp4 文件在 video_dir 下生成
env.stop_save_video()
```

> **OrcaManipulation 录制模式参考**：`DataCollectionManager` 在 task RUNNING 时调 `begin_save_video`，task END 时调 `stop_save_video`，整段录制 mp4；回放脚本（`data_collection_replay.py`）同样 `save_video=True` 录制回放过程用于验证轨迹。Euler Lesson 7 借鉴此模式：行走开始时 `begin_save_video`，行走结束时 `stop_save_video`。

**套 B：WebSocket 实时流（参考 `orca_gym/sensor/rgbd_camera.py` 与 `envs/aloha/aloha_orcagym_task.py`）**——适合在线策略推理时同步取帧（后台线程解码 H.264，`get_frame()` 返回最新帧）。

```python
from orca_gym.sensor.rgbd_camera import CameraWrapper

# camera_config: {camera_name: color_port}，端口取自 XML 的 user 字段第一个值
camera_config = {"camera_head": 7070}
cameras = [CameraWrapper(name=name, port=port) for name, port in camera_config.items()]
for cam in cameras:
    cam.start()  # 后台线程连接 ws://localhost:{port}，解码 H.264 流

# 取最新帧（bgr24 ndarray + image_index）
frame, idx = cameras[0].get_frame(format="bgr24", size=(640, 480))

# 结束
for cam in cameras:
    cam.stop()
```

**关键点**：
- **端口是 `user` 字段，不是分辨率**：`user="7070 7071"` 表示 color 流端口 7070、depth 流端口 7071（RGBD = RGB + Depth）。`CameraWrapper` 默认连 color 端口取 RGB 流。
- `get_current_frame()` 返回 `-1` 表示摄像头未使能（检查 XML 的 `<camera user="color_port depth_port">` 标签与 Studio 场景是否加载了含摄像头的模型）
- `get_frame_png(path)` 是异步的：调用后需轮询 `{path}/color/` 目录直到出现 PNG 文件且文件大小稳定（参考 `run_d12_act.py` 的 `max_wait=0.5s` 轮询逻辑）
- PNG 文件名格式为 `{camera_name}_color_0.png`，多摄像头时每个 camera 一个文件
- `CaptureMode` 枚举（`ASYNC`/`SYNC`）控制录制模式，默认 `ASYNC`
- **Lesson 7 推荐**：行走录制用套 A（`begin_save_video` + `get_frame_png`）；若策略需要实时视觉反馈，叠加套 B（`CameraWrapper`）

### 2.1 Lesson 4：状态查询 API 验证（G1 关节/body/site/IMU/接触）

**文件**：`OrcaPlayground/examples/euler/04_query_api/query_api.py`

**验证内容**：G1 全套状态查询——29 关节 qpos/qvel/qacc、关键 body 位姿、imu site、IMU/关节传感器、脚底接触、pelvis 基座坐标系变换。

**验证点**：
1. `query_joint_qpos(["{agent}/left_hip_pitch_joint", ..., "{agent}/right_wrist_yaw_joint"])` 返回 29 个关节角度，形状正确（每个 hinge joint 长度 1）
2. `query_joint_qvel/qacc` 同理，与 `env.data.qvel/qacc` 对应切片一致
3. `get_body_xpos_xmat_xquat(["{agent}/pelvis", "{agent}/torso_link", "{agent}/left_ankle_roll_link"])` 返回扁平数组，pelvis z≈0.793（初始高度）
4. `query_site_pos_and_quat(["{agent}/imu"])` 返回 imu site 位姿，位于 torso_link 内
5. `query_sensor_data(["{agent}/imu_quat", "{agent}/imu_gyro", "{agent}/imu_acc", "{agent}/left_hip_pitch_pos"])` 返回正确维度数据（quat=4, gyro=3, acc=3, jointpos=1）
6. `query_actuator_torques(["{agent}/left_hip_pitch", ...])` 返回 29 个执行器力矩
7. `query_contact_simple()` 返回脚底接触列表（G1 站立时 ankle_roll_link 与地面接触）
8. `query_position_body_B("{agent}/torso_link", "{agent}/pelvis")` 返回躯干相对骨盆的位置（基座坐标系变换）
9. `body_subtree_mass("{agent}/torso_link")` 返回正标量（含躯干+双臂质量）

### 2.2 Lesson 5：外力应用与状态设置验证（G1 推力/摩擦/足部接触/mocap 位姿写入）

**文件**：`OrcaPlayground/examples/euler/05_force_apply/force_apply.py`

**验证内容**：对 G1 躯干/骨盆施力、清力、修改脚底摩擦、查询接触力，并通过 XML 内置的 mocap body 验证 `set_mocap_pos_and_quat`。

> **说明**：`g1_29dof_camera.xml` 已内置 mocap body `ActorManipulator_Anchor`（`mocap="true"`）+ 测试 box `manipulation_box`（free joint）+ weld 等式约束 `anchor_box_weld`（绑定 anchor↔box，详见 §2.0.2）。本课在验证 `apply_body_force`/`clear_*`/`set_geom_friction`/`query_contact_force` 的基础上，**额外验证 `set_mocap_pos_and_quat` 驱动 mocap body**——写入 mocap 位姿后，weld 约束使 box 跟随移动（不需要 OrcaStudio 场景锚点）。

**验证点**：
1. `apply_body_force("{agent}/torso_link", [0, 0, 200], [0, 0, 0])` 后 G1 被抬起（pelvis z 上升），`env.data.xfrc_applied` 可读到力值
2. `clear_body_force("{agent}/torso_link")` 后 xfrc 清零，G1 自由落体
3. `clear_all_forces()` 清除全部外力
4. `set_geom_friction({"{agent}/left_ankle_roll_link_geom0": [0.01, 0.005, 0.0001]})` 降低脚底摩擦后，G1 站立稳定性下降（可能滑动）
5. `query_contact_force(contact_ids)` 返回脚底接触力（法向力 ≈ G1 重力的一半，单脚）
6. `apply_body_force("{agent}/pelvis", [50, 0, 0], [0, 0, 0])` 侧向推力使 G1 倾倒，观察接触变化
7. `set_mocap_pos_and_quat("ActorManipulator_Anchor", [0.7, 0, 0.5], [1,0,0,0])` 写入 mocap 位姿后，`env.data.mocap_pos[0]` 读回值一致；weld 约束驱动 box，步进 100 帧后 `get_body_xpos_xmat_xquat(["manipulation_box"])` 的 xpos ≈ `[0.7, 0, 0.5]`（atol=0.05）

### 2.3 Lesson 6：雅可比与逆运动学验证（G1 足端/躯干雅可比）

**文件**：`OrcaPlayground/examples/euler/06_jacobian/jacobian_ik.py`

**验证内容**：G1 雅可比计算（足端 body、imu site）、基于雅可比的躯干姿态控制或足端位置控制。

**验证点**：
1. `mj_jacBody(jacp, jacr, body_id=pelvis_id)` 返回 `(3, nv)` 雅可比矩阵，nv=35（6 free + 29 旋转）
2. `mj_jacSite(jacp, jacr, "{agent}/imu")` 返回 imu site 雅可比
3. `query_site_xvalp_xvalr(["{agent}/imu"])` 返回 imu site 速度，与 `jacp @ env.data.qvel` 数值一致
4. **IK 演示**：以左腿为目标，计算 `left_ankle_roll_link` 相对 pelvis 的雅可比，用伪逆 `qvel_leg = pinv(jacp_leg) @ (target_foot_pos - foot_pos)` 迭代调整左腿关节，使左脚到达目标位置（在 Studio 视口可见左脚移动）
5. `mj_jac_site(["{agent}/imu", "{agent}/left_ankle_roll_link"])` 批量返回多个雅可比

### 2.4 Lesson 7：Studio 视频录制与截帧验证（G1 行走录制）

**文件**：`OrcaPlayground/examples/euler/07_studio_capture/studio_capture.py`

**验证内容**：运行 G1 行走控制程序的同时，验证 camera 传感器的**视频录制**（`begin_save_video`/`stop_save_video`）与**视频截帧**（`get_frame_png`）两项功能，并查询帧索引/时间戳。

> **摄像头已内置**：`g1_29dof_camera.xml` 头部 `camera_head` body 已含摄像头传感器（`user="7070 7071"` 声明 color/depth 端口），G1 拖入场景后自动使能，**无需额外配置**。API 流程见 §2.0.5。
>
> **行走控制**：本例子运行 G1 行走控制程序（基于 ONNX 策略 locomotion），参考原 `examples/g1/run_g1_sim.py` 的控制逻辑，但使用 `envs/euler/robots/` 下的独立模型与策略资源。脚本结构：加载 G1 Euler Env → 扫描 agent_name → 加载 `models/dec_loco/model_6600.onnx` 策略 → 行走开始时 `begin_save_video` → 循环步进（策略输出 ctrl + 中途 `get_frame_png` 截帧）→ 行走结束时 `stop_save_video`。
>
> **录制模式参考**：借鉴 `OrcaManipulation` 的 `DataCollectionManager` 模式——task 开始时 `begin_save_video(video_dir)`，task 结束时 `stop_save_video()`，整段录制 mp4 用于验证行走效果（类似 OrcaManipulation 回放脚本 `data_collection_replay.py` 录制回放视频验证轨迹）。

**验证点**：
1. **摄像头使能检查**：`get_current_frame()` 初始返回值 ≥ 0（若返回 -1 说明摄像头未使能，检查场景是否加载了含 `camera_head` 的 G1、XML `user` 端口字段是否正确）
2. **视频录制**：`begin_save_video("/tmp/g1_walk_video", capture_mode=CaptureMode.ASYNC)`（path 是**目录**，参考 OrcaManipulation 用法）开始录制后，步进若干帧（如 500 帧 = 10 秒），`get_next_frame()` 帧索引递增
3. **视频截帧**：行走中途 `get_frame_png("/tmp/g1_frames")` 抓帧，返回 `{"camera_head": {"pos":..., "quat":...}}`，轮询 `/tmp/g1_frames/color/camera_head_color_0.png` 文件生成且大小稳定（参考 `run_d12_act.py` 的 `max_wait=0.5s` 轮询）
4. **时间戳查询**：`get_camera_time_stamp(last_frame_index)` 返回时间戳字典（键为相机名，值为时间戳列表）
5. **录制完成**：`stop_save_video()` 后 `/tmp/g1_walk_video/` 目录下生成 mp4 文件，包含 G1 行走画面
6. **内容验证**：录制期间 G1 在策略控制下行走，mp4 视频与截帧 PNG 内容应可见 locomotion 运动

### 2.5 Lesson 8：完整体操作与 Studio 拖拽验证（G1）

**文件**：`OrcaPlayground/examples/euler/08_body_manipulation/body_manipulation.py`

**验证内容**：运行 G1 行走控制程序的同时，验证 Studio UI 拖拽 G1 机体、锚定/释放、等式约束更新，并通过 XML 内置的 mocap+weld 验证 `modify_equality_objects` 重绑定驱动 G1 动力学。

> **依赖**：`g1_29dof_camera.xml` 已内置 mocap body `ActorManipulator_Anchor` + weld 等式约束 `anchor_box_weld`（默认绑定 anchor↔box，详见 §2.0.2）。本课在 Studio UI 拖拽验证之外，**额外通过 `modify_equality_objects` 将 weld 的 obj2id 从 box 重绑到 G1 pelvis**，验证 mocap 驱动 G1 动力学（不增删 equality 节点，仅改 obj1id/obj2id）。若 OrcaStudio 场景也提供锚点 body，二者可共存。
>
> **行走控制**：与 Lesson 7 相同，本例子运行 G1 行走控制程序（基于 ONNX 策略 locomotion），参考原 `examples/g1/run_g1_sim.py` 的控制逻辑，使用 `envs/euler/robots/` 下的独立资源。在行走过程中测试体操作：拖拽行走中的 G1、锚定后释放、观察其恢复行走。

**验证点**：
1. 在 Studio UI 中用鼠标拖拽 G1 的 pelvis 或 torso_link，`do_body_manipulation()` 检测到拖拽并锚定该 body
2. 锚定后 G1 跟随鼠标移动/旋转（mocap + equality 联动），行走控制暂停
3. 释放鼠标，`release_body_anchored()` 清除锚定，G1 恢复物理仿真与策略控制
4. `update_equality_constraints(...)` 切换锚点约束类型（weld/ball），观察 G1 跟随行为差异
5. `anchor_actor("{agent}/pelvis", anchor_type)` 程序化锚定 pelvis，G1 悬停在指定位置
6. 拖拽过程中 `query_body_xpos_xmat_xquat(["{agent}/pelvis"])` 实时返回位姿变化
7. **mocap 驱动 box（默认绑定）**：`set_mocap_pos_and_quat("ActorManipulator_Anchor", [0.7, 0, 0.5], [1,0,0,0])` 后步进 100 帧，`manipulation_box` 的 xpos ≈ `[0.7, 0, 0.5]`（atol=0.05），Studio 视口可见球体 anchor 拖动 box
8. **停用 equality 解耦**：`update_equality_constraints(eq_name="anchor_box_weld", active=False)` 后移动 mocap，box 不再跟随（自由落体或保持原位）
9. **重绑 equality 驱动 G1 pelvis**：`modify_equality_objects(eq_name="anchor_box_weld", obj2id=pelvis_id)` 将 weld 的 obj2id 从 box 改为 pelvis，`set_mocap_pos_and_quat("ActorManipulator_Anchor", pelvis_pos+[0.2,0,0.1])` 后步进 200 帧，G1 pelvis 位移 > 0.05m（Studio 视口可见 G1 被 mocap 拖动）—— **此为 mocap 驱动 G1 动力学的核心验证**

### 2.6 Example 目录结构

```
OrcaPlayground/examples/euler/
├── 04_query_api/
│   └── query_api.py              # Lesson 4：G1 状态查询（在线）
├── 05_force_apply/
│   └── force_apply.py            # Lesson 5：G1 外力应用（在线）
├── 06_jacobian/
│   └── jacobian_ik.py            # Lesson 6：G1 雅可比 IK（在线）
├── 07_studio_capture/
│   └── studio_capture.py         # Lesson 7：G1 行走录制（在线，camera 已内置）
└── 08_body_manipulation/
    └── body_manipulation.py      # Lesson 8：G1 行走中 Studio 拖拽（在线）

OrcaPlayground/envs/euler/
├── simple_env.py                 # SimpleEulerEnv（已有）
├── g1_base_env.py                # G1 Euler 基类（新增，加载 g1_29dof_camera.xml + 场景扫描 + 行走控制）
├── g1_locomotion.py              # G1 行走控制逻辑（新增，ONNX 策略推理 + 键盘控制，参考 run_g1_sim.py）
├── query_api_env.py              # Lesson 4 Env 子类（继承 g1_base_env）
├── force_apply_env.py            # Lesson 5 Env 子类
├── jacobian_env.py               # Lesson 6 Env 子类
├── studio_capture_env.py         # Lesson 7 Env 子类（行走 + 视频采集）
├── body_manipulation_env.py      # Lesson 8 Env 子类（行走 + 体操作）
├── online_verifier.py            # 在线判定框架（新增，见 §4）
└── robots/                       # Euler 专用 G1 资源（独立存档，不与 examples/g1 共享）
    ├── g1_29dof_camera.xml       # G1 模型（含头部摄像头）
    ├── meshes/                   # STL 网格（60 个）
    ├── models/                   # ONNX 策略（dec_loco + mimic 动作集）
    ├── config/                   # 运行配置（g1_29dof_hist.yaml）
    └── requirements.txt          # 依赖清单
```

> **G1 Env 基类设计**：`g1_base_env.py` 封装 G1 场景的公共逻辑——加载 `robots/g1_29dof_camera.xml`、调用 `model_scanner` 解析 agent_name、定义 G1 关节/执行器/传感器后缀常量。`g1_locomotion.py` 封装行走控制逻辑（ONNX 策略推理、键盘控制、站立/行走切换），供 Lesson 7/8 复用。各 Lesson 的 Env 子类继承 `g1_base_env`，按需重写 `reset_model`/`step`/`reward`。

---

## 3. Example 运行逻辑设计

### 3.1 统一运行框架

所有 Lesson 4–8 共用一个统一的运行框架，由 `g1_base_env.py` 的 `G1BaseEnv` 提供公共逻辑，各 Lesson 子类按需重写钩子方法。运行框架的核心职责：

1. **启动**：创建 Env → 扫描 agent_name → 加载策略（如需）→ 初始化 `OnlineVerifier`
2. **循环**：按 frame_skip 步进，每个控制周期插入验证点（数值判定 + 人工观察提示）
3. **结束**：输出判定报告，清理资源

```python
# g1_base_env.py 核心运行框架（伪代码）
class G1BaseEnv(OrcaGymEulerEnv):

    def run_lesson(self, num_steps: int, verifier: OnlineVerifier):
        """统一运行入口：子类通过重写 verify_step/observe_step 插入验证逻辑"""
        self.reset()
        verifier.observe("start", "请在 Studio 视口观察 G1 初始姿态：应站立在地面上")

        for step in range(num_steps):
            # 1. 策略推理（Lesson 7/8 需要，Lesson 4/5/6 可空操作）
            ctrl = self.compute_ctrl(step)
            self.do_simulation(ctrl, self.frame_skip)

            # 2. 数值判定（子类重写）
            self.verify_step(step, verifier)

            # 3. 人工观察提示（子类重写，阶段性打印）
            self.observe_step(step, verifier)

            # 4. 渲染
            self.render()

        # 5. 结束判定 + 报告
        self.verify_final(verifier)
        verifier.report()

    def compute_ctrl(self, step: int) -> np.ndarray:
        """子类重写：Lesson 4/5/6 返回零控；Lesson 7/8 返回 ONNX 策略输出"""
        return np.zeros(self.action_dim)

    def verify_step(self, step: int, verifier: OnlineVerifier):
        """子类重写：每个控制周期插入数值判定"""
        pass

    def observe_step(self, step: int, verifier: OnlineVerifier):
        """子类重写：阶段性打印人工观察提示"""
        pass

    def verify_final(self, verifier: OnlineVerifier):
        """子类重写：运行结束后的最终判定"""
        pass
```

### 3.2 启动流程

所有 Lesson 的启动流程一致，封装在 `G1BaseEnv.initialize_simulation` 中：

```python
# query_api.py 启动流程（所有 Lesson 通用）
from envs.euler.g1_base_env import G1BaseEnv
from envs.euler.online_verifier import OnlineVerifier
from envs.common.model_scanner import scan_scene_for_template, build_suffix_template

class QueryApiEnv(G1BaseEnv):
    def run(self):
        verifier = OnlineVerifier("Lesson 4: 状态查询 API")
        self.run_lesson(num_steps=100, verifier=verifier)

if __name__ == "__main__":
    env = QueryApiEnv(
        orcagym_addr="127.0.0.1:50051",
        time_step=0.001,
        frame_skip=20,
        model_path="envs/euler/robots/g1_29dof_camera.xml",
    )
    env.run()
```

### 3.3 循环步进逻辑

每个控制周期（frame_skip=20，即 20ms / 50Hz）执行：

| 步骤 | 动作 | 说明 |
|------|------|------|
| 1 | `ctrl = self.compute_ctrl(step)` | 策略推理（Lesson 4/5/6 零控，Lesson 7/8 ONNX） |
| 2 | `self.do_simulation(ctrl, self.frame_skip)` | 物理步进 |
| 3 | `self.verify_step(step, verifier)` | 数值判定：读写 env 数据，assert 预期 |
| 4 | `self.observe_step(step, verifier)` | 人工观察：阶段性打印 Studio 视口提示 |
| 5 | `self.render()` | 渲染到 Studio 视口 |

### 3.4 行走控制集成（Lesson 7/8）

Lesson 7/8 需要运行 G1 行走控制程序，复用 `g1_locomotion.py` 的 ONNX 策略推理逻辑：

```python
# g1_locomotion.py 行走控制核心（参考 run_g1_sim.py）
class G1Locomotion:
    def __init__(self, policy_path: str, agent_name: str, env):
        self.policy = onnxruntime.InferenceSession(policy_path)
        self.agent_name = agent_name
        self.env = env
        # ... 策略状态初始化（hist 长度等，参考 config/g1_29dof_hist.yaml）

    def compute_action(self, obs: dict) -> np.ndarray:
        """观测 → ONNX 推理 → ctrl 输出"""
        # 1. 组装观测向量（关节角、角速度、IMU、上一帧动作等）
        obs_vec = self._build_obs(obs)
        # 2. ONNX 推理
        action = self.policy.run(None, {"obs": obs_vec[None, :]})[0][0]
        # 3. 动作后处理（clip、缩放等）
        return np.clip(action, self.ctrl_low, self.ctrl_high)
```

Lesson 7/8 的 Env 子类重写 `compute_ctrl`：

```python
class StudioCaptureEnv(G1BaseEnv):
    def initialize_simulation(self):
        super().initialize_simulation()
        self.locomotion = G1Locomotion(
            policy_path="envs/euler/robots/models/dec_loco/model_6600.onnx",
            agent_name=self.agent_name,
            env=self,
        )

    def compute_ctrl(self, step: int) -> np.ndarray:
        obs = self._get_obs()
        return self.locomotion.compute_action(obs)
```

---

## 4. 在线判定逻辑设计

### 4.1 判定原则：人工观察 + 数值读写判定

每个 Lesson 的 example 脚本同时执行两种验证：

| 验证类型 | 机制 | 输出 |
|---------|------|------|
| **数值读写判定** | 脚本运行时实时读写 `env.data`/查询 API，与预期值比较（`np.allclose`、范围检查、维度检查） | pass/fail + 数值详情 |
| **人工观察** | 脚本阶段性打印 Studio 视口观察提示，引导用户确认视觉效果 | 观察项 checklist（用户手动确认） |

**设计原则**：
1. **判定逻辑内嵌脚本**：不依赖外部测试框架，example 脚本自身即验证脚本
2. **实时读写数据**：运行循环中直接调 `env.query_*`/读 `env.data`，与预期值比较
3. **容差合理**：物理仿真有数值误差，用 `np.allclose(atol=1e-3)` 或范围检查而非精确相等
4. **报告可追溯**：运行结束输出 JSON 报告，含每个判定项的 pass/fail + 实际值 + 预期值

### 4.2 数值判定框架

`OnlineVerifier` 类封装判定收集与报告输出，所有 Lesson 共用：

```python
# online_verifier.py
import json
import numpy as np
from datetime import datetime

class OnlineVerifier:
    """在线判定器：运行中收集判定项，结束后输出报告"""

    def __init__(self, lesson_name: str):
        self.lesson_name = lesson_name
        self.checks = []       # [{"name", "passed", "actual", "expected", "detail"}]
        self.observations = [] # [{"name", "prompt", "step"}]
        self._observed_names = set()

    def check(self, name: str, condition: bool, actual=None, expected=None, detail: str = ""):
        """数值判定：condition 为 True 则通过"""
        self.checks.append({
            "name": name,
            "passed": bool(condition),
            "actual": str(actual) if actual is not None else None,
            "expected": str(expected) if expected is not None else None,
            "detail": detail,
        })
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}: actual={actual}, expected={expected} {detail}")

    def check_allclose(self, name: str, actual, expected, atol=1e-3, detail: str = ""):
        """数值近似判定（np.allclose 封装）"""
        actual_arr = np.asarray(actual)
        expected_arr = np.asarray(expected)
        if actual_arr.shape != expected_arr.shape:
            self.check(name, False, actual_arr.shape, expected_arr.shape, f"shape mismatch {detail}")
            return
        passed = np.allclose(actual_arr, expected_arr, atol=atol)
        self.check(name, passed, actual_arr.tolist(), expected_arr.tolist(),
                   f"atol={atol} {detail}")

    def check_range(self, name: str, value, low, high, detail: str = ""):
        """范围判定：low <= value <= high"""
        passed = low <= value <= high
        self.check(name, passed, value, f"[{low}, {high}]", detail)

    def observe(self, name: str, prompt: str, step: int = 0):
        """人工观察项：打印提示，等用户在 Studio 视口确认"""
        if name not in self._observed_names:
            self.observations.append({"name": name, "prompt": prompt, "step": step})
            self._observed_names.add(name)
        print(f"  [OBSERVE] {name}: {prompt}")

    def report(self) -> dict:
        """输出判定报告并返回 JSON"""
        passed_count = sum(1 for c in self.checks if c["passed"])
        total = len(self.checks)
        all_passed = passed_count == total

        report = {
            "lesson": self.lesson_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total,
                "passed": passed_count,
                "failed": total - passed_count,
                "all_passed": all_passed,
            },
            "checks": self.checks,
            "observations": self.observations,
        }

        print("\n" + "=" * 60)
        print(f"判定报告: {self.lesson_name}")
        print(f"数值判定: {passed_count}/{total} passed")
        print(f"人工观察: {len(self.observations)} 项（请在上方 [OBSERVE] 提示处确认）")
        print(f"总结: {'ALL PASS' if all_passed else 'SOME FAILED'}")
        print("=" * 60)

        # 写入 JSON 文件
        report_path = f"/tmp/euler_{self.lesson_name.replace(' ', '_').replace(':', '')}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已写入: {report_path}")

        return report
```

### 4.3 各 Lesson 判定标准与 assert 设计

每个 Lesson 的 Env 子类重写 `verify_step`/`verify_final`/`observe_step`，嵌入具体判定逻辑。

#### 4.3.1 Lesson 4：状态查询判定

```python
class QueryApiEnv(G1BaseEnv):
    def verify_step(self, step: int, verifier: OnlineVerifier):
        agent = self.agent_name
        if step == 0:
            # 1. 关节 qpos 维度（29 个 hinge joint，每个长度 1）
            joint_names = [f"{agent}/{s}" for s in G1_ROT_JOINT_SUFFIXES]
            qpos = self.query_joint_qpos(joint_names)
            verifier.check("joint_qpos_dim", len(qpos) == 29, len(qpos), 29)

            # 2. qpos 与 env.data.qpos 切片一致
            expected = self.data.qpos[7:]  # 跳过 free base 前 7 维
            verifier.check_allclose("joint_qpos_vs_data", qpos, expected, atol=1e-6)

            # 3. pelvis 初始高度 ≈ 0.793
            pelvis = self.get_body_xpos_xmat_xquat([f"{agent}/pelvis"])
            pelvis_z = pelvis[0][2]
            verifier.check_range("pelvis_initial_height", pelvis_z, 0.75, 0.85,
                                 "G1 站立初始高度")

            # 4. IMU sensor 维度
            imu_quat = self.query_sensor_data([f"{agent}/imu_quat"])
            verifier.check("imu_quat_dim", len(imu_quat[0]) == 4, len(imu_quat[0]), 4)

            # 5. body_subtree_mass 为正
            torso_mass = self.body_subtree_mass(f"{agent}/torso_link")
            verifier.check("torso_subtree_mass_positive", torso_mass > 0, torso_mass, ">0")

            # 6. 基座坐标系变换：torso 相对 pelvis 的位置
            torso_B = self.query_position_body_B(f"{agent}/torso_link", f"{agent}/pelvis")
            verifier.check_range("torso_rel_pelvis_z", torso_B[2], 0.1, 0.3,
                                 "躯干在骨盆上方")

    def observe_step(self, step: int, verifier: OnlineVerifier):
        if step == 0:
            verifier.observe("g1_standing", "Studio 视口：G1 应站立在地面上，双臂自然下垂")
        if step == 50:
            verifier.observe("g1_stable", "Studio 视口：G1 应保持稳定站立，无抖动/倾倒")
```

#### 4.3.2 Lesson 5：外力应用判定

```python
class ForceApplyEnv(G1BaseEnv):
    def verify_step(self, step: int, verifier: OnlineVerifier):
        agent = self.agent_name

        if step == 10:
            # 施力前：记录 pelvis z
            self._z_before = self.get_body_xpos_xmat_xquat([f"{agent}/pelvis"])[0][2]

            # 施加向上力
            self.apply_body_force(f"{agent}/torso_link", [0, 0, 200], [0, 0, 0])

        elif step == 30:
            # 施力后：pelvis z 应上升
            z_after = self.get_body_xpos_xmat_xquat([f"{agent}/pelvis"])[0][2]
            verifier.check("force_lift_pelvis", z_after > self._z_before + 0.01,
                           z_after, f">{self._z_before + 0.01}", "施力后 pelvis 上升")

            # xfrc_applied 可读到力值
            torso_id = self.model.body(f"{agent}/torso_link").id
            xfrc = self.data.xfrc_applied[torso_id, :3]
            verifier.check("xfrc_recorded", np.any(xfrc != 0), xfrc.tolist(), "non-zero",
                           "xfrc_applied 记录了施加的力")

            # 清力
            self.clear_body_force(f"{agent}/torso_link")

        elif step == 35:
            # 清力后：xfrc 清零
            torso_id = self.model.body(f"{agent}/torso_link").id
            xfrc = self.data.xfrc_applied[torso_id, :3]
            verifier.check("xfrc_cleared", np.all(xfrc == 0), xfrc.tolist(), "zeros",
                           "清力后 xfrc 归零")

        elif step == 50:
            # 查询接触力：单脚法向力 ≈ G1 重力的一半
            contacts = self.query_contact_simple()
            if contacts:
                contact_ids = [c["id"] for c in contacts]
                forces = self.query_contact_force(contact_ids)
                total_normal = sum(abs(f[0]) for f in forces)  # 法向力近似
                g1_mass = self.body_subtree_mass(f"{agent}/pelvis") + \
                          self.body_subtree_mass(f"{agent}/torso_link")
                half_weight = g1_mass * 9.81 / 2
                verifier.check_range("contact_normal_force", total_normal,
                                     half_weight * 0.5, half_weight * 1.5,
                                     "接触法向力 ≈ G1 重力一半")

    def observe_step(self, step: int, verifier: OnlineVerifier):
        if step == 10:
            verifier.observe("force_applied", "Studio 视口：G1 应被向上抬起（torso 施加 200N 向上力）")
        elif step == 30:
            verifier.observe("force_cleared", "Studio 视口：清力后 G1 应自由落体回落")
```

#### 4.3.3 Lesson 6：雅可比判定

```python
class JacobianEnv(G1BaseEnv):
    def verify_step(self, step: int, verifier: OnlineVerifier):
        agent = self.agent_name
        if step == 0:
            # 1. pelvis 雅可比形状 (3, nv)，nv=35
            pelvis_id = self.model.body(f"{agent}/pelvis").id
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            self.mj_jacBody(jacp, jacr, body_id=pelvis_id)
            verifier.check("jac_shape", jacp.shape == (3, 35), jacp.shape, (3, 35),
                           "pelvis 雅可比形状")

            # 2. imu site 速度与 jacp @ qvel 一致
            site_vel = self.query_site_xvalp_xvalr([f"{agent}/imu"])
            jacp_site = np.zeros((3, self.model.nv))
            jacr_site = np.zeros((3, self.model.nv))
            imu_site_id = self.model.site(f"{agent}/imu").id
            self.mj_jacSite(jacp_site, jacr_site, site_id=imu_site_id)
            expected_vel = jacp_site @ self.data.qvel
            verifier.check_allclose("site_vel_vs_jac", site_vel[0][:3], expected_vel,
                                    atol=1e-4, detail="imu site 速度 = jacp @ qvel")

            # 3. IK 演示：左脚目标位置
            foot_pos = self.get_body_xpos_xmat_xquat([f"{agent}/left_ankle_roll_link"])[0][:3]
            target = foot_pos + np.array([0.0, 0.05, 0.1])  # 抬高 10cm
            for _ in range(50):
                jacp_foot = np.zeros((3, self.model.nv))
                jacr_foot = np.zeros((3, self.model.nv))
                foot_id = self.model.body(f"{agent}/left_ankle_roll_link").id
                self.mj_jacBody(jacp_foot, jacr_foot, body_id=foot_id)
                cur_pos = self.get_body_xpos_xmat_xquat(
                    [f"{agent}/left_ankle_roll_link"])[0][:3]
                delta = target - cur_pos
                qvel_leg = np.linalg.pinv(jacp_foot[:, 7:]) @ delta  # 跳过 free base
                # 逐步调整关节角（仅演示，实际需 clip + 阻尼）
                self.data.qpos[7:] += qvel_leg * 0.01
                self.mj_forward()
            final_pos = self.get_body_xpos_xmat_xquat(
                [f"{agent}/left_ankle_roll_link"])[0][:3]
            verifier.check_allclose("ik_foot_target", final_pos, target, atol=0.02,
                                    detail="IK 迭代后左脚到达目标位置")

    def observe_step(self, step: int, verifier: OnlineVerifier):
        if step == 0:
            verifier.observe("ik_foot_movement", "Studio 视口：左脚应移动到目标位置（抬高约 10cm）")
```

#### 4.3.4 Lesson 7：视频录制与截帧判定

```python
import os
import glob

class StudioCaptureEnv(G1BaseEnv):
    def run_lesson(self, num_steps: int, verifier: OnlineVerifier):
        """重写运行流程：加入 begin_save_video / stop_save_video"""
        self.reset()
        agent = self.agent_name

        # 1. 摄像头使能检查
        frame_idx = self.get_current_frame()
        verifier.check("camera_enabled", frame_idx >= 0, frame_idx, ">=0",
                       "摄像头使能检查")

        # 2. 开始录制（path 是目录）
        video_dir = "/tmp/g1_walk_video"
        os.makedirs(video_dir, exist_ok=True)
        self.begin_save_video(video_dir, capture_mode=CaptureMode.ASYNC)
        verifier.observe("recording_started", "Studio 视口：G1 开始行走，正在录制视频")

        # 3. 循环步进（ONNX 策略控制行走）
        prev_frame = frame_idx
        for step in range(num_steps):
            ctrl = self.compute_ctrl(step)
            self.do_simulation(ctrl, self.frame_skip)
            self.render()

            # 帧索引递增检查
            if step % 50 == 0:
                cur_frame = self.get_next_frame()
                if step > 0:
                    verifier.check(f"frame_index_increasing_{step}",
                                   cur_frame > prev_frame, cur_frame, f">{prev_frame}",
                                   "帧索引递增")
                prev_frame = cur_frame

        # 4. 视频截帧
        frame_dir = "/tmp/g1_frames"
        os.makedirs(frame_dir, exist_ok=True)
        camera_info = self.get_frame_png(frame_dir)
        verifier.check("get_frame_png_returns_cameras", "camera_head" in camera_info,
                       list(camera_info.keys()), ["camera_head"], "截帧返回相机列表")

        # 轮询 PNG 文件生成
        png_path = None
        for _ in range(20):  # max_wait 0.5s × 20 = 10s
            pngs = glob.glob(f"{frame_dir}/color/camera_head_color_*.png")
            if pngs:
                png_path = pngs[0]
                if os.path.getsize(png_path) > 100:  # 文件大小稳定
                    break
            time.sleep(0.5)
        verifier.check("png_file_generated", png_path is not None and os.path.getsize(png_path) > 100,
                       png_path, "exists & size>100", "PNG 截帧文件生成")

        # 5. 时间戳查询
        timestamps = self.get_camera_time_stamp(last_frame_index=prev_frame)
        verifier.check("timestamp_returned", "camera_head" in timestamps,
                       list(timestamps.keys()), ["camera_head"], "时间戳查询返回")

        # 6. 停止录制
        self.stop_save_video()

        # 7. mp4 文件生成检查
        mp4s = glob.glob(f"{video_dir}/*.mp4")
        verifier.check("mp4_file_generated", len(mp4s) > 0, mp4s, "non-empty",
                       "录制完成后 mp4 文件生成")

        verifier.report()

    def observe_step(self, step: int, verifier: OnlineVerifier):
        if step == 0:
            verifier.observe("g1_walking", "Studio 视口：G1 应在策略控制下行走")
```

#### 4.3.5 Lesson 8：体操作与拖拽判定

```python
class BodyManipulationEnv(G1BaseEnv):
    def verify_step(self, step: int, verifier: OnlineVerifier):
        agent = self.agent_name

        if step == 50:
            # 行走中：记录 pelvis 初始位姿
            self._pelvis_before = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[0][:3]

        elif step == 100:
            # 程序化锚定 pelvis
            self.anchor_actor(f"{agent}/pelvis", anchor_type="weld")
            verifier.observe("anchor_pelvis",
                             "Studio 视口：G1 pelvis 被锚定，应悬停在当前位置")

        elif step == 120:
            # 锚定后：检查 G1 不再自由运动（位姿接近锚定时）
            pelvis_after = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[0][:3]
            verifier.check_allclose("anchored_position_stable",
                                     pelvis_after, self._pelvis_before, atol=0.05,
                                     detail="锚定后 pelvis 位置稳定")

        elif step == 150:
            # 释放锚定
            self.release_body_anchored()
            verifier.observe("release_anchor",
                             "Studio 视口：释放锚定，G1 恢复物理仿真与行走")

        elif step == 200:
            # 释放后：G1 恢复运动（位姿有变化）
            pelvis_final = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[0][:3]
            moved = np.linalg.norm(pelvis_final - self._pelvis_before) > 0.01
            verifier.check("released_resumes_motion", moved,
                           pelvis_final.tolist(), "moved from anchor",
                           "释放后 G1 恢复运动")

        # === XML 内置 mocap + weld 约束验证（不依赖 Studio 锚点）===
        elif step == 250:
            # 默认绑定 anchor↔box：mocap 驱动 box
            box_before = self.get_body_xpos_xmat_xquat(["manipulation_box"])[0][:3]
            self._box_before = box_before
            target_mocap = np.array([0.7, 0.0, 0.5])
            self.set_mocap_pos_and_quat("ActorManipulator_Anchor",
                                        target_mocap.tolist(), [1, 0, 0, 0])

        elif step == 350:
            # 步进 100 帧后 box 应跟随 mocap
            box_after = self.get_body_xpos_xmat_xquat(["manipulation_box"])[0][:3]
            verifier.check_allclose("mocap_drives_box_via_weld",
                                     box_after, [0.7, 0.0, 0.5], atol=0.05,
                                     detail="mocap 移动后 box 跟随（weld 约束）")

            # 停用 equality：mocap 不再驱动 box
            self.update_equality_constraints(eq_name="anchor_box_weld", active=False)
            self.set_mocap_pos_and_quat("ActorManipulator_Anchor",
                                        [0.9, 0.0, 0.7], [1, 0, 0, 0])

        elif step == 450:
            # 停用后 box 不跟随
            box_idle = self.get_body_xpos_xmat_xquat(["manipulation_box"])[0][:3]
            verifier.check("eq_disable_decouples_box",
                           np.linalg.norm(box_idle - np.array([0.7, 0.0, 0.5])) < 0.1,
                           box_idle.tolist(), "≈[0.7,0,0.5]",
                           "停用 equality 后 box 不再跟随 mocap")

            # 重绑 equality：obj2id 从 box 改为 G1 pelvis
            pelvis_id = self.model.body(f"{agent}/pelvis").id
            self.modify_equality_objects(eq_name="anchor_box_weld",
                                          obj2id=pelvis_id)
            verifier.check("eq_rebound_to_pelvis",
                           self._mjModel.eq_obj2id[0] == pelvis_id,
                           self._mjModel.eq_obj2id[0], pelvis_id,
                           "weld obj2id 重绑到 pelvis")

            # 重新激活并移动 mocap 驱动 G1 pelvis
            self.update_equality_constraints(eq_name="anchor_box_weld", active=True)
            pelvis_pos = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[0][:3]
            self._pelvis_pre_drive = pelvis_pos.copy()
            self.set_mocap_pos_and_quat("ActorManipulator_Anchor",
                                        (pelvis_pos + np.array([0.2, 0.0, 0.1])).tolist(),
                                        [1, 0, 0, 0])

        elif step == 650:
            # 步进 200 帧后 G1 pelvis 被 mocap 驱动
            pelvis_driven = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[0][:3]
            displacement = np.linalg.norm(pelvis_driven - self._pelvis_pre_drive)
            verifier.check("mocap_drives_g1_pelvis",
                           displacement > 0.05,
                           displacement, ">0.05",
                           "重绑 equality 后 mocap 驱动 G1 pelvis 位移")

    def observe_step(self, step: int, verifier: OnlineVerifier):
        if step == 0:
            verifier.observe("g1_walking", "Studio 视口：G1 应在策略控制下行走")
        if step == 80:
            verifier.observe("manual_drag",
                             "请在 Studio 视口用鼠标拖拽 G1 pelvis，观察锚定效果")
        if step == 250:
            verifier.observe("mocap_drive_box",
                             "Studio 视口：绿色球体 anchor 应拖动橙色 box 移动到 [0.7,0,0.5]")
        if step == 450:
            verifier.observe("mocap_drive_g1",
                             "Studio 视口：重绑 equality 后 mocap anchor 应拖动 G1 pelvis（整机位移）")
```

> **说明**：上述 `verify_step` 中 `self._mjModel.eq_obj2id[0]` 是 P2 架构约束下**唯一允许的 model 字段读访问**（用于验证 `modify_equality_objects` 写入正确），实际 Env 公共方法应通过 `query_equality_objects` 等 K3 查询 API 读取，避免在 example 脚本中直接访问 `_mjModel`。

### 4.4 输出与报告

每个 Lesson 运行结束后，`OnlineVerifier.report()` 输出：

1. **控制台摘要**：pass/fail 计数 + 人工观察项数 + ALL PASS / SOME FAILED
2. **JSON 报告文件**：写入 `/tmp/euler_Lesson_N_report.json`，含完整判定明细

报告结构：

```json
{
  "lesson": "Lesson 4: 状态查询 API",
  "timestamp": "2026-06-26T10:30:00",
  "summary": {
    "total_checks": 8,
    "passed": 8,
    "failed": 0,
    "all_passed": true
  },
  "checks": [
    {
      "name": "joint_qpos_dim",
      "passed": true,
      "actual": "29",
      "expected": "29",
      "detail": ""
    },
    ...
  ],
  "observations": [
    {"name": "g1_standing", "prompt": "Studio 视口：G1 应站立在地面上", "step": 0}
  ]
}
```

**验收标准**：
- `all_passed == true`：全部数值判定通过
- 人工观察项由用户在运行过程中逐项确认（脚本打印 `[OBSERVE]` 提示）

---

## 5. 验收总览

### 5.1 子阶段验收清单

| 子阶段 | 验收项 | Example | 判定方式 |
|--------|--------|---------|---------|
| 3.1 状态查询 | `query_*`/`get_body_*`/`jnt_*` 在线运行正确 | Lesson 4 | 数值判定（维度/范围/一致性）+ 人工观察（G1 站立姿态） |
| 3.2 状态设置与外力 | 力应用/设置/mocap 位姿写入方法在线运行正确 | Lesson 5 | 数值判定（pelvis z 变化/xfrc 清零/接触力/mocap 驱动 box）+ 人工观察（抬起/回落/box 跟随） |
| 3.3 雅可比 | `mj_jac*`/IK 在线运行正确 | Lesson 6 | 数值判定（雅可比形状/site 速度一致性/IK 收敛）+ 人工观察（左脚移动） |
| 3.4 Studio 交互 | 视频/帧/内容文件方法在线运行正确 | Lesson 7 | 数值判定（帧索引递增/PNG 生成/mp4 生成）+ 人工观察（G1 行走） |
| 3.5 约束与体操作 | 约束/锚定/释放/mocap 驱动 G1 动力学在线运行正确 | Lesson 8 | 数值判定（锚定稳定/释放后运动/mocap 驱动 box/equality 停用解耦/重绑驱动 pelvis）+ 人工观察（拖拽效果/anchor 拖动 box/anchor 拖动 G1） |

### 5.2 阶段四完成标准

- [ ] Lesson 4–8 端到端 example 在线模式全部运行通过
- [ ] 每个 example 的 `OnlineVerifier` 报告 `all_passed == true`
- [ ] 人工观察项由用户逐项确认
- [ ] K1–K12 架构约束在线运行中无违反（无穿墙访问回退）
- [ ] JSON 报告文件归档（`/tmp/euler_Lesson_N_report.json`）

### 5.3 不在阶段四范围

| 项 | 归属 | 说明 |
|----|------|------|
| `EulerOrchestrator` 耦合编排 | 后续 phase | Euler 非刚体求解器与 MuJoCo 耦合，单独设计 |
| `OrcaGymLocalEnv` 老代码实际迁移 | 后续 phase | 选 OrcaPlayground 代表性 Env 实际迁移，产出迁移指南 |
| 多 agent 联合仿真编排 | 后续 phase | 多 `OrcaGymEuler` 实例编排，单独设计 |

---

## 附录 A：参考资料

- 架构文档：`docs/design/architecture/orca_gym_euler_architecture.md`
- 阶段一开发文档：`docs/design/development/orca_gym_euler_skeleton_development.md`
- 阶段二开发文档：`docs/design/development/orca_gym_euler_phase2_filling_development.md`
- 阶段三开发文档：`docs/design/development/orca_gym_euler_phase3_filling_development.md`（功能填充 + 离线单元测试）
- OrcaPlayground Euler 课程：`<repo-root>/OrcaPlayground/examples/euler/TUTORIAL.md`
- OrcaPlayground G1（阶段四统一场景）：Euler 专用模型 `<repo-root>/OrcaPlayground/envs/euler/robots/g1_29dof_camera.xml`（含头部摄像头）；原 G1 例子（参考来源）`<repo-root>/OrcaPlayground/examples/g1/README.md`、`<repo-root>/OrcaPlayground/examples/g1/run_g1_sim.py`
- OrcaPlayground D12 ACT（camera 截帧 API 参考）：`<repo-root>/OrcaPlayground/examples/d12/act/run_d12_act.py`（`_get_camera_images` 的 `get_frame_png` 用法）
- OrcaPlayground Aloha（camera WebSocket 实时流参考）：`<repo-root>/OrcaPlayground/envs/aloha/aloha_orcagym_task.py`（`CameraWrapper(name, port)` 用法）
- OrcaManipulation（camera 视频录制 + 回放验证参考）：`<repo-root>/OrcaManipulation/src/dataStorage/abstract_data_storage.py`（`begin_save_video`/`stop_save_video` 封装，path 为目录）、`<repo-root>/OrcaManipulation/src/dataCollectionManager/data_collection_manager.py`（task RUNNING/END 触发录制）、`<repo-root>/OrcaManipulation/src/examples/dataCollection/data_collection_replay.py`（回放录制验证）
- OrcaGym camera API 文档：`<repo-root>/OrcaGym/doc/api_detail/core.md`（`begin_save_video`/`stop_save_video`/`get_current_frame`/`get_camera_time_stamp`/`get_frame_png` 签名与说明）、`<repo-root>/OrcaGym/orca_gym/sensor/rgbd_camera.py`（`CameraWrapper` WebSocket 实时流实现）
- OrcaGymLocalEnv 老体系：`orca_gym/environment/orca_gym_local_env.py`、`orca_gym/core/orca_gym_local.py`