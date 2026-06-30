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
| 环境 | conda `orca` 环境，离线模式 | conda `orca` 环境 + OrcaStudio + G1 关卡 |
| 数据来源 | 离线加载 `g1_29dof_camera.xml` 的真实 MuJoCo 数据 | 在线 gRPC 连接 Studio，真实场景运行数据 |
| 判定方式 | unittest assert（数值判定） | 人工观察 + 数值判定（脚本内嵌） |

阶段三的离线单元测试验证**功能正确性**（API 调用链路、数值计算）；阶段四的在线 example 验证**真实运行效果**（含 Studio 交互、gRPC 通信、摄像头流、ONNX 策略推理等在线特性）。

### 1.3 前置条件

- 阶段三全部子阶段（3.1–3.5）验收通过
- OrcaStudio 已安装并可启动，具备含 G1 机器人的关卡
- G1 Euler 专用资源已就位（`OrcaPlayground/envs/euler/robots/`）
- 宿主机 CUDA 环境（ONNX 策略推理如需 GPU）

### 1.4 手工验证流程

阶段四采用「人工 + 自动」交替的双轨验证流程，每个 Lesson 按以下 5 步执行：

| 步骤 | 执行方 | 动作 | 说明 |
|------|--------|------|------|
| 1 | 人工 | 启动 OrcaStudio，加载含 1 个 G1 机器人的关卡，点击运行 | 关卡运行后 Studio 进入仿真态，监听 gRPC 端口等待 Env 连接 |
| 2 | 人工 | 运行 `OrcaPlayground/examples/euler/` 下对应 Lesson 脚本 | 脚本启动后通过 gRPC 连接 Studio，创建 `OrcaGymEulerEnv` 实例 |
| 3 | 自动 | 脚本驱动 euler env 实例完成相关功能 | 仿真步进 / locomotion 行走 / 视频录制截帧 / 体操作等，由 `run_lesson` 框架统一编排（§3.1） |
| 4 | 人工 | 用户根据教程指导观察 Studio 视口画面 | 确认画面结果符合预期（G1 站立/行走/抬起/拖拽等），对应 `OnlineVerifier.observe()` 提示项（§4.1） |
| 5 | 自动 | 脚本自动检查中间输出并评估用例是否通过 | 通过 euler 公共 API 读取数据、检查保存的文件（mp4/PNG/JSON），`OnlineVerifier.check()` 输出数值判定结论（§4.2） |

> **流程要点**：
> - 步骤 1–2 为人工启动环节，确保 Studio 关卡与脚本 env 实例一一对应（agent_name 扫描见 §2.0.4）。
> - 步骤 3 与步骤 4/5 在 `run_lesson` 循环中**交织进行**：每个控制周期先 `do_simulation` 步进（自动），再 `verify_step` 数值判定（自动），再 `observe_step` 打印观察提示（人工）。
> - 步骤 4 的人工观察项与步骤 5 的数值判定项**独立计数**，二者共同构成 Lesson 的验收结论（§4.4 报告格式）。
> - 脚本运行结束后，`OnlineVerifier.report()` 输出 JSON 报告到 `/tmp/euler_Lesson_N_report.json`，用户据报告 + 视口观察综合判断 Lesson 是否通过。

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

> **设计要点**：这三个测试对象是独立于 G1 本体的辅助 fixture，不干扰 G1 的 29 自由度行走控制。`anchor_box_weld` 的 `body2` 默认指向 `manipulation_box`，在线测试中可通过 `modify_equality_objects(eq_ids=[0], obj2_names=[f"{agent}/pelvis"])` 将约束重绑到 G1 pelvis，验证 mocap 驱动 G1 动力学（不增删 equality 节点，仅改 obj1id/obj2id）。

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
#    异步写 PNG 到 {image_path}/color/{camera_name}_color_0.png，返回 None
#    验证方式：轮询目录下 PNG 文件生成（参考 run_d12_act.py max_wait=0.5s 轮询）
env.get_frame_png("/tmp/g1_frames")

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
3. `get_body_xpos_xmat_xquat(["{agent}/pelvis", "{agent}/torso_link", "{agent}/left_ankle_roll_link"])` 返回 `dict[body_name -> {"xpos","xmat","xquat"}]`，pelvis `xpos[2]`≈0.793（初始高度）
4. `query_site_pos_and_mat(["{agent}/imu"])` 返回 imu site 的 `{"pos","mat"}`，位于 torso_link 内
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
7. `set_mocap_pos_and_quat({"ActorManipulator_Anchor": {"pos": [0.7, 0, 0.5], "quat": [1,0,0,0]}})`（dict 形参）写入 mocap 位姿后，`env.data.mocap_pos("ActorManipulator_Anchor")` 读回值一致；weld 约束驱动 box，步进 100 帧后 `get_body_xpos_xmat_xquat(["manipulation_box"])["manipulation_box"]["xpos"]` ≈ `[0.7, 0, 0.5]`（atol=0.05）

### 2.3 Lesson 6：雅可比与逆运动学验证（G1 足端/躯干雅可比）

**文件**：`OrcaPlayground/examples/euler/06_jacobian/jacobian_ik.py`

**验证内容**：G1 雅可比计算（足端 body、imu site）、基于雅可比的躯干姿态控制或足端位置控制。

**验证点**：
1. `mj_jacBody(jacp, jacr, body_name=f"{agent}/pelvis")` 返回 `(3, nv)` 雅可比矩阵，nv=35（6 free + 29 旋转）
2. `mj_jacSite(jacp, jacr, site_name=f"{agent}/imu")` 返回 imu site 雅可比
3. `query_site_xvalp_xvalr(["{agent}/imu"])` 返回 `(xvalp_dict, xvalr_dict)`，`xvalp_dict[site_name]` 与 `jacp @ env.data.qvel` 数值一致
4. **IK 演示**：以左腿为目标，计算 `left_ankle_roll_link` 相对 pelvis 的雅可比，用伪逆 `qvel_leg = pinv(jacp_leg) @ (target_foot_pos - foot_pos)` 迭代调整左腿关节，使左脚到达目标位置（在 Studio 视口可见左脚移动）
5. `mj_jac_site(["{agent}/imu", "{agent}/camera_head_site"])` 批量返回多个 site 雅可比（注：`left_ankle_roll_link` 为 body，单 body 雅可比用 `mj_jacBody`）

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
3. **视频截帧**：行走中途 `get_frame_png("/tmp/g1_frames")` 异步抓帧（返回 None，PNG 写入目录），轮询 `/tmp/g1_frames/color/camera_head_color_0.png` 文件生成且大小稳定（参考 `run_d12_act.py` 的 `max_wait=0.5s` 轮询）
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
6. 拖拽过程中 `get_body_xpos_xmat_xquat(["{agent}/pelvis"])` 实时返回位姿变化
7. **mocap 驱动 box（默认绑定）**：`set_mocap_pos_and_quat({"ActorManipulator_Anchor": {"pos": [0.7, 0, 0.5], "quat": [1,0,0,0]}})` 后步进 100 帧，`manipulation_box` 的 xpos ≈ `[0.7, 0, 0.5]`（atol=0.05），Studio 视口可见球体 anchor 拖动 box
8. **停用 equality 解耦**：`update_equality_constraints([{"type": 0, "obj1_id": -1, "obj2_id": -1, "data": np.zeros(mujoco.mjNEQDATA)}])`（按索引 0 写 type=0 停用 eq[0]）后移动 mocap，box 不再跟随（自由落体或保持原位）
9. **重绑 equality 驱动 G1 pelvis**：`modify_equality_objects(eq_ids=[0], obj2_names=[f"{agent}/pelvis"])` 将 eq[0] weld 的 obj2id 从 box 改为 pelvis，`set_mocap_pos_and_quat({"ActorManipulator_Anchor": {"pos": pelvis_pos+[0.2,0,0.1], "quat": [1,0,0,0]}})` 后步进 200 帧，G1 pelvis 位移 > 0.05m（Studio 视口可见 G1 被 mocap 拖动）—— **此为 mocap 驱动 G1 动力学的核心验证**

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
        """统一运行入口：子类通过重写钩子方法插入验证逻辑"""
        self.reset()
        verifier.observe("start", "请在 Studio 视口观察 G1 初始姿态：应站立在地面上")

        # 循环前钩子：Lesson 7 用于 begin_save_video，其他 Lesson 可空操作
        self.before_loop(verifier)

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

        # 循环后钩子：Lesson 7 用于 stop_save_video + mp4 检查，其他 Lesson 可空操作
        self.after_loop(verifier)

        # 5. 结束判定 + 报告
        self.verify_final(verifier)
        verifier.report()

    def compute_ctrl(self, step: int) -> np.ndarray:
        """子类重写：Lesson 4/5/6 返回零控；Lesson 7/8 返回 ONNX 策略输出"""
        return np.zeros(self.model.nu)

    def before_loop(self, verifier: OnlineVerifier):
        """子类重写：循环前准备（如 Lesson 7 begin_save_video）"""
        pass

    def verify_step(self, step: int, verifier: OnlineVerifier):
        """子类重写：每个控制周期插入数值判定"""
        pass

    def observe_step(self, step: int, verifier: OnlineVerifier):
        """子类重写：阶段性打印人工观察提示"""
        pass

    def after_loop(self, verifier: OnlineVerifier):
        """子类重写：循环后收尾（如 Lesson 7 stop_save_video + mp4 检查）"""
        pass

    def verify_final(self, verifier: OnlineVerifier):
        """子类重写：运行结束后的最终判定"""
        pass
```

### 3.2 启动流程

所有 Lesson 的启动流程一致，封装在 `G1BaseEnv.initialize_simulation` 中。各 Lesson 的脚本入口结构相同，仅 Env 子类名、verifier 名称、num_steps 不同：

```python
# query_api.py 启动流程（所有 Lesson 通用模板）
from envs.euler.g1_base_env import G1BaseEnv
from envs.euler.online_verifier import OnlineVerifier
from envs.common.model_scanner import scan_scene_for_template, build_suffix_template

class QueryApiEnv(G1BaseEnv):
    def run(self):
        verifier = OnlineVerifier("Lesson 4: 状态查询 API")
        self.run_lesson(num_steps=100, verifier=verifier)

if __name__ == "__main__":
    env = QueryApiEnv(
        frame_skip=20,
        orcagym_addr="127.0.0.1:50051",
        agent_names=["g1"],
        time_step=0.001,
        model_xml_path="envs/euler/robots/g1_29dof_camera.xml",
    )
    env.run()
```

各 Lesson 的脚本入口差异（Env 子类名 → verifier 名称 / num_steps）：

| Lesson | 脚本文件 | Env 子类 | verifier 名称 | num_steps |
|--------|---------|---------|--------------|-----------|
| 4 | `04_query_api/query_api.py` | `QueryApiEnv` | "Lesson 4: 状态查询 API" | 100 |
| 5 | `05_force_apply/force_apply.py` | `ForceApplyEnv` | "Lesson 5: 外力应用" | 100 |
| 6 | `06_jacobian/jacobian_ik.py` | `JacobianEnv` | "Lesson 6: 雅可比 IK" | 100 |
| 7 | `07_studio_capture/studio_capture.py` | `StudioCaptureEnv` | "Lesson 7: 视频录制" | 500 |
| 8 | `08_body_manipulation/body_manipulation.py` | `BodyManipulationEnv` | "Lesson 8: 体操作" | 700 |

> **说明**：Lesson 7/8 的 `num_steps` 较大（500/700）是因为需要覆盖行走录制 / 拖拽锚定 / equality 重绑等多个阶段；其他参数（`frame_skip`/`orcagym_addr`/`agent_names`/`time_step`/`model_xml_path`）各 Lesson 完全一致。`initialize_simulation` 在 `G1BaseEnv` 中统一实现，子类通过重写 `compute_ctrl`/`before_loop`/`verify_step`/`observe_step`/`after_loop`/`verify_final` 钩子插入差异化逻辑（§3.3）。

### 3.3 循环步进逻辑

每个控制周期（frame_skip=20，即 20ms / 50Hz）执行：

| 阶段 | 步骤 | 动作 | 说明 |
|------|------|------|------|
| 循环前 | 0 | `self.before_loop(verifier)` | 准备工作（Lesson 7 `begin_save_video`，其他 Lesson 空操作） |
| 循环中 | 1 | `ctrl = self.compute_ctrl(step)` | 策略推理（Lesson 4/5/6 零控，Lesson 7/8 ONNX） |
| 循环中 | 2 | `self.do_simulation(ctrl, self.frame_skip)` | 物理步进 |
| 循环中 | 3 | `self.verify_step(step, verifier)` | 数值判定：读写 env 数据，assert 预期 |
| 循环中 | 4 | `self.observe_step(step, verifier)` | 人工观察：阶段性打印 Studio 视口提示 |
| 循环中 | 5 | `self.render()` | 渲染到 Studio 视口 |
| 循环后 | 6 | `self.after_loop(verifier)` | 收尾工作（Lesson 7 `stop_save_video` + mp4 检查，其他 Lesson 空操作） |
| 结束 | 7 | `self.verify_final(verifier)` + `verifier.report()` | 最终判定 + 输出 JSON 报告 |

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

            # 2. qpos 与 env.data.qpos 切片一致（dict 拼接为连续数组后比较）
            expected = self.data.qpos[7:]  # 跳过 free base 前 7 维
            qpos_arr = np.concatenate([qpos[j] for j in joint_names])
            verifier.check_allclose("joint_qpos_vs_data", qpos_arr, expected, atol=1e-6)

            # 3. pelvis 初始高度 ≈ 0.793
            pelvis = self.get_body_xpos_xmat_xquat([f"{agent}/pelvis"])
            pelvis_z = pelvis[f"{agent}/pelvis"]["xpos"][2]
            verifier.check_range("pelvis_initial_height", pelvis_z, 0.75, 0.85,
                                 "G1 站立初始高度")

            # 4. IMU sensor 维度
            imu_quat = self.query_sensor_data([f"{agent}/imu_quat"])
            verifier.check("imu_quat_dim", len(imu_quat[f"{agent}/imu_quat"]) == 4,
                           len(imu_quat[f"{agent}/imu_quat"]), 4)

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
            self._z_before = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"][2]

            # 施加向上力
            self.apply_body_force(f"{agent}/torso_link", [0, 0, 200], [0, 0, 0])

        elif step == 30:
            # 施力后：pelvis z 应上升
            z_after = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"][2]
            verifier.check("force_lift_pelvis", z_after > self._z_before + 0.01,
                           z_after, f">{self._z_before + 0.01}", "施力后 pelvis 上升")

            # xfrc_applied 可读到力值（DataView 只读视图，按 body_id 索引）
            torso_id = self.model.body_name2id(f"{agent}/torso_link")
            xfrc = self.data.xfrc_applied[torso_id, :3]
            verifier.check("xfrc_recorded", np.any(xfrc != 0), xfrc.tolist(), "non-zero",
                           "xfrc_applied 记录了施加的力")

            # 清力
            self.clear_body_force(f"{agent}/torso_link")

        elif step == 35:
            # 清力后：xfrc 清零
            torso_id = self.model.body_name2id(f"{agent}/torso_link")
            xfrc = self.data.xfrc_applied[torso_id, :3]
            verifier.check("xfrc_cleared", np.all(xfrc == 0), xfrc.tolist(), "zeros",
                           "清力后 xfrc 归零")

        elif step == 50:
            # 查询接触力：单脚法向力 ≈ G1 重力的一半
            contacts = self.query_contact_simple()
            if contacts:
                # contact id 为 contacts 列表索引（query_contact_simple 不返回 id 字段）
                contact_ids = list(range(len(contacts)))
                forces = self.query_contact_force(contact_ids)  # dict[id -> (6,)]
                total_normal = sum(abs(f[0]) for f in forces.values())  # 法向力近似
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
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            self.mj_jacBody(jacp, jacr, body_name=f"{agent}/pelvis")
            verifier.check("jac_shape", jacp.shape == (3, 35), jacp.shape, (3, 35),
                           "pelvis 雅可比形状")

            # 2. imu site 速度与 jacp @ qvel 一致
            xvalp, xvalr = self.query_site_xvalp_xvalr([f"{agent}/imu"])
            jacp_site = np.zeros((3, self.model.nv))
            jacr_site = np.zeros((3, self.model.nv))
            self.mj_jacSite(jacp_site, jacr_site, site_name=f"{agent}/imu")
            expected_vel = jacp_site @ self.data.qvel
            verifier.check_allclose("site_vel_vs_jac", xvalp[f"{agent}/imu"],
                                    expected_vel, atol=1e-4,
                                    detail="imu site 速度 = jacp @ qvel")

            # 3. IK 演示：左脚目标位置
            foot_pos = self.get_body_xpos_xmat_xquat(
                [f"{agent}/left_ankle_roll_link"]
                )[f"{agent}/left_ankle_roll_link"]["xpos"]
            target = foot_pos + np.array([0.0, 0.05, 0.1])  # 抬高 10cm
            for _ in range(50):
                jacp_foot = np.zeros((3, self.model.nv))
                jacr_foot = np.zeros((3, self.model.nv))
                self.mj_jacBody(jacp_foot, jacr_foot,
                                body_name=f"{agent}/left_ankle_roll_link")
                cur_pos = self.get_body_xpos_xmat_xquat(
                    [f"{agent}/left_ankle_roll_link"]
                    )[f"{agent}/left_ankle_roll_link"]["xpos"]
                delta = target - cur_pos
                qvel_leg = np.linalg.pinv(jacp_foot[:, 7:]) @ delta  # 跳过 free base
                # 合规写入：复制 qpos → 修改 → set_joint_qpos（W1，不直接写 data.qpos）
                qpos = self.data.qpos.copy()
                qpos[7:] += qvel_leg * 0.01  # 逐步调整关节角（仅演示，实际需 clip + 阻尼）
                self.set_joint_qpos(qpos)
                self.mj_forward()
            final_pos = self.get_body_xpos_xmat_xquat(
                [f"{agent}/left_ankle_roll_link"]
                )[f"{agent}/left_ankle_roll_link"]["xpos"]
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
    # 截帧目录与视频目录（after_loop 复用）
    _video_dir = "/tmp/g1_walk_video"
    _frame_dir = "/tmp/g1_frames"

    def before_loop(self, verifier: OnlineVerifier):
        """循环前：摄像头使能检查 + 开始录制"""
        # 1. 摄像头使能检查
        frame_idx = self.get_current_frame()
        verifier.check("camera_enabled", frame_idx >= 0, frame_idx, ">=0",
                       "摄像头使能检查")

        # 2. 开始录制（path 是目录）
        os.makedirs(self._video_dir, exist_ok=True)
        self.begin_save_video(self._video_dir, capture_mode=CaptureMode.ASYNC)
        verifier.observe("recording_started",
                         "Studio 视口：G1 开始行走，正在录制视频")

        # 记录初始帧索引，供 verify_step 帧索引递增检查
        self._prev_frame = frame_idx

    def verify_step(self, step: int, verifier: OnlineVerifier):
        """循环中：每 50 步检查帧索引递增"""
        if step % 50 == 0:
            cur_frame = self.get_next_frame()
            if step > 0:
                verifier.check(f"frame_index_increasing_{step}",
                               cur_frame > self._prev_frame, cur_frame,
                               f">{self._prev_frame}", "帧索引递增")
            self._prev_frame = cur_frame

    def observe_step(self, step: int, verifier: OnlineVerifier):
        """循环中：阶段性人工观察提示"""
        if step == 0:
            verifier.observe("g1_walking", "Studio 视口：G1 应在策略控制下行走")
        elif step == 250:
            verifier.observe("walking_stable",
                             "Studio 视口：G1 行走应稳定，录制中段画面正常")

    def after_loop(self, verifier: OnlineVerifier):
        """循环后：截帧 + 时间戳 + 停止录制 + mp4 检查"""
        # 3. 视频截帧（get_frame_png 异步写 PNG 到目录，返回 None）
        os.makedirs(self._frame_dir, exist_ok=True)
        self.get_frame_png(self._frame_dir)

        # 轮询 PNG 文件生成
        png_path = None
        for _ in range(20):  # max_wait 0.5s × 20 = 10s
            pngs = glob.glob(f"{self._frame_dir}/color/camera_head_color_*.png")
            if pngs:
                png_path = pngs[0]
                if os.path.getsize(png_path) > 100:  # 文件大小稳定
                    break
            time.sleep(0.5)
        verifier.check("png_file_generated",
                       png_path is not None and os.path.getsize(png_path) > 100,
                       png_path, "exists & size>100", "PNG 截帧文件生成")

        # 4. 时间戳查询
        timestamps = self.get_camera_time_stamp(last_frame_index=self._prev_frame)
        verifier.check("timestamp_returned", "camera_head" in timestamps,
                       list(timestamps.keys()), ["camera_head"], "时间戳查询返回")

        # 5. 停止录制
        self.stop_save_video()

        # 6. mp4 文件生成检查
        mp4s = glob.glob(f"{self._video_dir}/*.mp4")
        verifier.check("mp4_file_generated", len(mp4s) > 0, mp4s, "non-empty",
                       "录制完成后 mp4 文件生成")
```

> **结构说明**：Lesson 7 通过 `before_loop`（摄像头检查 + begin_save_video）、`verify_step`（帧索引递增）、`observe_step`（行走观察提示）、`after_loop`（截帧 + 时间戳 + stop_save_video + mp4 检查）四个钩子拆分原内联逻辑，符合 §3.1 `run_lesson` 框架的「人工观察 + 数值判定交织」流程（§1.4 步骤 3/4/5）。`verifier.report()` 由 `run_lesson` 在 `after_loop` 之后统一调用，不在子类内显式调用。

#### 4.3.5 Lesson 8：体操作与拖拽判定

```python
class BodyManipulationEnv(G1BaseEnv):
    def verify_step(self, step: int, verifier: OnlineVerifier):
        agent = self.agent_name

        if step == 50:
            # 行走中：记录 pelvis 初始位姿
            self._pelvis_before = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]

        elif step == 100:
            # 程序化锚定 pelvis
            self.anchor_actor(f"{agent}/pelvis", anchor_type="weld")
            verifier.observe("anchor_pelvis",
                             "Studio 视口：G1 pelvis 被锚定，应悬停在当前位置")

        elif step == 120:
            # 锚定后：检查 G1 不再自由运动（位姿接近锚定时）
            pelvis_after = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]
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
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]
            moved = np.linalg.norm(pelvis_final - self._pelvis_before) > 0.01
            verifier.check("released_resumes_motion", moved,
                           pelvis_final.tolist(), "moved from anchor",
                           "释放后 G1 恢复运动")

        # === XML 内置 mocap + weld 约束验证（不依赖 Studio 锚点）===
        elif step == 250:
            # 默认绑定 anchor↔box：mocap 驱动 box
            self._box_before = self.get_body_xpos_xmat_xquat(
                ["manipulation_box"])["manipulation_box"]["xpos"]
            # set_mocap_pos_and_quat 接受 dict 形参（W2/S3 契约）
            self.set_mocap_pos_and_quat(
                {"ActorManipulator_Anchor": {"pos": [0.7, 0.0, 0.5], "quat": [1, 0, 0, 0]}}
            )

        elif step == 350:
            # 步进 100 帧后 box 应跟随 mocap
            box_after = self.get_body_xpos_xmat_xquat(
                ["manipulation_box"])["manipulation_box"]["xpos"]
            verifier.check_allclose("mocap_drives_box_via_weld",
                                     box_after, [0.7, 0.0, 0.5], atol=0.05,
                                     detail="mocap 移动后 box 跟随（weld 约束）")

            # 停用 equality[0]：写 type=0，mocap 不再驱动 box
            # update_equality_constraints 按 eq_list 索引写 _mjModel.eq_*（C3 契约）
            import mujoco
            self.update_equality_constraints([{
                "type": 0, "obj1_id": -1, "obj2_id": -1,
                "data": np.zeros(mujoco.mjNEQDATA),
            }])
            self.set_mocap_pos_and_quat(
                {"ActorManipulator_Anchor": {"pos": [0.9, 0.0, 0.7], "quat": [1, 0, 0, 0]}}
            )

        elif step == 450:
            # 停用后 box 不跟随
            box_idle = self.get_body_xpos_xmat_xquat(
                ["manipulation_box"])["manipulation_box"]["xpos"]
            verifier.check("eq_disable_decouples_box",
                           np.linalg.norm(box_idle - np.array([0.7, 0.0, 0.5])) < 0.1,
                           box_idle.tolist(), "≈[0.7,0,0.5]",
                           "停用 equality 后 box 不再跟随 mocap")

            # 重绑 equality[0]：obj2id 从 box 改为 G1 pelvis
            # modify_equality_objects 仅改 obj1id/obj2id，不改 type（C3 契约）
            self.modify_equality_objects(eq_ids=[0], obj2_names=[f"{agent}/pelvis"])

            # 验证重绑：通过公共查询 API equality_object_ids（不触 _mjModel，P2 合规）
            # 注：equality_object_ids 需在 Env 层扩展为公共方法（委托 _gym）
            obj1_id, obj2_id = self.equality_object_ids(0)
            pelvis_id = self.model.body_name2id(f"{agent}/pelvis")
            verifier.check("eq_rebound_to_pelvis",
                           obj2_id == pelvis_id, obj2_id, pelvis_id,
                           "weld obj2id 重绑到 pelvis")

            # 重新激活 weld 类型（step 350 已置 type=0 且 obj1_id=-1，需恢复完整绑定）
            # obj1 固定为 mocap anchor（step 350 被置 -1，此处恢复为 anchor body id）
            mocap_id = self.model.body_name2id("ActorManipulator_Anchor")
            self.update_equality_constraints([{
                "type": mujoco.mjtEq.mjEQ_WELD,
                "obj1_id": mocap_id, "obj2_id": pelvis_id,
                "data": np.zeros(mujoco.mjNEQDATA),
            }])

            pelvis_pos = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]
            self._pelvis_pre_drive = pelvis_pos.copy()
            self.set_mocap_pos_and_quat({
                "ActorManipulator_Anchor": {
                    "pos": (pelvis_pos + np.array([0.2, 0.0, 0.1])).tolist(),
                    "quat": [1, 0, 0, 0],
                }
            })

        elif step == 650:
            # 步进 200 帧后 G1 pelvis 被 mocap 驱动
            pelvis_driven = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]
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

> **说明**：上述 `verify_step` 中通过 `self.equality_object_ids(0)` 验证 `modify_equality_objects` 写入正确，符合 P2 架构约束（不触 `_mjModel`）。`equality_object_ids` 当前仅在 `OrcaGymEuler`（Gym 层）实现，**需在 `OrcaGymEulerEnv` 扩展为公共方法**（委托 `self._gym.equality_object_ids(eq_idx)`，见架构 §7 / AGENTS 规则 4「缺失功能时扩展公共方法」）。同理，`equality_data_width`/`n_equality`/`mocap_body_names` 若 example 需要也应一并扩展到 Env 层。example 中 `np.zeros(mujoco.mjNEQDATA)` 使用 MuJoCo 公共常量，不属于私有访问。

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

## 6. 实施步骤与教程

本章节将 §2–§5 的设计落地为可执行的实施计划，分为总体框架搭建（步骤 0）与 5 个 Lesson 实施步骤（步骤 1–5）。每个步骤包含**实施任务清单**、**教程文档大纲**（面向用户的使用教程）、**验收方案**（明确的通过条件）。

### 6.1 实施顺序总览

各步骤存在依赖关系，须按顺序实施：

```
步骤 0：总体框架（g1_base_env + online_verifier + 资源）
   │
   ├── 步骤 1：Lesson 4（状态查询）── 依赖步骤 0
   │     │
   │     └── 步骤 2：Lesson 5（外力应用）── 依赖步骤 0、复用步骤 1 的查询 API
   │           │
   │           └── 步骤 3：Lesson 6（雅可比 IK）── 依赖步骤 0、复用步骤 1/2 的查询/写入 API
   │                 │
   │                 └── 步骤 4：Lesson 7（视频录制）── 依赖步骤 0、复用 g1_locomotion
   │                       │
   │                       └── 步骤 5：Lesson 8（体操作）── 依赖步骤 0、复用步骤 4 的 locomotion + 步骤 2 的 equality API
```

| 步骤 | 产出物 | 依赖 | 教程文档 |
|------|--------|------|---------|
| 0 | `g1_base_env.py`、`online_verifier.py`、`model_scanner.py`、`g1_locomotion.py`、`robots/` 资源 | 阶段三完成 | `TUTORIAL.md` 总览 + `00_setup.md` |
| 1 | `04_query_api/query_api.py`、`query_api_env.py` | 步骤 0 | `04_query_api.md` |
| 2 | `05_force_apply/force_apply.py`、`force_apply_env.py` | 步骤 0、1 | `05_force_apply.md` |
| 3 | `06_jacobian/jacobian_ik.py`、`jacobian_env.py` | 步骤 0、1、2 | `06_jacobian.md` |
| 4 | `07_studio_capture/studio_capture.py`、`studio_capture_env.py` | 步骤 0 | `07_studio_capture.md` |
| 5 | `08_body_manipulation/body_manipulation.py`、`body_manipulation_env.py` | 步骤 0、2、4 | `08_body_manipulation.md` |

> **说明**：步骤 4（Lesson 7）仅依赖步骤 0 的 `g1_locomotion.py`，与步骤 1–3 无强依赖，可并行实施。但建议按顺序实施，便于早期发现问题。

### 6.2 步骤 0：总体框架搭建 ✅ 已完成

#### 6.2.1 实施任务清单

| 任务 | 文件 | 说明 |
|------|------|------|
| 0.1 资源准备 | `OrcaPlayground/envs/euler/robots/g1_29dof_camera.xml` | 在原 G1 模型基础上加装摄像头传感器（`user="7070 7071"`）、mocap body `ActorManipulator_Anchor`、测试 box `manipulation_box`、weld 约束 `anchor_box_weld`（详见 §2.0.2） |
| 0.2 资源准备 | `OrcaPlayground/envs/euler/robots/models/dec_loco/model_6600.onnx` | 复制原 `examples/g1/` 下的 ONNX 行走策略，供 Lesson 7/8 使用 |
| 0.3 资源准备 | `OrcaPlayground/envs/euler/robots/config/g1_29dof_hist.yaml` | 复制原 G1 配置（hist 长度、关节顺序等） |
| 0.4 框架代码 | `OrcaPlayground/envs/euler/online_verifier.py` | 实现 `OnlineVerifier` 类（§4.2），含 `check`/`check_allclose`/`check_range`/`observe`/`report` 方法 |
| 0.5 框架代码 | `OrcaPlayground/envs/common/model_scanner.py` | 实现 `build_suffix_template` + `scan_scene_for_template`（§2.0.4），通过 gRPC 扫描场景中的 G1 agent_name |
| 0.6 框架代码 | `OrcaPlayground/envs/euler/g1_base_env.py` | 实现 `G1BaseEnv(OrcaGymEulerEnv)` 基类（§3.1），含 `run_lesson`/`compute_ctrl`/`before_loop`/`verify_step`/`observe_step`/`after_loop`/`verify_final` 钩子；定义 G1 关节/执行器/传感器后缀常量 |
| 0.7 框架代码 | `OrcaPlayground/envs/euler/g1_locomotion.py` | 实现 `G1Locomotion` 类（§3.4），封装 ONNX 策略推理 + 观测组装 + 动作后处理，供 Lesson 7/8 复用 |
| 0.8 目录结构 | `OrcaPlayground/examples/euler/` | 创建 `04_query_api/`、`05_force_apply/`、`06_jacobian/`、`07_studio_capture/`、`08_body_manipulation/` 五个子目录 |
| 0.9 教程文档 | `OrcaPlayground/examples/euler/TUTORIAL.md` | 总览教程：环境准备、Studio 启动、课程索引、通用运行方式（§1.4 流程） |
| 0.10 教程文档 | `OrcaPlayground/examples/euler/00_setup.md` | 环境搭建教程：conda 环境、OrcaStudio 关卡加载、资源路径、首次连通性测试 |

#### 6.2.2 教程文档大纲

**`TUTORIAL.md`（总览）**：
1. OrcaGym Euler 阶段四在线验证简介
2. 环境准备（指向 `00_setup.md`）
3. 手工验证 5 步流程（§1.4）
4. 课程索引：Lesson 4–8 链接
5. 通用运行方式：启动 Studio → 运行脚本 → 观察视口 → 查看报告
6. 常见问题排查（gRPC 连接失败、摄像头未使能、agent_name 扫描失败等）

**`00_setup.md`（环境搭建）**：
1. 前置条件确认（conda orca 环境、OrcaStudio 已安装）
2. 资源路径说明（`envs/euler/robots/` 目录结构）
3. OrcaStudio 启动与关卡加载步骤（含截图位）
4. 首次连通性测试脚本：`python -c "from envs.common.model_scanner import scan_scene_for_template; print(scan_scene_for_template('127.0.0.1:50051', 0.002, template))"`
5. 验证：能扫描到 G1 agent_name 即环境就绪

#### 6.2.3 验收方案

| 验收项 | 通过条件 | 验证方法 |
|--------|---------|---------|
| 资源就位 | `g1_29dof_camera.xml`/`model_6600.onnx`/`g1_29dof_hist.yaml` 存在且可加载 | `ls envs/euler/robots/` 检查；`mujoco.MjModel.from_xml_path()` 加载不报错 |
| `OnlineVerifier` 可用 | `check`/`check_allclose`/`check_range`/`observe`/`report` 方法均能调用 | 单元测试：构造 verifier，调各方法，检查 JSON 报告生成 |
| `model_scanner` 可用 | 能扫描到 Studio 场景中的 G1 agent_name | 启动 Studio + G1 关卡，运行连通性测试脚本 |
| `G1BaseEnv` 可实例化 | `G1BaseEnv(...)` 构造不报错，`reset()` 返回初始观测 | 离线测试（`skip_grpc_load=True`）构造 + reset |
| `G1Locomotion` 可用 | ONNX 策略加载成功，`compute_action(obs)` 返回合法 ctrl | 离线测试：构造 mock obs，检查输出形状与 ctrl_range |
| 教程文档完整 | `TUTORIAL.md` + `00_setup.md` 内容覆盖上述大纲 | 人工审阅 |

#### 6.2.4 完成情况

| 任务 | 状态 | 备注 |
|------|------|------|
| 0.1 `g1_29dof_camera.xml` | ✅ | nq=43, nv=41, nu=29（含 G1 free joint + 29 hinge + box free joint） |
| 0.2 `model_6600.onnx` | ✅ | ONNX 输入 [1,500]，输出 [1,12]，12 维腿部动作 |
| 0.3 `g1_29dof_hist.yaml` | ✅ | 含 history_loco_height_config（400 维历史观测） |
| 0.4 `online_verifier.py` | ✅ | check/check_allclose/check_range/observe/report 全部可用，JSON 报告生成 |
| 0.5 `model_scanner.py` | ✅ | build_suffix_template + scan_scene_for_template 已实现 |
| 0.6 `g1_base_env.py` | ✅ | run_lesson 框架 + before_loop/verify_step/observe_step/after_loop/verify_final 钩子；离线实例化通过 |
| 0.7 `g1_locomotion.py` | ✅ | ONNX 推理通过，compute_action 返回 (29,)，ctrl 范围合法 [-0.19, 0.32] |
| 0.8 目录结构 | ✅ | 04_query_api ~ 08_body_manipulation 五个子目录已规划 |
| 0.9 `TUTORIAL.md` | ✅ | 含阶段四快速开始、5 步验证流程、课程索引 |
| 0.10 `00_setup.md` | ✅ | 含前置条件、资源路径、Studio 加载、连通性测试 |

**验收结论**：步骤 0 全部验收项通过（资源就位 + 框架可用 + ruff SLF001 零告警）。
- `model_scanner` 实际连通性验证需在 OrcaStudio + G1 关卡在线环境执行，离线部分已通过。
- `onnxruntime`、`pyyaml` 已安装到 conda `orca` 环境。

### 6.3 步骤 1：Lesson 4 实施步骤（状态查询）

#### 6.3.1 实施任务清单

| 任务 | 文件 | 说明 |
|------|------|------|
| 1.1 Env 子类 | `OrcaPlayground/envs/euler/query_api_env.py` | 实现 `QueryApiEnv(G1BaseEnv)`，重写 `verify_step`（§4.3.1 的 9 项查询验证）+ `observe_step`（G1 站立观察提示） |
| 1.2 脚本入口 | `OrcaPlayground/examples/euler/04_query_api/query_api.py` | 按 §3.2 模板实现 `if __name__ == "__main__"` 入口，`num_steps=100` |
| 1.3 教程文档 | `OrcaPlayground/examples/euler/04_query_api.md` | 面向用户的使用教程（见 6.3.2） |

#### 6.3.2 教程文档大纲（`04_query_api.md`）

1. **课程目标**：验证 G1 全套状态查询 API（关节/body/site/sensor/接触/质量）在线运行正确
2. **前置条件**：步骤 0 完成；OrcaStudio 已加载 G1 关卡并运行
3. **操作步骤**：
   - 步骤 1（人工）：启动 OrcaStudio，加载含 1 个 G1 的关卡，点击运行
   - 步骤 2（人工）：`cd OrcaPlayground && python examples/euler/04_query_api/query_api.py`
   - 步骤 3（自动）：脚本自动步进 100 帧，每帧执行查询验证
   - 步骤 4（人工）：观察 Studio 视口 G1 站立姿态（应稳定不倒）
   - 步骤 5（自动）：脚本输出判定报告到 `/tmp/euler_Lesson_4_状态查询_API_report.json`
4. **预期结果**：
   - 控制台输出 9 项 `[PASS]` 数值判定（关节维度/qpos 一致性/pelvis 高度/imu 维度等）
   - JSON 报告 `all_passed == true`
   - 视口观察：G1 站立地面，双臂自然下垂
5. **验证 API 列表**：`query_joint_qpos`/`query_joint_qvel`/`query_joint_qacc`/`get_body_xpos_xmat_xquat`/`query_site_pos_and_mat`/`query_sensor_data`/`query_actuator_torques`/`query_contact_simple`/`query_position_body_B`/`body_subtree_mass`
6. **故障排查**：
   - `pelvis_initial_height` 不在 [0.75, 0.85]：检查 G1 初始 keyframe 是否正确
   - `imu_quat_dim` 失败：检查 XML 中 imu_quat sensor 定义
   - gRPC 连接失败：参考 `00_setup.md` 连通性测试

#### 6.3.3 验收方案

| 验收项 | 通过条件 | 验证方法 |
|--------|---------|---------|
| 脚本可运行 | `python query_api.py` 启动后连接 Studio 成功，无异常退出 | 人工运行脚本 |
| 数值判定全通过 | `all_passed == true`，9 项 check 全 PASS | 查看 `/tmp/euler_Lesson_4_*.json` 报告 |
| 人工观察通过 | G1 站立稳定，无抖动/倾倒 | 用户在 Studio 视口确认 |
| API 覆盖完整 | 9 项验证点全部执行（无跳过） | 检查报告 `checks` 数组长度 ≥ 9 |
| 教程文档完整 | `04_query_api.md` 覆盖上述大纲 6 节 | 人工审阅 |

### 6.4 步骤 2：Lesson 5 实施步骤（外力应用）

#### 6.4.1 实施任务清单

| 任务 | 文件 | 说明 |
|------|------|------|
| 2.1 Env 子类 | `OrcaPlayground/envs/euler/force_apply_env.py` | 实现 `ForceApplyEnv(G1BaseEnv)`，重写 `verify_step`（§4.3.2：step 10 施力/step 30 检查抬起+xfrc/step 35 清力/step 50 接触力）+ `observe_step`（抬起/回落观察提示） |
| 2.2 脚本入口 | `OrcaPlayground/examples/euler/05_force_apply/force_apply.py` | 按 §3.2 模板，`num_steps=100` |
| 2.3 教程文档 | `OrcaPlayground/examples/euler/05_force_apply.md` | 面向用户的使用教程 |

#### 6.4.2 教程文档大纲（`05_force_apply.md`）

1. **课程目标**：验证 `apply_body_force`/`clear_body_force`/`clear_all_forces`/`set_geom_friction`/`query_contact_force`/`set_mocap_pos_and_quat` 在线运行正确
2. **前置条件**：步骤 0、1 完成；OrcaStudio 已加载 G1 关卡并运行
3. **操作步骤**：同 Lesson 4 五步流程，脚本路径改为 `examples/euler/05_force_apply/force_apply.py`
4. **预期结果**：
   - step 10：G1 被 200N 向上力抬起（pelvis z 上升 > 0.01m）
   - step 30：`xfrc_applied` 记录力值；清力后 step 35 xfrc 归零
   - step 50：接触法向力 ≈ G1 重力一半
   - mocap 驱动 box：`set_mocap_pos_and_quat` 后 box 跟随到 [0.7, 0, 0.5]
   - JSON 报告 `all_passed == true`
5. **视口观察**：G1 抬起 → 回落 → box 被 anchor 拖动
6. **验证 API 列表**：`apply_body_force`/`clear_body_force`/`clear_all_forces`/`set_geom_friction`/`query_contact_simple`/`query_contact_force`/`set_mocap_pos_and_quat`/`get_body_xpos_xmat_xquat`
7. **故障排查**：
   - `force_lift_pelvis` 失败：检查 `apply_body_force` 的 body_name 是否正确（`{agent}/torso_link`）
   - `mocap_drives_box` 失败：检查 XML 中 weld 约束 `anchor_box_weld` 是否 active

#### 6.4.3 验收方案

| 验收项 | 通过条件 | 验证方法 |
|--------|---------|---------|
| 脚本可运行 | `python force_apply.py` 连接 Studio 成功 | 人工运行 |
| 施力抬起 | `force_lift_pelvis` PASS（z_after > z_before + 0.01） | 查看报告 |
| xfrc 记录/清零 | `xfrc_recorded` + `xfrc_cleared` 均 PASS | 查看报告 |
| 接触力合理 | `contact_normal_force` 在 [half_weight×0.5, half_weight×1.5] | 查看报告 |
| mocap 驱动 box | `mocap_drives_box_via_weld` PASS（atol=0.05） | 查看报告 |
| 人工观察通过 | G1 抬起/回落/box 跟随符合预期 | 用户视口确认 |
| 教程文档完整 | `05_force_apply.md` 覆盖 7 节大纲 | 人工审阅 |

### 6.5 步骤 3：Lesson 6 实施步骤（雅可比 IK）

#### 6.5.1 实施任务清单

| 任务 | 文件 | 说明 |
|------|------|------|
| 3.1 Env 子类 | `OrcaPlayground/envs/euler/jacobian_env.py` | 实现 `JacobianEnv(G1BaseEnv)`，重写 `verify_step`（§4.3.3：pelvis 雅可比形状/imu site 速度一致性/IK 迭代）+ `observe_step`（左脚移动观察） |
| 3.2 脚本入口 | `OrcaPlayground/examples/euler/06_jacobian/jacobian_ik.py` | 按 §3.2 模板，`num_steps=100` |
| 3.3 教程文档 | `OrcaPlayground/examples/euler/06_jacobian.md` | 面向用户的使用教程 |

#### 6.5.2 教程文档大纲（`06_jacobian.md`）

1. **课程目标**：验证 `mj_jacBody`/`mj_jacSite`/`query_site_xvalp_xvalr`/`mj_jac_site` 在线运行正确，IK 迭代收敛
2. **前置条件**：步骤 0、1、2 完成；OrcaStudio 已加载 G1 关卡并运行
3. **操作步骤**：同五步流程，脚本路径 `examples/euler/06_jacobian/jacobian_ik.py`
4. **预期结果**：
   - pelvis 雅可比形状 (3, 35)（nv=6 free + 29 旋转）
   - imu site 速度 = `jacp_site @ qvel`（atol=1e-4）
   - IK 迭代 50 次后左脚到达目标位置（atol=0.02）
   - JSON 报告 `all_passed == true`
5. **视口观察**：左脚抬高约 10cm 到达目标位置
6. **验证 API 列表**：`mj_jacBody`/`mj_jacSite`/`query_site_xvalp_xvalr`/`mj_jac_site`/`get_body_xpos_xmat_xquat`/`set_joint_qpos`/`mj_forward`
7. **故障排查**：
   - `jac_shape` 失败：检查 `self.model.nv` 是否为 35（G1 29 自由度 + 6 free base）
   - `site_vel_vs_jac` 失败：检查 `query_site_xvalp_xvalr` 返回的 dict key 是否为 `{agent}/imu`
   - `ik_foot_target` 不收敛：增加迭代次数或调整步长 `qvel_leg * 0.01`

#### 6.5.3 验收方案

| 验收项 | 通过条件 | 验证方法 |
|--------|---------|---------|
| 脚本可运行 | `python jacobian_ik.py` 连接 Studio 成功 | 人工运行 |
| 雅可比形状 | `jac_shape` PASS（(3, 35)） | 查看报告 |
| site 速度一致性 | `site_vel_vs_jac` PASS（atol=1e-4） | 查看报告 |
| IK 收敛 | `ik_foot_target` PASS（atol=0.02） | 查看报告 |
| 人工观察通过 | 左脚移动到目标位置（抬高约 10cm） | 用户视口确认 |
| 教程文档完整 | `06_jacobian.md` 覆盖 7 节大纲 | 人工审阅 |

### 6.6 步骤 4：Lesson 7 实施步骤（视频录制）

#### 6.6.1 实施任务清单

| 任务 | 文件 | 说明 |
|------|------|------|
| 4.1 Env 子类 | `OrcaPlayground/envs/euler/studio_capture_env.py` | 实现 `StudioCaptureEnv(G1BaseEnv)`，重写 `before_loop`（摄像头检查+begin_save_video）/`verify_step`（帧索引递增）/`observe_step`（行走观察）/`after_loop`（截帧+时间戳+stop_save_video+mp4 检查）；重写 `compute_ctrl` 调用 `G1Locomotion` |
| 4.2 脚本入口 | `OrcaPlayground/examples/euler/07_studio_capture/studio_capture.py` | 按 §3.2 模板，`num_steps=500` |
| 4.3 教程文档 | `OrcaPlayground/examples/euler/07_studio_capture.md` | 面向用户的使用教程 |

#### 6.6.2 教程文档大纲（`07_studio_capture.md`）

1. **课程目标**：验证 `begin_save_video`/`stop_save_video`/`get_current_frame`/`get_next_frame`/`get_frame_png`/`get_camera_time_stamp` 在线运行正确，G1 行走录制产出 mp4 + PNG
2. **前置条件**：步骤 0 完成（含 `g1_locomotion.py`）；OrcaStudio 已加载含摄像头的 G1 关卡并运行
3. **操作步骤**：
   - 步骤 1（人工）：启动 OrcaStudio，加载含 G1（含 `camera_head`）的关卡，点击运行
   - 步骤 2（人工）：`cd OrcaPlayground && python examples/euler/07_studio_capture/studio_capture.py`
   - 步骤 3（自动）：脚本 `before_loop` 检查摄像头使能 + 开始录制；循环 500 帧 ONNX 策略行走；`after_loop` 截帧 + 停止录制 + 检查 mp4
   - 步骤 4（人工）：观察 Studio 视口 G1 行走画面（应稳定行走 10 秒）
   - 步骤 5（自动）：脚本输出判定报告
4. **预期结果**：
   - `camera_enabled` PASS（frame_idx ≥ 0）
   - 帧索引递增（每 50 步检查）
   - PNG 截帧文件生成（`/tmp/g1_frames/color/camera_head_color_*.png`，size > 100）
   - 时间戳查询返回 camera_head
   - mp4 文件生成（`/tmp/g1_walk_video/*.mp4`）
   - JSON 报告 `all_passed == true`
5. **视口观察**：G1 在策略控制下稳定行走 10 秒
6. **验证 API 列表**：`begin_save_video`/`stop_save_video`/`get_current_frame`/`get_next_frame`/`get_frame_png`/`get_camera_time_stamp`
7. **产出物**：
   - `/tmp/g1_walk_video/*.mp4`：行走视频
   - `/tmp/g1_frames/color/camera_head_color_*.png`：截帧图片
   - `/tmp/euler_Lesson_7_*.json`：判定报告
8. **故障排查**：
   - `camera_enabled` 失败（frame_idx=-1）：检查 XML `<camera user="7070 7071">` 端口字段；确认 Studio 已启动视频流服务
   - `mp4_file_generated` 失败：检查 `video_dir` 目录权限；确认 `stop_save_video` 已调用
   - G1 不行走：检查 ONNX 策略路径 `models/dec_loco/model_6600.onnx` 是否存在

#### 6.6.3 验收方案

| 验收项 | 通过条件 | 验证方法 |
|--------|---------|---------|
| 脚本可运行 | `python studio_capture.py` 连接 Studio 成功 | 人工运行 |
| 摄像头使能 | `camera_enabled` PASS（frame_idx ≥ 0） | 查看报告 |
| 帧索引递增 | 多个 `frame_index_increasing_*` PASS | 查看报告 |
| PNG 截帧生成 | `png_file_generated` PASS（文件存在且 size > 100） | 查看报告 + `ls /tmp/g1_frames/color/` |
| 时间戳查询 | `timestamp_returned` PASS | 查看报告 |
| mp4 生成 | `mp4_file_generated` PASS（`/tmp/g1_walk_video/*.mp4` 存在） | 查看报告 + `ls /tmp/g1_walk_video/` |
| 人工观察通过 | G1 稳定行走 10 秒，mp4/PNG 内容可见 locomotion | 用户视口 + 播放 mp4 确认 |
| 教程文档完整 | `07_studio_capture.md` 覆盖 8 节大纲 | 人工审阅 |

### 6.7 步骤 5：Lesson 8 实施步骤（体操作）

#### 6.7.1 实施任务清单

| 任务 | 文件 | 说明 |
|------|------|------|
| 5.1 Env 子类 | `OrcaPlayground/envs/euler/body_manipulation_env.py` | 实现 `BodyManipulationEnv(G1BaseEnv)`，重写 `verify_step`（§4.3.5：step 0-200 拖拽锚定/step 250-350 mocap 驱动 box/step 450-650 equality 重绑驱动 pelvis）+ `observe_step`（行走/拖拽/anchor 拖动 box/G1 提示）；重写 `compute_ctrl` 调用 `G1Locomotion` |
| 5.2 脚本入口 | `OrcaPlayground/examples/euler/08_body_manipulation/body_manipulation.py` | 按 §3.2 模板，`num_steps=700` |
| 5.3 教程文档 | `OrcaPlayground/examples/euler/08_body_manipulation.md` | 面向用户的使用教程 |
| 5.4 Env 层 API 扩展 | `OrcaGymEulerEnv`（`orca_gym/environment/euler/orca_gym_euler_env.py`） | 扩展 `equality_object_ids` 公共方法（委托 `self._gym.equality_object_ids`），见 §4.3.5 说明 |

#### 6.7.2 教程文档大纲（`08_body_manipulation.md`）

1. **课程目标**：验证 `do_body_manipulation`/`anchor_actor`/`release_body_anchored`/`update_equality_constraints`/`modify_equality_objects`/`set_mocap_pos_and_quat`/`equality_object_ids` 在线运行正确
2. **前置条件**：步骤 0、2、4 完成；OrcaStudio 已加载含 mocap+box+weld 的 G1 关卡并运行
3. **操作步骤**：
   - 步骤 1（人工）：启动 OrcaStudio，加载含 G1（含 `ActorManipulator_Anchor`/`manipulation_box`/`anchor_box_weld`）的关卡，点击运行
   - 步骤 2（人工）：`cd OrcaPlayground && python examples/euler/08_body_manipulation/body_manipulation.py`
   - 步骤 3（自动）：脚本循环 700 帧，分阶段验证拖拽锚定/mocap 驱动 box/equality 重绑驱动 pelvis
   - 步骤 4（人工）：按 `[OBSERVE]` 提示在 Studio 视口拖拽 G1 pelvis，观察锚定/释放效果；观察 anchor 拖动 box/G1
   - 步骤 5（自动）：脚本输出判定报告
4. **预期结果**：
   - step 80：人工拖拽 G1 pelvis，`do_body_manipulation` 检测并锚定
   - step 250-350：mocap 驱动 box 到 [0.7, 0, 0.5]（atol=0.05）
   - step 450：停用 equality 后 box 不跟随；重绑 equality 到 pelvis
   - step 650：mocap 驱动 G1 pelvis 位移 > 0.05m
   - JSON 报告 `all_passed == true`
5. **视口观察**：
   - G1 行走中可被鼠标拖拽锚定
   - 释放后恢复行走
   - 绿色球体 anchor 拖动橙色 box
   - 重绑后 anchor 拖动 G1 整机
6. **验证 API 列表**：`do_body_manipulation`/`anchor_actor`/`release_body_anchored`/`update_equality_constraints`/`modify_equality_objects`/`set_mocap_pos_and_quat`/`equality_object_ids`/`get_body_xpos_xmat_xquat`
7. **故障排查**：
   - `mocap_drives_box_via_weld` 失败：检查 XML weld 约束 `anchor_box_weld` 是否 active；确认 `set_mocap_pos_and_quat` 的 dict 参数格式
   - `eq_disable_decouples_box` 失败：检查 `update_equality_constraints` 的 `type=0` 是否正确写入
   - `eq_rebound_to_pelvis` 失败：确认 `equality_object_ids` 已在 Env 层扩展（任务 5.4）
   - `mocap_drives_g1_pelvis` 失败：检查重绑后 weld type 是否恢复为 `mjEQ_WELD`（step 450 已置 0，需在重绑后重新激活）

#### 6.7.3 验收方案

| 验收项 | 通过条件 | 验证方法 |
|--------|---------|---------|
| 脚本可运行 | `python body_manipulation.py` 连接 Studio 成功 | 人工运行 |
| 拖拽锚定 | 用户拖拽 G1 pelvis，G1 跟随鼠标 | 用户视口确认 |
| 释放恢复 | 释放鼠标后 G1 恢复运动 | 用户视口确认 |
| mocap 驱动 box | `mocap_drives_box_via_weld` PASS（atol=0.05） | 查看报告 |
| equality 停用解耦 | `eq_disable_decouples_box` PASS | 查看报告 |
| equality 重绑 | `eq_rebound_to_pelvis` PASS（obj2_id == pelvis_id） | 查看报告 |
| mocap 驱动 G1 | `mocap_drives_g1_pelvis` PASS（位移 > 0.05m） | 查看报告 |
| 人工观察通过 | 拖拽/anchor 拖动 box/anchor 拖动 G1 符合预期 | 用户视口确认 |
| Env 层 API 扩展 | `equality_object_ids` 在 Env 层可调用 | `env.equality_object_ids(0)` 返回 tuple |
| 教程文档完整 | `08_body_manipulation.md` 覆盖 7 节大纲 | 人工审阅 |

### 6.8 教程文档编写规范

所有 Lesson 教程文档（`04_query_api.md` ~ `08_body_manipulation.md`）遵循统一结构：

| 章节 | 内容 | 说明 |
|------|------|------|
| 1. 课程目标 | 验证的 API 列表 + 预期行为 | 一句话概括 |
| 2. 前置条件 | 依赖步骤 + Studio 状态 | 明确依赖关系 |
| 3. 操作步骤 | §1.4 五步流程的具体化 | 含具体命令行 |
| 4. 预期结果 | 数值判定项 + 期望值 | 对应 `verifier.check` 项 |
| 5. 视口观察 | 人工观察项描述 | 对应 `verifier.observe` 项 |
| 6. 验证 API 列表 | 本课涉及的所有公共 API | 便于回顾 |
| 7. 产出物 | 生成的文件路径（mp4/PNG/JSON） | Lesson 7/8 有产出物 |
| 8. 故障排查 | 常见失败原因 + 解决方案 | 对应 `check` 项的失败场景 |

> **文档风格**：
> - 面向用户（非开发者），避免内部实现细节
> - 命令行用代码块标注，路径用反引号
> - 故障排查按「失败现象 → 原因 → 解决方案」三段式
> - 每个 Lesson 文档控制在 100 行以内，保持简洁

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