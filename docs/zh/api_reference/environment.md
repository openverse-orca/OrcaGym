# 🌍 Environment API

Gymnasium 环境接口，提供标准的强化学习环境抽象。

## 主要类

| 类 | 说明 |
|----|------|
| **`OrcaGymEulerEnv`** | 推荐的环境基类 |
| `OrcaGymVectorEnv` | 向量化并行环境 |

---

## OrcaGymEulerEnv

`OrcaGymEulerEnv` 是你编写自定义环境的推荐基类。它封装了仿真核心，让你专注于任务逻辑。

### 构造参数

```python
class OrcaGymEulerEnv:
    def __init__(
        self,
        frame_skip: int,           # 每次 step() 对应的物理仿真步数
        orcagym_addr: str,         # gRPC 服务端地址（如 "localhost:50051"）
        agent_names: list[str],    # 智能体名称列表
        time_step: float,          # 物理仿真时间步长（秒）
        *,
        model_xml_path: str | None = None,   # 本地 XML 路径（离线模式）
        skip_grpc_load: bool = False,        # True → 离线模式
        render_mode: str = "human",          # "human" / "none"
        sync_render: bool = False,
        **kwargs,
    )
```

### 公共属性

```python
data: OrcaGymDataView          # 完整状态只读视图
model: OrcaGymModel            # 模型结构信息
sim_config: SimConfig          # 求解器参数配置
ctrl: np.ndarray               # getter 返回当前 actuator_force；setter 设置控制输入
init_qpos: np.ndarray          # 缓存的初始广义坐标
init_qvel: np.ndarray          # 缓存的初始广义速度
frame_skip: int                # step() 对应的物理步数
seed: int                      # 随机种子

@property
dt: float                      # 环境时间步长 = sim_config.timestep × frame_skip
```

### 仿真控制

```python
def do_simulation(ctrl: np.ndarray, n_frames: int) -> None
```
核心步进方法。设置控制 → 步进 n_frames → 自动同步状态。调用后 `self.data` 已为最新状态。ctrl 形状必须为 `(nu,)`。

```python
def set_ctrl(ctrl: np.ndarray) -> None
def mj_step(nstep: int) -> None
def mj_forward() -> None
```
`set_ctrl` 设置控制输入但不步进；`mj_step` / `mj_forward` 为低级仿真控制，通常不需要直接调用。

### 状态设置

```python
def set_joint_qpos(qpos: np.ndarray) -> None   # 设置广义坐标（全量）
def set_joint_qvel(qvel: np.ndarray) -> None   # 设置广义速度（全量）
```
设置后需调用 `mj_forward()` 更新派生量。

### 力应用

```python
def apply_body_force(body_name: str, force: np.ndarray, torque: np.ndarray) -> None
def clear_body_force(body_name: str) -> None
def clear_all_forces() -> None
def mj_apply_force_at_site(site_name: str, force: np.ndarray, torque: np.ndarray) -> None
def mj_clear_xfrc_applied_for_site(site_name: str) -> None
```

### Mocap 与几何体设置

```python
def set_mocap_pos_and_quat(mocap_pos_and_quat_dict: dict) -> None
def set_geom_friction(geom_friction_dict: dict) -> None
def add_extra_weight(weight_load_dict: dict) -> None
```

### 状态查询（按名称，无需 ID）

```python
# 关节查询
def query_joint_qpos(joint_names: list[str]) -> dict[str, np.ndarray]
def query_joint_qvel(joint_names: list[str]) -> dict[str, np.ndarray]
def query_joint_qacc(joint_names: list[str]) -> dict[str, np.ndarray]
def query_joint_offsets(joint_names: list[str]) -> dict[str, np.ndarray]   # 关节偏移
def query_joint_lengths(joint_names: list[str]) -> dict[str, np.ndarray]   # 关节长度
def query_joint_dofadrs(joint_names: list[str]) -> dict[str, int]           # 关节 dof 起始地址
def jnt_qposadr(joint_name: str) -> int
def jnt_dofadr(joint_name: str) -> int

# Body 位姿
def get_body_xpos_xmat_xquat(body_name_list: list[str]) -> dict
def get_body_xpos_xmat_xquat_xvel(body_name_list: list[str]) -> dict

# Site 查询
def query_site_pos_and_mat(site_names: list[str]) -> dict
def query_site_size(site_names: list[str]) -> dict[str, np.ndarray]

# 传感器/执行器/接触
def query_sensor_data(sensor_names: list[str]) -> dict[str, np.ndarray]
def query_actuator_torques(actuator_names: list[str]) -> dict[str, np.ndarray]
def query_contact_simple() -> list[dict]
def query_contact_force(contact_ids: list[int]) -> dict[int, np.ndarray]
def get_cfrc_ext() -> np.ndarray
def get_goal_bounding_box(geom_name: str) -> np.ndarray   # geom 包围盒半尺寸 (3,)
def body_subtree_mass(body_name: str) -> float
```

### 基座坐标系变换

```python
def query_site_pos_and_quat_B(site_names, base_body_list) -> dict
def query_site_xvalp_xvalr(site_names) -> tuple[dict, dict]
def query_site_xvalp_xvalr_B(site_names, base_body_list) -> tuple[dict, dict]
def query_velocity_body_B(ee_body, base_body) -> np.ndarray       # 6D 速度（基座系）
def query_position_body_B(ee_body, base_body) -> np.ndarray       # 3D 位置（基座系）
def query_orientation_body_B(ee_body, base_body) -> np.ndarray    # 四元数（基座系）
def query_joint_axes_B(joint_names, base_body) -> dict            # 关节轴方向（基座系）
```

### 里程计查询

```python
def query_robot_velocity_odom(base_body, initial_base_pos, initial_base_quat) -> tuple
def query_robot_position_odom(base_body, initial_base_pos, initial_base_quat) -> np.ndarray
def query_robot_orientation_odom(base_body, initial_base_pos, initial_base_quat) -> np.ndarray
```

### 雅可比

```python
def mj_jacBody(jacp: np.ndarray, jacr: np.ndarray, body_name: str) -> None
def mj_jacSite(jacp: np.ndarray, jacr: np.ndarray, site_name: str) -> None
def mj_jac_site(site_names: list[str]) -> dict[str, dict]
```

### 等式约束与体操作

Env 层公共原语（程序化操作应使用以下方法编排）：

```python
def equality_find_slot_by_body(body_name: str) -> int          # 查找含指定 body 的等式约束槽位，未找到返回 -1
def equality_constraint(slot: int) -> dict                     # 读取单个等式约束完整数据
def equality_update(
    slot: int,
    *,
    eq_type: int | None = None,       # mjtEq 类型常量
    obj1_name: str | None = None,     # 新的 obj1 body 名称
    obj2_name: str | None = None,     # 新的 obj2 body 名称
    data: np.ndarray | None = None,   # 约束数据 (mjNEQDATA,)
    active: bool | None = None,       # 是否激活
    solref: np.ndarray | None = None, # 求解器参考参数 (2,)
    solimp: np.ndarray | None = None, # 求解器 impedance 参数 (5,)
    forward: bool = True,             # 写入后是否调用 mj_forward
) -> None
```

> ⚠️ **注意**：以下方法在 Env 层不存在：
> - `update_equality_constraints` / `modify_equality_objects`：仅存在于
>   `OrcaGymEuler`（gym 层）与 `MuJoCoSimCore`（sim 层）。
>   `modify_equality_objects` 签名为
>   `modify_equality_objects(eq_ids: list[int], obj1_ids=None, obj2_ids=None)`，
>   参数为 int 列表而非 names。
> - `update_anchor_equality_constraints` / `anchor_actor` /
>   `release_body_anchored` / `do_body_manipulation`：为 Env 层内部 `_` 前缀
>   方法（`_anchor_actor` / `_release_body_anchored` / `_do_body_manipulation`），
>   由 `render()` 内部驱动的 UI 抓取状态机使用，不应直接调用。
>   程序化操作应使用 `equality_find_slot_by_body` + `equality_constraint` +
>   `equality_update` + `set_mocap_pos_and_quat` 原语编排。

### 只读查询

```python
def geom_friction(geom_name: str) -> np.ndarray   # geom 摩擦系数 (3,) [sliding, torsion, rolling]
```

### Studio 交互

```python
def render(simulate_index: int = -1, request_idr: bool = False) -> np.ndarray | None
    # 渲染到 Studio。simulate_index 透传到引擎相机管线用于帧对齐，
    # -1 表示由服务端自增（默认值）。启用客户端录制时应传入 >=0 的递增值。
    # request_idr=True 请求引擎在本次渲染输出一个 IDR 关键帧（录制段起点使用，
    # 配合 save_streaming 内部默认的前向截断使视频从关键帧开始）。

# 以下方法已废弃（引擎侧 MP4 录制 RPC 已删除），调用时发出 DeprecationWarning：
def begin_save_video(file_path, capture_mode=0) -> None       # [Deprecated] no-op
def stop_save_video() -> None                                  # [Deprecated] no-op
def get_current_frame() -> int                                 # [Deprecated] 返回 -1
def get_next_frame() -> int                                    # [Deprecated] 返回 0
def get_camera_time_stamp(last_frame_index) -> dict            # [Deprecated] 返回 {}

def get_frame_png(image_path) -> None
def load_content_file(content_file_name, **kwargs) -> None
```

### 相机录制 API（客户端 PyAV remux）

```python
def save_streaming(
    camera_name: str,
    camera_type: str,
    file_path: str,
    start_simulate_index: int,
    end_simulate_index: int,
) -> Future[RemuxResult]
    # 保存指定相机 [start, end] 区间的视频流为 MP4。**非阻塞**，返回 Future。
    # 通过 VideoRecorderManager 统一接口操作：幂等启动录制器，并在该相机的
    # 等待任务队列中注册一个区间保存任务。
    # 每个区间任务独立携带自己的 start/end，可同时注册多个互不干扰的区间。
    # 当接收线程收到 simulate_index >= end 的帧时，由保存 worker 线程异步执行
    # PyAV remux，不阻塞接收线程与上层调用线程。
    # 因此可容忍「物理仿真步 → 引擎渲染 → 取帧」的延迟（如保存 0-500 时缓存
    # 可能只有 490 帧，任务会等第 500 帧到达后再保存）。
    # 内部默认前向截断到区间内第一个关键帧（``truncate_to_keyframe=True``），
    # 保证输出视频可正常播放（配合录制起点 ``render(request_idr=True)``
    # 使视频从关键帧开始）。
    # env.close() 会自动保存未完成的录制任务（阻塞等待 remux 完成）。
    # 前置条件：已调用 ``start_streaming`` 启动推流。
    # 内部 remux_range 使用 timestamp_ns 作为 PTS 时间基（非固定 FPS），
    # 并返回 RemuxResult（含 frame_indices / timestamps_ns 帧号↔物理 index 映射）。

def set_render_fps(fps: int) -> None
    # 设置渲染帧率（render FPS）。控制 render() 调用引擎渲染的频率：
    # 同步渲染（sync_render=True）每隔 1/fps 个物理步渲染一帧；
    # 异步渲染（sync_render=False）每隔 1/fps 秒渲染一帧。

def set_sync_render(enabled: bool) -> None
    # 设置是否启用同步渲染。启用录制做帧对齐时需开启（enabled=True），
    # 使 render() 每物理步调用引擎渲染并透传 simulate_index。

def set_video_recorder_manager(manager: VideoRecorderManager | None) -> None
    # 注入 VideoRecorderManager 实例。环境层相机属性查询/设置与录制统一
    # 转发到该管理器。为 None 时，后续首次调用相机/录制接口会由
    # CreateVideoRecorderManager(self.stub, self.loop) 惰性创建。
```

底层 ``VideoRecorderManager`` 统一接口（``orca_gym.recorder`` 模块）。
相机属性查询/设置与推流状态机由 ``VideoRecorderManager`` 直接基于 gRPC stub
（``GrpcServiceStub``）实现（实际执行者），环境层（``OrcaGymLocalEnv`` /
``OrcaGymEulerEnv``）仅做转发。

```python
from orca_gym.recorder import CreateVideoRecorderManager, RemuxResult
from concurrent.futures import Future

# stub 为 gRPC 能力 stub（GrpcServiceStub），提供相机属性查询/设置 + 推流状态
# 切换的接口；可为 None（仅使用录制能力，不提供相机配置）。
# loop 为所属环境的事件循环（self.loop），用于同步桥接 stub 异步接口。
manager = CreateVideoRecorderManager(stub=env.stub, loop=env.loop)
manager.start_recorder(camera_name, color_port=7070)      # 幂等启动
future: Future[RemuxResult] = manager.save_streaming(
    camera_name, file_path, start_idx, end_idx
)  # 注册区间保存任务，非阻塞返回
result: RemuxResult = future.result()                    # 等待保存完成（可选）
# result.file_path / result.frame_count / result.frame_indices / result.timestamps_ns
manager.stop_all_and_save() -> dict[str, RemuxResult]    # env.close() 自动保存（阻塞等待）
```

> 任务队列抽象：等待队列中的每个保存任务（``RecordingTask``）独立携带触发回调
> （``trigger_fn``）与执行逻辑（``execute``）。触发条件是回调函数
> ``(task, current_simulate_index) -> bool``，接收线程逐帧轮询判断，便于后续扩展
> 新的任务类型（如按时间戳、按帧数触发）。

### 实时视频可视化（``VideoStreamViewer``）

``VideoRecorderManager`` 提供实时可视化能力，启动**独立子进程**建立
WebSocket 连接接收 H.264 码流、解码、用 matplotlib 渲染显示。子进程与主
进程完全解耦，不读取主进程的滚动缓存，不阻塞接收线程、保存 worker 与
上层仿真主线程。

```python
from orca_gym.recorder import CreateVideoRecorderManager, VideoStreamViewer

manager = CreateVideoRecorderManager(stub=env.stub, loop=env.loop)
manager.start_recorder(camera_name, color_port=7070)   # 先确保录制器已启动

# 非阻塞启动可视化窗口（子进程独立建立 WebSocket + matplotlib 显示）
viewer: VideoStreamViewer = manager.start_viewer(camera_name, window_name=None)
viewer.is_running

manager.get_viewer(camera_name)        # 获取查看器（未启动返回 None）
manager.stop_viewer(camera_name)       # 停止某相机窗口
manager.stop_all_viewers()             # 停止所有窗口
manager.get_viewer_stats()             # 所有窗口状态统计
```

独立使用 ``VideoStreamViewer``（不经过管理器）:

```python
from orca_gym.recorder import VideoStreamViewer

viewer = VideoStreamViewer(recorder, window_name="Camera")
viewer.start()
# ... 仿真循环中 ...
viewer.stop()
```

依赖 ``matplotlib`` / ``numpy`` / ``av`` / ``websockets`` / ``opencv-python``。
窗口通过关闭按钮或调用 ``viewer.stop()`` / ``manager.stop_viewer()`` 关闭
（内部通过 ``stop_event`` 信号通知子进程退出）。

### 相机属性查询/设置 + 推流状态机

```python
def get_camera_names() -> list[str]
def get_camera_properties(camera_name: str) -> GetCameraPropertiesResponse
def set_camera_properties(
    camera_name: str,
    **kwargs,   # 可选参数：capture_rgb, capture_depth, capture_normal, capture_object_color, random_object_color, use_nvenc, nvenc_gpu_index, width, height, vertical_fov, near_clip, far_clip, gamma, color_port, depth_port, use_dds, dds_topic, dds_stream_id
) -> None
def set_streaming_enabled(camera_name: str, enabled: bool) -> None
def make_camera_viewport_active(actor_name: str, entity_name: str) -> None
```

状态机约束：
- `camera_name` 可通过 `get_camera_names()` 枚举获取
- `set_camera_properties` 仅在 `Idle` 状态允许；`Streaming` 状态下需先调用 `set_streaming_enabled(False)` 回到 `Idle` 再设置属性
- `set_streaming_enabled(True)` 进入 `Streaming` 状态后，对应端口（如 7070/7071）开始监听并推流
- 客户端 PyAV 录制由 `save_streaming` 控制，与本组接口正交（但需要先 `set_streaming_enabled(True)` 启动推流）

### Studio 桥接

```python
def studio_bridge()   # 返回 OrcaStudio 桥接对象（K9 方法访问模式）
```

### 名称空间解析（多智能体）

继承自 `OrcaGymEnvMixin`，用于在多智能体环境中自动为实体名称添加智能体前缀。

```python
@property
agent_num: int                          # 智能体数量

def body(name: str, agent_id: int | None = None) -> str       # body 名称 → "agent_name_body_name"
def joint(name: str, agent_id: int | None = None) -> str      # 关节名称
def actuator(name: str, agent_id: int | None = None) -> str   # 执行器名称
def site(name: str, agent_id: int | None = None) -> str       # site 名称
def mocap(name: str, agent_id: int | None = None) -> str      # mocap 名称
def sensor(name: str, agent_id: int | None = None) -> str     # 传感器名称
```

> `agent_id=None` 时默认使用第一个智能体（索引 0）。

**示例：**
```python
# 单智能体环境（agent_names=["robot_1"]）
env.body("pelvis")    # → "robot_1_pelvis"
env.joint("leg_l_1")  # → "robot_1_leg_l_1"

# 多智能体环境（agent_names=["robot_1", "robot_2"]）
env.body("pelvis", agent_id=1)  # → "robot_2_pelvis"
```

### 动作/观测空间生成

继承自 `OrcaGymEnvMixin`，用于便捷地生成符合 Gymnasium 规范的动作空间和观测空间。

```python
def generate_action_space(bounds: np.ndarray) -> gym.Space      # (nu, 2) → Box 动作空间
def generate_observation_space(obs: np.ndarray | dict) -> gym.Space  # 从观测样例生成观测空间
```

`generate_action_space` 会自动处理 ±inf 边界（裁剪到 float32 可表示范围），避免 gymnasium 的溢出警告。

**示例：**
```python
# 根据执行器控制范围生成动作空间
ctrlrange = self.model.get_actuator_ctrlrange()  # (nu, 2)
self.action_space = self.generate_action_space(ctrlrange)

# 根据观测样例生成观测空间
obs_sample = self._get_obs()
self.observation_space = self.generate_observation_space(obs_sample)
```

### 随机种子

```python
def set_seed_value(seed: int) -> list[int]     # 设置随机数种子，返回种子列表
```

设置后可通过 `self.np_random` 使用 `RandomState` 实例。

### 重置（Gymnasium 标准接口）

```python
def reset(*, seed: int | None = None, options: dict | None = None) -> tuple[ObsType, dict]
```

由 `OrcaGymEnvMixin` 提供，编排顺序：`set_seed_value` → `reset_simulation` → `reset_model` → `render`。

> 子类应复写 `reset_model()` 而非直接复写 `reset()`。

### 抽象方法（子类必须实现）

```python
def step(action: np.ndarray) -> tuple[ObsType, float, bool, bool, dict]
def reset_model() -> tuple[dict, dict]
def _get_obs() -> np.ndarray | dict
```

### 生命周期方法

```python
def initialize_grpc()
def initialize_simulation()     # 加载模型
def reset_simulation()          # 重置状态
def init_qpos_qvel()            # 缓存初始状态
def set_time_step(time_step)    # 设置时间步长
def pause_simulation()
def close()                     # 关闭连接
```

### 使用示例

```python
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class MyRobotEnv(OrcaGymEulerEnv):
    """最简环境：Box 观测 + Box 动作，离线模式。"""

    def __init__(self, model_xml_path: str):
        super().__init__(
            frame_skip=5,
            orcagym_addr="localhost:50051",
            agent_names=["robot_1"],
            time_step=0.001,
            model_xml_path=model_xml_path,
            skip_grpc_load=True,   # 离线模式
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        obs_sample = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
        ]).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = False
        truncated = False
        info = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs: np.ndarray) -> float:
        return 0.0  # 替换为你的奖励函数

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
        self.set_joint_qpos(qpos)
        self.set_joint_qvel(qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}


# 使用
if __name__ == "__main__":
    env = MyRobotEnv(model_xml_path="/path/to/scene.xml")
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()
```

---

## OrcaGymVectorEnv

向量化环境，并行执行多个环境。继承自 Gymnasium `VectorEnv`。

```python
class OrcaGymVectorEnv(VectorEnv):
    def __init__(self, num_envs: int, worker_index: int, entry_point: str, **kwargs)
    def step(actions) -> tuple[obs, rewards, terminations, truncations, infos]
    def reset(*, seed=None, options=None) -> tuple[obs, infos]
```

### 公共属性

```python
num_envs: int                        # 并行环境数量
observation_space: gym.Space         # 批量观测空间
single_observation_space: gym.Space  # 单环境观测空间
action_space: gym.Space              # 批量动作空间
single_action_space: gym.Space       # 单环境动作空间
```

### 主要方法

```python
def reset(*, seed=None, options=None) -> tuple[ObsType, list[dict]]
def step(actions: ActType) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, list[dict]]
def render() -> None
def close() -> None
```

### 返回值说明

| 返回值 | 形状/类型 | 说明 |
|--------|-----------|------|
| `observations` | `(num_envs, *obs_shape)` | 所有环境的观测 |
| `rewards` | `(num_envs,)` | 所有环境的奖励 |
| `terminated` | `(num_envs,) bool` | 是否终止 |
| `truncated` | `(num_envs,) bool` | 是否截断 |
| `infos` | `list[dict]` | 每个环境的 info 字典 |

---

## RewardType

模块路径：`from orca_gym.environment.orca_gym_env import RewardType`

```python
class RewardType:
    SPARSE = "sparse"
    DENSE = "dense"
```
