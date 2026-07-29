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
def render() -> np.ndarray | None              # 渲染到 Studio
def begin_save_video(file_path, capture_mode=0) -> None
def stop_save_video() -> None
def get_current_frame() -> int
def get_next_frame() -> int
def get_camera_time_stamp(last_frame_index) -> dict
def get_frame_png(image_path) -> None
def load_content_file(content_file_name, **kwargs) -> None
```

### 摄像头传感器配置

```python
def set_camera_sensor_info(
    actor_name: str,
    capture_rgb: bool,
    capture_depth: bool,
    save_mp4_file: bool = False,
    use_dds: bool = False,
    **kwargs,   # 可选扩展参数：capture_normal, width, height, vertical_fov, near_clip, far_clip, gamma, color_port, depth_port, ...
) -> None
def make_camera_viewport_active(actor_name: str, entity_name: str) -> None
```

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
