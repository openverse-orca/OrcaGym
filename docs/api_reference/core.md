# 🧬 Core API

核心仿真接口，封装了 MuJoCo 物理引擎和模型管理功能。

## 主要类

| 类 | 说明 |
|----|------|
| `OrcaGymModel` | 静态模型信息（几何、关节、执行器等） |
| `OrcaGymDataView` | 仿真状态只读视图（qpos、qvel 等） |
| `SimConfig` | 求解器参数配置 |

---

## OrcaGymModel — 静态模型信息

包含所有在仿真过程中不变的信息，可通过 `env.model` 访问。

### 维度属性

```python
model.nq: int        # qpos 长度
model.nv: int        # qvel/qacc 长度（自由度数）
model.nu: int        # 执行器数量
model.ngeom: int     # 几何体数量
model.neq: int       # 等式约束数量
model.nmocap: int    # mocap body 数量
```

### 实体类型术语

| 实体 | 说明 |
|------|------|
| **Body** | 刚体，物理仿真基本单元。有质量、惯性、位置、姿态。 |
| **Joint** | 关节，连接 body 的约束。定义相对运动（旋转/滑动/自由）。 |
| **Actuator** | 执行器，驱动机器人的元件（电机等）。对应动作空间维度。 |
| **Geom** | 几何体，用于碰撞检测的几何形状。 |
| **Site** | 标记点，不参与物理仿真。用于标记关键位置。 |
| **Sensor** | 传感器，测量物理量的虚拟设备。 |
| **Equality** | 等式约束，强制两个 body 满足特定关系。常用于抓取。 |
| **Mocap Body** | 虚拟 body，可自由移动，不受物理约束。 |

### 名称↔ID 映射

| 实体 | name→id | id→name | 获取全部信息 |
|------|---------|---------|------------|
| Body | `body_name2id(n)` | `body_id2name(i)` | `get_body_dict()` |
| Joint | `joint_name2id(n)` | `joint_id2name(i)` | `get_joint_dict()` |
| Actuator | `actuator_name2id(n)` | `actuator_id2name(i)` | `get_actuator_dict()` |
| Geom | `geom_name2id(n)` | `geom_id2name(i)` | `get_geom_dict()` |
| Site | `site_name2id(n)` | `site_id2name(i)` | `get_site_dict()` |
| Sensor | `sensor_name2id(n)` | `sensor_id2name(i)` | `gen_sensor_dict()` |
| Mesh | `mesh_name2id(n)` | `mesh_id2name(i)` | `get_mesh_dict()` |

### 其他查询

```python
def get_body_names()
def get_actuator_ctrlrange() -> np.ndarray    # (nu, 2) 控制范围
def get_joint_qposrange(joint_names) -> np.ndarray
def get_eq_list() -> list
def get_mocap_dict() -> dict
def get_geom_body_name(geom_id: int) -> str
def get_geom_body_id(geom_id: int) -> int
```

---

## OrcaGymDataView — 仿真状态只读视图

通过 `env.data` 访问。`do_simulation()` 后自动更新为最新状态。

### 基本状态字段

```python
qpos: np.ndarray       # (nq,)  广义坐标
qvel: np.ndarray       # (nv,)  广义速度
qacc: np.ndarray       # (nv,)  广义加速度
qfrc_bias: np.ndarray  # (nv,)  偏置力（重力+科氏力+离心力）
time: float            # 仿真时间（秒）
```

### 扩展字段

```python
xfrc_applied: np.ndarray    # 外力（只读，写入用 apply_body_force）
actuator_force: np.ndarray  # 执行器力
contact: list               # 接触列表
cfrc_ext: np.ndarray        # 外部约束力 (nbody, 6)
```

### Body 查询（按名称，无需 ID）

```python
def body_xpos(body_name: str) -> np.ndarray       # 世界坐标位置 (3,)
def body_xquat(body_name: str) -> np.ndarray      # 四元数 [w,x,y,z] (4,)
def body_xmat(body_name: str) -> np.ndarray       # 旋转矩阵扁平存储 (9,)
def body_cvel(body_name: str) -> np.ndarray       # 空间速度 [ang(3), lin(3)] (6,)
def body_subtree_mass(body_name: str) -> float    # 子树总质量
```

### Site 查询

```python
def site_xpos(site_name: str) -> np.ndarray       # 世界坐标位置 (3,)
def site_xmat(site_name: str) -> np.ndarray       # 旋转矩阵扁平存储 (9,)
```

### Geom 查询

```python
def geom_xpos(geom_name: str) -> np.ndarray       # 世界坐标位置 (3,)
def geom_xmat(geom_name: str) -> np.ndarray       # 旋转矩阵扁平存储 (9,)
def geom_size(geom_name: str) -> np.ndarray       # 尺寸 (3,)
```

### Mocap 查询

```python
def mocap_pos(body_name: str) -> np.ndarray       # mocap 位置 (3,)
def mocap_quat(body_name: str) -> np.ndarray      # mocap 四元数 [w,x,y,z] (4,)
```

---

## SimConfig — 求解器配置

通过 `env.sim_config` 访问。修改在下次仿真步进时生效。

### Property

| 属性 | 类型 | 说明 |
|------|------|------|
| `timestep` | `float` | 物理时间步长 |
| `integrator` | `int` | 积分器类型（0=Euler, 1=RK4） |
| `iterations` | `int` | 求解器迭代次数 |
| `gravity` | `np.ndarray(3,)` | 重力向量 |

### 方法

```python
def load_from_dict(config: dict) -> None    # 批量设置参数
def to_dict() -> dict                       # 导出配置为字典
```

### 使用示例

```python
env.sim_config.timestep = 0.002
env.sim_config.iterations = 100
env.sim_config.load_from_dict({"integrator": 0, "iterations": 100})
```

---

## 辅助枚举与函数

### AnchorType

```python
class AnchorType:
    NONE = 0   # 无锚定
    WELD = 1   # 焊接锚定（完全固定位置和姿态）
    BALL = 2   # 球关节锚定（固定位置，允许旋转）
```

### CaptureMode

```python
class CaptureMode:
    ASYNC = 0  # 异步视频捕获
    SYNC = 1   # 同步视频捕获
```

### 工具函数

```python
def get_qpos_size(joint_type: int) -> int  # 关节在 qpos 中的元素数
def get_dof_size(joint_type: int) -> int   # 关节自由度数
```
