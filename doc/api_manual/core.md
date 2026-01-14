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


###### `OrcaGymBase.print_opt_config`

Signature:

```python
def print_opt_config(self)
```


###### `OrcaGymData.update_qfrc_bias`

Signature:

```python
def update_qfrc_bias(self, qfrc_bias)
```


#### `class OrcaGymLocal`

> OrcaGym 本地仿真接口


###### `OrcaGymLocal.init_simulation`

Signature:

```python
async def init_simulation(self, model_xml_path)
```


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

</details>
