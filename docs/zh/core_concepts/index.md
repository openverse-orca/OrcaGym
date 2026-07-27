# 🧠 核心概念

理解 OrcaGym 中的关键概念，帮助你更好地使用环境。

## 关键对象关系

在使用 OrcaGym 时，你通过环境对象与仿真世界交互：

```
env (Gymnasium Environment)
  ├── model (OrcaGymModel)    — 模型信息（结构，不变）
  ├── data  (OrcaGymDataView) — 仿真状态（动态，每步变化）
  └── sim_config              — 仿真参数配置
```

---

## 概念速查表

| 概念 | 说明 |
|------|------|
| **Model** | 静态模型信息（几何、关节、执行器、传感器） |
| **Data** | 动态仿真状态（位置、速度、加速度、时间） |
| **SimConfig** | 仿真参数（时间步长、求解器设置、重力） |
| **qpos** | 广义坐标（位置），长度 nq |
| **qvel** | 广义速度，长度 nv |
| **ctrl** | 控制输入，长度 nu |
| **等式约束** | 用于物体操作的连接约束 |
| **Mocap Body** | 可通过设置位姿来"操控"的特殊 body |

---

## 仿真 = 模型 + 数据

OrcaGym 将仿真世界分为两部分：

| 概念 | 类型 | 比喻 | 例子 |
|------|------|------|------|
| `env.model` | `OrcaGymModel` | 机器人的**说明书**（不会变） | 有几个关节、每个关节叫什么名字 |
| `env.data` | `OrcaGymDataView` | 机器人的**当前状态**（每步都变） | 关节现在转了多少度、速度是多少 |

```python
# model — 静态，描述结构
print(env.model.nq)            # 位置变量数
print(env.model.nv)            # 速度变量数
print(env.model.nu)            # 控制维度数

# data — 动态，反映当前状态
print(env.data.qpos)           # 当前位置 → 每一步仿真后都会变
print(env.data.qvel)           # 当前速度
print(env.data.time)           # 仿真时间
```

---

## 仿真时间

```
time_step  = 0.001 秒    ← 物理引擎每步的时间
frame_skip = 20          ← 每次 step() 物理引擎走几步
dt = 0.001 × 20 = 0.02秒 ← 你的控制指令每隔多久更新一次（50Hz）
```

控制频率：`control_hz = 1.0 / dt`

---

## 维度约定

| 变量 | 长度 | 含义 |
|------|------|------|
| `model.nq` | 广义坐标数 | qpos 长度 |
| `model.nv` | 自由度数 | qvel 长度 |
| `model.nu` | 执行器数 | 控制输入长度 |

---

## 关节类型与 qpos/qvel 维度

不同关节类型在 `qpos` 和 `qvel` 中占用不同数量的元素：

| 关节类型 | qpos 大小 | qvel 大小 | 示例 |
|----------|-----------|-----------|------|
| FREE | 7 (3 pos + 4 quat) | 6 (3 lin + 3 ang) | 自由飞行体 |
| BALL | 4 (quaternion) | 3 (angular velocity) | 球关节 |
| HINGE | 1 (angle) | 1 (angular velocity) | 旋转关节 |
| SLIDE | 1 (displacement) | 1 (linear velocity) | 滑动关节 |

---

## 阅读顺序建议

1. [Model / Data / Config](model-data-opt.md) — 理解三种核心数据对象
2. [Gymnasium 接口](gym-interface.md) — 理解标准 RL 接口
3. [数据流](data-flow.md) — 理解数据如何在仿真中流动
4. [架构总览](architecture-overview.md) — 理解整体分层设计、API 边界、调用流
5. [系统架构](architecture.md) — 理解组件设计、API 契约、封装隔离与迁移指南
