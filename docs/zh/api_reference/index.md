# 📖 API 参考

OrcaGym 的 API 文档，帮助你快速查找需要的接口。

## 模块索引

| 模块 | 说明 |
|------|------|
| [🧬 Core API](core.md) | 仿真核心、模型、状态、求解器配置 |
| [🌍 Environment API](environment.md) | Gymnasium 环境基类和方法 |
| [🎬 Scene API](scene.md) | 场景管理、Actor、灯光、材质 |
| [📷 Sensor API](sensor.md) | 相机、传感器数据获取 |
| [🔧 Utils API](utils.md) | 逆运动学、关节控制、旋转工具 |

## 快速导航

### 新手入门

1. 从 [Environment API](environment.md) 的 `OrcaGymEulerEnv` 开始——这是编写环境的主入口
2. 了解 [Core API](core.md) 中的 Model、Data、SimConfig
3. 查看 [Utils API](utils.md) 中的控制器和旋转工具

### 按任务查找

| 任务 | 相关 API |
|------|----------|
| 创建机器人训练环境 | `OrcaGymEulerEnv` |
| 读取仿真状态 | `env.data` → `env.data.qpos` / `env.data.body_xpos(name)` |
| 设置求解器参数 | `env.sim_config` → `env.sim_config.timestep = 0.002` |
| 查询 body/site 状态 | `env.query_*()` / `env.get_body_*()` |
| 施加外力 | `env.apply_body_force(name, force, torque)` |
| 抓取/拖拽物体 | Mocap + 等式约束 |
| 相机图像获取 | `CameraWrapper` |
| 场景中放置物体 | `OrcaGymScene` |
| 逆运动学控制 | `InverseKinematicsController` |
| 关节力矩控制 | `JointController` |
| 旋转/姿态转换 | `rotations` |
| 录制视频 | `begin_save_video` / `stop_save_video` |

## 关键概念速查

| 概念 | 说明 | 详见 |
|------|------|------|
| **Body** | 刚体，物理仿真基本单元 | [Core API](core.md) |
| **Joint** | 关节，连接 body 的约束 | [Core API](core.md) |
| **Actuator** | 执行器，驱动机器人的元件 | [Core API](core.md) |
| **Geom** | 几何体，碰撞检测形状 | [Core API](core.md) |
| **Site** | 标记点，不参与物理仿真 | [Core API](core.md) |
| **Sensor** | 传感器，测量物理量 | [Sensor API](sensor.md) |
| **Mocap Body** | 可自由移动的虚拟 body | [Core API](core.md) |
| **Equality Constraint** | 等式约束，连接两个 body | [Core API](core.md) |
| **qpos/qvel/qacc** | 广义坐标/速度/加速度 | [Core API](core.md) |
| **Frame Skip** | 每次 step() 的物理步数 | [Environment API](environment.md) |

## API 使用规范

以下是使用 OrcaGym API 的推荐方式：

| ✅ 推荐 | ❌ 避免 |
|---------|--------|
| `env.data.qpos` | 直接访问 MuJoCo 内部数据结构 |
| `env.data.body_xpos("link1")` | 通过内部 ID 访问 body |
| `env.sim_config.timestep = 0.002` | 直接修改求解器原始参数 |
| `env.apply_body_force("link1", f, tau)` | 直接写入外力数组 |
| `env.do_simulation(ctrl, n_frames)` | 手动步进和同步 |
| `env.query_joint_qpos([...])` | 绕道访问内部数据结构 |
