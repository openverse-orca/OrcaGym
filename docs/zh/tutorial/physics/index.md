# ⚛️ 物理仿真

OrcaGym 的物理仿真基于 MuJoCo 引擎，提供了高精度的刚体动力学和约束求解。

> 完整可运行代码见 [OrcaPlayground examples/euler/](https://github.com/OrcaGym/OrcaPlayground)。

## 章节导航

- [🔧 MuJoCo 后端](mujoco-backend.md) — 模型加载、求解器配置、步进控制
- [📐 状态管理](state-management.md) — qpos/qvel/qacc、状态设置与读取
- [💥 接触与力](contacts-forces.md) — 接触检测、力查询、外力注入
- [🔄 外力应用与 IK](force-apply.md) — 外力施加、状态写入、雅可比、逆运动学
- [🔗 等式约束](equality-constraints.md) — WELD/CONNECT 约束、mocap 锚定、抓取操作
- [🧤 软体与柔性体](soft-bodies.md) — MuJoCo Flex 系统（实验性）

## 快速参考

| 操作 | API | 说明 |
|------|-----|------|
| 步进+同步（推荐） | `env.do_simulation(ctrl, n)` | 原子操作，自动同步 data |
| 推进 n 步物理 | `env.mj_step(n)` | 执行 n 次物理步进 |
| 前向更新 | `env.mj_forward()` | 刷新派生量（body 位姿、传感器） |
| 同步视图 | `env._sync_view()` | 同步到 DataView |
| 雅可比矩阵 | `env.mj_jacBody(jacp, jacr, name)` | 位置/旋转雅可比 |
| Site 雅可比 | `env.mj_jacSite(jacp, jacr, name)` | Site 点雅可比 |
| 施加外力 | `env.apply_body_force(name, f, τ)` | 对 body 施加力/力矩 |
| 清除外力 | `env.clear_body_force(name)` | 清除指定 body 的外力 |
| 清除全部力 | `env.clear_all_forces()` | 清除所有外力 |
| 设置摩擦 | `env.set_geom_friction({name: arr})` | 设置 geom 摩擦系数 |
| 设置 mocap | `env.set_mocap_pos_and_quat({name: {...}})` | 设置 mocap body 位姿 |
| 设置求解器 | `env.sim_config.timestep = 0.002` | 时间步长配置 |

## 物理引擎关键参数

通过 `env.sim_config` 配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `timestep` | 0.001 (G1) / 0.002 | 物理仿真步长（秒） |
| `iterations` | 100 | 求解器迭代次数 |
| `integrator` | 0 (Euler) | 积分器类型（0=Euler, 1=RK4） |
| `gravity` | `[0, 0, -9.81]` | 重力向量 |
| `tolerance` | 1e-8 | 求解器容忍度 |

## G1 人形机器人标准配置

G1 Euler 示例（Lesson 4-9）使用以下标准参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `time_step` | 0.001s | 物理步长 1ms（MuJoCo 1000Hz） |
| `frame_skip` | 20 | 每控制周期 20 物理步 |
| `dt` | 0.02s | 控制频率 50Hz |
| `integrator` | 0 (Euler) | 半隐式 Euler 积分 |
