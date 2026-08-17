# 🏗️ 架构总览

本文从分层视角描述 `OrcaGymEulerEnv` + `OrcaGymEuler` 系统的整体架构，厘清各层职责与 API 边界，帮助开发者判断「应在哪一层开发」以及「应维护哪一层」。

组件设计细节、API 契约、封装隔离机制见 [architecture.md](architecture.md)。

---

## 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│  用户代码 (User Code)                                            │
│  业务环境子类、任务定义、奖励函数、观测构造                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 继承 OrcaGymEulerEnv，使用 env.data / env.sim_config
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  RL 训练框架 (RSL-RL / SB3)                                      │
│  策略训练、rollout 调度、obs / action / reward 流转               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ env.reset() / env.step() / env.do_simulation()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  环境层：OrcaGymEulerEnv                                         │
│  gym.Env 实现、公共 API 契约、OrcaGymEnvMixin                    │
│  .data / .model / .sim_config / .apply_body_force() / .query_*()│
└───────────────────────────┬─────────────────────────────────────┘
                            │ 组合（非继承），委托到 _gym
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  仿真核心层：OrcaGymEuler (Facade)                               │
│  MuJoCoSimCore / ModelRegistry / SimConfig                      │
│  通过后端选择在两条互斥路径之间二选一                              │
└───────┬───────────────────────────────────────┬─────────────────┘
        │                                       │
        │  backend="mujoco"                     │  backend="euler"
        ▼                                       ▼
┌───────────────────────────────┐  ┌──────────────────────────────┐
│  MuJoCo 后端（CPU）            │  │  Euler 后端（GPU）            │
│  MjModel / MjData / mj_step   │  │  Euler 引擎（自治物理）       │
│  opt.* 求解器参数              │  │  对外提供 MuJoCo 风格 API     │
│  纯 MuJoCo，无编排无耦合       │  │  D2H 数据提取（qpos/xpos 等） │
└───────────────┬───────────────┘  └──────────────┬───────────────┘
                │                                 │
                │ sync_to_view()                  │ D2H 数据提取（qpos/xpos 等）
                ▼                                 ▼
        ┌──────────────────────────────────────────────┐
        │  OrcaGymDataView（统一状态视图）              │
        │  env.data 读取一致，屏蔽后端差异              │
        └──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  外部渲染器（可选，旁路系统）                                     │
│  消费 qpos / sim_time 快照，不参与物理步进主路径                  │
└─────────────────────────────────────────────────────────────────┘
```

| 层次 | 项目 | Python 包 | 职责 |
|------|------|----------|------|
| 用户代码 | 业务仓库 | — | 业务环境子类、奖励函数、观测构造 |
| RL 训练框架 | RSL-RL / SB3 | — | 策略训练、rollout 调度 |
| 环境层 | OrcaGym | `orca_gym` | gym.Env 实现、公共 API 契约、MuJoCo 语义接口 |
| 仿真核心层 | OrcaGym | `orca_gym` | 仿真核心 Facade + 后端选择与委托 |
| MuJoCo 后端 | MuJoCo | `mujoco` | CPU 刚体动力学求解（开源标准） |
| Euler 后端 | Euler | — | GPU 物理仿真（自治引擎） |
| 外部渲染器 | OrcaStudio / OrcaLab | — | 渲染、场景同步、交互（旁路系统，可选） |

### 双后端互斥选择

OrcaGym 作为仿真框架，接入两个**互不隶属**的物理后端，运行时二选一：

- **MuJoCo 后端**（开源标准）：纯 CPU MuJoCo 刚体仿真，不涉及编排与耦合。
- **Euler 后端**（Orca 团队自研）：Euler 作为完整物理引擎自治运行。OrcaGym 通过 Euler 提供的 MuJoCo 风格 API 驱动仿真，并通过 D2H 接口提取 `qpos`/`xpos` 等数据到 CPU 供渲染使用。

两条路径互斥：加载一个后端时不涉及另一个。后端选择对用户代码透明，`env.data` / `env.do_simulation()` 等公共 API 行为一致。

> **当前实现状态**：`_euler` 字段为占位（恒为 `None`），`has_euler()` 恒返回 `False`，当前仅 MuJoCo 后端可用。Euler 后端的接入将在后续版本实现。

> **外部渲染器是旁路系统**：仅消费 `qpos`/`sim_time` 快照，不参与物理步进主路径。渲染器缺席时环境仍可正常 step。

---

## API 层次与角色界定

### 用户开发层

用户代码仅与以下两层 API 交互，**不得穿透到下层内部对象**（`_mjModel`/`_mjData`/`_sim` 等）：

| API | 来源 | 用途 |
|-----|------|------|
| `env.data` | `OrcaGymDataView` | 读取 `qpos`/`qvel`/`body_xpos(name)` 等状态 |
| `env.model` | `OrcaGymModel` | 查询模型结构（维度、名称映射） |
| `env.sim_config` | `SimConfig` | 配置 timestep / integrator / iterations / gravity |
| `env.ctrl` | `np.ndarray` | 设置控制输入 |
| `env.do_simulation(ctrl, n)` | — | 仿真步进 |
| `env.set_joint_qpos()` / `env.apply_body_force()` / `env.clear_body_force()` | — | 状态写入、外力注入 |
| `env.body()` / `env.joint()` / `env.actuator()` / `env.site()` | `OrcaGymEnvMixin` | 名称空间解析（自动添加 agent 前缀） |
| `env.render()` | — | 外部渲染器交互 |
| `gym.Env` 标准接口 | Gymnasium | `reset()` / `step()` / `observation_space` / `action_space` |

**用户开发范式**：

```python
class MyTaskEnv(OrcaGymEulerEnv):
    def _get_obs(self) -> dict:
        # ✅ 走公共 API
        return {
            "qpos": self.data.qpos.copy(),
            "body_pos": self.data.body_xpos("link1"),
        }

    def compute_reward(self) -> float:
        # ✅ 走公共 API
        return float(self.data.body_xpos("target")[2])

    def _apply_disturbance(self):
        # ✅ 走公共 API
        self.apply_body_force("link1", force=[0, 0, 10], torque=[0, 0, 0])
```

### 开发者维护层

| 层次 | 维护方 | 维护内容 |
|------|--------|---------|
| **环境层** `OrcaGymEulerEnv` | OrcaGym 团队 | gym.Env 实现、公共 API 契约、Mixin 公共方法 |
| **仿真核心层** `OrcaGymEuler` 及子组件 | OrcaGym 团队 | Facade 委托、`MuJoCoSimCore` / `ModelRegistry` / `SimConfig`、后端选择 |
| **MuJoCo 后端** MuJoCo | 上游 | `mujoco` 库 |
| **Euler 后端** Euler | Euler 团队 | 自治物理引擎，对外提供 MuJoCo 风格 API |
| **外部渲染器** OrcaStudio / OrcaLab | 各自团队 | 渲染器、交互逻辑 |

**开发者扩展原则**：当公共 API 不满足用户需求时，在 `OrcaGymEulerEnv` 增加公共方法（委托到 `_gym` 公共 API），或在 `OrcaGymDataView` 增加字段访问器，**不得引导用户穿墙访问内部对象**。

---

## 关键调用流

### step 主路径

```
用户代码 / RL 框架
    │ env.step(action)  或  env.do_simulation(ctrl, n_frames)
    ▼
OrcaGymEulerEnv
    │ 委托 _gym.do_simulation()
    ▼
OrcaGymEuler
    │ 按所选后端委托步进
    ▼
┌─────────────────────────────────────────────────────┐
│  MuJoCo 后端                │  Euler 后端            │
│  _sim.set_ctrl()           │  Euler 驱动步进         │
│  _sim.step(nstep)          │  （内部自治完成物理）    │
│  mj_step × nstep           │                        │
└─────────────────────────────────────────────────────┘
    │
    │ 状态同步到视图（MuJoCo: sync_to_view / Euler: D2H 提取）
    ▼
OrcaGymDataView  ←── env.data 读取一致
```

### 渲染旁路

```
用户代码
    │ env.render()
    ▼
OrcaGymEulerEnv
    │ 委托渲染器消费状态快照
    ▼
外部渲染器（独立进程/机器）
    │ 场景同步、渲染、视频帧捕获
```

渲染路径与物理步进路径**完全解耦**：渲染器仅消费 `qpos`/`sim_time` 快照，不触碰物理步进。

### 状态写入与外力注入

```
用户代码
    │ env.apply_body_force(name, force, torque)
    ▼
OrcaGymEulerEnv
    │ 委托 _gym.apply_body_force()
    ▼
OrcaGymEuler
    │ 按所选后端写入外力（具体机制由后端决定）
    ▼
后端内部（对用户不可见）
```

外力注入是**显式且可追踪的**：通过公共 API 注入，后端内部自行处理力的应用机制。

---

## 封装边界

```
用户可见                    │  用户不可见（内部）
─────────────────────────────┼──────────────────────────────────
env.data (DataView)         │  env._gym
env.model (OrcaGymModel)    │  env._gym._sim
env.sim_config (SimConfig)  │  env._gym._sim._mjModel / _mjData
env.ctrl                    │  env._gym._studio
env.do_simulation()         │  env._gym._registry
env.apply_body_force()      │  env._gym._euler
env.query_*()               │  env._gym._opt
env.body() / joint() / ...  │
```

- **左列**：公共 API（L1），用户和 AI 应使用，IDE 自动补全可见
- **右列**：内部组件（L2/L3），`_` 前缀约定 + ruff SLF001 静态检查 + AGENTS.md 约束，禁止外部访问

详细契约与隔离机制见 [architecture.md](architecture.md) §6–§7。
