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
│  MuJoCoSimCore / OrcaStudioBridge / ModelRegistry /             │
│  SimConfig / EulerOrchestrator                                  │
└──────┬─────────────────────────────────────────────┬────────────┘
       │                                             │
       │ mj_step / mj_forward                        │ gRPC 通信
       ▼                                             ▼
┌─────────────────────────────────┐  ┌─────────────────────────────┐
│  MuJoCo Runtime (刚体求解器)     │  │  OrcaStudio 系统             │
│  MjModel / MjData / mj_step     │  │  渲染、场景同步、视频保存      │
│  opt.* 求解器参数                │  │  物体操纵、相机控制            │
└─────────────────────────────────┘  └─────────────────────────────┘
       │
       │ 外力耦合 / 同步周期 (SyncCycleConfig)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  引擎层：Euler Runtime (orca.euler)                              │
│  多物理场仿真、Model / State / Control、求解器调度、零拷贝耦合     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ import orca.flow
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  框架层：OrcaFlow (orca.flow)                                    │
│  GPU 编程框架、多后端编译、flow.kernel / flow.array               │
└─────────────────────────────────────────────────────────────────┘
```

| 层次 | 项目 | Python 包 | 职责 |
|------|------|----------|------|
| 用户代码 | 业务仓库 | — | 业务环境子类、奖励函数、观测构造 |
| RL 训练框架 | RSL-RL / SB3 | — | 策略训练、rollout 调度 |
| 环境层 | OrcaGym | `orca_gym` | gym.Env 实现、公共 API 契约、MuJoCo 语义接口 |
| 仿真核心层 | OrcaGym | `orca_gym` | 仿真核心 Facade + 子组件编排 |
| 刚体运行时 | MuJoCo | `mujoco` | 刚体动力学求解 |
| 非刚体运行时 | Euler | `orca.euler` | 多物理场仿真、求解器调度、零拷贝耦合 |
| Studio 系统 | OrcaStudio | — | 渲染、场景同步、交互（旁路系统） |
| 框架层 | OrcaFlow | `orca.flow` | GPU 编程框架、多后端编译 |

> **OrcaStudio 是旁路系统**：通过 gRPC 与仿真核心通信，不参与 `mj_step` 主路径，不影响物理仿真。Studio 缺席时环境仍可正常 step。

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
| `env.render()` / `env.begin_save_video()` | — | Studio 交互 |
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
| **仿真核心层** `OrcaGymEuler` 及子组件 | OrcaGym 团队 | Facade 编排、`MuJoCoSimCore` / `OrcaStudioBridge` / `ModelRegistry` / `SimConfig` / `EulerOrchestrator` |
| **刚体运行时** MuJoCo | 上游 | `mujoco` 库 |
| **非刚体运行时** Euler | Euler 团队 | Model/State/Control、求解器、耦合编排 |
| **框架层** OrcaFlow | Flow 团队 | GPU kernel 编译、多后端调度 |
| **Studio 系统** OrcaStudio | Studio 团队 | 渲染器、gRPC 服务、交互逻辑 |

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
    │ _sim.set_ctrl() → _sim.step(nstep)
    ▼
MuJoCoSimCore
    │ mj_step × nstep
    ▼
MuJoCo Runtime  ←── (EulerOrchestrator 启用时) ── Euler Runtime / OrcaFlow
    │
    │ _sim.sync_to_view()
    ▼
OrcaGymDataView  ←── env.data 读取一致
```

### 渲染旁路

```
用户代码
    │ env.render()
    ▼
OrcaGymEulerEnv
    │ 委托 _studio.render(qpos, sim_time)
    ▼
OrcaStudioBridge  ──gRPC──►  OrcaStudio 系统（独立进程/机器）
                                    │ 场景同步、渲染、视频帧捕获
```

渲染路径与物理步进路径**完全解耦**：Studio 仅消费 `qpos`/`sim_time` 快照，不触碰 `mj_step`。

### 状态写入与外力注入

```
用户代码
    │ env.apply_body_force(name, force, torque)
    ▼
OrcaGymEulerEnv
    │ 委托 _gym.apply_body_force()
    ▼
OrcaGymEuler
    │ _sim.apply_body_force(body_id, force, torque)
    │ (可选) _euler.notify_external_force(...)
    ▼
MuJoCoSimCore
    │ 写入 xfrc_applied（内部细节，对用户不可见）
```

外力注入是**显式且可追踪的**：`EulerOrchestrator` 启用时可感知外力注入，保证 MuJoCo 与 Euler 耦合一致性。

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
