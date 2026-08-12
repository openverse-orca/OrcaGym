# Euler 连通：框架规定、开发方案与不确定项

> 用途：开工前对齐（黄总审阅 / 与鑫华讨论）  
> 日期：2026-08-11  
> 权威详设（未统一前勿单独当开工依据）：  
> - Gym：`orcagym_euler_integration.md`  
> - Euler：`OrcaEuler/Docs/Design/architecture/euler_in_orcagym_integration.md`  
> - GPU 后端：`OrcaEuler/.../solver_mujoco_backend_architecture.md`（方案 Z）

---

## 1. 当前框架规定（已定角色）

| 组件 | 角色 | 说明 |
|------|------|------|
| **OrcaGym** | **主控框架** | `OrcaGymEuler` / `OrcaGymEulerEnv`：Env 生命周期、步进编排、后端切换、与 Studio 刚体桥接 |
| **MuJoCo** | 独立刚体引擎（CPU） | `_mjModel` / `_mjData`，由 `MuJoCoSimCore` 持有 |
| **OrcaEuler** | 独立非刚体引擎 | `Model` / `State` / `Control` + `Solver.step()`；不感知 OrcaGym |
| **OrcaPlayground** | **示例 / 验收脚本** | `examples/euler/*` 挂在 Gym Env 上；**不是**主控框架，不靠「专用分支」当主控制脚本 |
| **OrcaStudio / OrcaEulerRender** | 场景源与渲染 | 双文件 mjcf+ESDF；刚体走 Gym Bridge；非刚体走 Euler `RenderClient` |

**北向数据契约（两边文档一致部分）：**

- Gym 写 Euler：`State.body_q` / `body_qd`（刚体输入）
- Euler 写、Gym 回流：`State.body_f` → MuJoCo `xfrc_applied`（不走 `qfrc_applied`）
- 力语义：COM 世界系 wrench，1:1
- 索引：靠 `body_label` 的 **basename** ↔ mjcf `body name`（Euler 已具备；Gym 建 map 待接）

**实现现状：** Gym 骨架已有；`_euler` 仍为 `None`，`step_with_coupling` ≈ 纯 MuJoCo。Euler 侧 `add_mjcf` + `add_esdf` 双注入与 `body_f` 契约文档已齐（见权威集成文档）。

---

## 2. 开发方案（文档中的目标形态）

### 2.1 主路径（Gym 编排，CPU）

一个同步周期（等齿比先行）四阶段：

1. MuJoCo 刚体解算  
2. Gym：`_mjData` → Euler `body_q` / `body_qd`（含四元数 wxyz→xyzw）  
3. Euler `Solver.step`  
4. Gym：`body_f` → `xfrc_applied`  

跨引擎同步在 **OrcaGymEuler**；Euler 内部多求解器由 **CouplingOrchestrator**（不对 MuJoCo）。

### 2.2 GPU 路径（方案 Z，**本里程碑不做**）

- 文档两边已标明：当前只做 CPU 四阶段；方案 Z 后续  
- 目标形态仍是：`SimConfig.use_gpu_backend`；Euler `SolverMujoco`；Gym Adapter；耦合可下沉到 Euler 内部

### 2.3 建议分工（与口头口径一致）

| 负责方 | 范围 |
|--------|------|
| 鑫华 / OrcaGym | Env、同步、后端协议、Adapter、`SimConfig` |
| OrcaEuler | 北向接口、`add_esdf`、求解器；（若做 GPU）`SolverMujoco` |
| OrcaPlayground | Gym API 稳定后补示例，不承担主控实现 |
| 开工前 | 统一设计文档交黄总审阅；不确定项再拉会 |

### 2.4 可变形「独立验收」线（可并行、不经 Gym）

Studio `PullEsdf` → Euler SemiImplicit → 生产 `RenderClient` → Studio。  
用于打通物理→渲染；**不等于**「Gym 连通引擎」已完成。

---

## 3. 曾不确定项（2026-08-12 回写）

| 项 | 状态 | 结论 |
|----|------|------|
| U1 文档口径 | **已对齐** | CPU 跨引擎在 Gym；方案 Z 后续；权威两篇已镜像 J2/J3 |
| U2 第一刀路径 | **已拍板** | A：CPU 四阶段（见 §4） |
| U3 里程碑边界 | **已拍板** | 交付 = Gym 填 `_euler` + 四阶段耦合；独立可变形线 / 双引擎 render 可后置 |
| U4 场景注入 | **已拍板** | Gym 双注入必选：`add_mjcf` + `PullEsdf`→`add_esdf(_from_string)`；Euler API 已齐，Gym 接线待做 |
| U5 齿比 | 仍等产品 | 本里程碑等齿比；M:N 后续 |
| U6 Playground | **已定** | 示例仓，不是主控 |

---

## 4. 已拍板（2026-08-11）

1. **主控 = OrcaGym；Playground = 示例。**  
2. **第一里程碑 = U2-A（CPU 四阶段）+ 场景 P（Gym 直接 PullEsdf）。**  
3. **简洁开工方案：** [`euler_cpu_coupling_dev_plan.md`](euler_cpu_coupling_dev_plan.md)（按 Canvas 模块依赖排序）。

GPU 方案 Z、CouplingOrchestrator、M:N 齿比不进本里程碑。

---

## 5. 相关路径

| 文档 | 路径 |
|------|------|
| 本文 | `/home/hjadmin/OrcaApr24/OrcaGym/docs/zh/core_concepts/euler_framework_status_and_open_items.md` |
| **CPU 耦合开工方案（简版）** | `/home/hjadmin/OrcaApr24/OrcaGym/docs/zh/core_concepts/euler_cpu_coupling_dev_plan.md` |
| Gym 集成 | `/home/hjadmin/OrcaApr24/OrcaGym/docs/zh/core_concepts/orcagym_euler_integration.md` |
| Euler 集成 | `/home/hjadmin/OrcaApr24/OrcaEuler/Docs/Design/architecture/euler_in_orcagym_integration.md` |
| 方案 Z | `/home/hjadmin/OrcaApr24/OrcaEuler/Docs/Design/architecture/solver_mujoco_backend_architecture.md` |
| 可变形链路 TODO | `/home/hjadmin/OrcaApr24/OrcaEngine2409/Gems/OrcaEulerRender/Docs/development/TODO_euler_deformable_full_pipeline.md` |
