# OrcaGym × Euler：CPU 四阶段耦合开发方案（简版）

> 日期：2026-08-11  
> 已拍板：**路径 A**（先打通 CPU 四阶段耦合）+ **场景 P**（Gym 直接 `PullEsdf`）  
> 模块划分见 Canvas：`orcagym-euler-integration-status.canvas.tsx`  
> 详设：[`orcagym_euler_integration.md`](orcagym_euler_integration.md)

---

## 1. 目标与边界

**第一里程碑目标：**  
Studio Play → Gym 拉 MJCF + PullEsdf → 填 `_euler` → 一个 SyncCycle 跑通四阶段（刚体步 → 状态同步 → Euler.step → 力回流）。

**不做（后续）：** GPU 方案 Z、`CouplingOrchestrator`、M:N 齿比、模式 B 零拷贝、Playground 当主控。

**角色：** Gym 主控；MuJoCo / Euler 并列独立引擎；跨引擎同步只在 `OrcaGymEuler`。

---

## 2. 架构一图

```
OrcaGymEulerEnv
  └─ OrcaGymEuler
       ├─ MuJoCoSimCore  ←── OrcaStudioBridge.LoadLocalEnv (MJCF)     [已通]
       ├─ Euler Runtime  ←── PullEsdf (50451) → add_esdf_from_string  [待通]
       └─ step_with_coupling
            ① MuJoCo step
            ② _sync_mujoco_to_euler_state   (xpos/xquat/cvel → body_q/qd)
            ③ Euler Solver.step
            ④ _sync_euler_force_to_mujoco   (body_f → xfrc_applied)
```

索引：`Model.body_label` ↔ mjcf `body name` → `body_index_map`。

---

## 3. 现状（相对 Canvas）

| 模块 | 状态 | 说明 |
|------|------|------|
| OrcaGymEulerEnv / MuJoCoSimCore / Bridge→MJCF | 已实现 | 纯刚体可跑 |
| step_with_coupling | 骨架 | `_euler is None` 时 = 纯 MuJoCo |
| `add_mjcf` / `Model.body_label` / `add_esdf`（Euler 仓） | **已实现** | 双注入必选；Gym 尚未接线。契约：`orcagym_euler_integration.md` §4.3.1 / §6.2 / §7.1.1 |
| `body_f` → `xfrc_applied` 约定 | **文档已钉** | 写回流前整表清零；basename map；`rigid_body_mode="external"` |
| Euler Runtime 持有、PullEsdf 注入、body_map、`_sync_*`、四阶段 | 待实现 | 本方案主体（唐鑫华） |
| 双引擎 render() | 待实现 | 可后置到耦合验收后 |
| CouplingOrchestrator / 方案 Z / 模式 B | 未来 | **不进本里程碑** |

---

## 4. 实现顺序（按依赖）

### M0 — 冻结与验收标准（0.5d）

- 文档口径：跨引擎编排在 Gym；等齿比 1:1；力走 `xfrc_applied`。
- 验收场景：Bear（或等价）Studio Play，至少 1 个可变形体 + 若干刚体。
- 通过标准：连续 N 个 SyncCycle 无崩；`body_f` 非零时能写入 `xfrc_applied`；可选看粒子/位姿不飞。

### M1 — 场景注入：PullEsdf → Euler Runtime

| 项 | 内容 |
|----|------|
| 做什么 | Env 初始化：现有 MJCF 路径不变；**必选** `add_mjcf`（刚体感知）；新增对 `50451` 的 `PullEsdf`；`add_esdf_from_string` + **一次** finalize；构造并挂上 `_euler` |
| 落点 | `orca_gym_euler_env.py`、`orca_gym_euler.py`；可薄封装 Pull 客户端（复用 Engine 测试 client 或正式 RenderClient） |
| 验收 | `has_euler()==True`；`particle_count` / body 数 > 0 |
| 负责 | Gym；Euler 仅修北向缺口（若有） |

### M2 — body_index_map

| 项 | 内容 |
|----|------|
| 做什么 | `_build_body_index_map`：`body_label` ↔ mj 名；预构建 `_mj_bodies_arr` / `_euler_idxs_arr` 与 `body_q/qd` 缓冲 |
| 依赖 | M1；刚体名与 ESDF/`add_mjcf` 感知侧一致（缺则补 `add_mjcf` 或约定 label） |
| 验收 | 映射表完整；无对应 body 打警告但不静默错绑 |
| 负责 | Gym |

### M3 — 双侧同步（模式 A）

| 项 | 内容 |
|----|------|
| 做什么 | `_sync_mujoco_to_euler_state`（含 wxyz→xyzw）；`_sync_euler_force_to_mujoco`（清零后写 `xfrc_applied`） |
| 验收 | 单测/脚本：同步后 `body_q` 与 `xpos` 一致；力通道 1:1 |
| 负责 | Gym |

### M4 — 四阶段 step_with_coupling

| 项 | 内容 |
|----|------|
| 做什么 | `_euler` 非空时走 ①②③④；仍无 Euler 时保持纯 MuJoCo |
| 验收 | Studio Play + Gym Env：耦合步进稳定；对比「关耦合」基线 |
| 负责 | Gym |

### M5 — 双引擎 render（可后置）

| 项 | 内容 |
|----|------|
| 做什么 | 刚体：现有 Bridge；非刚体：`euler.render()` / RenderClient |
| 验收 | 视口同时见刚体 + 非刚体 |
| 负责 | Gym 编排 + Euler/Engine 渲染通道 |

### M6 — 示例与回归

| 项 | 内容 |
|----|------|
| 做什么 | Playground `examples/euler/*` 挂正式 Gym API；冒烟脚本 |
| 说明 | Playground 只做示例，不承担主控实现 |

---

## 5. 分工（简）

| 方 | 范围 |
|----|------|
| OrcaGym | PullEsdf 接入、Runtime 持有、map、`_sync_*`、四阶段、Env API |
| OrcaEuler | 北向缺口（label / add_esdf / step）；不感知 Gym |
| OrcaEngine/Studio | `PullEsdf` 稳定、场景可导出 |
| OrcaPlayground | API 稳定后补示例 |

---

## 6. 风险（只列会挡路的）

1. **刚体名对不齐** → map 失败；需 ESDF/`add_mjcf` 与 mjcf `body name` 约定一致。  
2. **50451 与 Mujoco gRPC 两端口** → Env 初始化要同时连对。  
3. **Gym 文档 vs 方案 Z** → 本里程碑只做 CPU；GPU 另开文档，不混进本排期。

---

## 7. 相关路径

| 文档/代码 | 路径 |
|-----------|------|
| 本方案 | `OrcaGym/docs/zh/core_concepts/euler_cpu_coupling_dev_plan.md` |
| **GV任务计划** | `/home/hjadmin/OrcaApr24/OrcaEuler/Docs/Design/development/gymEuler_dev_plan.md` |
| 集成详设 | `OrcaGym/docs/zh/core_concepts/orcagym_euler_integration.md` |
| 开工不确定项 | `OrcaGym/docs/zh/core_concepts/euler_framework_status_and_open_items.md` |
| Gym 编排 | `OrcaGym/orca_gym/core/euler/orca_gym_euler.py` |
| Env | `OrcaGym/orca_gym/environment/euler/orca_gym_euler_env.py` |
