# OrcaGym Euler 阶段四目录重组与 Lesson 4–9 重新验收方案

## 1. 文档定位

本文是 `OrcaGymEulerEnv` 阶段四的**目录重组与重新验收**指导文档，承接
`docs/design/development/orca_gym_euler_phase4_online_validation_development.md`（下称"原 phase4 文档"）。

### 1.1 背景与动机

原 phase4 文档已指导完成 Lesson 4–7 的实施，目录布局为：

```
OrcaPlayground/
├── envs/euler/                 ← 共享代码层（Lesson 4–8 Env 子类、辅助工具）
│   ├── robots/                 ← G1 资产（mesh/ONNX/config/XML）
│   ├── scenes/simple_pendulum.xml
│   ├── g1_base_env.py
│   ├── g1_locomotion.py
│   ├── online_verifier.py
│   ├── simple_env.py
│   ├── query_api_env.py
│   ├── force_apply_env.py
│   ├── jacobian_env.py
│   └── studio_capture_env.py
└── examples/euler/0X_xxx/      ← 脚本入口 + 教程文档
```

该布局存在三个问题：

1. **example 不独立**：每个 example 通过 `from envs.euler.xxx import ...` 依赖 `envs/euler/`
   共享目录，无法单独拷出运行。
2. **资产与代码混杂**：G1 mesh/ONNX/config 埋在 `envs/euler/robots/` 下，与 Env 代码同级，
   不便于资产单独管理与版本化。
3. **Lesson 7 职责耦合**：原 Lesson 7（studio_capture）把"行走控制验证"和"视频采集验证"
   耦合在一起，导致行走问题（瘫倒/乱踹）与视频采集问题难以独立定位。

### 1.2 重组目标

1. **删除 `envs/euler/` 目录**，消除共享代码层。
2. **每个 example 完全自包含**：目录内含全部所需 `.py`，仅依赖已安装的 `orca_gym` 包
   （Euler 体系）+ 同目录文件 + `assets/`。
3. **资产集中到 `OrcaPlayground/assets/`**，与代码分离。
4. **零 `envs.*` 项目内 import**：`grep -r "from envs\." examples/euler/` 零命中。
5. **不扩大范围**：仅处理 `envs/euler` 与 `examples/euler` 两个目录；`envs/common`、
   `envs/g1`、`envs/fluid` 等原有目录不动（但其功能被提取后 example 不再依赖它们）。
6. **Euler 体系纯净**：example 仅依赖 `orca_gym.environment.euler.*`，不引入老 `OrcaGymLocalEnv`
   体系（Local 老主路径将被 Euler 替代，禁止 Euler 教程反向依赖）。
7. **Lesson 7 拆分**：原 Lesson 7（studio_capture）拆为 Lesson 7（locomotion，专注行走控制）
   + Lesson 8（video_capture，专注视频采集），原 Lesson 8（body_manipulation）顺延为 Lesson 9。

### 1.3 适用范围

- **重新验收**：Lesson 4–6（已实施并迁移到新结构，已通过验收）。
- **课程拆分**：原 Lesson 7（studio_capture）拆分为 Lesson 7（locomotion）+ Lesson 8（video_capture），
  需重新实施与验收。
- **新设计**：Lesson 9（原 Lesson 8 body_manipulation 顺延，未实施，按新结构从零设计）。

> **拆分动机**：原 Lesson 7 把行走控制与视频采集耦合，行走链路（ONNX 推理 → PD 控制 →
> motor 力矩）出问题时（瘫倒/乱踹），难以在视频采集的噪声中独立定位。拆分后：
> - Lesson 7 locomotion 先把行走控制链路跑通（纯行走，无视频），便于聚焦定位行走问题。
> - Lesson 8 video_capture 在行走已验证的基础上，再叠加视频/帧/时间戳采集 API 验证。

---

## 2. 目录重组方案

### 2.1 目标目录结构

```
OrcaPlayground/
├── assets/                                   ← 资产集中（§2.2）
│   ├── g1/
│   │   ├── g1_29dof_camera.xml
│   │   ├── config/g1_29dof_hist.yaml
│   │   ├── meshes/*.STL
│   │   └── models/
│   │       ├── dec_loco/model_6600.onnx
│   │       └── mimic/.../*.onnx
│   └── scenes/
│       └── simple_pendulum.xml
├── examples/euler/
│   ├── 00_setup.md
│   ├── TUTORIAL.md
│   ├── requirements.txt
│   ├── 01_hello_euler/
│   │   ├── hello_euler.py
│   │   └── simple_env.py                     ← 复制（仅依赖 OrcaGymEulerEnv + assets/scenes）
│   ├── 02_online_render/
│   │   ├── online_render.py
│   │   └── simple_env.py
│   ├── 03_rl_ppo/
│   │   ├── train_ppo.py
│   │   └── simple_env.py
│   ├── 04_query_api/
│   │   ├── 04_query_api.md
│   │   ├── query_api.py
│   │   ├── g1_base_env.py                    ← 复制（资产路径改指向 assets/g1）
│   │   ├── scene_scanner.py                  ← 提取自 envs/common/model_scanner.py（§2.3）
│   │   ├── online_verifier.py                ← 复制
│   │   └── query_api_env.py                  ← 复制
│   ├── 05_force_apply/
│   │   ├── 05_force_apply.md
│   │   ├── force_apply.py
│   │   ├── g1_base_env.py
│   │   ├── scene_scanner.py
│   │   ├── online_verifier.py
│   │   └── force_apply_env.py
│   ├── 06_jacobian/
│   │   ├── 06_jacobian.md
│   │   ├── jacobian_ik.py
│   │   ├── g1_base_env.py
│   │   ├── scene_scanner.py
│   │   ├── online_verifier.py
│   │   └── jacobian_env.py
│   ├── 07_locomotion/                        ← 拆分自原 studio_capture：专注行走控制
│   │   ├── 07_locomotion.md
│   │   ├── locomotion.py
│   │   ├── g1_base_env.py
│   │   ├── scene_scanner.py
│   │   ├── g1_locomotion.py                  ← 含内联 HistoryHandler + PD 控制器（§2.4）
│   │   ├── online_verifier.py
│   │   └── locomotion_env.py
│   ├── 08_video_capture/                     ← 拆分自原 studio_capture：专注视频采集
│   │   ├── 08_video_capture.md
│   │   ├── video_capture.py
│   │   ├── g1_base_env.py
│   │   ├── scene_scanner.py
│   │   ├── g1_locomotion.py                  ← 复用 07 的行走控制
│   │   ├── online_verifier.py
│   │   └── video_capture_env.py
│   └── 09_body_manipulation/                 ← 原 Lesson 8 顺延（§4）
│       ├── 09_body_manipulation.md
│       ├── body_manipulation.py
│       ├── g1_base_env.py
│       ├── scene_scanner.py
│       ├── g1_locomotion.py
│       ├── online_verifier.py
│       └── body_manipulation_env.py
└── envs/                                     ← euler/ 子目录删除；common/、g1/、fluid/ 原样保留
```

### 2.2 资产目录 `assets/` 组织

| 原路径 | 新路径 |
|--------|--------|
| `envs/euler/robots/g1_29dof_camera.xml` | `assets/g1/g1_29dof_camera.xml` |
| `envs/euler/robots/config/g1_29dof_hist.yaml` | `assets/g1/config/g1_29dof_hist.yaml` |
| `envs/euler/robots/meshes/*.STL` | `assets/g1/meshes/*.STL` |
| `envs/euler/robots/models/dec_loco/model_6600.onnx` | `assets/g1/models/dec_loco/model_6600.onnx` |
| `envs/euler/robots/models/mimic/**/*.onnx` | `assets/g1/models/mimic/**/*.onnx` |
| `envs/euler/scenes/simple_pendulum.xml` | `assets/scenes/simple_pendulum.xml` |

> **mesh 路径**：`g1_29dof_camera.xml` 内 mesh 引用为相对路径 `meshes/xxx.STL`，MuJoCo
> 加载时以 XML 所在目录为基准，迁移后 XML 仍在 `assets/g1/`，mesh 在 `assets/g1/meshes/`，
> 相对路径不变，**无需改 XML**。

### 2.3 `scene_scanner.py` 提取（替代 `envs/common/model_scanner.py`）

`g1_base_env.py` 原依赖 `envs.common.model_scanner`，其传递依赖链
`model_scanner → envs.fluid.sim_env.SimEnv → envs/fluid/trajectory/*` 是项目内深依赖，
无法简单复制。经分析，example 仅用 `model_scanner` 的 3 个函数：

| 用量 | 作用 | 依赖 |
|------|------|------|
| `build_suffix_template` | 构造后缀模板（纯数据） | 无 |
| `scan_scene_for_template` | 扫描场景实例 | → `probe_scene_model` → 探针 env |
| `require_complete_matches` | 校验匹配数量 | 纯逻辑 |

**关键决策：探针改用 `OrcaGymEulerEnv`（Euler 新主路径），禁用 `SimEnv`/`OrcaGymLocalEnv`
（Local 老主路径，将被 Euler 替代）。**

`probe_scene_model` 仅调用探针 env 的 `model.get_body_names()` / `get_joint_dict()` /
`get_actuator_dict()` / `get_site_dict()` / `_sensor_dict` / `close()`，这些方法在
`OrcaGymEulerEnv.model`（委托 `orca_gym` 包的 `OrcaGymModel`）上完全可用，与探针基类无关。

**提取后 `scene_scanner.py`（约 150 行）保留**：

- 数据类：`SuffixTemplate` / `InstanceMatch` / `SceneScanReport` / `SceneModelNames`
- 函数：`build_suffix_template` / `scan_scene_for_template` / `probe_scene_model`（探针改
  `OrcaGymEulerEnv`）/ `match_robot_instances` + 三个内部辅助 / `require_complete_matches`
  （简化为纯数量校验 + raise）

**移除（example 不需要）**：`AssetUiHint` / `ASSET_UI_HINTS` / `log_scene_scan_report` /
`_build_ui_hint_message` / `_emit_terminal_hint` / `ordered_match_names`。

**依赖**：`orca_gym.environment.euler.orca_gym_euler_env`（Euler）+ `numpy` + stdlib。
**零 `envs.*` 项目内 import**。

### 2.4 `HistoryHandler` 内联 + PD 控制器加入 `g1_locomotion.py`

`g1_locomotion.py` 原依赖 `envs.g1.utils.history_handler.HistoryHandler`（43 行，仅依赖
`numpy` + `orca_gym.log` 已安装包）。因仅 `g1_locomotion` 一处使用且代码极短，**直接内联**
到 `g1_locomotion.py` 中作为模块级类，消除 `envs.g1` 依赖。

**PD 控制器**：G1 执行器是 `motor`（力矩控制，`ctrlrange` 为 N·m），而 ONNX 策略输出的是
**位置目标** `q_target`（弧度量级）。需在 `g1_locomotion.py` 的 `compute_action` 末尾加 PD
转换，返回力矩而非位置目标（与 `envs/g1/g1_env.py` 的 PD 实现一致）：

```python
# tau = Kp*(q_target - q) + Kd*(0 - qd)
tau = self.joint_kp * (q_target - dof_pos[0]) - self.joint_kd * dof_vel[0]
tau = np.clip(tau, -self.motor_effort_limit, self.motor_effort_limit)
return tau
```

PD 参数（`JOINT_KP`/`JOINT_KD`/`motor_effort_limit_list`）从 `g1_29dof_hist.yaml` 加载。
此修复使 `compute_action` 返回力矩（量级 ~50 N·m），传给 motor 执行器足以支撑站立。

> ** Lesson 7 拆分背景**：原 studio_capture 把行走控制与视频采集耦合，行走链路出问题时
> （如缺 PD 控制器导致瘫倒/乱踹）难以独立定位。拆分后 Lesson 7 locomotion 专注行走控制链路，
> Lesson 8 video_capture 在行走已验证基础上再叠加视频采集。

### 2.5 每个 example 的依赖清单（重组后）

| example | 同目录 .py | 依赖的 `orca_gym` 模块 |
|---------|-----------|----------------------|
| 01/02/03 | `simple_env.py` | `orca_gym.environment.euler.orca_gym_euler_env` |
| 04 | `g1_base_env` + `scene_scanner` + `online_verifier` + `query_api_env` | 同上 |
| 05 | `g1_base_env` + `scene_scanner` + `online_verifier` + `force_apply_env` | 同上 |
| 06 | `g1_base_env` + `scene_scanner` + `online_verifier` + `jacobian_env` | 同上 |
| 07 | `g1_base_env` + `scene_scanner` + `g1_locomotion` + `online_verifier` + `locomotion_env` | 同上 |
| 08 | `g1_base_env` + `scene_scanner` + `g1_locomotion` + `online_verifier` + `video_capture_env` | 同上 |
| 09 | `g1_base_env` + `scene_scanner` + `g1_locomotion` + `online_verifier` + `body_manipulation_env` | 同上 |

**`OrcaGymEulerEnv` 传递依赖全在 `orca_gym.core.euler.*` + `orca_gym.utils/protos/log`
（通用基础设施），零 Local 体系引用。**

### 2.6 迁移步骤

#### 步骤 1：建 `assets/` 并迁移资产
- `mkdir -p assets/g1/{config,meshes,models/dec_loco,models/mimic} assets/scenes`
- 按 §2.2 表移动文件，保持子结构。
- 删除 `envs/euler/robots/requirements.txt`（合并到 `examples/euler/requirements.txt`）。

#### 步骤 2：提取 `scene_scanner.py`
- 从 `envs/common/model_scanner.py` 提取（§2.3），探针改 `OrcaGymEulerEnv`。
- 放入 04/05/06/07/08/09 各一份。

#### 步骤 3：内联 `HistoryHandler` + 加入 PD 控制器
- 将 `envs/g1/utils/history_handler.py` 的 `HistoryHandler` 类内联到 `g1_locomotion.py`。
- 在 `compute_action` 末尾加 PD 转换（§2.4），返回力矩。
- 放入 07/08/09 各一份（这三个 lesson 用 ONNX 行走）。

#### 步骤 4：复制 .py 到各 example
- 按 §2.5 表，把 `envs/euler/*.py` 复制到各 example 目录。
- 改写 import 为同目录直接 import（见下表）。

| 原写法 | 新写法（example 内） |
|--------|---------------------|
| `from envs.euler.g1_base_env import ...` | `from g1_base_env import ...` |
| `from envs.euler.online_verifier import ...` | `from online_verifier import ...` |
| `from envs.euler.query_api_env import ...` | `from query_api_env import ...` |
| `from envs.euler.simple_env import ...` | `from simple_env import ...` |
| `from envs.euler.g1_locomotion import ...` | `from g1_locomotion import ...` |
| `from envs.euler.studio_capture_env import ...` | 拆分：`from locomotion_env import ...` / `from video_capture_env import ...` |
| `from envs.euler.force_apply_env import ...` | `from force_apply_env import ...` |
| `from envs.euler.jacobian_env import ...` | `from jacobian_env import ...` |
| `from envs.common.model_scanner import ...` | `from scene_scanner import ...` |
| `from envs.g1.utils.history_handler import HistoryHandler` | 删除（已内联） |

#### 步骤 5：改写资产路径
每个 example 内的 `g1_base_env.py` 顶部：

```python
# example 在 examples/euler/0X_xxx/，__file__ 上溯 4 层到项目根
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")
_ROBOTS_DIR = os.path.join(_ASSETS_DIR, "g1")
G1_MODEL_XML = os.path.join(_ROBOTS_DIR, "g1_29dof_camera.xml")
G1_CONFIG_YAML = os.path.join(_ROBOTS_DIR, "config", "g1_29dof_hist.yaml")
G1_LOCO_ONNX = os.path.join(_ROBOTS_DIR, "models", "dec_loco", "model_6600.onnx")
```

每个 `simple_env.py` 顶部：

```python
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SCENE_XML = os.path.join(_PROJECT_ROOT, "assets", "scenes", "simple_pendulum.xml")
```

#### 步骤 6：清理各 example 脚本
- 删除 `sys.path.insert(PROJECT_ROOT)` 与 `from envs.euler...` 行。
- 改为同目录直接 import。

#### 步骤 7：删除 `envs/euler/` 整个目录

#### 步骤 8：原 Lesson 7 拆分为 Lesson 7 + Lesson 8
- 将原 `07_studio_capture/` 拆为 `07_locomotion/`（行走控制）+ `08_video_capture/`（视频采集）。
- `07_locomotion/`：从原 `studio_capture_env.py` 提取行走控制逻辑为 `locomotion_env.py`，
  移除视频采集（begin/stop_save_video、get_frame_png、get_camera_time_stamp），新增行走
  稳定性数值判定（基座高度、未摔倒、关节力矩范围）。
- `08_video_capture/`：保留原视频采集逻辑为 `video_capture_env.py`，复用 07 的 `g1_locomotion.py`
  驱动行走，在行走过程中录制视频/截帧/查询时间戳。
- 原 `08_body_manipulation/` 规划顺延为 `09_body_manipulation/`。

---

## 3. Lesson 4–8 重新验收方案

### 3.1 通用验收流程（每个 Lesson）

| 步骤 | 执行方 | 动作 |
|------|--------|------|
| 1 | 人工 | 启动 OrcaStudio，加载含 1 个 G1（或 pendulum，仅 01/02/03）的关卡，点击运行 |
| 2 | 人工 | `cd OrcaPlayground && python examples/euler/0X_xxx/xxx.py` |
| 3 | 自动 | 脚本驱动 env 步进，输出数值判定 + 人工观察提示 |
| 4 | 人工 | 按 `[OBSERVE]` 提示在 Studio 视口确认 |
| 5 | 自动 | 脚本输出 JSON 报告，退出码 0/1 |

### 3.2 通用验收通过条件（每个 Lesson 同时满足）

1. **脚本可运行**：`python xxx.py` 连接 Studio 成功，无 `ModuleNotFoundError`。
2. **独立性**：`grep -r "from envs\." examples/euler/0X_xxx/` 零命中。
3. **资产路径**：`python -c "from g1_base_env import G1_MODEL_XML; import os; print(os.path.exists(G1_MODEL_XML))"` 返回 `True`（仅 G1 lesson）。
4. **ruff SLF001**：`python -m ruff check --select SLF001 examples/euler/0X_xxx/` 零报警。
5. **数值判定**：JSON 报告 `all_passed == true`。
6. **人工观察**：视口画面符合教程预期。
7. **教程文档**：`.md` 文档路径、命令、目录结构已更新为新结构。

### 3.3 Lesson 4 重新验收（04_query_api）

**验证内容**：G1 状态查询 API（关节/body/site/IMU/接触）。

**执行命令**：
```bash
cd OrcaPlayground && conda activate orca
python examples/euler/04_query_api/query_api.py
```

**目录自查**：
- `examples/euler/04_query_api/` 含 `query_api.py` / `g1_base_env.py` / `scene_scanner.py` /
  `online_verifier.py` / `query_api_env.py` / `04_query_api.md`。
- `grep -r "from envs\." examples/euler/04_query_api/` 零命中。

**教程文档更新点（`04_query_api.md`）**：
- 删除"依赖 `envs/euler/` 共享目录"相关说明。
- 运行命令改为 `cd OrcaPlayground && python examples/euler/04_query_api/query_api.py`（无需额外
  `PYTHONPATH`，因同目录直接 import）。
- 新增"目录结构"小节，说明本目录自包含。
- 资产路径说明改为 `assets/g1/g1_29dof_camera.xml`。

**数值判定项**（与原 phase4 §4.3.1 一致，9 项）：query_joint_qpos/qvel/qacc、
get_body_xpos_xmat_xquat、query_site_pos_and_mat、query_sensor_data、query_actuator_torques、
query_contact_simple、query_position_body_B、body_subtree_mass。

**验收通过条件**：§3.2 全部满足 + 9 项数值判定 PASS。

**状态**：✅ 已通过验收。

### 3.4 Lesson 5 重新验收（05_force_apply）

**验证内容**：外力应用 + 摩擦修改 + 接触力查询 + mocap 位姿写入（XML 内置 anchor↔box weld）。

**执行命令**：
```bash
python examples/euler/05_force_apply/force_apply.py
```

**教程文档更新点（`05_force_apply.md`）**：同 §3.3，路径/命令/目录结构更新。

**数值判定项**（与原 phase4 §4.3.2 一致）：apply_body_force 抬起、clear_body_force、
clear_all_forces、set_geom_friction、query_contact_force、pelvis 侧向推力、
set_mocap_pos_and_quat 驱动 box（atol=0.05）。

**验收通过条件**：§3.2 全部满足 + 数值判定 PASS。

**状态**：✅ 已通过验收。

### 3.5 Lesson 6 重新验收（06_jacobian）

**验证内容**：雅可比计算 + 阻尼最小二乘 IK。

**执行命令**：
```bash
python examples/euler/06_jacobian/jacobian_ik.py
```

**教程文档更新点（`06_jacobian.md`）**：同 §3.3。

**数值判定项**（与原 phase4 §4.3.3 一致）：pelvis 雅可比形状、imu site 速度一致性、
IK 迭代收敛（atol=0.02）。

**验收通过条件**：§3.2 全部满足 + 数值判定 PASS。

**状态**：✅ 已通过验收。

### 3.6 Lesson 7 重新验收（07_locomotion）

> **拆分背景**：原 Lesson 7（studio_capture）把行走控制与视频采集耦合。拆分后本课专注
> 行走控制链路验证，不含视频采集，便于聚焦定位行走问题（瘫倒/乱踹等）。

**验证内容**：G1 ONNX 行走控制链路（ONNX 推理 + PD 控制器 + 行走稳定性）。

**执行命令**：
```bash
python examples/euler/07_locomotion/locomotion.py
```

**目录自查**（07 含 `g1_locomotion.py`，且其内已内联 `HistoryHandler` + PD 控制器）：
- `examples/euler/07_locomotion/` 含 `locomotion.py` / `g1_base_env.py` /
  `scene_scanner.py` / `g1_locomotion.py` / `online_verifier.py` / `locomotion_env.py` /
  `07_locomotion.md`。
- `grep -r "from envs\." examples/euler/07_locomotion/` 零命中。
- `grep -r "HistoryHandler" examples/euler/07_locomotion/g1_locomotion.py` 命中（内联确认）。
- `grep -r "joint_kp\|PD" examples/euler/07_locomotion/g1_locomotion.py` 命中（PD 控制器确认）。

**教程文档更新点（`07_locomotion.md`）**：
- 路径/命令/目录结构更新。
- ONNX 策略路径说明改为 `assets/g1/models/dec_loco/model_6600.onnx`。
- 配置文件路径改为 `assets/g1/config/g1_29dof_hist.yaml`。
- 说明 PD 控制器：策略输出位置目标 `q_target`，经 `tau = Kp*(q_target-q) + Kd*(0-qd)` 转力矩
  后传给 motor 执行器（G1 执行器是力矩控制）。
- 删除视频采集相关内容（移至 Lesson 8）。

**数值判定项**（行走稳定性，5 项）：
1. `base_height_stable`：基座高度维持在合理范围（0.6-0.9m，DEFAULT_BASE_HEIGHT=0.78 附近）。
2. `not_fallen`：基座俯仰/横滚角未超过阈值（< 0.8 rad，约 45°，判定未摔倒）。
3. `joint_torque_within_limit`：关节力矩未持续触限（clip 比例 < 50%）。
4. `standing_at_start`：前 50 步 G1 保持站立（基座高度 > 0.6m）。
5. `policy_action_finite`：ONNX 输出无 NaN/Inf。

**人工观察项**（2 项）：
- `g1_standing`：Studio 视口 G1 应站立，不瘫倒。
- `g1_walking_stable`：Studio 视口 G1 行走应稳定，不乱踹（双腿交替迈步，非剧烈抖动）。

**验收通过条件**：§3.2 全部满足 + 5 项数值判定 PASS + 2 项人工观察确认。

**状态**：⏳ 待实施（原 studio_capture 拆分 + 行走问题修复）。

> **当前已知问题**：行走链路缺 PD 控制器导致瘫倒（已修复，加 PD 后返回力矩），但加 PD 后
> 出现"乱踹"现象，需进一步定位（见 §6 待定位问题）。

### 3.7 Lesson 8 重新验收（08_video_capture）

> **拆分背景**：原 Lesson 7 视频采集部分独立成课。在 Lesson 7 行走已验证的基础上，
> 叠加 Studio 视频/帧/时间戳采集 API 验证。

**验证内容**：G1 行走录制 + 视频截帧 + 帧索引/时间戳查询。

**执行命令**：
```bash
python examples/euler/08_video_capture/video_capture.py
```

**目录自查**（08 复用 07 的 `g1_locomotion.py` 驱动行走）：
- `examples/euler/08_video_capture/` 含 `video_capture.py` / `g1_base_env.py` /
  `scene_scanner.py` / `g1_locomotion.py` / `online_verifier.py` / `video_capture_env.py` /
  `08_video_capture.md`。
- `grep -r "from envs\." examples/euler/08_video_capture/` 零命中。

**教程文档更新点（`08_video_capture.md`）**：
- 路径/命令/目录结构更新。
- 说明本课依赖 Lesson 7 行走控制已验证（复用 `g1_locomotion.py`）。
- ONNX 策略路径说明改为 `assets/g1/models/dec_loco/model_6600.onnx`。

**数值判定项**（视频采集 API，5 项，沿用原 studio_capture）：
1. `camera_enabled`：`get_current_frame() >= 0`（摄像头使能）。
2. `frame_index_increasing_{step}`：每 50 步帧索引递增。
3. `png_file_generated`：`get_frame_png` 生成 PNG 文件（size > 100）。
4. `timestamp_returned`：`get_camera_time_stamp` 返回 `camera_head` 键。
5. `mp4_file_generated`：`stop_save_video` 后 mp4 文件生成。

**人工观察项**（1 项）：
- `g1_walking_in_video`：录制视频中 G1 行走画面正常（依赖 Lesson 7 行走已跑通）。

**验收通过条件**：§3.2 全部满足 + 5 项数值判定 PASS + 1 项人工观察确认 + **Lesson 7 已通过**
（行走链路前置依赖）。

**状态**：⏳ 待实施（原 studio_capture 拆分）。

### 3.8 重新验收总览表

| Lesson | 目录 | 数值判定数 | 人工观察数 | 关键独立性问题 | 状态 |
|--------|------|-----------|-----------|--------------|------|
| 04_query_api | 04_query_api/ | 9 | — | `scene_scanner` 探针改 Euler | ✅ 已通过 |
| 05_force_apply | 05_force_apply/ | 7 | — | 同上 | ✅ 已通过 |
| 06_jacobian | 06_jacobian/ | 3 | 1 | 同上 | ✅ 已通过 |
| 07_locomotion | 07_locomotion/ | 5 | 2 | + `HistoryHandler` 内联 + PD 控制器 + 行走稳定性 | ⏳ 待实施 |
| 08_video_capture | 08_video_capture/ | 5 | 1 | 拆分自原 studio_capture | ⏳ 待实施 |

---

## 4. Lesson 9 实施方案（新结构，原 Lesson 8 顺延）

### 4.1 验证内容

运行 G1 行走控制程序的同时，验证 Studio UI 拖拽 G1 机体、锚定/释放、等式约束更新，并通过
XML 内置的 mocap+weld 验证 `modify_equality_objects` 重绑定驱动 G1 动力学。

> **依赖**：`assets/g1/g1_29dof_camera.xml` 已内置 mocap body `ActorManipulator_Anchor` +
> weld 等式约束 `anchor_box_weld`（默认绑定 anchor↔box，详见原 phase4 §2.0.2）。
>
> **行走控制**：复用 07 的 `g1_locomotion.py`（含内联 `HistoryHandler` + PD 控制器），
> ONNX 策略路径 `assets/g1/models/dec_loco/model_6600.onnx`。

### 4.2 目录结构

```
examples/euler/09_body_manipulation/
├── 09_body_manipulation.md          ← 教程文档（§4.6）
├── body_manipulation.py             ← 脚本入口（§4.4）
├── body_manipulation_env.py         ← Env 子类（§4.3）
├── g1_base_env.py                   ← 复制（资产路径改指向 assets/g1）
├── scene_scanner.py                 ← 复制（探针 OrcaGymEulerEnv）
├── g1_locomotion.py                 ← 复制（含内联 HistoryHandler + PD 控制器）
└── online_verifier.py               ← 复制
```

### 4.3 `BodyManipulationEnv` 实现（`body_manipulation_env.py`）

继承 `G1BaseEnv`，重写钩子插入体操作验证。`compute_ctrl` 委托 `G1Locomotion.compute_action`。

```python
class BodyManipulationEnv(G1BaseEnv):
    def initialize_simulation(self):
        super().initialize_simulation()
        self.locomotion = G1Locomotion(agent_name=self.agent_name)

    def compute_ctrl(self, step: int) -> np.ndarray:
        return self.locomotion.compute_action(self)

    def verify_step(self, step: int, verifier: OnlineVerifier):
        agent = self.agent_name
        if step == 50:
            # 行走中：记录 pelvis 初始位姿
            self._pelvis_before = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]
        elif step == 100:
            # 程序化锚定 pelvis（weld）
            self.anchor_actor(f"{agent}/pelvis", anchor_type="weld")
            verifier.observe("anchor_pelvis",
                             "Studio 视口：G1 pelvis 被锚定，应悬停在当前位置")
        elif step == 120:
            # 锚定后：pelvis 位置稳定（atol=0.05）
            pelvis_after = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]
            verifier.check_allclose("anchored_position_stable",
                                     pelvis_after, self._pelvis_before, atol=0.05,
                                     detail="锚定后 pelvis 位置稳定")
        elif step == 150:
            # 释放锚定
            self.release_body_anchored()
            verifier.observe("release_anchor",
                             "Studio 视口：释放锚定，G1 恢复物理仿真与行走")
        elif step == 200:
            # 释放后：G1 恢复运动（位移 > 0.01m）
            pelvis_final = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]
            moved = np.linalg.norm(pelvis_final - self._pelvis_before) > 0.01
            verifier.check("released_resumes_motion", moved, moved, True,
                           detail="释放后 G1 恢复运动")
        # ...（mocap 驱动 box / equality 重绑 / mocap 驱动 G1 pelvis，见原 phase4 §4.3.4）
```

### 4.4 `body_manipulation.py` 脚本入口

```python
env = BodyManipulationEnv(
    frame_skip=G1_FRAME_SKIP,
    orcagym_addr=args.addr,
    agent_names=["g1"],
    time_step=G1_TIME_STEP,
    model_xml_path=G1_MODEL_XML,
)
verifier = OnlineVerifier("Lesson 9: 体操作与 equality")
report = env.run_lesson(num_steps=700, verifier=verifier)
```

### 4.5 Env 层 API 扩展（`equality_object_ids`）

原 phase4 §4.5 要求在 `OrcaGymEulerEnv` 扩展 `equality_object_ids` 公共方法（读取等式约束
绑定的 object id），供 Lesson 9 验证 `modify_equality_objects` 重绑结果。此扩展仍需执行，
详见原 phase4 文档。

### 4.6 教程文档大纲（`09_body_manipulation.md`）

1. **课程目标**：验证 `do_body_manipulation`/`anchor_actor`/`release_body_anchored`/
   `update_equality_constraints`/`modify_equality_objects`/`set_mocap_pos_and_quat`/
   `equality_object_ids` 在线运行正确。
2. **前置条件**：OrcaStudio 已加载含 G1（含 `ActorManipulator_Anchor`/`manipulation_box`/
   `anchor_box_weld`）的关卡并运行；`assets/g1/g1_29dof_camera.xml` 就位；
   **Lesson 7 行走控制已通过**（本课行走中叠加体操作）。
3. **目录结构**：本目录自包含，含 `body_manipulation.py`/`body_manipulation_env.py`/
   `g1_base_env.py`/`scene_scanner.py`/`g1_locomotion.py`/`online_verifier.py`。
4. **运行步骤**：
   - 步骤 1（人工）：启动 OrcaStudio，加载含 G1（含 mocap+box+weld）的关卡，点击运行。
   - 步骤 2（人工）：`cd OrcaPlayground && python examples/euler/09_body_manipulation/body_manipulation.py`
   - 步骤 3（自动）：脚本循环 700 帧，分阶段验证拖拽锚定/mocap 驱动 box/equality 重绑驱动 pelvis。
   - 步骤 4（人工）：按 `[OBSERVE]` 提示在 Studio 视口拖拽 G1 pelvis，观察锚定/释放效果；
     观察 anchor 拖动 box/G1。
   - 步骤 5（自动）：脚本输出 JSON 报告。
5. **预期结果**：
   - step 80：人工拖拽 G1 pelvis，`do_body_manipulation` 检测并锚定。
   - step 250-350：mocap 驱动 box 到 [0.7, 0, 0.5]（atol=0.05）。
   - step 450：停用 equality 后 box 不跟随。
   - step 500：重绑 equality 到 pelvis（obj2_id == pelvis_id）。
   - step 650：mocap 驱动 G1 pelvis 位移 > 0.05m。
   - JSON 报告 `all_passed == true`。
6. **视口观察**：G1 行走中可被鼠标拖拽锚定；释放后恢复行走；绿色球体 anchor 拖动橙色 box；
   重绑后 anchor 拖动 G1 整机。
7. **验证 API 列表**：`do_body_manipulation`/`anchor_actor`/`release_body_anchored`/
   `update_equality_constraints`/`modify_equality_objects`/`set_mocap_pos_and_quat`/
   `equality_object_ids`/`get_body_xpos_xmat_xquat`。
8. **故障排查**：
   - `mocap_drives_box_via_weld` 失败：检查 XML weld 约束 `anchor_box_weld` 是否 active；
     确认 `set_mocap_pos_and_quat` 的 dict 参数格式。
   - `eq_disable_decouples_box` 失败：检查 `update_equality_constraints` 的 `type=0` 是否正确写入。
   - `eq_rebound_to_pelvis` 失败：确认 `equality_object_ids` 已在 Env 层扩展（§4.5）。
   - `mocap_drives_g1_pelvis` 失败：检查重绑后 weld type 是否恢复为 `mjEQ_WELD`（step 450 已置 0，
     需在重绑后重新激活）。

### 4.7 Lesson 9 验收方案

| 验收项 | 通过条件 | 验证方法 |
|--------|---------|---------|
| 脚本可运行 | `python body_manipulation.py` 连接 Studio 成功 | 人工运行 |
| 独立性 | `grep -r "from envs\." examples/euler/09_body_manipulation/` 零命中 | grep |
| ruff SLF001 | 零报警 | `ruff check --select SLF001` |
| 锚定稳定 | `anchored_position_stable` PASS（atol=0.05） | 查看报告 |
| 释放恢复 | `released_resumes_motion` PASS（位移 > 0.01m） | 查看报告 |
| mocap 驱动 box | `mocap_drives_box_via_weld` PASS（atol=0.05） | 查看报告 |
| equality 停用解耦 | box 不跟随（视口确认） | 视口 + 报告 |
| equality 重绑 | `eq_rebound_to_pelvis` PASS（obj2_id == pelvis_id） | 查看报告 |
| mocap 驱动 G1 | `mocap_drives_g1_pelvis` PASS（位移 > 0.05m） | 查看报告 |
| 人工观察通过 | 拖拽/anchor 拖动 box/anchor 拖动 G1 符合预期 | 用户视口确认 |
| Env 层 API 扩展 | `equality_object_ids` 在 Env 层可调用 | `env.equality_object_ids(0)` 返回 tuple |
| 教程文档完整 | `09_body_manipulation.md` 覆盖 8 节大纲 | 人工审阅 |
| 前置依赖 | Lesson 7 locomotion 已通过 | 查看历史验收记录 |

---

## 5. 整体验收检查清单

### 5.1 目录结构验收
- [ ] `envs/euler/` 目录已删除。
- [ ] `assets/g1/` 含 XML/config/meshes/models，路径与 §2.2 一致。
- [ ] `assets/scenes/simple_pendulum.xml` 就位。
- [ ] 每个 example 目录含 §2.1 表所列全部 `.py` + `.md`。
- [ ] 原 `07_studio_capture/` 已拆分为 `07_locomotion/` + `08_video_capture/`。
- [ ] 原 `08_body_manipulation/` 规划已顺延为 `09_body_manipulation/`。

### 5.2 独立性验收（硬指标）
- [ ] `grep -r "from envs\." examples/euler/` 零命中。
- [ ] 每个 example 目录单独 `python -c "import <entry>"` 无 `ModuleNotFoundError`。

### 5.3 Euler 体系纯净验收
- [ ] `grep -r "OrcaGymLocalEnv\|orca_gym_local\|sim_env" examples/euler/` 零命中。
- [ ] `grep -r "orca_gym" examples/euler/` 仅命中 `orca_gym.environment.euler.*` 或
  `orca_gym.utils/protos/log`（通用基础设施）。

### 5.4 资产路径验收
- [ ] 每个 G1 lesson 目录：`python -c "from g1_base_env import G1_MODEL_XML, G1_CONFIG_YAML, G1_LOCO_ONNX; import os; print(all(os.path.exists(p) for p in [G1_MODEL_XML, G1_CONFIG_YAML, G1_LOCO_ONNX]))"` 返回 `True`。
- [ ] 01/02/03：`python -c "from simple_env import _SCENE_XML; import os; print(os.path.exists(_SCENE_XML))"` 返回 `True`。

### 5.5 ruff 静态检查
- [ ] `python -m ruff check --select SLF001 examples/euler/` 零报警。

### 5.6 功能验收（需 OrcaStudio 在线）
- [ ] Lesson 4：9 项数值判定 PASS。✅
- [ ] Lesson 5：数值判定 PASS。✅
- [ ] Lesson 6：3 项数值判定 + 1 项人工观察 PASS。✅
- [ ] Lesson 7：5 项数值判定 + 2 项人工观察 PASS（行走稳定性）。
- [ ] Lesson 8：5 项数值判定 + 1 项人工观察 PASS（视频采集，前置依赖 Lesson 7）。
- [ ] Lesson 9：§4.7 全部验收项 PASS（前置依赖 Lesson 7）。

### 5.7 教程文档验收
- [ ] Lesson 4–6 的 `.md` 路径/命令/目录结构已更新为新结构。✅
- [ ] Lesson 7 `07_locomotion.md` 覆盖行走控制链路 + PD 控制器说明。
- [ ] Lesson 8 `08_video_capture.md` 覆盖视频采集 API + 行走前置依赖说明。
- [ ] Lesson 9 `09_body_manipulation.md` 覆盖 §4.6 八节大纲。
- [ ] `TUTORIAL.md` 总纲已更新课程编号（7=locomotion, 8=video_capture, 9=body_manipulation）。

---

## 6. 待定位问题与风险

### 6.1 Lesson 7 行走"乱踹"问题（待定位）

**现象**：加 PD 控制器后（修复瘫倒），G1 不再瘫倒，但出现"腿乱踹"现象（双腿剧烈抖动/乱踢，
非正常交替迈步）。

**已排除**：
- ✅ PD 控制器缺失（已加，不再瘫倒）。
- ✅ 资产路径（已验证 `True`）。
- ✅ import 自洽（已验证 OK）。

**待排查方向**（按优先级）：
1. **观测布局不匹配**：`g1_locomotion.py` 的 `_build_obs` 拼接顺序是否与 ONNX 模型
   `model_6600.onnx` 训练时的观测布局一致（参考 `envs/g1/rl_policy/decoupled_locomotion_stand_height.py`）。
   维度错位会导致策略输出错误动作。
2. **history 初始化/更新顺序**：`compute_action` 中 `_update_history` 在 ONNX 推理之后调用，
   首帧 history 全零是否合理；`history_loco_height_config` 的 key 顺序与 `_get_obs_history` 的
   sorted key 拼接是否一致。
3. **PD 参数/力矩限位**：`JOINT_KP`/`JOINT_KD` 是否与原 `envs/g1` 实际使用的一致；
   `motor_effort_limit_list` clip 是否过紧导致力矩饱和。
4. **frame_skip/time_step**：`G1_FRAME_SKIP=20`、`G1_TIME_STEP=0.001`（控制频率 50Hz，
   物理步长 1ms），与原 `envs/g1` 配置是否一致。
5. **phase_time 用墙钟时间**：`_get_phase_time` 用 `time.time() - self._start_time`，而
   `run_lesson` 的 RTF 限速会导致墙钟与仿真时间不同步，步态相位可能漂移。原体系可能用仿真时间。
6. **初始姿态/重置**：`reset_model` 是否将 G1 重置到正确的站立 keyframe。

**定位方法**：
- 对照 `envs/g1/rl_policy/decoupled_locomotion_stand_height.py` 的 `prepare_obs` 逐步比对
  观测拼接顺序与缩放。
- 打印 ONNX 输入 obs 的 shape 与各分段数值范围，与原策略对比。
- 单步调试：固定 q_target=default_dof_angles（无策略），仅 PD 保持站立，验证 PD 本身正确。

### 6.2 代码重复风险
- `g1_base_env.py` 有 6 份副本（04/05/06/07/08/09），`online_verifier.py` 6 份，
  `scene_scanner.py` 6 份，`g1_locomotion.py` 3 份。后续改基类需同步多份。
  - **缓解**：用户已确认接受此代价以换取独立性；改动时用 `diff` 跨目录核对。

### 6.3 资产路径硬编码
- 依赖 `__file__` 上溯 4 层定位项目根，若 example 被移出项目会失效。
  - **缓解**：当前不可接受 example 移出项目的场景；路径计算有单元覆盖（§5.4）。

### 6.4 回滚
迁移前用 git 对 `envs/euler/` 与 `examples/euler/` 打标签（如 `pre-restructure`）。
若验收失败，`git checkout pre-restructure -- envs/euler examples/euler` 即可恢复原结构。

---

## 7. 附录：`scene_scanner.py` 探针实现要点

```python
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

def probe_scene_model(orcagym_addr: str, time_step: float) -> SceneModelNames:
    """探针 env：用 OrcaGymEulerEnv 连接 Studio，读取场景模型名表。

    用 OrcaGymEulerEnv（Euler 新主路径）替代原 model_scanner 的 SimEnv
    （依赖 envs.fluid.sim_env → Local 老主路径），避免 Euler 教程反向依赖 Local 体系。
    仅调用 model.get_* 方法，与探针基类无关。
    """
    probe = OrcaGymEulerEnv(
        frame_skip=1,
        orcagym_addr=orcagym_addr,
        agent_names=["SceneProbe"],
        time_step=time_step,
    )
    try:
        m = probe.model
        site_dict = m.get_site_dict() if hasattr(m, "get_site_dict") else {}
        sensor_dict = getattr(m, "_sensor_dict", {})
        return SceneModelNames(
            bodies=set(m.get_body_names()),
            joints=set(m.get_joint_dict().keys()),
            actuators=set(m.get_actuator_dict().keys()),
            sites=set(site_dict.keys()),
            sensors=set(sensor_dict.keys()),
        )
    finally:
        probe.close()
```

> **注意**：`_sensor_dict` 为 `OrcaGymModel` 的内部属性，此处探针读取属于"一次性场景扫描"
> 用途，与架构 §7 的"运行时 API 隔离"不冲突（探针 env 在扫描后立即 close，不参与仿真循环）。
> 若后续 `OrcaGymModel` 提供公共 `get_sensor_dict()`，应改用公共方法。
