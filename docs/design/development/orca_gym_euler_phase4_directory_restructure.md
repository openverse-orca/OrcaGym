# OrcaGym Euler 阶段四目录重组与 Lesson 4–8 重新验收方案

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

该布局存在两个问题：

1. **example 不独立**：每个 example 通过 `from envs.euler.xxx import ...` 依赖 `envs/euler/`
   共享目录，无法单独拷出运行。
2. **资产与代码混杂**：G1 mesh/ONNX/config 埋在 `envs/euler/robots/` 下，与 Env 代码同级，
   不便于资产单独管理与版本化。

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

### 1.3 适用范围

- **重新验收**：Lesson 4–7（已实施，需迁移到新结构并重新跑通）。
- **新设计**：Lesson 8（未实施，按新结构从零设计）。

---

## 2. 目录重组方案

### 2.1 目标目录结构

```
OrcaPlayground/
├── assets/                                   ← 新建：资产集中（§2.2）
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
│   ├── 07_studio_capture/
│   │   ├── 07_studio_capture.md
│   │   ├── studio_capture.py
│   │   ├── g1_base_env.py
│   │   ├── scene_scanner.py
│   │   ├── g1_locomotion.py                  ← 含内联 HistoryHandler（§2.4）
│   │   ├── online_verifier.py
│   │   └── studio_capture_env.py
│   └── 08_body_manipulation/                 ← 新建（Lesson 8，§4）
│       ├── 08_body_manipulation.md
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

### 2.4 `HistoryHandler` 内联到 `g1_locomotion.py`

`g1_locomotion.py` 原依赖 `envs.g1.utils.history_handler.HistoryHandler`（43 行，仅依赖
`numpy` + `orca_gym.log` 已安装包）。因仅 `g1_locomotion` 一处使用且代码极短，**直接内联**
到 `g1_locomotion.py` 中作为模块级类，消除 `envs.g1` 依赖。

### 2.5 每个 example 的依赖清单（重组后）

| example | 同目录 .py | 依赖的 `orca_gym` 模块 |
|---------|-----------|----------------------|
| 01/02/03 | `simple_env.py` | `orca_gym.environment.euler.orca_gym_euler_env` |
| 04 | `g1_base_env` + `scene_scanner` + `online_verifier` + `query_api_env` | 同上 |
| 05 | `g1_base_env` + `scene_scanner` + `online_verifier` + `force_apply_env` | 同上 |
| 06 | `g1_base_env` + `scene_scanner` + `online_verifier` + `jacobian_env` | 同上 |
| 07 | `g1_base_env` + `scene_scanner` + `g1_locomotion` + `online_verifier` + `studio_capture_env` | 同上 |
| 08 | `g1_base_env` + `scene_scanner` + `g1_locomotion` + `online_verifier` + `body_manipulation_env` | 同上 |

**`OrcaGymEulerEnv` 传递依赖全在 `orca_gym.core.euler.*` + `orca_gym.utils/protos/log`
（通用基础设施），零 Local 体系引用。**

### 2.6 迁移步骤

#### 步骤 1：建 `assets/` 并迁移资产
- `mkdir -p assets/g1/{config,meshes,models/dec_loco,models/mimic} assets/scenes`
- 按 §2.2 表移动文件，保持子结构。
- 删除 `envs/euler/robots/requirements.txt`（合并到 `examples/euler/requirements.txt`）。

#### 步骤 2：提取 `scene_scanner.py`
- 从 `envs/common/model_scanner.py` 提取（§2.3），探针改 `OrcaGymEulerEnv`。
- 放入 04/05/06/07/08 各一份。

#### 步骤 3：内联 `HistoryHandler`
- 将 `envs/g1/utils/history_handler.py` 的 `HistoryHandler` 类内联到 `g1_locomotion.py`。
- 放入 07/08 各一份（仅这两个 lesson 用 ONNX 行走）。

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
| `from envs.euler.studio_capture_env import ...` | `from studio_capture_env import ...` |
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

---

## 3. Lesson 4–7 重新验收方案

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

### 3.6 Lesson 7 重新验收（07_studio_capture）

**验证内容**：G1 行走录制 + 视频截帧 + 帧索引/时间戳查询。

**执行命令**：
```bash
python examples/euler/07_studio_capture/studio_capture.py
```

**目录自查**（07 额外含 `g1_locomotion.py`，且其内已内联 `HistoryHandler`）：
- `examples/euler/07_studio_capture/` 含 `studio_capture.py` / `g1_base_env.py` /
  `scene_scanner.py` / `g1_locomotion.py` / `online_verifier.py` / `studio_capture_env.py` /
  `07_studio_capture.md`。
- `grep -r "from envs\." examples/euler/07_studio_capture/` 零命中。
- `grep -r "HistoryHandler" examples/euler/07_studio_capture/g1_locomotion.py` 命中（内联确认）。

**教程文档更新点（`07_studio_capture.md`）**：
- 路径/命令/目录结构更新。
- ONNX 策略路径说明改为 `assets/g1/models/dec_loco/model_6600.onnx`。
- 配置文件路径改为 `assets/g1/config/g1_29dof_hist.yaml`。
- 删除"依赖 `envs/euler/g1_locomotion`"说明，改为"本目录内 `g1_locomotion.py`"。

**数值判定项**（与原 phase4 §4.3.4 一致，5 项）：camera_enabled、frame_index_increasing、
png_file_generated、timestamp_returned、mp4_file_generated。

**人工观察项**：g1_walking、walking_stable。

**验收通过条件**：§3.2 全部满足 + 5 项数值判定 PASS + 2 项人工观察确认。

### 3.7 重新验收总览表

| Lesson | 目录 | 数值判定数 | 人工观察数 | 关键独立性问题 |
|--------|------|-----------|-----------|--------------|
| 04_query_api | 04_query_api/ | 9 | — | `scene_scanner` 探针改 Euler |
| 05_force_apply | 05_force_apply/ | 7 | — | 同上 |
| 06_jacobian | 06_jacobian/ | 3 | 1 | 同上 |
| 07_studio_capture | 07_studio_capture/ | 5 | 2 | + `HistoryHandler` 内联确认 |

---

## 4. Lesson 8 实施方案（新结构）

### 4.1 验证内容

运行 G1 行走控制程序的同时，验证 Studio UI 拖拽 G1 机体、锚定/释放、等式约束更新，并通过
XML 内置的 mocap+weld 验证 `modify_equality_objects` 重绑定驱动 G1 动力学。

> **依赖**：`assets/g1/g1_29dof_camera.xml` 已内置 mocap body `ActorManipulator_Anchor` +
> weld 等式约束 `anchor_box_weld`（默认绑定 anchor↔box，详见原 phase4 §2.0.2）。
>
> **行走控制**：复用 07 的 `g1_locomotion.py`（含内联 `HistoryHandler`），ONNX 策略路径
> `assets/g1/models/dec_loco/model_6600.onnx`。

### 4.2 目录结构

```
examples/euler/08_body_manipulation/
├── 08_body_manipulation.md          ← 教程文档（§4.6）
├── body_manipulation.py             ← 脚本入口（§4.4）
├── body_manipulation_env.py         ← Env 子类（§4.3）
├── g1_base_env.py                   ← 复制（资产路径改指向 assets/g1）
├── scene_scanner.py                 ← 复制（探针 OrcaGymEulerEnv）
├── g1_locomotion.py                 ← 复制（含内联 HistoryHandler）
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
            verifier.check("released_resumes_motion", moved,
                           pelvis_final.tolist(), "moved from anchor",
                           "释放后 G1 恢复运动")
        # === mocap + weld 约束验证（不依赖 Studio 锚点）===
        elif step == 250:
            # 默认绑定 anchor↔box：mocap 驱动 box
            self._box_before = self.get_body_xpos_xmat_xquat(
                ["manipulation_box"])["manipulation_box"]["xpos"]
            self.set_mocap_pos_and_quat(
                {"ActorManipulator_Anchor": {"pos": [0.7, 0.0, 0.5], "quat": [1, 0, 0, 0]}})
        elif step == 350:
            # 步进 100 帧后 box 跟随 mocap（atol=0.05）
            box_after = self.get_body_xpos_xmat_xquat(
                ["manipulation_box"])["manipulation_box"]["xpos"]
            verifier.check_allclose("mocap_drives_box_via_weld",
                                     box_after, [0.7, 0.0, 0.5], atol=0.05,
                                     detail="mocap 驱动 box 到目标位姿")
        elif step == 450:
            # 停用 equality[0]：box 不再跟随
            self.update_equality_constraints([{
                "type": 0, "obj1_id": -1, "obj2_id": -1,
                "data": np.zeros(mujoco.mjNEQDATA)}])
            verifier.observe("eq_disabled",
                             "Studio 视口：equality 停用，box 不再跟随 anchor")
        elif step == 500:
            # 重绑 equality[0] weld 的 obj2 到 G1 pelvis
            self.modify_equality_objects(eq_ids=[0], obj2_names=[f"{agent}/pelvis"])
            # 校验 obj2_id 已改为 pelvis
            obj1_id, obj2_id = self.equality_object_ids(0)
            pelvis_id = self.model.body_name2id(f"{agent}/pelvis")
            verifier.check("eq_rebound_to_pelvis",
                           obj2_id == pelvis_id, obj2_id, pelvis_id,
                           "equality 重绑到 pelvis")
        elif step == 550:
            # mocap 驱动 G1 pelvis：写入位姿偏移
            self._pelvis_pre_drive = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]
            self.set_mocap_pos_and_quat({
                "ActorManipulator_Anchor": {
                    "pos": (self._pelvis_pre_drive + np.array([0.2, 0, 0.1])).tolist(),
                    "quat": [1, 0, 0, 0]}})
        elif step == 650:
            # 步进 100 帧后 G1 pelvis 位移 > 0.05m
            pelvis_driven = self.get_body_xpos_xmat_xquat(
                [f"{agent}/pelvis"])[f"{agent}/pelvis"]["xpos"]
            moved = np.linalg.norm(pelvis_driven - self._pelvis_pre_drive) > 0.05
            verifier.check("mocap_drives_g1_pelvis", moved,
                           pelvis_driven.tolist(), "moved > 0.05m",
                           "mocap 驱动 G1 pelvis")

    def observe_step(self, step: int, verifier: OnlineVerifier):
        if step == 0:
            verifier.observe("g1_walking", "Studio 视口：G1 应在策略控制下行走")
        elif step == 80:
            verifier.observe("manual_drag_hint",
                             "可在 Studio 视口用鼠标拖拽 G1 pelvis，观察锚定效果")

    def after_loop(self, verifier: OnlineVerifier):
        # 恢复 equality 默认绑定，避免影响后续测试
        try:
            self.modify_equality_objects(eq_ids=[0], obj2_names=["manipulation_box"])
        except Exception:
            pass
```

### 4.4 脚本入口（`body_manipulation.py`）

按原 phase4 §3.2 模板，`num_steps=700`。

```python
def main() -> None:
    args = parse_args()
    env = BodyManipulationEnv(
        frame_skip=G1_FRAME_SKIP,
        orcagym_addr=args.addr,
        agent_names=["g1"],
        time_step=G1_TIME_STEP,
        model_xml_path=G1_MODEL_XML,
    )
    verifier = OnlineVerifier("Lesson 8: 体操作")
    try:
        report = env.run_lesson(num_steps=700, verifier=verifier)
    finally:
        env.close()
    if not report["summary"]["all_passed"]:
        sys.exit(1)
```

### 4.5 Env 层 API 扩展（`orca_gym` 包）

`equality_object_ids` 需在 `OrcaGymEulerEnv` 扩展（委托 `self._gym`）：

```python
def equality_object_ids(self, eq_id: int) -> tuple[int, int]:
    """返回 equality[eq_id] 的 (obj1id, obj2id)（委托 self._gym）。"""
    return self.loop.run_until_complete(self._gym.equality_object_ids(eq_id))
```

> **架构合规**：通过 Env 层公共方法委托，不穿墙访问 `_mjModel.eq_obj1id`。

### 4.6 教程文档大纲（`08_body_manipulation.md`）

1. **课程目标**：验证 `do_body_manipulation`/`anchor_actor`/`release_body_anchored`/
   `update_equality_constraints`/`modify_equality_objects`/`set_mocap_pos_and_quat`/
   `equality_object_ids` 在线运行正确。
2. **前置条件**：OrcaStudio 已加载含 G1（含 `ActorManipulator_Anchor`/`manipulation_box`/
   `anchor_box_weld`）的关卡并运行；`assets/g1/g1_29dof_camera.xml` 就位。
3. **目录结构**：本目录自包含，含 `body_manipulation.py`/`body_manipulation_env.py`/
   `g1_base_env.py`/`scene_scanner.py`/`g1_locomotion.py`/`online_verifier.py`。
4. **运行步骤**：
   - 步骤 1（人工）：启动 OrcaStudio，加载含 G1（含 mocap+box+weld）的关卡，点击运行。
   - 步骤 2（人工）：`cd OrcaPlayground && python examples/euler/08_body_manipulation/body_manipulation.py`
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

### 4.7 Lesson 8 验收方案

| 验收项 | 通过条件 | 验证方法 |
|--------|---------|---------|
| 脚本可运行 | `python body_manipulation.py` 连接 Studio 成功 | 人工运行 |
| 独立性 | `grep -r "from envs\." examples/euler/08_body_manipulation/` 零命中 | grep |
| ruff SLF001 | 零报警 | `ruff check --select SLF001` |
| 锚定稳定 | `anchored_position_stable` PASS（atol=0.05） | 查看报告 |
| 释放恢复 | `released_resumes_motion` PASS（位移 > 0.01m） | 查看报告 |
| mocap 驱动 box | `mocap_drives_box_via_weld` PASS（atol=0.05） | 查看报告 |
| equality 停用解耦 | box 不跟随（视口确认） | 视口 + 报告 |
| equality 重绑 | `eq_rebound_to_pelvis` PASS（obj2_id == pelvis_id） | 查看报告 |
| mocap 驱动 G1 | `mocap_drives_g1_pelvis` PASS（位移 > 0.05m） | 查看报告 |
| 人工观察通过 | 拖拽/anchor 拖动 box/anchor 拖动 G1 符合预期 | 用户视口确认 |
| Env 层 API 扩展 | `equality_object_ids` 在 Env 层可调用 | `env.equality_object_ids(0)` 返回 tuple |
| 教程文档完整 | `08_body_manipulation.md` 覆盖 8 节大纲 | 人工审阅 |

---

## 5. 整体验收检查清单

### 5.1 目录结构验收
- [ ] `envs/euler/` 目录已删除。
- [ ] `assets/g1/` 含 XML/config/meshes/models，路径与 §2.2 一致。
- [ ] `assets/scenes/simple_pendulum.xml` 就位。
- [ ] 每个 example 目录含 §2.1 表所列全部 `.py` + `.md`。

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
- [ ] Lesson 4：9 项数值判定 PASS。
- [ ] Lesson 5：数值判定 PASS。
- [ ] Lesson 6：3 项数值判定 + 1 项人工观察 PASS。
- [ ] Lesson 7：5 项数值判定 + 2 项人工观察 PASS。
- [ ] Lesson 8：§4.7 全部验收项 PASS。

### 5.7 教程文档验收
- [ ] Lesson 4–7 的 `.md` 路径/命令/目录结构已更新为新结构。
- [ ] Lesson 8 的 `.md` 覆盖 §4.6 八节大纲。

---

## 6. 风险与回滚

### 6.1 风险
- **代码重复**：`g1_base_env.py` 有 5 份副本（04/05/06/07/08），`online_verifier.py` 5 份，
  `scene_scanner.py` 5 份，`g1_locomotion.py` 2 份。后续改基类需同步多份。
  - **缓解**：用户已确认接受此代价以换取独立性；改动时用 `diff` 跨目录核对。
- **资产路径硬编码**：依赖 `__file__` 上溯 4 层定位项目根，若 example 被移出项目会失效。
  - **缓解**：当前不可接受 example 移出项目的场景；路径计算有单元覆盖（§5.4）。

### 6.2 回滚
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
