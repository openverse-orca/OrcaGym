# OrcaGym AI 开发指南

本文件为 AI 代理（如 Trae、Cursor 等）在本仓库工作时提供强制规则。AI 代理必须严格遵守。

## 规则 1：测试与调试环境

AI 代理执行测试、调试、运行脚本时，**必须使用 `orca` conda 环境**。

```bash
# 正确
conda activate orca
python -m pytest tests/...
python some_script.py

# 错误 — 不要使用 base 或其他环境
conda activate base
python some_script.py
```

`orca` 是 README 推荐的环境名称，已安装本项目所有依赖。使用其他环境会导致依赖缺失或版本不一致。

## 规则 2：Euler 开发的架构约束

`OrcaGymEulerEnv` + `OrcaGymEuler` 的开发**必须严格遵照架构文档约束**：

`docs/design/architecture/orca_gym_euler_architecture.md`

该文档定义了：

- 组件划分（`OrcaGymEulerEnv` / `OrcaGymEuler` / `MuJoCoSimCore` / `OrcaStudioBridge` / `ModelRegistry` / `SimConfig` / `OrcaGymDataView`）
- API 契约（L1/L2/L3 三层 + R/W/S/C/N 五类规则）
- 封装隔离机制（M0-M7 多层封装隔离机制）
- 步进编排契约
- 迁移策略

### 冲突处理

若开发过程中发现以下情况，**必须暂停并提交用户决策**，不得自行绕过：

1. 架构文档与现有代码存在矛盾
2. 架构文档的某项约束无法满足
3. 需要引入架构文档未定义的新组件或新 API
4. 需要修改架构文档定义的契约（如放宽封装隔离、暴露内部组件等）

**禁止的行为**：

- 为绕过约束而修改架构文档
- 在 `OrcaGymEuler` / `OrcaGymEulerEnv` 中暴露 `_mjModel` / `_mjData`
- 在 `OrcaGymDataView` 之外提供 `_mjData` 访问通道

配套的开发阶段分解见：`docs/design/development/orca_gym_euler_development.md`

## 规则 3：GPU 加速与 Sandbox 旁路

Euler 体系使用 GPU 加速时（MuJoCoFlow / Flow 在 GPU 上求解），**无法在 TRAE sandbox 内正确运行**。sandbox 剥离了所有进程能力，导致 `cuInit` 返回 `CUDA_ERROR_304`。AI agent 必须使用 **TRAE 命令白名单** 旁路 sandbox，才能调用 GPU。

> 用户侧的白名单配置教程见 `DEVELOPER_GUIDE.md`。本规则约束 AI agent 的命令格式。

### 核心规则

1. **GPU 命令必须以白名单解释器路径开头**。直接使用 `<conda-base>/envs/orca/bin/python` 作为命令首 token（可通过 `conda info --base` 解析 `<conda-base>`）。

2. **禁止使用 shell 管道 `|`**。管道会触发 IDE 用 `trae-sandbox` 包裹命令，重新引入能力限制，导致 `CUDA_ERROR_304`。包括 `| tail`、`| grep`、`2>&1 | ...` 等所有管道构造。

3. **输出捕获用重定向，不用管道**。如需捕获输出，将日志重定向到文件，再单独读取：
   ```bash
   # 正确 — 重定向到文件（通常安全）
   <conda-base>/envs/orca/bin/python script.py > /tmp/out.log 2>&1

   # 错误 — 管道触发 sandbox 包裹
   <conda-base>/envs/orca/bin/python script.py 2>&1 | tail -30
   ```

4. **若需切换目录，用 `cd` 链接**。`cd` 已在白名单中，`cd <repo-root> && <conda-base>/envs/orca/bin/python script.py` 整条链在宿主执行。

### 命令格式示例

```bash
# ✅ 正确 — 白名单解释器直接调用，无管道
<conda-base>/envs/orca/bin/python -m pytest tests/orca_gym/core/euler/...

# ✅ 正确 — cd 链接 + 白名单解释器
cd <repo-root> && <conda-base>/envs/orca/bin/python some_script.py

# ✅ 正确 — 重定向到文件捕获输出
<conda-base>/envs/orca/bin/python some_script.py > /tmp/out.log 2>&1

# ❌ 错误 — 管道触发 sandbox 包裹，GPU 不可用
<conda-base>/envs/orca/bin/python some_script.py 2>&1 | tail -30

# ❌ 错误 — 非白名单首 token
bash -c "<conda-base>/envs/orca/bin/python some_script.py"
```

### 识别 sandbox 包裹

若命令日志中出现 `trae-sandbox '...'` 前缀，说明命令被包裹（白名单未匹配或使用了管道）。此时 GPU 不可用，需简化命令：以白名单解释器路径开头，移除管道。

### CPU 测试无需旁路

仅使用 CPU 或纯 NumPy 的测试（如 `OrcaGymDataView` 数据结构测试、`SimConfig` 构造测试）可在 sandbox 内直接运行，无需白名单旁路。应将 GPU 依赖的测试与 CPU 测试分离，GPU 测试标记为仅在 sandbox 外运行。

## 规则 4：API 隔离强制

本仓库采用 `_` 前缀社区约定 + ruff SLF001 静态检查 + `__dir__` 控制，引导 AI 和用户走公共 API（架构 §7）。

### 禁止穿墙访问

不得访问以下 `_` 前缀内部属性（类内部合法的 `self._xxx` 委托除外）：

- `env._gym` / `env._stub` / `env._channel` / `env._studio_bridge`
- `env._gym._sim` / `env._gym._sim._mjData` / `env._gym._sim._mjModel`
- 任何自研类的 `_` 前缀属性

> `env.gym` / `env.stub` / `env.channel` 在 `OrcaGymEulerEnv` 中不存在（直接继承 `gym.Env`，Python 原生 `AttributeError`）。

### 必须使用公共 API

| 操作 | 正确 | 禁止 |
|------|------|------|
| 读取状态 | `env.data.qpos` / `env.data.body_xpos(name)` / `env.query_*()` | `env._gym._sim._mjData.qpos` |
| 写入状态 | `env.set_joint_qpos()` / `env.apply_body_force()` | `env._gym._sim._mjData.xfrc_applied[...]` |
| 步进 | `env.do_simulation(ctrl, n_frames)` / `env.step()` | `env._gym._sim._mjData.step()` |
| 求解器配置 | `env.sim_config.timestep = 0.002` | `env._gym._sim._mjModel.opt.timestep = 0.002` |

### 必须执行 ruff

提交代码前必须执行，零报警方可提交：

    <conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca_gym/

### 缺失功能时扩展公共方法

若公共 API 不满足需求，**暂停并提交用户决策**，不得自行穿墙访问内部属性。扩展方式：
- 在 `OrcaGymEulerEnv` 增加公共方法（委托到 `_gym` 公共 API）
- 在 `OrcaGymEuler` 增加公共方法（委托到 `_sim` 公共 API）
- 在 `OrcaGymDataView` 增加字段访问器

## 规则 5：Protobuf 代码生成

### 必须使用 orca 环境

Python 侧的 proto 生成脚本 `orca_gym/protos/generate_proto.py` **必须在 `orca` conda 环境中执行**。

`orca` 环境安装的 `protobuf` 和 `grpc-tools` 版本与 OrcaStudio C++ 侧（gRPC 1.51.1）严格匹配。使用其他环境（如 `base`）会导致生成的 `mjc_message_pb2.py` / `mjc_message_pb2_grpc.py` 版本不兼容，引发运行时序列化错误或 gRPC 调用失败。

```bash
# 正确 — 使用 orca 环境的 python 解释器
<conda-base>/envs/orca/bin/python orca_gym/protos/generate_proto.py

# 错误 — 使用 base 或系统 python
python orca_gym/protos/generate_proto.py
```

> `<conda-base>` 可通过 `conda info --base` 解析。具体解释器路径见 `~/.trae-cn/memory/user_profile.md` 的 `${ORCA_PYTHON}`。

### 同步规则

- proto 文件是**手动生成**的，不是自动编译的
- 修改 `orca_gym/protos/mjc_message.proto` 后，必须运行 `generate_proto.py` 重新生成 pb 文件
- **C++ 侧与 Python 侧的 proto 文件必须保持一致**：修改一处后必须同步修改另一侧（C++ 侧 proto 在 OrcaEngine2409 仓库 `Gems/Mujoco/Code/Source/GrpcService/protos/mjc_message.proto`）并各自重新生成
- C++ 侧生成脚本与配置指引见 OrcaEngine2409 仓库根目录 `AGENTS.md`
