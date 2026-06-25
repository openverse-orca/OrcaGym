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
- 封装隔离机制（M1-M6 六层机制）
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
- 跳过 `__getattr__` 拦截机制
- 在 `OrcaGymDataView` 之外提供 `_mjData` 访问通道

配套的开发阶段分解见：`docs/design/development/orca_gym_euler_development.md`

## 规则 3：GPU 加速与 Sandbox 旁路

Euler 体系使用 GPU 加速时（MuJoCoFlow / Flow 在 GPU 上求解），**无法在 TRAE sandbox 内正确运行**。sandbox 剥离了所有进程能力，导致 `cuInit` 返回 `CUDA_ERROR_304`。AI agent 必须使用 **TRAE 命令白名单** 旁路 sandbox，才能调用 GPU。

> 用户侧的白名单配置教程见 `DEVELOPMENT_GUIDE.md`。本规则约束 AI agent 的命令格式。

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
