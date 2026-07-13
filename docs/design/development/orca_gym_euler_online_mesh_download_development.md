SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2026 The OrcaGym Contributors

# OrcaGym Euler 在线场景 mesh 资源自动下载补全文档

## 1. 文档定位

### 1.1 文档目标

本文针对 `OrcaGymEulerEnv` 在线加载 OrcaStudio 场景 XML 时**不会自动下载 mesh/hfield 资源（STL/PNG 等）**的缺陷，进行根因分析并给出根治方案。

**问题表象**：在线模式下，Euler 体系的 `OrcaGymEulerEnv` 加载 G1 等场景时，`MjModel.from_xml_path` 报告 STL 文件缺失，仿真初始化失败。Local 体系加载同一场景无此问题。

**问题定性**：这是 Euler 体系自身的功能缺陷，不是 example 代码问题。根据 `AGENTS.md` 规则 2，example 层面无法干净地解决，必须由 OrcaGym 开发者在 Euler 体系内补全。

### 1.2 上游约束

| 文档 | 约束范围 |
|------|---------|
| `docs/design/architecture/orca_gym_euler_architecture.md` | §5.4 OrcaStudioBridge 职责（通信与场景同步）、§7 封装隔离机制 M0-M7、§6 API 契约 |
| `AGENTS.md` | 规则 2（Euler 架构约束、冲突处理必须暂停提交用户决策）、规则 4（API 隔离强制，禁止穿墙） |

### 1.3 修订原则

1. **不破坏封装隔离**：补全逻辑必须落在 `OrcaStudioBridge` 内部，example 代码不得触 `_gym`/`_studio` 等私有属性在两步之间插入下载。
2. **不暴露内部组件**：不通过 `@property` 暴露 bridge，不放宽 M0-M7 任何机制。
3. **复用 Local 体系成熟逻辑**：`process_xml_file` / `process_xml_node` / 原子落盘 + 文件锁逻辑已在 Local 体系验证，原样迁移语义、按 Euler 组件契约改写签名。
4. **离线模式不退化**：`skip_grpc_load=True` 路径不触发 gRPC，缺失资源直接报错（与 Local 离线行为对齐）。

---

## 2. 问题详述

### 2.1 现象

在线模式（`skip_grpc_load=False`）下，`OrcaGymEulerEnv` 实例化时 `initialize_simulation` 抛出 MuJoCo 资源缺失错误，形如：

```
Error: could not find mesh file 'g1/.../foot.stl'
```

而先用 Local 体系跑一次同一场景（如 `examples/g1/run_g1_sim.py`），STL 被下载到 `~/.orcagym/tmp/` 后，Euler 再加载同一 XML 即成功——证明问题不在 XML 本身，而在 Euler 缺少 mesh 下载步骤。

### 2.2 复现路径

`OrcaGymEulerEnv.__init__` → `initialize_simulation()`（[orca_gym_euler_env.py:197](../../../orca_gym/environment/euler/orca_gym_euler_env.py#L197)）：

```python
def initialize_simulation(self) -> Tuple[Any, OrcaGymDataView]:
    if self._skip_grpc_load:
        model_xml_path = self._local_xml_path
    else:
        model_xml_path = self.loop.run_until_complete(self._gym.load_model_xml())
    # ↑ 仅下载 XML 本体，未下载 mesh
    self.loop.run_until_complete(self._gym.init_simulation(model_xml_path))
    # ↑ MjModel.from_xml_path 因 STL 缺失而失败
    ...
```

两步之间没有 mesh 下载钩子，且两步都通过 `_gym`（私有属性）调用，example 代码无法在中间插入逻辑而不穿墙（违反 `AGENTS.md` 规则 4）。

### 2.3 根因对比

| 维度 | Local 体系（正确） | Euler 体系（缺陷） |
|------|-------------------|-------------------|
| XML 下载入口 | [orca_gym_local.py:241](../../../orca_gym/core/orca_gym_local.py#L241) `load_model_xml` | [orca_studio_bridge.py:90](../../../orca_gym/core/euler/orca_studio_bridge.py#L90) `load_model_xml` |
| 在线 XML 拉取 | [orca_gym_local.py:792](../../../orca_gym/core/orca_gym_local.py#L792) `load_local_env` | [orca_studio_bridge.py:109](../../../orca_gym/core/euler/orca_studio_bridge.py#L109) `_load_model_xml_online` |
| **XML 解析 + mesh 递归下载** | **[orca_gym_local.py:723](../../../orca_gym/core/orca_gym_local.py#L723) `process_xml_file` + [orca_gym_local.py:493](../../../orca_gym/core/orca_gym_local.py#L493) `process_xml_node`** | **❌ 无对应实现** |
| 资源下载 + 落盘 | [orca_gym_local.py:403](../../../orca_gym/core/orca_gym_local.py#L403) `load_content_file`（gRPC + 原子写盘 + 文件锁） | [orca_studio_bridge.py:496](../../../orca_gym/core/euler/orca_studio_bridge.py#L496) `load_content_file`（**薄 gRPC 包装，仅发请求、不捕获响应、不落盘**） |

**两个关键缺口**：

1. **缺递归解析逻辑**：Euler 的 `_load_model_xml_online` 下载 XML 后直接返回路径，不解析 `mesh`/`hfield` 节点的 `file` 属性，不检查本地是否存在对应 STL/PNG。
2. **缺落盘逻辑**：Euler 的 `load_content_file` docstring 自述"Bridge 层为薄 gRPC 包装：仅发起请求，文件落盘由上层处理"，但**没有任何上层组件实际承担落盘**——这是设计与实现脱节的缺口。

### 2.4 不能在 example 层面解决的原因

- `load_model_xml` 和 `init_simulation` 都是 `OrcaGymEuler` 的方法，example 通过 `env._gym`（私有）调用即违反 `AGENTS.md` 规则 4。
- `OrcaGymEulerEnv` 的 `initialize_simulation` 在 `__init__` 内一次性编排完两步，外部子类没有合法的注入点。
- 即便子类复写 `initialize_simulation`，也无法合法访问 `_gym._studio` 触发下载。

按 `AGENTS.md` 规则 2，必须暂停并提交用户决策，由开发者在 Euler 体系内补全。

---

## 3. 方案设计

### 3.1 根治方案

**落点**：全部补全逻辑落在 `OrcaStudioBridge` 内部（架构 §5.4 已定义其职责为"通信与场景同步"，mesh 下载属于场景同步）。

**改动范围**：
- 仅修改 `orca_gym/core/euler/orca_studio_bridge.py`。
- `OrcaGymEuler` / `OrcaGymEulerEnv` / `MuJoCoSimCore` / example 代码**零改动**。

**新增/修改方法**：

| 方法 | 可见性 | 职责 | 对应 Local 来源 |
|------|--------|------|----------------|
| `process_xml_file(file_path)` | 公共（async） | 读取 XML、解析根节点、调用 `process_xml_node` | [orca_gym_local.py:723](../../../orca_gym/core/orca_gym_local.py#L723) |
| `process_xml_node(node)` | 公共（async） | 递归遍历 `mesh`/`hfield` 节点，缺失文件调用 `_download_asset_to_cache` | [orca_gym_local.py:493](../../../orca_gym/core/orca_gym_local.py#L493) |
| `_download_asset_to_cache(file_name)` | 私有（async） | gRPC `LoadContentFile` + 捕获响应 + 原子落盘到 `xml_file_dir` + 文件锁 | [orca_gym_local.py:403](../../../orca_gym/core/orca_gym_local.py#L403) |
| `_load_model_xml_online` | 修改 | XML 落盘后、返回路径前，调用 `await self.process_xml_file(file_path)` | — |

**调用链（修复后）**：

```
OrcaGymEulerEnv.initialize_simulation
  └── OrcaGymEuler.load_model_xml
        └── OrcaStudioBridge.load_model_xml
              └── _load_model_xml_online
                    1. gRPC 拉取 XML 文件名 + 内容（已有）
                    2. 原子写盘 XML（已有）
                    3. ★ await process_xml_file(file_path)  ← 新增
                    │     └── process_xml_node(root)
                    │           ├── mesh/hfield 节点：检查 file 存在
                    │           │   └── 缺失 → _download_asset_to_cache
                    │           │                ├── gRPC LoadContentFile
                    │           │                ├── 捕获 response.content
                    │           │                └── 原子写盘 + file_lock
                    │           └── 其他节点：递归子节点
                    4. 返回 XML 路径（此时 mesh 已全部就位）
  └── OrcaGymEuler.init_simulation
        └── MuJoCoSimCore.init_simulation
              └── MjModel.from_xml_path  ← STL 已在缓存，成功
```

### 3.2 离线模式行为

`_stub is None` 时：
- `_load_model_xml_online` 不被调用（`load_model_xml` 走 `configure_offline` 配置的本地路径分支）。
- `process_xml_file` 仍会被调用吗？**会**——在离线分支也应检查本地 mesh 完整性。但离线模式下 `_download_asset_to_cache` 不能触 gRPC，应直接抛 `FileNotFoundError`（对齐 Local 的 `_skip_grpc_load` 行为，[orca_gym_local.py:514-518](../../../orca_gym/core/orca_gym_local.py#L514)）。

**实现选择**：`process_xml_file` 在 `load_model_xml` 的**两个分支后统一调用**（离线分支返回本地路径后、在线分支返回缓存路径后），保证离线模式也能尽早发现资源缺失并给出清晰错误，而不是等到 `MjModel.from_xml_path` 报底层 MuJoCo 错误。

修正后的 `load_model_xml`：

```python
async def load_model_xml(self) -> str:
    if self._stub is None:
        if self._local_xml_path is None:
            raise RuntimeError("Offline mode but no local_xml_path configured")
        if not os.path.isfile(self._local_xml_path):
            raise FileNotFoundError(f"local_xml_path not found: {self._local_xml_path}")
        xml_path = self._local_xml_path
    else:
        xml_path = await self._load_model_xml_online()
    # ★ 统一在此处检查/补全 mesh 资源
    await self.process_xml_file(xml_path)
    return xml_path
```

### 3.3 架构合规性核查

| 约束 | 核查 |
|------|------|
| §5.4 OrcaStudioBridge 职责 | ✅ mesh 下载属"场景同步"，落点正确 |
| §5.4 依赖反转 | ✅ bridge 不持有 `_mjData`，下载逻辑只操作本地文件系统与 gRPC |
| §5.4 不碰 `mj_step` | ✅ 下载在 `init_simulation` 之前完成，不涉步进 |
| M0-M7 隔离机制 | ✅ 全部新增方法在 bridge 内部，无新暴露 `_mjData`/`_mjModel` 通道 |
| K9 Studio 桥接访问 | ✅ 不通过 `@property` 暴露 bridge，Env 侧零改动 |
| `AGENTS.md` 规则 4 | ✅ example 无需穿墙，公共 API 调用链不变 |
| `AGENTS.md` 规则 2 冲突处理 | ✅ 本文档即为"提交用户决策"环节，方案落地前需用户确认 |

**无需修改架构文档**：本方案在 §5.4 已定义的职责范围内补全实现，不引入新组件、不放宽契约。`process_xml_file` / `process_xml_node` 可加入 §5.4 的方法清单（属文档增补，非契约修改）。

---

## 4. 实现细节

### 4.1 `_download_asset_to_cache`（私有，核心新增）

从 Local 的 `load_content_file`（[orca_gym_local.py:403](../../../orca_gym/core/orca_gym_local.py#L403)）抽取语义，按 Euler bridge 契约改写：

```python
async def _download_asset_to_cache(self, content_file_name: str) -> str:
    """从 Studio 下载资源文件并原子落盘到 xml_file_dir。

    离线模式（_stub is None）抛 FileNotFoundError，对齐 Local 离线行为。

    Args:
        content_file_name: 资源文件名（如 "g1/foot.stl"）。

    Returns:
        本地缓存文件绝对路径。
    """
    if self._stub is None:
        raise FileNotFoundError(
            f"Offline mode: missing mesh/asset '{content_file_name}' "
            f"(place file under xml assets dir: {self.xml_file_dir})"
        )
    content_file_path = os.path.join(self.xml_file_dir, content_file_name)
    async with file_lock(content_file_path, timeout=30):
        if os.path.exists(content_file_path):
            return content_file_path  # 已存在（可能在等锁期间被其他进程创建）
        request = mjc_message_pb2.LoadContentFileRequest(
            file_name=content_file_name, file_dir=""
        )
        response = await self._stub.LoadContentFile(request)
        if response.status != mjc_message_pb2.LoadContentFileResponse.SUCCESS:
            raise Exception(f"LoadContentFile failed for '{content_file_name}'")
        content = response.content
        if not content:
            raise Exception(f"LoadContentFile returned empty content for '{content_file_name}'")
        # 原子化保存：先写临时文件，再 move
        os.makedirs(os.path.dirname(content_file_path), exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            dir=os.path.dirname(content_file_path),
            delete=False,
            prefix=f"{os.path.basename(content_file_name)}_",
            suffix=".tmp",
        )
        try:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            shutil.move(temp_file.name, content_file_path)
        except Exception:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
            raise
    return content_file_path
```

**与 Local 版本的差异**：
- Local 版本支持 `remote_file_dir` / `local_file_dir` / `temp_file_path` 参数；Euler 版本简化为固定落盘到 `xml_file_dir`（Euler 的资源目录即 XML 缓存目录，由 `xml_file_dir` property 统一管理）。
- Local 版本通过 `self._skip_grpc_load` 判定离线；Euler 版本通过 `self._stub is None` 判定（Euler 的离线即 stub 为 None）。

### 4.2 `process_xml_node`（公共，递归）

原样迁移 Local 逻辑（[orca_gym_local.py:493](../../../orca_gym/core/orca_gym_local.py#L493)）：

```python
async def process_xml_node(self, node) -> None:
    """递归处理 XML 节点，下载缺失的 mesh/hfield 资源。"""
    if node.tag in ('mesh', 'hfield'):
        content_file_name = node.get('file')
        if content_file_name is not None:
            content_file_path = os.path.join(self.xml_file_dir, content_file_name)
            if not os.path.exists(content_file_path):
                await self._download_asset_to_cache(content_file_name)
    else:
        for child in node:
            await self.process_xml_node(child)
```

**简化点**：Local 版在 `process_xml_node` 内额外套了 `file_lock`；Euler 版将锁下沉到 `_download_asset_to_cache`，避免双重加锁。`_download_asset_to_cache` 内部已有 `file_lock` + 存在性二次检查，保证多进程安全。

### 4.3 `process_xml_file`（公共，入口）

原样迁移 Local 逻辑（[orca_gym_local.py:723](../../../orca_gym/core/orca_gym_local.py#L723)）：

```python
async def process_xml_file(self, file_path: str) -> None:
    """解析 XML 文件，下载缺失的 mesh/hfield 资源。"""
    import xml.etree.ElementTree as ET
    with open(file_path, 'r') as f:
        xml_content = f.read()
    root = ET.fromstring(xml_content)
    await self.process_xml_node(root)
```

### 4.4 `_load_model_xml_online` 修改

在 XML 原子落盘后、`return` 前不需要新增调用——按 §3.2 的修正，`process_xml_file` 统一在 `load_model_xml` 公共入口的两个分支后调用。`_load_model_xml_online` 本身**不改**。

### 4.5 既有 `load_content_file` 的处理

Euler 现有 `load_content_file`（[orca_studio_bridge.py:496](../../../orca_gym/core/euler/orca_studio_bridge.py#L496)）是薄 gRPC 包装，已被 `OrcaGymEuler.load_content_file`（[orca_gym_euler.py:343](../../../orca_gym/core/euler/orca_gym_euler.py#L343)）委托暴露。**保留不动**，避免影响既有调用方。新增的 `_download_asset_to_cache` 不复用它（因为它不捕获响应、不落盘），而是独立实现 gRPC + 落盘。

> **可选重构（非必须）**：若希望消除重复，可将 `load_content_file` 重构为返回 `bytes`，`_download_asset_to_cache` 调用它再落盘。但这是代码整洁度优化，不在本次修复必做范围。

---

## 5. 测试验证

### 5.1 单元测试（CPU，可在 sandbox 内运行）

**文件**：`tests/orca_gym/core/euler/test_orca_studio_bridge_mesh_download.py`

| 用例 | 验证点 |
|------|--------|
| `test_process_xml_node_downloads_missing_mesh` | mock `_download_asset_to_cache`，构造含 `mesh file="a.stl"` 的 XML 节点，缓存目录无 `a.stl`，断言被调用 |
| `test_process_xml_node_skips_existing_mesh` | 缓存目录已存在 `a.stl`，断言 `_download_asset_to_cache` 不被调用 |
| `test_process_xml_node_recurses_children` | `asset` 节点下嵌套 `mesh`，断言递归命中 |
| `test_process_xml_file_parses_and_dispatches` | 写入临时 XML 文件，断言 `process_xml_node` 被以 root 调用 |
| `test_download_asset_offline_raises` | `_stub=None`，断言抛 `FileNotFoundError` 且消息含 `xml_file_dir` |
| `test_download_asset_writes_atomically` | mock gRPC stub 返回固定 bytes，断言文件原子落盘到 `xml_file_dir`，临时文件被清理 |
| `test_download_asset_concurrent_safe` | 同一文件并发调用，第二次因 `file_lock` + 存在性检查跳过下载（mock gRPC 仅被调用一次） |

### 5.2 集成测试（需 OrcaStudio 在线，标 GPU/sandbox 外）

**前置**：OrcaStudio 服务端加载 G1 场景，清空 `~/.orcagym/tmp/`。

| 用例 | 验证点 |
|------|--------|
| `test_online_load_g1_downloads_stl` | `OrcaGymEulerEnv` 在线实例化，断言 `~/.orcagym/tmp/` 下出现 G1 的 STL 文件，`init_simulation` 不抛错 |
| `test_online_load_idempotent` | 连续实例化两次，第二次 gRPC `LoadContentFile` 不被重复调用（缓存命中） |

### 5.3 回归测试

- 离线模式（`skip_grpc_load=True` + 本地 XML + 本地 mesh）：行为不变，`process_xml_file` 检查通过不下载。
- 既有 `load_content_file` 调用方：行为不变（方法签名未改）。

---

## 6. 风险与影响范围

### 6.1 影响范围

| 组件 | 是否改动 | 说明 |
|------|---------|------|
| `orca_studio_bridge.py` | ✅ 改 | 新增 3 方法、修改 `load_model_xml` |
| `orca_gym_euler.py` | ❌ 不改 | `load_model_xml` 委托不变 |
| `orca_gym_euler_env.py` | ❌ 不改 | `initialize_simulation` 编排不变 |
| `mujoco_sim_core.py` | ❌ 不改 | `init_simulation` 不变 |
| example 代码 | ❌ 不改 | 公共 API 调用链不变 |
| 架构文档 | 可选增补 | §5.4 方法清单补列 `process_xml_file`/`process_xml_node`（非契约修改） |

### 6.2 风险

| 风险 | 等级 | 缓解 |
|------|------|------|
| gRPC `LoadContentFile` 响应字段名与 Local 不一致 | 低 | Local 已用 `response.content`（[orca_gym_local.py:411](../../../orca_gym/core/orca_gym_local.py#L411)），Euler proto 共用同一 `mjc_message_pb2`，字段一致 |
| mesh `file` 属性含子目录（如 `g1/foot.stl`） | 中 | `os.path.join(xml_file_dir, file_name)` 自动处理；`_download_asset_to_cache` 内 `os.makedirs(dirname, exist_ok=True)` 确保子目录存在 |
| 大型场景 mesh 众多，串行下载慢 | 低 | 可后续优化为并发下载（`asyncio.gather`），本次修复优先正确性 |
| 在线模式首次加载因下载变慢 | 低 | 属预期行为，优于直接失败；幂等缓存保证后续加载快 |
| `file_lock` 跨进程语义在 Windows 的行为 | 低 | Local 体系已用同一 `file_lock`（`orca_gym.utils.dir_utils`）长期运行验证 |

### 6.3 不解决的问题（显式排除）

- **Euler 体系的 `load_content_file` 薄包装设计争议**：本方案不重构既有 `load_content_file`，仅新增 `_download_asset_to_cache` 承担落盘。是否统一两者为单一方法属独立重构议题。
- **mesh 并发下载优化**：本次修复保持串行，与 Local 行为一致。
- **Studio 端主动推送 mesh**：本方案仍走客户端拉取（`LoadContentFile`），不改服务端协议。

---

## 7. 实施清单

- [ ] `orca_studio_bridge.py` 新增 `_download_asset_to_cache` / `process_xml_node` / `process_xml_file`
- [ ] `orca_studio_bridge.py` 修改 `load_model_xml`：两分支后统一调用 `process_xml_file`
- [ ] 新增单元测试 `tests/orca_gym/core/euler/test_orca_studio_bridge_mesh_download.py`
- [ ] 执行 `ruff check --select SLF001 orca_gym/` 零报警
- [ ] 在线集成测试（需 OrcaStudio + G1 场景）
- [ ] 可选：架构文档 §5.4 方法清单增补 `process_xml_file` / `process_xml_node`
- [ ] 用户确认后合并
