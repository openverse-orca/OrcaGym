# OrcaGym API Manual: `orca_gym/core`

> **📖 这是什么文档？**  
> 这是 `orca_gym/core` 模块的完整 API 参考手册，采用“索引 + 详情”的版式设计，便于快速查找和深入学习。

## 📚 文档说明

### 文档特点

- **索引优先**：每个模块和类都提供索引表格，方便快速浏览和定位
- **详情展开**：点击或展开详情部分，查看完整的方法签名、参数说明和使用示例
- **面向本地环境**：本手册主要覆盖本地环境实现，远程环境相关内容已省略
- **仅公开接口**：只列出 public 符号（不以下划线开头），聚焦实际可用的 API

### 如何使用本手册

1. **快速查找**：使用下方的模块索引表格，找到你需要的模块
2. **浏览类列表**：进入模块后，先看“Classes（索引）”表格，了解有哪些类
3. **查看方法**：每个类都有“方法索引”表格，快速了解可用方法
4. **深入阅读**：展开“方法详情”部分，查看完整的签名、参数说明和使用示例

### 相关文档

- **快速概览**：查看 [`API_REFERENCE.md`](../API_REFERENCE.md) 了解整体架构和典型调用链
- **详细参考**：查看 [`api_detail/core.md`](../api_detail/core.md) 获取自动生成的完整 API 签名列表
- **Environment 模块**：查看 [`api_manual/environment.md`](environment.md) 了解环境层接口

---

## 📦 Modules（索引）

快速浏览所有模块，点击模块名跳转到详细内容：

| Module | 说明 |
| --- | --- |
| [`orca_gym/core/orca_gym_local.py`](#orca_gymcoreorca_gym_localpy) | **本地 MuJoCo Backend**：本地 MuJoCo 仿真引擎的核心实现（最常用） |
| [`orca_gym/core/orca_gym_model.py`](#orca_gymcoreorca_gym_modelpy) | **模型信息**：静态模型信息封装，提供 body/joint/actuator 等查询接口 |
| [`orca_gym/core/orca_gym_data.py`](#orca_gymcoreorca_gym_datapy) | **仿真数据**：动态仿真状态封装，包含 qpos/qvel/qacc 等状态 |
| [`orca_gym/core/orca_gym_opt_config.py`](#orca_gymcoreorca_gym_opt_configpy) | **优化配置**：MuJoCo 仿真器优化参数配置（timestep/solver 等） |

---

## `orca_gym/core/orca_gym_opt_config.py`

> OrcaGymOptConfig - MuJoCo 仿真器优化配置

### Module docstring

OrcaGymOptConfig - MuJoCo 仿真器优化配置

本模块提供 MuJoCo 仿真器优化参数的封装类，用于配置物理仿真器的各种参数。
这些参数影响仿真的精度、稳定性和性能。

使用场景:
    - 在环境初始化时从服务器获取配置
    - 通过 env.gym.opt 访问配置对象
    - 调整物理仿真精度和性能平衡

典型用法:
    ```python
    # 配置通过 OrcaGymLocal 的初始化自动获取
    env = OrcaGymLocalEnv(...)
    # 访问配置
    timestep = env.gym.opt.timestep
    gravity = env.gym.opt.gravity
    solver = env.gym.opt.solver
    ```

### Classes（索引）

| Class | 摘要 |
| --- | --- |
| `OrcaGymOptConfig` | MuJoCo 仿真器优化配置容器 |

### Classes（详情）

#### `class OrcaGymOptConfig`

> MuJoCo 仿真器优化配置容器

