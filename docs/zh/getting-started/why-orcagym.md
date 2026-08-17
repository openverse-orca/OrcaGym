# 🧬 为什么选择 OrcaGym

在众多机器人仿真平台中，OrcaGym 的独特优势是什么？

## 与主流仿真平台对比

| 特性 | OrcaGym | Isaac Gym | MuJoCo (原生) | PyBullet | SAPIEN |
|------|---------|-----------|---------------|----------|--------|
| Gymnasium API | ✅ 完全兼容 | ❌ 自定义 VecEnv | 需手动封装 | ✅ | ❌ |
| 双物理后端 | ✅ MuJoCo (CPU) + Euler (GPU) | ❌ PhysX only | ❌ MuJoCo only | ❌ Bullet only | ❌ PhysX only |
| 分布式部署 | ✅ 原生支持 | ❌ 单机 | ❌ 单机 | ❌ 单机 | ❌ 单机 |
| 光线追踪 | ✅ 可选接入 | ❌ | ❌ | ❌ | ✅ |
| GPU 加速 | ✅ Euler 后端 | ✅ (原生) | ❌ (CPU) | ❌ (CPU) | ✅ |
| 多智能体 | ✅ 原生 | ⚠️ 需手动 | ⚠️ 需手动 | ⚠️ 需手动 | ✅ |
| 开源 | ✅ MIT | ✅ 非商业 | ✅ Apache 2.0 | ✅ | ✅ |

## 核心优势详解

### 1. 标准化的 RL 接口

OrcaGym 严格遵循 Gymnasium 规范，意味着：

- 现有 RL 算法库**无需修改**即可运行
- `env.step()`、`env.reset()` 等调用完全符合预期
- 支持 `Dict` 和 `Box` 两种观测空间

```python
# 与任何 Gymnasium 兼容的 RL 库一起使用
from stable_baselines3 import PPO

env = gym.make("YourEnv-v0", ...)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

### 2. 云原生分布式架构

与单机仿真器不同，OrcaGym 天然支持：

- **本地模式**：Python 进程内直接驱动 MuJoCo，适合开发调试
- **远程模式**：连接远程仿真服务，适合大规模训练
- **混合模式**：训练在远端，策略执行在本地

```
开发阶段：本地模式 → 快速迭代
部署阶段：远程模式 → 弹性扩展
```

### 3. 可扩展的可视化接入

OrcaGym 的状态视图（`env.data`）可被外部可视化工具消费，支持接入 OrcaStudio/OrcaLab 等外部平台进行场景可视化与调试（外部工具为可选项，不影响物理仿真）。

## 适用场景

| 场景 | 推荐理由 |
|------|----------|
| **足式机器人控制** | 高精度接触模型 + 标准 RL 接口 |
| **机械臂操控** | 逆运动学 + 等式约束 + Mocap 操控 |
| **多智能体协作** | 原生多智能体 + 异步环境 |
| **视觉 RL** | 光线追踪 + RGB-D 传感器 |
| **大规模分布式训练** | 多节点扩展 |
| **机器人教学** | 标准化 API + 丰富示例 |

## 局限性

- 核心包支持双后端：MuJoCo（CPU，已完整可用）与 Euler（GPU，预留接口，集成方案待定）
- 远程可视化/渲染依赖外部工具（如 OrcaStudio/OrcaLab）
- 社区尚在发展初期，第三方示例较少
