# 🐋 什么是 OrcaGym

OrcaGym 是一个**开源、云原生的机器人仿真平台**，提供与 OpenAI Gym / Gymnasium 接口完全兼容的机器人仿真环境。

## 一句话概括

**OrcaGym = Gymnasium API + 双物理后端（MuJoCo / Euler）+ 分布式通信**

## 核心定位

传统机器人仿真方案往往在保真度和计算效率之间面临取舍。OrcaGym 通过以下方式弥合这一差距：

1. **标准化接口**：完全兼容 Gymnasium API，零成本迁移现有 RL 算法
2. **双物理后端**：核心包支持 MuJoCo（开源、CPU、纯刚体）与 Euler（自研、GPU）两条互不隶属的路径，通过 `SimConfig.backend` 选择
3. **云原生架构**：实现本地/远程混合部署
4. **可扩展渲染**：可通过外部可视化工具（如 OrcaStudio/OrcaLab）接入光线追踪等高质量渲染

## 主要特性

### 🎮 Gymnasium API 兼容性

```python
import gymnasium as gym

env = gym.make("YourEnv-v0", frame_skip=5, orcagym_addr="localhost:50051")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

与 Stable-Baselines3、RLlib、CleanRL 等主流 RL 库无缝对接。

### ⚡ 双物理后端

OrcaGym 采用双后端架构，两条路径互不隶属，通过 `SimConfig.backend` 选择：

| 后端 | 定位 | 特点 | 适用场景 |
|------|------|------|----------|
| **MuJoCo** | 开源标准路径 | 高精度刚体动力学（CPU） | 足式机器人、机械臂操控、快速原型 |
| **Euler** | Orca 团队自研路径 | GPU 加速 | 大规模并行训练、高保真仿真 |

> **当前实现状态**：MuJoCo 后端已完整可用；Euler 后端为预留接口（`_euler=None`），具体集成方案待定。

### 🌐 分布式部署

支持从本地开发到大规模远程训练的灵活部署：
- **本地模式**：Python 进程内直接运行 MuJoCo，适合开发调试
- **远程模式**：连接远程服务器进行物理计算和渲染，适合大规模训练

### 📷 传感器与感知

- IMU（加速度计、陀螺仪）
- 力/扭矩传感器
- RGB-D 相机
- 接触力传感器

### 🤖 多智能体支持

原生支持同构/异构多智能体场景，智能体间可独立或协作。

## 许可证

MIT License — 完全开源，可自由用于学术和商业用途。
