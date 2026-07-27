# 🐋 什么是 OrcaGym

OrcaGym 是一个**开源、云原生的机器人仿真平台**，提供与 OpenAI Gym / Gymnasium 接口完全兼容的机器人仿真环境。

## 一句话概括

**OrcaGym = Gymnasium API + MuJoCo 物理引擎 + 分布式通信 + OrcaStudio/OrcaLab 云平台**

## 核心定位

传统机器人仿真方案往往在保真度和计算效率之间面临取舍。OrcaGym 通过以下方式弥合这一差距：

1. **标准化接口**：完全兼容 Gymnasium API，零成本迁移现有 RL 算法
2. **多物理后端**：通过 OrcaStudio/OrcaLab 集成 MuJoCo、PhysX、ODE
3. **云原生架构**：实现本地/远程混合部署
4. **逼真渲染**：光线追踪为视觉 RL 任务提供高质量观察

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

### ⚡ 多物理后端

| 后端 | 特点 | 适用场景 |
|------|------|----------|
| **MuJoCo** | 高精度刚体动力学 | 足式机器人、机械臂操控 |
| **PhysX** | GPU 加速、大规模并行 | 群体仿真、复杂场景 |
| **ODE** | 开源通用 | 快速原型、教育用途 |

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
