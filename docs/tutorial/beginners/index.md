# 🎓 新手入门

欢迎来到 OrcaGym 新手教程！本系列从**零**开始，用渐进的方式带你掌握 OrcaGym。

---

## 适合谁读？

- **刚接触 OrcaGym**，想从零开始上手的开发者
- **机器人仿真新手**，希望理解仿真环境的基本概念
- **有 Python 基础**，想学习机器人仿真的学生或工程师

## 前置知识

| 知识点 | 说明 |
|--------|------|
| Python 基础 | 熟悉函数、类、`import` |
| NumPy 基础 | 了解 `np.array` 的创建和基本操作 |
| 机器人学基础 | 了解"关节"、"末端执行器"等概念（可选，教程中会解释） |

> **不需要** RL（强化学习）背景知识！本教程聚焦于仿真本身。

---

## 学习路径

我们设计了一条**由浅入深、层层递进**的学习路径。每一步只引入一个新概念：

```
🔌 Hello World 了解最简仿真循环是什么样的
 │
 └── 🎬 场景搭建 学会往场景里放机器人、物体、灯光
 │
 └── 🏗️ 第一个环境 学会写自己的环境类
 │
 └── 📡 读取状态 学会查询关节角度、body 位姿
 │
 └── 🦾 让机器人动起来 理解 qpos/qvel，控制关节
 │
 └── 📷 相机与视觉 获取 RGB-D 相机图像
 │
 └── 🎮 简单控制器 编写 PD 控制器
 │
 └── 🏆 搭建一个任务 组合所有知识，完成到达任务
 │
 └── 🧠 PPO 训练 用 SB3 RL 训练控制器
```

| 章节 | 新概念 | 预计时间 |
|------|--------|----------|
| [🔌 Hello World](hello-world.md) | 离线模式、`do_simulation`、`data.qpos` | 5 分钟 |
| [🎬 场景搭建](scene-setup.md) | `OrcaGymScene`、`Actor`、资产摆放 | 15 分钟 |
| [🏗️ 第一个环境](your-first-env.md) | 继承 `OrcaGymEulerEnv`、实现 `step`/`reset_model`/`_get_obs` | 15 分钟 |
| [📡 读取状态](state-queries.md) | `query_joint_qpos`、`get_body_xpos_xmat_xquat`、传感器 | 15 分钟 |
| [🦾 让机器人动起来](move-a-joint.md) | `qpos`/`qvel`、`set_joint_qpos`、`do_simulation` | 20 分钟 |
| [📷 相机与视觉](camera-and-vision.md) | `CameraWrapper`、RGB-D 图像获取 | 15 分钟 |
| [🎮 简单控制器](simple-controller.md) | PD 控制器原理、参数调优 | 20 分钟 |
| [🏆 搭建一个任务](build-a-task.md) | 组合一切：感知→决策→控制，完成到达目标 | 30 分钟 |
| [🧠 PPO 训练](ppo-training.md) | SB3 PPO、奖励函数设计、离线训练 | 30 分钟 |

---

## 关键概念速览

### 仿真 = 模型 + 数据

OrcaGym 将仿真世界分为两部分：

| 概念 | 类型 | 比喻 | 例子 |
|------|------|------|------|
| `env.model` | `OrcaGymModel` | 机器人的**说明书**（不会变） | 有几个关节、每个关节叫什么名字 |
| `env.data` | `OrcaGymDataView` | 机器人的**当前状态**（每步都变） | 关节现在转了多少度、速度是多少 |

```python
# model — 静态，描述结构
print(env.model.nq) # 一共几个位置变量
print(env.model.nv) # 一共几个速度变量
print(env.model.nu) # 一共几个执行器（控制维度）

# data — 动态，反映当前状态（：OrcaGymDataView，零拷贝只读视图）
print(env.data.qpos) # 当前位置 → 每一步仿真后都会变
print(env.data.qvel) # 当前速度
print(env.data.time) # 仿真时间
```

### 仿真时间

```
time_step = 0.001 秒 ← 物理引擎每步的时间（很小，保证精度）
frame_skip = 20 ← 每次 step() 物理引擎走几步
dt = 0.001 × 20 = 0.02秒 ← 你的控制指令每隔多久更新一次（50Hz）
```

### 环境类层次

```
gymnasium.Env
 └── OrcaGymEulerEnv # 👈 推荐（当前主路径）
 ├── 组合 OrcaGymEuler 仿真核心
 ├── .data → OrcaGymDataView（完整状态零拷贝只读视图）
 ├── .model → OrcaGymModel（模型结构信息）
 ├── .sim_config → SimConfig（求解器配置）
 └── .ctrl → np.ndarray（当前控制输入）

 └── OrcaGymLocalEnv # 维护模式，不推荐新项目使用
```

> **推荐**：`OrcaGymEulerEnv` 是当前推荐的新入口。 `OrcaGymLocalEnv` 处于维护模式，逐步废弃。
> 可运行的完整示例见 [OrcaPlayground examples/euler/](https://github.com/OrcaGym/OrcaPlayground)。

---

## 下一步

从 [🔌 Hello World](hello-world.md) 开始，5 分钟跑通你的第一个仿真！
