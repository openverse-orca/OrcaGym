# 🏗️ 第一个环境 — 写自己的环境类

上一节我们用最简环境跑通了仿真循环。这一节，你将学会如何**写一个完整的环境类**来控制仿真。

## 为什么要写环境类？

要控制仿真（物理步进、读取状态），你需要继承环境基类。

**推荐使用 `OrcaGymEulerEnv`**（Euler 新主路径）， `OrcaGymLocalEnv` 处于维护模式。

一个环境类 = 场景的"驱动程序"：

```
MuJoCo XML → OrcaGymEulerEnv → 驱动仿真（循环运行）
```

## 最小环境骨架（，推荐）

环境类需要实现 **3 个核心方法**：

```
__init__() — 初始化（设置 action_space、observation_space）
step(action) — 执行一步仿真，返回 (obs, reward, terminated, truncated, info)
reset_model() — 重置到初始状态
_get_obs() — 收集观测数据
```

> **与 RL 环境的关键区别**：`step()` 返回 5 元组（Gymnasium 标准），不是老版的 4 元组。
> `terminated` 表示任务完成/失败（如机器人摔倒），`truncated` 表示超时截断（达到 max_episode_steps）。

下面是一个**可运行**的完整环境（离线模式，无需 Studio）：

```python
"""
my_first_env.py — 一个最小的自定义环境
基于 [OrcaPlayground examples/euler/01_hello_euler/](https://github.com/OrcaGym/OrcaPlayground) 的简化版
"""

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class MyFirstEnv(OrcaGymEulerEnv):
 """最简环境：观测 = 关节位置 + 速度，动作 = 力矩控制，奖励 = 0"""

 metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

 def __init__(self, model_xml_path, **kwargs):
 # ── 父类初始化（自主编排生命周期）──
 super().__init__(
 frame_skip=kwargs.pop("frame_skip", 5),
 orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
 agent_names=kwargs.pop("agent_names", ["agent0"]),
 time_step=kwargs.pop("time_step", 0.002),
 model_xml_path=model_xml_path,
 skip_grpc_load=kwargs.pop("skip_grpc_load", True), # 默认离线
 **kwargs,
 )

 # ── 动作空间 ──
 self.action_space = spaces.Box(
 low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
 )

 # ── 观测空间 ──
 obs_sample = self._get_obs()
 self.observation_space = spaces.Box(
 low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32
 )

 # ── 观测获取 ───────────────────────────────────────
 def _get_obs(self) -> np.ndarray:
 """返回当前观测。零拷贝直接读 data。"""
 return np.concatenate([
 self.data.qpos.copy(),
 self.data.qvel.copy(),
 ]).astype(np.float32)

 # ── 仿真步进 ───────────────────────────────────────
 def step(self, action: np.ndarray):
 """
 执行一步仿真。
 action: 形状 (nu,)，每个值 ∈ [-1, 1]
 返回: (obs, reward, terminated, truncated, info) — Gymnasium 5 元组
 """
 action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)

 # 1. 执行仿真：do_simulation 内部自动同步 data
 self.do_simulation(action, self.frame_skip)

 # 2. 获取新观测
 obs = self._get_obs()

 # 3. 奖励 & 终止条件
 reward = 0.0 # 此处恒为 0（替换为你的奖励函数）
 terminated = False # 此处永不终止（替换为你的终止条件）
 truncated = False
 info = {"time": float(self.data.time)}

 return obs, reward, terminated, truncated, info

 # ── 重置 ───────────────────────────────────────────
 def reset_model(self):
 """回到初始状态。"""
 # 合规写法：通过 set_joint_qpos / set_joint_qvel 设置状态
 self.set_joint_qpos(self.init_qpos) # 恢复到 XML 中定义的初始 qpos
 self.set_joint_qvel(self.init_qvel) # 恢复到 XML 中定义的初始 qvel
 self.mj_forward() # 更新派生量（body 位姿、传感器等）
 self._sync_view() # 同步到 DataView（env.data）
 return self._get_obs(), {}


# ============================================================
# 使用
# ============================================================
if __name__ == "__main__":
 SCENE_XML = "/path/to/your/scene.xml"

 env = MyFirstEnv(model_xml_path=SCENE_XML)
 obs, _ = env.reset()
 print(f"nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")
 print(f"obs.shape={obs.shape}")

 for i in range(10):
 action = env.action_space.sample() # 随机动作
 obs, reward, terminated, truncated, _ = env.step(action)
 print(f" Step {i}: reward={reward:.3f}, time={_:.4f}s")

 env.close()
```

## 核心概念拆解

### `do_simulation` — 一步到位的仿真步进

```python
self.do_simulation(ctrl, self.frame_skip)
```

这一行在 中等价于：

```python
# 内部：_step_with_coupling(ctrl, n_frames, dt)
# → _sim.set_ctrl(ctrl) + _sim.step(n_frames)
# 然后：_sync_view()
# data 自动同步为最新状态
```

> **关键优势**：`do_simulation()` 返回后 `self.data` 已自动更新，无需手动 `update_data()`。

### `self.data` — 完整状态只读视图

中 `self.data` 是 `OrcaGymDataView`，提供零拷贝只读视图：

| 属性 | 含义 | 形状 |
|------|------|------|
| `self.data.qpos` | 广义位置 | `(nq,)` |
| `self.data.qvel` | 广义速度 | `(nv,)` |
| `self.data.qacc` | 广义加速度 | `(nv,)` |
| `self.data.time` | 仿真时间 | 标量 |
| `self.data.xfrc_applied` | 外力（只读） | `(nbody, 6)` |

> ⚠️ `self.data.qpos` 是零拷贝视图，直接读即可。若需保存历史值，调用 `.copy()`。

### `self.model` — 模型结构查询

| 属性/方法 | 含义 |
|-----------|------|
| `self.model.nq` | 广义坐标维度 |
| `self.model.nv` | 广义速度维度 |
| `self.model.nu` | 执行器数量 |
| `self.model.body_name2id(name)` | Body 名称 → ID |
| `self.model.joint_name2id(name)` | Joint 名称 → ID |
| `self.model.get_joint_dict()` | 所有关节信息字典 |
| `self.model.get_geom_dict()` | 所有几何体信息字典 |

### 状态写入

修改状态后**必须调用 `mj_forward()`**，否则派生量（body 位姿、传感器等）不会更新：

```python
# ✅ 正确写法
qpos = self.data.qpos.copy()
qpos[0] = 0.5 # 修改关节 0 的角度
self.set_joint_qpos(qpos) # 合规写入
self.mj_forward() # ← 必须！更新派生量
self._sync_view() # 同步到 DataView

# ❌ 错误：直接写 data.qpos（只读视图）
# self.data.qpos[0] = 0.5 # 违反封装契约

# ❌ 错误：缺少 mj_forward
# self.set_joint_qpos(qpos)
# # 此时读 body_xpos 会是旧值
```

### `reset_model` 的标准范式

```python
def reset_model(self):
 qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
 qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
 self.set_joint_qpos(qpos) # 合规写入
 self.set_joint_qvel(qvel) # 合规写入
 self.mj_forward() # 更新派生量
 self._sync_view() # 同步 DataView
 return self._get_obs(), {}
```

- `self.init_qpos` / `self.init_qvel`：父类在 `initialize_simulation()` 后缓存的初始状态
- `self.np_random`：父类通过 `set_seed_value()` 创建的随机数生成器

### 环境生命周期

```
MyFirstEnv(model_xml_path=..., skip_grpc_load=True)
 └── OrcaGymEulerEnv.__init__()
 ├── initialize_grpc() # 离线模式 skip
 ├── pause_simulation()
 ├── set_time_step(time_step)
 ├── initialize_simulation() # 加载 model_xml → 创建 MuJoCo 实例
 ├── reset_simulation() # reset_data + sync_to_view
 └── init_qpos_qvel() # 缓存 init_qpos / init_qvel

env.reset() [来自 OrcaGymEnvMixin]
 ├── reset_simulation() → 恢复初始状态
 └── reset_model() → 你的自定义重置逻辑

env.step(action) ← 重复 N 次
 ├── do_simulation(ctrl, frame_skip) # 步进 + 自动 sync
 ├── _get_obs()
 └── 返回 (obs, reward, terminated, truncated, info)

env.close()
 └── 关闭 gRPC 通道（离线模式 no-op）
```

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `ValueError: Action dimension mismatch` | `action.shape` ≠ `(nu,)` | 检查 `action.reshape(env.model.nu)` |
| 观测数据"不对" | 没在 `mj_forward()` 后读 data | `reset_model` 中确认调用了 `mj_forward()` |
| 观测全是 NaN | 在 `mj_forward()` 前读了派生量 | 用 `do_simulation()` 代替手动操作 |
| `AttributeError: 'OrcaGymEulerEnv' object has no attribute 'gym'` | 用了 API | 没有 `env.gym`，用 `env.data` / `env.model` |
| `TypeError: step() returns 5 values` | 旧代码只解包 4 个值 | Gymnasium 标准：`obs, reward, terminated, truncated, info = env.step(action)` |

## 下一步

环境类写好了。接下来学习如何**读取更多状态信息**：[📡 读取状态](state-queries.md)。
