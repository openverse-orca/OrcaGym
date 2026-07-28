# 🔌 Hello World — 跑通第一个仿真

目标：5 分钟内跑通一个最简仿真循环（无需 OrcaStudio，离线模式）。

---

## 前提

- 已安装 OrcaGym（能 `from orca_gym.environment.euler import OrcaGymEulerEnv`）
- 有一个简单的 MuJoCo XML 场景文件（本节使用内置示例）

---

## 完整代码

下面是一个**可以直接运行**的最简示例。把它保存为 `hello_orcagym.py`：

```python
"""
hello_orcagym.py — OrcaGym 最简示例

功能：创建环境 → 随机动作驱动 200 步 → 关闭
前提：不需要 OrcaStudio（离线模式，skip_grpc_load=True）
"""
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class HelloEnv(OrcaGymEulerEnv):
 """最简环境：用随机动作驱动仿真，打印状态。"""

 def __init__(self, model_xml_path, **kwargs):
 super().__init__(
 frame_skip=kwargs.pop("frame_skip", 5),
 orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
 agent_names=kwargs.pop("agent_names", ["agent0"]),
 time_step=kwargs.pop("time_step", 0.002),
 model_xml_path=model_xml_path,
 skip_grpc_load=True, # 离线模式，不需要 Studio
 **kwargs,
 )
 # 动作空间 = Box
 self.action_space = spaces.Box(
 low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
 )
 obs = self._get_obs()
 self.observation_space = spaces.Box(
 low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
 )

 def step(self, action):
 action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
 self.do_simulation(action, self.frame_skip)
 obs = self._get_obs()
 reward = 0.0
 terminated = False
 truncated = False
 info = {"time": float(self.data.time)}
 return obs, reward, terminated, truncated, info

 def reset_model(self):
 self.set_joint_qpos(self.init_qpos)
 self.set_joint_qvel(self.init_qvel)
 self.mj_forward()
 self._sync_view()
 return self._get_obs(), {}

 def _get_obs(self) -> np.ndarray:
 return self.data.qpos.copy().astype(np.float32)


# ============================================================
# 使用
# ============================================================
if __name__ == "__main__":
 # 场景 XML 路径（请替换为你本地的场景文件）
 SCENE_XML = "/path/to/your/scene.xml"

 print("=" * 60)
 print("Hello OrcaGym — 离线模式最简示例")
 print("=" * 60)

 # 1. 创建环境（离线模式，直接加载本地 XML）
 env = HelloEnv(model_xml_path=SCENE_XML, frame_skip=5, time_step=0.002)
 print(f"[1/4] 环境创建成功: nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")

 # 2. 验证状态访问
 print(f"[2/4] 状态访问: qpos.shape={env.data.qpos.shape}, time={env.data.time:.4f}")

 # 3. reset
 obs, info = env.reset()
 print(f"[3/4] reset 成功: obs.shape={obs.shape}")

 # 4. 步进循环（随机动作，200 步）
 total_reward = 0.0
 for step in range(200):
 action = env.action_space.sample()
 obs, reward, terminated, truncated, info = env.step(action)
 total_reward += reward
 if (step + 1) % 50 == 0:
 print(f"[4/4] step {step + 1}/200: obs={obs}, reward={reward:.4f}")

 print(f"[4/4] 步进完成: 总奖励={total_reward:.4f}")
 env.close()
 print("=" * 60)
 print("Hello OrcaGym 完成！")
```

运行：

```bash
python hello_orcagym.py
```

---

## 逐行解释

### 环境构造

```python
class HelloEnv(OrcaGymEulerEnv):
 def __init__(self, model_xml_path, **kwargs):
 super().__init__(
 frame_skip=5, # 每次 step() 推 5 个物理步
 orcagym_addr="localhost:50051", # gRPC 地址（离线模式不需要）
 agent_names=["agent0"], # agent 名称列表
 time_step=0.002, # 每个物理步 0.002 秒
 model_xml_path=model_xml_path, # 本地 MuJoCo XML 场景路径
 skip_grpc_load=True, # True = 离线模式
 )
```

关键参数：
- `model_xml_path`：本地 MuJoCo XML 场景文件的路径。离线模式下用此路径直接加载。
- `skip_grpc_load=True`：跳过 gRPC 连接，纯本地 MuJoCo 仿真。
- `frame_skip=5`：每次 `step()` 调用时物理引擎推 5 步。控制频率 = 1 / (time_step × frame_skip) = 100 Hz。
- `time_step=0.002`：每个物理步 0.002 秒。

### 需要实现的三个抽象方法

```python
def step(self, action): # 执行一步仿真 → 返回 (obs, reward, terminated, truncated, info)
def reset_model(self): # 重置到初始状态 → 返回 (obs, info)
def _get_obs(self) -> dict | np.ndarray: # 收集观测数据
```

### 核心循环

```python
obs, info = env.reset() # 回到初始状态
obs, reward, terminated, truncated, info = env.step(action) # 前进一步
```

| 变量 | 含义 | 类型 |
|------|------|------|
| `obs` | 观测数据（由 `_get_obs()` 返回） | `np.ndarray` 或 `dict` |
| `reward` | 奖励 | `float` |
| `terminated` | 任务是否完成/失败（如摔倒） | `bool` |
| `truncated` | 是否超时截断（达到 max_episode_steps） | `bool` |
| `info` | 额外调试信息 | `dict` |

### 离线 vs 在线模式

| 模式 | `skip_grpc_load` | 需要 Studio | 适用场景 |
|------|------------------|------------|----------|
| 离线（推荐入门） | `True` | 否 | 训练、快速测试、本地开发 |
| 在线 | `False`（默认） | 是 | 渲染可视化、人工观察、视频录制 |

---

## 常见问题

### `FileNotFoundError: model XML not found`

**原因**：`model_xml_path` 指向的 XML 文件不存在。

**解决**：确认文件路径正确。你可以从 [OrcaPlayground](https://github.com/OrcaGym/OrcaPlayground) 获取示例场景文件。

### `ModuleNotFoundError: No module named 'orca_gym'`

**原因**：OrcaGym 未安装。

**解决**：按照 [安装指南](../getting-started/installation.md) 安装。

### 离线模式下 `env.render()` 不起作用？

离线模式下 `render()` 是 no-op（因为没连 Studio）。如需可视化，请使用在线模式（`skip_grpc_load=False`）并启动 OrcaStudio。

---

## 下一步

你已经跑通了最简仿真！接下来学习如何**往场景里放东西**：[🎬 场景搭建](scene-setup.md)。
