# 🔄 数据流

理解仿真数据如何在 OrcaGym 中流动，是正确使用环境的关键。

## 核心数据流

```
           RL Policy
               │
               ▼ action (nu,)
    ┌──────────────────────┐
    │  OrcaGymEulerEnv      │
    │   ┌────────────────┐  │
    │   │ do_simulation() │  │  ← 核心步进方法
    │   │   ├─ 设置控制输入  │  │
    │   │   ├─ 物理步进      │  │
    │   │   └─ 同步状态      │  │
    │   └───────┬────────┘  │
    │           ▼            │
    │   仿真状态 (data)       │
    │   qpos/qvel/time/...   │
    │           │            │
    │   ┌───────▼────────┐  │
    │   │ _get_obs()     │──┼──▶ obs
    │   │ compute_reward()│──┼──▶ reward
    │   └────────────────┘  │
    └──────────────────────┘
               │
               ▼ obs
           RL Policy
```

## step() 内部发生了什么

```python
# 当你调用 env.step(action) 时：
env.step(action)
  │
  ├─▶ do_simulation(ctrl, n_frames)
  │     ├─▶ 设置控制输入 ctrl
  │     └─▶ 执行 n_frames 步物理仿真
  │
  ├─▶ _get_obs()          # 构建观测
  ├─▶ compute_reward()    # 计算奖励
  │
  └─▶ 返回 (obs, reward, terminated, truncated, info)
```

## 状态同步规则

修改仿真状态后，必须执行某些操作以确保数据一致：

| 修改操作 | 必须执行 |
|----------|----------|
| `set_joint_qpos()` | `mj_forward()` |
| `set_joint_qvel()` | `mj_forward()` |
| `set_mocap_pos_and_quat()` | `mj_forward()` |

> **重要**：`do_simulation()` 返回后数据已自动同步。只有手动修改状态时，才需要手动调用 `mj_forward()`。

## 为什么修改 qpos 后需要 mj_forward？

MuJoCo 有很多"派生量"（body 位姿、传感器值、接触力等）需要 `mj_forward` 才能刷新。只改 `qpos` 不 `forward`，会导致：

- Body 位姿为 NaN 或旧值
- 传感器数据不正确
- 接触状态不一致

```python
# ✅ 正确
env.set_joint_qpos(qpos)
env.mj_forward()          # 刷新所有派生量

# 现在可以安全读取
body_pos = env.data.body_xpos("end_effector")

# ❌ 错误 —— 忘记 forward
env.set_joint_qpos(...)
body_pos = env.data.body_xpos("end_effector")  # 可能 NaN！
```

## 常见数据不同步问题

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| 读到旧状态 | 修改状态后没同步 | `do_simulation()` 已自动同步；手动操作后需要 `mj_forward()` |
| 位姿 NaN | 修改 qpos 后没调 `mj_forward()` | 修改状态后先 `mj_forward()` |
| 传感器值不变 | sensordata 依赖 forward | `mj_forward()` 后再读传感器 |
| 接触力异常 | contact 在 step 后刷新 | step 后立即读接触 |
