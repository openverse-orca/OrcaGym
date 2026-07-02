# OrcaGymEulerEnv 开发者速查手册

面向 `OrcaGymEulerEnv` 开发者的上手速查。配套架构契约见 `docs/design/architecture/orca_gym_euler_architecture.md` §6.8，可运行示例见 `OrcaPlayground/examples/euler/01~09` 课程。

本手册不展开实现细节——请通过 example 课程理解完整开发流程。

## 1. 三种使用模式

| 模式 | 适用 | 入口 | 参考课程 |
|------|------|------|----------|
| **A. 直接实例化** | 验证脚本、单进程交互、原型调试 | `env = MyEnv(...)` | 01/02/04~09 |
| **B. `make_env` 工厂** | RL 训练、SB3 集成、VecEnv 多进程 | `env = make_env(args); env = Monitor(env)` | 03 |
| **C. `gym.make` 注册** | 发布为可复用包、第三方调用 | `gym.make("MyEnv-v0")` | — |

选择原则：RL 训练用 B；其余用 A；发布用 C。三种方式的构造参数契约一致（架构 §6.8.1 E1）。

## 2. 最小骨架

### 2.1 直接实例化（模式 A）

```python
env = MyEnv(
    frame_skip=20,
    orcagym_addr="127.0.0.1:50051",
    agent_names=["g1"],
    time_step=0.001,
    model_xml_path="g1.xml",
    skip_grpc_load=False,      # True = 离线（无 Studio 远端）
    render_mode="human",
)
try:
    obs, info = env.reset()
    for _ in range(N):
        action = policy(obs)   # 或 np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()                # 必须调用：释放 gRPC / Studio 资源
```

### 2.2 RL 训练（模式 B，SB3 集成）

```python
def make_env(args, rank=0, seed=0):
    """返回 thunk（架构 §6.8.1 E2，SubprocVecEnv 要求 callable）。"""
    def _init():
        env = MyEnv(orcagym_addr=args.addr, skip_grpc_load=not args.online, ...)
        env.reset(seed=seed + rank)
        return env
    return _init

# 单 env 训练
env = make_env(args)()
env = Monitor(env)
model = PPO("MlpPolicy", env=env, ...)
model.learn(total_timesteps=N)
env.close()

# VecEnv 并发训练（多核加速 rollout）
env = SubprocVecEnv([make_env(args, rank=i, seed=0) for i in range(n_envs)])
env = VecMonitor(env)
model = PPO("MlpPolicy", env=env, ...)
model.learn(total_timesteps=N)
env.close()
```

完整训练/评估流程见 `examples/euler/03_rl_ppo/train_ppo.py`。

## 3. 生命周期要点

| 阶段 | API | 说明 |
|------|-----|------|
| 实例化 | `MyEnv(...)` | 构造参数显式传入；`skip_grpc_load=True` 可离线跑（无 Studio） |
| 重置 | `env.reset()` | 返回 `(obs, info)`；子类只需复写 `reset_model()` |
| 步进 | `env.step(action)` | 返回 Gymnasium 五元组；**唯一对外步进入口**（架构 §6.4 S5） |
| 渲染 | `env.render()` | 由 `render_mode` 控制 |
| 关闭 | `env.close()` | **必须显式调用**（推荐 `try/finally`），释放 gRPC/视频资源 |

关键：`step` 内部通过 `do_simulation` 编排物理步进；PD 控制在 `step` 内以 `frame_skip=1` 闭环实现（架构 §6.4 S6）。**不要**在主循环直接调 `do_simulation`。

## 4. 常见陷阱

- **忘记 `close()`**：gRPC channel 与 Studio 视频编码器不会自动释放，导致端口/文件泄漏。用 `try/finally`。
- **绕过 `step` 步进**：直接调 `do_simulation` 作为主路径违反 §6.4 S5，丢失 PD 闭环/obs/reward/truncated 语义。
- **穿墙访问**：`env._gym._sim._mjData` 违反封装（架构 §7），用 `env.data.qpos` / `env.query_*()`。
- **`reset` 复写错误**：`reset()` 由 Mixin 提供，子类复写 `reset_model()`，不要复写 `reset()`。
- **VecEnv 非 new 实例**：`make_env` 必须每次返回新实例（架构 §6.8.1 E2），否则多进程共享状态。

## 5. 开发一个新 Env 的流程

1. 继承 `OrcaGymEulerEnv`（或 `G1BaseEnv` 若复用 run_lesson 框架）
2. 复写三个 Gymnasium hook：`step` / `reset_model` / `_get_obs`（架构 §5.1）
3. 如需 PD 控制：复写 `_pd_controller` hook（架构 §6.4 S6），不要复写 `do_simulation`
4. 如需 RL：定义 `observation_space` / `action_space` + 复写 `_compute_reward` / `_is_terminated`
5. 提供 `make_env` 工厂（RL 场景）或直接实例化（验证场景）
6. `ruff check --select SLF001` 零报警方可提交

## 6. Example 课程索引

按难度递进，每个课程验证一组 API：

| 课程 | 主题 | 学习重点 |
|------|------|----------|
| 01_hello_euler | Hello Euler | 最小 env：`step`/`reset_model`/`_get_obs` 三件套 |
| 02_pose_control | 位姿控制 | 关节控制、`set_joint_qpos` 写入 |
| 03_rl_ppo | RL 训练 | `make_env` 工厂 + SB3 PPO + `evaluate_policy`（模式 B 标杆） |
| 04_query_api | 状态查询 | `query_*` / `data` 读取 + run_lesson 框架 |
| 05_force_apply | 外力施加 | `apply_body_force` / `clear_body_force` |
| 06_jacobian | 雅可比 | `data.body_xpos` / 动力学查询 |
| 07_locomotion | 行走控制 | `_pd_controller` 闭环 hook + ONNX 策略 |
| 08_video_capture | 视频截帧 | `begin_save_video` / `stop_save_video` |
| 09_body_manipulation | 体操作 | 等式约束无状态原语 + 交互式循环 |

建议从 01 开始逐课阅读，01–03 覆盖标准 Gymnasium 用法，04–09 覆盖 OrcaGym 特色 API。
