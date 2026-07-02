# OrcaGym Euler Gym API 对齐重构文档：修复 step/reset_model 接口断层

## 1. 文档定位

### 1.1 文档目标

本文针对 `OrcaGymEulerEnv` 在 Gymnasium 标准接口（`step` / `reset_model` / `_get_obs`）上的设计与示例缺陷，进行 P0~P2 级修复。

**重构背景**：架构文档已修订（见 `orca_gym_euler_architecture.md` §5.1 / §5.9 / §6.4 / §8.3 / §12.4.1），明确：
- `reset` 由 `OrcaGymEnvMixin` 编排，子类应复写 `reset_model` hook（对齐 Gym `MujocoEnv`）
- `do_simulation` 是 `step()` 内部调用的仿真原语，不是对外主入口
- `step()` 是 RL 训练与外部运行循环的唯一入口（§6.4 S5）
- Locomotion PD 控制应在 `step()` 内部以 `frame_skip=1` 多帧步进（§6.4 S6）

但**代码层面**仍存在三类断层，导致架构约束无法落地：

| 级别 | 问题 | 后果 |
|------|------|------|
| **P0** | example 4-9 的 `G1BaseEnv` 未实现 `step`，`run_lesson` 直接调 `do_simulation` | 环境无法接 SB3/RLlib 训练；示例示范反模式 |
| **P1** | EulerEnv 的 `step`/`reset_model`/`_get_obs` 仅有 `raise NotImplementedError`，无 docstring 模板 | 用户不知道如何复写 |
| **P2** | EulerEnv 类 docstring 未列出继承自 Mixin 的方法 | 新用户找不到 `reset` 在哪 |

### 1.2 上游约束

| 文档 | 约束范围 |
|------|---------|
| `docs/design/architecture/orca_gym_euler_architecture.md` | §5.1 EulerEnv 契约、§5.9 Mixin 归属、§6.4 步进契约 S1-S6、§8.3 步进模式、§12.4.1 骨架签名 |
| `AGENTS.md` | 规则 1（orca conda 环境）、规则 4（API 隔离强制，do_simulation 是公共步进 API） |

### 1.3 修订原则

1. **不破坏架构约束**：K1–K14 + M0-M7 机制在修复后仍须满足。
2. **不与 Gym 生态冲突**：保留 `reset_model` / `_get_obs` / `do_simulation` 命名（Gym `MujocoEnv` 十年公开 hook 约定），**不**改名为 `reset`/`get_obs`/`_do_simulation`。
3. **不禁止方式 B**：架构 §6.4 S3 已将方式 B 定位为"高级逃生通道，默认不推荐"，保留作为接触富优化等场景的逃生口，仅约束示例不得以方式 B 作为主路径。
4. **示例先行**：P0 的 example 修复必须先于代码模板落地，作为后续用户的参照样板。

---

## 2. 问题详述

### 2.1 P0：example 4-9 接口断层

**现状**（`OrcaPlayground/examples/euler/04_query_api/g1_base_env.py`）：
- `G1BaseEnv(OrcaGymEulerEnv)` 继承基类未复写 `step`（基类 `raise NotImplementedError`）
- `run_lesson`（g1_base_env.py:252-309）直接调用 `self.do_simulation(ctrl, self.frame_skip)`（g1_base_env.py:302），绕过 `step`

**问题**：
1. `G1BaseEnv` 无法接入 SB3/RLlib/CleanRL 训练（`step` 抛 `NotImplementedError`）
2. `run_lesson` 是 verifier 驱动的演示循环，语义上与 RL 训练循环不同，但当前直接调 `do_simulation` 让用户误以为这是标准步进方式
3. example 4-9 作为教学样例却在示范反模式，误导用户认为"调 `do_simulation` 就是步进环境"

**影响范围**：`OrcaPlayground/examples/euler/04_query_api/` 至 `09_*` 下所有继承 `G1BaseEnv` 的子类。

### 2.2 P1：EulerEnv hook 缺模板

**现状**（`orca_gym/environment/euler/orca_gym_euler_env.py:1383-1395`）：
```python
def step(self, action) -> tuple[dict, dict]:
    raise NotImplementedError("step 待子类实现")

def reset_model(self) -> tuple[dict, dict]:
    raise NotImplementedError("reset_model 待子类实现")

def _get_obs(self) -> dict:
    raise NotImplementedError("_get_obs 待子类实现")
```

**问题**：三个 hook 仅有 `raise`，无 docstring。用户不知道：
- `step` 内部应按什么顺序调 `do_simulation` → `_get_obs` → 组织五元组
- `reset_model` 应调用 `set_joint_qpos`/`set_joint_qvel`/`mj_forward`/`_sync_view`
- `_get_obs` 是 `step` 与 `reset_model` 共用的观测 hook

**注**：架构文档 §12.4.1 已补充这三个 hook 的 docstring 模板（本次架构修订完成），但代码文件 `orca_gym_euler_env.py` 尚未同步落地。

### 2.3 P2：EulerEnv docstring 未列继承方法

**现状**：`OrcaGymEulerEnv` 类 docstring 仅列出"使用契约"和"禁止"，未列出从 `OrcaGymEnvMixin` 继承的公共方法。

**问题**：新用户读 EulerEnv 时只看到一堆 `NotImplementedError`，找不到 `reset` 的实现位置（`reset` 在 Mixin），造成论述 1 的"继承链错位"困惑。

---

## 3. 修复方案

### 3.1 P0 修复：G1BaseEnv 补 step + run_lesson 改用 step

#### 3.1.1 G1BaseEnv 补标准 step 实现

为 `G1BaseEnv` 增加标准 `step` 实现，使其可被 SB3 训练。由于 G1 为 locomotion 机器人，采用 §6.4 S6 的 PD 控制模式：

```python
# g1_base_env.py
class G1BaseEnv(OrcaGymEulerEnv):
    MAX_EPISODE_STEPS = 1000  # 由子类按需覆写

    def step(self, action: np.ndarray):
        """标准 Gymnasium step 接口，内部组织 PD 控制循环。

        遵循架构 §6.4 S6：以 frame_skip=1 多帧 do_simulation 实现精细 PD 控制。
        """
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        target = self._action_to_target(action)

        # PD 控制内循环：精细步进
        for _ in range(self.frame_skip):
            ctrl = self._pd_controller(self.data.qpos, self.data.qvel, target)
            self.do_simulation(ctrl, 1)   # frame_skip=1，含 Euler 耦合

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated = self._is_terminated(obs)
        self._step_count += 1
        truncated = self._step_count >= self.MAX_EPISODE_STEPS
        info: dict[str, Any] = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info
```

**子类需复写的 hook**（保留为抽象方法或提供默认实现）：
- `_action_to_target(action)` — 动作空间到 PD 目标的映射
- `_pd_controller(qpos, qvel, target)` — PD 控制律
- `_compute_reward(obs, action)` — 奖励函数
- `_is_terminated(obs)` — 终止条件
- `reset_model()` — 重置（G1 已有则保留）
- `_get_obs()` — 观测（G1 已有则保留）

> **设计权衡**：`step` 实现在 `G1BaseEnv` 而非 `OrcaGymEulerEnv` 基类。理由：基类不知道控制律、奖励、终止条件，保留 `raise NotImplementedError` 符合 Gym `MujocoEnv` 抽象基类约定。`G1BaseEnv` 作为"locomotion 机器人基类"提供 PD 模式样板，子类只需复写 hook。

#### 3.1.2 run_lesson 改用 step

将 `run_lesson` 内部的 `self.do_simulation(ctrl, self.frame_skip)` 改为 `self.step(ctrl)`：

```python
# g1_base_env.py — run_lesson 内部
# 修改前
self.do_simulation(ctrl, self.frame_skip)

# 修改后
obs, reward, terminated, truncated, info = self.step(ctrl)
```

**注意**：`run_lesson` 的 verifier 逻辑可能需要从 `step` 返回的 `obs`/`info` 中读取状态，而非直接读 `self.data`。需检查 verifier 的取数路径，必要时把验证所需字段补入 `info`。

#### 3.1.3 子类兼容性检查

检查 `examples/euler/05_*` ~ `09_*` 下继承 `G1BaseEnv` 的子类：
- 若子类已自行实现 `step`，需确认是否与 `G1BaseEnv.step` 的 PD 模式冲突，必要时重构为复写 hook（`_pd_controller` 等）
- 若子类依赖 `run_lesson` 直接调 `do_simulation` 的旧行为，需同步改为读 `step` 返回值

### 3.2 P1 修复：EulerEnv hook 落地 docstring 模板

同步架构文档 §12.4.1 的 docstring 模板到代码文件 `orca_gym/environment/euler/orca_gym_euler_env.py`：

```python
# orca_gym_euler_env.py
def step(self, action) -> tuple:
    """Gymnasium 标准步进接口（子类必须复写）。

    标准实现模板:
        self.do_simulation(action, self.frame_skip)   # 或 PD 循环见架构 §6.4 S6
        obs = self._get_obs()
        reward = ...
        terminated = ...
        truncated = self._step_count >= self.MAX_EPISODE_STEPS
        info = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info

    禁止:
        不要在外部运行循环里绕过 step() 直接调 do_simulation（架构 §6.4 S5）。
        不要复写 do_simulation 作为步进主路径，应在 step() 内调用它。
    """
    raise NotImplementedError("step 待子类实现")

def reset_model(self) -> tuple[dict, dict]:
    """Gymnasium MuJoCo 标准 hook（子类必须复写，由 reset() 调用）。

    标准实现模板:
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
        self.set_joint_qpos(qpos)
        self.set_joint_qvel(qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}

    说明:
        - 这是 Gym MujocoEnv 十年公开 hook 约定，不要直接复写 reset()。
        - reset() 由 OrcaGymEnvMixin 编排（seed + reset_simulation + reset_model + render）。
    """
    raise NotImplementedError("reset_model 待子类实现")

def _get_obs(self) -> dict:
    """Gymnasium MuJoCo 标准 hook（子类必须复写，step 与 reset_model 共用）。

    标准实现模板:
        theta = float(self.data.qpos[0])
        theta_dot = float(self.data.qvel[0])
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    说明:
        - `_` 前缀表示 protected（类族内部），子类复写是 Python 常规操作。
        - 不要改名为 get_obs，保持与 Gym MujocoEnv 命名一致。
    """
    raise NotImplementedError("_get_obs 待子类实现")
```

### 3.3 P2 修复：EulerEnv docstring 列出继承方法

同步架构文档 §5.1 的设计契约到 `OrcaGymEulerEnv` 类 docstring：

```python
class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env):
    """OrcaGym Euler 双引擎环境。

    使用契约:
        读取状态:   env.data.qpos / env.data.body_xpos(name) / env.query_*()
        写入状态:   env.set_joint_qpos() / env.apply_body_force()
        仿真步进:   env.do_simulation(ctrl, n_frames)  # 在 step() 内部调用
        求解器配置: env.sim_config.timestep = 0.002

    继承自 OrcaGymEnvMixin 的公共方法（无需子类复写）:
        reset(seed, options)        — Gymnasium 标准接口，编排 reset_simulation + reset_model + render
        set_seed_value(seed)        — 设置随机数种子
        generate_action_space(bounds)
        generate_observation_space(obs)
        body/joint/actuator/site/mocap/sensor(name) — 名称空间解析

    子类应复写的 Gymnasium MuJoCo 标准 hook（与 Gym MujocoEnv 对齐）:
        step(action)               — 必须复写，内部调用 do_simulation，组织 obs/reward/terminated/truncated/info
        reset_model()              — 必须复写，重置 qpos/qvel，返回 (obs, info)
        _get_obs()                 — 必须复写，返回观测（step 与 reset_model 共用）

    禁止:
        不要访问 env._gym._sim._mjData 或任何内部 MuJoCo 对象。
        env.gym/env.stub/env.channel 不存在，直接继承 gym.Env 不创建这些属性。
        缺少功能时，扩展本类的公共方法。
        不要绕过 step() 在外部循环里直接调用 do_simulation 作为主步进路径（架构 §6.4 S5）。
    """
```

### 3.4 P1 补充：Locomotion PD 控制样例

在 `OrcaPlayground/examples/euler/` 下新增一个 locomotion PD 控制样例，示范 §6.4 S6 的标准模式。该样例应：
- 继承 `G1BaseEnv`（复用其 `step` 的 PD 循环骨架）
- 仅复写 `_action_to_target` / `_pd_controller` / `_compute_reward` / `_is_terminated` / `reset_model` / `_get_obs`
- 可被 SB3 直接训练（`gym.make` + `PPO`）

> **说明**：具体样例代码随 P0 的 `G1BaseEnv` 修复一并落地，本设计文档不预先写死，避免与实际 G1 PD 控制律实现脱节。落地时参照 `examples/euler/01_hello_euler/simple_env.py` 的 Pendulum 范式，扩展为多自由度 PD。

---

## 4. 实施步骤

### 步骤 1：EulerEnv docstring + hook 模板落地（P1 + P2）

**文件**：`orca_gym/environment/euler/orca_gym_euler_env.py`

1. 更新 `OrcaGymEulerEnv` 类 docstring（§3.3）
2. 为 `step` / `reset_model` / `_get_obs` 补 docstring 模板（§3.2）
3. 保持三个方法仍 `raise NotImplementedError`（不提供默认实现，符合抽象基类约定）

**验收**：
- docstring 与架构文档 §5.1 / §12.4.1 一致
- `ruff check --select SLF001 orca_gym/` 零报警
- 骨架验收测试（§12.5）仍全部通过

### 步骤 2：G1BaseEnv 补 step + run_lesson 改用 step（P0）

**文件**：`OrcaPlayground/examples/euler/04_query_api/g1_base_env.py`

1. 在 `G1BaseEnv` 增加 `step` 实现（§3.1.1），内含 PD 控制循环
2. 抽取 `_action_to_target` / `_pd_controller` / `_compute_reward` / `_is_terminated` 为子类 hook（默认实现或抽象）
3. 将 `run_lesson` 内部的 `self.do_simulation(ctrl, self.frame_skip)` 改为 `self.step(ctrl)`（§3.1.2）
4. 检查 verifier 取数路径，必要时把验证字段补入 `step` 返回的 `info`

**验收**：
- `G1BaseEnv` 可通过 `gym.make` 实例化并执行 `env.reset()` + `env.step(action)` 不抛 `NotImplementedError`
- `run_lesson` 走 `step` 路径，verifier 仍能正确验证
- 子类（05~09）若复写 `step`，需确认与 `G1BaseEnv.step` 不冲突

### 步骤 3：子类兼容性扫描（P0 收尾）

**范围**：`OrcaPlayground/examples/euler/05_*` ~ `09_*`

1. 列出所有继承 `G1BaseEnv` 的子类
2. 检查是否自行实现 `step` / `do_simulation` 调用
3. 重构为复写 `G1BaseEnv` 的 hook（`_pd_controller` 等），而非绕过 `step`

**验收**：
- 所有子类通过 `step` 接口步进，无直接 `do_simulation` 主路径调用
- `ruff check --select SLF001` 零报警

### 步骤 4：Locomotion PD 样例（P1）

**范围**：`OrcaPlayground/examples/euler/` 下新增样例目录

1. 创建一个可直接接 SB3 训练的 G1 locomotion 样例
2. 复用 `G1BaseEnv.step` 的 PD 骨架，仅复写 hook
3. 提供最小可运行的 `PPO` 训练脚本片段

**验收**：
- 样例可通过 SB3 `PPO` 训练若干步不报错
- 样例代码结构清晰，作为后续 locomotion 开发模板

---

## 5. 不变更项

以下**不在本次重构范围**，保持原状：

| 项 | 理由 |
|----|------|
| `reset_model` 命名 | Gym `MujocoEnv` 十年公开 hook 约定，改名切断生态兼容 |
| `_get_obs` 命名 | 同上，`_` 表示 protected，子类复写是 Python 常规 |
| `do_simulation` 公开性 | Gym `MujocoEnv` 公开方法，且 Locomotion PD 需在 `step` 内多次调用，改 `_do_simulation` 会阻塞高级用法并违反 AGENTS.md 规则 4 |
| 架构方式 B 保留 | §6.4 S3 已定位为"高级逃生通道，默认不推荐"，禁止会切断接触富优化等场景的逃生口 |
| `OrcaGymEnvMixin` 提供 `reset` | Mixin 模式本身合规，问题在文档化不足，已在架构 §5.1/§5.9 补齐 |

---

## 6. 风险与回退

### 6.1 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `G1BaseEnv.step` 的 PD 循环与现有 verifier 取数时机不一致 | 中 | verifier 读到非预期状态 | 在 `step` 返回的 `info` 中补齐验证字段 |
| 子类自行实现 `step` 与基类冲突 | 低 | 子类行为不一致 | 步骤 3 扫描时统一为复写 hook 模式 |
| `run_lesson` 改用 `step` 后性能下降（多一层五元组组装） | 低 | 演示循环变慢 | 可接受，正确性优先；必要时 verifier 走 `step` 的 fast path |
| Locomotion PD 样例依赖 GPU | 中 | sandbox 内无法运行 | 按 AGENTS.md 规则 3 用白名单解释器旁路 |

### 6.2 回退

若 P0 修复导致 example 4-9 大面积回归，可临时：
1. 保留 `G1BaseEnv.step` 实现（不回退，因其是正确路径）
2. `run_lesson` 暂时双路径：优先 `step`，verifier 不兼容时 fallback `do_simulation`（标注 deprecated）
3. 在后续迭代中消除 fallback

---

## 7. 验收清单

### 7.1 代码层验收

| # | 验收项 | 验证方式 |
|---|--------|---------|
| V1 | `OrcaGymEulerEnv` 类 docstring 列出继承自 Mixin 的方法 | 源码审查 |
| V2 | `step`/`reset_model`/`_get_obs` 有 docstring 模板 | 源码审查 |
| V3 | `G1BaseEnv.step` 实现完整，可被 `gym.make` 调用 | 运行时测试 |
| V4 | `run_lesson` 走 `step` 路径 | 源码审查 + 运行时 |
| V5 | example 05~09 子类无直接 `do_simulation` 主路径调用 | `grep do_simulation` 扫描 |
| V6 | `ruff check --select SLF001 orca_gym/` 零报警 | ruff 执行 |
| V7 | Locomotion PD 样例可接 SB3 `PPO` 训练 | 运行时测试 |

### 7.2 架构一致性验收

| # | 验收项 | 架构依据 |
|---|--------|---------|
| A1 | EulerEnv docstring 与架构 §5.1 一致 | §5.1 |
| A2 | hook 模板与架构 §12.4.1 一致 | §12.4.1 |
| A3 | `step` 内部调 `do_simulation`，符合 §6.4 S5 | §6.4 S5 |
| A4 | Locomotion PD 用 §6.4 S6 模式 | §6.4 S6 |
| A5 | K1–K14 + M0-M7 机制未被破坏 | §12.3, §7 |

---

## 8. 与架构文档的对应关系

| 本文档章节 | 架构文档章节 | 关系 |
|-----------|-------------|------|
| §2.1 P0 | §6.4 S5（step 是唯一入口） | 落地 S5 约束 |
| §3.1 P0 修复 | §6.4 S6 + §8.3 PD 模式 | 落地 S6 + PD 样例 |
| §3.2 P1 hook 模板 | §12.4.1 骨架签名 | 同步 docstring 到代码 |
| §3.3 P2 docstring | §5.1 设计契约 | 同步契约到代码 |
| §5 不变更项 | §5.9 + §6.4 S3 + §8.3 | 保持 hook 命名与方式 B |

本文档是架构文档的**代码落地实施计划**，不引入新架构决策。所有设计决策已在上游架构文档修订中完成。
