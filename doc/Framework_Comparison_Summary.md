# Isaac Gym vs OrcaGym 框架对比总结

## 快速对比表

| 特性维度 | Isaac Gym (legged_gym) | OrcaGym | 迁移难度 |
|---------|----------------------|---------|---------|
| **物理引擎** | NVIDIA Isaac Gym (PhysX) | Mujoco / 其他物理引擎 | ⭐⭐⭐ |
| **执行模式** | 同步向量化 (GPU) | 异步分布式 (CPU/GPU) | ⭐⭐⭐ |
| **环境接口** | VecEnv (自定义) | Gymnasium (标准) | ⭐⭐ |
| **数据结构** | PyTorch Tensor | NumPy Array | ⭐⭐ |
| **RL 算法** | RSL-RL (内置PPO) | SB3/RLlib (多种算法) | ⭐⭐ |
| **配置系统** | 嵌套类 | 字典 | ⭐ |
| **并行策略** | 单机 GPU 并行 | 多机分布式 | ⭐⭐⭐ |
| **通信方式** | 本地内存 | 本地或远程通信 | ⭐⭐ |
| **观测计算** | 批量 Tensor 操作 | 单个 NumPy 操作 | ⭐⭐ |
| **奖励计算** | 批量 Tensor 操作 | 单个 NumPy 操作 | ⭐⭐ |

*难度: ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐ 困难*

---

## 核心架构对比

### Isaac Gym 架构特点

```
特点:
✅ GPU 加速，单机性能极强
✅ 低延迟，张量并行计算
✅ 批量环境同步执行
✅ 内置地形生成
✅ 与 Isaac Sim 生态集成

限制:
❌ 仅支持 NVIDIA GPU
❌ 多机部署需要 NvLink 互联（成本高）
❌ 环境接口不标准
❌ RL 库选择有限 (RSL-RL)
❌ 调试困难 (黑盒物理引擎)
```

### OrcaGym 架构特点

```
特点:
✅ 标准 Gymnasium 接口
✅ 支持多种物理引擎
✅ 分布式部署能力
✅ 兼容主流 RL 库
✅ 灵活的场景配置
✅ 支持实时可视化

限制:
❌ 单机性能不如 Isaac Gym (CPU vs GPU)
❌ 多机部署需要额外配置
❌ 地形配置需要 OrcaStudio
```

---

## 关键代码转换示例

### 1. 环境初始化

**Isaac Gym:**
```python
# 使用 task_registry
env, env_cfg = task_registry.make_env(name="anymal_c_rough", args=args)
# env.num_envs = 4096 个环境同时运行
```

**OrcaGym:**
```python
# 使用 gymnasium.make
env = gym.make(
    "LeggedGym-v0",
    orcagym_addr="localhost:50051",
    agent_names=["anymal_000"],
    # ...
)
```

### 2. 观测计算

**Isaac Gym:**
```python
def compute_observations(self):
    # 批量计算，所有环境一次性
    self.obs_buf = torch.cat((
        self.base_lin_vel * self.obs_scales.lin_vel,  # [4096, 3]
        self.base_ang_vel * self.obs_scales.ang_vel,   # [4096, 3]
        self.projected_gravity,                         # [4096, 3]
        self.commands[:, :3] * self.commands_scale,    # [4096, 3]
        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # [4096, 12]
        self.dof_vel * self.obs_scales.dof_vel,       # [4096, 12]
        self.actions                                    # [4096, 12]
    ), dim=-1)  # 结果: [4096, 48]
```

**OrcaGym:**
```python
def get_obs(self, sensor_data, qpos_buffer, qvel_buffer, ...):
    # 单个 agent 计算
    obs = np.concatenate([
        self._body_lin_vel,                             # [3]
        self._body_ang_vel,                             # [3]
        self._body_orientation,                         # [3]
        self._command_values,                           # [4]
        (self._leg_joint_qpos - self._neutral_joint_values),  # [12]
        self._leg_joint_qvel,                           # [12]
        self._action,                                   # [12]
    ]).astype(np.float32)  # 结果: [49]
    
    obs *= self._obs_scale_vec
    return {"observation": obs, ...}
```

### 3. 奖励计算

**Isaac Gym:**
```python
def compute_reward(self):
    # 批量计算所有环境的奖励
    self.rew_buf[:] = 0.  # [4096]
    for i in range(len(self.reward_functions)):
        name = self.reward_names[i]
        rew = self.reward_functions[i]() * self.reward_scales[name]  # [4096]
        self.rew_buf += rew
        self.episode_sums[name] += rew

def _reward_tracking_lin_vel(self):
    # 返回批量奖励
    lin_vel_error = torch.sum(
        torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), 
        dim=1
    )  # [4096]
    return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)  # [4096]
```

**OrcaGym:**
```python
def compute_reward(self, achieved_goal, desired_goal):
    # 单个 agent 计算奖励
    total_reward = 0.0  # scalar
    
    for reward_fn in self._reward_functions:
        if reward_fn["coeff"] == 0:
            continue
        reward = reward_fn["function"](reward_fn["coeff"])  # scalar
        total_reward += reward
    
    return total_reward  # scalar

def _compute_reward_follow_command_linvel(self, coeff):
    # 返回单个奖励
    lin_vel_error = np.sum(
        np.square(self._command["lin_vel"][:2] - self._body_lin_vel[:2])
    )  # scalar
    reward = np.exp(-lin_vel_error / tracking_sigma) * coeff * self.dt  # scalar
    return reward  # scalar
```

### 4. 环境重置

**Isaac Gym:**
```python
def reset_idx(self, env_ids):
    """重置指定环境的子集"""
    if len(env_ids) == 0:
        return
    
    # 批量重置 DOF
    self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
        0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
    )  # [N, 12]
    self.dof_vel[env_ids] = 0.
    
    # 批量重置根状态
    self.root_states[env_ids] = self.base_init_state
    self.root_states[env_ids, :3] += self.env_origins[env_ids]
    self.root_states[env_ids, :2] += torch_rand_float(
        -1., 1., (len(env_ids), 2), device=self.device
    )
    
    # 一次性应用到所有指定环境
    self.gym.set_dof_state_tensor_indexed(
        self.sim,
        gymtorch.unwrap_tensor(self.dof_state),
        gymtorch.unwrap_tensor(env_ids.to(dtype=torch.int32)),
        len(env_ids)
    )
```

**OrcaGym:**
```python
def reset_agents(self, agents: list[LeggedRobot]):
    """重置指定的 agents"""
    if len(agents) == 0:
        return
    
    joint_qpos = {}
    joint_qvel = {}
    
    # 遍历每个 agent 分别重置
    for agent in agents:
        # 单个 agent 重置
        agent_qpos, agent_qvel = agent.reset(
            self.np_random, 
            height_map=self._height_map
        )
        joint_qpos.update(agent_qpos)
        joint_qvel.update(agent_qvel)
    
    # 批量应用到物理引擎
    self.set_joint_qpos(joint_qpos)
    self.set_joint_qvel(joint_qvel)
    self.mj_forward()
    self.update_data()
```

### 5. 配置格式

**Isaac Gym:**
```python
class AnymalCRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        
    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}
        action_scale = 0.25
        decimation = 4
    
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            orientation = -1.0
            lin_vel_z = -2.0
            torques = -0.00001
```

**OrcaGym:**
```python
AnymalCConfig = {
    # 关节配置
    "base_joint_name": "base",
    "leg_joint_names": [
        "LF_HAA", "LF_HFE", "LF_KFE",
        # ...
    ],
    "neutral_joint_angles": {
        "LF_HAA": 0.0,
        "LF_HFE": 0.4,
        # ...
    },
    
    # 控制配置
    "actuator_type": "position",
    "kps": [80., 80., 80.] * 4,  # 12 个关节
    "kds": [2., 2., 2.] * 4,
    "action_scale": [0.25] * 12,
    
    # 奖励配置
    "reward_coeff": {
        "rough_terrain": {
            "follow_command_linvel": 10.0,
            "follow_command_angvel": 5.0,
            "body_orientation": -1.0,
            "body_lin_vel": -2.0,
            "torques": -0.00001,
        }
    },
    
    # Domain randomization
    "randomize_friction": True,
    "friction_range": [0.5, 1.25],
    "push_robots": True,
    "push_interval_s": 15,
}
```

---

## 训练流程对比

### Isaac Gym 训练流程

```python
# 1. 注册环境
task_registry.register(
    "anymal_c_rough",
    AnymalCRough,
    AnymalCRoughCfg(),
    AnymalCRoughCfgPPO()
)

# 2. 创建环境
env, env_cfg = task_registry.make_env(name="anymal_c_rough")

# 3. 创建 PPO runner
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name="anymal_c_rough")

# 4. 训练
ppo_runner.learn(num_learning_iterations=10000)
```

**特点**:
- 一体化流程
- 自定义 PPO 实现 (RSL-RL)
- 环境和算法强耦合

### OrcaGym 训练流程

```python
# 1. 注册环境
gym.register(
    id='LeggedGym-v0',
    entry_point='envs.legged_gym.legged_gym_env:LeggedGymEnv',
)

# 2. 创建向量化环境
env = SubprocVecEnv([
    make_env("LeggedGym-v0", i, config) 
    for i in range(num_envs)
])

# 3. 创建 PPO 模型 (Stable-Baselines3)
model = PPO("MlpPolicy", env, learning_rate=3e-4, ...)

# 4. 训练
model.learn(total_timesteps=10000000)
model.save("final_model")
```

**特点**:
- 模块化设计
- 标准 RL 库 (SB3/RLlib)
- 环境和算法解耦

---

## 性能对比

### 吞吐量对比 (样本/秒)

| 场景 | Isaac Gym | OrcaGym (本地) | OrcaGym (分布式) |
|------|-----------|--------------|---------------|
| 最大吞吐量 | ~100,000 | ~50,000 | ~1,000,000 |
| 多机扩展性 | 支持 (需 NvLink) | 支持 | 支持 |
| 延迟 | ~1ms | ~1ms | 取决于网络 |

**说明**:
- Isaac Gym 单机性能极强，多机扩展需要 NvLink（硬件成本高）
- OrcaGym 本地模式性能约为 Isaac Gym 的 50%
- OrcaGym 分布式模式可通过通用网络扩展，实现 10 倍以上的吞吐量提升
- 单机模式下两者延迟都在 1ms 级别

### 资源消耗对比

| 资源 | Isaac Gym | OrcaGym |
|------|-----------|---------|
| 内存消耗 | ~64GB | ~64GB |
| GPU 依赖 | 必需 NVIDIA GPU | 支持各类 GPU（兼容 PyTorch） |
| 多机部署 | 支持 (需 NvLink 互联) | 支持 (通用网络) |

---

## 适用场景建议

### 选择 Isaac Gym 的场景

✅ **适合**:
- 单机训练，有 NVIDIA GPU
- 追求极致的训练速度
- 需要与 Isaac Sim 生态集成
- 简单的训练场景

❌ **不适合**:
- 需要低成本的多机扩展（NvLink 成本高）
- 需要灵活的 RL 算法切换
- 需要与其他工具集成
- 复杂的仿真场景

### 选择 OrcaGym 的场景

✅ **适合**:
- 需要低成本的多机扩展（使用通用网络）
- 需要使用多种 RL 算法 (SB3/RLlib)
- 需要灵活的物理引擎选择
- 需要与 Gymnasium 生态集成
- 需要实时可视化和调试
- 需要与真实机器人系统对接

❌ **不适合**:
- 追求极致单机性能
- 不想配置额外的服务器
- 简单的原型验证

---

## 迁移成本估算

### 小型项目 (1-2 个机器人)
- **时间**: 1-2 周
- **难度**: 中等
- **建议**: 参考现有示例，快速上手

### 中型项目 (3-5 个机器人)
- **时间**: 2-4 周
- **难度**: 中等偏高
- **建议**: 模块化迁移，逐步测试

### 大型项目 (复杂场景、多机器人)
- **时间**: 1-2 月
- **难度**: 高
- **建议**: 团队协作，分阶段迁移

---

## 常见迁移陷阱

### ❌ 陷阱 1: 直接照搬批量逻辑

```python
# 错误: 直接使用批量操作
def compute_reward(self):
    rewards = np.zeros(len(self.agents))  # ❌ OrcaGym 是单个 agent
    for i, agent in enumerate(self.agents):
        rewards[i] = agent.compute_reward()
    return rewards
```

```python
# 正确: 单个 agent 逻辑
def compute_reward(self, achieved_goal, desired_goal):
    total_reward = 0.0  # ✅ 单个标量
    for reward_fn in self._reward_functions:
        total_reward += reward_fn["function"](reward_fn["coeff"])
    return total_reward
```

### ❌ 陷阱 2: 忽略坐标系转换

```python
# 错误: 使用全局坐标
reward = -np.sum(np.square(self._body_lin_vel[:2]))  # ❌ 全局坐标

# 正确: 转换到局部坐标
body_lin_vel_local, _, _ = global2local(
    body_orientation_quat, 
    body_lin_vel_global, 
    body_ang_vel_global
)
reward = -np.sum(np.square(body_lin_vel_local[:2]))  # ✅ 局部坐标
```

### ❌ 陷阱 3: 配置参数不对应

```python
# 错误: 直接复制 Isaac Gym 配置
cfg.control.decimation = 4  # ❌ Isaac Gym 术语

# 正确: 使用 OrcaGym 术语
LeggedEnvConfig["ACTION_SKIP"] = 4  # ✅ OrcaGym 术语
```

---

## 总结

### Isaac Gym 优势
1. **性能**: 单机训练速度极快
2. **易用性**: 一体化工具链
3. **生态**: 与 NVIDIA 生态集成

### OrcaGym 优势
1. **灵活性**: 支持多种物理引擎和 RL 库
2. **扩展性**: 分布式部署能力
3. **标准化**: Gymnasium 接口兼容性
4. **可维护性**: 模块化设计

### 最佳实践
1. **先理解再迁移**: 充分理解两个框架的差异
2. **模块化迁移**: 分步骤逐个模块迁移
3. **充分测试**: 每个模块单独测试
4. **性能调优**: 迁移完成后再进行性能优化
5. **文档记录**: 记录每个关键决策和修改

---

**参考**: 完整迁移指南见 [Isaac_Gym_Migration_Guide.md](./Isaac_Gym_Migration_Guide.md)

