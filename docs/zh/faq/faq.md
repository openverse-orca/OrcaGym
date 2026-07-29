# ❓ 常见问题

## 安装与配置

### Q: 安装后导入 `orca_gym` 失败？

```bash
# 确认安装
pip show orca-gym

# 如果显示找不到，重新安装
pip install orca-gym --force-reinstall
```

### Q: MuJoCo 找不到 GLFW 库？

```bash
# Linux
# 注意：libglew 的包名随 Ubuntu 版本不同
#   - Ubuntu 22.04+：libglew2.2
#   - Ubuntu 20.04：libglew2.1
sudo apt-get install libglfw3 libglew2.2 libosmesa6

# macOS
brew install glfw glew
```

### Q: 如何连接 OrcaStudio/OrcaLab？

1. 从 [orca3d.cn](http://orca3d.cn/) 下载并安装
2. 打开软件，点击"运行"按钮
3. 默认 gRPC 地址为 `localhost:50051`

## 仿真运行

### Q: 仿真无法启动，报 "Load local env failed"？

常见原因：
1. 场景配置错误（关节/body 名称重复）
2. 模型初始姿态重叠
3. mesh/纹理资源缺失 → 等待几秒后重试
4. OrcaStudio/OrcaLab 未正常启动

### Q: 发现数据是 NaN？

通常是修改状态后未触发正向计算导致。**推荐通过 `do_simulation()` 完成状态更新**——该方法在 `mj_step` 后会自动调用 `update_data()` 同步 `env.data`，无需手动调用同步逻辑：

```python
# 正确做法：通过 do_simulation 触发步进 + 自动同步
env.do_simulation(ctrl, n_frames=1)
print(env.data.qpos)  # 已是最新状态
```

若确需在子类中重排状态后做一次正向运动学计算，请在**子类内部**通过公共 API 或类内同步机制完成，不要在用户代码中直接调用 `mj_forward()` 或内部 `_sync_view()` 等私有方法（违反 API 隔离）。

### Q: 步进后读到的是旧数据？

`do_simulation()` 内部已自动同步数据，返回后 `env.data` 就是最新状态。如需在 `step()` 之外修改状态，请通过 `do_simulation()` 或子类内部同步机制完成，避免直接调用私有方法。

### Q: 如何提高仿真速度？

1. 设置 `render_mode="none"`
2. 增加 `timestep`（如 0.002 或 0.005）
3. 减少 `frame_skip`
4. 使用向量化环境
5. 简化碰撞几何体

### Q: 远程模式和本地模式如何选择？

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 开发调试 | 本地 | MuJoCo 直连，没有网络延迟 |
| 单机训练 | 本地 + 向量化 | 多进程并行 |
| 大规模分布式 | 远程 | 仿真在服务端，训练在客户端 |
| 需要 PhysX 后端 | 远程 | PhysX 仅在服务端可用 |

## 环境开发

### Q: 如何自定义环境？

继承 `OrcaGymEulerEnv`，实现 `step()`、`reset_model()`、`_get_obs()` 方法即可。

```python
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

class MyEnv(OrcaGymEulerEnv):
    def step(self, action): ...
    def reset_model(self): ...
    def _get_obs(self): ...
```

### Q: action_space 的维度怎么来的？

来自模型中的执行器数量 (`model.nu`)：

```python
print(f"执行器数: {env.model.nu}")
print(f"动作空间: {env.action_space}")
# Box(low=ctrlrange_min, high=ctrlrange_max, shape=(nu,), float32)
```

### Q: 如何添加传感器到观测中？

```python
def _get_obs(self):
    # 关节状态
    proprio = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    # 传感器数据
    sensors = self.query_sensor_data(["imu_acc", "imu_gyro"])

    return np.concatenate([
        proprio,
        sensors["imu_acc"],
        sensors["imu_gyro"],
    ]).astype(np.float32)
```

## 迁移

### Q: 从 Isaac Gym 迁移的关键区别？

| Isaac Gym | OrcaGym |
|-----------|---------|
| `VecEnv` 接口 | `Gymnasium.Env` 接口 |
| PyTorch Tensor 批量操作 | NumPy Array 逐个操作 |
| GPU 单进程 4096 env | 多进程向量化 |
| RSL-RL | Stable-Baselines3 / RLlib |
| PhysX | MuJoCo (本地) |

### Q: 从原生 MuJoCo 环境迁移？

1. 将基类改为 `OrcaGymEulerEnv`
2. 使用 `env.data` 代替直接访问 MuJoCo 内部数据
3. 使用 `env.sim_config` 代替直接访问求解器参数
4. 使用 `env.apply_body_force()` 代替直接写外力

## 其他

### Q: 如何贡献代码？

参见 [贡献指南](https://github.com/openverse-orca/OrcaGym#贡献)。

### Q: 如何获取帮助？

请通过 [GitHub Issues](https://github.com/openverse-orca/OrcaGym/issues) 提交问题。
