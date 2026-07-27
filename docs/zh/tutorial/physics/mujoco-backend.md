# 🔧 MuJoCo 后端

OrcaGym 的 Euler 模式直接使用 MuJoCo 作为物理引擎。环境初始化时自动完成模型加载。

> 完整可运行代码见 [OrcaPlayground examples/euler/](https://github.com/OrcaGym/OrcaPlayground)。

## 模型加载

模型加载在环境初始化时自动完成，无需手动操作：

```python
# 离线模式：从本地 XML 文件加载
env = MyEnv(
    model_xml_path="path/to/scene.xml",
    skip_grpc_load=True,   # True = 离线模式，直接加载本地 XML
)

# 在线模式：通过 gRPC 从 OrcaStudio 获取模型 XML
env = MyEnv(
    model_xml_path="path/to/scene.xml",
    skip_grpc_load=False,  # False（默认）= 在线模式
)
```

内部流程：
1. 加载 XML → 创建 MuJoCo 实例（`mjModel` + `mjData`）
2. 查询并填充 Model / Data 信息
3. 初始化所有字典（body, joint, actuator, ...）
4. 缓存初始状态到 `init_qpos` / `init_qvel`

## 资源缓存

MuJoCo 模型依赖的 mesh 和 hfield 文件会缓存在 `~/.orcagym/tmp/` 目录。

## 仿真控制

### 步进控制

```python
# ✅ 推荐：do_simulation（原子操作，自动同步 data）
env.do_simulation(ctrl, n_frames=20)
# 等价于：set_ctrl → mj_step(20) → _sync_view

# 手动控制（需要自己同步）
env.set_ctrl(ctrl)
env.mj_step(nstep=20)
env._sync_view()           # 同步状态视图

# 前向计算（刷新派生量，不推进时间）
env.mj_forward()

# 纯 MuJoCo 步进
env.mj_step(nstep=20)
```

### ctrl 设置

```python
# ctrl 是模型执行器的控制输入，长度 = model.nu
ctrl = np.zeros(env.model.nu, dtype=np.float64)
env.do_simulation(ctrl, env.frame_skip)
```

## 求解器配置

通过 `env.sim_config` 读写求解器参数：

```python
# 读写单个参数
env.sim_config.timestep = 0.002
env.sim_config.iterations = 100
env.sim_config.integrator = 0       # 0=Euler, 1=RK4
env.sim_config.gravity = np.array([0., 0., -9.81])

# 批量设置
env.sim_config.load_from_dict({
    "integrator": 0,
    "iterations": 100,
    "tolerance": 1e-8,
})

# 导出配置
config = env.sim_config.to_dict()
```

### 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `timestep` | float | 0.002 | 物理步长（秒） |
| `iterations` | int | 100 | 求解器迭代次数 |
| `integrator` | int | 0 | 0=Euler, 1=RK4 |
| `gravity` | ndarray | [0,0,-9.81] | 重力加速度 |
| `tolerance` | float | 1e-8 | 求解器容忍度 |

## 时间步长 vs 控制频率

```
物理步长 (timestep)    = 0.001 秒  # 物理引擎每步的时间（默认）
控制步长 (frame_skip)  = 20       # 每次 step() 执行多少物理步
环境步长 (dt)          = 0.02 秒  # timestep × frame_skip
控制频率               = 50 Hz    # 1 / dt
```

```python
print(f"物理步长: {env.sim_config.timestep:.4f}s")
print(f"frame_skip: {env.frame_skip}")
print(f"控制步长: {env.dt:.4f}s")
print(f"控制频率: {1.0/env.dt:.1f}Hz")
```

### G1 标准配置

G1 人形机器人使用以下标准参数（来自 Euler 示例）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `time_step` | 0.001 | 物理步长 1ms |
| `frame_skip` | 20 | 每控制周期 20 物理步 |
| `dt` | 0.02s | 控制频率 50Hz |

## 调试与性能分析

```python
# 查看约束计数
counts = env.get_constraint_counts()
print(f"等式约束: {counts.get('nefc', 0)}, 接触: {counts.get('ncon', 0)}")

# 查看模型信息
print(f"nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")
print(f"nbody={env.model.nbody}, ngeom={env.model.ngeom}")
print(f"njnt={env.model.njnt}, nsite={env.model.nsite}")
```
