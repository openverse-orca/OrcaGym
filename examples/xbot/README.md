# XBot在OrcaGym中运行
参考自http://github.com/roboterax/humanoid-gym.git
## ✅ 当前状态

**`run_xbot_orca.py`已经可以在OrcaGym中稳定运行！**

机器人现在可以：
- ✅ 稳定站立
- ✅ 平稳行走
- ✅ 使用humanoid-gym预训练模型
- ✅ 策略文件集成在项目内（config目录）
- ✅ **支持 MUSA GPU 加速**（已验证）

### 🚀 MUSA GPU 支持

XBot 项目已完全支持 MUSA（摩尔线程）显卡：
- ✅ PyTorch JIT 模型可以直接在 MUSA GPU 上运行
- ✅ 策略推理在 GPU 上执行，性能最优
- ✅ 自动检测 MUSA GPU（优先级：MUSA > CUDA > CPU）

**使用 MUSA GPU：**
```bash
python run_xbot_orca.py --orcagym_addr 192.168.1.88:50051 --device musa
# 输出: [INFO] Policy loaded on device: musa:0
```

**自动检测（推荐）：**
```bash
python run_xbot_orca.py --orcagym_addr 192.168.1.88:50051 --device auto
```

## 🚀 使用方法

### 方法1: 自动运行（固定速度）

```bash
conda activate orca
python examples/xbot/run_xbot_orca.py
```

**速度调整**: 编辑`run_xbot_orca.py`第92-94行：
```python
CMD_VX = 0.4   # 前向速度 (m/s)
CMD_VY = 0.0   # 侧向速度 (m/s)
CMD_DYAW = 0.0 # 转向速度 (rad/s)
```

### 方法2: 键盘控制（WASD）⭐

```bash
conda activate orca
python examples/xbot/run_xbot_keyboard.py
```

**按键说明**:
- `W` - 前进
- `S` - 后退
- `A` - 左转
- `D` - 右转
- `Q` - 左平移
- `E` - 右平移
- `LShift` - 加速（Turbo模式，2倍速度）
- `Space` - 停止
- `R` - 手动重置环境 ⭐
- `Esc` - 退出程序

**特性**:
- ✅ **不会自动重置**: 即使检测到摔倒或超时，机器人也会继续运行
- ✅ **手动控制**: 只有按R键才会重置环境
- ✅ **实时速度调整**: 按住按键即时响应

## 📊 性能指标

使用humanoid-gym预训练模型：
- Episode长度: 262步 (26.2秒)
- 行走距离: 1.05m
- 平均速度: 0.4 m/s
- 姿态稳定: Roll/Pitch < 5°

## 🔧 核心组件

### 环境
- **`envs/xbot_gym/xbot_simple_env.py`** - XBot环境实现
  - 基于`OrcaGymLocalEnv`
  - 实现PD控制、观察空间、decimation

### 配置和策略
- **`config/policy_example.pt`** - 预训练策略模型 ⭐
  - 来自humanoid-gym项目
  - 已集成在项目内
  - 无需外部依赖

- **`config/xbot_train_config.yaml`** - 训练配置文件

### 运行脚本
- **`run_xbot_orca.py`** - 自动运行脚本（固定速度）
  - 加载预训练策略
  - 设置固定命令速度
  - 实时监控和诊断

- **`run_xbot_keyboard.py`** - 键盘控制脚本 ⭐
  - WASD控制移动方向
  - 实时调整速度
  - 支持Turbo加速模式


## 📝 关键配置

```python
# PD控制参数
kps = [200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15]
kds = [10.0] * 12
tau_limit = 200.0
action_scale = 0.25

# 仿真参数
timestep = 0.001s  # 1ms
decimation = 10    # 策略100Hz
frame_stack = 15   # 观察堆叠
```

## ⚠️ 注意事项

1. **必须使用orca conda环境**
2. **OrcaStudio必须先启动**
3. **XBot-L必须已加载到场景中**
4. **初始高度约0.88m**（OrcaStudio默认spawn高度）

## 🎯 性能测试

### 性能测试脚本

使用 `run_xbot_benchmark.py` 进行性能测试：

```bash
# 测试 MUSA GPU 性能
python run_xbot_benchmark.py --device musa --warmup 100 --iterations 1000

# 对比所有可用设备（MUSA GPU, CUDA GPU, CPU）
python run_xbot_benchmark.py --compare_all --warmup 100 --iterations 1000

# 测试批量推理性能
python run_xbot_benchmark.py --device musa --batch_sizes 1 4 8 16 32
```

**测试指标：**
- ⏱️ 推理时间（平均、中位数、P50/P95/P99）
- 🚀 吞吐量（FPS、推理/秒）
- 💾 内存使用情况
- 📊 批量推理性能对比

**参数说明：**
- `--device`: 测试设备 (cpu, cuda, musa, auto)
- `--warmup`: 预热迭代次数（默认: 100）
- `--iterations`: 测试迭代次数（默认: 1000）
- `--batch_sizes`: 批量推理测试的批量大小
- `--compare_all`: 对比所有可用设备
- `--no_batch`: 跳过批量推理测试

## 🎯 下一步

- ✅ 核心功能已完成
- ✅ 可以稳定运行
- ✅ 性能测试工具已就绪
- 📈 如需训练自定义模型，可参考humanoid-gym项目

---

**项目状态**: ✅ 可用
