# 🎯 动作空间

动作空间定义了智能体可以向环境输出的控制命令。

## 自动生成

OrcaGym 从 MuJoCo 执行器控制范围自动生成动作空间：

```python
class MyEnv(OrcaGymEulerEnv):
    def __init__(self, ...):
        super().__init__(...)
        
        # 方式 1：自动生成
        ctrl_range = self.model.get_actuator_ctrlrange()  # (nu, 2)
        self.action_space = self.generate_action_space(ctrl_range)
        
        # 方式 2：手动定义
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )
```

## 动作空间类型

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| 位置控制 | 动作 = 目标关节位置 | 机械臂 |
| 速度控制 | 动作 = 目标关节速度 | 移动机器人 |
| 力矩控制 | 动作 = 目标力矩 | 直接力控制 |
| 增量控制 | 动作 = 当前位置的偏移 | 精细操控 |

## 理解动作维度

```python
# 每个执行器的信息
actuator_dict = env.model.get_actuator_dict()

for name, info in actuator_dict.items():
    print(f"{name}:")
    print(f"  类型: {info['TrnType']}")
    print(f"  控制范围: {info['CtrlRange']}")
    print(f"  力范围: {info['ForceRange']}")
    print(f"  齿轮比: {info['GearRatio']}")
    print(f"  关联关节: {info['JointName']}")
```

## 动作缩放

```python
def scale_action(action, low, high):
    """将 [-1, 1] 的动作缩放到 [low, high]"""
    return low + (action + 1.0) * 0.5 * (high - low)

# 使用执行器实际范围缩放
ctrl_range = env.model.get_actuator_ctrlrange()
low = ctrl_range[:, 0]
high = ctrl_range[:, 1]
scaled_action = scale_action(normalized_action, low, high)
```

## 动作空间与 step()

```python
# step() 接受与 action_space 匹配的动作
action = env.action_space.sample()     # 正确：来自 action_space
obs, reward, _, _, _ = env.step(action)

# 或直接传递 ctrl
ctrl = np.zeros(env.model.nu)           # 自定义
env.do_simulation(ctrl, env.frame_skip) # 直接控制
```
