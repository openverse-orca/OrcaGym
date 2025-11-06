# 腰部关节数据保存功能说明

## 概述

本功能为OrcaGym仿真环境添加了腰部关节数据的保存功能，确保在数据采集和保存过程中包含完整的腰部运动信息。

## 新增的观测数据

### 腰部关节观测数据

在`get_obs()`方法中新增了以下观测数据：

1. **waist_joint_qpos**: 腰部关节当前角度
   - 类型: `np.float32`
   - 形状: `(1,)`
   - 范围: `[-π/2, π/2]` 弧度

2. **waist_joint_qpos_sin**: 腰部关节角度的sin值
   - 类型: `np.float32`
   - 形状: `(1,)`
   - 范围: `[-1, 1]`

3. **waist_joint_qpos_cos**: 腰部关节角度的cos值
   - 类型: `np.float32`
   - 形状: `(1,)`
   - 范围: `[-1, 1]`

4. **waist_joint_vel**: 腰部关节角速度
   - 类型: `np.float32`
   - 形状: `(1,)`
   - 范围: `[-π, π]` rad/s

### 观测数据缩放

在`_setup_obs_scale()`方法中添加了腰部关节的缩放参数：

```python
# 腰部关节缩放参数
waist_qpos_scale = np.pi / 2  # 腰部关节角度范围 [-π/2, π/2]
max_waist_joint_vel = np.pi  # 腰部关节角速度范围 rad/s

"waist_joint_qpos": np.ones(1, dtype=np.float32) / waist_qpos_scale,
"waist_joint_qpos_sin": np.ones(1, dtype=np.float32),
"waist_joint_qpos_cos": np.ones(1, dtype=np.float32),
"waist_joint_vel": np.ones(1, dtype=np.float32) / max_waist_joint_vel,
```

## 新增的动作数据

### 动作空间扩展

在`_setup_action_range()`方法中扩展了动作空间：

```python
self._action_range = np.concatenate([
    # ... 原有动作数据 ...
    [[-np.pi/2, np.pi/2]],  # waist joint angle (新增)
], dtype=np.float32, axis=0)
```

### 动作数据结构

在`on_teleoperation_action()`方法中，动作数组现在包含：

```
action = [
    action_l_B,           # 0-5  : 左臂末端位置和旋转
    ctrl_l,              # 6-12 : 左臂关节控制
    grasp_value_l,       # 13   : 左臂抓取值
    action_r_B,          # 14-19: 右臂末端位置和旋转
    ctrl_r,              # 20-26: 右臂关节控制
    grasp_value_r,       # 27   : 右臂抓取值
    waist_angle,         # 28   : 腰部关节角度 (新增)
]
```

### 动作处理

在`on_playback_action()`方法中添加了腰部关节动作的处理：

```python
# 处理腰部关节action
waist_angle = action[28]
self.set_waist_control(waist_angle)
```

## 数据保存格式

### HDF5文件结构

```
dataset.h5
├── observations/
│   ├── waist_joint_qpos          # 腰部关节角度序列
│   ├── waist_joint_qpos_sin      # 腰部关节角度sin值序列
│   ├── waist_joint_qpos_cos      # 腰部关节角度cos值序列
│   └── waist_joint_vel           # 腰部关节角速度序列
├── actions/
│   └── actions                   # 完整动作序列 (包含腰部关节)
└── metadata                      # 元数据信息
```

### 元数据信息

```json
{
  "description": "包含腰部关节数据的机器人遥操作数据",
  "waist_joint_info": {
    "angle_range": "[-π/2, π/2]",
    "velocity_range": "[-π, π]",
    "position_in_action": 28,
    "observation_keys": [
      "waist_joint_qpos",
      "waist_joint_qpos_sin", 
      "waist_joint_qpos_cos",
      "waist_joint_vel"
    ]
  },
  "total_steps": 1000,
  "action_dimension": 29,
  "created_at": "2024-01-01T12:00:00"
}
```

## 使用方法

### 数据采集

在遥操作模式下，腰部关节数据会自动包含在观测和动作中：

```python
# 遥操作时自动包含腰部数据
obs = env.get_obs()
action = env.step(action)

# 观测数据包含腰部信息
waist_angle = obs['waist_joint_qpos']
waist_velocity = obs['waist_joint_vel']

# 动作数据包含腰部控制
waist_action = action[28]
```

### 数据回放

在回放模式下，腰部关节动作会被自动处理：

```python
# 回放时自动处理腰部动作
action = np.array([...])  # 包含29个元素的动作数组
env.step(action)  # 第28个元素会被用作腰部关节控制
```

### 数据加载

```python
import h5py
import numpy as np

def load_waist_data(filename):
    with h5py.File(filename, 'r') as f:
        # 读取腰部关节观测数据
        waist_qpos = f['observations']['waist_joint_qpos'][:]
        waist_vel = f['observations']['waist_joint_vel'][:]
        
        # 读取动作数据
        actions = f['actions']['actions'][:]
        waist_actions = actions[:, 28]  # 第28个元素是腰部关节角度
        
        return waist_qpos, waist_vel, waist_actions
```

## 技术细节

### 数据获取

腰部关节数据通过以下方法获取：

```python
# 获取当前角度
waist_joint_value = self.get_waist_angle()

# 获取角速度
waist_joint_velocity = self._env.data.qvel[self._waist_jnt_dof]
```

### 数据归一化

腰部关节数据会根据以下范围进行归一化：

- **角度范围**: `[-π/2, π/2]` → `[-1, 1]`
- **角速度范围**: `[-π, π]` → `[-1, 1]`
- **sin/cos值**: 已经是 `[-1, 1]` 范围

### 数据一致性

确保观测数据和动作数据的一致性：

- 观测数据反映当前状态
- 动作数据反映控制指令
- 两者在时间上同步

## 优势

1. **完整记录**: 包含腰部关节的完整运动信息
2. **数据丰富**: 提供角度、角速度、sin/cos值等多种表示
3. **便于分析**: 支持后续的数据分析和可视化
4. **训练友好**: 便于训练包含腰部的控制策略
5. **回放支持**: 支持完整的数据回放和复现

## 注意事项

1. **数据维度**: 动作空间从28维扩展到29维
2. **兼容性**: 需要更新相关的数据处理代码
3. **存储空间**: 数据量略有增加
4. **性能影响**: 对性能影响很小

## 相关文件

- `envs/manipulation/dual_arm_robot.py` - 主要实现文件
- `waist_data_saving_example.py` - 使用示例
- `WAIST_DATA_SAVING_README.md` - 本说明文档

## 更新日志

- **v1.0.0**: 初始版本，添加腰部关节数据保存功能
- 扩展观测数据空间
- 扩展动作数据空间
- 添加数据归一化处理
- 支持HDF5格式保存
