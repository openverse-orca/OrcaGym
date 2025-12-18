# 双臂机器人配置系统使用指南

## 概述

`run_dual_arm_sim_with_config.py` 是支持灵活机器人配置的增强版本，您可以通过命令行参数指定机器人配置。

## 与原版本的对比

| 特性 | run_dual_arm_sim.py | run_dual_arm_sim_with_config.py |
|------|---------------------|----------------------------------|
| 机器人配置 | 自动推断 | 支持显式指定 |
| 配置参数 | ❌ 不支持 | ✅ 支持 --robot_config 和 --robot_configs |
| 列出配置 | ❌ 不支持 | ✅ 支持 --list_configs |
| 多机器人不同配置 | ❌ 不支持 | ✅ 支持 |
| 向后兼容 | - | ✅ 完全兼容 |

## 快速开始

### 1. 查看所有可用配置

```bash
python run_dual_arm_sim_with_config.py --list_configs
```

输出示例：
```
============================================================
可用的机器人配置：
============================================================
  1. openloong
  2. openloong_hand_fix_base
  3. openloong_gripper_2f85_fix_base
  4. openloong_gripper_2f85_mobile_base
  5. d12
  6. hand
  7. gripper_2f85

============================================================
使用方法：
============================================================
  单个配置: --robot_config openloong
  多个配置: --robot_configs 'robot1:openloong,robot2:d12'
```

### 2. 使用默认配置（最简单）

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

系统会根据机器人名称自动推断配置（与原版本行为一致）。

### 3. 显式指定配置

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --robot_config openloong \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

### 4. 多个机器人使用不同配置

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names "robot1_usda robot2_usda" \
    --robot_configs "robot1_usda:openloong,robot2_usda:d12" \
    --pico_ports "8001 8002" \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

## 常用场景示例

### 场景1：遥操作数据采集

```bash
python run_dual_arm_sim_with_config.py \
    --orcagym_address localhost:50051 \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --robot_config openloong \
    --pico_ports 8001 \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --action_step 5 \
    --ctrl_device vr \
    --record_length 60 \
    --teleoperation_rounds 10 \
    --level shop \
    --withvideo True
```

### 场景2：回放演示数据

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --robot_config openloong \
    --run_mode playback \
    --dataset ./records_tmp/shop/dual_arm_2025-01-01_12-00-00.hdf5 \
    --playback_mode random \
    --realtime_playback True
```

### 场景3：训练模仿学习策略

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --robot_config openloong \
    --run_mode imitation \
    --dataset ./records_tmp/shop/dual_arm_2025-01-01_12-00-00.hdf5 \
    --algo bc \
    --level shop
```

### 场景4：测试训练好的策略

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --robot_config openloong \
    --run_mode rollout \
    --model_file ./trained_models_tmp/model.pth \
    --rollout_times 10 \
    --level shop
```

### 场景5：数据增强

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --robot_config openloong \
    --run_mode augmentation \
    --dataset ./records_tmp/shop/dual_arm_2025-01-01_12-00-00.hdf5 \
    --augmented_noise 0.005 \
    --augmented_rounds 10 \
    --level shop
```

## 多 Pico 设备配置

如果要使用多个 Pico VR 设备，需要先设置 ADB 反向端口转发：

```bash
# 查看设备序列号
adb devices

# PICO_01:8001 -----> PC:8001
adb -s <device_serial_number_01> reverse tcp:8001 tcp:8001

# PICO_02:8001 -----> PC:8002
adb -s <device_serial_number_02> reverse tcp:8001 tcp:8002
```

然后在脚本参数中指定多个端口：

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names "robot1_usda robot2_usda" \
    --robot_configs "robot1_usda:openloong,robot2_usda:openloong" \
    --pico_ports "8001 8002" \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

## 参数说明

### 配置相关参数

- `--list_configs`: 列出所有可用的机器人配置并退出
- `--robot_config <name>`: 指定所有机器人使用的配置名称
- `--robot_configs <mapping>`: 为不同机器人指定不同配置（格式：robot1:config1,robot2:config2）

### 基础参数

- `--orcagym_address`: gRPC 服务器地址（默认：localhost:50051）
- `--agent_names`: 机器人名称，多个用空格分隔
- `--pico_ports`: Pico 服务器端口，多个用空格分隔

### 运行模式

- `--run_mode`: teleoperation / playback / imitation / rollout / augmentation
- `--action_type`: end_effector_ik / end_effector_osc / joint_pos / joint_motor
- `--action_step`: 每个动作的仿真步数

### 任务配置

- `--task_config`: 任务配置文件路径
- `--level`: 存储级别或场景名称

### 数据集

- `--dataset`: 数据集文件路径
- `--record_length`: 录制时长（秒）
- `--withvideo`: 是否输出视频（True/False）

### 其他参数

详见 `python run_dual_arm_sim_with_config.py --help`

## 常见问题

### Q: 两个脚本有什么区别？

A: `run_dual_arm_sim_with_config.py` 在原版基础上添加了配置参数支持，使用更灵活。如果不指定配置参数，两者行为完全一致。

### Q: 我应该使用哪个脚本？

A: 
- 如果需要显式指定配置或使用多机器人不同配置：使用 `run_dual_arm_sim_with_config.py`
- 如果只使用默认配置：两者都可以

### Q: 如何知道机器人应该用哪个配置？

A: 使用 `--list_configs` 查看所有配置，或让系统自动推断。

### Q: 配置文件在哪里？

A: 在 `envs/manipulation/robots/configs/` 目录下

### Q: 如何添加新配置？

A: 在 `envs/manipulation/robots/configs/` 创建 `xxx_config.py` 文件，定义配置字典即可。详见配置系统文档。

## 相关文档

- [机器人配置系统详细文档](../../envs/manipulation/robots/configs/README.md)
- [配置快速开始](../../envs/manipulation/robots/configs/快速开始.md)
- [系统更新说明](../../ROBOT_CONFIG_UPDATE.md)

## 技术支持

如有问题，请参考：
1. 运行 `--help` 查看参数说明
2. 运行 `--list_configs` 查看可用配置
3. 查阅相关文档

