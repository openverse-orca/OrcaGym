# 从 run_dual_arm_sim.py 迁移到 run_dual_arm_sim_with_config.py

## 为什么要迁移？

新版本 `run_dual_arm_sim_with_config.py` 提供了更灵活的机器人配置支持：

✅ 可以显式指定机器人配置  
✅ 支持为不同机器人使用不同配置  
✅ 可以列出所有可用配置  
✅ 完全向后兼容，无需修改现有命令  

## 迁移步骤

### 不需要修改的情况

如果您的命令不需要显式指定配置，**无需任何修改**！只需将脚本名称替换即可：

#### 之前
```bash
python run_dual_arm_sim.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

#### 之后
```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

行为完全一致！系统会自动根据机器人名称推断配置。

### 利用新功能

#### 1. 显式指定配置

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --robot_config openloong \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

#### 2. 多机器人不同配置

```bash
python run_dual_arm_sim_with_config.py \
    --agent_names "robot1_usda robot2_usda" \
    --robot_configs "robot1_usda:openloong,robot2_usda:d12" \
    --pico_ports "8001 8002" \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

#### 3. 查看可用配置

```bash
python run_dual_arm_sim_with_config.py --list_configs
```

## 命令对照表

| 原命令 | 新命令 | 说明 |
|--------|--------|------|
| `python run_dual_arm_sim.py [args]` | `python run_dual_arm_sim_with_config.py [args]` | 直接替换，无需修改参数 |
| 无对应命令 | `--list_configs` | 新增：列出可用配置 |
| 无对应命令 | `--robot_config openloong` | 新增：指定配置 |
| 无对应命令 | `--robot_configs "r1:c1,r2:c2"` | 新增：多机器人配置 |

## 批量迁移脚本

如果您有很多脚本需要更新，可以使用以下命令批量替换：

```bash
# 在当前目录下查找并替换
find . -name "*.sh" -type f -exec sed -i 's/run_dual_arm_sim\.py/run_dual_arm_sim_with_config.py/g' {} \;
```

## 常见场景迁移示例

### 场景1：基础遥操作

#### 之前
```bash
python run_dual_arm_sim.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --pico_ports 8001 \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

#### 之后（选项A - 保持原样）
```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --pico_ports 8001 \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

#### 之后（选项B - 显式配置）
```bash
python run_dual_arm_sim_with_config.py \
    --agent_names openloong_gripper_2f85_fix_base_usda \
    --robot_config openloong \
    --pico_ports 8001 \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

### 场景2：多机器人

#### 之前（只能使用相同配置）
```bash
python run_dual_arm_sim.py \
    --agent_names "robot1_usda robot2_usda" \
    --pico_ports "8001 8002" \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

#### 之后（可以使用不同配置）
```bash
python run_dual_arm_sim_with_config.py \
    --agent_names "robot1_usda robot2_usda" \
    --robot_configs "robot1_usda:openloong,robot2_usda:d12" \
    --pico_ports "8001 8002" \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

### 场景3：回放数据

#### 之前
```bash
python run_dual_arm_sim.py \
    --run_mode playback \
    --dataset ./records_tmp/shop/data.hdf5 \
    --agent_names openloong_gripper_2f85_fix_base_usda
```

#### 之后（无需修改）
```bash
python run_dual_arm_sim_with_config.py \
    --run_mode playback \
    --dataset ./records_tmp/shop/data.hdf5 \
    --agent_names openloong_gripper_2f85_fix_base_usda
```

## 迁移检查清单

- [ ] 将脚本名称从 `run_dual_arm_sim.py` 改为 `run_dual_arm_sim_with_config.py`
- [ ] （可选）添加 `--robot_config` 参数显式指定配置
- [ ] （可选）如果使用多机器人，考虑使用 `--robot_configs` 指定不同配置
- [ ] 测试命令是否正常工作
- [ ] 更新相关文档和脚本

## 回退方案

如果遇到问题，可以随时回退到原版本：

```bash
# 只需将脚本名称改回去
python run_dual_arm_sim.py [原来的参数]
```

原版本 `run_dual_arm_sim.py` 仍然可用且功能不变。

## 推荐做法

1. **测试环境先试用**：在测试环境先使用新版本，确认无误后再应用到生产环境
2. **保留原脚本**：保留原版本脚本作为备份
3. **逐步迁移**：先从简单的命令开始迁移，再处理复杂场景
4. **使用显式配置**：在生产环境建议使用 `--robot_config` 显式指定配置，避免依赖自动推断

## 获取帮助

如有疑问：

```bash
# 查看帮助
python run_dual_arm_sim_with_config.py --help

# 列出可用配置
python run_dual_arm_sim_with_config.py --list_configs
```

或参考：
- [配置系统使用指南](README_CONFIG.md)
- [详细配置文档](../../envs/manipulation/robots/configs/README.md)

