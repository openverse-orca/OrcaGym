# Bug 修复：robot_configs 未定义错误

## 问题描述

运行以下命令时出现错误：
```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'dexforce_w1' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

错误信息：
```
NameError: name 'robot_configs' is not defined
```

## 根本原因

在 `orca_gym/scripts/dual_arm_manipulation.py` 中：

1. `run_dual_arm_sim()` 函数中获取了 `robot_configs` 参数
2. 但在调用 `run_example()` 函数时没有传递 `robot_configs` 参数
3. 导致 `run_example()` 函数内部使用 `robot_configs` 时出现未定义错误

## 修复内容

### 修改 1: 添加 `robot_configs` 参数到 `run_example` 函数

**文件**: `orca_gym/scripts/dual_arm_manipulation.py`

```python
def run_example(orcagym_addr : str,
                agent_names : str,
                pico_ports : str,
                record_path : str,
                run_mode : str,
                action_type : str,
                action_step : int,
                algo_config : str,
                ctrl_device : str,
                max_episode_steps : int,
                playback_mode : str,
                rollout_times : int,
                ckpt_path : str,
                augmented_noise : float,
                augmented_rounds : int,
                teleoperation_rounds : int,
                sample_range : float,
                realtime_playback : bool,
                current_file_path : str,
                task_config : str,
                augmentation_path : str,
                output_video : bool,
                sync_codec : bool,
                robot_configs : dict = None):  # ← 新增参数
```

### 修改 2: 传递 `robot_configs` 参数到 `run_example` 调用

**文件**: `orca_gym/scripts/dual_arm_manipulation.py`

```python
for config in algo_config:
    run_example(orcagym_addr,
                agent_names,
                pico_ports,
                record_path,
                run_mode,
                action_type,
                action_step,
                config,
                ctrl_device,
                max_episode_steps,
                playback_mode,
                rollout_times,
                ckpt_path,
                augmented_noise,
                augmented_rounds,
                teleoperation_rounds,
                sample_range,
                realtime_playback,
                current_file_path,
                task_config=task_config,
                augmentation_path=augmented_path,
                output_video=withvideo,
                sync_codec=sync_codec,
                robot_configs=robot_configs  # ← 新增参数
                )
```

### 修改 3: 注册 `dexforce_w1` 配置

**文件**: `envs/manipulation/robots/configs/robot_config_registry.py`

```python
try:
    from .dexforce_w1_config import dexforce_w1_config
    cls.register_config("dexforce_w1", dexforce_w1_config)
except ImportError:
    pass
```

## 验证步骤

1. **检查配置是否可用**：
```bash
python run_dual_arm_sim_with_config.py --list_configs
```

应该能看到 `dexforce_w1` 在可用配置列表中。

2. **测试原命令**：
```bash
python run_dual_arm_sim_with_config.py \
    --robot_config 'dexforce_w1' \
    --task_config scan_task.yaml \
    --action_type end_effector_osc \
    --level 'shop_ik_1'
```

应该不再出现 `NameError` 错误。

## 相关文件

| 文件 | 修改内容 |
|------|---------|
| `orca_gym/scripts/dual_arm_manipulation.py` | 添加 `robot_configs` 参数支持 |
| `envs/manipulation/robots/configs/robot_config_registry.py` | 注册 `dexforce_w1` 配置 |

## 注意事项

1. **配置文件已存在**：`dexforce_w1_config.py` 已经存在于 `envs/manipulation/robots/configs/` 目录下
2. **自动加载**：配置注册表会自动加载所有 `*_config.py` 文件，但为了更快的加载，显式注册了 `dexforce_w1`
3. **向后兼容**：如果不提供 `robot_configs` 参数（默认为 `None`），系统会自动根据机器人名称推断配置

## 测试建议

### 测试 1: 列出配置
```bash
cd examples/openpi
python run_dual_arm_sim_with_config.py --list_configs
```

预期输出应包含 `dexforce_w1`。

### 测试 2: 使用 dexforce_w1 配置
```bash
cd examples/openpi
python run_dual_arm_sim_with_config.py \
    --agent_names your_robot_name \
    --robot_config dexforce_w1 \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

应该能正常启动而不报错。

### 测试 3: 不指定配置（自动推断）
```bash
cd examples/openpi
python run_dual_arm_sim_with_config.py \
    --agent_names your_robot_name \
    --run_mode teleoperation \
    --action_type end_effector_osc \
    --level shop
```

应该能自动推断配置并正常运行。

## 问题排查

如果仍然遇到问题：

1. **检查配置文件是否存在**：
```bash
ls envs/manipulation/robots/configs/dexforce_w1_config.py
```

2. **检查配置内容**：
```bash
cat envs/manipulation/robots/configs/dexforce_w1_config.py
```

3. **测试配置加载**：
```python
from envs.manipulation.robots.configs.robot_config_registry import get_robot_config
config = get_robot_config("test", config_name="dexforce_w1")
print(config)
```

4. **检查 Python 路径**：
确保当前目录在 Python 路径中：
```bash
cd /home/orcash/OrcaGym/OrcaGym
python your_script.py
```

## 总结

✅ 修复了 `robot_configs` 未定义的错误  
✅ 添加了 `dexforce_w1` 配置的显式注册  
✅ 保持了向后兼容性  
✅ 所有功能应该正常工作  

现在您可以使用 `--robot_config dexforce_w1` 参数来运行您的脚本了！

