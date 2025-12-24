# XBot配置文件

## 📦 策略文件

### policy_example.pt
- **来源**: [humanoid-gym](http://github.com/roboterax/humanoid-gym.git)
- **大小**: 2.02 MB
- **格式**: TorchScript JIT模型
- **用途**: XBot机器人的预训练行走策略

**特点**:
- ✅ 已集成在项目内，无需外部依赖
- ✅ 支持稳定的双足行走
- ✅ 可以在OrcaGym环境中直接加载使用

## 📋 训练配置

### xbot_train_config.yaml
- **用途**: XBot训练的环境配置
- **包含**: 物理参数、奖励设置、训练超参数等

## 🔄 更新策略文件

如果需要使用新训练的策略：

1. 从humanoid-gym训练完成后，导出策略：
```bash
cd /path/to/humanoid-gym
python humanoid/scripts/export_policy.py --task XBotL_free
```

2. 复制到config目录：
```bash
cp humanoid-gym/logs/XBot_ppo/exported/policies/policy_*.pt \
   OrcaGym/examples/xbot/config/policy_example.pt
```

3. 直接运行即可生效：
```bash
python examples/xbot/run_xbot_orca.py
```

