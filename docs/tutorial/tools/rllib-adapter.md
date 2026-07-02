# 🤖 RLlib 适配器

OrcaGym 提供 RLlib 适配器，支持分布式 RL 训练。

## 集成方式

```python
from orca_adapters.rllib import appo_catalog

# 使用 APPO 算法训练
```

### 指标回调

```python
from orca_adapters.rllib import metrics_callback
```

## 配置

在 RLlib 配置中指定 OrcaGym 环境：

```python
config = {
 "env": "YourOrcaGymEnv-v0",
 "env_config": {
 "frame_skip": 20,
 "orcagym_addr": "localhost:50051",
 "agent_names": ["agent0"],
 "time_step": 0.001,
 },
 # RLlib 特定配置...
}
```

## 训练示例

OrcaGym + RLlib 的训练循环使用 RLlib 的标准流程，只需将环境注册为 OrcaGym 环境即可。
