from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.base import Encoder, ActorCriticEncoder
from ray.rllib.core.models.configs import MLPHeadConfig, MLPEncoderConfig, ActorCriticEncoderConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.utils.numpy import convert_to_numpy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

class DictAPPOCatalog(PPOCatalog):
    """自定义 Catalog 处理字典观测空间"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _get_encoder_config(
            self,) -> MLPEncoderConfig:
        """为字典观测空间创建多分支编码器配置"""
        # 跳过父类方法，不调用 super()
        if not isinstance(self.observation_space, gym.spaces.Dict):
            return super()._get_encoder_config(framework)
        
        # 创建多分支配置
        branch_configs = {}
        for key, subspace in self.observation_space.spaces.items():
            branch_configs[key] = MLPEncoderConfig(
                input_dim=subspace.shape[0],
                hidden_layer_dims=self.model_config_dict["fcnet_hiddens"],
                hidden_layer_activation=self.model_config_dict["fcnet_activation"],
                output_dim=self.model_config_dict["fcnet_hiddens"][-1],
                output_activation=self.model_config_dict["post_fcnet_activation"],
            )
        
        # 返回为字典空间定制的编码器配置
        return DictEncoderConfig(branch_configs=branch_configs)
    
    def build_actor_critic_encoder(self, framework: str) -> ActorCriticEncoder:
        """构建处理字典空间的演员-评论家编码器"""
        if not isinstance(self.observation_space, gym.spaces.Dict):
            return super().build_actor_critic_encoder(framework)
        
        encoder_config = self._get_encoder_config(framework)
        encoder = encoder_config.build(framework)
        
        return DictActorCriticEncoder(
            encoder=encoder,
            action_space=self.action_space,
            model_config_dict=self.model_config_dict,
        )

class DictEncoderConfig(MLPEncoderConfig):
    """字典观测的编码器配置"""
    
    def __init__(self, branch_configs: dict):
        super().__init__(input_dim=None)
        self.branch_configs = branch_configs
        
    def build(self, framework: str = "torch") -> "DictEncoder":
        return DictEncoder(self)

class DictEncoder(Encoder):
    """处理字典观测的编码器模块"""
    
    def __init__(self, config: DictEncoderConfig):
        super().__init__()
        self.branches = nn.ModuleDict()
        
        # 为观测字典的每个键创建独立的MLP分支
        for key, branch_config in config.branch_configs.items():
            mlp_config = MLPEncoderConfig(
                input_dim=branch_config.input_dim,
                hidden_layer_dims=branch_config.hidden_layer_dims,
                hidden_layer_activation=branch_config.hidden_layer_activation,
                output_dim=branch_config.output_dim,
                output_activation=branch_config.output_activation
            )
            self.branches[key] = mlp_config.build(framework="torch")
        
        # 计算总的输出维度
        self.output_dim = sum(
            branch_config.output_dim 
            for branch_config in config.branch_configs.values()
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @output_dim.setter
    def output_dim(self, value):
        self._output_dim = value
        
    def _forward(self, inputs: dict, **kwargs) -> torch.Tensor:
        branch_outputs = []
        
        # 处理每个观测分支
        for key, branch in self.branches.items():
            # 确保输入中包含所有键
            if key not in inputs:
                raise ValueError(f"Observation missing required key: {key}")
                
            # 获取输入并确保正确的形状
            obs_data = inputs[key]
            if not isinstance(obs_data, torch.Tensor):
                obs_data = torch.as_tensor(obs_data, dtype=torch.float32)
                
            # 通过分支处理
            branch_out = branch(obs_data)
            branch_outputs.append(branch_out)
        
        # 沿着特征维度拼接所有分支输出
        return torch.cat(branch_outputs, dim=-1)

class DictActorCriticEncoder(ActorCriticEncoder):
    """处理字典观测空间的演员-评论家编码器"""
    
    def __init__(self, encoder: Encoder, *args, **kwargs):
        super().__init__(encoder, *args, **kwargs)
    
    def __call__(self, inputs: dict, **kwargs) -> dict:
        # 通过共享编码器处理字典观测
        encoder_out = self.encoder(inputs)
        
        # 返回统一格式的编码输出
        return {
            ENCODER_OUT: encoder_out,
            "actor_out": encoder_out,
            "critic_out": encoder_out,
        }

class DictAPPOTorchRLModule(DefaultPPOTorchRLModule):
    """处理字典观测的自定义PPO RL模块"""
    
    def setup(self):
        # 使用我们的自定义Catalog
        self.config.catalog_class = DictAPPOCatalog
        self.config.catalog = self.config.get_catalog()
        catalog = self.config.catalog
        
        # 确保共享编码器用于演员和评论家
        encoder_config = ActorCriticEncoderConfig()

        # 构建共享编码器
        self.encoder = catalog.build_actor_critic_encoder(
            encoder_config=encoder_config,
            framework=self.framework
        )

        # 构建演员和评论家头部
        actor_head = catalog.build_actor_head(framework=self.framework)
        critic_head = catalog.build_critic_head(framework=self.framework)

        # 配置值函数共享
        self.pi = actor_head(self.encoder.output_dim)
        self.vf = critic_head(self.encoder.output_dim)