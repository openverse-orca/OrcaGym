#!/usr/bin/env python3
"""
脚本用于从Stable Baselines3 PPO模型中提取PyTorch模型并打印出来
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def print_model_structure(model, model_name="Model"):
    """打印模型结构"""
    print(f"\n{'='*60}")
    print(f"{model_name} 结构:")
    print(f"{'='*60}")
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            print(f"{name}: {module}")
            if hasattr(module, 'weight') and module.weight is not None:
                print(f"  Weight shape: {module.weight.shape}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"  Bias shape: {module.bias.shape}")
            print(f"  Parameters: {params:,}")
            print()
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"{'='*60}\n")

def print_model_parameters(model, model_name="Model"):
    """打印模型参数"""
    print(f"\n{'='*60}")
    print(f"{model_name} 参数详情:")
    print(f"{'='*60}")
    
    for name, param in model.named_parameters():
        print(f"{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Data type: {param.dtype}")
        print(f"  Requires grad: {param.requires_grad}")
        print(f"  Device: {param.device}")
        if param.numel() <= 20:  # 如果参数数量少，打印具体值
            print(f"  Values: {param.data}")
        else:
            print(f"  Min: {param.data.min().item():.6f}")
            print(f"  Max: {param.data.max().item():.6f}")
            print(f"  Mean: {param.data.mean().item():.6f}")
            print(f"  Std: {param.data.std().item():.6f}")
        print()

def extract_pytorch_model_from_sb3(model_path=None, config_path=None):
    """从Stable Baselines3模型提取PyTorch模型"""
    
    if model_path is None:
        # 查找最新的模型文件
        trained_models_dir = Path(__file__).parent.parent / "trained_models_tmp"
        if not trained_models_dir.exists():
            print("未找到训练模型目录")
            return None, None
        
        # 获取有.zip文件的模型目录
        model_dirs = [d for d in trained_models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            print("未找到任何训练模型")
            return None, None
        
        # 查找包含.zip文件的目录
        model_dir_with_zip = None
        for model_dir in model_dirs:
            zip_files = list(model_dir.glob("*.zip"))
            if zip_files:
                model_dir_with_zip = model_dir
                break
        
        if model_dir_with_zip is None:
            print("未找到包含.zip文件的模型目录")
            return None, None
        
        print(f"使用模型目录: {model_dir_with_zip}")
        
        # 查找.zip文件
        zip_files = list(model_dir_with_zip.glob("*.zip"))
        model_path = zip_files[0]
        config_path = model_dir_with_zip / "config.json"
    
    print(f"加载模型: {model_path}")
    print(f"加载配置: {config_path}")
    
    # 加载配置
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"模型配置: {json.dumps(config, indent=2)}")
    
    # 加载SB3模型
    try:
        sb3_model = PPO.load(str(model_path))
        print("成功加载Stable Baselines3模型")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None
    
    # 提取PyTorch模型
    policy = sb3_model.policy
    print(f"策略类型: {type(policy)}")
    
    # 获取actor和critic网络
    actor_net = policy.mlp_extractor.policy_net
    critic_net = policy.mlp_extractor.value_net
    action_net = policy.action_net
    value_net = policy.value_net
    
    print("\n提取的PyTorch模型组件:")
    print(f"Actor网络: {type(actor_net)}")
    print(f"Critic网络: {type(critic_net)}")
    print(f"Action网络: {type(action_net)}")
    print(f"Value网络: {type(value_net)}")
    
    return {
        'policy': policy,
        'actor_net': actor_net,
        'critic_net': critic_net,
        'action_net': action_net,
        'value_net': value_net,
        'sb3_model': sb3_model
    }, config

def create_standalone_pytorch_model(policy):
    """创建独立的PyTorch模型"""
    
    class StandalonePolicy(nn.Module):
        def __init__(self, sb3_policy):
            super().__init__()
            self.features_extractor = sb3_policy.features_extractor
            self.mlp_extractor = sb3_policy.mlp_extractor
            self.action_net = sb3_policy.action_net
            self.value_net = sb3_policy.value_net
            
        def forward(self, obs):
            # 特征提取
            features = self.features_extractor(obs)
            
            # 策略和价值网络
            latent_pi = self.mlp_extractor.forward_actor(features)
            latent_vf = self.mlp_extractor.forward_critic(features)
            
            # 动作分布
            mean_actions = self.action_net(latent_pi)
            values = self.value_net(latent_vf)
            
            return mean_actions, values
    
    return StandalonePolicy(policy)

def main():
    """主函数"""
    print("Stable Baselines3 PPO模型转PyTorch格式")
    print("="*60)
    
    # 提取模型
    models, config = extract_pytorch_model_from_sb3()
    
    if models is None:
        print("模型提取失败")
        return
    
    # 打印各个组件的结构
    print_model_structure(models['actor_net'], "Actor网络")
    print_model_structure(models['critic_net'], "Critic网络")
    print_model_structure(models['action_net'], "Action网络")
    print_model_structure(models['value_net'], "Value网络")
    
    # 创建独立模型
    standalone_model = create_standalone_pytorch_model(models['policy'])
    print_model_structure(standalone_model, "独立PyTorch模型")
    
    # 打印参数详情
    print_model_parameters(standalone_model, "独立PyTorch模型参数")
    
    # 保存PyTorch模型
    output_path = Path(__file__).parent / "extracted_pytorch_model.pth"
    torch.save(standalone_model.state_dict(), output_path)
    print(f"PyTorch模型已保存到: {output_path}")
    
    # 测试模型推理
    print("\n测试模型推理:")
    print("-" * 40)
    
    # 创建示例输入
    device = next(standalone_model.parameters()).device
    obs_space = models['sb3_model'].observation_space
    
    # 创建示例观察
    if hasattr(obs_space, 'spaces'):  # Dict观察空间
        sample_obs = {}
        for key, space in obs_space.spaces.items():
            sample_obs[key] = torch.randn(1, *space.shape).to(device)
    else:  # Box观察空间
        sample_obs = torch.randn(1, *obs_space.shape).to(device)
    
    print(f"输入观察形状: {sample_obs}")
    
    # 推理
    with torch.no_grad():
        actions, values = standalone_model(sample_obs)
        print(f"输出动作形状: {actions.shape}")
        print(f"输出价值形状: {values.shape}")
        print(f"动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        print(f"价值范围: [{values.min().item():.3f}, {values.max().item():.3f}]")

if __name__ == "__main__":
    main()