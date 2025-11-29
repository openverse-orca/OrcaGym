#!/usr/bin/env python3
"""
脚本用于从Stable Baselines3 PPO模型和RLLib APPO模型中提取PyTorch模型并打印出来
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
import json
import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


# RLLib imports
try:
    import ray
    from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import DefaultAPPOTorchRLModule
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec, RLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.utils.numpy import convert_to_numpy
    RLLIB_AVAILABLE = True
except ImportError:
    _logger.warning("Warning: RLLib not available. RLLib checkpoint support will be disabled.")
    RLLIB_AVAILABLE = False

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))


def parse_model_config_string(model_config_str: str) -> Dict[str, Any]:
    """
    解析 DefaultModelConfig 字符串为字典
    
    Args:
        model_config_str: 如 "DefaultModelConfig(fcnet_hiddens=[512, 256, 128], ...)"
        
    Returns:
        解析后的配置字典
    """
    if not model_config_str:
        return {}
    
    import re
    import ast
    
    # 提取括号内的参数
    match = re.search(r'DefaultModelConfig\((.*)\)$', model_config_str.strip())
    if not match:
        return {}
    
    params_str = match.group(1)
    return _parse_parameters(params_str)


def _parse_parameters(params_str: str) -> Dict[str, Any]:
    """解析参数字符串"""
    import re
    import ast
    
    config = {}
    params = _split_parameters(params_str)
    
    for param in params:
        if '=' not in param:
            continue
            
        key, value_str = param.split('=', 1)
        key = key.strip()
        value_str = value_str.strip()
        
        config[key] = _parse_value(value_str)
    
    return config


def _split_parameters(params_str: str) -> list:
    """智能分割参数字符串，处理嵌套结构"""
    params = []
    current_param = ""
    bracket_count = 0
    square_bracket_count = 0
    curly_bracket_count = 0
    in_quotes = False
    quote_char = None
    
    for char in params_str:
        if not in_quotes:
            if char in ['"', "'"]:
                in_quotes = True
                quote_char = char
            elif char == '(':
                bracket_count += 1
            elif char == ')':
                bracket_count -= 1
            elif char == '[':
                square_bracket_count += 1
            elif char == ']':
                square_bracket_count -= 1
            elif char == '{':
                curly_bracket_count += 1
            elif char == '}':
                curly_bracket_count -= 1
            elif char == ',' and bracket_count == 0 and square_bracket_count == 0 and curly_bracket_count == 0:
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
                continue
        else:
            if char == quote_char:
                in_quotes = False
                quote_char = None
        
        current_param += char
    
    if current_param.strip():
        params.append(current_param.strip())
    
    return params


def _parse_value(value_str: str) -> Any:
    """解析单个参数值"""
    import ast
    
    value_str = value_str.strip()
    
    if value_str == 'None':
        return None
    elif value_str in ['True', 'False']:
        return value_str == 'True'
    elif (value_str.startswith('"') and value_str.endswith('"')) or \
         (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]
    elif value_str.startswith('[') and value_str.endswith(']'):
        try:
            return ast.literal_eval(value_str)
        except:
            return value_str
    elif value_str.startswith('{') and value_str.endswith('}'):
        try:
            return ast.literal_eval(value_str)
        except:
            return value_str
    else:
        try:
            return float(value_str) if '.' in value_str else int(value_str)
        except ValueError:
            return value_str

def print_model_structure(model, model_name="Model"):
    """打印模型结构"""
    _logger.info(f"\n{'='*60}")
    _logger.info(f"{model_name} 结构:")
    _logger.info(f"{'='*60}")
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            _logger.info(f"{name}: {module}")
            if hasattr(module, 'weight') and module.weight is not None:
                _logger.info(f"  Weight shape: {module.weight.shape}")
            if hasattr(module, 'bias') and module.bias is not None:
                _logger.info(f"  Bias shape: {module.bias.shape}")
            _logger.info(f"  Parameters: {params:,}")
            print()
    
    _logger.info(f"总参数数量: {total_params:,}")
    _logger.info(f"可训练参数数量: {trainable_params:,}")
    _logger.info(f"{'='*60}\n")

def print_model_parameters(model, model_name="Model"):
    """打印模型参数"""
    _logger.info(f"\n{'='*60}")
    _logger.info(f"{model_name} 参数详情:")
    _logger.info(f"{'='*60}")
    
    for name, param in model.named_parameters():
        _logger.info(f"{name}:")
        _logger.info(f"  Shape: {param.shape}")
        _logger.info(f"  Data type: {param.dtype}")
        _logger.info(f"  Requires grad: {param.requires_grad}")
        _logger.info(f"  Device: {param.device}")
        if param.numel() <= 20:  # 如果参数数量少，打印具体值
            _logger.info(f"  Values: {param.data}")
        else:
            _logger.info(f"  Min: {param.data.min().item():.6f}")
            _logger.info(f"  Max: {param.data.max().item():.6f}")
            _logger.info(f"  Mean: {param.data.mean().item():.6f}")
            _logger.info(f"  Std: {param.data.std().item():.6f}")
        print()

def load_rllib_checkpoint(checkpoint_path: str) -> Tuple[DefaultAPPOTorchRLModule, Dict[str, Any]]:
    """
    从RLLib checkpoint加载模型和配置
    
    Args:
        checkpoint_path: checkpoint目录路径
        
    Returns:
        rllib_module: 加载的RLLib模块
        config: 模型配置信息
    """
    if not RLLIB_AVAILABLE:
        raise ImportError("RLLib is not available. Please install ray[rllib] to use this functionality.")
    
    checkpoint_path = Path(checkpoint_path)
    
    # 读取checkpoint元数据
    with open(checkpoint_path / "rllib_checkpoint.json", "r") as f:
        checkpoint_meta = json.load(f)
    
    # 读取算法状态
    with open(checkpoint_path / "algorithm_state.pkl", "rb") as f:
        algorithm_state = pickle.load(f)
    
    # 读取类构造函数参数
    with open(checkpoint_path / "class_and_ctor_args.pkl", "rb") as f:
        class_and_ctor_args = pickle.load(f)
    
    # 读取RL模块状态
    rl_module_path = checkpoint_path / "learner_group" / "learner" / "rl_module"
    with open(rl_module_path / "module_state.pkl", "rb") as f:
        module_state = pickle.load(f)
    
    with open(rl_module_path / "class_and_ctor_args.pkl", "rb") as f:
        module_class_and_ctor_args = pickle.load(f)
    
    # 读取策略状态
    policy_path = rl_module_path / "default_policy"
    with open(policy_path / "module_state.pkl", "rb") as f:
        policy_state = pickle.load(f)
    
    with open(policy_path / "class_and_ctor_args.pkl", "rb") as f:
        policy_class_and_ctor_args = pickle.load(f)
    
    # 从配置中提取观察空间和动作空间信息
    rl_module_specs = module_class_and_ctor_args.get("rl_module_specs", {})
    if "default_policy" in rl_module_specs:
        default_policy_spec = rl_module_specs["default_policy"]
        observation_space = default_policy_spec.observation_space
        action_space = default_policy_spec.action_space
    else:
        # 尝试从算法状态中获取
        observation_space = algorithm_state.get("observation_space")
        action_space = algorithm_state.get("action_space")
    
    # 如果还是None，尝试从策略参数中获取
    if observation_space is None or action_space is None:
        policy_class_and_ctor_args = policy_class_and_ctor_args
        if policy_class_and_ctor_args:
            ctor_args_and_kwargs = policy_class_and_ctor_args.get("ctor_args_and_kwargs", ({}, {}))
            if isinstance(ctor_args_and_kwargs, tuple) and len(ctor_args_and_kwargs) >= 2:
                kwargs = ctor_args_and_kwargs[1]
                if isinstance(kwargs, dict):
                    observation_space = kwargs.get("observation_space")
                    action_space = kwargs.get("action_space")
    
    _logger.info(f"Observation space: {observation_space}")
    _logger.info(f"Action space: {action_space}")
    
    if observation_space is None or action_space is None:
        raise ValueError("无法从checkpoint中提取观察空间或动作空间信息")
    
    # 从配置文件中读取 model_config
    checkpoint_parent_path = checkpoint_path.parent
    with open(checkpoint_parent_path / "params.json", "r") as f:
        config = json.load(f)
        model_config_str = config.get("_model_config", "")
    
    # 解析 model_config 字符串为字典
    model_config = parse_model_config_string(model_config_str)
    _logger.info(f"解析的模型配置: {model_config}")
    
    # 创建RL模块规范
    rl_module_spec = RLModuleSpec(
        observation_space=observation_space,
        action_space=action_space,
        module_class=DefaultAPPOTorchRLModule,
        model_config=model_config
    )
    
    # 创建RL模块实例
    module_class = rl_module_spec.module_class
    rllib_module = module_class(
        observation_space=rl_module_spec.observation_space,
        action_space=rl_module_spec.action_space,
        inference_only=rl_module_spec.inference_only,
        learner_only=rl_module_spec.learner_only,
        model_config=rl_module_spec.model_config
    )
    
    # 加载状态 - 处理键名不匹配的问题
    try:
        rllib_module.load_state_dict(policy_state)
        _logger.info("成功直接加载状态字典")
    except RuntimeError as e:
        _logger.info(f"直接加载状态失败，尝试修复键名: {e}")
        
        # 创建新的状态字典，修复键名
        new_state_dict = {}
        for key, value in policy_state.items():
            # 将numpy数组转换为torch张量，创建可写副本避免内存共享警告
            if isinstance(value, np.ndarray):
                # 创建可写的副本，避免内存共享警告
                value = torch.from_numpy(value.copy())
            
            # 修复键名映射
            new_key = key
            if key.startswith("_old_encoder."):
                new_key = key.replace("_old_encoder.", "encoder.")
            elif key.startswith("_old_pi."):
                new_key = key.replace("_old_pi.", "pi.")
            
            new_state_dict[new_key] = value
        
        # 尝试加载修复后的状态字典
        try:
            rllib_module.load_state_dict(new_state_dict, strict=False)
            _logger.info("成功加载修复后的状态字典")
        except Exception as e2:
            _logger.info(f"修复后加载仍然失败: {e2}")
            # 如果还是失败，尝试只加载匹配的键
            model_state_dict = rllib_module.state_dict()
            compatible_state_dict = {}
            incompatible_keys = []
            
            for key, value in new_state_dict.items():
                if key in model_state_dict:
                    if value.shape == model_state_dict[key].shape:
                        compatible_state_dict[key] = value
                    else:
                        incompatible_keys.append(f"{key}: checkpoint shape {value.shape} vs model shape {model_state_dict[key].shape}")
                else:
                    incompatible_keys.append(f"{key}: key not found in model")
            
            if incompatible_keys:
                _logger.info(f"发现 {len(incompatible_keys)} 个不兼容的参数:")
                for key_info in incompatible_keys[:5]:  # 只显示前5个
                    _logger.info(f"  - {key_info}")
                if len(incompatible_keys) > 5:
                    _logger.info(f"  ... 还有 {len(incompatible_keys) - 5} 个不兼容参数")
            
            rllib_module.load_state_dict(compatible_state_dict, strict=False)
            _logger.info(f"成功加载 {len(compatible_state_dict)} 个兼容的参数")
    
    return rllib_module, module_class_and_ctor_args


def extract_pytorch_model_from_rllib(checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """从RLLib checkpoint提取PyTorch模型"""
    
    if not RLLIB_AVAILABLE:
        _logger.info("RLLib不可用，无法加载RLLib checkpoint")
        return None, None
    
    _logger.info(f"加载RLLib checkpoint: {checkpoint_path}")
    
    try:
        # 加载RLLib模块
        rllib_module, config = load_rllib_checkpoint(checkpoint_path)
        _logger.info("成功加载RLLib模块")
        
        # 提取模型组件
        models = {
            'rllib_module': rllib_module,
            'encoder': getattr(rllib_module, 'encoder', None),
            'pi': getattr(rllib_module, 'pi', None),
            'vf': getattr(rllib_module, 'vf', None),
        }
        
        # 打印可用的组件
        _logger.info("\n可用的模型组件:")
        for name, component in models.items():
            if component is not None:
                _logger.info(f"  {name}: {type(component)}")
            else:
                _logger.info(f"  {name}: None")
        
        return models, config
        
    except Exception as e:
        _logger.info(f"加载RLLib checkpoint失败: {e}")
        return None, None


def extract_pytorch_model_from_sb3(model_path=None, config_path=None):
    """从Stable Baselines3模型提取PyTorch模型"""
    
    if model_path is None:
        # 查找最新的模型文件
        trained_models_dir = Path(__file__).parent.parent / "trained_models_tmp"
        if not trained_models_dir.exists():
            _logger.info("未找到训练模型目录")
            return None, None
        
        # 获取有.zip文件的模型目录
        model_dirs = [d for d in trained_models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            _logger.info("未找到任何训练模型")
            return None, None
        
        # 查找包含.zip文件的目录
        model_dir_with_zip = None
        for model_dir in model_dirs:
            zip_files = list(model_dir.glob("*.zip"))
            if zip_files:
                model_dir_with_zip = model_dir
                break
        
        if model_dir_with_zip is None:
            _logger.info("未找到包含.zip文件的模型目录")
            return None, None
        
        _logger.info(f"使用模型目录: {model_dir_with_zip}")
        
        # 查找.zip文件
        zip_files = list(model_dir_with_zip.glob("*.zip"))
        model_path = zip_files[0]
        config_path = model_dir_with_zip / "config.json"
    
    _logger.info(f"加载模型: {model_path}")
    _logger.info(f"加载配置: {config_path}")
    
    # 加载配置
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        _logger.info(f"模型配置: {json.dumps(config, indent=2)}")
    
    # 加载SB3模型
    try:
        sb3_model = PPO.load(str(model_path))
        _logger.info("成功加载Stable Baselines3模型")
    except Exception as e:
        _logger.info(f"加载模型失败: {e}")
        return None, None
    
    # 提取PyTorch模型
    policy = sb3_model.policy
    _logger.info(f"策略类型: {type(policy)}")
    
    # 获取actor和critic网络
    actor_net = policy.mlp_extractor.policy_net
    critic_net = policy.mlp_extractor.value_net
    action_net = policy.action_net
    value_net = policy.value_net
    
    _logger.info("\n提取的PyTorch模型组件:")
    _logger.info(f"Actor网络: {type(actor_net)}")
    _logger.info(f"Critic网络: {type(critic_net)}")
    _logger.info(f"Action网络: {type(action_net)}")
    _logger.info(f"Value网络: {type(value_net)}")
    
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


def create_standalone_rllib_model(rllib_module):
    """创建独立的RLLib PyTorch模型"""
    
    class StandaloneRLLibPolicy(nn.Module):
        def __init__(self, rllib_module):
            super().__init__()
            self.rllib_module = rllib_module
            
        def forward(self, obs):
            """
            前向传播，返回动作和价值
            
            Args:
                obs: 观察张量 [batch_size, obs_dim]
                
            Returns:
                actions: 动作张量 [batch_size, action_dim]
                values: 价值张量 [batch_size, 1]
            """
            # 使用deterministic=True确保输出确定性动作
            result = self.rllib_module.forward_exploration(
                {Columns.OBS: obs}, 
                deterministic=True
            )
            
            # 提取动作
            if isinstance(result, dict):
                if 'actions' in result:
                    actions = result['actions']
                elif 'action' in result:
                    actions = result['action']
                elif 'action_dist_inputs' in result:
                    actions = result['action_dist_inputs']
                else:
                    action_dist = result.get('action_dist', result)
                    if hasattr(action_dist, 'mean'):
                        actions = action_dist.mean
                    elif hasattr(action_dist, 'deterministic_sample'):
                        actions = action_dist.deterministic_sample()
                    else:
                        raise ValueError(f"无法从结果中提取动作: {result}")
            else:
                action_dist = result
                if hasattr(action_dist, 'mean'):
                    actions = action_dist.mean
                elif hasattr(action_dist, 'deterministic_sample'):
                    actions = action_dist.deterministic_sample()
                else:
                    raise ValueError(f"无法从动作分布中提取动作: {action_dist}")
            
            # 获取价值估计
            try:
                value_result = self.rllib_module.forward_inference(
                    {Columns.OBS: obs}
                )
                if isinstance(value_result, dict):
                    values = value_result.get('vf_preds', value_result.get('value', None))
                    if values is None:
                        # 尝试从其他可能的键获取价值
                        for key in ['vf_preds', 'value', 'values', 'vf']:
                            if key in value_result:
                                values = value_result[key]
                                break
                else:
                    values = value_result
                
                # 如果仍然无法获取价值，返回零
                if values is None:
                    values = torch.zeros(obs.shape[0], 1, device=obs.device)
                    
            except Exception as e:
                _logger.info(f"获取价值估计失败: {e}")
                # 如果无法获取价值，返回零
                values = torch.zeros(obs.shape[0], 1, device=obs.device)
            
            return actions, values
    
    return StandaloneRLLibPolicy(rllib_module)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从SB3或RLLib checkpoint提取PyTorch模型")
    parser.add_argument("--checkpoint", type=str, help="模型checkpoint路径")
    parser.add_argument("--type", type=str, choices=["sb3", "rllib"], 
                       help="模型类型: sb3 或 rllib")
    parser.add_argument("--output", type=str, help="输出PyTorch模型路径")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="只分析模型结构，不保存模型")
    
    args = parser.parse_args()
    
    _logger.info("PyTorch模型提取和分析工具")
    _logger.info("="*60)
    
    # 确定模型类型和路径
    if args.type and args.checkpoint:
        model_type = args.type
        checkpoint_path = args.checkpoint
    else:
        # 自动检测模型类型
        trained_models_dir = Path(__file__).parent.parent / "trained_models_tmp"
        if not trained_models_dir.exists():
            _logger.info("未找到训练模型目录")
            return
        
        # 查找最新的模型文件
        model_dirs = [d for d in trained_models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            _logger.info("未找到任何训练模型")
            return
        
        # 优先查找RLLib checkpoint
        rllib_checkpoint = None
        sb3_model = None
        
        for model_dir in model_dirs:
            # 查找RLLib checkpoint
            rllib_dirs = list(model_dir.glob("APPO_*"))
            if rllib_dirs:
                latest_rllib = max(rllib_dirs, key=lambda x: x.stat().st_mtime)
                checkpoint_dirs = list(latest_rllib.glob("checkpoint_*"))
                if checkpoint_dirs:
                    rllib_checkpoint = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)
                    break
            
            # 查找SB3模型
            zip_files = list(model_dir.glob("*.zip"))
            if zip_files:
                sb3_model = zip_files[0]
                break
        
        if rllib_checkpoint:
            model_type = "rllib"
            checkpoint_path = str(rllib_checkpoint)
            _logger.info(f"自动检测到RLLib checkpoint: {checkpoint_path}")
        elif sb3_model:
            model_type = "sb3"
            checkpoint_path = str(sb3_model)
            _logger.info(f"自动检测到SB3模型: {checkpoint_path}")
        else:
            _logger.info("未找到支持的模型文件")
            return
    
    # 根据模型类型提取模型
    if model_type == "sb3":
        _logger.info("处理Stable Baselines3模型...")
        models, config = extract_pytorch_model_from_sb3(checkpoint_path)
        
        if models is None:
            _logger.info("SB3模型提取失败")
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
        
        # 测试模型推理
        _logger.info("\n测试模型推理:")
        _logger.info("-" * 40)
        
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
        
        _logger.info(f"输入观察形状: {sample_obs}")
        
        # 推理
        with torch.no_grad():
            actions, values = standalone_model(sample_obs)
            _logger.info(f"输出动作形状: {actions.shape}")
            _logger.info(f"输出价值形状: {values.shape}")
            _logger.info(f"动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
            _logger.info(f"价值范围: [{values.min().item():.3f}, {values.max().item():.3f}]")
    
    elif model_type == "rllib":
        _logger.info("处理RLLib APPO模型...")
        models, config = extract_pytorch_model_from_rllib(checkpoint_path)
        
        if models is None:
            _logger.info("RLLib模型提取失败")
            return
        
        # 打印各个组件的结构
        if models['encoder'] is not None:
            print_model_structure(models['encoder'], "编码器网络")
        if models['pi'] is not None:
            print_model_structure(models['pi'], "策略网络")
        if models['vf'] is not None:
            print_model_structure(models['vf'], "价值网络")
        
        # 创建独立模型
        standalone_model = create_standalone_rllib_model(models['rllib_module'])
        print_model_structure(standalone_model, "独立RLLib PyTorch模型")
        
        # 打印参数详情
        print_model_parameters(standalone_model, "独立RLLib PyTorch模型参数")
        
        # 测试模型推理
        _logger.info("\n测试模型推理:")
        _logger.info("-" * 40)
        
        # 创建示例输入
        device = next(standalone_model.parameters()).device
        obs_space = models['rllib_module'].observation_space
        
        # 创建示例观察
        if hasattr(obs_space, 'shape'):
            sample_obs = torch.randn(1, *obs_space.shape).to(device)
        else:
            # 对于复杂的观察空间，使用默认形状
            sample_obs = torch.randn(1, 48).to(device)  # 假设48维观察
        
        _logger.info(f"输入观察形状: {sample_obs.shape}")
        
        # 推理
        with torch.no_grad():
            actions, values = standalone_model(sample_obs)
            _logger.info(f"输出动作形状: {actions.shape}")
            _logger.info(f"输出价值形状: {values.shape}")
            _logger.info(f"动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
            _logger.info(f"价值范围: [{values.min().item():.3f}, {values.max().item():.3f}]")
    
    # 保存PyTorch模型
    if not args.analyze_only:
        if args.output:
            output_path = Path(args.output)
        else:
            # 将.pth文件保存在输入checkpoint的目录下
            checkpoint_dir = Path(args.checkpoint).parent
            output_path = checkpoint_dir / f"extracted_{model_type}_pytorch_model.pth"
        
        torch.save(standalone_model.state_dict(), output_path)
        _logger.info(f"\nPyTorch模型已保存到: {output_path}")
    else:
        _logger.info("\n仅分析模式，未保存模型")

if __name__ == "__main__":
    main()