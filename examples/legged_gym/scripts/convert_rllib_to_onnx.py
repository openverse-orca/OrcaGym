import os
import pickle
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import onnx
import onnxruntime as ort
from pathlib import Path

# RLlib imports
import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import DefaultAPPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.core.columns import Columns
import tree

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class OnnxableRLlibPolicy(nn.Module):
    """将RLlib策略包装为可导出ONNX的PyTorch模块"""
    
    def __init__(self, rllib_module: DefaultAPPOTorchRLModule):
        super().__init__()
        self.rllib_module = rllib_module
        
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        前向传播，只返回动作（用于推理）
        
        Args:
            observation: 观察张量 [batch_size, obs_dim]
            
        Returns:
            action: 动作张量 [batch_size, action_dim]
        """
        # 使用deterministic=True确保输出确定性动作
        result = self.rllib_module.forward_exploration(
            {Columns.OBS: observation}, 
            deterministic=True
        )
        
        # 处理不同的返回格式
        if isinstance(result, dict):
            # 如果返回的是字典，尝试获取动作
            if 'actions' in result:
                action = result['actions']
            elif 'action' in result:
                action = result['action']
            elif 'action_dist_inputs' in result:
                # 对于action_dist_inputs，直接使用作为动作输出
                action = result['action_dist_inputs']
            else:
                # 如果没有直接的action键，尝试从action_dist中获取
                action_dist = result.get('action_dist', result)
                if hasattr(action_dist, 'mean'):
                    action = action_dist.mean
                elif hasattr(action_dist, 'deterministic_sample'):
                    action = action_dist.deterministic_sample()
                else:
                    raise ValueError(f"无法从结果中提取动作: {result}")
        else:
            # 如果返回的是动作分布对象
            action_dist = result
            if hasattr(action_dist, 'mean'):
                action = action_dist.mean
            elif hasattr(action_dist, 'deterministic_sample'):
                action = action_dist.deterministic_sample()
            else:
                raise ValueError(f"无法从动作分布中提取动作: {action_dist}")
            
        return action


def load_rllib_checkpoint(checkpoint_path: str) -> Tuple[DefaultAPPOTorchRLModule, Dict[str, Any]]:
    """
    从RLlib checkpoint加载模型和配置
    
    Args:
        checkpoint_path: checkpoint目录路径
        
    Returns:
        rllib_module: 加载的RLlib模块
        config: 模型配置信息
    """
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
    # 从RL模块规范中获取
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
        # 从策略参数中获取
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
    
    # 创建RL模块规范
    rl_module_spec = RLModuleSpec(
        observation_space=observation_space,
        action_space=action_space,
        module_class=DefaultAPPOTorchRLModule,
        model_config=module_class_and_ctor_args.get("model_config", {})
    )
    
    # 创建RL模块实例 - 使用新的API
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


def create_dummy_observation(observation_space, batch_size: int = 1) -> torch.Tensor:
    """
    创建虚拟观察数据用于ONNX导出
    
    Args:
        observation_space: 观察空间
        batch_size: 批次大小
        
    Returns:
        dummy_obs: 虚拟观察张量
    """
    if hasattr(observation_space, 'shape'):
        # 连续观察空间
        obs_shape = (batch_size,) + observation_space.shape
        dummy_obs = torch.randn(obs_shape, dtype=torch.float32)
        # 裁剪到 [-10, 10]
        dummy_obs = torch.clamp(dummy_obs, -10, 10)
    elif hasattr(observation_space, 'spaces'):
        # 字典观察空间
        dummy_obs = {}
        for key, space in observation_space.spaces.items():
            if hasattr(space, 'shape'):
                obs_shape = (batch_size,) + space.shape
                dummy_obs[key] = torch.randn(obs_shape, dtype=torch.float32)
        # 对于字典观察，我们假设主要观察在'observation'键下
        if 'observation' in dummy_obs:
            dummy_obs = dummy_obs['observation']
        else:
            # 如果没有'observation'键，取第一个
            dummy_obs = list(dummy_obs.values())[0]
    else:
        raise ValueError(f"Unsupported observation space type: {type(observation_space)}")
    
    return dummy_obs


def convert_rllib_to_onnx(checkpoint_path: str, output_path: str, batch_size: int = 1):
    """
    将RLlib checkpoint转换为ONNX模型
    
    Args:
        checkpoint_path: RLlib checkpoint目录路径
        output_path: 输出ONNX文件路径
        batch_size: 批次大小
    """
    _logger.info(f"Loading RLlib checkpoint from: {checkpoint_path}")
    
    # 初始化Ray（如果还没有初始化）
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # 加载RLlib模块
    rllib_module, config = load_rllib_checkpoint(checkpoint_path)
    rllib_module.eval()
    
    _logger.info(f"Model loaded successfully")
    _logger.info(f"Model config: {config}")
    
    # 创建可导出的ONNX模块
    onnx_policy = OnnxableRLlibPolicy(rllib_module)
    onnx_policy.eval()
    
    # 创建虚拟观察数据
    observation_space = rllib_module.config.observation_space
    dummy_obs = create_dummy_observation(observation_space, batch_size)
    
    _logger.info(f"Dummy observation shape: {dummy_obs.shape}")
    
    # 导出为ONNX
    _logger.info(f"Exporting to ONNX: {output_path}")
    
    # 设置动态轴
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    torch.onnx.export(
        onnx_policy,
        args=(dummy_obs,),
        f=output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    _logger.info(f"ONNX model exported successfully to: {output_path}")


def verify_onnx_model(onnx_path: str, checkpoint_path: str, num_tests: int = 5):
    """
    验证ONNX模型的正确性
    
    Args:
        onnx_path: ONNX模型路径
        checkpoint_path: 原始checkpoint路径
        num_tests: 测试次数
    """
    _logger.info(f"Verifying ONNX model: {onnx_path}")
    
    # 加载原始RLlib模型
    rllib_module, _ = load_rllib_checkpoint(checkpoint_path)
    rllib_module.eval()
    
    # 创建ONNX运行时会话
    ort_session = ort.InferenceSession(onnx_path)
    
    # 获取观察空间信息
    observation_space = rllib_module.config.observation_space
    
    # 运行多次测试
    for i in range(num_tests):
        _logger.info(f"Test {i+1}/{num_tests}")
        
        # 创建随机观察
        dummy_obs = create_dummy_observation(observation_space, batch_size=1)
        
        # 运行ONNX模型
        onnx_input = {'input': dummy_obs.numpy()}
        onnx_output = ort_session.run(None, onnx_input)[0]
        
        # 运行原始PyTorch模型
        with torch.no_grad():
            torch_result = rllib_module.forward_exploration(
                {Columns.OBS: dummy_obs}, 
                deterministic=True
            )
            
            # 处理不同的返回格式
            if isinstance(torch_result, dict):
                if 'actions' in torch_result:
                    torch_action = torch_result['actions'].numpy()
                elif 'action' in torch_result:
                    torch_action = torch_result['action'].numpy()
                elif 'action_dist_inputs' in torch_result:
                    torch_action = torch_result['action_dist_inputs'].numpy()
                else:
                    raise ValueError(f"无法从PyTorch结果中提取动作: {torch_result}")
            else:
                if hasattr(torch_result, 'mean'):
                    torch_action = torch_result.mean.numpy()
                elif hasattr(torch_result, 'deterministic_sample'):
                    torch_action = torch_result.deterministic_sample().numpy()
                else:
                    raise ValueError(f"无法从PyTorch结果中提取动作: {torch_result}")
        
        # 比较输出
        _logger.info(f"  ONNX output shape: {onnx_output.shape}")
        _logger.info(f"  PyTorch output shape: {torch_action.shape}")
        
        # 计算差异
        diff = np.abs(onnx_output - torch_action)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        _logger.info(f"  Max difference: {max_diff:.6f}")
        _logger.info(f"  Mean difference: {mean_diff:.6f}")
        
        # 检查差异是否在可接受范围内
        if max_diff > 1e-5:
            _logger.warning(f"  WARNING: Large difference detected!")
        else:
            _logger.info(f"  ✓ Outputs match within tolerance")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="Convert RLlib checkpoint to ONNX model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to RLlib checkpoint directory")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for ONNX model")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for ONNX model (default: 1)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify the converted ONNX model")
    parser.add_argument("--num_tests", type=int, default=5,
                       help="Number of verification tests (default: 5)")
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.checkpoint_path}")
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 转换模型
        convert_rllib_to_onnx(args.checkpoint_path, args.output_path, args.batch_size)
        
        # 验证模型（如果请求）
        if args.verify:
            verify_onnx_model(args.output_path, args.checkpoint_path, args.num_tests)
        
        _logger.info("Conversion completed successfully!")
        
    except Exception as e:
        _logger.error(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()
