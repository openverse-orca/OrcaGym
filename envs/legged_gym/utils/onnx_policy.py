"""
ONNX策略加载器 - 迁移自Lite3_rl_deploy
用于加载和运行ONNX格式的策略模型

参考文件:
- Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp
"""

import onnxruntime as ort
import numpy as np
from typing import Union, Optional
import os

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class ONNXPolicy:
    """
    ONNX策略加载和推理类
    
    参考: lite3_test_policy_runner_onnx.hpp
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        num_threads: int = 1,
        providers: Optional[list] = None
    ):
        """
        初始化ONNX策略
        
        Args:
            model_path: ONNX模型文件路径
            device: 设备类型 ('cpu' 或 'cuda')
            num_threads: 线程数
            providers: ONNX Runtime提供者列表
        """
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
        
        self.device = device
        
        # 设置ONNX Runtime会话选项
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 设置提供者
        if providers is None:
            if device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        # 创建ONNX Runtime会话
        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {model_path}: {e}")
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        
        _logger.info(f"[ONNX Policy] Model loaded: {model_path}")
        _logger.info(f"[ONNX Policy] Input: {self.input_name}, shape: {input_shape}")
        _logger.info(f"[ONNX Policy] Output: {self.output_name}, shape: {output_shape}")
        _logger.info(f"[ONNX Policy] Providers: {self.session.get_providers()}")
        
        # 检查是否支持动态batch
        # 如果第一个维度是None或字符串（如'batch'），则支持动态batch
        # 如果第一个维度是整数，则不支持动态batch（固定batch_size）
        if len(input_shape) > 0:
            first_dim = input_shape[0]
            self.supports_dynamic_batch = (
                first_dim is None or isinstance(first_dim, str)
            )
            self.fixed_batch_size = first_dim if isinstance(first_dim, int) else None
        else:
            self.supports_dynamic_batch = False
            self.fixed_batch_size = None
        
        if not self.supports_dynamic_batch:
            if self.fixed_batch_size is not None:
                print(f"[ONNX Policy] Model has fixed batch_size={self.fixed_batch_size}, "
                      f"will process batch inputs sequentially")
            else:
                _logger.info(f"[ONNX Policy] Model batch size unknown, will process batch inputs sequentially")
        else:
            _logger.info(f"[ONNX Policy] Model supports dynamic batch size")
        
        # 验证输入输出维度
        if len(input_shape) >= 2 and input_shape[1] != 45:
            _logger.warning(f"[WARNING] Expected input dimension 45, got {input_shape[1]}")
        if len(output_shape) >= 2 and output_shape[1] != 12:
            _logger.warning(f"[WARNING] Expected output dimension 12, got {output_shape[1]}")
    
    def __call__(self, obs: Union[np.ndarray, list]) -> np.ndarray:
        """
        运行策略推理
        
        Args:
            obs: 观测 [N, 45] 或 [45]
        
        Returns:
            actions: 动作 [N, 12] 或 [12]
        """
        # 转换为numpy数组
        if isinstance(obs, list):
            obs_np = np.array(obs, dtype=np.float32)
        else:
            obs_np = np.asarray(obs, dtype=np.float32)
        
        # 确保是float32类型
        obs_np = obs_np.astype(np.float32)
        
        # 处理单样本情况
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        # 验证维度
        if obs_np.shape[1] != 45:
            raise ValueError(f"Expected observation dimension 45, got {obs_np.shape[1]}")
        
        batch_size = obs_np.shape[0]
        
        # 如果模型不支持动态batch且输入是批量数据，需要逐个处理
        if not self.supports_dynamic_batch and batch_size > 1:
            # 逐个处理每个样本
            actions_list = []
            for i in range(batch_size):
                single_obs = obs_np[i:i+1]  # 保持2D形状 [1, 45]
                try:
                    outputs = self.session.run(
                        [self.output_name],
                        {self.input_name: single_obs}
                    )
                    actions_list.append(outputs[0][0])  # [12]
                except Exception as e:
                    raise RuntimeError(f"ONNX inference failed for sample {i}: {e}")
            
            # 拼接结果
            actions_np = np.stack(actions_list, axis=0)  # [N, 12]
        else:
            # 支持动态batch或单样本，直接处理
            try:
                outputs = self.session.run(
                    [self.output_name],
                    {self.input_name: obs_np}
                )
            except Exception as e:
                raise RuntimeError(f"ONNX inference failed: {e}")
            
            # 获取输出
            actions_np = outputs[0]  # [N, 12]
        
        # 如果输入是单样本，返回单样本
        if single_sample:
            actions_np = actions_np.squeeze(0)
        
        return actions_np
    
    def get_info(self) -> dict:
        """
        获取策略信息
        
        Returns:
            包含策略信息的字典
        """
        return {
            "model_path": self.model_path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_shape": self.session.get_inputs()[0].shape,
            "output_shape": self.session.get_outputs()[0].shape,
            "providers": self.session.get_providers(),
        }


def load_onnx_policy(model_path: str, device: str = "cpu") -> ONNXPolicy:
    """
    便捷函数：加载ONNX策略
    
    Args:
        model_path: ONNX模型文件路径
        device: 设备类型
    
    Returns:
        ONNXPolicy实例
    """
    return ONNXPolicy(model_path, device=device)

