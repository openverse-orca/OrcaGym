#!/usr/bin/env python3
"""
gRPC推理服务器
使用配置文件中的ONNX模型进行推理
"""

import os
import sys
import yaml
import argparse
import logging
import time
import numpy as np
import grpc
from concurrent import futures
from pathlib import Path
from typing import Dict, Any, Optional


# 导入生成的gRPC代码

from examples.legged_gym.scripts.proto import inference_pb2
from examples.legged_gym.scripts.proto import inference_pb2_grpc

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



# 导入ONNX运行时
try:
    import onnxruntime as ort
except ImportError:
    _logger.error("Error: onnxruntime not installed.")
    _logger.performance("Please install: pip install onnxruntime")
    sys.exit(1)

logger = logging.getLogger(__name__)


class ONNXInferenceService(inference_pb2_grpc.InferenceServiceServicer):
    """ONNX推理服务"""
    
    def __init__(self, config_path: str):
        """
        初始化推理服务
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.models = self._load_onnx_models()
        self.request_count = 0
        self.model_cache = {}
        
        logger.info(f"Loaded {len(self.models)} ONNX models")
        for model_name, model in self.models.items():
            logger.info(f"  - {model_name}: {model}")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _load_onnx_models(self) -> Dict[str, ort.InferenceSession]:
        """加载ONNX模型"""
        models = {}
        
        if 'model_file' not in self.config or 'onnx' not in self.config['model_file']:
            logger.error("No ONNX models found in config")
            return models
        
        onnx_config = self.config['model_file']['onnx']
        
        for model_name, model_path in onnx_config.items():
            try:
                # 检查模型文件是否存在
                if not os.path.exists(model_path):
                    logger.warning(f"ONNX model file not found: {model_path}")
                    continue
                
                # 加载ONNX模型
                logger.info(f"Loading ONNX model: {model_path}")
                # 显式指定GPU优先
                providers = [
                    'CUDAExecutionProvider',  # 优先尝试GPU
                    'CPUExecutionProvider'    # GPU不可用时回退到CPU
                ]
                session = ort.InferenceSession(model_path, providers=providers)
                
                # 获取模型输入输出信息
                input_names = [input.name for input in session.get_inputs()]
                output_names = [output.name for output in session.get_outputs()]
                logger.info(f"Model {model_name}: inputs={input_names}, outputs={output_names}")
                
                models[model_name] = session
                
            except Exception as e:
                logger.error(f"Failed to load ONNX model {model_name} from {model_path}: {e}")
                continue
        
        return models
    
    def _prepare_onnx_input(self, obs: Dict[str, np.ndarray], model: ort.InferenceSession) -> Dict[str, np.ndarray]:
        """准备ONNX模型输入"""
        # 获取模型的输入信息
        input_names = [input.name for input in model.get_inputs()]
        onnx_input = {}
        
        # 根据模型输入名称准备数据
        for input_name in input_names:
            if input_name == "observation_achieved_goal":
                onnx_input[input_name] = np.array([obs["achieved_goal"]], dtype=np.float32)
            elif input_name == "observation_desired_goal":
                onnx_input[input_name] = np.array([obs["desired_goal"]], dtype=np.float32)
            elif input_name == "observation_observation":
                onnx_input[input_name] = np.array([obs["observation"]], dtype=np.float32)
            else:
                # 对于其他输入，尝试从obs中获取
                if input_name in obs:
                    onnx_input[input_name] = np.array([obs[input_name]], dtype=np.float32)
                else:
                    # 如果找不到对应的输入，使用默认值
                    logger.warning(f"Input '{input_name}' not found in observation, using zeros")
                    input_shape = model.get_inputs()[0].shape
                    if len(input_shape) > 1:
                        onnx_input[input_name] = np.zeros((1,) + tuple(input_shape[1:]), dtype=np.float32)
                    else:
                        onnx_input[input_name] = np.zeros((1,), dtype=np.float32)
        
        return onnx_input
    
    def Predict(self, request, context):
        """处理单次推理请求"""
        self.request_count += 1
        logger.info(f"Received inference request #{self.request_count}")
        
        try:
            # 解析请求数据
            observation = np.array(request.observation, dtype=np.float32)
            desired_goal = np.array(request.desired_goal, dtype=np.float32)
            achieved_goal = np.array(request.achieved_goal, dtype=np.float32)
            model_type = request.model_type
            deterministic = request.deterministic
            
            logger.debug(f"Model type: {model_type}, Deterministic: {deterministic}")
            logger.debug(f"Observation shape: {observation.shape}")
            logger.debug(f"Desired goal shape: {desired_goal.shape}")
            logger.debug(f"Achieved goal shape: {achieved_goal.shape}")
            
            # 准备观察数据
            obs = {
                "observation": observation,
                "desired_goal": desired_goal,
                "achieved_goal": achieved_goal
            }
            
            # 选择模型
            if model_type not in self.models:
                available_models = list(self.models.keys())
                error_msg = f"Model type '{model_type}' not found. Available models: {available_models}"
                logger.error(error_msg)
                return inference_pb2.InferenceResponse(
                    action=[],
                    states=[],
                    success=False,
                    error_message=error_msg
                )
            
            model = self.models[model_type]
            
            # 准备ONNX输入
            onnx_input = self._prepare_onnx_input(obs, model)
            
            # 执行推理
            start_time = time.time()
            onnx_outputs = model.run(None, onnx_input)
            inference_time = time.time() - start_time
            
            # 获取动作输出
            action = onnx_outputs[0][0]  # 假设第一个输出是动作
            
            # 确保动作在合理范围内
            action = np.clip(action, -100, 100)
            
            # 生成状态信息（如果有的话）
            states = None
            if len(onnx_outputs) > 1:
                states = onnx_outputs[1][0]  # 假设第二个输出是状态
            
            logger.debug(f"Inference completed in {inference_time:.4f}s")
            logger.debug(f"Action shape: {action.shape}")
            
            # 返回响应
            response = inference_pb2.InferenceResponse(
                action=action.tolist(),
                states=states.tolist() if states is not None else [],
                success=True,
                error_message=""
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return inference_pb2.InferenceResponse(
                action=[],
                states=[],
                success=False,
                error_message=str(e)
            )
    
    def BatchPredict(self, request, context):
        """处理批量推理请求"""
        logger.info(f"Received batch inference request with {len(request.requests)} items")
        
        responses = []
        for req in request.requests:
            # 复用单次推理逻辑
            single_response = self.Predict(req, context)
            responses.append(single_response)
        
        return inference_pb2.BatchInferenceResponse(responses=responses)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "loaded_models": list(self.models.keys()),
            "request_count": self.request_count,
            "config_path": getattr(self, 'config_path', 'unknown')
        }
        
        for model_name, model in self.models.items():
            try:
                input_info = [(input.name, input.shape) for input in model.get_inputs()]
                output_info = [(output.name, output.shape) for output in model.get_outputs()]
                info[f"{model_name}_inputs"] = input_info
                info[f"{model_name}_outputs"] = output_info
            except Exception as e:
                logger.warning(f"Failed to get info for model {model_name}: {e}")
        
        return info


def serve(config_path: str, port: int = 50051, max_workers: int = 10):
    """启动gRPC服务器"""
    
    # 创建推理服务
    inference_service = ONNXInferenceService(config_path)
    
    # 创建gRPC服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        inference_service, server
    )
    
    server_address = f"[::]:{port}"
    server.add_insecure_port(server_address)
    
    logger.info(f"Starting gRPC server on {server_address}")
    logger.info("Model information:")
    model_info = inference_service.get_model_info()
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    server.start()
    
    try:
        # 保持服务器运行
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


def test_server(config_path: str, port: int = 50051):
    """测试服务器连接"""
    try:
        from grpc_client import create_grpc_client
        
        # 创建客户端
        client = create_grpc_client(f"localhost:{port}", timeout=5.0)
        
        # 测试推理
        test_obs = {
            "observation": np.random.normal(0, 1, 50).astype(np.float32),
            "desired_goal": np.random.normal(0, 1, 3).astype(np.float32),
            "achieved_goal": np.random.normal(0, 1, 3).astype(np.float32)
        }
        
        _logger.info("Testing gRPC server...")
        
        # 获取可用的模型类型
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        available_models = list(config['model_file']['onnx'].keys())
        
        for model_type in available_models:
            _logger.info(f"Testing model: {model_type}")
            action, states = client.predict(test_obs, model_type=model_type, deterministic=True)
            _logger.info(f"  Action shape: {action.shape}")
            _logger.info(f"  States shape: {states.shape if states is not None else None}")
        
        client.close()
        _logger.info("Test successful!")
        
    except Exception as e:
        _logger.error(f"Test failed: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="gRPC ONNX Inference Server")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration file")
    parser.add_argument("--port", type=int, default=50151, 
                       help="Server port")
    parser.add_argument("--max-workers", type=int, default=10, 
                       help="Max worker threads")
    parser.add_argument("--test", action="store_true", 
                       help="Test server connection")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    if args.test:
        test_server(args.config, args.port)
    else:
        serve(args.config, args.port, args.max_workers)


if __name__ == "__main__":
    main() 