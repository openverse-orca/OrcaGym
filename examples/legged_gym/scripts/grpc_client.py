import grpc
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
import logging

# 导入生成的gRPC代码
try:
    from .proto import inference_pb2
    from .proto import inference_pb2_grpc
except ImportError:
    # 如果还没有生成gRPC代码，使用占位符
    inference_pb2 = None
    inference_pb2_grpc = None

logger = logging.getLogger(__name__)


class GrpcInferenceClient:
    """gRPC推理客户端"""
    
    def __init__(self, server_address: str = "localhost:50051", 
                 timeout: float = 5.0,
                 max_retries: int = 3):
        """
        初始化gRPC客户端
        
        Args:
            server_address: 服务器地址，格式为 "host:port"
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.server_address = server_address
        self.timeout = timeout
        self.max_retries = max_retries
        self.channel = None
        self.stub = None
        self._connect()
    
    def _connect(self):
        """建立gRPC连接"""
        try:
            if inference_pb2 is None:
                raise ImportError("gRPC protobuf files not generated. Please run: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. inference.proto")
            
            self.channel = grpc.insecure_channel(self.server_address)
            self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

            logger.info(f"Successfully connected to gRPC server at {self.server_address}")
            
        except Exception as e:
            logger.error(f"Failed to connect to gRPC server: {e}")
            raise

    
    def predict(self, obs: Dict[str, np.ndarray], 
                model_type: str = "default",
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        执行推理
        
        Args:
            obs: 观察数据字典，包含 'observation', 'desired_goal', 'achieved_goal' 等键
            model_type: 模型类型标识
            deterministic: 是否确定性推理
            
        Returns:
            Tuple[action, states]: 动作数组和状态数组（可选）
        """
        if self.stub is None:
            logger.error("gRPC client not connected")
            return None, None
        
        # 准备请求数据
        request = inference_pb2.InferenceRequest(
            observation=obs.get('observation', []).flatten().tolist(),
            desired_goal=obs.get('desired_goal', []).flatten().tolist(),
            achieved_goal=obs.get('achieved_goal', []).flatten().tolist(),
            model_type=model_type,
            deterministic=deterministic
        )
        
        # 发送请求，带重试机制
        for attempt in range(self.max_retries):
            try:
                # 使用grpc.RpcError的超时机制
                response = self.stub.Predict(request, timeout=self.timeout)
                
                if not response.success:
                    raise RuntimeError(f"Inference failed: {response.error_message}")
                
                # 转换响应数据
                action = np.array(response.action, dtype=np.float32)
                states = np.array(response.states, dtype=np.float32) if response.states else None
                
                return action, states
                
            except grpc.RpcError as e:
                logger.error(f"gRPC request failed after {self.max_retries} attempts: {e}")
                return None, None
    
    def batch_predict(self, obs_list: list, 
                     model_type: str = "default",
                     deterministic: bool = True) -> list:
        """
        批量推理
        
        Args:
            obs_list: 观察数据列表
            model_type: 模型类型标识
            deterministic: 是否确定性推理
            
        Returns:
            List[Tuple[action, states]]: 动作和状态列表
        """
        if self.stub is None:
            raise RuntimeError("gRPC client not connected")
        
        # 准备批量请求
        requests = []
        for obs in obs_list:
            request = inference_pb2.InferenceRequest(
                observation=obs.get('observation', []).flatten().tolist(),
                desired_goal=obs.get('desired_goal', []).flatten().tolist(),
                achieved_goal=obs.get('achieved_goal', []).flatten().tolist(),
                model_type=model_type,
                deterministic=deterministic
            )
            requests.append(request)
        
        batch_request = inference_pb2.BatchInferenceRequest(requests=requests)
        
        # 发送批量请求
        for attempt in range(self.max_retries):
            try:
                # 使用grpc.RpcError的超时机制
                batch_response = self.stub.BatchPredict(batch_request, timeout=self.timeout)
                
                results = []
                for response in batch_response.responses:
                    if not response.success:
                        raise RuntimeError(f"Batch inference failed: {response.error_message}")
                    
                    action = np.array(response.action, dtype=np.float32)
                    states = np.array(response.states, dtype=np.float32) if response.states else None
                    results.append((action, states))
                
                return results
                
            except grpc.RpcError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"gRPC batch request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    logger.error(f"gRPC batch request failed after {self.max_retries} attempts: {e}")
                    raise
    
    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
            logger.info("gRPC client connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数，用于创建客户端实例
def create_grpc_client(server_address: str = "localhost:50051", 
                      timeout: float = 5.0,
                      max_retries: int = 3) -> GrpcInferenceClient:
    """
    创建gRPC推理客户端
    
    Args:
        server_address: 服务器地址
        timeout: 超时时间
        max_retries: 最大重试次数
        
    Returns:
        GrpcInferenceClient: 客户端实例
    """
    return GrpcInferenceClient(server_address, timeout, max_retries) 