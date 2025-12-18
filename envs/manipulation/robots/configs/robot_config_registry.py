"""
机器人配置注册表模块
用于管理和动态加载不同的机器人配置
"""

from typing import Dict, Any, Optional
import importlib
import os


class RobotConfigRegistry:
    """机器人配置注册表，管理所有可用的机器人配置"""
    
    _configs: Dict[str, Dict[str, Any]] = {}
    _initialized: bool = False
    
    @classmethod
    def register_config(cls, config_name: str, config_dict: Dict[str, Any]) -> None:
        """
        注册一个机器人配置
        
        Args:
            config_name: 配置名称
            config_dict: 配置字典
        """
        cls._configs[config_name] = config_dict
    
    @classmethod
    def get_config(cls, config_name: str) -> Dict[str, Any]:
        """
        获取指定名称的配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            配置字典
            
        Raises:
            ValueError: 如果配置不存在
        """
        if not cls._initialized:
            cls._load_all_configs()
        
        if config_name not in cls._configs:
            available_configs = ", ".join(cls._configs.keys())
            raise ValueError(
                f"Robot configuration '{config_name}' not found. "
                f"Available configurations: {available_configs}"
            )
        
        return cls._configs[config_name]
    
    @classmethod
    def get_config_for_robot(cls, robot_name: str, config_name: Optional[str] = None) -> Dict[str, Any]:
        """
        根据机器人名称获取配置
        
        Args:
            robot_name: 机器人名称
            config_name: 可选的配置名称，如果提供则直接使用该配置
            
        Returns:
            配置字典
        """
        print("get_config_for_robot: ", robot_name, config_name)
        if not cls._initialized:
            cls._load_all_configs()
        
        # 如果显式指定了配置名称，直接使用
        if config_name:
            return cls.get_config(config_name)
        
        # 否则根据机器人名称推断配置
        # 优先进行精确匹配
        if robot_name in cls._configs:
            return cls._configs[robot_name]
        
        # 如果没有精确匹配，尝试前缀匹配
        for config_key in cls._configs.keys():
            if robot_name.startswith(config_key):
                return cls._configs[config_key]
        
        # 如果都没找到，抛出错误
        available_configs = ", ".join(cls._configs.keys())
        raise ValueError(
            f"No configuration found for robot '{robot_name}'. "
            f"Available configurations: {available_configs}"
        )
    
    @classmethod
    def list_available_configs(cls) -> list:
        """
        列出所有可用的配置
        
        Returns:
            配置名称列表
        """
        if not cls._initialized:
            cls._load_all_configs()
        
        return list(cls._configs.keys())
    
    @classmethod
    def _load_all_configs(cls) -> None:
        """加载所有配置文件"""
        if cls._initialized:
            return
        
        # 导入所有配置
        try:
            from .openloong_config import openloong_config
            cls.register_config("openloong", openloong_config)
            cls.register_config("openloong_hand_fix_base", openloong_config)
            cls.register_config("openloong_gripper_2f85_fix_base", openloong_config)
            cls.register_config("openloong_gripper_2f85_mobile_base", openloong_config)
        except ImportError:
            pass
        
        try:
            from .d12_config import d12_config
            cls.register_config("d12", d12_config)
        except ImportError:
            pass
        
        try:
            from .hand_config import hand_config
            cls.register_config("hand", hand_config)
        except ImportError:
            pass
        
        try:
            from .gripper_2f85_config import gripper_2f85_config
            cls.register_config("gripper_2f85", gripper_2f85_config)
        except ImportError:
            pass
        
        try:
            from .dexforce_w1_config import dexforce_w1_config
            cls.register_config("dexforce_w1", dexforce_w1_config)
        except ImportError:
            pass
        
        # 尝试加载其他可能的配置文件
        config_dir = os.path.dirname(__file__)
        for filename in os.listdir(config_dir):
            if filename.endswith("_config.py") and not filename.startswith("_"):
                module_name = filename[:-3]  # 去掉 .py
                config_var_name = module_name  # 配置变量名通常与文件名相同
                
                # 跳过已加载的
                if module_name in ["openloong_config", "d12_config", "hand_config", "gripper_2f85_config"]:
                    continue
                
                try:
                    module = importlib.import_module(f".{module_name}", package="envs.manipulation.robots.configs")
                    if hasattr(module, config_var_name):
                        config = getattr(module, config_var_name)
                        # 从文件名提取配置名称（去掉 _config 后缀）
                        config_name = module_name.replace("_config", "")
                        cls.register_config(config_name, config)
                except (ImportError, AttributeError):
                    pass
        
        cls._initialized = True
    
    @classmethod
    def reset(cls) -> None:
        """重置注册表（主要用于测试）"""
        cls._configs.clear()
        cls._initialized = False


# 便捷函数
def get_robot_config(robot_name: str, config_name: Optional[str] = None) -> Dict[str, Any]:
    """
    获取机器人配置的便捷函数
    
    Args:
        robot_name: 机器人名称
        config_name: 可选的配置名称
        
    Returns:
        配置字典
    """
    return RobotConfigRegistry.get_config_for_robot(robot_name, config_name)


def list_available_configs() -> list:
    """
    列出所有可用配置的便捷函数
    
    Returns:
        配置名称列表
    """
    return RobotConfigRegistry.list_available_configs()

