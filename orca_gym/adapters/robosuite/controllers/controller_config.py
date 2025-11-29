import json
import os

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


# 打开并读取 JSON 文件
def load_config(config_name) -> dict:
    # 获取当前路径
    current_path = os.path.dirname(__file__)
    _logger.info(f"current_path:  {current_path}")
    # 获取配置文件路径
    config_file_path = current_path + f"/config/{config_name}.json"

    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config