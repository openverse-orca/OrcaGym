import json
import os

# 打开并读取 JSON 文件
def load_config(config_name) -> dict:
    # 获取当前路径
    current_path = os.path.dirname(__file__)
    print("current_path: ", current_path)
    # 获取配置文件路径
    config_file_path = current_path + f"/config/{config_name}.json"

    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config