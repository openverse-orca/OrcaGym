import os
import sys
import time

def create_tmp_dir(dir_name):
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 设置要检查的目录路径
    dir_path = os.path.join(current_dir, dir_name)

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录 '{dir_path}' 已创建。")


def formate_now():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())