import sys
import os

# 添加 libs 目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录
libs_dir = os.path.join(current_dir, 'libs')  # 构建 libs 目录路径
sys.path.append(libs_dir)  # 将 libs 目录添加到 sys.path

import openloong_wbc

# 创建 OrcaGym_Interface 实例
orcagym = openloong_wbc.OrcaGym_Interface()

# 调用方法
orcagym.setRPY([1.0, 0.5, 0.0])
orcagym.setBaseQuat([0.0, 1.0, 0.0, 0.0])

# 创建 OpenLoongWBC 实例
open_env = openloong_wbc.OpenLoongWBC("urdf_path", 0.01, "json_path", "log_path", 42, 42)

# 调用类方法
open_env.InitLogger()