import os
import subprocess

# 设置环境变量
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ":/home/william/repo/Quadruped-PyMPC/quadruped_pympc/acados/lib"
os.environ['ACADOS_SOURCE_DIR'] = "/home/william/repo/Quadruped-PyMPC/quadruped_pympc/acados"

# 运行目标 Python 脚本
subprocess.run(["python", "./quadruped_ctrl.py"])