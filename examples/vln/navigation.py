import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime


current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(project_root)# 这里的目录应该是～/
project_root = project_root +'/vln_policy'
#debug取消注释下面
# project_root = '/home/fuxin/vln_policy'
# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)

os.chdir(project_root+'/vlfm')
from vlfm.reality.run_orca import run_policy_orca
from vlfm.run import run_policy

if __name__ == "__main__":
    # run_policy()
    run_policy_orca()
