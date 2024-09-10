import sys
import os
import numpy as np

# 添加 libs 目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录
libs_dir = os.path.join(current_dir, 'libs')  # 构建 libs 目录路径
sys.path.append(libs_dir)  # 将 libs 目录添加到 sys.path

import openloong_dyn_ctrl
from openloong_dyn_ctrl import OpenLoongWBC, OrcaGym_Interface, ButtonState


model_nq = 38
model_nv = 37

# 创建 OrcaGym_Interface 实例
orcagym_interface = OrcaGym_Interface(0.01)

# 调用方法
orcagym_interface.setRPY([1.0, 0.5, 0.0])
orcagym_interface.setBaseQuat([0.0, 1.0, 0.0, 0.0])

qpos = np.zeros(model_nq)
qvel = np.zeros(model_nv)
sensordata_quat = np.zeros(4)
sensordata_vel = np.zeros(3)
sensordata_gyro = np.zeros(3)
sensordata_acc = np.zeros(3)
xpos = np.zeros(3)

JntIdQpos = list(range(model_nq))
JntIdQvel = list(range(model_nv))
orcagym_interface.setJointOffsetQpos(JntIdQpos)
orcagym_interface.setJointOffsetQvel(JntIdQvel)

orcagym_interface.updateSensorValues(qpos, qvel, sensordata_quat, sensordata_vel, sensordata_gyro, sensordata_acc, xpos)


buttonState = ButtonState()

# 创建 records目录
if not os.path.exists("./records"):
    os.makedirs("./records")

# 创建 OpenLoongWBC 实例
open_env = OpenLoongWBC("./external/openloong-dyn-control/models/AzureLoong.urdf", 
                                      0.01, 
                                      "./external/openloong-dyn-control/common/joint_ctrl_config.json", 
                                      "./records/datalog.log", 
                                      model_nq, 
                                      model_nv)

# 调用类方法
open_env.InitLogger()

open_env.Runsimulation(buttonState, orcagym_interface, 0.01)

# 输出测试结果
print("Run test successfully!")