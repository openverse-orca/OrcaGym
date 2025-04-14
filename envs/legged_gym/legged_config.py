import numpy as np
from envs.legged_gym.robot_config.Lite3_config import Lite3Config
from envs.legged_gym.robot_config.go2_config import Go2Config
from envs.legged_gym.robot_config.A01B_config import A01BConfig
from envs.legged_gym.robot_config.AzureLoong_config import AzureLoongConfig
from envs.legged_gym.robot_config.g1_config import g1Config

LeggedEnvConfig = {
    "TIME_STEP" : 0.005,                 # 仿真步长200Hz

    "FRAME_SKIP_REALTIME" : 1,           # 200Hz 推理步长
    "FRAME_SKIP_SHORT" : 4,              # 200Hz * 4 = 50Hz 推理步长
    "FRAME_SKIP_LONG" : 10,              # 200Hz * 10 = 20Hz 训练步长

    "EPISODE_TIME_VERY_SHORT" : 2,       # 每个episode的时间长度
    "EPISODE_TIME_SHORT" : 5,           
    "EPISODE_TIME_LONG" : 20,


    "phy_low" : {
        "iterations" : 10,
        "noslip_iterations" : 0,
        "ccd_iterations" : 0,
        "sdf_iterations" : 0,
    },
    "phy_high" : {
        "iterations" : 100,
        "noslip_iterations" : 10,
        "ccd_iterations" : 50,
        "sdf_iterations" : 10,  
    },
    "phy_config" : "phy_high",
}

LeggedObsConfig = {
    # 对观测数据进行缩放实现归一化，以便神经网络更好的训练
    # 噪声也按同等比例缩放归一化，然后再乘以噪声系数以便对不同的部分模拟不同的噪声水平
    "scale" : {
        "lin_vel" : 2.0,
        "ang_vel" : 0.25,
        "qpos" : 1.0,
        "qvel" : 0.05,
        "height" : 5.0
    },

    "noise" : {
        "noise_level" : 1.0,
        "qpos" : 0.01,
        "qvel" : 1.5,
        "lin_vel" : 0.1,
        "ang_vel" : 0.2,
        "orientation" : 0.05,
        "height" : 0.1
    }
}

LeggedRobotConfig = {
    "Lite3": Lite3Config,
    "go2": Go2Config,
    "A01B": A01BConfig,
    "AzureLoong": AzureLoongConfig,
    "g1": g1Config,     
}