import numpy as np
from envs.legged_gym.robot_config.Lite3_config import Lite3Config
from envs.legged_gym.robot_config.go2_config import Go2Config
from envs.legged_gym.robot_config.A01B_config import A01BConfig
from envs.legged_gym.robot_config.AzureLoong_config import AzureLoongConfig
from envs.legged_gym.robot_config.g1_config import g1Config

LeggedEnvConfig = {
    "TIME_STEP" : 0.005,                 # 仿真步长1000Hz ~ 200Hz
    "FRAME_SKIP" : 1,                    # PD函数计算步长200Hz
    "ACTION_SKIP" : 4,                  # 训练和推理步长50Hz

    "EPISODE_TIME_VERY_SHORT" : 2,       # 每个episode的时间长度
    "EPISODE_TIME_SHORT" : 5,           
    "EPISODE_TIME_LONG" : 20,


    "phy_low" : {
        "iterations" : 10,
        "noslip_iterations" : 0,
        "ccd_iterations" : 0,
        "sdf_iterations" : 0,
        "filterparent" : "disable"
    },
    "phy_high" : {
        "iterations" : 100,
        "noslip_iterations" : 10,
        "ccd_iterations" : 50,
        "sdf_iterations" : 10,  
        "filterparent" : "disable"
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

CurriculumConfig = {
    "terrain" : {
        "default" : {
            "offset" : [0, 0, 0],
        },
        "brics_low" : {
            "offset" : [0, -30, 0],
        },
        "brics_high" : {
            "offset" : [0, 30, 0],
        },
        "stair_low" : {
            "offset" : [-30, -30, 0],
        },
        "stair_high" : {
            "offset" : [-30, 0, 0],
        },
        "slop_10" : {
            "offset" : [60, -30, 0],
        },
        "slop_5" : {
            "offset" : [60, 30, 0],
        },
        "stair_low_flat" : {
            "offset" : [0, 30, 0],
        },
        "stair_mid_flat" : {
            "offset" : [10, 30, 0],
        },
        "perlin_smooth" : {
            "offset" : [-30, 30, 0],
        },
        "perlin_rough" : {
            "offset" : [0, 30, 0],
        },
        "ellipsoid_low" : {
            "offset" : [-30, 30, 0],
        },
        "ellipsoid_high" : {
            "offset" : [0, 30, 0],
        },
    },

    "curriculum_levels" : {
        # "basic_moving" : [
        #     {"terrain" : "default" ,          "command_type": "move_slowly",},
        #     {"terrain" : "default" ,          "command_type": "move_medium",},
        # ],
        "flat_terrain" : [
            {"terrain" : "default" ,          "command_type": "move_medium",},
            {"terrain" : "default" ,          "command_type": "move_fast",},
            {"terrain" : "ellipsoid_low" ,    "command_type": "move_medium",},
            {"terrain" : "ellipsoid_low" ,    "command_type": "move_fast",},
            {"terrain" : "brics_low" ,        "command_type": "move_medium", },
            {"terrain" : "brics_low" ,        "command_type": "move_fast", },
            {"terrain" : "stair_low_flat" ,   "command_type": "move_medium",},
            {"terrain" : "stair_low_flat" ,   "command_type": "move_fast",},
        ],
        "rough_terrain" : [
            {"terrain" : "default" ,          "command_type": "move_fast",},
            {"terrain" : "brics_low" ,        "command_type": "move_medium", },
            {"terrain" : "default" ,          "command_type": "move_fast",},
            {"terrain" : "brics_low" ,        "command_type": "move_fast",},
            {"terrain" : "stair_low_flat" ,   "command_type": "move_medium",},
            {"terrain" : "default" ,          "command_type": "move_fast",},
            {"terrain" : "stair_low_flat" ,   "command_type": "move_fast",},
            {"terrain" : "stair_low",       "command_type": "move_medium",},
            {"terrain" : "default" ,          "command_type": "move_fast",},
            {"terrain" : "stair_low" ,       "command_type": "move_medium",},
            {"terrain" : "slop_5" ,          "command_type": "move_medium", },
            {"terrain" : "default" ,          "command_type": "move_fast",},
            {"terrain" : "slop_5" ,          "command_type": "move_fast",},
            {"terrain" : "brics_high" ,       "command_type": "move_medium", },
            {"terrain" : "default" ,          "command_type": "move_fast",},
            {"terrain" : "brics_high" ,       "command_type": "move_medium",},
            {"terrain" : "stair_high" ,       "command_type": "move_medium",},
            {"terrain" : "default" ,          "command_type": "move_fast",},
            {"terrain" : "slop_10" ,          "command_type": "move_medium", },
            {"terrain" : "default" ,          "command_type": "move_fast",},
            {"terrain" : "slop_10" ,          "command_type": "move_medium",},
        ],
    },


    "ground_contact_body_names" : ["Floor_Floor", 
                                    "terrain_perlin_smooth_usda_terrain",
                                    "terrain_perlin_rough_usda_terrain",
                                    "terrain_perlin_smooth_slope_usda_terrain",
                                    "terrain_perlin_rough_slope_usda_terrain",
                                    "terrain_stair_low_usda_terrain", 
                                    "terrain_stair_high_usda_terrain",    
                                    "terrain_brics_usda_terrain",        
                                    "terrain_stair_low_flat_usda_terrain",  
                                    "terrain_stair_mid_flat_usda_terrain",  
                                    "terrain_brics_low_usda_terrain",      
                                    "terrain_brics_high_usda_terrain",     
                                    "terrain_slop_10_usda_terrain",        
                                    "terrain_slop_20_usda_terrain",       
                                    "terrain_ellipsoid_low_usda_terrain",
                                    ],
        

    "curriculum_commands" : {
        "stand_still" : {
            "command_lin_vel_range_x" : [-0.0, 0.0], # x direction for forward max speed
            "command_lin_vel_range_y" : [-0.0, 0.0], # y direction for left/right max speed
            "command_lin_vel_threshold" : [-0.0, 0.0], # min linear velocity to trigger moving
            "command_ang_vel_range" : 0.0,  # max turning rate
            "command_resample_interval" : 20, # second to resample the command
            "distance" : 0.0,
            "rating" : 0.5,
            "terminate_threshold" : 10,
        },

        "spot_turn" : {
            "command_lin_vel_range_x" : [-0.1, 0.1], # x direction for forward max speed
            "command_lin_vel_range_y" : [-0.0, 0.0], # y direction for left/right max speed
            "command_lin_vel_threshold" : [-0.0, 0.0], # min linear velocity to trigger moving
            "command_ang_vel_range" : np.pi / 4,  # max turning rate
            "command_resample_interval" : 2, # second to resample the command
            "distance" : 0.0,
            "rating" : 0.5,
            "terminate_threshold" : 3,
        },

        "move_slowly" : {
            "command_lin_vel_range_x" : [-0.5, 0.5], # x direction for forward max speed
            "command_lin_vel_range_y" : [-0.0, 0.0], # y direction for left/right max speed
            "command_lin_vel_threshold" : [-0.25, 0.25], # min linear velocity to trigger moving
            "command_ang_vel_range" : np.pi / 4,  # max turning rate
            "command_resample_interval" : 4, # second to resample the command
            "distance" : 1.0,
            "rating" : 0.5,
            "terminate_threshold" : 10,
        },

        "move_medium" : {
            "command_lin_vel_range_x" : [-0.8, 0.8], # x direction for forward
            "command_lin_vel_range_y" : [-0.1, 0.1], # y direction for left/right
            "command_lin_vel_threshold" : [-0.2, 0.2], # min linear velocity to trigger moving
            "command_ang_vel_range" : np.pi / 4,  # max turning rate
            "command_resample_interval" : 4, # second to resample the command
            "distance" : 3.0,
            "rating" : 0.5,
            "terminate_threshold" : 10,
        },

        "move_fast" : {
            "command_lin_vel_range_x" : [-1.0, 1.5], # x direction for forward max speed
            "command_lin_vel_range_y" : [-0.3, 0.3], # y direction for left/right max speed
            "command_lin_vel_threshold" : [-0.2, 0.3], # min linear velocity to trigger moving
            "command_ang_vel_range" : np.pi / 4,  # max turning rate
            "command_resample_interval" : 4, # second to resample the command
            "distance" : 5.0,
            "rating" : 0.5,
            "terminate_threshold" : 3,
        },
    },
}