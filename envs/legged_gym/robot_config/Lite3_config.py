
import numpy as np

RewardConfig = {
    "follow_command": {
        "alive" : 0,                     # 存活奖励
        "success" : 0,                   # 成功奖励
        "failure" : 0,                   # 失败惩罚
        "leg_contact" : 1,               # 腿部身体接触惩罚
        "body_contact" : 1,              # 身体接触惩罚
        "foot_touch" : 0,                # 重踏惩罚
        "joint_angles" : 0.1,            # 关节偏离自然站立角度惩罚
        "joint_accelerations" : 2.5e-7,  # 关节加速度惩罚
        "limit" : 0.01,                 # Action极限值惩罚
        "action_rate" : 0.01,           # Action平滑
        "base_gyro" : 0,                
        "base_accelerometer" : 0,
        "follow_command_linvel" : 1,    # 跟随指令速度奖励
        "follow_command_angvel" : 0.5,  # 跟随指令角速度奖励
        "height" : 0,                   # 身体高度惩罚
        "body_lin_vel" : 2,             # 身体上下线速度惩罚
        "body_ang_vel" : 0.05,         # 身体倾斜角速度惩罚
        "body_orientation" : 0,         # 身体姿态惩罚
        "feet_air_time" : 1,          # 足底离地时间，小于给定的世间惩罚
        "feet_self_contact" : 0,        # 足底自接触惩罚
        "feet_slip" : 0,             # 接触时，足底线速度
        "feet_wringing" : 0,         # 接触时，足底角速度
        "feet_fitted_ground" : 0.01,    # 鼓励对角步态，避免单侧滑步
        "fly" : 0.1,                    # 四足离地惩罚
        "stepping" : 0.1,                 # 无指令时，踏步惩罚
        "torques" : 1e-5,                # 关节力矩惩罚
        "joint_qpos_limits" : 10.0,      # 关节角度极限值惩罚
        # "joint_qvel_limits" : 1.0,       # 关节速度极限值惩罚
        # "soft_torque_limit" : 1.0,       # 避免关节力矩过大
        "contact_no_vel" : 0,            # 接触时，足底线速度越小越好
    },
    "stand_still": {
        "alive" : 0,                     # 存活奖励
        "success" : 0,                   # 成功奖励
        "failure" : 0,                   # 失败惩罚
        "leg_contact" : 1,               # 腿部身体接触惩罚
        "body_contact" : 1,              # 身体接触惩罚
        "foot_touch" : 0,                # 重踏惩罚
        "joint_angles" : 0.2,            # 关节偏离自然站立角度惩罚
        "joint_accelerations" : 2.5e-7,  # 关节加速度惩罚
        "limit" : 0.01,                 # Action极限值惩罚
        "action_rate" : 0.01,           # Action平滑
        "base_gyro" : 0,                
        "base_accelerometer" : 0,
        "follow_command_linvel" : 0.1,    # 跟随指令速度奖励
        "follow_command_angvel" : 5.0,  # 跟随指令角速度奖励
        "height" : 0,                   # 身体高度惩罚
        "body_lin_vel" : 2,             # 身体上下线速度惩罚
        "body_ang_vel" : 0.05,         # 身体倾斜角速度惩罚
        "body_orientation" : 0,         # 身体姿态惩罚
        "feet_air_time" : 0.1,          # 足底离地时间，小于给定的世间惩罚
        "feet_self_contact" : 0,        # 足底自接触惩罚
        "feet_slip" : 0,             # 接触时，足底线速度
        "feet_wringing" : 0,         # 接触时，足底角速度
        "feet_fitted_ground" : 0,    # 鼓励对角步态，避免单侧滑步
        "fly" : 0.01,                    # 四足离地惩罚0, 0
        "stepping" : 0.5,                 # 无指令时，踏步惩罚
        "torques" : 1e-6,                # 关节力矩惩罚
        "joint_qpos_limits" : 10.0,      # 关节角度极限值惩罚
        # "joint_qvel_limits" : 1.0,       # 关节速度极限值惩罚
        # "soft_torque_limit" : 1.0,       # 避免关节力矩过大
        "contact_no_vel" : 0,            # 接触时，足底线速度越小越好
    }
}

Lite3Config = {
        
        # The order of the joints should be the same as they have been defined in the xml file.
        "base_joint_name" :     "torso",
        "leg_joint_names" :     ["FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint", 
                                "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
                                "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
                                "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"],
        
        # Init the robot in a standing position. Keep the order of the joints same as the joint_names 
        # for reset basic pos or computing the reward easily.
        "neutral_joint_angles" : {"FL_HipX_joint": 0.0, "FL_HipY_joint": -0.8, "FL_Knee_joint": 1.5,
                                "FR_HipX_joint": 0.0, "FR_HipY_joint": -0.8, "FR_Knee_joint": 1.5,
                                "HL_HipX_joint": 0.0, "HL_HipY_joint": -1.0, "HL_Knee_joint": 1.5,
                                "HR_HipX_joint": 0.0, "HR_HipY_joint": -1.0, "HR_Knee_joint": 1.5},
        
        "base_neutral_height_offset" : 0.16,    # the offset from max height to standing natural height
        "base_born_height_offset" : 0.001,       # the offset from max height to standing natural height



        # The order of the actuators should be the same as they have been defined in the xml file.
        "actuator_names" :      ["FL_HipX_actuator", "FL_HipY_actuator", "FL_Knee_actuator",
                                "FR_HipX_actuator", "FR_HipY_actuator", "FR_Knee_actuator",
                                "HL_HipX_actuator", "HL_HipY_actuator", "HL_Knee_actuator",
                                "HR_HipX_actuator", "HR_HipY_actuator", "HR_Knee_actuator"],

        "actuator_type" :        "position",  # "torque" or "position"
        # "kps" :                  [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
        "kps" :                  [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25],
        "kds" :                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

        "action_scale" :         [
            0.2,    # joint name="FL_HipX_joint" joint axis="-1 0 0" range="-0.523 0.523", neutral=0.0
            1.0,    # joint name="FL_HipY_joint" joint axis="0 -1 0" range="-2.67 0.314", neutral=-1.0
            0.8,    # joint name="FL_Knee_joint" joint axis="0 -1 0" range="0.524 2.792", neutral=1.8

            0.2,    # joint name="FR_HipX_joint" joint axis="-1 0 0" range="-0.523 0.523", neutral=0.0
            1.0,    # joint name="FR_HipY_joint" joint axis="0 -1 0" range="-2.67 0.314", neutral=-1.0
            0.8,    # joint name="FR_Knee_joint" joint axis="0 -1 0" range="0.524 2.792", neutral=1.8

            0.2,    # joint name="HL_HipX_joint" joint axis="-1 0 0" range="-0.523 0.523", neutral=0.0
            1.0,    # joint name="HL_HipY_joint" joint axis="0 -1 0" range="-2.67 0.314", neutral=-1.0
            0.8,    # joint name="HL_Knee_joint" joint axis="0 -1 0" range="0.524 2.792", neutral=1.8

            0.2,    # joint name="HR_HipX_joint" joint axis="-1 0 0" range="-0.523 0.523", neutral=0.0
            1.0,    # joint name="HR_HipY_joint" joint axis="0 -1 0" range="-2.67 0.314", neutral=-1.0
            0.8,    # joint name="HR_Knee_joint" joint axis="0 -1 0" range="0.524 2.792", neutral=1.8
        ],
        "soft_joint_qpos_limit": 0.9,       # percentage of urdf limits, values above this limit are penalized
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["FL_site", "FR_site", "HL_site", "HR_site"],
        
        "sensor_imu_framequat_name" :           "imu_quat",
        "sensor_imu_gyro_name" :                "imu_omega",
        "sensor_imu_accelerometer_name" :       "imu_acc",
        "sensor_foot_touch_names" : ["FL_touch", "FR_touch", "HL_touch", "HR_touch"],   # Maintain the same order as contact_site_names
        "use_imu_sensor" : False,

        "ground_contact_body_names" : ["Floor_Floor", 
                                        "terrain_perlin_smooth_usda_terrain",
                                        "terrain_perlin_rough_usda_terrain",
                                        "terrain_perlin_smooth_slope_usda_terrain",
                                        "terrain_perlin_rough_slope_usda_terrain",
                                        "terrain_stair_low_usda_terrain", 
                                        "terrain_stair_high_usda_terrain",    
                                        "terrain_brics_usda_terrain",                                     
                                        ],
        
        "base_contact_body_names" : ["torso", "FL_HIP", "FR_HIP", "HL_HIP", "HR_HIP"],
        "leg_contact_body_names" : ["FL_THIGH", "FL_SHANK", 
                                    "FR_THIGH", "FR_SHANK", 
                                    "HL_THIGH", "HL_SHANK", 
                                    "HR_THIGH", "HR_SHANK"],
        "foot_body_names" : ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"],  # Maintain the same order as contact_site_names
        "foot_fitted_ground_pairs" : [[0, 3], [1, 2]],                   # For the quadruped robot, the left front and right rear feet should touch the ground at the same time.

        # Config for reward
        "reward_coeff" : RewardConfig,



        # Robot's Self-Weight: Approximately 149.2 Newtons.
        # Static Foot Forces: Front feet ~44.8 N each; rear feet ~29.8 N each.
        # Dynamic Foot Forces (Walking at 1 m/s): Approximately 89.5 Newtons per foot during ground contact.
        # Front vs. Rear Legs: Front legs bear more force due to weight distribution and dynamics. 
        "foot_touch_force_threshold" : 100.0,
        "foot_touch_force_air_threshold" : 0.01,
        "foot_touch_force_step_threshold" : 5.0,
        "foot_touch_air_time_ideal" : 0.8,  # Go2 robot standing height is 0.4m. The ideal median stride rate for a Trot is around 0.4 seconds
        "foot_square_wave" : {
            "p5" :          0.5,
            "phase_freq" :  0.8,
            "eps" :         0.2,
        },

        # Config for randomization
        "randomize_friction" :      True,
        "friction_range" :          [0.5, 1.25],
        "randomize_base_mass" :     True,
        "added_mass_range" :        [-0.5, 1.5],
        "push_robots" :             True,
        "push_interval_s" :         5,
        "max_push_vel_xy" :         1.0,
        "pos_random_range" :        0.5,    # randomize the x,y position of the robot in each episode
        
        # Config for ccurriculum learning
        "curriculum_learning" :     True,
        "curriculum_levels" : {
            "stand_still" : [
                {"name" : "default" ,           "offset" : [0, 0, 0],       "distance": 0.0, "rating": 0.5, "command_type": "stand_still", "terminate_threshold": 10},
                {"name" : "default" ,           "offset" : [0, 0, 0],       "distance": 0.0, "rating": 0.5, "command_type": "spot_turn", "terminate_threshold": 10},
                {"name" : "smooth" ,           "offset" : [-30, 30, 0],       "distance": 0.0, "rating": 0.5, "command_type": "stand_still", "terminate_threshold": 10},
                {"name" : "smooth" ,           "offset" : [-30, 30, 0],       "distance": 0.0, "rating": 0.5, "command_type": "spot_turn", "terminate_threshold": 10},
            ],
            "basic_moving" : [
                {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 1.0, "rating": 0.5, "command_type": "move_slowly", "terminate_threshold": 10},
                {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 3.0, "rating": 0.5, "command_type": "move_medium", "terminate_threshold": 10},
                {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 0.0, "rating": 0.5, "command_type": "spot_turn", "terminate_threshold": 10},
            ],
            "flat_terrain" : [
                {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 5.0, "rating": 0.5, "command_type": "move_fast", "terminate_threshold": 10},
                {"name" : "smooth" ,                "offset" : [-30, 30, 0],   "distance": 5.0, "rating": 0.5, "command_type": "move_fast",  "terminate_threshold": 10},
                {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 5.0, "rating": 0.5, "command_type": "move_fast", "terminate_threshold": 10},
                {"name" : "smooth" ,                "offset" : [-30, 30, 0],   "distance": 5.0, "rating": 0.5, "command_type": "move_fast",  "terminate_threshold": 10},
            ],
            "rough_terrain" : [
                {"name" : "rough" ,                 "offset" : [-0, 30, 0],   "distance": 3.0, "rating": 0.5, "command_type": "move_medium",  "terminate_threshold": 10},
                {"name" : "smooth_slope" ,          "offset" : [0, -30, 0],    "distance": 3.0, "rating": 0.5, "command_type": "move_medium",  "terminate_threshold": 10},
                {"name" : "rough" ,                 "offset" : [-0, 30, 0],   "distance": 3.0, "rating": 0.5, "command_type": "move_medium",  "terminate_threshold": 10},
                {"name" : "rough_slope" ,           "offset" : [30, 0, 0],    "distance": 3.0, "rating": 0.5, "command_type": "move_medium",  "terminate_threshold": 10},
                {"name" : "rough" ,                 "offset" : [-0, 30, 0],   "distance": 3.0, "rating": 0.5, "command_type": "move_medium",  "terminate_threshold": 10},
                {"name" : "terrain_stairs_low" ,    "offset" : [-30, -30, 0],   "distance": 3.0, "rating": 0.5, "command_type": "move_medium",  "terminate_threshold": 10},
           ],
        },
        "curriculum_commands" : {
            "stand_still" : {
                "command_lin_vel_range_x" : [-0.0, 0.0], # x direction for forward max speed
                "command_lin_vel_range_y" : [-0.0, 0.0], # y direction for left/right max speed
                "command_lin_vel_threshold" : [-0.0, 0.0], # min linear velocity to trigger moving
                "command_ang_vel_range" : 0.0,  # max turning rate
                "command_resample_interval" : 20, # second to resample the command
            },

            "spot_turn" : {
                "command_lin_vel_range_x" : [-0.1, 0.1], # x direction for forward max speed
                "command_lin_vel_range_y" : [-0.0, 0.0], # y direction for left/right max speed
                "command_lin_vel_threshold" : [-0.0, 0.0], # min linear velocity to trigger moving
                "command_ang_vel_range" : np.pi / 4,  # max turning rate
                "command_resample_interval" : 2, # second to resample the command
            },

            "move_slowly" : {
                "command_lin_vel_range_x" : [-0.5, 0.5], # x direction for forward max speed
                "command_lin_vel_range_y" : [-0.0, 0.0], # y direction for left/right max speed
                "command_lin_vel_threshold" : [-0.1, 0.1], # min linear velocity to trigger moving
                "command_ang_vel_range" : np.pi / 4,  # max turning rate
                "command_resample_interval" : 4, # second to resample the command
            },

            "move_medium" : {
                "command_lin_vel_range_x" : [-1.0, 1.0], # x direction for forward
                "command_lin_vel_range_y" : [-0.1, 0.1], # y direction for left/right
                "command_lin_vel_threshold" : [-0.2, 0.2], # min linear velocity to trigger moving
                "command_ang_vel_range" : np.pi / 4,  # max turning rate
                "command_resample_interval" : 4, # second to resample the command
            },

            "move_fast" : {
                "command_lin_vel_range_x" : [-1.0, 1.5], # x direction for forward max speed
                "command_lin_vel_range_y" : [-0.3, 0.3], # y direction for left/right max speed
                "command_lin_vel_threshold" : [-0.2, 0.3], # min linear velocity to trigger moving
                "command_ang_vel_range" : np.pi / 4,  # max turning rate
                "command_resample_interval" : 4, # second to resample the command
            },
        },

        # Config for logging
        "log_env_ids" :     [0],
        "log_agent_names" : ["Lite3_000"],
        
        # Config for visualization
        "visualize_command_agent_names" : ["Lite3_000"],
        "command_indicator_name" : "command_indicator_mocap",

        # Config for playable agent
        "playable_agent_name" : "Lite3_000",
        
        # Config for nn network training
        "pi" : [512, 256, 128],  # 策略网络结构
        "vf" : [512, 256, 128],   # 值函数网络结构
        "n_steps" : 64,  # 每个环境采样步数
        "batch_size" : 4096,  # 批次大小            
        "learning_rate" : 0.0003,  # 学习率
        "gamma" : 0.99,  # 折扣因子
        "clip_range" : 0.2,  # PPO剪切范围
        "ent_coef" : 0.01,  # 熵系数
        "max_grad_norm" : 1,  # 最大梯度范数
    }