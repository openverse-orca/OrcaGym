import numpy as np

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
        "mpr_iterations" : 0,
        "sdf_iterations" : 0,
    },
    "phy_high" : {
        "iterations" : 100,
        "noslip_iterations" : 10,
        "mpr_iterations" : 50,
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
    "go2": {
        
        # The order of the joints should be the same as they have been defined in the xml file.
        "base_joint_name" :     "base",
        "leg_joint_names" :     ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        
        # Init the robot in a standing position. Keep the order of the joints same as the joint_names 
        # for reset basic pos or computing the reward easily.
        "neutral_joint_angles" : {"FL_hip_joint": 0.0, "FL_thigh_joint": 0.7, "FL_calf_joint": -1.5,
                                "FR_hip_joint": 0.0, "FR_thigh_joint": 0.7, "FR_calf_joint": -1.5,
                                "RL_hip_joint": 0.0, "RL_thigh_joint": 0.7, "RL_calf_joint": -1.5,
                                "RR_hip_joint": 0.0, "RR_thigh_joint": 0.7, "RR_calf_joint": -1.5},
        
        "base_neutral_height_offset" : 0.12,    # the offset from max height to standing natural height
        "base_born_height_offset" : 0.001,       # the offset from max height to standing natural height



        # The order of the actuators should be the same as they have been defined in the xml file.
        "actuator_names" :      ["FL_hip_actuator", "FL_thigh_actuator", "FL_calf_actuator",
                                "FR_hip_actuator", "FR_thigh_actuator", "FR_calf_actuator",
                                "RL_hip_actuator", "RL_thigh_actuator", "RL_calf_actuator",
                                "RR_hip_actuator", "RR_thigh_actuator", "RR_calf_actuator"],

        "actuator_type" :        "position",  # "torque" or "position"
        "action_scale" :         0.5,
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["FL_site", "FR_site", "RL_site", "RR_site"],
        
        "sensor_imu_framequat_name" :           "imu_quat",
        "sensor_imu_gyro_name" :                "imu_omega",
        "sensor_imu_accelerometer_name" :       "imu_acc",
        "sensor_foot_touch_names" : ["FL_touch", "FR_touch", "RL_touch", "RR_touch"],

        "ground_contact_body_names" : ["Floor_Floor", 
                                        "terrain_perlin_smooth_usda_terrain",
                                        "terrain_perlin_rough_usda_terrain",
                                        "terrain_perlin_smooth_slope_usda_terrain",
                                        "terrain_perlin_rough_slope_usda_terrain",
                                        "terrain_stair_low_usda_terrain", 
                                        "terrain_stair_high_usda_terrain",                                        
                                        ],
        
        "base_contact_body_names" : ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"],
        "leg_contact_body_names" : ["FL_thigh", "FL_calf", 
                                    "FR_thigh", "FR_calf", 
                                    "RL_thigh", "RL_calf", 
                                    "RR_thigh", "RR_calf"],
        "foot_body_names" : ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],

        # Robot's Self-Weight: Approximately 149.2 Newtons.
        # Static Foot Forces: Front feet ~44.8 N each; rear feet ~29.8 N each.
        # Dynamic Foot Forces (Walking at 1 m/s): Approximately 89.5 Newtons per foot during ground contact.
        # Front vs. Rear Legs: Front legs bear more force due to weight distribution and dynamics. 
        "foot_touch_force_threshold" : 100.0,
        "foot_touch_force_air_threshold" : 0.01,
        "foot_touch_air_time_ideal" : 0.4,  # Go2 robot standing height is 0.4m. The ideal median stride rate for a Trot is around 0.4 seconds
        

        # Config for randomization
        "randomize_friction" :      True,
        "friction_range" :          [0.5, 1.25],
        "randomize_base_mass" :     False,
        "added_mass_range" :        [-1., 1.],
        "push_robots" :             True,
        "push_interval_s" :         15,
        "max_push_vel_xy" :         1.0,
        "pos_random_range" :        2.0,    # randomize the x,y position of the robot in each episode
        
        # Config for ccurriculum learning
        "curriculum_learning" :     True,
        "curriculum_levels" : [
            {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "smooth" ,                "offset" : [-55, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "rough" ,                 "offset" : [-0, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "smooth_slope" ,          "offset" : [0, -55, 0],    "distance": 3.0, "rating": 0.5, "command_type": "slope", },
            {"name" : "rough_slope" ,           "offset" : [55, 0, 0],    "distance": 3.0, "rating": 0.5, "command_type": "slope", },
            {"name" : "terrain_stairs_low" ,    "offset" : [-55, -55, 0],   "distance": 3.0, "rating": 0.5, "command_type": "stairs", },
            {"name" : "terrain_stairs_high" ,   "offset" : [-55, 0, 0],    "distance": 2.0, "rating": 0.5, "command_type": "stairs", },
        ],
        "curriculum_commands" : {
            "flat_plane" : {
                "command_lin_vel_range_x" : 1.5, # x direction for forward max speed
                "command_lin_vel_range_y" : 0.3, # y direction for left/right max speed
                "command_lin_vel_threshold" : 0.2, # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },
            
            "slope" : {
                "command_lin_vel_range_x" : 1.0, # x direction for forward
                "command_lin_vel_range_y" : 0.2, # y direction for left/right
                "command_lin_vel_threshold" : 0.2, # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },
            
            "stairs" : {
                "command_lin_vel_range_x" : 0.5, # x direction for forward
                "command_lin_vel_range_y" : 0.1, # y direction for left/right
                "command_lin_vel_threshold" : 0.0, # min linear velocity to trigger moving
                "command_ang_vel_range" : 0.5,  # max turning rate
                "command_resample_interval" : 20, # second to resample the command
            },
        },

        # Config for logging
        "log_env_ids" :     [0],
        "log_agent_names" : ["go2_000"],
        
        # Config for visualization
        "visualize_command_agent_names" : ["go2_000"],
        "command_indicator_name" : "command_indicator_mocap",

        # Config for playable agent
        "playable_agent_name" : "go2_000",
    },
    "A01B": {
        # The order of the joints should be the same as they have been defined in the xml file.
        "base_joint_name" :     "trunk",
        "leg_joint_names" :     ["fr_joint0", "fr_joint1", "fr_joint2", 
                                "fl_joint0", "fl_joint1", "fl_joint2",
                                "hr_joint0", "hr_joint1", "hr_joint2",
                                "hl_joint0", "hl_joint1", "hl_joint2"],
        
        
        # Init the robot in a standing position. Keep the order of the joints same as the joint_names 
        # for reset basic pos or computing the reward easily.
        "neutral_joint_angles" : {"fr_joint0": 0.0, "fr_joint1": -0.7, "fr_joint2": 1.3,
                                "fl_joint0": 0.0, "fl_joint1": -0.7, "fl_joint2": 1.3,
                                "hr_joint0": 0.0, "hr_joint1": -0.7, "hr_joint2": 1.3,
                                "hl_joint0": 0.0, "hl_joint1": -0.7, "hl_joint2": 1.3},
        
        "base_neutral_height_offset" : 0.14,    # the offset from max height to standing natural height
        "base_born_height_offset" : 0.001,       # the offset from max height to standing natural height


        # The order of the actuators should be the same as they have been defined in the xml file.
        "actuator_names" :      ["fr_abad_actuator", "fr_thigh_actuator", "fr_calf_actuator",
                                "fl_abad_actuator", "fl_thigh_actuator", "fl_calf_actuator",
                                "hr_abad_actuator", "hr_thigh_actuator", "hr_calf_actuator",
                                "hl_abad_actuator", "hl_thigh_actuator", "hl_calf_actuator"],

        "actuator_type" :        "position",  # "torque" or "position"
        "action_scale" :         0.5,
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["fr_site", "fl_site", "hr_site", "hl_site"],
        
        "sensor_imu_framequat_name" :           "imu_quat",
        "sensor_imu_gyro_name" :                "imu_omega",
        "sensor_imu_accelerometer_name" :       "imu_acc",
        "sensor_foot_touch_names" : ["fr_touch", "fl_touch", "rr_touch", "rl_touch"],

        "ground_contact_body_names" : ["Floor_Floor", 
                                        "terrain_perlin_smooth_usda_terrain",
                                        "terrain_perlin_rough_usda_terrain",
                                        "terrain_perlin_smooth_slope_usda_terrain",
                                        "terrain_perlin_rough_slope_usda_terrain",
                                        "terrain_stair_low_usda_terrain", 
                                        "terrain_stair_high_usda_terrain",                                        
                                        ],
        
        "base_contact_body_names" : ["trunk_link", "fl_abad", "fr_abad", "hl_abad", "hr_abad"],

        "leg_contact_body_names" : ["fl_thigh", "fl_calf", 
                                    "fr_thigh", "fr_calf", 
                                    "hl_thigh", "hl_calf", 
                                    "hr_thigh", "hr_calf"],
        
        "foot_body_names" : ["fl_foot", "fr_foot", "hl_foot", "hr_foot"],


        # 机器狗的自重： 约 61.44 kg。
        # 静止时每只脚的受力： 约 150.7 N。
        # 以 1 m/s 速度行走时，每只脚在触地时的受力： 约 301.4 N。        
        "foot_touch_force_threshold" : 350.0,
        "foot_touch_force_air_threshold" : 0.01,
        "foot_touch_air_time_ideal" : 0.6,  
        

        # Config for randomization
        "randomize_friction" :      True,
        "friction_range" :          [0.5, 1.25],
        "randomize_base_mass" :     False,
        "added_mass_range" :        [-1., 1.],
        "push_robots" :             True,
        "push_interval_s" :         15,
        "max_push_vel_xy" :         1.5,
        "pos_random_range" :        2.0,    # randomize the x,y position of the robot in each episode
        
        # Config for ccurriculum learning
        "curriculum_learning" :     True,
        "curriculum_levels" : [
            {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 7.5, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "smooth" ,                "offset" : [-55, 55, 0],   "distance": 7.5, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "rough" ,                 "offset" : [-0, 55, 0],   "distance": 7.5, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "smooth_slope" ,          "offset" : [0, -55, 0],    "distance": 4.5, "rating": 0.5, "command_type": "slope", },
            {"name" : "rough_slope" ,           "offset" : [55, 0, 0],    "distance": 4.5, "rating": 0.5, "command_type": "slope", },
            {"name" : "terrain_stairs_low" ,    "offset" : [-55, -55, 0],   "distance": 4.5, "rating": 0.5, "command_type": "stairs", },
            {"name" : "terrain_stairs_high" ,   "offset" : [-55, 0, 0],    "distance": 3.0, "rating": 0.5, "command_type": "stairs", },
        ],
        "curriculum_commands" : {
            "flat_plane" : {
                "command_lin_vel_range_x" : 2.25, # x direction for forward max speed
                "command_lin_vel_range_y" : 0.45, # y direction for left/right max speed
                "command_lin_vel_threshold" : 0.3, # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },
            
            "slope" : {
                "command_lin_vel_range_x" : 1.5, # x direction for forward
                "command_lin_vel_range_y" : 0.3, # y direction for left/right
                "command_lin_vel_threshold" : 0.3, # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },
            
            "stairs" : {
                "command_lin_vel_range_x" : 0.75, # x direction for forward
                "command_lin_vel_range_y" : 0.15, # y direction for left/right
                "command_lin_vel_threshold" : 0.0, # min linear velocity to trigger moving
                "command_ang_vel_range" : 0.5,  # max turning rate
                "command_resample_interval" : 20, # second to resample the command
            },
        },

        # Config for logging
        "log_env_ids" :     [0],
        "log_agent_names" : ["A01B_000"],
        
        # Config for visualization
        "visualize_command_agent_names" : ["A01B_000"],
        "command_indicator_name" : "command_indicator_mocap",

        # Config for playable agent
        "playable_agent_name" : "A01B_000",        
    },
    "AzureLoong": {
        # The order of the joints should be the same as they have been defined in the xml file.
        "base_joint_name" :     "float_base",
        "leg_joint_names" :     [
                                # "J_arm_r_01", # "J_arm_r_02", 
                                # "J_arm_l_01", # "J_arm_l_02",
                                 "J_hip_r_roll", 
                                #  "J_hip_r_yaw", 
                                 "J_hip_r_pitch", "J_knee_r_pitch", 
                                #  "J_ankle_r_pitch", 
                                 # "J_ankle_r_roll",
                                 "J_hip_l_roll", 
                                #  "J_hip_l_yaw", 
                                 "J_hip_l_pitch", "J_knee_l_pitch", 
                                #  "J_ankle_l_pitch", 
                                 # "J_ankle_l_roll",
                                ],
        
        
        # Init the robot in a standing position. Keep the order of the joints same as the joint_names 
        # for reset basic pos or computing the reward easily.
        "neutral_joint_angles" : {
                                #   "J_arm_r_01": 0.0, # "J_arm_r_02": 1.2,
                                #   "J_arm_l_01": 0.0,  # "J_arm_l_02": -1.2,
                                  "J_hip_r_roll": -0.1, 
                                #   "J_hip_r_yaw": 0.2, 
                                  "J_hip_r_pitch": 0.5, "J_knee_r_pitch": -1.1, 
                                #   "J_ankle_r_pitch": 0.6, 
                                #   "J_ankle_r_roll": 0.1,
                                  "J_hip_l_roll": 0.1, 
                                #   "J_hip_l_yaw": -0.2, 
                                  "J_hip_l_pitch": 0.5, "J_knee_l_pitch": -1.1, 
                                #   "J_ankle_l_pitch": 0.6, 
                                #   "J_ankle_l_roll": -0.1,
                                  },
        
        "base_neutral_height_offset" : 0.12,    # the offset from max height tnpo standing natural height
        "base_born_height_offset" : 0.01,       # the offset from max height to standing natural height


        # The order of the actuators should be the same as they have been defined in the xml file.
        "actuator_names" :      [
                                #  "P_arm_r_01", # "P_arm_r_02",
                                #  "P_arm_l_01", # "P_arm_l_02",
                                 "P_hip_r_roll", 
                                #  "P_hip_r_yaw", 
                                 "P_hip_r_pitch", "P_knee_r_pitch", 
                                #  "P_ankle_r_pitch", 
                                #  "P_ankle_r_roll",
                                 "P_hip_l_roll", 
                                #  "P_hip_l_yaw", 
                                 "P_hip_l_pitch", "P_knee_l_pitch", 
                                #  "P_ankle_l_pitch", 
                                #  "P_ankle_l_roll",
                                 ],

        "actuator_type" :        "position",  # "torque" or "position"
        "action_scale" :         0.5,
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["lf-tc", "rf-tc",],
        
        "sensor_imu_framequat_name" :           "baselink-quat",
        "sensor_imu_gyro_name" :                "baselink-gyro",
        "sensor_imu_accelerometer_name" :       "baselink-velocity",
        "sensor_foot_touch_names" : ["lf-touch", "rf-touch"],

        "ground_contact_body_names" : ["Floor_Floor", 
                                        "terrain_perlin_smooth_usda_terrain",
                                        "terrain_perlin_rough_usda_terrain",
                                        "terrain_perlin_smooth_slope_usda_terrain",
                                        "terrain_perlin_rough_slope_usda_terrain",
                                        "terrain_stair_low_usda_terrain", 
                                        "terrain_stair_high_usda_terrain",                                        
                                        ],
        
        "base_contact_body_names" : ["base_link", 
                                     "Link_arm_r_01", "Link_arm_r_02", "Link_arm_r_03", "Link_arm_r_04", "Link_arm_r_05", "Link_arm_r_06", "Link_arm_r_07",
                                     "Link_arm_l_01", "Link_arm_l_02", "Link_arm_l_03", "Link_arm_l_04", "Link_arm_l_05", "Link_arm_l_06", "Link_arm_l_07",
                                     "Link_waist_pitch", "Link_waist_roll", "Link_waist_yaw", "Link_head_pitch", "Link_head_yaw",
                                     ],

        "leg_contact_body_names" : ["Link_hip_l_roll", "Link_hip_l_yaw", "Link_hip_l_pitch", 
                                     "Link_hip_r_roll", "Link_hip_r_yaw", "Link_hip_r_pitch",
                                    "Link_knee_l_pitch", "Link_knee_r_pitch", ],
        
        "foot_body_names" : ["Link_ankle_l_pitch", "Link_ankle_l_roll", 
                             "Link_ankle_r_pitch", "Link_ankle_r_roll"],


        # 机器人总质量 (M): 约 78.35 千克
        # 垂直跳跃高度 (h)：假设机器人在单腿跳跃时的垂直跳跃高度与水平距离相同，即 0.3 米
        # 减速距离 (d<sub>landing</sub>)：如果实际的减速距离更大（例如，机器人脚掌有更好的缓冲机制），则减速度和受力将会降低。例如，若减速距离为 0.1 米，则减速度为 29.43 米/秒²，受力为 2,305 牛顿。
        "foot_touch_force_threshold" : 2500.0,
        "foot_touch_force_air_threshold" : 0.01,
        "foot_touch_air_time_ideal" : 0.5,  
        

        # Config for randomization
        "randomize_friction" :      True,
        "friction_range" :          [0.5, 1.25],
        "randomize_base_mass" :     False,
        "added_mass_range" :        [-1., 1.],
        "push_robots" :             True,
        "push_interval_s" :         15,
        "max_push_vel_xy" :         1.0,
        "pos_random_range" :        2.0,    # randomize the x,y position of the robot in each episode
        
        # Config for ccurriculum learning
        "curriculum_learning" :     True,
        "curriculum_levels" : [
            {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 3.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "smooth" ,                "offset" : [-55, 55, 0],   "distance": 3.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "rough" ,                 "offset" : [-0, 55, 0],   "distance": 3.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "smooth_slope" ,          "offset" : [0, -55, 0],    "distance": 2.0, "rating": 0.5, "command_type": "slope", },
            {"name" : "rough_slope" ,           "offset" : [55, 0, 0],    "distance": 2.0, "rating": 0.5, "command_type": "slope", },
            {"name" : "terrain_stairs_low" ,    "offset" : [-55, -55, 0],   "distance": 2.0, "rating": 0.5, "command_type": "stairs", },
            {"name" : "terrain_stairs_high" ,   "offset" : [-55, 0, 0],    "distance": 1.0, "rating": 0.5, "command_type": "stairs", },
        ],
        "curriculum_commands" : {
            "flat_plane" : {
                "command_lin_vel_range_x" : 1.0, # x direction for forward max speed
                "command_lin_vel_range_y" : 0.2, # y direction for left/right max speed
                "command_lin_vel_threshold" : 0.2, # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },
            
            "slope" : {
                "command_lin_vel_range_x" : 1.0, # x direction for forward
                "command_lin_vel_range_y" : 0.2, # y direction for left/right
                "command_lin_vel_threshold" : 0.2, # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },
            
            "stairs" : {
                "command_lin_vel_range_x" : 0.3, # x direction for forward
                "command_lin_vel_range_y" : 0.0, # y direction for left/right
                "command_lin_vel_threshold" : 0.1, # min linear velocity to trigger moving
                "command_ang_vel_range" : 0.5,  # max turning rate
                "command_resample_interval" : 20, # second to resample the command
            },
        },
        
        # Config for logging
        "log_env_ids" :     [0],
        "log_agent_names" : ["AzureLoong_000"],
        
        # Config for visualization
        "visualize_command_agent_names" : ["AzureLoong_000"],
        "command_indicator_name" : "command_indicator_mocap",

        # Config for playable agent
        "playable_agent_name" : "AzureLoong_000",        
    }    
}