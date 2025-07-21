Go2Config = {
        
        # The order of the joints should be the same as they have been defined in the xml file.
        "base_joint_name" :     "base",
        "leg_joint_names" :     ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        
        # Init the robot in a standing position. Keep the order of the joints same as the joint_names 
        # for reset basic pos or computing the reward easily.
        # "neutral_joint_angles" : {"FL_hip_joint": 0.0, "FL_thigh_joint": 0.7, "FL_calf_joint": -1.5,
        #                         "FR_hip_joint": 0.0, "FR_thigh_joint": 0.7, "FR_calf_joint": -1.5,
        #                         "RL_hip_joint": 0.0, "RL_thigh_joint": 0.7, "RL_calf_joint": -1.5,
        #                         "RR_hip_joint": 0.0, "RR_thigh_joint": 0.7, "RR_calf_joint": -1.5},

        "neutral_joint_angles" : {"FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
                                "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
                                "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
                                "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5},

        # "neutral_joint_angles" : {"FL_hip_joint": 0.0, "FL_thigh_joint": 1.0, "FL_calf_joint": -1.8,
        #                         "FR_hip_joint": 0.0, "FR_thigh_joint": 1.0, "FR_calf_joint": -1.8,
        #                         "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.8,
        #                         "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.8},

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
        "sensor_foot_touch_names" : ["FL_touch", "FR_touch", "RL_touch", "RR_touch"],   # Maintain the same order as contact_site_names
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
        
        "base_contact_body_names" : ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"],
        "leg_contact_body_names" : ["FL_thigh", "FL_calf", 
                                    "FR_thigh", "FR_calf", 
                                    "RL_thigh", "RL_calf", 
                                    "RR_thigh", "RR_calf"],
        "foot_body_names" : ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],  # Maintain the same order as contact_site_names
        "foot_fitted_ground_pairs" : [[0, 3], [1, 2]],                   # For the quadruped robot, the left front and right rear feet should touch the ground at the same time.

        # Config for reward
        "reward_coeff" : {
            "alive" : 0,
            "success" : 0,
            "failure" : 0,
            "contact" : 1,
            "foot_touch" : 0,
            "joint_angles" : 0.12, # 0.1
            "joint_accelerations" : 2.5e-7,
            "limit" : 0,
            "action_rate" : 0.01,
            "base_gyro" : 0,
            "base_accelerometer" : 0,
            "follow_command_linvel" : 1.3, # 1.0
            "follow_command_angvel" : 0.8, # 0.5
            "height" : 0,
            "body_lin_vel" : 2,
            "body_ang_vel" : 0.05,
            "body_orientation" : 1,
            "feet_air_time" : 1,
            "feet_self_contact" : 0,
            "feet_slip" : 0.05,
            "feet_wringing" : 0.0,
            "feet_fitted_ground" : 0.1,
            "fly" : 0.1,
            "stepping" : 0.12,  # 0.1            
        },

        # Robot's Self-Weight: Approximately 149.2 Newtons.
        # Static Foot Forces: Front feet ~44.8 N each; rear feet ~29.8 N each.
        # Dynamic Foot Forces (Walking at 1 m/s): Approximately 89.5 Newtons per foot during ground contact.
        # Front vs. Rear Legs: Front legs bear more force due to weight distribution and dynamics. 
        "foot_touch_force_threshold" : 100.0,
        "foot_touch_force_air_threshold" : 0.01,
        "foot_touch_force_step_threshold" : 5.0,
        "foot_touch_air_time_ideal" : 0.4,  # Go2 robot standing height is 0.4m. The ideal median stride rate for a Trot is around 0.4 seconds
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
        "push_interval_s" :         15,
        "max_push_vel_xy" :         1.0,
        "pos_random_range" :        2.0,    # randomize the x,y position of the robot in each episode
        
        # Config for ccurriculum learning
        "curriculum_learning" :     True,
        "curriculum_levels" : [
            # basic moving skills
            # {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 2.0, "rating": 0.5, "command_type": "move_slowly", },
            # {"name" : "smooth" ,                "offset" : [-55, 55, 0],   "distance": 2.0, "rating": 0.5, "command_type": "move_slowly", },
            # {"name" : "rough" ,                 "offset" : [-0, 55, 0],   "distance": 2.0, "rating": 0.5, "command_type": "move_slowly", },
            # {"name" : "smooth_slope" ,          "offset" : [0, -55, 0],    "distance": 2.0, "rating": 0.5, "command_type": "move_slowly", },

            # advanced moving skills
            {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "smooth" ,                "offset" : [-55, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "rough" ,                 "offset" : [-0, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            # {"name" : "smooth_slope" ,          "offset" : [0, -55, 0],    "distance": 3.0, "rating": 0.5, "command_type": "slope", },
            # {"name" : "rough" ,                "offset" : [-55, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            # {"name" : "rough_slope" ,           "offset" : [55, 0, 0],    "distance": 3.0, "rating": 0.5, "command_type": "slope", },
            # {"name" : "rough" ,                "offset" : [-55, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            # {"name" : "terrain_stairs_low" ,    "offset" : [-55, -55, 0],   "distance": 3.0, "rating": 0.5, "command_type": "stairs", },
            # {"name" : "rough" ,                "offset" : [-55, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            # {"name" : "terrain_brics" ,         "offset" : [55, -55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "slope", },
        ],
        "curriculum_commands" : {
            "move_slowly" : {
                "command_lin_vel_range_x" : [-0.0, 0.5], # x direction for forward max speed
                "command_lin_vel_range_y" : [-0.0, 0.0], # y direction for left/right max speed
                "command_lin_vel_threshold" : [0, 0.2], # min linear velocity to trigger moving
                "command_ang_vel_range" : 0.5,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },

            "flat_plane" : {
                "command_lin_vel_range_x" : [-0.5, 1.5], # x direction for forward max speed
                "command_lin_vel_range_y" : [-0.3, 0.3], # y direction for left/right max speed
                "command_lin_vel_threshold" : [-0.1, 0.2], # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },
            
            "slope" : {
                "command_lin_vel_range_x" : [-0.3, 1.0], # x direction for forward
                "command_lin_vel_range_y" : [-0.2, 0.2], # y direction for left/right
                "command_lin_vel_threshold" : [-0.1, 0.2], # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 20, # second to resample the command
            },
            
            "stairs" : {
                "command_lin_vel_range_x" : [0, 0.5], # x direction for forward
                "command_lin_vel_range_y" : [-0.1, 0.1], # y direction for left/right
                "command_lin_vel_threshold" : [0.0, 0.1], # min linear velocity to trigger moving
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