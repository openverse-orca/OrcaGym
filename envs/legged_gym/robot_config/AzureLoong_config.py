AzureLoongConfig = {
        # The order of the joints should be the same as they have been defined in the xml file.
        "base_joint_name" :     "float_base",
        "leg_joint_names" :     [
                                # "J_arm_r_01", # "J_arm_r_02", 
                                # "J_arm_l_01", # "J_arm_l_02",
                                 "J_hip_r_roll", 
                                #  "J_hip_r_yaw", 
                                 "J_hip_r_pitch", "J_knee_r_pitch", 
                                 "J_ankle_r_pitch", 
                                 # "J_ankle_r_roll",
                                 "J_hip_l_roll", 
                                #  "J_hip_l_yaw", 
                                 "J_hip_l_pitch", "J_knee_l_pitch", 
                                 "J_ankle_l_pitch", 
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
                                  "J_ankle_r_pitch": 0.605, 
                                #   "J_ankle_r_roll": 0.1,
                                  "J_hip_l_roll": 0.1, 
                                #   "J_hip_l_yaw": -0.2, 
                                  "J_hip_l_pitch": 0.5, "J_knee_l_pitch": -1.1, 
                                  "J_ankle_l_pitch": 0.605, 
                                #   "J_ankle_l_roll": -0.1,
                                  },
        
        "base_neutral_height_offset" : 0.10,    # the offset from max height tnpo standing natural height
        "base_born_height_offset" : 0.01,       # the offset from max height to standing natural height


        # The order of the actuators should be the same as they have been defined in the xml file.
        "actuator_names" :      [
                                #  "P_arm_r_01", # "P_arm_r_02",
                                #  "P_arm_l_01", # "P_arm_l_02",
                                 "P_hip_r_roll", 
                                #  "P_hip_r_yaw", 
                                 "P_hip_r_pitch", "P_knee_r_pitch", 
                                 "P_ankle_r_pitch", 
                                #  "P_ankle_r_roll",
                                 "P_hip_l_roll", 
                                #  "P_hip_l_yaw", 
                                 "P_hip_l_pitch", "P_knee_l_pitch", 
                                 "P_ankle_l_pitch", 
                                #  "P_ankle_l_roll",
                                 ],

        "actuator_type" :        "position",  # "torque" or "position"
        "action_scale" :         0.5,
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["rf-tc-front", "rf-tc-back", "lf-tc-front", "lf-tc-back"],
        
        "sensor_imu_framequat_name" :           "baselink-quat",
        "sensor_imu_gyro_name" :                "baselink-gyro",
        "sensor_imu_accelerometer_name" :       "baselink-velocity",
        "sensor_foot_touch_names" : ["rf-touch-front", "rf-touch-back", "lf-touch-front", "lf-touch-back"],  # Maintain the same order as contact_site_names
        "use_imu_sensor" : False,

        
        "base_contact_body_names" : ["base_link", 
                                     "Link_arm_r_01", "Link_arm_r_02", "Link_arm_r_03", "Link_arm_r_04", "Link_arm_r_05", "Link_arm_r_06", "Link_arm_r_07",
                                     "Link_arm_l_01", "Link_arm_l_02", "Link_arm_l_03", "Link_arm_l_04", "Link_arm_l_05", "Link_arm_l_06", "Link_arm_l_07",
                                     "Link_waist_pitch", "Link_waist_roll", "Link_waist_yaw", "Link_head_pitch", "Link_head_yaw",
                                     ],

        "leg_contact_body_names" : ["Link_hip_l_roll", "Link_hip_l_yaw", "Link_hip_l_pitch", 
                                     "Link_hip_r_roll", "Link_hip_r_yaw", "Link_hip_r_pitch",
                                    "Link_knee_l_pitch", "Link_knee_r_pitch", ],
        
        "foot_body_names" : ["Link_ankle_r_roll", "Link_ankle_r_roll", 
                             "Link_ankle_l_roll", "Link_ankle_l_roll"],  # Maintain the same order as contact_site_names
        "foot_fitted_ground_pairs" : [[0, 1], [2, 3]],                   # For the humanoid robot, the feet's front and back should touch the ground at the same time.

        # Config for reward
        "reward_coeff" : {
            "alive" : 0,
            "success" : 0,
            "failure" : 100,
            "contact" : 1,
            "foot_touch" : 0,
            "joint_angles" : 0.1,
            "joint_accelerations" : 2.5e-7,
            "limit" : 0,
            "action_rate" : 0.1,
            "base_gyro" : 0,
            "base_accelerometer" : 0,
            "follow_command_linvel" : 10, # From gymloong
            "follow_command_angvel" : 5,  # From gymloong
            "height" : 0,
            "body_lin_vel" : 2,
            "body_ang_vel" : 0.05,
            "body_orientation" : 0,
            "feet_air_time" : 1,
            "feet_self_contact" : 1,
            "feet_slip" : 0.5,
            "feet_wringing" : 0.0,
            "feet_fitted_ground" : 1.0,
            "fly" : 5.0,
            "stepping" : 1.0,
        },

        # Config from gymloong
        "foot_touch_force_threshold" : 1500.0,
        "foot_touch_force_air_threshold" : 0.01,
        "foot_touch_force_step_threshold" : 5.0,
        "foot_touch_air_time_ideal" : 0.5,  
        "foot_square_wave" : {
            "p5" :          0.5,
            "phase_freq" :  1.0,
            "eps" :         0.2,
        },

        # Config for randomization
        "randomize_friction" :      True,
        "friction_range" :          [0.5, 1.25],
        "randomize_base_mass" :     True,
        "added_mass_range" :        [-1., 10.],
        "push_robots" :             True,
        "push_interval_s" :         2,
        "max_push_vel_xy" :         0.5,
        "pos_random_range" :        2.0,    # randomize the x,y position of the robot in each episode
        
        # Config for ccurriculum learning
        "curriculum_learning" :     True,
        "curriculum_levels" : [
            {"name" : "default" ,               "offset" : [0, 0, 0],       "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "smooth" ,                "offset" : [-55, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "rough" ,                 "offset" : [-0, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "smooth_slope" ,          "offset" : [0, -55, 0],    "distance": 3.0, "rating": 0.5, "command_type": "slope", },
            {"name" : "rough" ,                 "offset" : [-0, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "rough_slope" ,           "offset" : [55, 0, 0],    "distance": 3.0, "rating": 0.5, "command_type": "slope", },
            {"name" : "rough" ,                 "offset" : [-0, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            # {"name" : "terrain_stairs_low" ,    "offset" : [-55, -55, 0],   "distance": 3.0, "rating": 0.5, "command_type": "stairs", },
            # {"name" : "rough" ,                 "offset" : [-0, 55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "flat_plane", },
            {"name" : "terrain_brics" ,         "offset" : [55, -55, 0],   "distance": 5.0, "rating": 0.5, "command_type": "slope", },
            # {"name" : "terrain_stairs_high" ,   "offset" : [-55, 0, 0],    "distance": 2.0, "rating": 0.5, "command_type": "stairs", },
        ],
        "curriculum_commands" : {
            "flat_plane" : {
                "command_lin_vel_range_x" : [-0.3, 1.5], # x direction for forward max speed
                "command_lin_vel_range_y" : [-0.3, 0.3], # y direction for left/right max speed
                "command_lin_vel_threshold" : [-0.1, 0.2], # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },
            
            "slope" : {
                "command_lin_vel_range_x" : [-0.3, 1.0], # x direction for forward
                "command_lin_vel_range_y" : [-0.3, 0.3], # y direction for left/right
                "command_lin_vel_threshold" : [-0.1, 0.2], # min linear velocity to trigger moving
                "command_ang_vel_range" : 1.0,  # max turning rate
                "command_resample_interval" : 7, # second to resample the command
            },
            
            "stairs" : {
                "command_lin_vel_range_x" : [0, 0.5], # x direction for forward
                "command_lin_vel_range_y" : [-0.1, 0.1], # y direction for left/right
                "command_lin_vel_threshold" : [0, 0.1], # min linear velocity to trigger moving
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
        
        # Config for nn network training
        "pi" : [512, 256, 128],  # 策略网络结构
        "vf" : [512, 256, 128],   # 值函数网络结构
        "n_steps" : 32,  # 每个环境采样步数
        "batch_size" : 1024,  # 批次大小            
        "learning_rate" : 0.0005,  # 学习率
        "gamma" : 0.99,  # 折扣因子
        "clip_range" : 0.2,  # PPO剪切范围
        "ent_coef" : 0.01,  # 熵系数
        "max_grad_norm" : 1,  # 最大梯度范数
    }