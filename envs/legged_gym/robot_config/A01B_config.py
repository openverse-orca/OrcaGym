A01BConfig = {
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
        "sensor_foot_touch_names" : ["fr_touch", "fl_touch", "rr_touch", "rl_touch"],   # Maintain the same order as contact_site_names
        "use_imu_sensor" : False,

        
        "base_contact_body_names" : ["trunk_link", "fl_abad", "fr_abad", "hl_abad", "hr_abad"],

        "leg_contact_body_names" : ["fl_thigh", "fl_calf", 
                                    "fr_thigh", "fr_calf", 
                                    "hl_thigh", "hl_calf", 
                                    "hr_thigh", "hr_calf"],
        
        "foot_body_names" : ["fl_foot", "fr_foot", "hl_foot", "hr_foot"],   # Maintain the same order as contact_site_names

        # Config for reward
        "reward_coeff" : {
            "alive" : 0,
            "success" : 0,
            "failure" : 0,
            "contact" : 1,
            "foot_touch" : 0,
            "joint_angles" : 0.1,
            "joint_accelerations" : 2.5e-7,
            "limit" : 0,
            "action_rate" : 0.01,
            "base_gyro" : 0,
            "base_accelerometer" : 0,
            "follow_command_linvel" : 1,
            "follow_command_angvel" : 0.5,
            "height" : 0,
            "body_lin_vel" : 2,
            "body_ang_vel" : 0.05,
            "body_orientation" : 0,
            "feet_air_time" : 1,
            "feet_self_contact" : 1,
            "feet_slip" : 0.5,
            "feet_wringing" : 0.0,
            "feet_fitted_ground" : 0.1,
            "fly" : 1,
            "stepping" : 1,
        },

        # 机器狗的自重： 约 61.44 kg。
        # 静止时每只脚的受力： 约 150.7 N。
        # 以 1 m/s 速度行走时，每只脚在触地时的受力： 约 301.4 N。        
        "foot_touch_force_threshold" : 350.0,
        "foot_touch_force_air_threshold" : 0.01,
        "foot_touch_force_step_threshold" : 5.0,
        "foot_touch_air_time_ideal" : 0.6,  

        # Config for randomization
        "randomize_friction" :      True,
        "friction_range" :          [0.5, 1.25],
        "randomize_base_mass" :     True,
        "added_mass_range" :        [-1., 10.],
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