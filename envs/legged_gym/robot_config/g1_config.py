g1Config = {
        # The order of the joints should be the same as they have been defined in the xml file.
        "base_joint_name" :     "floating_base_joint",
        "leg_joint_names" :     [
                                "left_hip_pitch_joint",
                                "left_hip_roll_joint",
                                "left_hip_yaw_joint",
                                "left_knee_joint",
                                "left_ankle_pitch_joint",
                                "left_ankle_roll_joint",
                                
                                "right_hip_pitch_joint",
                                "right_hip_roll_joint",
                                "right_hip_yaw_joint",
                                "right_knee_joint",
                                "right_ankle_pitch_joint",
                                "right_ankle_roll_joint",
                                
                                "waist_yaw_joint",
                                "waist_roll_joint",
                                "waist_pitch_joint",
                                
                                "left_shoulder_pitch_joint",
                                "left_shoulder_roll_joint",
                                "left_shoulder_yaw_joint",
                                "left_elbow_joint",
                                "left_wrist_roll_joint",
                                "left_wrist_pitch_joint",
                                "left_wrist_yaw_joint",
                                
                                "right_shoulder_pitch_joint",
                                "right_shoulder_roll_joint",
                                "right_shoulder_yaw_joint",
                                "right_elbow_joint",
                                "right_wrist_roll_joint",
                                "right_wrist_pitch_joint",
                                "right_wrist_yaw_joint",
        ],
        
        
        # Init the robot in a standing position. Keep the order of the joints same as the joint_names 
        # for reset basic pos or computing the reward easily.
        "neutral_joint_angles" : {
                                "left_hip_pitch_joint": 0.0,
                                "left_hip_roll_joint": 0.0,
                                "left_hip_yaw_joint": 0.0,
                                "left_knee_joint": 0.0,
                                "left_ankle_pitch_joint": 0.0,
                                "left_ankle_roll_joint": 0.0,
                                
                                "right_hip_pitch_joint": 0.0,
                                "right_hip_roll_joint": 0.0,
                                "right_hip_yaw_joint": 0.0,
                                "right_knee_joint": 0.0,
                                "right_ankle_pitch_joint": 0.0,
                                "right_ankle_roll_joint": 0.0,
                                
                                "waist_yaw_joint": 0.0,
                                "waist_roll_joint": 0.0,
                                "waist_pitch_joint": 0.0,
                                
                                "left_shoulder_pitch_joint": 0.2,
                                "left_shoulder_roll_joint": 0.2,
                                "left_shoulder_yaw_joint": 0.0,
                                "left_elbow_joint": 1.28,
                                "left_wrist_roll_joint": 0.0,
                                "left_wrist_pitch_joint": 0.0,
                                "left_wrist_yaw_joint": 0.0,
                                
                                "right_shoulder_pitch_joint": 0.2,
                                "right_shoulder_roll_joint": -0.2,
                                "right_shoulder_yaw_joint": 0.0,
                                "right_elbow_joint": 1.28,
                                "right_wrist_roll_joint": 0.0,
                                "right_wrist_pitch_joint": 0.0,
                                "right_wrist_yaw_joint": 0.0,
        },
        
        "neutral_joint_angles_coeff" : {
                                "left_hip_pitch_joint": 0.02,
                                "left_hip_roll_joint": 0.15,
                                "left_hip_yaw_joint": 0.15,
                                "left_knee_joint": 0.02,
                                "left_ankle_pitch_joint": 0.02,
                                "left_ankle_roll_joint": 0.02,
                                
                                "right_hip_pitch_joint": 0.02,
                                "right_hip_roll_joint": 0.15,
                                "right_hip_yaw_joint": 0.15,
                                "right_knee_joint": 0.02,
                                "right_ankle_pitch_joint": 0.02,
                                "right_ankle_roll_joint": 0.02,
                                
                                "waist_yaw_joint": 0.2,
                                "waist_roll_joint": 0.2,
                                "waist_pitch_joint": 0.2,
                                
                                "left_shoulder_pitch_joint": 0.15,
                                "left_shoulder_roll_joint": 0.2,
                                "left_shoulder_yaw_joint": 0.2,
                                "left_elbow_joint": 0.15,
                                "left_wrist_roll_joint": 0.2,
                                "left_wrist_pitch_joint": 0.2,
                                "left_wrist_yaw_joint": 0.2,
                                
                                "right_shoulder_pitch_joint": 0.15,
                                "right_shoulder_roll_joint": 0.2,
                                "right_shoulder_yaw_joint": 0.2,
                                "right_elbow_joint": 0.15,
                                "right_wrist_roll_joint": 0.2,
                                "right_wrist_pitch_joint": 0.2,
                                "right_wrist_yaw_joint": 0.2,
        },
        
        
        "base_neutral_height_offset" : 0.00,    # the offset from max height tnpo standing natural height
        "base_born_height_offset" : 0.01,       # the offset from max height to standing natural height


        # The order of the actuators should be the same as they have been defined in the xml file.
        "actuator_names" :      [
                                "left_hip_pitch_joint",
                                "left_hip_roll_joint",
                                "left_hip_yaw_joint",
                                "left_knee_joint",
                                "left_ankle_pitch_joint",
                                "left_ankle_roll_joint",
                                
                                "right_hip_pitch_joint",
                                "right_hip_roll_joint",
                                "right_hip_yaw_joint",
                                "right_knee_joint",
                                "right_ankle_pitch_joint",
                                "right_ankle_roll_joint",
                                
                                "waist_yaw_joint",
                                "waist_roll_joint",
                                "waist_pitch_joint",
                                
                                "left_shoulder_pitch_joint",
                                "left_shoulder_roll_joint",
                                "left_shoulder_yaw_joint",
                                "left_elbow_joint",
                                "left_wrist_roll_joint",
                                "left_wrist_pitch_joint",
                                "left_wrist_yaw_joint",
                                
                                "right_shoulder_pitch_joint",
                                "right_shoulder_roll_joint",
                                "right_shoulder_yaw_joint",
                                "right_elbow_joint",
                                "right_wrist_roll_joint",
                                "right_wrist_pitch_joint",
                                "right_wrist_yaw_joint",                                
        ],

        "actuator_type" :        "position",  # "torque" or "position"
        "action_scale" :         0.5,
        
        "action_scale_mask" :  {
                                "left_hip_pitch_joint" : 1.0,
                                "left_hip_roll_joint" : 1.0,
                                "left_hip_yaw_joint" : 1.0,
                                "left_knee_joint" : 1.0,
                                "left_ankle_pitch_joint" : 1.0,
                                "left_ankle_roll_joint" : 1.0,
                                
                                "right_hip_pitch_joint" : 1.0,
                                "right_hip_roll_joint" : 1.0,
                                "right_hip_yaw_joint" : 1.0,
                                "right_knee_joint" : 1.0,
                                "right_ankle_pitch_joint" : 1.0,
                                "right_ankle_roll_joint" : 1.0,
                                
                                "waist_yaw_joint" : 0.0,
                                "waist_roll_joint" : 0.0,
                                "waist_pitch_joint" : 0.0,
                                
                                "left_shoulder_pitch_joint" : 0.0,
                                "left_shoulder_roll_joint" : 0.0,
                                "left_shoulder_yaw_joint" : 0.0,
                                "left_elbow_joint" : 0.0,
                                "left_wrist_roll_joint" : 0.0,
                                "left_wrist_pitch_joint" : 0.0,
                                "left_wrist_yaw_joint" : 0.0,
                                
                                "right_shoulder_pitch_joint" : 0.0,
                                "right_shoulder_roll_joint" : 0.0,
                                "right_shoulder_yaw_joint" : 0.0,
                                "right_elbow_joint" : 0.0,
                                "right_wrist_roll_joint" : 0.0,
                                "right_wrist_pitch_joint" : 0.0,
                                "right_wrist_yaw_joint" : 0.0,                  
        },
        
        
        "imu_site_name" :       "imu_in_pelvis",
        "contact_site_names" :  ["lf-tc-front", "lf-tc-back", "rf-tc-front", "rf-tc-back"],
        
        "sensor_imu_framequat_name" :           "imu-pelvis-quat",
        "sensor_imu_gyro_name" :                "imu-pelvis-angular-velocity",
        "sensor_imu_accelerometer_name" :       "imu-pelvis-linear-acceleration",
        "sensor_foot_touch_names" : ["lf-touch-front", "lf-touch-back", "rf-touch-front", "rf-touch-back"],  # Maintain the same order as contact_site_names
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
        
        "base_contact_body_names" : ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link",
                                     ],

        "leg_contact_body_names" : ["left_hip_pitch_link", "left_hip_yaw_link", "left_knee_link",
                                    "right_hip_pitch_link", "right_hip_yaw_link", "right_knee_link",
                                     "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
                                     "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
                                    ],
        
        "foot_body_names" : ["left_ankle_roll_link", "left_ankle_roll_link", 
                             "right_ankle_roll_link", "right_ankle_roll_link"],  # Maintain the same order as contact_site_names
        "foot_fitted_ground_pairs" : [[0, 1], [2, 3]],                   # For the humanoid robot, the feet's front and back should touch the ground at the same time.

        # Config for reward
        "reward_coeff" : {
            "alive" : 0,
            "success" : 0,
            "failure" : 200,
            "contact" : 1,
            "foot_touch" : 0,
            "joint_angles" : 1.0,
            "joint_accelerations" : 2.5e-7,
            "limit" : 0,
            "action_rate" : 0.1,
            "base_gyro" : 0,
            "base_accelerometer" : 0,
            "follow_command_linvel" : 1,
            "follow_command_angvel" : 1,
            "height" : 0,
            "body_lin_vel" : 2,
            "body_ang_vel" : 0.05,
            "body_orientation" : 2,
            "feet_air_time" : 1,
            "feet_self_contact" : 1,
            "feet_slip" : 0.25,
            "feet_wringing" : 0.0,
            "feet_fitted_ground" : 1.0,
            "fly" : 1.0,
            "stepping" : 1.0,
        },

        # Config from gymloong
        "foot_touch_force_threshold" : 500.0,
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
        "randomize_base_mass" :     False,
        "added_mass_range" :        [-1., 1.],
        "push_robots" :             True,
        "push_interval_s" :         2,
        "max_push_vel_xy" :         0.5,
        "pos_random_range" :        2.0,    # randomize the x,y position of the robot in each episode
        
        # Config for ccurriculum learning
        "curriculum_learning" :     False, #True,
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
                "command_lin_vel_range_x" : [0, 1.0], # x direction for forward max speed
                "command_lin_vel_range_y" : [-0.1, 0.1], # y direction for left/right max speed
                "command_lin_vel_threshold" : [0, 0.2], # min linear velocity to trigger moving
                "command_ang_vel_range" : 0.5,  # max turning rate
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
        "log_agent_names" : ["g1_000"],
        
        # Config for visualization
        "visualize_command_agent_names" : ["g1_000"],
        "command_indicator_name" : "command_indicator_mocap",

        # Config for playable agent
        "playable_agent_name" : "g1_000",        
        
        
        # Config for nn network training
        "pi" : [512, 256, 128],  # 策略网络结构
        "vf" : [512, 256, 128],   # 值函数网络结构
        "n_steps" : 32,  # 每个环境采样步数
        "batch_size" : 1024  # 批次大小
    }   