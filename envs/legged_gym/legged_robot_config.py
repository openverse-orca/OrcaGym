import numpy as np

LeggedEnvConfig = {
    "TIME_STEP" : 0.005,                 # 仿真步长200Hz

    "FRAME_SKIP_REALTIME" : 1,           # 200Hz 推理步长
    "FRAME_SKIP_SHORT" : 4,              # 200Hz * 4 = 50Hz 推理步长
    "FRAME_SKIP_LONG" : 10,              # 200Hz * 10 = 20Hz 训练步长

    "EPISODE_TIME_VERY_SHORT" : 2,       # 每个episode的时间长度
    "EPISODE_TIME_SHORT" : 5,           
    "EPISODE_TIME_LONG" : 20,
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
        "leg_joint_names" :     ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        
        "base_joint_name" :     "base",
        
        "neutral_joint_angles" : {"FL_hip_joint": 0.0, "FL_thigh_joint": 0.7, "FL_calf_joint": -1.5,
                                "FR_hip_joint": 0.0, "FR_thigh_joint": 0.7, "FR_calf_joint": -1.5,
                                "RL_hip_joint": 0.0, "RL_thigh_joint": 0.7, "RL_calf_joint": -1.5,
                                "RR_hip_joint": 0.0, "RR_thigh_joint": 0.7, "RR_calf_joint": -1.5},
        
        "base_neutral_height_offset" : 0.12,    # the offset from max height to standing natural height
        "base_born_height_offset" : 0.001,       # the offset from max height to standing natural height

        "command_lin_vel_range_x" : 0.1,   # x direction for forward
        "command_lin_vel_range_y" : 0.0,   # y direction for left/right
        "command_ang_vel_range" : np.pi / 32,
        "command_ang_rate"      : 0.2,

        "actuator_names" :      ["FL_hip_actuator", "FL_thigh_actuator", "FL_calf_actuator",
                                "FR_hip_actuator", "FR_thigh_actuator", "FR_calf_actuator",
                                "RL_hip_actuator", "RL_thigh_actuator", "RL_calf_actuator",
                                "RR_hip_actuator", "RR_thigh_actuator", "RR_calf_actuator"],

        "actuator_type" :        "position",  # "torque" or "position"
        "action_scale" :         0.5,
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["FL_site", "FR_site", "RL_site", "RR_site"],

        "imu_mocap_name":       "imu_mocap",

        "sensor_imu_framequat_name" :           "imu_quat",
        "sensor_imu_gyro_name" :                "imu_omega",
        "sensor_imu_accelerometer_name" :       "imu_acc",
        "sensor_foot_touch_names" : ["FL_touch", "FR_touch", "RL_touch", "RR_touch"],

        "ground_contact_body_names" : ["Floor_Floor", "terrain_000_terrain", "terrain_001_terrain", "terrain_002_terrain", "terrain_003_terrain",
                                        "terrain_004_terrain", "terrain_005_terrain", "terrain_006_terrain", "terrain_007_terrain", "terrain_008_terrain",
                                        "terrain_009_terrain", "terrain_010_terrain", "terrain_011_terrain", "terrain_012_terrain", "terrain_013_terrain",
                                        "terrain_014_terrain", "terrain_015_terrain", "terrain_016_terrain", "terrain_017_terrain", "terrain_018_terrain",
                                        "terrain_019_terrain", "terrain_020_terrain", "terrain_021_terrain", "terrain_022_terrain", "terrain_023_terrain",
                                        "terrain_024_terrain", "terrain_025_terrain", "terrain_026_terrain", "terrain_027_terrain", "terrain_028_terrain",
                                        "terrain_029_terrain", "terrain_030_terrain", "terrain_031_terrain", "terrain_032_terrain", "terrain_033_terrain",
                                        "terrain_034_terrain", "terrain_035_terrain", "terrain_036_terrain", "terrain_037_terrain", "terrain_038_terrain",
                                        "terrain_039_terrain", "terrain_040_terrain", "terrain_041_terrain", "terrain_042_terrain", "terrain_043_terrain",
                                        "terrain_044_terrain", "terrain_045_terrain", "terrain_046_terrain", "terrain_047_terrain", "terrain_048_terrain",
                                        "terrain_049_terrain", "terrain_050_terrain", "terrain_051_terrain", "terrain_052_terrain", "terrain_053_terrain",
                                        "terrain_054_terrain", "terrain_055_terrain", "terrain_056_terrain", "terrain_057_terrain", "terrain_058_terrain",
                                        "terrain_059_terrain", "terrain_060_terrain", "terrain_061_terrain", "terrain_062_terrain", "terrain_063_terrain",],
        "base_contact_body_names" : ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"],
        "leg_contact_body_names" : ["FL_thigh", "FL_calf", 
                                    "FR_thigh", "FR_calf", 
                                    "RL_thigh", "RL_calf", 
                                    "RR_thigh", "RR_calf"],

        # Robot's Self-Weight: Approximately 149.2 Newtons.
        # Static Foot Forces: Front feet ~44.8 N each; rear feet ~29.8 N each.
        # Dynamic Foot Forces (Walking at 1 m/s): Approximately 89.5 Newtons per foot during ground contact.
        # Front vs. Rear Legs: Front legs bear more force due to weight distribution and dynamics. 
        "foot_touch_force_threshold" : 100.0,
        "foot_touch_force_air_threshold" : 0.01,
        "foot_touch_air_time_threshold" : 0.5,  # second

        # Config for logging
        "log_env_ids" :     [0],
        "log_agent_names" : ["go2_000"]
    },
    "A01B": {
        "leg_joint_names" :     ["fr_joint0", "fr_joint1", "fr_joint2", 
                                "fl_joint0", "fl_joint1", "fl_joint2",
                                "hr_joint0", "hr_joint1", "hr_joint2",
                                "hl_joint0", "hl_joint1", "hl_joint2"],
        
        "base_joint_name" :     "trunk",
        
        # Init the robot in a standing position. 
        # Maintain the order of the joints same as the joint_names for reset basic pos or computing the reward easily.
        "neutral_joint_angles" : {"fr_joint0": 0.0, "fr_joint1": -0.8, "fr_joint2": 1.8,
                                "fl_joint0": 0.0, "fl_joint1": -0.8, "fl_joint2": 1.8,
                                "hr_joint0": 0.0, "hr_joint1": -0.8, "hr_joint2": 1.8,
                                "hl_joint0": 0.0, "hl_joint1": -0.8, "hl_joint2": 1.8},

        "base_neutral_height_offset" : 0.25,    # the offset from max height to standing natural height
        

        "actuator_names" :      ["fr_abad_actuator", "fr_thigh_actuator", "fr_calf_actuator",
                                "fl_abad_actuator", "fl_thigh_actuator", "fl_calf_actuator",
                                "hr_abad_actuator", "hr_thigh_actuator", "hr_calf_actuator",
                                "hl_abad_actuator", "hl_thigh_actuator", "hl_calf_actuator"],

        "actuator_type" :        "torque",  # "torque" or "position"
        "action_scale" :         0.5,
                                 
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["fr_site", "fl_site", "hr_site", "hl_site"],

        "imu_mocap_name":       "imu_mocap",

        "sensor_imu_framequat_name" :           "imu_quat",
        "sensor_imu_gyro_name" :                "imu_omega",
        "sensor_imu_accelerometer_name" :       "imu_acc",
        "sensor_foot_touch_names" : ["fr_touch", "fl_touch", "rr_touch", "rl_touch"],

        "ground_contact_body_names" : ["Floor_Floor", "terrain_000_terrain", "terrain_001_terrain", "terrain_002_terrain", "terrain_003_terrain",
                                        "terrain_004_terrain", "terrain_005_terrain", "terrain_006_terrain", "terrain_007_terrain", "terrain_008_terrain",
                                        "terrain_009_terrain", "terrain_010_terrain", "terrain_011_terrain", "terrain_012_terrain", "terrain_013_terrain",
                                        "terrain_014_terrain", "terrain_015_terrain"],
        "base_contact_body_names" : ["trunk_link"],
        "leg_contact_body_names" : ["fr_thigh", "fr_calf", "fl_thigh", "fl_calf", "hr_thigh", "hr_calf", "hl_thigh", "hl_calf"],

        # 机器狗的自重： 约 61.44 kg。
        # 静止时每只脚的受力： 约 150.7 N。
        # 以 1 m/s 速度行走时，每只脚在触地时的受力： 约 301.4 N。        
        "foot_touch_force_threshold" : 350.0,
    }
}