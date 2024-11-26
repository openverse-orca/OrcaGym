
LeggedRobotConfig = {
    "Go2": {
        "leg_joint_names" :     ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        
        "base_joint_name" :     "base",
        
        "neutral_joint_angles" : {"FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.6,
                                "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.6,
                                "RL_hip_joint": 0.0, "RL_thigh_joint": 0.8, "RL_calf_joint": -1.6,
                                "RR_hip_joint": 0.0, "RR_thigh_joint": 0.8, "RR_calf_joint": -1.6},
        

        "actuator_names" :      ["FR_hip", "FR_thigh", "FR_calf",
                                "FL_hip", "FL_thigh", "FL_calf",
                                "RR_hip", "RR_thigh", "RR_calf",
                                "RL_hip", "RL_thigh", "RL_calf"],
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["base_contact_box", "base_contact_cylinder", "base_contact_sphere"],

        "sensor_imu_names" :    ["imu_quat", "imu_omega", "imu_acc"],
        "sensor_base_touch_names" : ["base_touch_box", "base_touch_cylinder", "base_touch_sphere"],

        "body_contact_force_threshold" : [0.5, 0.5, 0.5]
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

        "actuator_type" :        "position",  # "torque" or "position"
                                 
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["fr_site", "fl_site", "hr_site", "hl_site"],

        "imu_mocap_name":       "imu_mocap",

        "sensor_imu_framequat_name" :           "imu_quat",
        "sensor_imu_gyro_name" :                "imu_omega",
        "sensor_imu_accelerometer_name" :       "imu_acc",
        "sensor_foot_touch_names" : ["fr_touch", "fl_touch", "rr_touch", "rl_touch"],

        "ground_contact_body_names" : ["Floor_Floor"],
        "base_contact_body_names" : ["trunk_link"],
        "leg_contact_body_names" : ["fr_thigh", "fr_calf", "fl_thigh", "fl_calf", "hr_thigh", "hr_calf", "hl_thigh", "hl_calf"],

        # 机器狗的自重： 约 61.44 kg。
        # 静止时每只脚的受力： 约 150.7 N。
        # 以 1 m/s 速度行走时，每只脚在触地时的受力： 约 301.4 N。        
        "foot_touch_force_threshold" : 350.0,
    }
}