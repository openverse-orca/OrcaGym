
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
        
        "neutral_joint_angles" : {"FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.6,
                                "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.6,
                                "RL_hip_joint": 0.0, "RL_thigh_joint": 0.8, "RL_calf_joint": -1.6,
                                "RR_hip_joint": 0.0, "RR_thigh_joint": 0.8, "RR_calf_joint": -1.6},
        

        "actuator_names" :      ["fr_tau0", "fr_tau1", "fr_tau2",
                                "fl_tau0", "fl_tau1", "fl_tau2",
                                "hr_tau0", "hr_tau1", "hr_tau2",
                                "hl_tau0", "hl_tau1", "hl_tau2"],
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["fr_site", "fl_site", "hr_site", "hl_site"],

        "sensor_imu_framequat_name" :           "imu_quat",
        "sensor_imu_gyro_name" :                "imu_omega",
        "sensor_imu_accelerometer_name" :       "imu_acc",
        "sensor_foot_touch_names" : ["fr_touch", "fl_touch", "rr_touch", "rl_touch"],

        "base_contact_body_names" : ["trunk_link"],
        "leg_contact_body_names" : ["fr_thigh", "fr_calf", "fl_thigh", "fl_calf", "hr_thigh", "hr_calf", "hl_thigh", "hl_calf"],
    }
}