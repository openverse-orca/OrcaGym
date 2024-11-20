
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
    }
}