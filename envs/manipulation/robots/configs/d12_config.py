d12_config = {
    "robot_type": "dual_arm",
    "base": {
        "base_body_name": "base_link",
        "base_joint_name": "base_joint",
        "dummy_joint_name": "dummy_joint",
    },
    "right_arm": {
        "joint_names": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
     #   "neutral_joint_values": [0.698, -1.047, -0.9599, -0.2618, -1.57, 0.0, 0.0],
        "neutral_joint_values": [0.0, 0, -0, 0, 0, 0.0, 0.0],
        "motor_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
       # "position_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
     #   "neutral_joint_values": [-0.698, 1.047, 0.9599, 0.2618, 1.57, 0.0, 0.0],
        "neutral_joint_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "motor_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
        #"position_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
        "ee_center_site_name": "ee_center_site",
    },
}
