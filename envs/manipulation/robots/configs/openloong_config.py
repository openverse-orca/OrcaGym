openloong_config = {
    "robot_type": "dual_arm",
    "base": {
        "base_body_name": "base_link",
        "base_joint_name": "base_joint",
        "dummy_joint_name": "dummy_joint",
    },
    "neck": {
        "yaw_joint_name": "J_head_yaw",
        "pitch_joint_name": "J_head_pitch",
        "yaw_actuator_name": "M_head_yaw",
        "pitch_actuator_name": "M_head_pitch",
        "neck_center_site_name": "neck_center_site"
    },
    "right_arm": {
        "joint_names": ["J_arm_r_01", "J_arm_r_02", "J_arm_r_03", "J_arm_r_04", "J_arm_r_05", "J_arm_r_06", "J_arm_r_07"],
        "neutral_joint_values": [0.706, -0.594, -2.03, 1.65, -2.131, -0.316, -0.705],
        "motor_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
        "position_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["J_arm_l_01", "J_arm_l_02", "J_arm_l_03", "J_arm_l_04", "J_arm_l_05", "J_arm_l_06", "J_arm_l_07"],
        "neutral_joint_values": [-1.041, 0.721, 2.52, 1.291, 2.112, 0.063, 0.92],
        "motor_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
        "position_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
        "ee_center_site_name": "ee_center_site",
    },
}