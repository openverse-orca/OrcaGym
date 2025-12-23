"""
G1 23DOF 机器人配置
G1 23DOF Robot Configuration

基于 g1_23dof.xml 文件生成
注意：23DOF 版本只有 5 个手臂关节（没有 wrist_pitch 和 wrist_yaw），也没有灵巧手
"""

g1_23dof_config = {
    "robot_type": "dual_arm",
    "hand_type": "none",
    "base": {
        "base_body_name": "pelvis",              # 基座 body 名称
        "base_joint_name": "floating_base_joint", # 基座关节（free joint）
        "dummy_joint_name": "floating_base_joint", # 虚拟关节
    },
    "right_arm": {
        # 右臂关节（从肩部到手腕，共5个关节，没有 wrist_pitch 和 wrist_yaw）
        "joint_names": [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint"
        ],
        # 右臂中性位置（弧度）
        "neutral_joint_values": [0.0, 0.0, 0.0, 0.0, 0.0],
        # 右臂电机执行器名称（与关节名称相同）
        "motor_names": [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint"
        ],
        # 末端执行器中心点 site 名称
        # 注意：如果 XML 中没有定义 site，需要在 XML 中添加
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        # 左臂关节（从肩部到手腕，共5个关节）
        "joint_names": [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint"
        ],
        # 左臂中性位置（弧度）
        "neutral_joint_values": [0.0, 0.0, 0.0, 0.0, 0.0],
        # 左臂电机执行器名称（与关节名称相同）
        "motor_names": [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint"
        ],
        # 末端执行器中心点 site 名称
        "ee_center_site_name": "ee_center_site",
    },

    
}

