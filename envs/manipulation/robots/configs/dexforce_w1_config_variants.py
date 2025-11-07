"""
Dexforce W1 配置变体
用于测试不同的初始姿态配置

使用方法：
在 dexforce_w1_config.py 中引入并使用某个变体配置
"""

# 基础配置模板
BASE_CONFIG = {
    "robot_type": "dual_arm",
    "base": {
        "base_body_name": "base_link",
        "base_joint_name": "base_joint",
        "dummy_joint_name": "dummy_joint",
    },
}

DEFAULT_NEUTRAL = {
    "right": [-1.9, -1.0, -0.001, -0.4292, -1.5708, -1.5708, 0.0],
    "left": [1.9, 1.0, 0.001, 0.4292, 1.5708, 1.5708, 0.0],
}

# 变体1：原始配置（与OpenLoong相同）
VARIANT_ORIGINAL = {
    **BASE_CONFIG,
    "right_arm": {
        "joint_names": ["RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7"],
        "neutral_joint_values": [-1.9, 0.5, 0, 2.0, -1.5708, 0, 0],
        "motor_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
        "position_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7"],
        "neutral_joint_values": [1.9, -0.5, 0, 2.0, 1.5708, 0, 0],
        "motor_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
        "position_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
        "ee_center_site_name": "ee_center_site",
    },
}

# 变体2：反转J2（肩关节pitch）
VARIANT_FLIP_J2 = {
    **BASE_CONFIG,
    "right_arm": {
        "joint_names": ["RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7"],
        "neutral_joint_values": [-1.9, -0.5, 0, 2.0, -1.5708, 0, 0],  # J2: 0.5 -> -0.5
        "motor_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
        "position_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7"],
        "neutral_joint_values": [1.9, 0.5, 0, 2.0, 1.5708, 0, 0],  # J2: -0.5 -> 0.5
        "motor_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
        "position_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
        "ee_center_site_name": "ee_center_site",
    },
}

# 变体3：反转J4（肘关节）
VARIANT_FLIP_J4 = {
    **BASE_CONFIG,
    "right_arm": {
        "joint_names": ["RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7"],
        "neutral_joint_values": [-1.9, 0.5, 0, -2.0, -1.5708, 0, 0],  # J4: 2.0 -> -2.0
        "motor_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
        "position_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7"],
        "neutral_joint_values": [1.9, -0.5, 0, -2.0, 1.5708, 0, 0],  # J4: 2.0 -> -2.0
        "motor_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
        "position_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
        "ee_center_site_name": "ee_center_site",
    },
}

# 变体4：同时反转J2和J4
VARIANT_FLIP_J2_J4 = {
    **BASE_CONFIG,
    "right_arm": {
        "joint_names": ["RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7"],
        "neutral_joint_values": [-1.9, -0.5, 0, -2.0, -1.5708, 0, 0],
        "motor_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
        "position_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7"],
        "neutral_joint_values": [1.9, 0.5, 0, -2.0, 1.5708, 0, 0],
        "motor_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
        "position_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
        "ee_center_site_name": "ee_center_site",
    },
}

# 变体5：保守调整（小幅度）
VARIANT_CONSERVATIVE = {
    **BASE_CONFIG,
    "right_arm": {
        "joint_names": ["RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7"],
        "neutral_joint_values": [-1.5, 0.3, 0, 1.8, -1.3, 0, 0],
        "motor_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
        "position_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7"],
        "neutral_joint_values": [1.5, -0.3, 0, 1.8, 1.3, 0, 0],
        "motor_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
        "position_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
        "ee_center_site_name": "ee_center_site",
    },
}

# 变体6：完全对称（两臂使用相同值）
VARIANT_SYMMETRIC = {
    **BASE_CONFIG,
    "right_arm": {
        "joint_names": ["RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7"],
        "neutral_joint_values": [0, 0, 0, 1.5, 0, 0, 0],  # 简单对称姿态
        "motor_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
        "position_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": ["LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7"],
        "neutral_joint_values": [0, 0, 0, 1.5, 0, 0, 0],  # 与右臂相同
        "motor_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
        "position_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
        "ee_center_site_name": "ee_center_site",
    },
}

# 配置说明
VARIANT_DESCRIPTIONS = {
    "ORIGINAL": "原始配置（与OpenLoong相同）",
    "FLIP_J2": "反转J2（肩关节pitch）- 如果一个手臂向上一个向下",
    "FLIP_J4": "反转J4（肘关节）- 如果肘部弯曲方向相反",
    "FLIP_J2_J4": "同时反转J2和J4 - 组合修正",
    "CONSERVATIVE": "保守调整 - 小幅度修改所有关节",
    "SYMMETRIC": "完全对称 - 两臂使用相同值（用于诊断）",
}

# 使用示例：
# 在 dexforce_w1_config.py 中：
# from .dexforce_w1_config_variants import VARIANT_FLIP_J2 as dexforce_w1_config

