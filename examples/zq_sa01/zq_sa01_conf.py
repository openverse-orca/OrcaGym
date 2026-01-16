# ZQ SA01 机器人配置参数

# 控制参数
action_scale = 0.5
control_type = 'P'
decimation = 10  # 控制频率: 1000Hz / 10 = 100Hz

# PD控制器增益 - 刚度 (N·m/rad)
stiffness = {
    '1_joint': 50,   # 髋关节侧摆
    '2_joint': 50,   # 髋关节旋转
    '3_joint': 70,   # 髋关节俯仰
    '4_joint': 70,   # 膝关节
    '5_joint': 20,   # 踝关节俯仰
    '6_joint': 20    # 踝关节侧摆
}

# PD控制器增益 - 阻尼 (N·m·s/rad)
damping = {
    '1_joint': 5.0,
    '2_joint': 5.0,
    '3_joint': 7.0,
    '4_joint': 7.0,
    '5_joint': 2.0,
    '6_joint': 2.0
}

# 默认关节角度 (rad) - action=0时的目标姿态
default_joint_angles = {
    'leg_l1_joint': 0.0,     # 左髋侧摆
    'leg_l2_joint': 0.0,     # 左髋旋转
    'leg_l3_joint': -0.24,   # 左髋俯仰 (-13.75°)
    'leg_l4_joint': 0.48,    # 左膝关节 (27.5°)
    'leg_l5_joint': -0.24,   # 左踝俯仰 (-13.75°)
    'leg_l6_joint': 0.0,     # 左踝侧摆
    'leg_r1_joint': 0.0,     # 右髋侧摆
    'leg_r2_joint': 0.0,     # 右髋旋转
    'leg_r3_joint': -0.24,   # 右髋俯仰
    'leg_r4_joint': 0.48,    # 右膝关节
    'leg_r5_joint': -0.24,   # 右踝俯仰
    'leg_r6_joint': 0.0      # 右踝侧摆
}

# 关节限位 (rad)
joint_limits = {
    'leg_l1_joint': (-0.523, 0.523),    # ±30°
    'leg_l2_joint': (-0.3, 0.3),        # ±17°
    'leg_l3_joint': (-1.204, 1.204),    # ±69°
    'leg_l4_joint': (-0.15, 2.268),     # -8.6° ~ 130°
    'leg_l5_joint': (-1.0, 0.6),        # -57° ~ 34°
    'leg_l6_joint': (-0.6, 0.6),        # ±34°
    'leg_r1_joint': (-0.523, 0.523),
    'leg_r2_joint': (-0.3, 0.3),
    'leg_r3_joint': (-1.204, 1.204),
    'leg_r4_joint': (-0.15, 2.268),
    'leg_r5_joint': (-1.0, 0.6),
    'leg_r6_joint': (-0.6, 0.6)
}

# 力矩限制 (N·m)
torque_limits = {
    'all_joints': 200.0  # 所有关节统一为 ±200 N·m
}

# 初始状态
init_base_pos = [0.0, 0.0, 1.1]  # 基座初始位置 (x, y, z)
init_base_rot = [0.0, 0.0, 0.0, 1.0]  # 基座初始朝向 (x, y, z, w)

# 仿真参数
dt = 0.001  # 仿真步长 (1ms)
gravity = [0.0, 0.0, -9.81]

# 关节名称列表（按顺序）
joint_names = [
    'leg_l1_joint', 'leg_l2_joint', 'leg_l3_joint',
    'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint',
    'leg_r1_joint', 'leg_r2_joint', 'leg_r3_joint',
    'leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint'
]

# 脚部链接索引（用于接触力检测）
feet_links = ['leg_l6_link', 'leg_r6_link']

# PD控制器计算公式说明
"""
目标位置计算:
    target_pos = action * action_scale + default_joint_angles + motor_zero_offset
    
力矩计算:
    position_error = target_pos - current_pos
    torque = Kp * position_error - Kd * current_vel
    torque_final = clip(torque * torque_multiplier, -torque_limit, +torque_limit)
    
其中:
    - Kp: stiffness (刚度)
    - Kd: damping (阻尼)
    - motor_zero_offset: 电机零点偏移（域随机化）
    - torque_multiplier: 力矩乘数（域随机化）
"""


