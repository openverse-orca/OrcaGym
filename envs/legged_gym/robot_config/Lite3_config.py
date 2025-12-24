
import numpy as np

RewardConfig = {
    "flat_terrain": {
        "alive" : 0,                     # 存活奖励
        "success" : 0,                   # 成功奖励
        "failure" : 0,                   # 失败惩罚
        "leg_contact" : 2,               # 腿部身体接触惩罚
        "body_contact" : 10,              # 身体接触惩罚
        "foot_touch" : 0.00,             # 重踏惩罚
        "joint_angles" : 0.5,            # 关节偏离自然站立角度惩罚
        "joint_accelerations" : 2.5e-7,  # 关节加速度惩罚
        "limit" : 0.00,                 # Action极限值惩罚
        "action_rate" : 0.01,           # Action平滑
        "base_gyro" : 0,                
        "base_accelerometer" : 0,
        "follow_command_linvel" : 1,    # 跟随指令速度奖励
        "follow_command_angvel" : 0.5,  # 跟随指令角速度奖励
        "height" : 0,                   # 身体高度惩罚
        "body_lin_vel" : 2,             # 身体上下线速度惩罚
        "body_ang_vel" : 0.05,         # 身体倾斜角速度惩罚
        "body_orientation" : 1,         # 身体姿态惩罚
        "feet_air_time" : 1,          # 足底离地时间，小于给定的世间惩罚
        "feet_self_contact" : 0,        # 足底自接触惩罚
        "feet_slip" : 0.0,             # 接触时，足底线速度
        "feet_wringing" : 0,         # 接触时，足底角速度
        "feet_fitted_ground" : 0.0,    # 鼓励对角步态，避免单侧滑步
        "feet_swing_height" : 0,    # 鼓励足底离地高度在理想范围内
        "fly" : 0.0,                    # 四足离地惩罚
        "stepping" : 0.0,                 # 无指令时，踏步惩罚
        "torques" : 1e-5,                # 关节力矩惩罚
        "joint_qpos_limits" : 0.0,      # 关节角度极限值惩罚
        "joint_qvel_limits" : 0.0,       # 关节速度极限值惩罚
        "torque_limits" : 0.0,       # 避免关节力矩过大
        "contact_no_vel" : 0,            # 接触时，足底线速度越小越好
    },
}

Lite3Config = {
        
        # The order of the joints should be the same as they have been defined in the xml file.
        "base_joint_name" :     "torso",
        "leg_joint_names" :     ["FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint", 
                                "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
                                "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
                                "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"],
        
        # Init the robot in a standing position. Keep the order of the joints same as the joint_names 
        # for reset basic pos or computing the reward easily.
        "neutral_joint_angles" : {"FL_HipX_joint": 0.0, "FL_HipY_joint": -1.0, "FL_Knee_joint": 1.8,
                                "FR_HipX_joint": 0.0, "FR_HipY_joint": -1.0, "FR_Knee_joint": 1.8,
                                "HL_HipX_joint": 0.0, "HL_HipY_joint": -1.0, "HL_Knee_joint": 1.8,
                                "HR_HipX_joint": 0.0, "HR_HipY_joint": -1.0, "HR_Knee_joint": 1.8},

        "neutral_joint_angles_coeff" : {"FL_HipX_joint": 1, "FL_HipY_joint": 0.0, "FL_Knee_joint": 0,
                                        "FR_HipX_joint": 1, "FR_HipY_joint": 0.0, "FR_Knee_joint": 0,
                                        "HL_HipX_joint": 1, "HL_HipY_joint": 0.0, "HL_Knee_joint": 0,
                                        "HR_HipX_joint": 1, "HR_HipY_joint": 0.0, "HR_Knee_joint": 0},
        
        "base_neutral_height_offset" : 0.16,    # the offset from max height to standing natural height
        "base_born_height_offset" : 0.001,       # the offset from max height to standing natural height



        # The order of the actuators should be the same as they have been defined in the xml file.
        "actuator_names" :      ["FL_HipX_actuator", "FL_HipY_actuator", "FL_Knee_actuator",
                                "FR_HipX_actuator", "FR_HipY_actuator", "FR_Knee_actuator",
                                "HL_HipX_actuator", "HL_HipY_actuator", "HL_Knee_actuator",
                                "HR_HipX_actuator", "HR_HipY_actuator", "HR_Knee_actuator"],

        "actuator_type" :        "position",  # "torque" or "position"
        "kps" :                  [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
        "kds" :                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        
        # ========== 迁移自Lite3_rl_deploy的参数 ==========
        # 已修改：Kd值从0.7改为1.0，以匹配原始实现
        # 注意: 原始实现中kd=1.0，现在已与原始实现保持一致
        
        # 观测缩放参数 (迁移自lite3_test_policy_runner_onnx.hpp)
        "omega_scale" :          0.25,      # 角速度缩放 (base_omega * omega_scale)
        "dof_vel_scale" :        0.05,      # 关节速度缩放 (dof_vel * dof_vel_scale)
        "max_cmd_vel" :          [0.8, 0.8, 0.8],  # 最大命令速度 [lin_x, lin_y, ang_yaw]
        
        # 默认关节位置 (用于观测计算，相对默认位置的偏移)
        # 注意: 原始实现中有两个版本:
        # - URDF_INIT: [0, -1.35453, 2.54948] * 4
        # - dof_pos_default_policy: [0.0, -0.8, 1.6] * 4
        # 这里使用策略中的默认值，与neutral_joint_angles不同
        "dof_pos_default_policy" : {
            "FL_HipX_joint": 0.0, "FL_HipY_joint": -0.8, "FL_Knee_joint": 1.6,
            "FR_HipX_joint": 0.0, "FR_HipY_joint": -0.8, "FR_Knee_joint": 1.6,
            "HL_HipX_joint": 0.0, "HL_HipY_joint": -0.8, "HL_Knee_joint": 1.6,
            "HR_HipX_joint": 0.0, "HR_HipY_joint": -0.8, "HR_Knee_joint": 1.6,
        },
        
        # 动作缩放 (迁移自action_scale_robot)
        # 注意: 原始实现中的action_scale与OrcaGym中的action_scale不同
        # 原始: [0.125, 0.25, 0.25] * 4
        # OrcaGym: [0.25] * 12
        # 如果需要完全匹配，可以使用原始值
        "action_scale_original" : [0.125, 0.25, 0.25,  # FL: HipX, HipY, Knee
                                   0.125, 0.25, 0.25,  # FR
                                   0.125, 0.25, 0.25,  # HL
                                   0.125, 0.25, 0.25], # HR
        "use_original_action_scale" : False,  # 是否使用原始动作缩放

        "action_scale" :         [
            0.25,    # joint name="FL_HipX_joint" joint axis="-1 0 0" range="-0.523 0.523", neutral=0.0
            0.25,    # joint name="FL_HipY_joint" joint axis="0 -1 0" range="-2.67 0.314", neutral=-1.0
            0.25,    # joint name="FL_Knee_joint" joint axis="0 -1 0" range="0.524 2.792", neutral=1.8

            0.25,    # joint name="FR_HipX_joint" joint axis="-1 0 0" range="-0.523 0.523", neutral=0.0
            0.25,    # joint name="FR_HipY_joint" joint axis="0 -1 0" range="-2.67 0.314", neutral=-1.0
            0.25,    # joint name="FR_Knee_joint" joint axis="0 -1 0" range="0.524 2.792", neutral=1.8

            0.25,    # joint name="HL_HipX_joint" joint axis="-1 0 0" range="-0.523 0.523", neutral=0.0
            0.25,    # joint name="HL_HipY_joint" joint axis="0 -1 0" range="-2.67 0.314", neutral=-1.0
            0.25,    # joint name="HL_Knee_joint" joint axis="0 -1 0" range="0.524 2.792", neutral=1.8

            0.25,    # joint name="HR_HipX_joint" joint axis="-1 0 0" range="-0.523 0.523", neutral=0.0
            0.25,    # joint name="HR_HipY_joint" joint axis="0 -1 0" range="-2.67 0.314", neutral=-1.0
            0.25,    # joint name="HR_Knee_joint" joint axis="0 -1 0" range="0.524 2.792", neutral=1.8
        ],
        "soft_joint_qpos_limit": 0.9,       # percentage of urdf limits, values above this limit are penalized
        "soft_joint_qvel_limit": 1.0,       # percentage of urdf limits, values above this limit are penalized
        "soft_torque_limit": 1.0,           # percentage of urdf limits, values above this limit are penalized

        # From Lite3 URDF @ https://github.com/DeepRoboticsLab/URDF_model/tree/c81de1e90f40662bf2dcb3c1e7c6fb0a9d4be92b
        "joint_qvel_range" : [26.2, 26.2, 17.3, 26.2, 26.2, 17.3, 26.2, 26.2, 17.3, 26.2, 26.2, 17.3],
        
        "imu_site_name" :       "imu",
        "contact_site_names" :  ["FL_site", "FR_site", "HL_site", "HR_site"],
        
        "sensor_imu_framequat_name" :           "imu_quat",
        "sensor_imu_gyro_name" :                "imu_omega",
        "sensor_imu_accelerometer_name" :       "imu_acc",
        "sensor_foot_touch_names" : ["FL_touch", "FR_touch", "HL_touch", "HR_touch"],   # Maintain the same order as contact_site_names
        "use_imu_sensor" : False,

        "compute_body_height" : False,      
        "observe_body_height" : False,       # 真机没有激光雷达，无法计算body高度，因此这里高度只用来做奖励，不用来观测
        "compute_body_orientation" : False,  # TODO:目前只支持水平方向的orientation奖励
        "compute_foot_height" : True,        # Foot 高度只用来做奖励，不用来观测

        "base_contact_body_names" : ["torso", "FL_HIP", "FR_HIP", "HL_HIP", "HR_HIP"],
        "leg_contact_body_names" : ["FL_THIGH", "FL_SHANK", 
                                    "FR_THIGH", "FR_SHANK", 
                                    "HL_THIGH", "HL_SHANK", 
                                    "HR_THIGH", "HR_SHANK"],
        "foot_body_names" : ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"],  # Maintain the same order as contact_site_names
        "foot_fitted_ground_pairs" : [[0, 3], [1, 2]],                   # For the quadruped robot, the left front and right rear feet should touch the ground at the same time.

        # Config for reward
        "reward_coeff" : RewardConfig,



        # Robot's Self-Weight: Approximately 149.2 Newtons.
        # Static Foot Forces: Front feet ~44.8 N each; rear feet ~29.8 N each.
        # Dynamic Foot Forces (Walking at 1 m/s): Approximately 89.5 Newtons per foot during ground contact.
        # Front vs. Rear Legs: Front legs bear more force due to weight distribution and dynamics. 
        "foot_touch_force_threshold" : 100.0,
        "foot_touch_force_air_threshold" : 0.01,
        "foot_touch_force_step_threshold" : 0.01,
        "foot_touch_air_time_ideal" : 0.2,  # Go2 robot standing height is 0.4m. The ideal median stride rate for a Trot is around 0.4 seconds
        "foot_square_wave" : {
            "p5" :          0.5,
            "phase_freq" :  0.8,
            "eps" :         0.2,
        },
        "foot_leg_period" : {
            "period" : 0.8,
            "offset" : 0.5,
            "stance_threshold" : 0.55,
            "swing_height" : 0.03,
        },

        # Config for randomization
        "randomize_friction" :      True,
        "friction_range" :          [0.75, 1.25],
        "randomize_base_mass" :     True,
        "added_mass_range" :        [-0.5, 1.5],
        "added_mass_pos_range" :    [-0.05, 0.05],
        "push_robots" :             True,
        "push_interval_s" :         15,
        "max_push_vel_xy" :         1.0,
        "pos_random_range" :        0.5,    # randomize the x,y position of the robot in each episode
        
        # Config for ccurriculum learning
        "curriculum_learning" :     True,

        # Config for logging
        "log_agent_names" : ["Lite3_000"],
        
        # Config for visualization
        "visualize_command_agent_names" : ["Lite3_000"],
        "command_indicator_name" : "command_indicator_mocap",

        # Config for playable agent
        "playable_agent_name" : "Lite3_000",
        
        # Config for nn network training
        "pi" : [512, 256, 128],  # 策略网络结构
        "vf" : [512, 256, 128],   # 值函数网络结构
        "n_steps" : 64,  # 每个环境采样步数
        "batch_size" : 4096,  # 批次大小            
        "learning_rate": {
            "initial_value": 3e-4,    # 初始学习率
            "end_fraction": 0.8,      # 线性衰减，0.1表示在训练结束前10%时达到最终学习率
            "final_value": 1e-4       # 最终学习率
        },
        "gamma" : 0.99,  # 折扣因子
        "clip_range": 0.2,  # 裁剪范围
        "ent_coef" : 0.01,  # 固定熵系数
        "max_grad_norm" : 1,  # 最大梯度范数

        # Config for rllib
        "fcnet_hiddens": [512, 256, 128],
        "rollout_fragment_length" : 4,  # 每个rollout的片段长度(rllib)
        "train_batch_size" : 32768,  # 训练批次大小  (rllib)
        "circular_buffer_num_batches" : 4,
        "circular_buffer_iterations_per_batch" : 2,
        "minibatch_size" : 4096,    # 批次大小  (rllib)
        "lr_schedule": {
            "initial_value": 3e-4,    # 初始学习率
            "end_fraction": 0.2,      # 线性衰减，0.1表示在训练结束前10%时达到最终学习率
            "final_value": 1e-4       # 最终学习率
        },
        "ent_coef_schedule" : {   # 线性衰减熵系数(rllib)
            "initial_value": 0.01,
            "end_fraction": 0.2,
            "final_value": 0.005,
        },
        "clip_param": 0.1, # 裁剪参数(rllib)
        "grad_clip" : 1.0, # 梯度裁剪(rllib)
        "grad_clip_by" : "global_norm", # 梯度裁剪方式(rllib)
        "vf_loss_coeff" : 0.5,
        "vf_share_layers" : False,
        "use_lstm" : False,
        "use_kl_loss" : True,
        "lstm_cell_size" : 256,
        "max_seq_len" : 64,
        "conv_filters" : None,
        "free_log_std" : False,
    }