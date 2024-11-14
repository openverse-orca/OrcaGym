import os
import sys
import time

import numpy as np

current_file_path = os.path.abspath('./../../')

if current_file_path not in sys.path:
    print("add path: ", current_file_path)
    sys.path.append(current_file_path)


import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime

# 
TIME_STEP = 0.01

def slerp(q1, q2, t):
    dot_product = np.dot(q1, q2)
    # 如果点积小于 0，反转一个四元数以取最短路径
    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product
    
    # 如果点积接近1，直接返回线性插值（防止浮点误差）
    if dot_product > 0.9995:
        return (1.0 - t) * q1 + t * q2
    
    # 计算角度 Omega
    theta_0 = np.arccos(dot_product)  # 初始角度
    sin_theta_0 = np.sin(theta_0)
    
    # 计算插值角度
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    # 计算权重
    s1 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    return s1 * q1 + s2 * q2

def generate_xpos_xquat():
    # 起点和终点坐标
    P1 = np.array([0.43226111, -0.06559586, 1.74047182])
    P2 = np.array([0.31422432, -0.51640537, 2.08510489])

    # 四元数，形式为(w, x, y, z)
    xquat1 = np.array([0.84184204, 0.42076756, 0.33788169, 0.00962368])
    xquat2 = np.array([0.66645221, 0.7345639, -0.02884642, -0.12419826])

    increasing = True  # 初始化为递增模式
    t = 0  # 初始化 t 值

    # 定义闭包函数
    def next_xpos_xquat():
        nonlocal t, increasing

        # 使用线性插值计算当前位置
        xpos = (1 - t) * P1 + t * P2
        # 使用 SLERP 进行四元数插值，保持四元数顺序为(w, x, y, z)
        xquat = slerp(xquat1, xquat2, t)

        # 更新 t 值
        if increasing:
            t += 0.01  # 每次递增 0.01
            if t >= 1:
                t = 1
                increasing = False  # 当达到终点时，切换为递减
        else:
            t -= 0.01  # 每次递减 0.01
            if t <= 0:
                t = 0
                increasing = True  # 当达到起点时，切换为递增

        return xpos, xquat  # 返回当前的插值位置和插值后的四元数

    return next_xpos_xquat  # 返回闭包函数

def register_env(orcagym_addr, env_name, env_index, control_freq=20) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point="envs.openloong.Openloong_arm_date_env:OpenloongArmEnv",
        kwargs={'frame_skip': 1,   
                'reward_type': "dense",
                'orcagym_addr': orcagym_addr, 
                'agent_names': ['AzureLoong'], 
                'time_step': TIME_STEP,
                'control_freq': control_freq},
        max_episode_steps=sys.maxsize,
        reward_threshold=0.0,
    )
    return env_id

def continue_training(env):
    observation, info = env.reset(seed=42)
    action = generate_xpos_xquat()

    while True:
        start_time = datetime.now()

        #action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action())

        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() < TIME_STEP:
            time.sleep(TIME_STEP - elapsed_time.total_seconds())

    

if __name__ == "__main__":
    """
    OSC运动算法控制青龙机器人机械臂的示例
    """
    try:
        orcagym_addr = "localhost:50051"
        print("simulation running... , orcagym_addr: ", orcagym_addr)
        
        env_name = "DateCtrl-v0"
        env_index = 0
        env_id = register_env(orcagym_addr, env_name, env_index, 20)
        print("Registering environment with id: ", env_id)

        env = gym.make(env_id)        
        print("Starting simulation...")

        continue_training(env)
    except KeyboardInterrupt:
        print("Simulation stopped")        
        env.close()