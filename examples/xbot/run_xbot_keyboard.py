#!/usr/bin/env python3
"""
XBot键盘控制 - 使用WASD控制机器人移动
基于run_xbot_orca.py，添加键盘控制功能
"""

from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from envs.xbot_gym.xbot_simple_env import XBotSimpleEnv
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
import torch
import numpy as np
import time


class XBotKeyboardController:
    """
    XBot键盘控制器
    
    按键映射:
        W/S - 前进/后退
        A/D - 左转/右转
        Q/E - 左平移/右平移
        LShift - 加速（Turbo模式）
        Space - 停止
        R - 重置环境
        Esc - 退出
    """
    
    def __init__(self, orcagym_addr: str):
        self.keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)
        
        # 速度参数
        self.base_forward_speed = 0.5  # 基础前进速度
        self.base_backward_speed = -0.2  # 基础后退速度
        self.base_strafe_speed = 0.2     # 基础平移速度
        self.base_turn_speed = 0.3       # 基础转向速度
        self.turbo_scale = 2           # Turbo模式加速倍数
        
        # 上一次的按键状态
        self.last_key_state = {}
        
        print("\n⌨️  键盘控制说明:")
        print("  W - 前进")
        print("  S - 后退")
        print("  A - 左转")
        print("  D - 右转")
        print("  Q - 左平移")
        print("  E - 右平移")
        print("  LShift - 加速（Turbo）")
        print("  Space - 停止")
        print("  R - 重置环境")
        print("  Esc - 退出")
        print()
    
    def get_command(self):
        """
        根据键盘状态计算命令速度
        
        返回:
            (vx, vy, dyaw, reset, stop)
        """
        self.keyboard.update()
        key_state = self.keyboard.get_state()
        
        vx = 0.0
        vy = 0.0
        dyaw = 0.0
        reset_flag = False
        stop_flag = False
        
        # W - 前进
        if key_state["W"] == 1:
            vx = self.base_forward_speed
        
        # S - 后退
        if key_state["S"] == 1:
            vx = self.base_backward_speed
        
        # Q - 左平移
        if key_state["Q"] == 1:
            vy = self.base_strafe_speed
        
        # E - 右平移
        if key_state["E"] == 1:
            vy = -self.base_strafe_speed
        
        # A - 左转
        if key_state["A"] == 1:
            dyaw = self.base_turn_speed
        
        # D - 右转
        if key_state["D"] == 1:
            dyaw = -self.base_turn_speed
        
        # LShift - Turbo加速
        if key_state["LShift"] == 1:
            vx *= self.turbo_scale
            vy *= self.turbo_scale
        
        # Space - 停止
        if key_state["Space"] == 1:
            vx = 0.0
            vy = 0.0
            dyaw = 0.0
            stop_flag = True
        
        # R - 重置（检测按下边沿）
        if self.last_key_state.get("R", 0) == 0 and key_state["R"] == 1:
            reset_flag = True
        
        # 保存当前按键状态
        self.last_key_state = key_state.copy()
        
        return vx, vy, dyaw, reset_flag, stop_flag
    
    def close(self):
        """关闭键盘控制器"""
        # KeyboardInput的close方法已在类内部实现
        pass


def main():
    print("="*80)
    print("🎮 XBot键盘控制 - OrcaGym")
    print("="*80)
    
    # 环境配置
    orcagym_addr = "localhost:50051"
    config = {
        "frame_skip": 10,
        "orcagym_addr": orcagym_addr,
        "agent_names": ["XBot-L"],
        "time_step": 0.001,
        "max_episode_steps": 10000,
        "render_mode": "human",
    }

    TIME_STEP = config['time_step']
    FRAME_SKIP = config['frame_skip']
    REALTIME_STEP = TIME_STEP * FRAME_SKIP
    
    print(f"\n⚙️  环境配置:")
    print(f"  - OrcaGym地址: {orcagym_addr}")
    print(f"  - 物理步长: {config['time_step']}s (1000Hz)")
    print(f"  - 策略频率: 100Hz")
    
    # 创建环境
    print("\n📦 创建环境...")
    env = XBotSimpleEnv(**config)
    
    # 加载策略
    policy_path = "/home/orca/OrcaWorkStation/humanoid-gym/logs/XBot_ppo/exported/policies/policy_example.pt"
    
    print(f"\n📦 加载策略: {policy_path}")
    try:
        policy = torch.jit.load(policy_path)
        policy.eval()
        print("✅ 策略加载成功")
    except Exception as e:
        print(f"❌ 策略加载失败: {e}")
        env.close()
        return
    
    # 创建键盘控制器
    print("\n🎮 初始化键盘控制器...")
    keyboard_controller = XBotKeyboardController(orcagym_addr)
    
    print("\n" + "="*80)
    print("🚀 开始运行...")
    print("="*80)
    print("\n提示: 按ESC退出，按R重置环境\n")
    
    # Reset
    obs, info = env.reset()
    
    episode_reward = 0.0
    episode_steps = 0
    total_episodes = 0
    
    try:
        while True:
            start_time = datetime.now()
            # 获取键盘命令
            vx, vy, dyaw, reset_flag, stop_flag = keyboard_controller.get_command()
            
            # 更新环境的命令速度
            env.set_command(vx, vy, dyaw)
            
            # 检查ESC退出
            key_state = keyboard_controller.keyboard.get_state()
            if key_state["Esc"] == 1:
                print("\n⚠️  用户按下ESC，退出程序")
                break
            
            # 检查重置
            if reset_flag:
                print(f"\n🔄 重置环境 (Episode {total_episodes}: {episode_steps}步, 奖励={episode_reward:.2f})")
                obs, info = env.reset()
                episode_reward = 0.0
                episode_steps = 0
                total_episodes += 1
                continue
            
            # 获取策略动作
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float()
                action = policy(obs_tensor).numpy()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            episode_reward += reward
            episode_steps += 1
            
            # 显示当前状态
            if episode_steps % 100 == 0:
                status = "🛑 停止" if stop_flag else f"➡️  vx={vx:.2f}, vy={vy:.2f}, dyaw={dyaw:.2f}"
                print(f"[Step {episode_steps:4d}] {status} | Reward: {episode_reward:.2f}")
            
            # ⚠️ 禁用自动重置 - 只在检测到摔倒或超时时提示，不自动reset
            if terminated or truncated:
                total_episodes += 1
                print(f"\n⚠️  检测到异常状态 (Episode {total_episodes}):")
                print(f"  - 步数: {episode_steps}")
                print(f"  - 奖励: {episode_reward:.2f}")
                print(f"  - 原因: {'摔倒' if terminated else '超时'}")
                print(f"  ℹ️  机器人将继续运行，按R键手动重置")
                print()
                
                # ⭐ 不自动重置，继续运行
                # obs, info = env.reset()
                # episode_reward = 0.0
                # episode_steps = 0
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
    
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    keyboard_controller.close()
    env.close()
    
    print("\n" + "="*80)
    print("✅ 程序结束")
    print("="*80)
    print(f"总Episodes: {total_episodes}")
    print()


if __name__ == '__main__':
    main()

