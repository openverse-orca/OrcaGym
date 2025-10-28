#!/usr/bin/env python3
"""
XBot运行脚本 - 完全基于OrcaGym框架
使用envs/xbot_gym/xbot_simple_env.py环境
"""

from datetime import datetime
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from envs.xbot_gym.xbot_simple_env import XBotSimpleEnv
import torch
import numpy as np
import math

def print_detailed_diagnostics(step, obs, action, env):
    """
    ⭐ 详细诊断输出 - 参考standaloneMujoco的调试格式
    """
    print(f"\n{'='*80}")
    print(f"🔍 详细诊断 [Step={step}, Policy Update={step//10}, Time={step*0.001:.2f}s]")
    print(f"{'='*80}")
    
    # 解析观测空间（47维）
    phase_sin, phase_cos = obs[0], obs[1]
    phase = math.atan2(phase_sin, phase_cos) / (2 * math.pi)
    if phase < 0:
        phase += 1.0
    
    cmd_vx = obs[2] / 2.0   # 恢复原始命令
    cmd_vy = obs[3] / 2.0
    cmd_dyaw = obs[4] / 1.0
    
    q_obs = obs[5:17]         # 关节位置偏差
    dq_obs = obs[17:29] / 0.05  # 关节速度（恢复）
    last_action = obs[29:41]  # 上一次动作
    omega = obs[41:44]        # 角速度
    euler = obs[44:47]        # 欧拉角
    
    print(f"\n📊 观测空间 (47维):")
    print(f"  - Gait Phase: {phase:.3f} (sin={phase_sin:.3f}, cos={phase_cos:.3f})")
    print(f"  - Commands: vx={cmd_vx:.2f}, vy={cmd_vy:.2f}, dyaw={cmd_dyaw:.2f}")
    print(f"  - Joint Pos: range=[{q_obs.min():.3f}, {q_obs.max():.3f}], mean={q_obs.mean():.3f}")
    print(f"  - Joint Vel: range=[{dq_obs.min():.2f}, {dq_obs.max():.2f}], mean={dq_obs.mean():.2f}")
    print(f"  - Last Action: range=[{last_action.min():.3f}, {last_action.max():.3f}], mean={last_action.mean():.3f}")
    print(f"  - Angular Vel: [{omega[0]:.2f}, {omega[1]:.2f}, {omega[2]:.2f}]")
    print(f"  - Euler: [{np.rad2deg(euler[0]):.1f}°, {np.rad2deg(euler[1]):.1f}°, {np.rad2deg(euler[2]):.1f}°]")
    
    print(f"\n🎮 动作输出 (12维):")
    print(f"  - Action: range=[{action.min():.3f}, {action.max():.3f}], mean={action.mean():.3f}")
    print(f"  - Action norm: {np.linalg.norm(action):.3f}")
    
    # PD控制信息（从环境获取）
    if hasattr(env, 'last_tau'):
        tau = env.last_tau
        print(f"\n⚙️  PD控制:")
        print(f"  - Target q: range=[{(env.action_scale * action).min():.3f}, {(env.action_scale * action).max():.3f}]")
        print(f"  - Torque τ: range=[{tau.min():.1f}, {tau.max():.1f}] N·m, max_abs={np.abs(tau).max():.1f}")
        print(f"  - Torque usage: {np.abs(tau).max()/env.tau_limit*100:.1f}% of limit")
    
    # Base状态
    if hasattr(env, 'last_base_pos'):
        base_pos = env.last_base_pos
        print(f"\n🤖 Base状态:")
        print(f"  - Position: ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})m")
        print(f"  - RPY: ({np.rad2deg(euler[0]):.2f}°, {np.rad2deg(euler[1]):.2f}°, {np.rad2deg(euler[2]):.2f}°)")
    
    print(f"{'='*80}")


def main():
    print("="*80)
    print("🚀 XBot运行测试 - OrcaGym框架（增强诊断版）")
    print("="*80)
    
    # 关键配置 - 匹配humanoid-gym
    config = {
        "frame_skip": 10,              # 单次物理步
        "orcagym_addr": "localhost:50051",
        "agent_names": ["XBot-L"],
        "time_step": 0.001,           # ⚠️ 1ms物理步长
        "max_episode_steps": 10000,
        "render_mode": "human",       # 可视化
    }

    TIME_STEP = config['time_step']
    FRAME_SKIP = config['frame_skip']
    REALTIME_STEP = TIME_STEP * FRAME_SKIP
    
    # ⭐ 命令速度配置（可调节参数）
    # 测试结果: 0.4 m/s 是最优速度（262步），降速反而性能下降
    # 速度选项:
    #   - 0.4 m/s: 262步 ✅ 最佳性能
    #   - 0.2 m/s: 232步 ⚠️ 略有下降
    #   - 0.15 m/s: 150步 ❌ 性能差
    CMD_VX = 0.0   # 前向速度（保持0.4 m/s最优）
    CMD_VY = 0.0   # 侧向速度
    CMD_DYAW = 0.0 # 转向速度
    
    print(f"\n⚙️  仿真配置:")
    print(f"  - 物理步长: {config['time_step']}s (1000Hz)")
    print(f"  - Decimation: 10 (在环境内部实现)")
    print(f"  - 策略频率: 100Hz")
    print(f"\n🎯 命令速度 (参考standaloneMujoco):")
    print(f"  - vx: {CMD_VX} m/s")
    print(f"  - vy: {CMD_VY} m/s")
    print(f"  - dyaw: {CMD_DYAW} rad/s")
    
    # 创建环境
    print("\n📦 创建环境...")
    env = XBotSimpleEnv(**config)
    
    # ⭐ 设置命令速度
    env.cmd_vx = CMD_VX
    env.cmd_vy = CMD_VY
    env.cmd_dyaw = CMD_DYAW
    
    print(f"✓ 环境创建成功")
    print(f"  - 观测空间: {env.observation_space.shape}")
    print(f"  - 动作空间: {env.action_space.shape}")
    print(f"  - 命令速度已设置: vx={env.cmd_vx}, vy={env.cmd_vy}, dyaw={env.cmd_dyaw}")
    
    # 加载策略 - 使用相对路径
    # 目录结构: OrcaWorkStation/OrcaGym/examples/xbot/ 和 OrcaWorkStation/humanoid-gym/
    policy_path = "../../../humanoid-gym/logs/XBot_ppo/exported/policies/policy_example.pt"
    
    print(f"\n📦 加载策略: {policy_path}")
    try:
        policy = torch.jit.load(policy_path)
        policy.eval()
        print(f"✓ 策略加载成功")
        use_policy = True
    except Exception as e:
        print(f"\n⚠️  无法加载策略: {e}")
        print("   使用零动作测试")
        use_policy = False
    
    # 运行
    print("\n" + "="*80)
    print("🚀 开始运行...")
    print("="*80)
    print("\n提示:")
    print("  - Pitch应该保持<20°，高度应该在0.85-0.95m")
    print("  - 每100步打印详细诊断信息")
    print("  - 参考standaloneMujoco: Pitch±1.5°，速度0.4m/s\n")
    
    obs, info = env.reset()
    
    episode_reward = 0.0
    episode_steps = 0
    max_steps = 2000  # 测试2000步
    
    # ⭐ 诊断间隔
    DIAGNOSTIC_INTERVAL = 100  # 每100步打印一次详细诊断
    
    try:
        while True:
            # 获取action
            start_time = datetime.now()
            if use_policy:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float()
                    action = policy(obs_tensor).numpy()
            else:
                # 零动作（站立测试）
                action = np.zeros(12, dtype=np.float32)
            
            # ⭐ 每100步打印详细诊断（在step之前，观察输入）
            # if step > 0 and step % DIAGNOSTIC_INTERVAL == 0:
            #     print_detailed_diagnostics(step, obs, action, env)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # 渲染
            env.render()
            
            # Episode结束
            if terminated or truncated:
                print(f"\n{'='*80}")
                print(f"❌ Episode结束")
                print(f"{'='*80}")
                print(f"  - Steps: {episode_steps}")
                print(f"  - Reward: {episode_reward:.2f}")
                if 'fall_reason' in info and info['fall_reason']:
                    print(f"  - 原因: {info['fall_reason']}")
                print(f"{'='*80}\n")
                
                # 打印最后的诊断信息
                # print_detailed_diagnostics(episode_steps, obs, action, env)
                
                obs, info = env.reset()
                episode_reward = 0.0
                episode_steps = 0

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
        # print(f"\n{'='*80}")
        # print(f"✅ 测试完成！运行了{max_steps}步")
        # print(f"{'='*80}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  运行被中断")
    
    finally:
        env.close()
        print("\n环境已关闭")

if __name__ == "__main__":
    main()

