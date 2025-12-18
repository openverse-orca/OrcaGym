"""
双臂机器人操作示例 - 支持灵活配置

用法示例：
1. 使用默认配置（根据机器人名称自动推断）：
   python run_dual_arm_sim_with_config.py --agent_names openloong_gripper_2f85_fix_base_usda

2. 显式指定配置：
   python run_dual_arm_sim_with_config.py --agent_names openloong_gripper_2f85_fix_base_usda --robot_config openloong

3. 多个机器人使用不同配置：
   python run_dual_arm_sim_with_config.py --agent_names "robot1 robot2" --robot_configs "robot1:openloong,robot2:d12"

4. 列出所有可用配置：
   python run_dual_arm_sim_with_config.py --list_configs
"""

import os
import sys

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

import orca_gym.scripts.dual_arm_manipulation as dual_arm_manipulation
from envs.manipulation.robots.configs.robot_config_registry import list_available_configs

import argparse


def parse_robot_configs(robot_configs_str: str) -> dict:
    """
    解析机器人配置字符串
    
    Args:
        robot_configs_str: 格式为 "robot1:config1,robot2:config2"
        
    Returns:
        配置字典 {robot_name: config_name}
    """
    if not robot_configs_str:
        return {}
    
    configs = {}
    for pair in robot_configs_str.split(','):
        if ':' in pair:
            robot_name, config_name = pair.split(':', 1)
            configs[robot_name.strip()] = config_name.strip()
    
    return configs


if __name__ == "__main__":

    # 如果要使用多Pico设备，需要设置adb反向端口转发
    # adb device 命令查看设备序列号

    # PICO_01:8001 -----> PC:8001
    # adb -s <device_serial_number_01> reverse tcp:8001 tcp:8001

    # PICO_02:8001 -----> PC:8002
    # adb -s <device_serial_number_02> reverse tcp:8001 tcp:8002  
    
    # 脚本参数增加--pico_ports参数，指定多个端口
    # 例如：--pico_ports "8001 8002"

    parser = argparse.ArgumentParser(
        description='运行双臂机器人操作示例，支持灵活配置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # ==================== 配置相关参数 ====================
    parser.add_argument('--list_configs', action='store_true', 
                        help='列出所有可用的机器人配置并退出')
    parser.add_argument('--robot_config', type=str, default=None,
                        help='指定所有机器人使用的配置名称（例如：openloong, d12）')
    parser.add_argument('--robot_configs', type=str, default=None,
                        help='为不同机器人指定不同配置，格式：robot1:config1,robot2:config2')
    
    # ==================== 基础参数 ====================
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', 
                        help='gRPC 服务器地址')
    parser.add_argument('--agent_names', type=str, default='openloong_gripper_2f85_fix_base_usda', 
                        help='机器人名称，多个机器人用空格分隔')
    parser.add_argument('--pico_ports', type=str, default='8001', 
                        help='Pico 服务器端口，多个端口用空格分隔')
    
    # ==================== 运行模式参数 ====================
    parser.add_argument('--run_mode', type=str, default='teleoperation', 
                        help='运行模式 (teleoperation / playback / imitation / rollout / augmentation)')
    parser.add_argument('--action_type', type=str, 
                        help='动作类型 (end_effector_ik / end_effector_osc / joint_pos / joint_motor)')
    parser.add_argument('--action_step', type=int, default=1, 
                        help='每个动作的仿真步数。end_effector 建议 5，joint_pos 建议 1')
    
    # ==================== 任务配置参数 ====================
    parser.add_argument('--task_config', type=str, 
                        help='任务配置文件路径')
    parser.add_argument('--level', type=str, 
                        help='存储级别或场景名称（例如：default, experiment, debug）')
    
    # ==================== 数据集参数 ====================
    parser.add_argument('--dataset', type=str, 
                        help='保存/加载数据集的文件路径')
    parser.add_argument('--record_length', type=int, default=1200, 
                        help='每个 episode 录制的时长（秒）')
    parser.add_argument('--withvideo', type=str, default='True',  
                        help='是否输出视频数据 (True / False)')
    parser.add_argument('--sync_codec', type=str, default='False',  
                        help='是否等待下一帧准备好再发送下一个命令 (True / False)')
    
    # ==================== 训练参数 ====================
    parser.add_argument('--algo', type=str, default='bc', 
                        help='训练策略使用的算法')
    parser.add_argument('--model_file', type=str, 
                        help='加载模型文件的路径（用于 rollout）')
    
    # ==================== 控制参数 ====================
    parser.add_argument('--ctrl_device', type=str, default='vr', 
                        help='控制设备类型')
    parser.add_argument('--sample_range', type=float, default=0.0, 
                        help='采样物体和目标位置的区域范围')
    
    # ==================== 回放参数 ====================
    parser.add_argument('--playback_mode', type=str, default='random', 
                        help='回放模式 (loop 或 random)')
    parser.add_argument('--rollout_times', type=int, default=10, 
                        help='rollout 测试次数')
    parser.add_argument('--realtime_playback', type=str, default='True', 
                        help='是否启用实时回放或 rollout (True / False)')
    
    # ==================== 数据增强参数 ====================
    parser.add_argument('--augmented_noise', type=float, default=0.005, 
                        help='数据增强的噪声强度')
    parser.add_argument('--augmented_rounds', type=int, default=10, 
                        help='数据增强的轮数')
    
    # ==================== 遥操作参数 ====================
    parser.add_argument('--teleoperation_rounds', type=int, default=100, 
                        help='遥操作的轮数')
    
    args = parser.parse_args()
    
    # ==================== 处理配置列表请求 ====================
    if args.list_configs:
        print("=" * 60)
        print("可用的机器人配置：")
        print("=" * 60)
        configs = list_available_configs()
        for i, config in enumerate(configs, 1):
            print(f"  {i}. {config}")
        print("\n" + "=" * 60)
        print("使用方法：")
        print("=" * 60)
        print("  单个配置: --robot_config openloong")
        print("  多个配置: --robot_configs 'robot1:openloong,robot2:d12'")
        print("\n示例命令：")
        print("  python run_dual_arm_sim_with_config.py \\")
        print("      --agent_names openloong_gripper_2f85_fix_base_usda \\")
        print("      --robot_config openloong \\")
        print("      --run_mode teleoperation")
        print("=" * 60)
        sys.exit(0)
    
    # ==================== 解析和设置机器人配置 ====================
    robot_configs = {}
    
    if args.robot_configs:
        # 为不同机器人指定不同配置
        robot_configs = parse_robot_configs(args.robot_configs)
        print(f"使用自定义机器人配置: {robot_configs}")
    elif args.robot_config:
        # 所有机器人使用相同配置
        # 使用特殊键 "__all__" 表示所有机器人都使用这个配置
        # 这样即使 agent_names 不匹配也能正确应用配置
        robot_configs = {"__all__": args.robot_config}
        print(f"所有机器人使用配置: {args.robot_config}")
        print(f"配置映射: {robot_configs}")
        print(f"Agent names from args: {args.agent_names}")
    else:
        print("未指定配置，将根据机器人名称自动推断配置")
    
    # 将配置添加到 args 中
    args.robot_configs_dict = robot_configs
    print(f"传递给 run_dual_arm_sim 的 robot_configs: {robot_configs}")
    
    # ==================== 运行主程序 ====================
    dual_arm_manipulation.run_dual_arm_sim(args, project_root, current_file_path)

