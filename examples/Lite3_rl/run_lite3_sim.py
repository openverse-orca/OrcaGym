"""
Lite3 ONNX策略仿真运行脚本
仿照legged_gym的仿真模式，在OrcaGym中运行Lite3的ONNX策略

使用方法:
    python run_lite3_sim.py --config configs/lite3_onnx_sim_config.yaml --remote localhost:50051
"""

import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime
import gymnasium as gym
import yaml

# 添加项目根目录到Python路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加legged_gym目录到路径，以便导入scripts模块
legged_gym_dir = os.path.join(project_root, "examples", "legged_gym")
if legged_gym_dir not in sys.path:
    sys.path.insert(0, legged_gym_dir)

from scripts.scene_util import clear_scene, publish_terrain, generate_height_map_file, publish_scene
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
from envs.legged_gym.legged_sim_env import LeggedSimEnv
from envs.legged_gym.legged_config import LeggedRobotConfig, LeggedObsConfig, CurriculumConfig, LeggedEnvConfig
from envs.legged_gym.utils.onnx_policy import load_onnx_policy
from envs.legged_gym.utils.lite3_obs_helper import compute_lite3_obs, get_dof_pos_default_policy
from envs.legged_gym.robot_config.Lite3_config import Lite3Config


# 仿真参数
TIME_STEP = 0.001                       # 1000 Hz for physics simulation
FRAME_SKIP = 5
ACTION_SKIP = 4

REALTIME_STEP = TIME_STEP * FRAME_SKIP * ACTION_SKIP  # 50 Hz for policy
CONTROL_FREQ = 1 / REALTIME_STEP        # 50 Hz for ppo policy

EPISODE_TIME_SHORT = LeggedEnvConfig["EPISODE_TIME_SHORT"]
MAX_EPISODE_STEPS = int(EPISODE_TIME_SHORT / REALTIME_STEP)  # 10 seconds


class KeyboardControl:
    """键盘控制类，用于控制机器人移动"""
    def __init__(self, orcagym_addr: str, env: LeggedSimEnv, command_model: dict):
        self.keyboard_controller = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)
        self._last_key_status = {"W": 0, "A": 0, "S": 0, "D": 0, "Q": 0, "E": 0, "Z": 0, "C": 0}
        self.env = env
        self.player_agent_lin_vel_x = {terrain_type: np.array(command_model[terrain_type]["forward_speed"]) 
                                       for terrain_type in command_model.keys()}
        self.player_agent_lin_vel_y = {terrain_type: np.array(command_model[terrain_type]["left_speed"]) 
                                       for terrain_type in command_model.keys()}
        self.player_agent_turn_angel = {terrain_type: np.array(command_model[terrain_type]["turn_speed"]) 
                                        for terrain_type in command_model.keys()}
        self.terrain_type = "flat_terrain"
        self.rl_control_mode = False  # False: 默认状态, True: RL控制状态
        
        # 命令速度缩放：降低后退速度，改善策略对负值命令的响应
        self.backward_speed_scale = 0.5  # 后退速度缩放（降低到50%，进一步改善稳定性）
        
        # 命令平滑：使用指数移动平均来平滑命令变化，减少突然变化导致的卡顿
        self.command_smoothing = True
        self.smoothing_alpha = 0.3  # 平滑系数（越小越平滑，0.3表示30%新值+70%旧值）
        self.last_lin_vel = np.zeros(3)
        self.last_ang_vel = 0.0

    def update(self):
        self.keyboard_controller.update()
        key_status = self.keyboard_controller.get_state()
        lin_vel = np.zeros(3)
        ang_vel = 0.0
        reborn = False
        enter_default_state = False
        enter_rl_control = False
        
        # z: 机器狗站立进入默认状态
        if self._last_key_status["Z"] == 0 and key_status["Z"] == 1:
            enter_default_state = True
            self.rl_control_mode = False
            print("进入默认状态（站立）")
        
        # c: 机器狗站立进入rl控制状态
        if self._last_key_status["C"] == 0 and key_status["C"] == 1:
            enter_rl_control = True
            self.rl_control_mode = True
            print("进入RL控制状态")
        
        # 只有在RL控制状态下才响应移动命令
        if self.rl_control_mode:
            # wasd: 前后左右移动
            # 参考legged_gym: Q键用[0]（左，负数），E键用[1]（右，正数）
            # left_speed配置: [-0.3, 0.3]，其中[0]是负数（左），[1]是正数（右）
            # 修复：再次反转左右控制，恢复A和S的原始速度使其更流畅
            if key_status["W"] == 1:  # 前
                lin_vel[0] = self.player_agent_lin_vel_x[self.terrain_type][1]  # 0.5 m/s，前进
            if key_status["S"] == 1:  # 后（进一步降低速度以改善稳定性）
                # 后退速度：降低到50%，避免策略对负值命令响应不佳导致卡住
                # -0.5 * 0.5 = -0.25 m/s，归一化后 = -0.5（而不是-1.0）
                lin_vel[0] = self.player_agent_lin_vel_x[self.terrain_type][0] * self.backward_speed_scale  # -0.25 m/s，后退
            if key_status["A"] == 1: 
                # 左移：使用left_speed[1] = 0.3 m/s，归一化后 = 1.0
                lin_vel[1] = self.player_agent_lin_vel_y[self.terrain_type][1]  # 0.3 m/s，左移
            if key_status["D"] == 1:  # 右（修复：移除速度缩放，使其与A键对称）
                # 右移：使用left_speed[0] = -0.3 m/s，归一化后 = -1.0（与A键对称）
                # 移除速度缩放，使左右移动完全对称
                lin_vel[1] = self.player_agent_lin_vel_y[self.terrain_type][0]  # -0.3 m/s，右移
            
            # qe: 顺逆时针旋转（参考legged_gym_env.py，A/D控制转向）
            # 但根据用户需求，Q/E控制旋转
            if key_status["Q"] == 1:  # 顺时针旋转（左转）
                ang_vel = self.player_agent_turn_angel[self.terrain_type]
            if key_status["E"] == 1:  # 逆时针旋转（右转）
                ang_vel = -self.player_agent_turn_angel[self.terrain_type]
            
            # 命令平滑处理（减少突然变化，改善稳定性，避免卡顿）
            if self.command_smoothing:
                # 指数移动平均：new = α*current + (1-α)*last
                # 平滑系数0.3表示：30%新值 + 70%旧值，使命令变化更平滑
                lin_vel = self.smoothing_alpha * lin_vel + (1 - self.smoothing_alpha) * self.last_lin_vel
                ang_vel = self.smoothing_alpha * ang_vel + (1 - self.smoothing_alpha) * self.last_ang_vel
            
            # 保存当前命令用于下次平滑
            self.last_lin_vel = lin_vel.copy()
            self.last_ang_vel = ang_vel
            
            # 调试：打印命令（仅在按下S键时）
            if key_status["S"] == 1:
                print(f"[DEBUG] 后退命令: lin_vel[0]={lin_vel[0]:.3f}, 归一化后={lin_vel[0]/0.5:.3f}, forward_speed配置={self.player_agent_lin_vel_x[self.terrain_type]}")
        else:
            # 非RL控制状态下，重置命令平滑的历史值
            if self.command_smoothing:
                self.last_lin_vel = np.zeros(3)
                self.last_ang_vel = 0.0

        self._last_key_status = key_status.copy()
        return lin_vel, ang_vel, reborn, self.terrain_type, enter_default_state, enter_rl_control, self.rl_control_mode


def register_env(orcagym_addr: str, 
                 env_name: str, 
                 env_index: int, 
                 agent_name: str,
                 ctrl_device: str,
                 max_episode_steps: int,
                 height_map: str,
                 ) -> str:
    """注册环境"""
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}_000"]
    print("Agent names: ", agent_names)
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'env_id': env_id,
                'agent_names': agent_names,
                'time_step': TIME_STEP,
                'action_skip': ACTION_SKIP,
                'max_episode_steps': max_episode_steps,
                'ctrl_device': ctrl_device,
                'control_freq': CONTROL_FREQ,
                'height_map': height_map,
                'robot_config': LeggedRobotConfig[agent_name],
                'legged_obs_config': LeggedObsConfig,
                'curriculum_config': CurriculumConfig,
                'legged_env_config': LeggedEnvConfig,
                }
    gym.register(
        id=env_id,
        entry_point='envs.legged_gym.legged_sim_env:LeggedSimEnv',
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def load_lite3_onnx_policy(model_path: str, device: str = "cpu"):
    """
    加载Lite3 ONNX策略
    
    Args:
        model_path: ONNX模型文件路径
        device: 设备类型 ('cpu' 或 'cuda')
    
    Returns:
        ONNXPolicy实例
    """
    # 检查设备可用性
    if device == "cuda":
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in available_providers:
                print(f"[WARNING] CUDAExecutionProvider not available. Available providers: {available_providers}")
                print(f"[WARNING] Falling back to CPU. Install onnxruntime-gpu to use GPU:")
                print(f"         pip install onnxruntime-gpu")
                device = "cpu"
            else:
                print(f"[INFO] Using GPU (CUDAExecutionProvider)")
        except ImportError:
            print(f"[WARNING] onnxruntime not installed. Falling back to CPU.")
            device = "cpu"
    
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(model_path):
        # 获取项目根目录（从当前脚本位置向上3级：Lite3_rl -> examples -> OrcaGym-dev）
        current_file_path = os.path.abspath(__file__)
        # __file__ 是 examples/Lite3_rl/run_lite3_sim.py
        # dirname 1: examples/Lite3_rl/
        # dirname 2: examples/
        # dirname 3: OrcaGym-dev/ (项目根目录)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        
        # 如果路径以 examples/ 开头，从项目根目录开始
        if model_path.startswith('examples/'):
            model_path = os.path.join(project_root, model_path)
        else:
            # 否则相对于当前脚本目录
            current_file_dir = os.path.dirname(current_file_path)
            model_path = os.path.join(current_file_dir, model_path)
    
    model_path = os.path.abspath(model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model file not found: {model_path}\n"
                               f"Current working directory: {os.getcwd()}\n"
                               f"Resolved path: {model_path}")
    
    print(f"Loading Lite3 ONNX policy from: {model_path}")
    print(f"Device: {device.upper()}")
    policy = load_onnx_policy(model_path, device=device)
    return policy


def compute_lite3_observation_from_env(env: LeggedSimEnv, agent_name: str):
    """
    从环境中提取数据并计算Lite3格式的45维观测
    
    Args:
        env: 仿真环境
        agent_name: agent名称
    
    Returns:
        obs: 45维观测向量
    """
    # 获取agent（LeggedRobot实例）
    agent_base = env._agents[agent_name]
    agent = agent_base.agent  # 获取LeggedRobot实例
    
    # 确保观测已更新（调用get_obs会更新内部状态）
    if not hasattr(agent, '_is_obs_updated') or not agent._is_obs_updated:
        # 如果观测未更新，需要先更新
        # 这里我们直接从agent的属性中获取，这些属性在step时会被更新
        pass
    
    # 1. 获取基础角速度 (base angular velocity)
    # agent._body_ang_vel 是在 get_obs 中计算的局部坐标系角速度
    base_ang_vel = agent._body_ang_vel.copy() if hasattr(agent, '_body_ang_vel') else np.zeros(3)
    
    # 2. 获取投影重力 (projected gravity)
    # 从body orientation计算重力方向
    # body_orientation是roll, pitch, yaw（但yaw被设为0）
    # 我们需要计算重力在body frame中的方向
    if hasattr(agent, '_body_orientation'):
        # 从roll和pitch计算重力方向
        roll = agent._body_orientation[0]
        pitch = agent._body_orientation[1]
        # 重力在body frame中的方向（归一化）
        projected_gravity = np.array([
            np.sin(pitch),
            -np.sin(roll) * np.cos(pitch),
            -np.cos(roll) * np.cos(pitch)
        ])
    else:
        projected_gravity = np.array([0, 0, -1])
    
    # 3. 获取命令速度 (commands)
    # 从agent获取命令
    # agent._command 是一个字典，包含 "lin_vel" (数组) 和 "ang_vel" (标量)
    if hasattr(agent, '_command') and isinstance(agent._command, dict):
        lin_vel = agent._command.get("lin_vel", np.zeros(3))
        ang_vel = agent._command.get("ang_vel", 0.0)
        
        # 获取配置参数用于归一化命令
        config = Lite3Config
        max_cmd_vel = np.array(config.get('max_cmd_vel', [0.8, 0.8, 0.8]))
        
        # 命令需要归一化到 [-1, 1] 范围
        # 观测计算函数会将归一化的命令乘以 max_cmd_vel
        # 实际速度范围：前进/后退 [-0.5, 0.5]，左右 [-0.3, 0.3]
        # 需要归一化到 [-1, 1]，使用实际速度范围的最大值
        # 前进/后退：除以 0.5（实际最大速度）
        # 左右：除以 0.3（实际最大速度）
        # 角速度：除以 max_cmd_vel[2]（通常是 0.8）
        forward_max = 0.5  # 配置中的最大前进速度
        lateral_max = 0.3  # 配置中的最大侧向速度
        
        commands_normalized = np.array([
            lin_vel[0] / forward_max if forward_max > 0 else 0.0,      # lin_vel_x 归一化到 [-1, 1]
            lin_vel[1] / lateral_max if lateral_max > 0 else 0.0,      # lin_vel_y 归一化到 [-1, 1]
            ang_vel / max_cmd_vel[2] if max_cmd_vel[2] > 0 else 0.0,  # ang_vel_yaw 归一化到 [-1, 1]
        ])
        
        # 限制在 [-1, 1] 范围内（防止超出范围）
        commands_normalized = np.clip(commands_normalized, -1.0, 1.0)
        
        commands = commands_normalized
    else:
        # 如果没有命令，使用零命令
        commands = np.array([0.0, 0.0, 0.0])
    
    # 4. 获取关节位置 (dof_pos)
    # agent._leg_joint_qpos 是当前的关节位置
    dof_pos = agent._leg_joint_qpos.copy() if hasattr(agent, '_leg_joint_qpos') else np.zeros(12)
    
    # 5. 获取关节速度 (dof_vel)
    # agent._leg_joint_qvel 是当前的关节速度
    dof_vel = agent._leg_joint_qvel.copy() if hasattr(agent, '_leg_joint_qvel') else np.zeros(12)
    
    # 6. 获取上一动作 (last_actions)
    # agent._last_action 是上一帧的动作
    last_actions = agent._last_action.copy() if hasattr(agent, '_last_action') else np.zeros(12)
    
    # 获取配置参数
    config = Lite3Config
    omega_scale = config.get('omega_scale', 0.25)
    dof_vel_scale = config.get('dof_vel_scale', 0.05)
    max_cmd_vel = np.array(config.get('max_cmd_vel', [0.8, 0.8, 0.8]))
    dof_pos_default = get_dof_pos_default_policy()
    
    # 计算Lite3格式观测
    obs = compute_lite3_obs(
        base_ang_vel=base_ang_vel,
        projected_gravity=projected_gravity,
        commands=commands,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        last_actions=last_actions,
        omega_scale=omega_scale,
        dof_vel_scale=dof_vel_scale,
        max_cmd_vel=max_cmd_vel,
        dof_pos_default=dof_pos_default
    )
    
    return obs


def run_simulation(env: gym.Env, 
                   agent_name: str,
                   policy,
                   time_step: float, 
                   frame_skip: int,
                   action_skip: int,
                   keyboard_control: KeyboardControl,
                   command_model: dict):
    """运行仿真循环"""
    obs, info = env.reset()
    
    dt = time_step * frame_skip * action_skip
    config = Lite3Config
    
    # 获取动作缩放和默认位置
    if config.get('use_original_action_scale', False):
        action_scale = np.array(config.get('action_scale_original', [0.25] * 12))
    else:
        action_scale = np.array(config.get('action_scale', [0.25] * 12))
    
    dof_pos_default = get_dof_pos_default_policy()
    
    # 初始化last_action
    last_action = np.zeros(12)
    
    # 初始化agent的last_action
    agent_name_full = f"{agent_name}_000"
    agent_base = env.unwrapped._agents[agent_name_full]
    agent = agent_base.agent
    if hasattr(agent, '_last_action'):
        agent._last_action = np.zeros(12)
    
    print("=" * 60)
    print("Lite3 ONNX策略仿真运行中...")
    print("=" * 60)
    print("控制说明:")
    print("  Z: 机器狗站立进入默认状态")
    print("  C: 机器狗站立进入RL控制状态")
    print("  W/S: 前进/后退（仅在RL控制状态下有效）")
    print("  A/D: 左移/右移（仅在RL控制状态下有效）")
    print("  Q/E: 顺时针/逆时针旋转（仅在RL控制状态下有效）")
    print("=" * 60)
    
    # 第一次循环前，先执行一步零动作来初始化环境状态
    action_init = np.zeros(env.action_space.shape[0])
    obs, reward, terminated, truncated, info = env.step(action_init)
    
    try:
        while True:
            start_time = datetime.now()
            
            # 更新键盘控制
            lin_vel, ang_vel, reborn, terrain_type, enter_default_state, enter_rl_control, rl_control_mode = keyboard_control.update()
            
            # 处理状态切换
            if enter_default_state:
                # 进入默认状态：重置环境并设置为默认姿态
                obs, info = env.reset()
                last_action = np.zeros(12)
                agent_base = env.unwrapped._agents[agent_name_full]
                agent = agent_base.agent
                if hasattr(agent, '_last_action'):
                    agent._last_action = np.zeros(12)
                # 执行一步零动作来初始化环境状态
                action_init = np.zeros(env.action_space.shape[0])
                obs, reward, terminated, truncated, info = env.step(action_init)
                continue
            
            if enter_rl_control:
                # 进入RL控制状态：确保环境已初始化
                if not hasattr(env.unwrapped, '_agents'):
                    obs, info = env.reset()
                    action_init = np.zeros(env.action_space.shape[0])
                    obs, reward, terminated, truncated, info = env.step(action_init)
            
            # 只有在RL控制状态下才运行策略
            if not rl_control_mode:
                # 默认状态：执行零动作保持站立
                action = np.zeros(env.action_space.shape[0])
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                # 控制频率（使用REALTIME_STEP，与xbot保持一致）
                elapsed_time = datetime.now() - start_time
                if elapsed_time.total_seconds() < REALTIME_STEP:
                    time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
                continue
            
            # 设置命令
            command_dict = {"lin_vel": lin_vel, "ang_vel": ang_vel}
            if hasattr(env, "setup_command"):
                env.setup_command(command_dict)
            else:
                env.unwrapped.setup_command(command_dict)
            
            # 验证命令是否正确设置到agent（调试后退问题）
            agent_base = env.unwrapped._agents[agent_name_full]
            agent = agent_base.agent
            if hasattr(agent, '_command') and lin_vel[0] < 0:
                actual_command = agent._command.get("lin_vel", None)
                if actual_command is not None:
                    print(f"[DEBUG] 后退命令检查: 设置={lin_vel[0]:.3f}, agent中={actual_command[0]:.3f}")
            
            # 计算Lite3格式观测
            lite3_obs = compute_lite3_observation_from_env(env.unwrapped, agent_name_full)
            
            # 运行策略
            policy_action = policy(lite3_obs)  # [12]
            
            # 将策略输出转换为目标关节位置
            # policy_action是相对于default位置的偏移，需要加上default位置
            target_dof_pos = policy_action * action_scale + dof_pos_default
            
            # 获取agent和当前状态
            agent_base = env.unwrapped._agents[agent_name_full]
            agent = agent_base.agent
            
            # 获取当前关节位置和速度
            current_dof_pos = agent._leg_joint_qpos.copy()
            current_dof_vel = agent._leg_joint_qvel.copy()
            
            kps = np.array(config['kps'])
            kds = np.array(config['kds'])
            
            # PD控制计算目标速度（用于位置控制）
            # 注意：OrcaGym使用位置控制，所以我们需要将目标位置转换为动作
            # 动作是相对于neutral位置的偏移
            neutral_joint_values = agent._neutral_joint_values.copy()
            action_offset = target_dof_pos - neutral_joint_values
            
            # 转换为环境动作格式
            # 动作需要归一化到action_space范围
            # 根据action_scale，动作范围是 [-action_scale, action_scale]
            # 但环境期望的是归一化的动作，所以需要除以action_scale
            normalized_action = action_offset / action_scale
            
            # 限制在[-1, 1]范围内
            action = np.clip(normalized_action, -1.0, 1.0)
            
            # 保存last_action用于下次观测计算（保存policy_action，即策略原始输出）
            agent._last_action = policy_action.copy()
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            # 控制频率（使用REALTIME_STEP，与xbot保持一致）
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
            
    except KeyboardInterrupt:
        print("\n退出仿真环境")
    finally:
        env.close()


def main(config: dict, remote: str = None):
    """主函数"""
    try:
        if remote is not None:
            orcagym_addresses = [remote]
        else:
            orcagym_addresses = config['orcagym_addresses']
        
        agent_name = config['agent_name']
        onnx_model_path = config['onnx_model_path']
        ctrl_device = config['ctrl_device']
        terrain_asset_paths = config['terrain_asset_paths']
        agent_asset_path = config['agent_asset_path']
        command_model = config['command_model']
        
        # 获取推理设备（默认CPU，可配置为cuda）
        inference_device = config.get('inference_device', 'cpu')
        
        # 加载ONNX策略
        policy = load_lite3_onnx_policy(onnx_model_path, device=inference_device)
        
        # 清空场景
        clear_scene(orcagym_addresses=orcagym_addresses)
        
        # 发布地形
        publish_terrain(
            orcagym_addresses=orcagym_addresses,
            terrain_asset_paths=terrain_asset_paths,
        )
        
        # 生成高度图
        height_map_file = generate_height_map_file(
            orcagym_addresses=orcagym_addresses,
        )
        
        # 放置机器人（调高10cm）
        publish_scene(
            orcagym_addresses=orcagym_addresses,
            agent_name=agent_name,
            agent_asset_path=agent_asset_path,
            agent_num=1,
            terrain_asset_paths=terrain_asset_paths
        )
        
        print("simulation running... , orcagym_addr: ", orcagym_addresses)
        env_name = "Lite3Sim-v0"
        env_id, kwargs = register_env(
            orcagym_addr=orcagym_addresses[0], 
            env_name=env_name, 
            env_index=0, 
            agent_name=agent_name, 
            ctrl_device=ctrl_device, 
            max_episode_steps=MAX_EPISODE_STEPS,
            height_map=height_map_file,
        )
        print("Registered environment: ", env_id)
        
        env = gym.make(env_id)
        print("Starting simulation...")
        
        # 设置摩擦系数
        friction_scale = config.get('friction_scale', None)
        if friction_scale is not None:
            env.unwrapped.setup_base_friction(friction_scale)
        
        # 创建键盘控制
        keyboard_control = KeyboardControl(orcagym_addresses[0], env, command_model)
        
        # 运行仿真
        run_simulation(
            env=env,
            agent_name=agent_name,
            policy=policy,
            time_step=TIME_STEP,
            frame_skip=FRAME_SKIP,
            action_skip=ACTION_SKIP,
            keyboard_control=keyboard_control,
            command_model=command_model,
        )
    finally:
        print("退出仿真环境")
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Lite3 ONNX policy in simulation')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--remote', type=str, help='Remote address of OrcaStudio (e.g., localhost:50051)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                       help='Inference device: cpu or cuda (overrides config file)')
    args = parser.parse_args()
    
    if args.config is None:
        # 使用默认配置
        config = {
            'orcagym_addresses': ['localhost:50051'],
            'agent_name': 'Lite3',
            'agent_asset_path': 'assets/prefabs/lite3_usda',
            'onnx_model_path': 'policy.onnx',
            'ctrl_device': 'keyboard',
            'inference_device': 'cpu',  # 默认使用CPU
            'terrain_asset_paths': ['assets/prefabs/terrain_test_usda'],
            'command_model': {
                'flat_terrain': {
                    'forward_speed': [-0.5, 0.5],
                    'left_speed': [-0.3, 0.3],
                    'turn_speed': 0.7853975,
                    'turbo_scale': 3.0,
                }
            },
            'friction_scale': 1.0,
        }
    else:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 命令行参数优先于配置文件
    if args.device is not None:
        config['inference_device'] = args.device
        print(f"[INFO] Using device from command line: {args.device}")
    
    main(config=config, remote=args.remote)

