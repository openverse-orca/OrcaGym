"""
ZQ SA01 人形机器人环境
基于 humanoid-gym 的控制策略，适配 OrcaGym 框架
"""

import numpy as np
from collections import deque
from typing import Optional
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()

np.set_printoptions(suppress=True, precision=6, floatmode='fixed')


class ZQSA01Env(OrcaGymLocalEnv):
    """ZQ SA01 双足人形机器人环境"""

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        frame_stack: int = 15,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs
        )
        
        self.verbose = verbose
        self.nu = int(self.model.nu)  # 12个关节
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        
        _logger.info(f"[ZQSA01] Model: nq={self.nq}, nv={self.nv}, nu={self.nu}")

        # PD控制参数（来自配置文件）
        self.kps = np.array([50, 50, 70, 70, 20, 20, 50, 50, 70, 70, 20, 20], dtype=np.float64)
        self.kds = np.array([5.0, 5.0, 7.0, 7.0, 2.0, 2.0, 5.0, 5.0, 7.0, 7.0, 2.0, 2.0], dtype=np.float64)
        self.tau_limit = 200.0
        self.action_scale = 0.5
        self.decimation = 10  # 控制频率 = 1000Hz / 10 = 100Hz
        
        # 默认关节位置（站立姿态）
        self.default_dof_pos = np.array([
            0.0, 0.0, -0.24, 0.48, -0.24, 0.0,  # 左腿
            0.0, 0.0, -0.24, 0.48, -0.24, 0.0   # 右腿
        ], dtype=np.float64)
        
        # 关节名称
        self.joint_names_str = [
            'leg_l1_joint', 'leg_l2_joint', 'leg_l3_joint',
            'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint',
            'leg_r1_joint', 'leg_r2_joint', 'leg_r3_joint',
            'leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint'
        ]
        self.joint_names = [self.joint(name) for name in self.joint_names_str]
        
        self.actuator_names_str = [
            'leg_l1_joint', 'leg_l2_joint', 'leg_l3_joint',
            'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint',
            'leg_r1_joint', 'leg_r2_joint', 'leg_r3_joint',
            'leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint'
        ]
        self.actuator_names = [self.actuator(name) for name in self.actuator_names_str]
        self.actuator_ids = [self.model.actuator_name2id(name) for name in self.actuator_names]
        # 观察历史
        self.frame_stack = frame_stack

        # 命令
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_dyaw = 0.0
        
        # 上一步动作（存储缩放后的动作，与Isaac Gym一致）
        self.last_action = np.zeros(12, dtype=np.float32)
        
        # 目标关节位置（用于PD控制）
        self.target_dof_pos = self.default_dof_pos.copy()
        # 观察维度: 命令(3) + 关节误差(12) + 关节速度(12) + 上次动作(12) + 角速度(3) + 欧拉角(3) = 45
        self.single_obs_dim = 45
        # 步数计数
        self.step_count = 0
        
        # 获取传感器名称
        self._detect_sensors()
        
        # 定义动作和观察空间
        self.action_space = spaces.Box(
            low=-18.0, high=18.0, shape=(12,), dtype=np.float32
        )
        full_obs_dim = self.single_obs_dim * self.frame_stack
        self.observation_space = spaces.Box(
            low=-18.0, high=18.0, shape=(full_obs_dim,), dtype=np.float32
        )
        
        _logger.info(f"[ZQSA01] PD gains: kp={self.kps[2]}, kd={self.kds[2]}, tau_limit={self.tau_limit}")
        _logger.info(f"[ZQSA01] Action space: {self.action_space.shape}")
        _logger.info(f"[ZQSA01] Observation space: {self.observation_space.shape}")

    def _detect_sensors(self):
        """自动检测传感器名称"""
        all_sensors = list(self.model._sensor_dict.keys())
        
        # 查找 IMU 传感器
        self.orientation_sensor = None
        self.angular_velocity_sensor = None
        
        for sensor in all_sensors:
            if 'orientation' in sensor.lower() or 'quat' in sensor.lower():
                self.orientation_sensor = sensor
            if 'angular' in sensor.lower() and 'velocity' in sensor.lower():
                self.angular_velocity_sensor = sensor
        
        if self.verbose:
            _logger.info(f"[ZQSA01] Detected sensors:")
            _logger.info(f"  - Orientation: {self.orientation_sensor}")
            _logger.info(f"  - Angular velocity: {self.angular_velocity_sensor}")

    def set_command(self, vx: float = 0.0, vy: float = 0.0, dyaw: float = 0.0):
        """设置运动命令"""
        self.cmd_vx = vx
        self.cmd_vy = vy
        self.cmd_dyaw = dyaw

    def set_action(self, action: np.ndarray):
        """处理策略输出的动作
        
        Args:
            action: 策略网络的原始输出 (12维)
        """
        # 限制动作范围（与Isaac Gym一致）
        action = np.clip(action, -100.0, 100.0)
        
        # 缩放动作并计算目标位置
        action_scaled = action * self.action_scale
        self.target_dof_pos = action_scaled + self.default_dof_pos
        
        # 保存缩放后的动作（用于下次观察，与Isaac Gym一致）
        self.last_action = action_scaled.astype(np.float32)
    
    def step(self, action):
        """执行一步仿真
        
        与 Isaac Gym 一致，每个 frame_skip 步骤都重新计算 PD 控制
        
        Args:
            action: 策略网络的原始输出，如果不为None则先调用set_action处理
        """
        # 如果传入了动作，先处理动作
        if action is not None:
            self.set_action(action)
        # 在每个仿真步骤中重新计算 PD 控制（与 Isaac Gym 一致）
        for _ in range(self.frame_skip):
            # 获取当前关节状态
            joint_qpos = self.query_joint_qpos(self.joint_names)
            joint_qvel = self.query_joint_qvel(self.joint_names)
            q = np.array([joint_qpos[jn] for jn in self.joint_names], dtype=np.float64).squeeze()
            dq = np.array([joint_qvel[jn] for jn in self.joint_names], dtype=np.float64).squeeze()
            
            # PD 控制计算力矩
            tau = self.kps * (self.target_dof_pos - q) - self.kds * dq
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)
            
            # 设置控制并仿真一步
            ctrl = np.zeros(self.nu, dtype=np.float32)
            for i in range(self.nu):
                ctrl[self.actuator_ids[i]] = tau[i]
            
            self.set_ctrl(ctrl)
            self.mj_step(nstep=1)
            self.gym.update_data()
        
        # 更新观察
        obs = self._get_obs()
        self.step_count += 1
        
        # 简单的奖励和终止条件
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """获取观察"""
        # 获取IMU数据
        if self.orientation_sensor:
            orientation_data = self.query_sensor_data([self.orientation_sensor])
            orientation = orientation_data[self.orientation_sensor].copy()
            # 转换为 [x,y,z,w] 格式
            quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]], dtype=np.float32)
            if np.allclose(quat, [0, 0, 0, 0], atol=1e-6):
                quat = np.array([0, 0, 0, 1], dtype=np.float32)
            quat = quat / np.linalg.norm(quat)
            euler = R.from_quat(quat).as_euler('xyz', degrees=False).astype(np.float32)
        else:
            euler = np.zeros(3, dtype=np.float32)
        
        if self.angular_velocity_sensor:
            ang_vel_data = self.query_sensor_data([self.angular_velocity_sensor])
            ang_vel = ang_vel_data[self.angular_velocity_sensor].copy().astype(np.float32)
        else:
            ang_vel = np.zeros(3, dtype=np.float32)
        
        # 获取关节状态
        joint_qpos = self.query_joint_qpos(self.joint_names)
        joint_qvel = self.query_joint_qvel(self.joint_names)
        
        dof_pos = np.array([joint_qpos[jn] for jn in self.joint_names], dtype=np.float32).squeeze()
        dof_vel = np.array([joint_qvel[jn] for jn in self.joint_names], dtype=np.float32).squeeze()
        
        # 调试：在首次调用时打印原始数据
        if self.step_count == 0 and self.verbose:
            _logger.info(f"[DEBUG] Raw sensor data:")
            _logger.info(f"  dof_pos: {dof_pos}")
            _logger.info(f"  dof_vel: {dof_vel}")
            _logger.info(f"  euler: {euler}")
            _logger.info(f"  ang_vel: {ang_vel}")
        
        # 构建单帧观察, 相位在主循环中设置
        # 缩放因子（与Isaac Gym训练时一致）
        lin_vel_scale = 2.0
        ang_vel_scale = 1.0
        
        single_obs = np.zeros(45, dtype=np.float32)
        idx = 0
        
        single_obs[idx] = self.cmd_vx * lin_vel_scale
        single_obs[idx+1] = self.cmd_vy * lin_vel_scale
        single_obs[idx+2] = self.cmd_dyaw * ang_vel_scale
        idx += 3

        # 关节位置误差（相对于默认位置）
        single_obs[idx:idx+12] = (dof_pos - self.default_dof_pos.astype(np.float32))
        idx += 12
        
        # 关节速度
        single_obs[idx:idx+12] = dof_vel * 0.05  # 缩放
        idx += 12
        
        # 上一步动作
        single_obs[idx:idx+12] = self.last_action
        idx += 12
               
        # 角速度
        single_obs[idx:idx+3] = ang_vel
        idx += 3

        # 欧拉角
        single_obs[idx:idx+3] = euler
        idx += 3
  
        return single_obs


    def reset_model(self):
        """重置环境"""
        # 重置基座位置
        try:
            base_joint = self.joint("root")
            self.set_joint_qpos({base_joint: np.array([0, 0, 1.1, 1, 0, 0, 0])})
        except:
            _logger.warning("[ZQSA01] Could not reset base joint")
        
        # 重置关节到默认位置
        joint_pos_dict = {
            self.joint_names[i]: self.default_dof_pos[i] 
            for i in range(12)
        }
        self.set_joint_qpos(joint_pos_dict)
        
        # 前进一步
        self.mj_forward()
        
        self.last_action = np.zeros(12, dtype=np.float32)
        self.target_dof_pos = self.default_dof_pos.copy()
        self.step_count = 0
        
        obs = self._get_obs()
        
        if self.verbose:
            _logger.info(f"[ZQSA01] Reset obs shape: {obs.shape}")
            _logger.info(f"  cmd: [{self.cmd_vx:.3f}, {self.cmd_vy:.3f}, {self.cmd_dyaw:.3f}]")
            _logger.info(f"  dof_pos: {obs[3:15]}")
            _logger.info(f"  dof_vel: {obs[15:27]}")
            _logger.info(f"  ang_vel: {obs[39:42]}")
            _logger.info(f"  euler: {obs[42:45]}")
        
        return obs, {}

    def render_callback(self, mode='human'):
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")
