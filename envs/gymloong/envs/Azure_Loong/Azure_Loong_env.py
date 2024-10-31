import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

# 导入基类 MujocoRobotEnv
from envs.robot_env import MujocoRobotEnv

# 导入配置类 LeggedRobotCfg
from envs.gymloong.envs.base.legged_robot_config import LeggedRobotCfg

class AzureLoongEnv(MujocoRobotEnv):
    """
    Azure Loong 环境，继承自 MujocoRobotEnv，整合 LeggedRobot 的逻辑。
    """

    def __init__(
        self,
        cfg: LeggedRobotCfg,
        args,
        grpc_address: str,
        agent_names: list[str],
        frame_skip: int = 1,
        **kwargs
    ):
        """
        初始化 Azure Loong 环境。

        Args:
            cfg (LeggedRobotCfg): 配置对象。
            grpc_address (str): gRPC 服务器地址。
            agent_names (list[str]): 代理名称列表。
            time_step (float): 仿真时间步长。
            frame_skip (int): 每个动作之间模拟的步数。
            model_path (str): MuJoCo 模型文件的路径。
        """
        self.cfg = cfg

        # 调用父类初始化
        super().__init__(
            frame_skip=frame_skip,
            grpc_address=grpc_address,
            agent_names=agent_names,
            time_step=time_step,
            n_actions=cfg.env.num_actions,
            observation_space=None,  # 将在下面定义
            action_space_type=None,
            action_step_count=None,
            **kwargs
        )

        # 初始化环境参数
        self._parse_cfg(self.cfg)

        # 初始化观测和动作空间
        self._init_spaces()

        # 初始化缓冲区和变量
        self._init_buffers()

        # 准备奖励函数
        self._prepare_reward_function()

        # 设置初始状态
        self.reset()

    def _parse_cfg(self, cfg):
        """
        解析配置文件。
        """
        self.dt = cfg.control.decimation * cfg.sim.dt
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = cfg.rewards.scales
        self.command_ranges = cfg.commands.ranges
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))

    def _init_spaces(self):
        """
        初始化观测空间和动作空间。
        """
        # 动作空间
        action_dim = self.cfg.env.num_actions
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

        # 观测空间
        obs_dim = self.cfg.env.num_observations
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _init_buffers(self):
        """
        初始化缓冲区和变量。
        """
        self.num_envs = 1  # 单环境
        self.device = 'cpu'  # 使用 CPU

        # 状态变量
        self.dof_pos = np.zeros(self.model.nq)
        self.dof_vel = np.zeros(self.model.nv)
        self.torques = np.zeros(self.model.nu)

        # 控制增益
        self.p_gains = np.zeros(self.model.nu)
        self.d_gains = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            actuator_name = self.model.actuator_id2name(i)
            joint_id = self.model.actuator_trnid[i, 0]
            joint_name = self.model.joint_id2name(joint_id)

            stiffness = self.cfg.control.stiffness.get(joint_name, 0.0)
            damping = self.cfg.control.damping.get(joint_name, 0.0)
            self.p_gains[i] = stiffness
            self.d_gains[i] = damping

        # 力矩限制
        self.torque_limits = self.model.actuator_ctrlrange[:, 1]

        # 其他变量
        self.episode_length = 0
        self.commands = np.zeros(self.cfg.commands.num_commands)
        self.episode_sums = {}

        # 默认关节位置
        self.default_dof_pos = np.array([
            self.cfg.init_state.default_joint_angles.get(self.model.joint_id2name(i), 0.0)
            for i in range(self.model.nq)
        ])

        # 观测和动作的缩放
        self.commands_scale = np.array([
            self.obs_scales.lin_vel,
            self.obs_scales.lin_vel,
            self.obs_scales.ang_vel,
            1.0  # heading
        ])

        # 上一个动作（用于计算动作变化率）
        self.last_actions = np.zeros(self.cfg.env.num_actions)

    def step(self, action):
        """
        环境的步进函数。

        Args:
            action (np.ndarray): 动作。

        Returns:
            observation (np.ndarray): 观测。
            reward (float): 奖励。
            done (bool): 是否结束。
            info (dict): 额外信息。
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.actions = action.copy()

        # 计算力矩并施加
        torques = self._compute_torques(action)
        self.apply_torques(torques)

        # 仿真步进
        for _ in range(self.cfg.control.decimation):
            self.sim.step()

        # 更新状态
        self._update_states()

        # 计算奖励
        reward = self._compute_reward()

        # 检查是否终止
        done = self._check_termination()

        # 获取观测
        observation = self._get_obs()

        # 增加时间步
        self.episode_length += 1
        if self.episode_length >= self.max_episode_length:
            done = True

        # 额外信息
        info = {}

        return observation, reward, done, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境。

        Returns:
            observation (np.ndarray): 初始观测。
        """
        super().reset(seed=seed)

        # 重置仿真状态
        self.sim.reset()
        self.sim.set_state(self.sim_state)
        self.sim.forward()

        # 重置变量
        self.episode_length = 0

        # 更新状态
        self._update_states()

        # 重置指令
        self._resample_commands()

        # 获取初始观测
        observation = self._get_obs()

        return observation

    def _compute_torques(self, action):
        """
        根据动作计算力矩。

        Args:
            action (np.ndarray): 动作。

        Returns:
            torques (np.ndarray): 力矩。
        """
        if self.cfg.control.control_type == "P":
            desired_pos = self.default_dof_pos + action * self.cfg.control.action_scale
            pos_error = desired_pos - self.dof_pos
            vel_error = -self.dof_vel
            torques = self.p_gains * pos_error + self.d_gains * vel_error
        elif self.cfg.control.control_type == "T":
            torques = action * self.cfg.control.action_scale
        else:
            raise ValueError(f"Unknown control type: {self.cfg.control.control_type}")

        # 限制力矩
        torques = np.clip(torques, -self.torque_limits, self.torque_limits)
        return torques

    def apply_torques(self, torques):
        """
        施加力矩到仿真。

        Args:
            torques (np.ndarray): 力矩。
        """
        self.sim.data.ctrl[:] = torques

    def _update_states(self):
        """
        更新状态变量。
        """
        self.dof_pos = self.sim.data.qpos.copy()
        self.dof_vel = self.sim.data.qvel.copy()

        # 基座位置和朝向
        self.base_pos = self.dof_pos[:3]
        self.base_quat = self.dof_pos[3:7]

        # 基座速度
        self.base_lin_vel = self.dof_vel[:3]
        self.base_ang_vel = self.dof_vel[3:6]

    def _get_obs(self):
        """
        获取观测。

        Returns:
            observation (np.ndarray): 观测数组。
        """
        # 构建观测
        observation = np.concatenate([
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self._get_projected_gravity(),
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,  # 上一个动作
            # 如果需要，高度测量等其他观测
        ])
        return observation

    def _get_projected_gravity(self):
        """
        计算重力在机器人坐标系下的表示。

        Returns:
            projected_gravity (np.ndarray): 重力向量。
        """
        # 获取重力向量
        gravity_vec = np.array(self.sim.model.opt.gravity)
        # 将重力向量转换到机器人坐标系
        base_quat_inv = self.base_quat.copy()
        base_quat_inv[1:] *= -1  # 共轭
        projected_gravity = self._quat_rotate(base_quat_inv, gravity_vec)
        return projected_gravity

    def _quat_rotate(self, quat, vec):
        """
        使用四元数旋转向量。

        Args:
            quat (np.ndarray): 四元数。
            vec (np.ndarray): 向量。

        Returns:
            rotated_vec (np.ndarray): 旋转后的向量。
        """
        import transforms3d.quaternions as tq
        return tq.quat2mat(quat).dot(vec)

    def _compute_reward(self):
        """
        计算奖励。

        Returns:
            reward (float): 奖励值。
        """
        reward = 0.0

        for name, func in zip(self.reward_names, self.reward_functions):
            rew = func() * self.reward_scales.get(name, 0.0)
            reward += rew
            # 记录奖励总和
            if name not in self.episode_sums:
                self.episode_sums[name] = 0.0
            self.episode_sums[name] += rew

        # 如果只允许正奖励
        if self.cfg.rewards.only_positive_rewards:
            reward = max(reward, 0.0)

        # 处理终止奖励
        if "termination" in self.reward_scales:
            termination_rew = self._reward_termination() * self.reward_scales["termination"]
            reward += termination_rew
            self.episode_sums["termination"] = self.episode_sums.get("termination", 0.0) + termination_rew

        return reward

    def _prepare_reward_function(self):
        """
        准备奖励函数。
        """
        self.reward_functions = []
        self.reward_names = []
        # 处理奖励缩放因子
        self.reward_scales = {k: v * self.dt for k, v in self.reward_scales.items() if v != 0}

        for name in self.reward_scales.keys():
            if name == "termination":
                continue
            func_name = f"_reward_{name}"
            if hasattr(self, func_name):
                self.reward_functions.append(getattr(self, func_name))
                self.reward_names.append(name)
            else:
                print(f"Warning: Reward function '{func_name}' not found.")

    def _check_termination(self):
        """
        检查是否终止。

        Returns:
            done (bool): 是否终止。
        """
        done = False

        # 检查终止条件
        if self.base_pos[2] < 0.2:  # 地面以下
            done = True
        if np.abs(self.base_quat[0]) > 0.5:  # 翻倒
            done = True

        return done

    def _resample_commands(self):
        """
        重新采样指令。
        """
        self.commands[0] = np.random.uniform(self.command_ranges.lin_vel_x[0], self.command_ranges.lin_vel_x[1])
        self.commands[1] = np.random.uniform(self.command_ranges.lin_vel_y[0], self.command_ranges.lin_vel_y[1])
        if self.cfg.commands.heading_command:
            self.commands[3] = np.random.uniform(self.command_ranges.heading[0], self.command_ranges.heading[1])
            # 根据 heading 计算 ang_vel_yaw
            forward = self._quat_rotate(self.base_quat, np.array([1.0, 0.0, 0.0]))
            heading = np.arctan2(forward[1], forward[0])
            self.commands[2] = np.clip(0.5 * self._wrap_to_pi(self.commands[3] - heading), -1.0, 1.0)
        else:
            self.commands[2] = np.random.uniform(self.command_ranges.ang_vel_yaw[0], self.command_ranges.ang_vel_yaw[1])

    def _wrap_to_pi(self, angle):
        """
        将角度限制在 [-pi, pi] 范围内。

        Args:
            angle (float): 角度。

        Returns:
            wrapped_angle (float): 限制后的角度。
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # 实现具体的奖励函数，例如：
    def _reward_tracking_lin_vel(self):
        """
        奖励机器人跟踪线速度命令。

        Returns:
            reward (float): 奖励值。
        """
        lin_vel_error = self.commands[:2] - self.base_lin_vel[:2]
        error = np.sum(np.square(lin_vel_error))
        reward = np.exp(-error / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_tracking_ang_vel(self):
        """
        奖励机器人跟踪角速度命令。

        Returns:
            reward (float): 奖励值。
        """
        ang_vel_error = self.commands[2] - self.base_ang_vel[2]
        reward = np.exp(-np.square(ang_vel_error) / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_torques(self):
        """
        惩罚机器人使用过大的力矩。

        Returns:
            reward (float): 奖励值（负值）。
        """
        reward = -np.sum(np.square(self.torques))
        return reward

    def _reward_action_rate(self):
        """
        惩罚动作变化过快。

        Returns:
            reward (float): 奖励值（负值）。
        """
        action_rate = self.actions - self.last_actions
        reward = -np.sum(np.square(action_rate))
        self.last_actions = self.actions.copy()
        return reward

    def _reward_termination(self):
        """
        终止奖励。

        Returns:
            reward (float): 奖励值。
        """
        return -1.0 if self._check_termination() else 0.0

    def close(self):
        """
        关闭环境。
        """
        if self.viewer is not None:
            self.viewer = None

    # 根据需要，添加其他方法和奖励函数
