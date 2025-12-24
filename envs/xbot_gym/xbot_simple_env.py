# xbot_simple_env.py
"""
基于 standaloneMujoco.py 的简化 XBot 环境
直接移植standalone代码的控制逻辑到OrcaGym框架
"""

import numpy as np
import math
import mujoco
from collections import deque
from typing import Optional, Tuple, Dict
from gymnasium import spaces

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class XBotSimpleEnv(OrcaGymLocalEnv):
    """
    简化的XBot环境，直接移植standalone_mujoco_sim.py的逻辑
    """

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        frame_stack: int = 15,
        verbose: bool = True,  # 控制日志输出
        **kwargs
    ):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs
        )
        
        self.verbose = verbose  # 保存verbose标志

        # 模型信息
        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        _logger.info(f"[XBotSimpleEnv] Model: nq={self.nq}, nv={self.nv}, nu={self.nu}")

        # 控制参数 - 恢复humanoid-gym的原始值
        # 因为我们会实现正确的decimation（10次内部循环）
        self.kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.float64)
        self.kds = np.array([10.0] * 12, dtype=np.float64)
        self.tau_limit = 200.0
        self.action_scale = 0.25
        
        # Decimation - 关键！匹配humanoid-gym
        self.decimation = 10  # 每次策略更新，内部循环10次物理步
        
        # 简单检查timestep
        if self.verbose:
            actual_timestep = self.gym._mjModel.opt.timestep
            _logger.performance(f"[XBotSimpleEnv] Physics timestep: {actual_timestep}s ({1.0/actual_timestep:.0f}Hz)")
        
        _logger.info(f"[XBotSimpleEnv] Using humanoid-gym PD gains: kp_max={np.max(self.kps)}, kd={self.kds[0]}, tau_limit={self.tau_limit}, action_scale={self.action_scale}, decimation={self.decimation}")

        # 观察历史
        self.frame_stack = frame_stack
        self.single_obs_dim = 47
        self.hist_obs = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.hist_obs.append(np.zeros(self.single_obs_dim, dtype=np.float32))

        # 命令 (初始不移动，先站稳)
        self.cmd_vx = 0.0  # 先站稳，不移动
        self.cmd_vy = 0.0
        self.cmd_dyaw = 0.0

        # 最后的动作
        self.last_action = np.zeros(12, dtype=np.float32)
        
        # 动作平滑 - 减缓脚步摆动频率
        # ⭐ 修改：关闭平滑，与standaloneMujoco一致
        self.use_action_filter = False  # 关闭动作平滑（standaloneMujoco无平滑）
        self.action_filter_alpha = 0.08  # 如果启用的话
        self.filtered_action = np.zeros(12, dtype=np.float32)

        # 控制数组
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        
        # Warmup阶段 - 渐进式启动避免摔倒
        # ⭐ 修改：关闭warmup，与standaloneMujoco一致
        self.step_count = 0
        self.warmup_steps = 500  # 如果启用的话
        self.use_warmup = False  # 关闭warmup（standaloneMujoco无warmup）
        
        # ⭐ 调试信息变量
        self.last_tau = np.zeros(12, dtype=np.float64)  # 最后的扭矩
        self.last_base_pos = np.zeros(3, dtype=np.float64)  # 最后的base位置
        
        # 基座名称（用于真实位置查询）
        # 从错误信息看，body名称格式是: XBot-L_usda_base_link
        # 尝试自动检测正确的base_link名称
        self.base_body_name = None
        try:
            all_bodies = self.model.get_body_names()
            # 尝试几种可能的命名模式
            candidates = [
                "XBot-L_usda_base_link",  # USDA导入格式
                f"{agent_names[0]}_base_link" if len(agent_names) > 0 else None,
                "base_link",
            ]
            for candidate in candidates:
                if candidate and candidate in all_bodies:
                    self.base_body_name = candidate
                    break
            
            if self.base_body_name is None:
                # 搜索包含"base"和"link"的body
                for body in all_bodies:
                    if "base" in body.lower() and "link" in body.lower():
                        self.base_body_name = body
                        break
        except:
            self.base_body_name = "base_link"  # 默认值
        
        _logger.info(f"[XBotSimpleEnv] Using base body name: {self.base_body_name}")
        
        # 上一次的基座位置（用于检测位移和估算角速度）
        self.last_base_pos = None
        self.last_base_euler = None
        self.last_base_quat = None

        # 定义动作和观察空间
        self.action_space = spaces.Box(
            low=-18.0, high=18.0, shape=(12,), dtype=np.float32
        )
        full_obs_dim = self.single_obs_dim * self.frame_stack
        self.observation_space = spaces.Box(
            low=-18.0, high=18.0, shape=(full_obs_dim,), dtype=np.float32
        )

        _logger.info(f"[XBotSimpleEnv] Initialized with frame_stack={frame_stack}")
        _logger.info(f"[XBotSimpleEnv] Action space: {self.action_space.shape}")
        _logger.info(f"[XBotSimpleEnv] Observation space: {self.observation_space.shape}")

    def set_command(self, vx: float = 0.0, vy: float = 0.0, dyaw: float = 0.0):
        """设置运动命令"""
        self.cmd_vx = vx
        self.cmd_vy = vy
        self.cmd_dyaw = dyaw
        _logger.info(f"[Command] Set to vx={vx:.2f}, vy={vy:.2f}, dyaw={dyaw:.2f}")
    
    def set_smoothness(self, alpha: float = 0.15):
        """
        设置动作平滑度
        alpha: 0.0-1.0，越小越平滑
          - 0.1: 非常平滑，脚步摆动很慢
          - 0.3: 中等平滑
          - 1.0: 无平滑，完全响应
        """
        self.action_filter_alpha = np.clip(alpha, 0.0, 1.0)
        _logger.info(f"[Config] Action smoothness set to alpha={self.action_filter_alpha:.2f}")
    
    def enable_warmup(self, enabled: bool = True, steps: int = 300):
        """启用/禁用warmup阶段"""
        self.use_warmup = enabled
        self.warmup_steps = steps
        _logger.info(f"[Config] Warmup {'enabled' if enabled else 'disabled'} ({steps} steps)")

    def quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转欧拉角
        quat: 可能是 [x,y,z,w] 或 [w,x,y,z] 格式
        """
        # OrcaGym返回的四元数可能是[w,x,y,z]格式
        if len(quat) == 4:
            # 尝试判断格式：如果第一个元素接近±1，可能是w在前
            if abs(quat[0]) > 0.9 and abs(quat[1]) < 0.5 and abs(quat[2]) < 0.5 and abs(quat[3]) < 0.5:
                # 可能是 [w,x,y,z] 格式
                w, x, y, z = quat
            else:
                # 可能是 [x,y,z,w] 格式
                x, y, z, w = quat
        else:
            x, y, z, w = 0, 0, 0, 1
            
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _build_single_observation(self) -> np.ndarray:
        """
        构建单帧观察 (47维) - 完全按照 standalone 的逻辑
        """
        # 获取关节状态 (最后12个)
        q = self.data.qpos[-12:].astype(np.float64)
        dq = self.data.qvel[-12:].astype(np.float64)

        # 获取IMU传感器数据 - 尝试多种传感器名称格式
        sensor_read_success = False
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        omega = np.zeros(3, dtype=np.float64)
        
        # 尝试多种传感器名称格式
        sensor_name_candidates = [
            ("orientation", "angular-velocity"),  # 原始名称
            ("XBot_orientation", "XBot_angular-velocity"),  # 加agent前缀
            ("XBot-L_orientation", "XBot-L_angular-velocity"),  # 加模型名前缀
        ]
        
        for ori_name, gyro_name in sensor_name_candidates:
            try:
                sensor_dict = self.query_sensor_data([ori_name, gyro_name])
                quat_wxyz = np.array(sensor_dict[ori_name], dtype=np.float64)
                quat = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
                omega = np.array(sensor_dict[gyro_name], dtype=np.float64)
                sensor_read_success = True
                if self.step_count == 1:
                    _logger.info(f"[Success] Using sensors: '{ori_name}', '{gyro_name}'")
                break
            except:
                continue
        
        # 如果所有尝试都失败，使用body查询获取姿态
        if not sensor_read_success and self.base_body_name:
            try:
                _, _, xquat = self.get_body_xpos_xmat_xquat([self.base_body_name])
                quat = xquat.copy()
                
                # 尝试估算角速度（从姿态变化）
                if self.last_base_quat is not None:
                    # 简单的差分估计: omega ≈ Δeuler / Δt
                    current_euler = self.quaternion_to_euler(quat)
                    if self.last_base_euler is not None:
                        delta_euler = current_euler - self.last_base_euler
                        dt = self.time_step * self.frame_skip  # 0.01s
                        omega = delta_euler / dt  # rad/s
                        # 限制范围避免异常值
                        # ⭐ 修改：放宽限制（standaloneMujoco无限制）
                        omega = np.clip(omega, -20.0, 20.0)
                
                sensor_read_success = True
                if self.step_count == 1:
                    _logger.info(f"[Fallback] Using body quaternion from '{self.base_body_name}' (omega estimated from Δeuler)")
            except:
                pass
        
        if not sensor_read_success and self.step_count == 1:
            _logger.error(f"[Error] Cannot read IMU sensors! Observation will have zero omega/euler.")

        # 转换欧拉角
        eu_ang = self.quaternion_to_euler(quat)

        # 构建观察向量 (完全按照standalone的顺序)
        obs = np.zeros(47, dtype=np.float32)
        obs[0] = math.sin(2 * math.pi * self.data.time / 0.64)  # 步态相位 sin
        obs[1] = math.cos(2 * math.pi * self.data.time / 0.64)  # 步态相位 cos
        obs[2] = self.cmd_vx * 2.0  # 线速度命令 x
        obs[3] = self.cmd_vy * 2.0  # 线速度命令 y
        obs[4] = self.cmd_dyaw * 1.0  # 角速度命令
        obs[5:17] = q * 1.0  # 关节位置 (12)
        obs[17:29] = dq * 0.05  # 关节速度 (12)
        obs[29:41] = self.last_action  # 历史动作 (12)
        obs[41:44] = omega  # 角速度 (3)
        obs[44:47] = eu_ang  # 欧拉角 (3)
        
        # 在前几步打印观察向量的关键部分，验证数据
        if self.step_count <= 3:
            print(f"[ObsDebug] Step={self.step_count} | "
                  f"cmd=[{obs[2]:.2f},{obs[3]:.2f},{obs[4]:.2f}] | "
                  f"q_range=[{np.min(q):.3f},{np.max(q):.3f}] | "
                  f"dq_range=[{np.min(dq):.3f},{np.max(dq):.3f}] | "
                  f"omega={omega} | "
                  f"euler={np.degrees(eu_ang)}")

        return np.clip(obs, -18.0, 18.0)

    def get_full_obs_vector(self) -> np.ndarray:
        """
        构建完整的观察向量 (47 * frame_stack)
        """
        single = self._build_single_observation()
        self.hist_obs.append(single)

        # 拼接历史观察
        stacked = np.zeros(self.single_obs_dim * self.frame_stack, dtype=np.float32)
        for i, h in enumerate(self.hist_obs):
            stacked[i * self.single_obs_dim:(i + 1) * self.single_obs_dim] = h

        return stacked

    def reset_model(self) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        """
        _logger.info("[XBotSimpleEnv] Resetting environment...")

        # 清空控制和历史
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.filtered_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0  # 重置步数计数器
        
        self.hist_obs.clear()
        for _ in range(self.frame_stack):
            self.hist_obs.append(np.zeros(self.single_obs_dim, dtype=np.float32))
        
        # 重置位置和姿态跟踪
        self.last_base_pos = None
        self.last_base_euler = None
        self.last_base_quat = None

        # ⭐ 修改：设置为"爬行模式"，降低初始高度（模拟standaloneMujoco）
        try:
            # 获取当前qpos
            current_qpos = self.data.qpos.copy()
            
            # 设置base高度为接近地面（standaloneMujoco显示约0.0m）
            # qpos[2]是base的z坐标
            CRAWL_HEIGHT = 0.05  # 5cm高度，接近地面
            current_qpos[2] = CRAWL_HEIGHT
            
            _logger.info(f"[爬行模式] 设置初始高度: {CRAWL_HEIGHT}m (原始约0.88m)")
            
            # 通过OrcaGym API设置qpos
            import asyncio
            if hasattr(self, 'loop') and self.loop:
                self.loop.run_until_complete(self.gym.set_qpos(current_qpos))
                self.gym.mj_forward()
                self.gym.update_data()
                _logger.info(f"[爬行模式] qpos已更新，新base高度: {current_qpos[2]:.3f}m")
            else:
                _logger.info(f"[警告] 无法设置qpos（loop不可用）")
        except Exception as e:
            _logger.info(f"[警告] 设置爬行模式失败: {e}")

        # 获取初始观察
        obs = self.get_full_obs_vector()

        _logger.info(f"[XBotSimpleEnv] Reset complete. Obs shape: {obs.shape}")
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步 - 添加平滑和warmup来提高稳定性
        
        action: 策略输出 [-18, 18] 范围的12维动作
        """
        self.step_count += 1
        
        # 裁剪动作到有效范围
        action = np.clip(action, -18.0, 18.0).astype(np.float32)
        
        # === 改进1: Warmup阶段，渐进增加动作幅度 ===
        if self.use_warmup and self.step_count <= self.warmup_steps:
            # 平滑的warmup曲线: 从0.1开始，逐渐增加到1.0
            warmup_progress = self.step_count / self.warmup_steps
            warmup_scale = 0.1 + 0.9 * warmup_progress  # 0.1 -> 1.0
            action = action * warmup_scale
            
            if self.step_count % 100 == 0:
                print(f"[Warmup] Step {self.step_count}/{self.warmup_steps}, "
                      f"scale={warmup_scale:.2f}, action_norm={np.linalg.norm(action):.2f}")
        
        # === 改进2: 动作平滑，减缓脚步摆动频率 ===
        if self.use_action_filter:
            # 指数移动平均: new = α*current + (1-α)*history
            self.filtered_action = (self.action_filter_alpha * action + 
                                   (1 - self.action_filter_alpha) * self.filtered_action)
            action_to_use = self.filtered_action
        else:
            action_to_use = action
        
        self.last_action = action_to_use.copy()

        # 计算目标位置 (action * action_scale)
        target_q = action_to_use * self.action_scale
        target_dq = np.zeros(12, dtype=np.float64)

        # === 关键修改：实现decimation，匹配humanoid-gym ===
        # 每次策略更新，内部循环多次物理步，每次都重新计算PD
        last_tau = None  # ⭐ 保存最后的扭矩用于诊断
        for _ in range(self.decimation):
            # 获取当前关节状态（每次物理步都更新）
            q = self.data.qpos[-12:].astype(np.float64)
            dq = self.data.qvel[-12:].astype(np.float64)

            # PD控制计算力矩
            tau = (target_q - q) * self.kps + (target_dq - dq) * self.kds
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)
            
            last_tau = tau.copy()  # ⭐ 保存最后的扭矩

            # 设置控制
            self.ctrl[-12:] = tau.astype(np.float32)

            # 单步物理仿真 (1ms if time_step=0.001)
            self.do_simulation(self.ctrl, 1)
        
        # ⭐ 保存调试信息
        self.last_tau = last_tau

        # 获取新的观察
        obs = self.get_full_obs_vector()

        # === 真实状态检测（基于body位置查询）===
        real_base_z = 0.0
        real_euler = np.array([0.0, 0.0, 0.0])
        real_omega = np.array([0.0, 0.0, 0.0])
        position_valid = False
        
        try:
            # 查询真实的基座位置和姿态
            xpos, xmat, xquat = self.get_body_xpos_xmat_xquat([self.base_body_name])
            real_base_z = float(xpos[2])
            
            # 在第1步打印原始四元数用于调试
            if self.step_count == 1:
                _logger.debug(f"\n[Debug] Raw quaternion from get_body_xpos_xmat_xquat: {xquat}")
                _logger.debug(f"[Debug] Interpreting as: w={xquat[0]:.3f}, x={xquat[1]:.3f}, y={xquat[2]:.3f}, z={xquat[3]:.3f}")
            
            real_euler = self.quaternion_to_euler(xquat)
            position_valid = True
            
            # ⭐ 保存base位置用于诊断
            self.last_base_pos = xpos.copy()
            
            # 尝试获取角速度
            try:
                sensor_dict = self.query_sensor_data(["angular-velocity"])
                real_omega = np.array(sensor_dict.get("angular-velocity", [0.0, 0.0, 0.0]))
            except:
                pass
        except Exception as e:
            # 如果查询失败，第一次打印错误和可用body列表
            if self.step_count == 1:
                _logger.error(f"\n⚠️ Warning: Cannot query base body '{self.base_body_name}'")
                _logger.error(f"   Error: {e}")
                try:
                    all_bodies = self.model.get_body_names()
                    _logger.info(f"   Available bodies: {all_bodies[:15]}")
                    # 尝试找到包含base的body
                    base_bodies = [b for b in all_bodies if 'base' in b.lower()]
                    if base_bodies:
                        _logger.info(f"   Bodies with 'base': {base_bodies}")
                except Exception as e2:
                    _logger.error(f"   Cannot list bodies: {e2}")
                print()
            
            # 使用qpos兜底
            try:
                real_base_z = float(self.data.qpos[2])
            except:
                real_base_z = 0.0
        
        # === 摔倒检测 (现在启用) ===
        is_fallen = False
        fall_reason = []
        roll_deg = 0.0
        pitch_deg = 0.0
        
        if position_valid:
            roll_deg = np.degrees(abs(real_euler[0]))
            pitch_deg = np.degrees(abs(real_euler[1]))
            
            # 高度检测
            # ⭐ 爬行模式：降低高度阈值（原0.7m → 0.02m）
            if real_base_z < 0.02:
                is_fallen = True
                fall_reason.append(f"高度过低({real_base_z:.2f}m<0.02m，爬行模式)")
            
            # 姿态检测
            if roll_deg > 30:
                is_fallen = True
                fall_reason.append(f"Roll过大({roll_deg:.1f}°>30°)")
            if pitch_deg > 30:
                is_fallen = True
                fall_reason.append(f"Pitch过大({pitch_deg:.1f}°>30°)")
        
        terminated = is_fallen
        truncated = False
        
        # === 奖励函数 (简单但有效) ===
        reward = 0.0
        
        if position_valid:
            # 1. 保持高度奖励 (目标0.9m)
            height_error = abs(real_base_z - 0.9)
            height_reward = np.exp(-height_error * 10.0)  # 越接近0.9m奖励越高
            reward += height_reward * 2.0
            
            # 2. 保持直立奖励
            orientation_error = (roll_deg + pitch_deg) / 180.0
            orientation_reward = np.exp(-orientation_error * 20.0)
            reward += orientation_reward * 3.0
            
            # 3. 存活奖励
            if not terminated:
                reward += 0.5
            
            # 4. 摔倒惩罚
            if terminated:
                reward -= 5.0
        
        # 保存当前位置和姿态用于下次对比和omega估算
        if position_valid:
            self.last_base_pos = xpos.copy()
            self.last_base_euler = real_euler.copy()
            self.last_base_quat = xquat.copy()
        info = {
            'base_z': real_base_z,
            'euler': real_euler,
            'is_fallen': is_fallen,
            'fall_reason': ' + '.join(fall_reason) if fall_reason else '',
            'reward': reward,
            'roll_deg': roll_deg,
            'pitch_deg': pitch_deg
        }
        
        # 摔倒打印
        if is_fallen and fall_reason:
            _logger.info(f"\n[FALL] Step={self.step_count} | {' + '.join(fall_reason)}\n")
        
        # 详细诊断：前10步每步打印，之后每200步打印
        should_print_diagnostic = (self.step_count <= 10) or (self.step_count % 200 == 0)
        
        if should_print_diagnostic:
            if position_valid:
                try:
                    # 计算位移（如果有上次位置）
                    displacement = ""
                    if self.last_base_pos is not None:
                        delta_pos = xpos - self.last_base_pos
                        delta_interval = 1 if self.step_count <= 10 else 200
                        displacement = f"Δ{delta_interval}步=({delta_pos[0]:.3f},{delta_pos[1]:.3f},{delta_pos[2]:.3f})"
                    
                    roll_deg = np.degrees(real_euler[0])
                    pitch_deg = np.degrees(real_euler[1])
                    yaw_deg = np.degrees(real_euler[2])
                    
                    # 状态标记
                    status = "✓"
                    if abs(roll_deg) > 30 or abs(pitch_deg) > 30:
                        status = "⚠️"
                    if abs(roll_deg) > 40 or abs(pitch_deg) > 40:
                        status = "❌"
                    # ⭐ 爬行模式：调整诊断阈值
                    if real_base_z < 0.02:
                        status = "❌"
                    
                    print(f"[{status}] Step={self.step_count:5d} | "
                          f"Pos=({xpos[0]:6.3f},{xpos[1]:6.3f},{xpos[2]:6.3f})m | "
                          f"Roll={roll_deg:6.1f}° | Pitch={pitch_deg:6.1f}° | Yaw={yaw_deg:6.1f}° | "
                          f"Reward={reward:6.2f} | "
                          f"Action={np.linalg.norm(action):5.2f} | "
                          f"Filtered={np.linalg.norm(action_to_use):5.2f} | "
                          f"Tau_max={np.max(np.abs(tau)):6.1f} | "
                          f"{displacement}")
                except Exception as e:
                    _logger.error(f"[Diagnostics] Step={self.step_count} position_valid=True but print failed: {e}")
            else:
                # 查询失败，尝试列出所有body看看名称
                if self.step_count == 200:
                    try:
                        all_bodies = self.model.get_body_names()
                        _logger.error(f"[Diagnostics] Cannot find '{self.base_body_name}'. Available bodies (first 10): {all_bodies[:10]}")
                    except:
                        pass
                _logger.error(f"[Diagnostics] Step={self.step_count} | position_valid=False, cannot query base position")

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # 简单测试
    _logger.info("XBotSimpleEnv module loaded successfully")

