import numpy as np


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

class JointController:
    def __init__(self, Kp=10.0, Ki=0.1, Kd=2.0, Kv=5.0, max_speed=80.0, ctrlrange=(-80, 80)):
        """
        初始化控制器参数
        :param Kp: 位置比例增益
        :param Ki: 积分增益
        :param Kd: 速度微分增益
        :param Kv: 速度误差增益（比例速度规划）
        :param max_speed: 最大允许速度（弧度/秒）
        :param ctrlrange: 驱动器的力矩范围（tuple）
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kv = Kv
        self.max_speed = max_speed
        self.ctrl_low, self.ctrl_high = ctrlrange
        
        # PID 状态变量
        self.integral = 0.0
        self.prev_error_pos = 0.0
        self.prev_error_vel = 0.0

    def _clamp(self, value, low, high):
        """限制值在指定范围内"""
        return np.clip(value, low, high)

    def compute_torque(self, target_qpos, current_qpos, current_qvel, dt):
        """
        计算关节力矩
        :param target_qpos: 用户输入的目标角度（弧度）
        :param current_qpos: 当前关节角度（弧度）
        :param current_qvel: 当前关节速度（弧度/秒）
        :param dt: 仿真步长（秒）
        :return: 输出力矩（Nm）
        """
        # 1. 比例速度规划：生成目标速度
        error_pos = target_qpos - current_qpos
        target_velocity = self.Kv * error_pos  # 比例控制生成速度指令
        
        # 2. 限制目标速度在物理范围内
        target_velocity = self._clamp(target_velocity, -self.max_speed, self.max_speed)

        # 3. PID 控制器计算
        # 位置误差
        error_pos = target_qpos - current_qpos
        
        # 速度误差（目标速度 - 当前速度）
        error_vel = target_velocity - current_qvel
        
        # 积分项（仅累积位置误差）
        self.integral += error_pos * dt
        
        # 抗积分饱和（限制积分项范围）
        integral_limit = 100.0  # 根据实际情况调整
        self.integral = self._clamp(self.integral, -integral_limit, integral_limit)
        
        # 微分项（位置误差变化率）
        derivative_pos = (error_pos - self.prev_error_pos) / dt if dt > 0 else 0.0
        
        # PID 输出
        torque = (
            self.Kp * error_pos 
            + self.Ki * self.integral 
            + self.Kd * derivative_pos 
            + self.Kv * error_vel  # 速度误差反馈
        )
        
        # 4. 限制力矩输出范围
        torque = self._clamp(torque, self.ctrl_low, self.ctrl_high)
        
        # 更新历史记录
        self.prev_error_pos = error_pos
        self.prev_error_vel = error_vel
        
        return torque