import sys
import os
import grpc
import numpy as np

class OrcaGymData:
    def __init__(self, model):
        """
        初始化数据容器，创建零初始化的状态数组
        
        在环境初始化时自动创建，通过 env.data 访问。
        存储仿真过程中的动态状态信息，每次仿真步进后由服务器更新。
        
        术语说明:
            - qpos (关节位置): 所有关节的广义坐标位置，包括旋转关节角度、平移关节位移等
            - qvel (关节速度): 所有关节的广义速度，对应 qpos 的导数
            - qacc (关节加速度): 所有关节的广义加速度，对应 qvel 的导数
            - qfrc_bias (偏置力): 包括重力、科里奥利力、离心力等被动力，用于动力学计算
        
        使用示例:
            ```python
            # 获取当前状态
            state = {
                "time": self.data.time,
                "qpos": self.data.qpos.copy(),  # 所有关节位置
                "qvel": self.data.qvel.copy(),  # 所有关节速度
                "qacc": self.data.qacc.copy(),  # 所有关节加速度
            }
            ```
        
        使用示例:
            ```python
            # 获取特定关节的状态
            q = self.data.qpos[-12:].astype(np.float64)  # 最后12个关节位置
            dq = self.data.qvel[-12:].astype(np.float64)  # 最后12个关节速度
            ```
        """
        self.model = model
        self.qpos = np.zeros(model.nq)
        self.qvel = np.zeros(model.nv)
        self.qacc = np.zeros(model.nv)
        self.qfrc_bias = np.zeros(model.nv)
        self.time = 0

    def update_qpos_qvel_qacc(self, qpos, qvel, qacc):
        """
        更新关节位置、速度和加速度数据
        
        通常在从服务器同步数据后调用，用于更新本地数据副本。
        这些数据用于构建观测空间和计算奖励。
        
        术语说明:
            - 观测空间 (Observation Space): 强化学习中智能体能够观察到的状态信息集合
            - 奖励 (Reward): 强化学习中用于评估动作好坏的标量信号
        
        使用示例:
            ```python
            # 从服务器获取最新状态后更新
            self.gym.update_data()  # 从服务器同步
            self.data.update_qpos_qvel_qacc(
                self.gym.data.qpos,
                self.gym.data.qvel,
                self.gym.data.qacc
            )
            ```
        """
        self.qpos = qpos
        self.qvel = qvel
        self.qacc = qacc

    def update_qfrc_bias(self, qfrc_bias):
        """
        更新关节偏置力数据
        
        术语说明:
            - 偏置力 (Bias Force): 包括重力、科里奥利力、离心力等被动力
            - 科里奥利力 (Coriolis Force): 由于物体在旋转参考系中运动产生的惯性力
            - 动力学计算: 根据力和力矩计算物体的加速度和运动状态
        
        使用示例:
            ```python
            # 更新偏置力（通常由服务器计算）
            self.data.update_qfrc_bias(self.gym.data.qfrc_bias)
            ```
        """
        self.qfrc_bias = qfrc_bias

