import sys
import os
import grpc
import numpy as np

class OrcaGymData:
    def __init__(self, model):
        """
        动态状态容器（本地副本）。

        用途：
        - 保存仿真过程中的动态状态（时间、关节状态、偏置力等）。
        - 在 `update_data()`（backend 同步）后由上层调用更新到该容器，供观测构建、奖励/调试读取。

        字段与 shape（与 `model.nq/nv` 对齐）：
        - `time`: 标量仿真时间
        - `qpos`: `(nq,)` 广义坐标（关节位置/位姿参数）
        - `qvel`: `(nv,)` 广义速度
        - `qacc`: `(nv,)` 广义加速度
        - `qfrc_bias`: `(nv,)` 偏置力（如重力/科里奥利/离心等项）

        术语速查（面向首次接触 MuJoCo/Gym 的读者）：
        - `nq/nv`: MuJoCo 维度参数；`nq` 对应 `qpos` 长度，`nv` 对应 `qvel/qacc/qfrc_*` 长度
        - 广义坐标/速度/加速度：MuJoCo 的“关节空间”状态表示；不同关节类型在 `qpos` 中占用元素数不同
        - `qfrc_bias`: 使系统在当前状态下“维持/平衡”所需的被动力项（常含重力、科里奥利、离心等），用于动力学相关计算/调试

        注意：
        - 该对象用于“读”；写入通常由 `update_*` 方法完成（上层完成同步后调用）。
        - 读取数组用于长期保存/跨步比较时，建议在调用侧使用 `copy()`，避免后续同步覆盖。
        """
        self.model = model
        self.qpos = np.zeros(model.nq)
        self.qvel = np.zeros(model.nv)
        self.qacc = np.zeros(model.nv)
        self.qfrc_bias = np.zeros(model.nv)
        self.time = 0

    def update_qpos_qvel_qacc(self, qpos, qvel, qacc):
        """
        同步关节状态（qpos/qvel/qacc）。

        参数：
        - `qpos`: `(nq,)` 广义坐标
        - `qvel`: `(nv,)` 广义速度
        - `qacc`: `(nv,)` 广义加速度

        说明：
        - 通常在 backend 完成数据同步后调用，用于刷新本地副本。
        - 该方法仅更新字段，不做一致性计算；需要运动学/传感器派生量一致时，应由上层按既定链路调用 `mj_forward()` 等接口。
        """
        self.qpos = qpos
        self.qvel = qvel
        self.qacc = qacc

    def update_qfrc_bias(self, qfrc_bias):
        """
        同步偏置力 `qfrc_bias`。

        参数：
        - `qfrc_bias`: `(nv,)` 偏置力向量（如重力/科里奥利/离心等项）

        说明：
        - 通常由 backend 计算并在数据同步后刷新到本地副本。
        """
        self.qfrc_bias = qfrc_bias

