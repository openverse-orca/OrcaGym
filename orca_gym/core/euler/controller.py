"""OrcaGym Euler 控制算法目录（GPU-native 控制 kernel + 用户句柄）。

本模块定义 device-side 控制 kernel（预编译 ``@flow.kernel``）与对应用户句柄类。
SingleWorld 封装原则（design §3.8 / guide §3.3）：``@flow.kernel``、device buffer、
``flow.array`` 全部封装于此，用户只通过 ``MuJoCoSimCoreEuler.register_pid_controller``
以 numpy/list/str 参数化配置，不接触任何 ``flow.*`` 对象。

未来扩展其它控制算法（微分滤波、自定义 PD 变体等）时，在本模块追加 ``@flow.kernel``
顶层函数与对应句柄类，并在 ``register_pid_controller`` 的 ``controller_type`` 分派处登记。

注意：本模块顶层 ``import orca.flow``，仅在 GPU(Euler) 后端调用路径被惰性导入
（``MuJoCoSimCoreEuler.register_pid_controller`` 内部 import），不参与 CPU-only 导入路径，
以维持 ``mujoco_sim_core_euler`` 在无 ``orca.flow`` 环境的可导入性。
"""

from __future__ import annotations

import numpy as np

import orca.flow as flow


@flow.kernel
def pd_kernel(
    q_target: flow.array2d(dtype=flow.float32),  # (1, nu)
    qpos: flow.array2d(dtype=flow.float32),  # (1, nq)
    qvel: flow.array2d(dtype=flow.float32),  # (1, nv)
    kp: flow.array(dtype=flow.float32),  # (nu,)
    kd: flow.array(dtype=flow.float32),  # (nu,)
    motor_limit: flow.array(dtype=flow.float32),  # (nu,)
    ctrl: flow.array2d(dtype=flow.float32),  # (1, nu)
    qpos_offset: int,
    qvel_offset: int,
):
    """GPU-native PD 力矩计算：tau = kp*(q_target - qpos) - kd*qvel，限幅写入 ctrl。

    与 ``07_locomotion.py`` 参考实现语义一致（guide §3.3.1），唯一差异是关节偏移
    ``qpos_offset``/``qvel_offset`` 由运行时 ``jnt_qposadr``/``jnt_dofadr`` 解析传入，
    而非硬编码自由基座的 7/6。

    依赖不变式：``ctrl`` 的执行器顺序与被驱动关节（``joint_names``）顺序一致；
    若 XML 中 actuator 顺序与关节顺序不一致，需额外构建 actuator→joint 索引映射（P1 暂不支持）。
    """
    dof_id = flow.tid()
    tau = kp[dof_id] * (q_target[0, dof_id] - qpos[0, qpos_offset + dof_id]) \
        - kd[dof_id] * qvel[0, qvel_offset + dof_id]
    ctrl[0, dof_id] = flow.clamp(tau, -motor_limit[dof_id], motor_limit[dof_id])


class PidController:
    """PD 控制器 device-side 句柄（numpy 入参，Flow 细节零泄露）。

    内部 device buffer（``q_target``/``kp``/``kd``/``motor_limit``，形状分别为
    ``(1, nu)``/``(nu,)``）全部封装为私有态；用户只通过 ``update_target`` /
    ``set_gains`` 两个纯 numpy 方法交互。预编译 ``pd_kernel`` 的注册信息与
    ``set_pre_step_kernel`` 编排细节不对外暴露。
    """

    def __init__(
        self,
        q_target_dev: "flow.array",
        kp_dev: "flow.array",
        kd_dev: "flow.array",
        motor_limit_dev: "flow.array",
    ) -> None:
        self._q_target = q_target_dev
        self._kp = kp_dev
        self._kd = kd_dev
        self._motor_limit = motor_limit_dev

    def update_target(self, q_target: np.ndarray) -> None:
        """H2D 上传目标关节角 ``q_target``（一次纯 H2D，形状 ``(nu,)``）。

        供控制循环每个周期更新一次目标（等价于 guide §8 step() 模式 3 中
        ``self._pid.update_target(action)``）。
        """
        self._q_target.assign(np.asarray(q_target, dtype=np.float32).reshape(1, -1))

    def set_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """H2D 更新 PD 增益 ``kp``/``kd``（形状 ``(nu,)``）。"""
        self._kp.assign(np.asarray(kp, dtype=np.float32))
        self._kd.assign(np.asarray(kd, dtype=np.float32))