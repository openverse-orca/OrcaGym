"""MuJoCo 仿真核心。

持有 _mjModel/_mjData，执行纯 MuJoCo 操作，不对外暴露内部对象。
属于 OrcaGymEuler 体系的 P1 基础设施骨架组件。

参见 docs/design/architecture/orca_gym_euler_architecture.md 第 5.3 节。
"""

from __future__ import annotations

import numpy as np
import mujoco


class MuJoCoSimCore:
    """MuJoCo 仿真核心，持有 _mjModel/_mjData，不对外暴露。

    设计契约:
        - _mjModel/_mjData 只存在于本类内部，不作为公共属性暴露。
        - 提供步进、前向、控制设置、外力注入等原子操作。
        - sync_to_view() 在 P2 阶段实现，将内部状态同步到 OrcaGymDataView。

    使用示例:
        ```python
        sim = MuJoCoSimCore()
        sim.init_simulation("scene.xml")
        sim.set_ctrl(np.zeros(sim.nu))
        sim.step(5)
        ```
    """

    def __init__(self) -> None:
        self._mjModel: mujoco.MjModel | None = None
        self._mjData: mujoco.MjData | None = None

    def init_simulation(self, model_xml_path: str) -> None:
        """从 MJCF 文件加载模型并创建数据对象。

        Args:
            model_xml_path: MJCF 场景 XML 文件路径。
        """
        self._mjModel = mujoco.MjModel.from_xml_path(model_xml_path)
        self._mjData = mujoco.MjData(self._mjModel)

    @property
    def nq(self) -> int:
        """广义坐标维度（qpos 长度）。"""
        self._require_initialized()
        return self._mjModel.nq

    @property
    def nv(self) -> int:
        """自由度维度（qvel/qacc 长度）。"""
        self._require_initialized()
        return self._mjModel.nv

    @property
    def nu(self) -> int:
        """执行器数量（动作空间维度）。"""
        self._require_initialized()
        return self._mjModel.nu

    def step(self, nstep: int) -> None:
        """执行 nstep 次物理仿真步进。

        Args:
            nstep: 步进次数，通常为 1 或 frame_skip。
        """
        self._require_initialized()
        mujoco.mj_step(self._mjModel, self._mjData, nstep)

    def forward(self) -> None:
        """执行前向计算，更新所有派生状态（位置、速度、加速度、力等）。

        在设置关节状态、mocap 位置等操作后调用，确保派生量一致。
        """
        self._require_initialized()
        mujoco.mj_forward(self._mjModel, self._mjData)

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置执行器控制输入。

        Args:
            ctrl: 控制输入数组，形状 (nu,)。
        """
        self._require_initialized()
        self._mjData.ctrl[:] = ctrl

    def apply_body_force(
        self, body_id: int, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """在指定 body 上施加外力/力矩（累加到 xfrc_applied）。

        力和力矩在 world frame 中表示，直接写入 _mjData.xfrc_applied。
        多次调用会累加；调用 clear_body_force/clear_all_forces 清零。

        Args:
            body_id: Body 索引（mujoco body id）。
            force: 力向量 [fx, fy, fz]，world frame。
            torque: 力矩向量 [tx, ty, tz]，world frame。
        """
        self._require_initialized()
        self._mjData.xfrc_applied[body_id, :3] += force
        self._mjData.xfrc_applied[body_id, 3:] += torque

    def clear_body_force(self, body_id: int) -> None:
        """清零指定 body 的外力。

        Args:
            body_id: Body 索引。
        """
        self._require_initialized()
        self._mjData.xfrc_applied[body_id] = 0

    def clear_all_forces(self) -> None:
        """清零所有 body 的外力。"""
        self._require_initialized()
        self._mjData.xfrc_applied[:] = 0

    def sync_to_view(self, view) -> None:
        """将 _mjData 状态同步到 OrcaGymDataView。

        基本状态（qpos/qvel/qacc/qfrc_bias/time）使用 copy；
        xfrc_applied 使用只读视图（不 copy，共享内存）；
        body/site 派生状态使用视图。

        Args:
            view: OrcaGymDataView 实例。
        """
        self._require_initialized()
        d = self._mjData
        # 基本状态（copy，避免后续 step 覆盖）
        view._qpos = d.qpos.copy()
        view._qvel = d.qvel.copy()
        view._qacc = d.qacc.copy()
        view._qfrc_bias = d.qfrc_bias.copy()
        view._time = float(d.time)
        # 扩展字段
        view._xfrc_applied = d.xfrc_applied  # 只读视图，不 copy
        view._actuator_force = d.actuator_force.copy()
        view._contact = [d.contact[i] for i in range(d.ncon)]
        # body 派生状态（视图，共享内存）
        view._xpos = d.xpos
        view._xquat = d.xquat
        view._xmat = d.xmat
        view._cvel = d.cvel
        # site 派生状态（视图）
        view._site_xpos = d.site_xpos
        view._site_xmat = d.site_xmat

    def _require_initialized(self) -> None:
        if self._mjModel is None or self._mjData is None:
            raise RuntimeError("仿真未初始化，请先调用 init_simulation()")
