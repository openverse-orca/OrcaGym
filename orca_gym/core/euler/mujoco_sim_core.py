"""MuJoCoSimCore — MuJoCo 仿真核心（阶段二填充）。

本模块属于 OrcaGym Euler 体系阶段二（P4-Step1），持有 `_mjModel`/`_mjData`，
是这两个 MuJoCo 对象**唯一**的存放位置（架构 §5.3）。

阶段二填充真实 MuJoCo 操作（init/step/forward/set_ctrl/sync_to_view/
reset_data/set_qpos_qvel），力应用方法（apply_body_force/clear_*）留待完整 P4。

`_mjModel`/`_mjData` 不能作为公共属性被外部访问（由 Gym 层隔离机制保证）。
"""

import mujoco
import numpy as np

from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


class MuJoCoSimCore:
    """MuJoCo 仿真核心，持有 _mjModel/_mjData。

    _mjModel/_mjData 只存在于此类内部，不对外暴露。
    通过 sync_to_view() 将状态同步到 OrcaGymDataView。

    使用契约:
        初始化:     sim.init_simulation("model.xml")
        重置:       sim.reset_data()
        步进:       sim.step(nstep=1)
        前向:       sim.forward()
        设控制:     sim.set_ctrl(ctrl_array)
        设状态:     sim.set_qpos_qvel(qpos, qvel)
        读状态:     sim.sync_to_view(data_view)  # 同步到 DataView
        应用力:     sim.apply_body_force(body_id, force, torque)  # 待完整 P4
        清力:       sim.clear_body_force(body_id) / sim.clear_all_forces()  # 待完整 P4

    禁止:
        外部不应直接访问本类的 _mjModel/_mjData。
        读取状态 → env.data（OrcaGymDataView）
    """

    def __init__(self) -> None:
        """初始化仿真核心。

        _mjModel/_mjData 初始化为 None，待 init_simulation() 填充。
        """
        self._mjModel = None    # mujoco.MjModel | None
        self._mjData = None     # mujoco.MjData | None

    # --- 生命周期方法 ---

    def init_simulation(self, model_xml_path: str) -> None:
        """加载 MuJoCo 模型并初始化仿真。

        Args:
            model_xml_path: MuJoCo 模型 XML 文件路径。
        """
        self._mjModel = mujoco.MjModel.from_xml_path(model_xml_path)
        self._mjData = mujoco.MjData(self._mjModel)

    def reset_data(self) -> None:
        """重置 MjData 到初始状态（mj_resetData）。

        供 OrcaGymEulerEnv.reset_simulation 调用。
        """
        if self._mjModel is None or self._mjData is None:
            raise RuntimeError("Simulation not initialized")
        mujoco.mj_resetData(self._mjModel, self._mjData)

    def step(self, nstep: int) -> None:
        """执行 nstep 步 MuJoCo 仿真。

        Args:
            nstep: 步进次数。
        """
        mujoco.mj_step(self._mjModel, self._mjData, nstep)

    def forward(self) -> None:
        """执行 MuJoCo 前向计算（不步进，仅更新派生量）。"""
        mujoco.mj_forward(self._mjModel, self._mjData)

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置控制输入到 _mjData.ctrl。

        Args:
            ctrl: 控制输入数组。
        """
        self._mjData.ctrl[:] = ctrl

    def set_qpos_qvel(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """设置广义坐标和速度（供 set_joint_qpos/qvel 使用）。

        注意：调用后需调用 forward() 以更新派生量。

        Args:
            qpos: 广义坐标数组。
            qvel: 广义速度数组。
        """
        self._mjData.qpos[:] = qpos
        self._mjData.qvel[:] = qvel

    def sync_to_view(self, view: OrcaGymDataView) -> None:
        """将 _mjData 状态同步到 OrcaGymDataView。

        基本字段采用零拷贝视图赋值；body/site 查询由 DataView 按需读取。

        Args:
            view: 待填充的 OrcaGymDataView 实例。
        """
        view._sync_from_mjdata(self._mjData, self._mjModel)

    # --- 力应用方法（待完整 P4）---

    def apply_body_force(self, body_id: int, force: np.ndarray, torque: np.ndarray) -> None:
        """对指定 body 施加外力/力矩（写入 _mjData.xfrc_applied）。

        Args:
            body_id: MuJoCo body id。
            force: 力向量 (3,)。
            torque: 力矩向量 (3,)。

        Raises:
            NotImplementedError: 阶段二不实现外力应用，留待完整 P4。
        """
        raise NotImplementedError("apply_body_force 待完整 P4 填充")

    def clear_body_force(self, body_id: int) -> None:
        """清除指定 body 的外力（清零 _mjData.xfrc_applied[body_id]）。

        Args:
            body_id: MuJoCo body id。

        Raises:
            NotImplementedError: 阶段二不实现外力清除，留待完整 P4。
        """
        raise NotImplementedError("clear_body_force 待完整 P4 填充")

    def clear_all_forces(self) -> None:
        """清除所有 body 的外力（清零 _mjData.xfrc_applied 全数组）。

        Raises:
            NotImplementedError: 阶段二不实现外力清除，留待完整 P4。
        """
        raise NotImplementedError("clear_all_forces 待完整 P4 填充")

    # --- 维度 property ---

    @property
    def nq(self) -> int:
        """广义坐标维度（qpos 维度）。"""
        return self._mjModel.nq

    @property
    def nv(self) -> int:
        """广义速度维度（qvel 维度）。"""
        return self._mjModel.nv

    @property
    def nu(self) -> int:
        """控制输入维度（ctrl 维度）。"""
        return self._mjModel.nu
