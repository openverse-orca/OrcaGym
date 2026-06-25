"""MuJoCoSimCore — MuJoCo 仿真核心（骨架）。

本模块属于 OrcaGym Euler 体系骨架阶段（P2-Step2），持有 `_mjModel`/`_mjData`，
是这两个 MuJoCo 对象**唯一**的存放位置（架构 §5.3）。

骨架阶段不执行真实 MuJoCo 操作，但属性定义和方法签名必须完整，
且 `_mjModel`/`_mjData` 不能作为公共属性被外部访问（由 Gym 层隔离机制保证）。
方法体 `raise NotImplementedError`，P4 填充阶段将填入真实仿真逻辑。
"""

import numpy as np

from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


class MuJoCoSimCore:
    """MuJoCo 仿真核心，持有 _mjModel/_mjData。

    _mjModel/_mjData 只存在于此类内部，不对外暴露。
    通过 sync_to_view() 将状态同步到 OrcaGymDataView。

    使用契约:
        初始化:     sim.init_simulation("model.xml")
        步进:       sim.step(nstep=1)
        前向:       sim.forward()
        设控制:     sim.set_ctrl(ctrl_array)
        读状态:     sim.sync_to_view(data_view)  # 同步到 DataView
        应用力:     sim.apply_body_force(body_id, force, torque)
        清力:       sim.clear_body_force(body_id) / sim.clear_all_forces()

    禁止:
        外部不应直接访问本类的 _mjModel/_mjData。
        读取状态 → env.data（OrcaGymDataView）
    """

    def __init__(self) -> None:
        """初始化仿真核心。

        _mjModel/_mjData 初始化为 None，待 init_simulation() 填充。
        骨架阶段不执行真实 MuJoCo 初始化。
        """
        self._mjModel = None    # mujoco.MjModel | None
        self._mjData = None     # mujoco.MjData | None

    # --- 生命周期方法 ---

    def init_simulation(self, model_xml_path: str) -> None:
        """加载 MuJoCo 模型并初始化仿真。

        Args:
            model_xml_path: MuJoCo 模型 XML 文件路径。

        Raises:
            NotImplementedError: 骨架阶段未实现真实初始化。
        """
        raise NotImplementedError("init_simulation 待 P4 填充")

    def step(self, nstep: int) -> None:
        """执行 nstep 步 MuJoCo 仿真。

        Args:
            nstep: 步进次数。

        Raises:
            NotImplementedError: 骨架阶段未实现真实步进。
        """
        raise NotImplementedError("step 待 P4 填充")

    def forward(self) -> None:
        """执行 MuJoCo 前向计算（不步进，仅更新派生量）。

        Raises:
            NotImplementedError: 骨架阶段未实现真实前向计算。
        """
        raise NotImplementedError("forward 待 P4 填充")

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置控制输入到 _mjData.ctrl。

        Args:
            ctrl: 控制输入数组。

        Raises:
            NotImplementedError: 骨架阶段未实现真实控制设置。
        """
        raise NotImplementedError("set_ctrl 待 P4 填充")

    def sync_to_view(self, view: OrcaGymDataView) -> None:
        """将 _mjData 状态同步到 OrcaGymDataView。

        Args:
            view: 待填充的 OrcaGymDataView 实例。

        Raises:
            NotImplementedError: 骨架阶段未实现真实同步。
        """
        raise NotImplementedError("sync_to_view 待 P4 填充")

    # --- 力应用方法 ---

    def apply_body_force(self, body_id: int, force: np.ndarray, torque: np.ndarray) -> None:
        """对指定 body 施加外力/力矩（写入 _mjData.xfrc_applied）。

        Args:
            body_id: MuJoCo body id。
            force: 力向量 (3,)。
            torque: 力矩向量 (3,)。

        Raises:
            NotImplementedError: 骨架阶段未实现真实力应用。
        """
        raise NotImplementedError("apply_body_force 待 P4 填充")

    def clear_body_force(self, body_id: int) -> None:
        """清除指定 body 的外力（清零 _mjData.xfrc_applied[body_id]）。

        Args:
            body_id: MuJoCo body id。

        Raises:
            NotImplementedError: 骨架阶段未实现真实力清除。
        """
        raise NotImplementedError("clear_body_force 待 P4 填充")

    def clear_all_forces(self) -> None:
        """清除所有 body 的外力（清零 _mjData.xfrc_applied 全数组）。

        Raises:
            NotImplementedError: 骨架阶段未实现真实力清除。
        """
        raise NotImplementedError("clear_all_forces 待 P4 填充")

    # --- 维度 property ---

    @property
    def nq(self) -> int:
        """广义坐标维度（qpos 维度）。

        Returns:
            nq 维度。

        Raises:
            NotImplementedError: 骨架阶段未持有真实 mjModel。
        """
        raise NotImplementedError("nq 待 P4 填充")

    @property
    def nv(self) -> int:
        """广义速度维度（qvel 维度）。

        Returns:
            nv 维度。

        Raises:
            NotImplementedError: 骨架阶段未持有真实 mjModel。
        """
        raise NotImplementedError("nv 待 P4 填充")

    @property
    def nu(self) -> int:
        """控制输入维度（ctrl 维度）。

        Returns:
            nu 维度。

        Raises:
            NotImplementedError: 骨架阶段未持有真实 mjModel。
        """
        raise NotImplementedError("nu 待 P4 填充")
