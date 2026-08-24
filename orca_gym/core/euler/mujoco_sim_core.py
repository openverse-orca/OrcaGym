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


_ACTOR_MANIPULATOR_BODY_NAMES = (
    "ORCA_MANIPULATOR_a3f5e2d1-7b8c-4f2a-9e6d-1c2b3a4f5d6e_Anchor",
    "ORCA_MANIPULATOR_a3f5e2d1-7b8c-4f2a-9e6d-1c2b3a4f5d6e_dummy",
    "ActorManipulator_Anchor",
    "ActorManipulator_dummy",
)


def disable_actor_manipulator_collision(model: mujoco.MjModel) -> int:
    """
    关闭 ActorManipulator 拖拽代理所有几何体的碰撞掩码（contype=conaffinity=0）。

    拖拽/抓取依赖 mocap anchor 与 weld 等号约束，与接触完全无关，故关闭碰撞不影响功能；
    但可消除 AR-001 中该代理埋地/远端碰撞体与无限平面贯产生的垃圾约束（junk efc）行。

    Args:
        model: 原始 mujoco.MjModel。

    Returns:
        被修改的 geom 数量（用于日志/断言）。
    """
    n_disabled = 0
    for body_name in _ACTOR_MANIPULATOR_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body_id:
                if model.geom_contype[gid] != 0 or model.geom_conaffinity[gid] != 0:
                    model.geom_contype[gid] = 0
                    model.geom_conaffinity[gid] = 0
                    n_disabled += 1
    return n_disabled


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
        # AR-001：模型加载后关闭 ActorManipulator 拖拽代理的碰撞掩码，
        # 消除其埋地/远端碰撞体与无限平面贯产生的垃圾约束行（不影响 mocap weld 拖拽）。
        self.disable_actor_manipulator_collision()

    def disable_actor_manipulator_collision(self) -> int:
        """关闭 ActorManipulator 拖拽代理几何体的碰撞掩码（可随时重试/重断言）。

        Returns:
            本次被修改的 geom 数量。
        """
        if self._mjModel is None:
            return 0
        return disable_actor_manipulator_collision(self._mjModel)

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
        view._sync_from_mjdata(self._mjData, self._mjModel)  # noqa: SLF001  core 层组件协作：SimCore 填充 DataView
        # --- 临时诊断：打印 CPU 同步的 qpos，与 GPU 后端对比渲染差异 ---
        _q = np.asarray(self._mjData.qpos, dtype=np.float64)
        _max = float(np.max(np.abs(_q))) if _q.size else 0.0
        print(
            f"[CPU sync_to_view] t={self._mjData.time:.4f} nq={_q.size} "
            f"finite={bool(np.isfinite(_q).all())} max|qpos|={_max:.4g} "
            f"qpos={np.round(_q, 4).tolist()}"
        )

    # --- 力应用方法（待完整 P4）---

    def apply_body_force(self, body_id: int, force: np.ndarray, torque: np.ndarray) -> None:
        """对指定 body 施加外力/力矩（写入 _mjData.xfrc_applied）。

        Args:
            body_id: MuJoCo body id。
            force: 力向量 (3,)。
            torque: 力矩向量 (3,)。
        """
        f = np.asarray(force, dtype=np.float64).reshape(3)
        tau = np.asarray(torque, dtype=np.float64).reshape(3)
        self._mjData.xfrc_applied[body_id, :3] = f
        self._mjData.xfrc_applied[body_id, 3:6] = tau

    def clear_body_force(self, body_id: int) -> None:
        """清除指定 body 的外力（清零 _mjData.xfrc_applied[body_id]）。

        Args:
            body_id: MuJoCo body id。
        """
        self._mjData.xfrc_applied[body_id, :6] = 0.0

    def clear_all_forces(self) -> None:
        """清除所有 body 的外力（清零 _mjData.xfrc_applied 全数组）。"""
        self._mjData.xfrc_applied[:] = 0.0

    def mj_apply_force_at_site(
        self, site_id: int, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """在 site 处施加力（等价 mujoco.mj_applyForce，写 xfrc_applied[site.bodyid]）。

        Args:
            site_id: MuJoCo site id。
            force: 力向量 (3,)（世界坐标系）。
            torque: 力矩向量 (3,)（世界坐标系）。
        """
        f = np.asarray(force, dtype=np.float64).reshape(3)
        tau = np.asarray(torque, dtype=np.float64).reshape(3)
        body_id = self._mjModel.site_bodyid[site_id]
        self._mjData.xfrc_applied[body_id, :3] += f
        self._mjData.xfrc_applied[body_id, 3:6] += tau

    def mj_clear_xfrc_applied_for_site(self, site_id: int) -> None:
        """清除 site 关联 body 的 xfrc。

        Args:
            site_id: MuJoCo site id。
        """
        body_id = self._mjModel.site_bodyid[site_id]
        self.clear_body_force(body_id)

    # --- 状态设置方法（阶段三 3.2.2）---

    def set_mocap_pos_and_quat(self, mocap_dict: dict[str, dict]) -> None:
        """设置 mocap body 位置/四元数（写 mocap_pos/mocap_quat）。

        Args:
            mocap_dict: dict[body_name -> {"pos": (3,), "quat": (4,) [w,x,y,z]}]。
                        body_name 必须是 mocap body（mocapid >= 0）。
        """
        for body_name, pose in mocap_dict.items():
            body_id = mujoco.mj_name2id(
                self._mjModel, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            mocap_id = int(self._mjModel.body_mocapid[body_id])
            if mocap_id >= 0:
                self._mjData.mocap_pos[mocap_id] = np.asarray(
                    pose["pos"], dtype=np.float64
                ).reshape(3)
                self._mjData.mocap_quat[mocap_id] = np.asarray(
                    pose["quat"], dtype=np.float64
                ).reshape(4)

    def set_geom_friction(self, geom_friction_dict: dict[str, np.ndarray]) -> None:
        """设置 geom 摩擦系数（写 _mjModel.geom_friction）。

        Args:
            geom_friction_dict: dict[geom_name -> friction (3,) [sliding, torsion, rolling]]。
        """
        for geom_name, friction in geom_friction_dict.items():
            geom_id = mujoco.mj_name2id(
                self._mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            self._mjModel.geom_friction[geom_id] = np.asarray(
                friction, dtype=np.float64
            ).reshape(3)

    def add_extra_weight(self, weight_load_dict: dict) -> None:
        """为 body 添加额外重量（修改 body_mass/body_inertia）。

        Args:
            weight_load_dict: dict[body_name -> weight (float, kg)]。
        """
        for body_name, weight in weight_load_dict.items():
            body_id = mujoco.mj_name2id(
                self._mjModel, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            self._mjModel.body_mass[body_id] += float(weight)
            # 简化惯性：按球体 I = 2/5 m r^2，r 取当前等价半径
            # 实际项目按需重算，此处仅同步 mass（保持质心/惯量张量不变）

    # --- 关节查询（阶段三 3.1.1）---

    def _joint_id(self, joint_name: str) -> int:
        """解析关节名称到 joint_id（内部辅助）。"""
        return mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    def _joint_qpos_len(self, joint_id: int) -> int:
        """关节 qpos 长度（按类型：free=7, ball=4, hinge/slide=1）。"""
        jtype = self._mjModel.jnt_type[joint_id]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            return 7
        if jtype == mujoco.mjtJoint.mjJNT_BALL:
            return 4
        return 1  # HINGE / SLIDE

    def _joint_qvel_len(self, joint_id: int) -> int:
        """关节 qvel/qacc 长度（按类型：free=6, ball=3, hinge/slide=1）。"""
        jtype = self._mjModel.jnt_type[joint_id]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            return 6
        if jtype == mujoco.mjtJoint.mjJNT_BALL:
            return 3
        return 1  # HINGE / SLIDE

    def query_joint_qpos(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qpos（按关节类型切片）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> qpos 切片 np.ndarray]。切片为 _mjData.qpos 的
            视图（零拷贝），长度按关节类型（free=7, ball=4, hinge/slide=1）。
        """
        result: dict[str, np.ndarray] = {}
        for name in joint_names:
            jid = self._joint_id(name)
            adr = int(self._mjModel.jnt_qposadr[jid])
            n = self._joint_qpos_len(jid)
            result[name] = self._mjData.qpos[adr:adr + n]
        return result

    def query_joint_qvel(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qvel（按 dof 偏移切片）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> qvel 切片 np.ndarray]。切片为 _mjData.qvel 的
            视图（零拷贝），长度按关节类型（free=6, ball=3, hinge/slide=1）。
        """
        result: dict[str, np.ndarray] = {}
        for name in joint_names:
            jid = self._joint_id(name)
            adr = int(self._mjModel.jnt_dofadr[jid])
            n = self._joint_qvel_len(jid)
            result[name] = self._mjData.qvel[adr:adr + n]
        return result

    def query_joint_qacc(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qacc（按 dof 偏移切片，qacc 与 qvel 共享 dof 地址）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> qacc 切片 np.ndarray]。切片为 _mjData.qacc 的
            视图（零拷贝），长度与 qvel 相同。
        """
        result: dict[str, np.ndarray] = {}
        for name in joint_names:
            jid = self._joint_id(name)
            adr = int(self._mjModel.jnt_dofadr[jid])
            n = self._joint_qvel_len(jid)
            result[name] = self._mjData.qacc[adr:adr + n]
        return result

    def query_joint_offsets(
        self, joint_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """查询关节 qpos/qvel/qacc 偏移量。

        Args:
            joint_names: 关节名称列表。

        Returns:
            (qpos_offsets, qvel_offsets, qacc_offsets) 三个 np.ndarray，
            长度等于 joint_names。qacc 偏移与 qvel 相同（共享 dof 地址）。
        """
        qpos_adrs = []
        qvel_adrs = []
        qacc_adrs = []
        for name in joint_names:
            jid = self._joint_id(name)
            qpos_adrs.append(int(self._mjModel.jnt_qposadr[jid]))
            dof_adr = int(self._mjModel.jnt_dofadr[jid])
            qvel_adrs.append(dof_adr)
            qacc_adrs.append(dof_adr)
        return (
            np.array(qpos_adrs, dtype=int),
            np.array(qvel_adrs, dtype=int),
            np.array(qacc_adrs, dtype=int),
        )

    def query_joint_lengths(
        self, joint_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """查询关节 qpos/qvel/qacc 长度。

        Args:
            joint_names: 关节名称列表。

        Returns:
            (qpos_lengths, qvel_lengths, qacc_lengths) 三个 np.ndarray，
            长度等于 joint_names。qacc 长度与 qvel 相同。
        """
        qpos_lens = []
        qvel_lens = []
        qacc_lens = []
        for name in joint_names:
            jid = self._joint_id(name)
            qpos_lens.append(self._joint_qpos_len(jid))
            n_qvel = self._joint_qvel_len(jid)
            qvel_lens.append(n_qvel)
            qacc_lens.append(n_qvel)
        return (
            np.array(qpos_lens, dtype=int),
            np.array(qvel_lens, dtype=int),
            np.array(qacc_lens, dtype=int),
        )

    def query_joint_dofadrs(self, joint_names: list[str]) -> dict[str, int]:
        """查询关节 dof 起始地址（jnt_dofadr）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> dof 起始地址 int]。
        """
        return {
            name: int(self._mjModel.jnt_dofadr[self._joint_id(name)])
            for name in joint_names
        }

    def jnt_qposadr(self, joint_name: str) -> int:
        """查询关节 qpos 起始地址。

        Args:
            joint_name: 关节名称。

        Returns:
            qpos 起始地址（int，非 numpy 标量）。
        """
        return int(self._mjModel.jnt_qposadr[self._joint_id(joint_name)])

    def jnt_dofadr(self, joint_name: str) -> int:
        """查询关节 qvel 起始地址（dof 地址）。

        Args:
            joint_name: 关节名称。

        Returns:
            qvel 起始地址（int，非 numpy 标量）。
        """
        return int(self._mjModel.jnt_dofadr[self._joint_id(joint_name)])

    # --- Body/Site 查询（阶段三 3.1.2）---

    def _body_id(self, body_name: str) -> int:
        """解析 body 名称到 body_id（内部辅助）。"""
        return mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_BODY, body_name)

    def _site_id(self, site_name: str) -> int:
        """解析 site 名称到 site_id（内部辅助）。"""
        return mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_SITE, site_name)

    def query_body_xpos_xmat_xquat(
        self, body_name_list: list[str]
    ) -> dict[str, dict]:
        """查询 body 位姿（xpos/xmat/xquat）。

        Args:
            body_name_list: body 名称列表。

        Returns:
            dict[body_name -> {"xpos": np.ndarray(3,),
                               "xmat": np.ndarray(3,3),
                               "xquat": np.ndarray(4,)}]。
            所有数组为 _mjData 的零拷贝视图。
        """
        result: dict[str, dict] = {}
        for name in body_name_list:
            bid = self._body_id(name)
            result[name] = {
                "xpos": self._mjData.xpos[bid],
                "xmat": self._mjData.xmat[bid].reshape(3, 3),
                "xquat": self._mjData.xquat[bid],
            }
        return result

    def query_body_xpos_xmat_xquat_xvel(
        self, body_name_list: list[str]
    ) -> dict[str, dict]:
        """查询 body 位姿 + 世界系线速度（mj_jacBody @ qvel）。

        通过 SimCore ``mj_jacBody`` 封装（阶段三 3.3.1 已实现）计算世界系线速度。

        Args:
            body_name_list: body 名称列表。

        Returns:
            dict[body_name -> {"xpos": np.ndarray(3,),
                               "xmat": np.ndarray(3,3),
                               "xquat": np.ndarray(4,),
                               "xvel": np.ndarray(3,)}]。
            ``xpos``/``xmat``/``xquat`` 为零拷贝视图；``xvel`` 为新建数组
            （jac @ qvel 计算结果，非视图）。
        """
        result: dict[str, dict] = {}
        nv = self._mjModel.nv
        for name in body_name_list:
            bid = self._body_id(name)
            # 世界系线速度 = jacp @ qvel（body 原点平移雅可比）
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            self.mj_jacBody(jacp, jacr, bid)
            xvel = jacp @ self._mjData.qvel
            result[name] = {
                "xpos": self._mjData.xpos[bid],
                "xmat": self._mjData.xmat[bid].reshape(3, 3),
                "xquat": self._mjData.xquat[bid],
                "xvel": xvel,
            }
        return result

    def query_site_pos_and_mat(self, site_names: list[str]) -> dict[str, dict]:
        """查询 site xpos/xmat。

        Args:
            site_names: site 名称列表。

        Returns:
            dict[site_name -> {"xpos": np.ndarray(3,),
                               "xmat": np.ndarray(3,3,)}]。
            所有数组为 _mjData 的零拷贝视图。
        """
        result: dict[str, dict] = {}
        for name in site_names:
            sid = self._site_id(name)
            result[name] = {
                "xpos": self._mjData.site_xpos[sid],
                "xmat": self._mjData.site_xmat[sid].reshape(3, 3),
            }
        return result

    def query_site_size(self, site_names: list[str]) -> dict[str, np.ndarray]:
        """查询 site 尺寸。

        Args:
            site_names: site 名称列表。

        Returns:
            dict[site_name -> site_size np.ndarray(3,)]。
            数组为 _mjModel.site_size 的零拷贝视图。
        """
        result: dict[str, np.ndarray] = {}
        for name in site_names:
            sid = self._site_id(name)
            result[name] = self._mjModel.site_size[sid]
        return result

    # --- 传感器/执行器/接触/Geom 查询（阶段三 3.1.3）---

    def _sensor_id(self, sensor_name: str) -> int:
        """解析传感器名称到 sensor_id（内部辅助）。"""
        return mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)

    def _actuator_id(self, actuator_name: str) -> int:
        """解析执行器名称到 actuator_id（内部辅助）。"""
        return mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

    def _geom_id(self, geom_name: str) -> int:
        """解析 geom 名称到 geom_id（内部辅助）。"""
        return mujoco.mj_name2id(self._mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    def query_sensor_data(
        self, sensor_names: list[str], sensor_info: dict
    ) -> dict[str, np.ndarray]:
        """查询传感器数据（按 adr/dim 切片 sensordata）。

        ``sensor_info`` 由 ``OrcaGymModel`` 提供（含每个传感器的 adr/dim），
        SimCore 不持有 ``OrcaGymModel``（解耦）。若 ``sensor_info`` 缺少某传感器
        条目，则回退到从 ``_mjModel`` 直接读取 adr/dim。

        Args:
            sensor_names: 传感器名称列表。
            sensor_info: 传感器元信息 dict，键为传感器名，值为含 ``adr``/``dim``
                键的 dict（来自 ``OrcaGymModel``）。

        Returns:
            dict[sensor_name -> sensordata 切片 np.ndarray]。切片为
            _mjData.sensordata 的零拷贝视图。
        """
        result: dict[str, np.ndarray] = {}
        for name in sensor_names:
            # 优先使用 sensor_info，缺失则回退到 _mjModel
            info = sensor_info.get(name) if sensor_info else None
            if info is not None and "adr" in info and "dim" in info:
                adr = int(info["adr"])
                dim = int(info["dim"])
            else:
                sid = self._sensor_id(name)
                adr = int(self._mjModel.sensor_adr[sid])
                dim = int(self._mjModel.sensor_dim[sid])
            result[name] = self._mjData.sensordata[adr:adr + dim]
        return result

    def query_actuator_torques(self, actuator_names: list[str]) -> dict[str, np.ndarray]:
        """查询执行器力矩（actuator_force 切片）。

        Args:
            actuator_names: 执行器名称列表。

        Returns:
            dict[actuator_name -> actuator_force 切片 np.ndarray]。切片为
            _mjData.actuator_force 的零拷贝视图（长度为该执行器的控制维度，
            通常为 1）。
        """
        result: dict[str, np.ndarray] = {}
        for name in actuator_names:
            aid = self._actuator_id(name)
            # actuator_force 索引为 actuator_id（每个执行器通常 1 维）
            adr = aid
            dim = int(self._mjModel.actuator_ctrlsz[aid]) if hasattr(
                self._mjModel, "actuator_ctrlsz"
            ) else 1
            # actuator_force 与 ctrl 共享布局，每执行器通常 1 维
            result[name] = self._mjData.actuator_force[adr:adr + max(dim, 1)]
        return result

    def query_contact_simple(self) -> list[dict]:
        """查询简单接触信息（遍历 contact 列表）。

        Returns:
            list[dict]，每个 dict 描述一个接触，含键：
            ``geom1`` (int)、``geom2`` (int)、``dist`` (float)、
            ``pos`` (np.ndarray(3,))、``frame`` (np.ndarray(9,) 扁平)。
            无接触时返回空列表。
        """
        contacts: list[dict] = []
        ncon = self._mjData.ncon
        for i in range(ncon):
            con = self._mjData.contact[i]
            contacts.append({
                "geom1": int(con.geom1),
                "geom2": int(con.geom2),
                "dist": float(con.dist),
                "pos": self._mjData.contact[i].pos,
                "frame": self._mjData.contact[i].frame,
            })
        return contacts

    def query_contact_force(self, contact_ids: list[int]) -> dict[int, np.ndarray]:
        """查询接触力（mj_contactForce）。

        Args:
            contact_ids: 接触索引列表（对应 _mjData.contact 数组下标）。

        Returns:
            dict[contact_id -> force np.ndarray(6,)]，前 3 分量为接触力，
            后 3 分量为接触力矩（由 ``mujoco.mj_contactForce`` 计算）。
        """
        result: dict[int, np.ndarray] = {}
        for cid in contact_ids:
            force = np.zeros(6)
            mujoco.mj_contactForce(self._mjModel, self._mjData, cid, force)
            result[cid] = force
        return result

    def get_cfrc_ext(self) -> np.ndarray:
        """查询外部约束力（cfrc_ext）。

        MuJoCo spatial vector 布局为 [torque(3), force(3)]，即
        ``[mx, my, mz, fx, fy, fz]``。线性力在 ``cfrc[bid, 3:]``，
        力矩在 ``cfrc[bid, :3]``。力/力矩均在以 subtree com 为原点的
        全局坐标系中（坐标轴与世界系对齐）。

        注意：需调用 ``mj_rnePostConstraint`` 后才有效（mj_step 默认不计算）。

        Returns:
            np.ndarray，形状 (nbody, 6)，为 _mjData.cfrc_ext 的零拷贝视图。
        """
        cfrc = self._mjData.cfrc_ext
        return cfrc

    def get_goal_bounding_box(self, geom_name: str) -> np.ndarray:
        """查询 geom 尺寸（bounding box）。

        Args:
            geom_name: geom 名称。

        Returns:
            np.ndarray(3,)，为 _mjModel.geom_size[geom_id] 的零拷贝视图。
            对于 box geom，值为半尺寸 (hx, hy, hz)。
        """
        gid = self._geom_id(geom_name)
        size = self._mjModel.geom_size[gid]
        return size

    # --- 雅可比计算（阶段三 3.3.1）---

    def mj_jacBody(
        self, jacp: np.ndarray, jacr: np.ndarray, body_id: int
    ) -> None:
        """计算 body 雅可比（mujoco.mj_jacBody，原地写 jacp/jacr）。

        Args:
            jacp: 平移雅可比矩阵，形状 (3, nv)，调用方预分配。
            jacr: 旋转雅可比矩阵，形状 (3, nv)，调用方预分配。
            body_id: MuJoCo body id。
        """
        mujoco.mj_jacBody(self._mjModel, self._mjData, jacp, jacr, body_id)

    def mj_jacSite(
        self, jacp: np.ndarray, jacr: np.ndarray, site_id: int
    ) -> None:
        """计算 site 雅可比（mujoco.mj_jacSite，原地写 jacp/jacr）。

        Args:
            jacp: 平移雅可比矩阵，形状 (3, nv)，调用方预分配。
            jacr: 旋转雅可比矩阵，形状 (3, nv)，调用方预分配。
            site_id: MuJoCo site id。
        """
        mujoco.mj_jacSite(self._mjModel, self._mjData, jacp, jacr, site_id)

    def mj_jac_site(self, site_names: list[str]) -> dict[str, dict]:
        """批量计算 site 雅可比（循环 mj_jacSite）。

        Args:
            site_names: site 名称列表。

        Returns:
            dict[site_name -> {"jacp": np.ndarray(3, nv),
                               "jacr": np.ndarray(3, nv)}]。
            jacp/jacr 为新建数组（每 site 独立分配，非视图）。
        """
        result: dict[str, dict] = {}
        nv = self._mjModel.nv
        for site_name in site_names:
            sid = self._site_id(site_name)
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacSite(self._mjModel, self._mjData, jacp, jacr, sid)
            result[site_name] = {"jacp": jacp, "jacr": jacr}
        return result

    # --- 等式约束（阶段三 3.5.1）---

    def update_equality_constraints(self, eq_list: list[dict]) -> None:
        """更新等式约束（写 _mjModel.eq_type/eq_obj1id/eq_obj2id/eq_data）。

        按 (obj1_id, obj2_id) 匹配槽位写入，对齐 OrcaGymLocal 语义。
        匹配失败时抛出 ValueError，避免硬编码索引破坏其他约束。

        Args:
            eq_list: 等式约束列表，每项为 dict，含键：
                - type: mjtEq 类型常量（如 mjEQ_CONNECT/WELD）。
                - obj1_id: 关联对象 1 的 id（用于匹配槽位）。
                - obj2_id: 关联对象 2 的 id（用于匹配槽位）。
                - data: 约束数据 np.ndarray（形状 (mjNEQDATA,)）。
                - new_obj1_id: 可选，匹配成功后写入的新 obj1 id。
                - new_obj2_id: 可选，匹配成功后写入的新 obj2 id。
        """
        model = self._mjModel
        for eq in eq_list:
            obj1_id = eq["obj1_id"]
            obj2_id = eq["obj2_id"]
            matched = False
            for i in range(model.neq):
                if (int(model.eq_obj1id[i]) == obj1_id and
                        int(model.eq_obj2id[i]) == obj2_id):
                    model.eq_type[i] = eq["type"]
                    # 支持可选的 obj id 变更（anchor 时改 obj2_id 指向 actor）
                    if "new_obj1_id" in eq:
                        model.eq_obj1id[i] = eq["new_obj1_id"]
                    if "new_obj2_id" in eq:
                        model.eq_obj2id[i] = eq["new_obj2_id"]
                    model.eq_data[i] = eq["data"]
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"未找到 (obj1_id={obj1_id}, obj2_id={obj2_id}) "
                    f"匹配的等式约束槽位，请检查 XML 中是否预定义了对应的 "
                    f"<equality><weld/connect body1=... body2=.../></equality>"
                )

    def modify_equality_objects(
        self,
        eq_ids: list[int],
        obj1_ids=None,
        obj2_ids=None,
    ) -> None:
        """修改等式约束关联对象（改 eq_obj1id/eq_obj2id）。

        Args:
            eq_ids: 等式约束索引列表。
            obj1_ids: 新的 obj1 id 列表（None 表示不修改）。
            obj2_ids: 新的 obj2 id 列表（None 表示不修改）。
        """
        model = self._mjModel
        for i, eq_id in enumerate(eq_ids):
            if obj1_ids is not None:
                model.eq_obj1id[eq_id] = obj1_ids[i]
            if obj2_ids is not None:
                model.eq_obj2id[eq_id] = obj2_ids[i]

    def set_equality_active(self, eq_idx: int, active: bool) -> None:
        """设置等式约束初始激活状态（写 _mjModel.eq_active0）。

        active/solref/solimp 无"按 (obj1_id, obj2_id) 匹配"语义，按 slot 索引
        直接写入，与 update_equality_constraints 的匹配写入区分。

        Args:
            eq_idx: 等式约束索引。
            active: 是否激活。
        """
        self._mjModel.eq_active0[eq_idx] = bool(active)

    def set_equality_solref(self, eq_idx: int, solref: np.ndarray) -> None:
        """设置等式约束 solver reference 参数（写 _mjModel.eq_solref）。

        Args:
            eq_idx: 等式约束索引。
            solref: solver reference 参数，形状 (2,)。
        """
        self._mjModel.eq_solref[eq_idx] = solref

    def set_equality_solimp(self, eq_idx: int, solimp: np.ndarray) -> None:
        """设置等式约束 solver impedance 参数（写 _mjModel.eq_solimp）。

        Args:
            eq_idx: 等式约束索引。
            solimp: solver impedance 参数，形状 (mjNEQIMP,) = (5,)，
                即 [d0, dWidth, dMid, dEnd, width]。
        """
        self._mjModel.eq_solimp[eq_idx] = solimp

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
