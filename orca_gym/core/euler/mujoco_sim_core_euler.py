"""MuJoCoSimCoreEuler — 单世界 Euler 后端编排层（P1，nworld=1）。

本模块属于 OrcaGym Euler 体系 P1，是 Euler 后端的仿真核心编排层，
与 CPU 版 `MuJoCoSimCore` 共享同一组方法签名（L1 对齐），内部把写操作
落到 ``solver.host``（lazy-dirty），读操作先 ``_ensure_host_fresh()``
（lazy-stale），GPU 推进（step/forward）前 ``_commit_if_dirty()``。

lazy 同步原语（_mark_dirty/_mark_stale/_commit_if_dirty/_ensure_host_fresh）
是本编排层的唯一同步入口，其余方法禁止直接调 flush/sync_to_host
（对齐 design §5.1「单一同步入口」约束）。

``_solver`` 通过字符串前向引用类型注释（`from __future__ import annotations`），
避免在 CPU-only 测试环境 import `orca.euler`；真正的 import 放在
`init_simulation` 内（见 Phase C1）。
"""

from __future__ import annotations

import warnings

import mujoco
import numpy as np

# ActorManipulator 拖拽代理（mocap anchor + 极轻 dummy 自由体）的地面安全停放高度。
_ACTOR_MANIPULATOR_SAFE_HEIGHT = 0.5  # m


def _actor_manipulator_body_ids(model: mujoco.MjModel) -> tuple[int, int]:
    """返回拖拽代理的 (anchor_id, dummy_id)，不存在则 (-1, -1)。

    匹配新旧两套命名：旧版 ``ActorManipulator_{Anchor,dummy}``，新版 UUID 化的
    ``ORCA_MANIPULATOR_<uuid>_{Anchor,dummy}``。
    """
    anchor_id = -1
    dummy_id = -1
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name is None:
            continue
        if name.startswith("ORCA_MANIPULATOR_") and name.endswith("_Anchor"):
            anchor_id = bid
        elif name.startswith("ORCA_MANIPULATOR_") and name.endswith("_dummy"):
            dummy_id = bid
    if anchor_id < 0 or dummy_id < 0:
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name is None:
                continue
            if name == "ActorManipulator_Anchor":
                anchor_id = bid
            elif name == "ActorManipulator_dummy":
                dummy_id = bid
    return anchor_id, dummy_id


def _park_actor_manipulator(model: mujoco.MjModel) -> int:
    """把拖拽代理停放到地面以上并关闭其 geom 碰撞，返回处理数量（0/1）。

    旧关卡导出的 ActorManipulator 代理被埋在 z=-1000（远低于 floor 平面 z=0），
    dummy 自由体（质量 ~4e-6 kg）因此深深穿透 floor；接触反力与指向 anchor 的
    软 weld 互相打架，在 GPU float32 下爆出 qacc ~1e6 并逐步累积为 NaN。
    将 anchor/dummy 移到地面以上（保持 x,y，仅抬升 z），并关闭代理 geom 的碰撞
    掩码（沿袭 CPU 版 AR-001），从根上消除接触与穿透。

    Args:
        model: 已加载的 host MjModel（就地修改）。

    Returns:
        处理的拖拽代理数量（0 或 1）。
    """
    anchor_id, dummy_id = _actor_manipulator_body_ids(model)
    if anchor_id < 0 or dummy_id < 0:
        return 0

    # 关闭代理 geom 碰撞掩码：拖拽只依赖 mocap weld，与接触无关。
    for bid in (anchor_id, dummy_id):
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid:
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0

    # 若停放在地面以下，则抬升到地面以上（保持 x,y，仅改 z）。
    x = float(model.body_pos[anchor_id][0])
    y = float(model.body_pos[anchor_id][1])
    z = float(model.body_pos[anchor_id][2])
    if z < 0.0:
        target = np.array([x, y, _ACTOR_MANIPULATOR_SAFE_HEIGHT], dtype=np.float64)
        # mocap anchor：body_pos 决定 reset 后的 mocap_pos。
        model.body_pos[anchor_id] = target
        # dummy 自由体：改 free joint 的初始位置（运行时位置来自 qpos0）。
        dummy_jnt = int(model.body_jntadr[dummy_id])
        if dummy_jnt >= 0:
            qadr = int(model.jnt_qposadr[dummy_jnt])
            model.qpos0[qadr:qadr + 3] = target
    return 1


class MuJoCoSimCoreEuler:
    """单世界编排层（nworld=1，对齐 design C3）。

    与 CPU 版 MuJoCoSimCore 共享同一组方法签名，内部把写操作落到
    ``solver.host``（lazy-dirty），读操作先 ``_ensure_host_fresh()``
    （lazy-stale），GPU 推进（step/forward）前 ``_commit_if_dirty()``。

    使用契约:
        初始化:     core.init_simulation("model.xml", device, nworld)
        重置:       core.reset_data()
        步进:       core.step(nstep=1)
        前向:       core.forward()
        设控制:     core.set_ctrl(ctrl_array)
        设状态:     core.set_qpos_qvel(qpos, qvel)
        读状态:     core.sync_to_view(data_view)

    禁止:
        外部不应直接访问本类的 _solver。
        其余方法不得直接调 flush/sync_to_host，统一走 lazy 同步原语。
    """

    def __init__(self) -> None:
        self._solver = None     # SolverMujocoSingleWorld | None
        self._nworld: int = 1
        self._host_dirty: bool = False
        self._host_stale: bool = True

    # ---- lazy 同步原语（内部，B2 交付）----

    def _mark_dirty(self) -> None:
        self._host_dirty = True

    def _mark_stale(self) -> None:
        self._host_stale = True

    def _commit_if_dirty(self) -> None:
        if self._host_dirty:
            self._solver.flush()
            self._host_dirty = False

    def _ensure_host_fresh(self) -> None:
        if self._host_stale:
            self._solver.sync_to_host()
            self._host_stale = False

    def _require_solver(self) -> None:
        """确保 solver 已初始化，否则抛 RuntimeError（与 CPU 版一致）。"""
        if self._solver is None:
            raise RuntimeError("Simulation not initialized")

    # ---- 维度 property（G 类，读 host model，无同步）----

    @property
    def nq(self) -> int:
        self._require_solver()
        return self._solver.mj_model.nq

    @property
    def nv(self) -> int:
        self._require_solver()
        return self._solver.mj_model.nv

    @property
    def nu(self) -> int:
        self._require_solver()
        return self._solver.mj_model.nu

    @property
    def mj_model(self):
        """返回 host MjModel（只读，供 SimConfig/ModelRegistry 绑定）。"""
        self._require_solver()
        return self._solver.mj_model

    # ---- A 类：生命周期方法（C1 交付）----

    def init_simulation(
        self, model_xml_path: str, device: str = "cuda", nworld: int = 1
    ) -> None:
        if nworld != 1:
            raise NotImplementedError(
                "P1 仅支持 nworld=1（多世界留给 MuJoCoSimCoreEulerMultiWorlds）。"
            )
        try:
            import orca.euler as euler
        except ImportError as e:
            raise RuntimeError(
                "Euler 后端不可用：orca.euler 未安装。"
                "请确认已安装 orca.euler 且 SimConfig.backend = SimBackend.EULER。"
            ) from e
        self._solver = euler.SolverMujocoSingleWorld(
            source=self._prepare_model(model_xml_path), device=device
        )
        self._nworld = nworld
        self._host_dirty = False
        self._host_stale = False   # 构造后 host 与 GPU 均为初始态，视为 fresh

    @staticmethod
    def _prepare_model(model_xml_path: str) -> "mujoco.MjModel":
        """加载 host MjModel 并做 GPU 后端所需兼容性预处理。

        两处预处理：

        1. **降级 no-slip 求解器**：mujoco_flow（fork 自 mujoco_warp 后端）上游
           未实现 no-slip 后处理，``put_model`` 对 ``noslip_iterations > 0`` 抛
           ``NotImplementedError``。因 OrcaGym 接入的 MJCF 由外部渲染端经 gRPC
           提供（可能启用 no-slip），此处清零 noslip_iterations 保证 GPU 后端可
           运行，代价是 GPU 结果与 CPU no-slip 配置存在近似差异。

        2. **停放 ActorManipulator 拖拽代理**：旧关卡导出的代理被埋在 z=-1000
           （远低于 floor 平面），导致极轻 dummy 自由体穿透 floor、接触反力与软
           weld 打架，在 float32 下爆出 qacc ~1e6 并累积为 NaN（见
           ``_park_actor_manipulator``）。

        Args:
            model_xml_path: MuJoCo 模型 XML 文件路径。

        Returns:
            host MjModel（已按需清零 noslip_iterations 并停放拖拽代理）。
        """
        model = mujoco.MjModel.from_xml_path(model_xml_path)
        if model.opt.noslip_iterations > 0:
            warnings.warn(
                f"GPU 后端不支持 no-slip 求解器，noslip_iterations 由 "
                f"{model.opt.noslip_iterations} 降级为 0（模型: {model_xml_path}）。"
                "GPU 物理结果与 CPU 的 no-slip 配置存在近似差异。",
                stacklevel=2,
            )
            model.opt.noslip_iterations = 0
        if _park_actor_manipulator(model) > 0:
            warnings.warn(
                "检测到 ActorManipulator 拖拽代理（mocap anchor + dummy 自由体）"
                "埋于地面以下，已将其停放到地面以上并关闭代理 geom 碰撞掩码，"
                "以消除 GPU float32 下的数值发散。",
                stacklevel=2,
            )
        return model

    def reset_data(self) -> None:
        self._require_solver()
        self._solver.reset()       # GPU reset + host 重置
        self._host_dirty = False
        self._host_stale = False

    def step(self, nstep: int) -> None:
        self._require_solver()
        self._commit_if_dirty()    # H2D：让写操作进入 step 前生效
        self._solver.step(nstep)
        self._mark_stale()         # 步进后 host 过期

    def forward(self) -> None:
        self._require_solver()
        self._commit_if_dirty()
        self._solver.forward()
        self._mark_stale()

    def sync_to_view(self, view) -> None:
        self._require_solver()
        self._ensure_host_fresh()      # D2H：世界 0（nworld=1）
        view._sync_from_mjdata(self._solver.host, self._solver.mj_model)  # noqa: SLF001

    # ---- B 类：状态写入方法（C2 交付）----

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self._require_solver()
        self._solver.host.ctrl[:] = ctrl
        self._mark_dirty()

    def set_qpos_qvel(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self._require_solver()
        host = self._solver.host
        host.qpos[:] = qpos
        host.qvel[:] = qvel
        self._mark_dirty()

    def set_mocap_pos_and_quat(self, mocap_dict: dict[str, dict]) -> None:
        self._require_solver()
        model = self._solver.mj_model
        host = self._solver.host
        for body_name, pose in mocap_dict.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id >= 0:
                host.mocap_pos[mocap_id] = np.asarray(
                    pose["pos"], dtype=np.float64
                ).reshape(3)
                host.mocap_quat[mocap_id] = np.asarray(
                    pose["quat"], dtype=np.float64
                ).reshape(4)
        self._mark_dirty()

    # ---- C 类：力应用方法（C3 交付）----

    def apply_body_force(self, body_id, force: np.ndarray, torque: np.ndarray) -> None:
        self._require_solver()
        # SolverMujocoSingleWorld 封装 body_id name 查询 + 写 host.xfrc_applied
        self._solver.apply_body_force(body_id, force, torque)
        self._mark_dirty()

    def clear_body_force(self, body_id: int) -> None:
        self._require_solver()
        self._solver.host.xfrc_applied[body_id, :6] = 0.0
        self._mark_dirty()

    def clear_all_forces(self) -> None:
        self._require_solver()
        self._solver.host.xfrc_applied[:] = 0.0
        self._mark_dirty()

    def mj_apply_force_at_site(self, site_id: int, force: np.ndarray, torque: np.ndarray) -> None:
        self._require_solver()
        host = self._solver.host
        body_id = self._solver.mj_model.site_bodyid[site_id]
        host.xfrc_applied[body_id, :3] += np.asarray(force, dtype=np.float64).reshape(3)
        host.xfrc_applied[body_id, 3:6] += np.asarray(torque, dtype=np.float64).reshape(3)
        self._mark_dirty()

    def mj_clear_xfrc_applied_for_site(self, site_id: int) -> None:
        self._require_solver()
        self.clear_body_force(int(self._solver.mj_model.site_bodyid[site_id]))

    # ---- D 类：状态查询方法（D1 交付）----
    # 18 个查询方法与 CPU 版逐字一致，仅前置 _require_solver/_ensure_host_fresh，
    # 且 _mjModel/_mjData 替换为 _solver.mj_model/_solver.host。

    def _joint_id(self, joint_name: str) -> int:
        """解析关节名称到 joint_id（内部辅助，无同步）。"""
        return mujoco.mj_name2id(
            self._solver.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )

    def _joint_qpos_len(self, joint_id: int) -> int:
        """关节 qpos 长度（按类型：free=7, ball=4, hinge/slide=1）。"""
        jtype = self._solver.mj_model.jnt_type[joint_id]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            return 7
        if jtype == mujoco.mjtJoint.mjJNT_BALL:
            return 4
        return 1  # HINGE / SLIDE

    def _joint_qvel_len(self, joint_id: int) -> int:
        """关节 qvel/qacc 长度（按类型：free=6, ball=3, hinge/slide=1）。"""
        jtype = self._solver.mj_model.jnt_type[joint_id]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            return 6
        if jtype == mujoco.mjtJoint.mjJNT_BALL:
            return 3
        return 1  # HINGE / SLIDE

    def query_joint_qpos(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        self._require_solver()
        self._ensure_host_fresh()
        model = self._solver.mj_model
        host = self._solver.host
        result: dict[str, np.ndarray] = {}
        for name in joint_names:
            jid = self._joint_id(name)
            adr = int(model.jnt_qposadr[jid])
            n = self._joint_qpos_len(jid)
            result[name] = host.qpos[adr:adr + n]
        return result

    def query_joint_qvel(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        self._require_solver()
        self._ensure_host_fresh()
        model = self._solver.mj_model
        host = self._solver.host
        result: dict[str, np.ndarray] = {}
        for name in joint_names:
            jid = self._joint_id(name)
            adr = int(model.jnt_dofadr[jid])
            n = self._joint_qvel_len(jid)
            result[name] = host.qvel[adr:adr + n]
        return result

    def query_joint_qacc(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        self._require_solver()
        self._ensure_host_fresh()
        model = self._solver.mj_model
        host = self._solver.host
        result: dict[str, np.ndarray] = {}
        for name in joint_names:
            jid = self._joint_id(name)
            adr = int(model.jnt_dofadr[jid])
            n = self._joint_qvel_len(jid)
            result[name] = host.qacc[adr:adr + n]
        return result

    def query_joint_offsets(
        self, joint_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_solver()
        self._ensure_host_fresh()
        model = self._solver.mj_model
        qpos_adrs = []
        qvel_adrs = []
        qacc_adrs = []
        for name in joint_names:
            jid = self._joint_id(name)
            qpos_adrs.append(int(model.jnt_qposadr[jid]))
            dof_adr = int(model.jnt_dofadr[jid])
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
        self._require_solver()
        self._ensure_host_fresh()
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
        self._require_solver()
        self._ensure_host_fresh()
        return {
            name: int(self._solver.mj_model.jnt_dofadr[self._joint_id(name)])
            for name in joint_names
        }

    def jnt_qposadr(self, joint_name: str) -> int:
        self._require_solver()
        self._ensure_host_fresh()
        return int(self._solver.mj_model.jnt_qposadr[self._joint_id(joint_name)])

    def jnt_dofadr(self, joint_name: str) -> int:
        self._require_solver()
        self._ensure_host_fresh()
        return int(self._solver.mj_model.jnt_dofadr[self._joint_id(joint_name)])

    def _body_id(self, body_name: str) -> int:
        """解析 body 名称到 body_id（内部辅助，无同步）。"""
        return mujoco.mj_name2id(
            self._solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )

    def _site_id(self, site_name: str) -> int:
        """解析 site 名称到 site_id（内部辅助，无同步）。"""
        return mujoco.mj_name2id(
            self._solver.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )

    def query_body_xpos_xmat_xquat(
        self, body_name_list: list[str]
    ) -> dict[str, dict]:
        self._require_solver()
        self._ensure_host_fresh()
        host = self._solver.host
        result: dict[str, dict] = {}
        for name in body_name_list:
            bid = self._body_id(name)
            result[name] = {
                "xpos": host.xpos[bid],
                "xmat": host.xmat[bid].reshape(3, 3),
                "xquat": host.xquat[bid],
            }
        return result

    def query_body_xpos_xmat_xquat_xvel(
        self, body_name_list: list[str]
    ) -> dict[str, dict]:
        self._require_solver()
        self._ensure_host_fresh()
        host = self._solver.host
        nv = self._solver.mj_model.nv
        result: dict[str, dict] = {}
        for name in body_name_list:
            bid = self._body_id(name)
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            self.mj_jacBody(jacp, jacr, bid)
            xvel = jacp @ host.qvel
            result[name] = {
                "xpos": host.xpos[bid],
                "xmat": host.xmat[bid].reshape(3, 3),
                "xquat": host.xquat[bid],
                "xvel": xvel,
            }
        return result

    def query_site_pos_and_mat(self, site_names: list[str]) -> dict[str, dict]:
        self._require_solver()
        self._ensure_host_fresh()
        host = self._solver.host
        result: dict[str, dict] = {}
        for name in site_names:
            sid = self._site_id(name)
            result[name] = {
                "xpos": host.site_xpos[sid],
                "xmat": host.site_xmat[sid].reshape(3, 3),
            }
        return result

    def query_site_size(self, site_names: list[str]) -> dict[str, np.ndarray]:
        self._require_solver()
        self._ensure_host_fresh()
        result: dict[str, np.ndarray] = {}
        for name in site_names:
            sid = self._site_id(name)
            result[name] = self._solver.mj_model.site_size[sid]
        return result

    def _sensor_id(self, sensor_name: str) -> int:
        """解析传感器名称到 sensor_id（内部辅助，无同步）。"""
        return mujoco.mj_name2id(
            self._solver.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )

    def _actuator_id(self, actuator_name: str) -> int:
        """解析执行器名称到 actuator_id（内部辅助，无同步）。"""
        return mujoco.mj_name2id(
            self._solver.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )

    def _geom_id(self, geom_name: str) -> int:
        """解析 geom 名称到 geom_id（内部辅助，无同步）。"""
        return mujoco.mj_name2id(
            self._solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )

    def query_sensor_data(
        self, sensor_names: list[str], sensor_info: dict
    ) -> dict[str, np.ndarray]:
        self._require_solver()
        self._ensure_host_fresh()
        model = self._solver.mj_model
        host = self._solver.host
        result: dict[str, np.ndarray] = {}
        for name in sensor_names:
            info = sensor_info.get(name) if sensor_info else None
            if info is not None and "adr" in info and "dim" in info:
                adr = int(info["adr"])
                dim = int(info["dim"])
            else:
                sid = self._sensor_id(name)
                adr = int(model.sensor_adr[sid])
                dim = int(model.sensor_dim[sid])
            result[name] = host.sensordata[adr:adr + dim]
        return result

    def query_actuator_torques(self, actuator_names: list[str]) -> dict[str, np.ndarray]:
        self._require_solver()
        self._ensure_host_fresh()
        model = self._solver.mj_model
        host = self._solver.host
        result: dict[str, np.ndarray] = {}
        for name in actuator_names:
            aid = self._actuator_id(name)
            adr = aid
            dim = int(model.actuator_ctrlsz[aid]) if hasattr(
                model, "actuator_ctrlsz"
            ) else 1
            result[name] = host.actuator_force[adr:adr + max(dim, 1)]
        return result

    def query_contact_simple(self) -> list[dict]:
        self._require_solver()
        self._ensure_host_fresh()
        host = self._solver.host
        contacts: list[dict] = []
        ncon = host.ncon
        for i in range(ncon):
            con = host.contact[i]
            contacts.append({
                "geom1": int(con.geom1),
                "geom2": int(con.geom2),
                "dist": float(con.dist),
                "pos": host.contact[i].pos,
                "frame": host.contact[i].frame,
            })
        return contacts

    def query_contact_force(self, contact_ids: list[int]) -> dict[int, np.ndarray]:
        self._require_solver()
        self._ensure_host_fresh()
        result: dict[int, np.ndarray] = {}
        for cid in contact_ids:
            force = np.zeros(6)
            mujoco.mj_contactForce(self._solver.mj_model, self._solver.host, cid, force)
            result[cid] = force
        return result

    def get_cfrc_ext(self) -> np.ndarray:
        self._require_solver()
        self._ensure_host_fresh()
        return self._solver.host.cfrc_ext

    def get_goal_bounding_box(self, geom_name: str) -> np.ndarray:
        self._require_solver()
        self._ensure_host_fresh()
        gid = self._geom_id(geom_name)
        return self._solver.mj_model.geom_size[gid]

    # ---- E 类：雅可比计算方法（E1 交付）----

    def mj_jacBody(
        self, jacp: np.ndarray, jacr: np.ndarray, body_id: int
    ) -> None:
        """计算 body 雅可比（mujoco.mj_jacBody，原地写 jacp/jacr）。"""
        self._require_solver()
        self._ensure_host_fresh()
        mujoco.mj_jacBody(
            self._solver.mj_model, self._solver.host, jacp, jacr, body_id
        )

    def mj_jacSite(
        self, jacp: np.ndarray, jacr: np.ndarray, site_id: int
    ) -> None:
        """计算 site 雅可比（mujoco.mj_jacSite，原地写 jacp/jacr）。"""
        self._require_solver()
        self._ensure_host_fresh()
        mujoco.mj_jacSite(
            self._solver.mj_model, self._solver.host, jacp, jacr, site_id
        )

    def mj_jac_site(self, site_names: list[str]) -> dict[str, dict]:
        """批量计算 site 雅可比（循环 mj_jacSite）。"""
        self._require_solver()
        self._ensure_host_fresh()
        result: dict[str, dict] = {}
        nv = self._solver.mj_model.nv
        for site_name in site_names:
            sid = self._site_id(site_name)
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacSite(
                self._solver.mj_model, self._solver.host, jacp, jacr, sid
            )
            result[site_name] = {"jacp": jacp, "jacr": jacr}
        return result

    # ---- F 类：模型参数写入方法（F1 交付，P1 抛错）----

    def _not_implemented_p2(self, method: str) -> None:
        raise NotImplementedError(
            f"MuJoCoSimCoreEuler.{method} 在 P1 未实现：Euler 后端修改模型常量 "
            "需引入 override_model + mjf_model 重建（design §4.3 P2 策略）。"
        )

    def set_geom_friction(self, geom_friction_dict: dict[str, np.ndarray]) -> None:
        self._not_implemented_p2("set_geom_friction")

    def add_extra_weight(self, weight_load_dict: dict) -> None:
        self._not_implemented_p2("add_extra_weight")

    def update_equality_constraints(self, eq_list: list[dict]) -> None:
        self._not_implemented_p2("update_equality_constraints")

    def modify_equality_objects(
        self,
        eq_ids: list[int],
        obj1_ids=None,
        obj2_ids=None,
    ) -> None:
        self._not_implemented_p2("modify_equality_objects")

    def set_equality_active(self, eq_idx: int, active: bool) -> None:
        self._not_implemented_p2("set_equality_active")

    def set_equality_solref(self, eq_idx: int, solref: np.ndarray) -> None:
        self._not_implemented_p2("set_equality_solref")

    def set_equality_solimp(self, eq_idx: int, solimp: np.ndarray) -> None:
        self._not_implemented_p2("set_equality_solimp")