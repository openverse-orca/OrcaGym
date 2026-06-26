"""OrcaGymDataView — MuJoCo 状态的完整只读视图（阶段二填充）。

本模块属于 OrcaGym Euler 体系阶段二（P4-Step1/Step4），提供 MuJoCo 状态的
完整只读视图，替代直接访问 `_mjData`（架构 §5.7, §7.4）。

阶段二 Step 1 填充 `_sync_from_mjdata`（基本字段零拷贝同步），供
MuJoCoSimCore.sync_to_view 调用。阶段二 Step 4 填充 body/site 查询方法
（按需通过 `_mj_data`/`_mj_model` 读取）。
"""

import mujoco
import numpy as np


class OrcaGymDataView:
    """MuJoCo 状态的完整只读视图。

    替代直接访问 _mjData。所有字段在 update_data() 后保证一致。
    用户永远不需要访问 _mjData。

    使用契约:
        读取状态:   env.data.qpos / env.data.body_xpos("link1")
        写入状态:   env.set_joint_qpos() / env.apply_body_force()

    禁止:
        不要通过 env._gym._sim._mjData 绕道访问。
        缺少字段时，扩展本类，不要绕道。
    """

    def __init__(self) -> None:
        # --- 基本状态（架构 §5.7，原 OrcaGymData 已有）---
        self.qpos: np.ndarray = np.array([])
        self.qvel: np.ndarray = np.array([])
        self.qacc: np.ndarray = np.array([])
        self.qfrc_bias: np.ndarray = np.array([])
        self.time: float = 0.0

        # --- 扩展字段（架构 §5.7，覆盖用户绕道访问的字段）---
        self.xfrc_applied: np.ndarray = np.array([])
        self.actuator_force: np.ndarray = np.array([])
        self.contact: list = []

        # --- 扩展字段（阶段三 3.1.4）---
        self.cfrc_ext: np.ndarray = np.array([])

        # --- 内部引用（不对外暴露，由 _sync_from_mjdata 设置）---
        self._mj_data = None
        self._mj_model = None

    # --- 内部同步方法（供 MuJoCoSimCore.sync_to_view 调用）---

    def _sync_from_mjdata(self, mj_data, mj_model) -> None:
        """从 MjData 同步基本字段（零拷贝视图）。

        供 MuJoCoSimCore.sync_to_view 调用。同步后 qpos/qvel 等字段为
        MjData 对应数组的视图（非拷贝），读取 env.data.qpos 直接反映
        _mjData 当前状态。body/site 查询方法通过 _mj_data/_mj_model
        按需读取。

        Args:
            mj_data: MuJoCo MjData 对象。
            mj_model: MuJoCo MjModel 对象。
        """
        self._mj_data = mj_data
        self._mj_model = mj_model
        self.qpos = mj_data.qpos
        self.qvel = mj_data.qvel
        self.qacc = mj_data.qacc
        self.qfrc_bias = mj_data.qfrc_bias
        self.time = float(mj_data.time)
        self.xfrc_applied = mj_data.xfrc_applied
        self.actuator_force = mj_data.actuator_force
        self.contact = mj_data.contact
        self.cfrc_ext = mj_data.cfrc_ext

    # --- body 查询方法 ---

    def body_xpos(self, body_name: str) -> np.ndarray:
        """查询 body 的世界坐标位置 (3,)。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 的世界坐标位置数组，形状 (3,)。
        """
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self._mj_data.body(body_id).xpos

    def body_xquat(self, body_name: str) -> np.ndarray:
        """查询 body 的世界坐标四元数 (4,)。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 的世界坐标四元数 [w, x, y, z]，形状 (4,)。
        """
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self._mj_data.body(body_id).xquat

    def body_xmat(self, body_name: str) -> np.ndarray:
        """查询 body 的世界坐标旋转矩阵 (9,)。

        MuJoCo 以扁平数组存储旋转矩阵（行优先），如需 (3, 3) 可调用
        ``.reshape(3, 3)``。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 的世界坐标旋转矩阵（扁平存储），形状 (9,)。
        """
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self._mj_data.body(body_id).xmat

    def body_cvel(self, body_name: str) -> np.ndarray:
        """查询 body 的空间速度 (6,)。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 的空间速度 [angular(3), linear(3)]，形状 (6,)。
        """
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self._mj_data.cvel[body_id]

    def body_subtree_mass(self, body_name: str) -> float:
        """查询 body 子树总质量。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 子树总质量（标量）。
        """
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return float(self._mj_model.body_subtreemass[body_id])

    # --- site 查询方法 ---

    def site_xpos(self, site_name: str) -> np.ndarray:
        """查询 site 的世界坐标位置 (3,)。

        Args:
            site_name: site 名称（已含 agent 前缀）。

        Returns:
            site 的世界坐标位置数组，形状 (3,)。
        """
        site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self._mj_data.site(site_id).xpos

    def site_xmat(self, site_name: str) -> np.ndarray:
        """查询 site 的世界坐标旋转矩阵 (9,)。

        MuJoCo 以扁平数组存储旋转矩阵（行优先），如需 (3, 3) 可调用
        ``.reshape(3, 3)``。

        Args:
            site_name: site 名称（已含 agent 前缀）。

        Returns:
            site 的世界坐标旋转矩阵（扁平存储），形状 (9,)。
        """
        site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self._mj_data.site(site_id).xmat

    # --- geom 查询方法（阶段三 3.1.4）---

    def _geom_id(self, geom_name: str) -> int:
        """解析 geom 名称到 geom_id（内部辅助）。"""
        return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    def geom_xpos(self, geom_name: str) -> np.ndarray:
        """查询 geom 的世界坐标位置 (3,)。

        Args:
            geom_name: geom 名称。

        Returns:
            geom 的世界坐标位置，形状 (3,)，为 _mj_data 的零拷贝视图。
        """
        gid = self._geom_id(geom_name)
        xpos = self._mj_data.geom_xpos[gid]
        return xpos

    def geom_xmat(self, geom_name: str) -> np.ndarray:
        """查询 geom 的世界坐标旋转矩阵 (9,)。

        MuJoCo 以扁平数组存储旋转矩阵（行优先），如需 (3, 3) 可调用
        ``.reshape(3, 3)``。

        Args:
            geom_name: geom 名称。

        Returns:
            geom 的世界坐标旋转矩阵（扁平存储），形状 (9,)，为 _mj_data 的零拷贝视图。
        """
        gid = self._geom_id(geom_name)
        xmat = self._mj_data.geom_xmat[gid]
        return xmat

    def geom_size(self, geom_name: str) -> np.ndarray:
        """查询 geom 的尺寸 (3,)。

        Args:
            geom_name: geom 名称。

        Returns:
            geom 的尺寸，形状 (3,)，为 _mj_model 的零拷贝视图。
            对于 box geom，值为半尺寸 (hx, hy, hz)。
        """
        gid = self._geom_id(geom_name)
        size = self._mj_model.geom_size[gid]
        return size

    # --- M3: __getattr__ 兜底（架构 §7.4）---

    def __getattr__(self, name: str):
        """兜底：访问不存在的字段时引导扩展，而非绕道访问 _mjData。

        __getattr__ 仅在常规属性查找失败时触发，因此已定义的字段
        (qpos/qvel/qacc/qfrc_bias/time/xfrc_applied/actuator_force/contact)
        不会进入此分支。

        Args:
            name: 被访问的属性名。

        Raises:
            AttributeError: 带引导文本，列出当前可用字段和方法，
                引导用户在 OrcaGymDataView 中扩展而非绕道。
        """
        raise AttributeError(
            f"'OrcaGymDataView' 没有字段 '{name}'。\n"
            f"  当前可用字段: {list(self.__dict__.keys())}\n"
            f"  当前可用方法: body_xpos, body_xquat, body_xmat, "
            f"body_cvel, body_subtree_mass, site_xpos, site_xmat, ...\n"
            f"  如果需要 '{name}'，请在 OrcaGymDataView 中添加该字段或方法。"
        )
