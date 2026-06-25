"""OrcaGymDataView — MuJoCo 状态的完整只读视图（骨架）。

本模块属于 OrcaGym Euler 体系骨架阶段（P1-Step2），提供 MuJoCo 状态的
完整只读视图，替代直接访问 `_mjData`（架构 §5.7, §7.4）。

骨架阶段 DataView 不持有真实数据，字段初始化为空数组/空列表/默认值，
方法体 `raise NotImplementedError`。P4 填充阶段将填入真实查询逻辑。
"""

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

    # --- body 查询方法 ---

    def body_xpos(self, body_name: str) -> np.ndarray:
        """查询 body 的世界坐标位置 (3,)。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 的世界坐标位置数组，形状 (3,)。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("body_xpos 待 P4 填充")

    def body_xquat(self, body_name: str) -> np.ndarray:
        """查询 body 的世界坐标四元数 (4,)。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 的世界坐标四元数 [w, x, y, z]，形状 (4,)。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("body_xquat 待 P4 填充")

    def body_xmat(self, body_name: str) -> np.ndarray:
        """查询 body 的世界坐标旋转矩阵 (3, 3)。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 的世界坐标旋转矩阵，形状 (3, 3)。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("body_xmat 待 P4 填充")

    def body_cvel(self, body_name: str) -> np.ndarray:
        """查询 body 的空间速度 (6,)。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 的空间速度 [angular(3), linear(3)]，形状 (6,)。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("body_cvel 待 P4 填充")

    def body_subtree_mass(self, body_name: str) -> float:
        """查询 body 子树总质量。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 子树总质量（标量）。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("body_subtree_mass 待 P4 填充")

    # --- site 查询方法 ---

    def site_xpos(self, site_name: str) -> np.ndarray:
        """查询 site 的世界坐标位置 (3,)。

        Args:
            site_name: site 名称（已含 agent 前缀）。

        Returns:
            site 的世界坐标位置数组，形状 (3,)。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("site_xpos 待 P4 填充")

    def site_xmat(self, site_name: str) -> np.ndarray:
        """查询 site 的世界坐标旋转矩阵 (3, 3)。

        Args:
            site_name: site 名称（已含 agent 前缀）。

        Returns:
            site 的世界坐标旋转矩阵，形状 (3, 3)。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("site_xmat 待 P4 填充")

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
