"""OrcaGymDataView — MuJoCo 状态的完整只读视图。

替代直接访问 _mjData，提供基本状态字段和 body/site 属性查询。
通过 __getattr__ 兜底，缺字段时引导扩展。

属于 OrcaGymEuler 体系的 P2 状态视图与配置组件。
参见 docs/design/architecture/orca_gym_euler_architecture.md 第 5.7 节。
"""

from __future__ import annotations

import numpy as np

from orca_gym.core.orca_gym_model import OrcaGymModel


class OrcaGymDataView:
    """MuJoCo 状态的完整只读视图，替代直接访问 _mjData。

    设计契约:
        - 所有字段在 sync_to_view() 后保证一致。
        - 用户永远不需要访问 _mjData。
        - xfrc_applied 是只读视图（写入用 apply_body_force）。
        - 缺字段时 __getattr__ 引导扩展，不绕道 _mjData。

    使用示例:
        ```python
        qpos = view.qpos
        body_pos = view.body_xpos("link1")
        site_pos = view.site_xpos("tip")
        ```
    """

    def __init__(self, model: OrcaGymModel) -> None:
        self._model = model
        # 基本状态（由 sync_to_view 填充）
        self._qpos: np.ndarray = np.zeros(model.nq)
        self._qvel: np.ndarray = np.zeros(model.nv)
        self._qacc: np.ndarray = np.zeros(model.nv)
        self._qfrc_bias: np.ndarray = np.zeros(model.nv)
        self._time: float = 0.0
        # 扩展字段
        self._xfrc_applied: np.ndarray | None = None
        self._actuator_force: np.ndarray | None = None
        self._contact: list = []
        # body 派生状态
        self._xpos: np.ndarray | None = None
        self._xquat: np.ndarray | None = None
        self._xmat: np.ndarray | None = None
        self._cvel: np.ndarray | None = None
        # site 派生状态
        self._site_xpos: np.ndarray | None = None
        self._site_xmat: np.ndarray | None = None

    # --- 基本状态（只读属性）---

    @property
    def qpos(self) -> np.ndarray:
        return self._qpos

    @property
    def qvel(self) -> np.ndarray:
        return self._qvel

    @property
    def qacc(self) -> np.ndarray:
        return self._qacc

    @property
    def qfrc_bias(self) -> np.ndarray:
        return self._qfrc_bias

    @property
    def time(self) -> float:
        return self._time

    @property
    def xfrc_applied(self) -> np.ndarray:
        """外力数组（只读视图，写入用 apply_body_force）。"""
        return self._xfrc_applied

    @property
    def actuator_force(self) -> np.ndarray:
        return self._actuator_force

    @property
    def contact(self) -> list:
        return self._contact

    # --- body 属性查询（按名称）---

    def _body_id(self, body_name: str) -> int:
        return self._model.body_name2id(body_name)

    def body_xpos(self, body_name: str) -> np.ndarray:
        """body 世界坐标位置 [x, y, z]。"""
        return self._xpos[self._body_id(body_name)]

    def body_xquat(self, body_name: str) -> np.ndarray:
        """body 世界坐标四元数 [w, x, y, z]。"""
        return self._xquat[self._body_id(body_name)]

    def body_xmat(self, body_name: str) -> np.ndarray:
        """body 世界坐标旋转矩阵（3x3，展平为 9 元素）。"""
        return self._xmat[self._body_id(body_name)]

    def body_cvel(self, body_name: str) -> np.ndarray:
        """body 质心速度 [vx, vy, vz, wx, wy, wz]。"""
        return self._cvel[self._body_id(body_name)]

    def body_subtree_mass(self, body_name: str) -> float:
        """body 子树总质量（静态模型属性）。"""
        return float(self._model.get_body_byname(body_name)["SubtreeMass"])

    # --- site 属性查询（按名称）---

    def _site_id(self, site_name: str) -> int:
        return self._model.site_name2id(site_name)

    def site_xpos(self, site_name: str) -> np.ndarray:
        """site 世界坐标位置 [x, y, z]。"""
        return self._site_xpos[self._site_id(site_name)]

    def site_xmat(self, site_name: str) -> np.ndarray:
        """site 世界坐标旋转矩阵（3x3，展平为 9 元素）。"""
        return self._site_xmat[self._site_id(site_name)]

    # --- 兜底引导 ---

    def __getattr__(self, name: str):
        # 仅当正常属性查找失败时触发。
        raise AttributeError(
            f"'OrcaGymDataView' 没有字段 '{name}'。\n"
            f"  当前可用字段: qpos, qvel, qacc, qfrc_bias, time, "
            f"xfrc_applied, actuator_force, contact\n"
            f"  当前可用方法: body_xpos, body_xquat, body_xmat, body_cvel, "
            f"body_subtree_mass, site_xpos, site_xmat\n"
            f"  如果需要 '{name}'，请在 OrcaGymDataView 中添加该字段或方法。"
        )

    def __dir__(self):
        return [
            "qpos", "qvel", "qacc", "qfrc_bias", "time",
            "xfrc_applied", "actuator_force", "contact",
            "body_xpos", "body_xquat", "body_xmat", "body_cvel",
            "body_subtree_mass", "site_xpos", "site_xmat",
        ]
