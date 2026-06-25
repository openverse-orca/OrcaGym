"""ModelRegistry — 模型注册与结构查询（骨架）。

本模块属于 OrcaGym Euler 体系骨架阶段（P2-Step1），负责构建
`OrcaGymModel`/`OrcaGymData` 并提供 body/equality 等模型信息查询
（架构 §5.5）。

骨架阶段 `ModelRegistry` 不执行真实模型构建，方法体 `raise
NotImplementedError`。P4 填充阶段将填入真实构建与查询逻辑。

注意:
    `build_orca_gym_data` 返回 `OrcaGymData`（原体系的状态容器），
    而非 `OrcaGymDataView`。`OrcaGymDataView` 由 `MuJoCoSimCore`
    的 `sync_to_view` 填充，不由此处构建。
"""

from typing import Tuple


class ModelRegistry:
    """模型注册与结构查询。

    构建 OrcaGymModel/OrcaGymData，提供 body/equality 等模型信息查询。

    使用契约:
        构建模型:   model = registry.build_orca_gym_model()
        查询结构:   mass = registry.body_subtree_mass("link1")

    禁止:
        不要通过本类直接访问 _mjModel.opt.*（用 SimConfig）。
    """

    def __init__(self, mj_model=None) -> None:
        """初始化模型注册器。

        Args:
            mj_model: MuJoCo 模型对象。骨架阶段不依赖真实 mjModel，
                仅存储引用供 P4 填充阶段使用。
        """
        self._mj_model = mj_model

    # --- 构建方法 ---

    def build_orca_gym_model(self):
        """构建 OrcaGymModel 实例（模型结构抽象，原样复用）。

        Returns:
            OrcaGymModel 实例。

        Raises:
            NotImplementedError: 骨架阶段未实现真实构建。
        """
        raise NotImplementedError("build_orca_gym_model 待 P4 填充")

    def build_orca_gym_data(self):
        """构建 OrcaGymData 实例（原体系的状态容器，非 DataView）。

        注意: 此处返回 OrcaGymData（原体系），不是 OrcaGymDataView。
        OrcaGymDataView 由 MuJoCoSimCore.sync_to_view 填充。

        Returns:
            OrcaGymData 实例。

        Raises:
            NotImplementedError: 骨架阶段未实现真实构建。
        """
        raise NotImplementedError("build_orca_gym_data 待 P4 填充")

    # --- 扩展查询方法（架构 §5.5，覆盖用户绕道访问的模型结构）---

    def body_subtree_mass(self, body_name: str) -> float:
        """查询 body 子树总质量。

        替代直接访问 _mjModel.body_subtreemass[id]。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        Returns:
            body 子树总质量（标量）。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("body_subtree_mass 待 P4 填充")

    def equality_data_width(self) -> int:
        """查询等式约束数据宽度（eq_data 每行元素数）。

        替代直接访问 _mjModel.eq_data.shape[1]。

        Returns:
            等式约束数据宽度。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("equality_data_width 待 P4 填充")

    def equality_object_ids(self, eq_idx: int) -> Tuple[int, int]:
        """查询等式约束关联的两个对象 id。

        替代直接访问 _mjModel.eq_obj1id[eq_idx] / eq_obj2id[eq_idx]。

        Args:
            eq_idx: 等式约束索引。

        Returns:
            (obj1_id, obj2_id) 元组。

        Raises:
            NotImplementedError: 骨架阶段未实现真实查询。
        """
        raise NotImplementedError("equality_object_ids 待 P4 填充")
