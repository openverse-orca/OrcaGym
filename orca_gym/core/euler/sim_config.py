"""SimConfig — MuJoCo 求解器参数配置。

替代直接访问 _mjModel.opt.*，提供 typed 的读写接口。
修改在下次 mj_step 时生效。

属于 OrcaGymEuler 体系的 P2 状态视图与配置组件。
参见 docs/design/architecture/orca_gym_euler_architecture.md 第 5.6 节。
"""

from __future__ import annotations

import mujoco


class SimConfig:
    """MuJoCo 求解器参数配置，替代 _mjModel.opt.* 直接访问。

    设计契约:
        - 覆盖 _mjModel.opt 的所有用户可访问字段。
        - 读写直接代理到 _mjModel.opt，修改在下次 mj_step 时生效。
        - 支持 load_from_dict() 批量设置和 to_dict() 导出。

    使用示例:
        ```python
        config = SimConfig(mj_model)
        config.timestep = 0.002
        config.load_from_dict({"integrator": 0, "iterations": 100})
        ```
    """

    # _mjModel.opt 的所有用户可访问字段（与 mujoco MjOption 结构对齐）
    _OPT_FIELDS = frozenset({
        "timestep", "impratio", "tolerance", "ls_tolerance",
        "noslip_tolerance", "ccd_tolerance",
        "gravity", "wind", "magnetic", "density", "viscosity",
        "o_margin", "o_solref", "o_solimp", "o_friction",
        "integrator", "cone", "jacobian", "solver",
        "iterations", "ls_iterations", "noslip_iterations", "ccd_iterations",
        "disableflags", "enableflags", "disableactuator",
        "sdf_initpoints", "sdf_iterations", "sleep_tolerance",
    })

    def __init__(self, mj_model: mujoco.MjModel) -> None:
        object.__setattr__(self, "_mj_model", mj_model)

    def __getattr__(self, name: str):
        # 仅当正常属性查找失败时触发。_mj_model 等真实属性不会进入此处。
        if name in self._OPT_FIELDS:
            return getattr(self._mj_model.opt, name)
        if name == "filterparent":
            return not (self.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
        raise AttributeError(
            f"'{type(self).__name__}' 没有配置项 '{name}'。\n"
            f"  可用配置项: {sorted(self._OPT_FIELDS)}\n"
            f"  如果需要新的配置项，请在 SimConfig._OPT_FIELDS 中添加。"
        )

    def __setattr__(self, name: str, value) -> None:
        if name in self._OPT_FIELDS:
            setattr(self._mj_model.opt, name, value)
        else:
            object.__setattr__(self, name, value)

    def __dir__(self):
        return sorted(self._OPT_FIELDS | {"filterparent", "load_from_dict", "to_dict"})

    @property
    def filterparent(self) -> bool:
        """是否过滤父级碰撞（从 disableflags 派生）。"""
        return not (self.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)

    @filterparent.setter
    def filterparent(self, value: bool) -> None:
        if value:
            # 开启过滤：清除 disableflags 中的 FILTERPARENT 位
            self.disableflags &= ~mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
        else:
            # 关闭过滤：设置 disableflags 中的 FILTERPARENT 位
            self.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT

    def load_from_dict(self, config: dict) -> None:
        """从字典批量设置配置参数。

        Args:
            config: 配置字典，键为字段名，值为参数值。
        """
        for key, value in config.items():
            if key in self._OPT_FIELDS or key == "filterparent":
                setattr(self, key, value)
            else:
                raise KeyError(
                    f"未知的配置项 '{key}'，可用: {sorted(self._OPT_FIELDS)}"
                )

    def to_dict(self) -> dict:
        """导出所有配置参数为字典。"""
        result = {field: getattr(self._mj_model.opt, field) for field in self._OPT_FIELDS}
        result["filterparent"] = self.filterparent
        return result
