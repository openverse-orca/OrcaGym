"""SimConfig — MuJoCo 求解器参数配置（骨架）。

本模块属于 OrcaGym Euler 体系骨架阶段（P1-Step1），提供 typed 的 MuJoCo
求解器参数读写接口，替代直接访问 `_mjModel.opt.*`。

骨架阶段不持有真实 `mjModel`，property 通过内部占位字段实现读写。
P4 填充阶段将改为委托 `mj_model.opt.*`。
"""

import numpy as np


class SimConfig:
    """MuJoCo 求解器参数配置。

    替代直接访问 _mjModel.opt.*。
    修改在下次 mj_step 时生效。

    使用契约:
        读取: ts = env.sim_config.timestep
        写入: env.sim_config.timestep = 0.002
        批量: env.sim_config.load_from_dict({"integrator": 0, "iterations": 100})

    禁止:
        不要通过 env._gym._sim._mjModel.opt.* 绕道访问。
    """

    def __init__(self, mj_model=None) -> None:
        """初始化求解器配置。

        Args:
            mj_model: MuJoCo 模型对象。骨架阶段不依赖真实 mjModel，
                使用内部占位字段；P4 填充阶段将委托 mj_model.opt.*。
        """
        # 骨架阶段：存储引用供 P4 填充，但 property 走占位字段
        self._mj_model = mj_model
        # 占位默认值（合理的 MuJoCo 默认）
        self._timestep: float = 0.002
        self._integrator: int = 0
        self._iterations: int = 100
        self._gravity: np.ndarray = np.array([0.0, 0.0, -9.81])

    # --- 骨架包含的 property（架构 §12.2）---

    @property
    def timestep(self) -> float:
        """物理仿真时间步长（秒）。"""
        return self._timestep

    @timestep.setter
    def timestep(self, value: float) -> None:
        self._timestep = float(value)

    @property
    def integrator(self) -> int:
        """积分器类型（MuJoCo mjtIntegrator 枚举值）。"""
        return self._integrator

    @integrator.setter
    def integrator(self, value: int) -> None:
        self._integrator = int(value)

    @property
    def iterations(self) -> int:
        """求解器迭代次数。"""
        return self._iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        self._iterations = int(value)

    @property
    def gravity(self) -> np.ndarray:
        """重力加速度向量 (3,)。"""
        return self._gravity

    @gravity.setter
    def gravity(self, value: np.ndarray) -> None:
        self._gravity = np.asarray(value, dtype=np.float64)

    # --- 批量读写 ---

    def load_from_dict(self, config: dict) -> None:
        """从字典批量加载配置。

        Args:
            config: 配置字典，键为属性名（timestep/integrator/iterations/gravity），
                值为对应类型的配置值。未提供的键保持原值不变。
        """
        if "timestep" in config:
            self.timestep = config["timestep"]
        if "integrator" in config:
            self.integrator = config["integrator"]
        if "iterations" in config:
            self.iterations = config["iterations"]
        if "gravity" in config:
            self.gravity = config["gravity"]

    def to_dict(self) -> dict:
        """导出当前配置为字典。

        Returns:
            包含 timestep/integrator/iterations/gravity 四个键的字典。
        """
        return {
            "timestep": self.timestep,
            "integrator": self.integrator,
            "iterations": self.iterations,
            "gravity": self.gravity,
        }
