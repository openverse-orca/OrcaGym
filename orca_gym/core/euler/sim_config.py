"""SimConfig — MuJoCo 求解器参数配置（阶段二填充）。

本模块属于 OrcaGym Euler 体系阶段二（P4-Step2），提供 typed 的 MuJoCo
求解器参数读写接口，替代直接访问 `_mjModel.opt.*`。

阶段二 Step 2 将 property 改为委托 `mj_model.opt.*`。未绑定时（init_simulation
前）getter 返回缓存默认值，setter 写入缓存；绑定后（_bind 调用后）getter/setter
委托真实 `_mj_model.opt.*`。`_bind` 仅切换引用不同步缓存值——绑定后 mj_model.opt
保留 XML 原值，由 Env 层（如 Step 6 的 `sim_config.timestep = self._time_step`）
显式重新应用需生效的缓存配置。
"""

import enum
from typing import Any

import numpy as np

#: SimConfig 构造期 opt 覆盖支持的键（timestep 走 time_step 构造参数，单独通道）。
OPT_OVERRIDE_KEYS: frozenset[str] = frozenset({"integrator", "gravity", "iterations"})


def validate_opt_overrides(overrides: dict[str, Any] | None) -> None:
    """校验构造期 opt 覆盖键（用户边界入口调用，越早失败越好）。

    Args:
        overrides: Env 构造参数 sim_config_overrides。None 或空 dict 直接通过。

    Raises:
        ValueError: 含非法键（含 "timestep" —— 该键与 time_step 构造参数双通道冲突）。
    """
    if not overrides:
        return
    invalid = set(overrides) - OPT_OVERRIDE_KEYS
    if invalid:
        raise ValueError(
            f"不支持的 sim_config_overrides 键: {sorted(invalid)}；"
            f"合法键: {sorted(OPT_OVERRIDE_KEYS)}（timestep 请用 time_step 构造参数）"
        )


class SimBackend(enum.Enum):
    """仿真后端选择（对齐 design §2.1 二选一）。"""

    MUJOCO = "mujoco"   # CPU 后端（现状 MuJoCoSimCore）
    EULER = "euler"     # Euler GPU 后端（P1 单世界）


class SimConfig:
    """MuJoCo 求解器参数配置。

    替代直接访问 _mjModel.opt.*。
    修改在下次 mj_step 时生效。

    使用契约:
        读取: ts = env.sim_config.timestep
        写入: env.sim_config.timestep = 0.002
        批量: env.sim_config.load_from_dict({"integrator": 0, "iterations": 100})
        绑定: sim_config._bind(mj_model)  # 供 OrcaGymEuler.init_simulation 调用

    禁止:
        不要通过 env._gym._sim._mjModel.opt.* 绕道访问。
    """

    def __init__(self, mj_model=None) -> None:
        """初始化求解器配置。

        Args:
            mj_model: MuJoCo 模型对象。None 时使用缓存占位字段，
                待 _bind(mj_model) 绑定后委托真实 mj_model.opt.*。
        """
        self._mj_model = mj_model
        # 缓存默认值（合理的 MuJoCo 默认）；未绑定时 getter 返回这些值。
        # 绑定后 getter 委托 mj_model.opt.*（保留 XML 原值），
        # Env 层负责显式重新应用需生效的缓存配置（如 timestep）。
        self._timestep: float = 0.002
        self._integrator: int = 0
        self._iterations: int = 100
        self._gravity: np.ndarray = np.array([0.0, 0.0, -9.81])
        # Euler 后端选择（不委托 mj_model，见 design §4.3）
        self._backend: SimBackend = SimBackend.MUJOCO
        self._device: str = "cuda"
        self._nworld: int = 1

    # --- 绑定方法（供 OrcaGymEuler.init_simulation 后调用）---

    def _bind(self, mj_model) -> None:
        """绑定真实 mjModel，切换 property 委托到 mj_model.opt.*。

        供 OrcaGymEuler.init_simulation 在加载 mjModel 后调用。
        绑定后 mj_model.opt 保留 XML 原值（不同步缓存），由 Env 层
        显式重新应用需生效的缓存配置（如 `sim_config.timestep = ts`）。

        Args:
            mj_model: MuJoCo MjModel 对象。
        """
        self._mj_model = mj_model

    # --- property（架构 §12.2）---

    @property
    def timestep(self) -> float:
        """物理仿真时间步长（秒）。"""
        if self._mj_model is not None:
            return float(self._mj_model.opt.timestep)
        return self._timestep

    @timestep.setter
    def timestep(self, value: float) -> None:
        v = float(value)
        self._timestep = v  # 始终缓存
        if self._mj_model is not None:
            if self._backend == SimBackend.EULER:
                raise RuntimeError(
                    "Euler 后端下 timestep 在 init_simulation 后不可修改。"
                    "请在 init_simulation 前通过 SimConfig.timestep = v 设置。"
                )
            self._mj_model.opt.timestep = v

    @property
    def integrator(self) -> int:
        """积分器类型（MuJoCo mjtIntegrator 枚举值）。"""
        if self._mj_model is not None:
            return int(self._mj_model.opt.integrator)
        return self._integrator

    @integrator.setter
    def integrator(self, value: int) -> None:
        v = int(value)
        self._integrator = v
        if self._mj_model is not None:
            if self._backend == SimBackend.EULER:
                raise RuntimeError(
                    "Euler 后端下 integrator 在 init_simulation 后不可修改。"
                    "请在 init_simulation 前通过 SimConfig.integrator = v 设置。"
                )
            self._mj_model.opt.integrator = v

    @property
    def iterations(self) -> int:
        """求解器迭代次数。"""
        if self._mj_model is not None:
            return int(self._mj_model.opt.iterations)
        return self._iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        v = int(value)
        self._iterations = v
        if self._mj_model is not None:
            if self._backend == SimBackend.EULER:
                raise RuntimeError(
                    "Euler 后端下 iterations 在 init_simulation 后不可修改。"
                    "请在 init_simulation 前通过 SimConfig.iterations = v 设置。"
                )
            self._mj_model.opt.iterations = v

    @property
    def gravity(self) -> np.ndarray:
        """重力加速度向量 (3,)。"""
        if self._mj_model is not None:
            return self._mj_model.opt.gravity
        return self._gravity

    @gravity.setter
    def gravity(self, value: np.ndarray) -> None:
        v = np.asarray(value, dtype=np.float64)
        self._gravity = v
        if self._mj_model is not None:
            if self._backend == SimBackend.EULER:
                raise RuntimeError(
                    "Euler 后端下 gravity 在 init_simulation 后不可修改。"
                    "请在 init_simulation 前通过 SimConfig.gravity = v 设置。"
                )
            self._mj_model.opt.gravity[:] = v

    # --- 后端选择（不委托 mj_model，见 design §4.3）---

    @property
    def backend(self) -> SimBackend:
        return self._backend

    @backend.setter
    def backend(self, value: SimBackend | str) -> None:
        self._backend = SimBackend(value)  # 支持 "euler" str 或 SimBackend 枚举

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = value

    @property
    def nworld(self) -> int:
        return self._nworld

    @nworld.setter
    def nworld(self, value: int) -> None:
        self._nworld = value

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
