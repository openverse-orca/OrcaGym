"""OrcaGymEulerEnv — OrcaGym Euler 环境 Facade。

继承 OrcaGymBaseEnv，组合 OrcaGymEuler 作为仿真核心，
提供 Gymnasium 兼容的环境接口。

属于 OrcaGymEuler 体系的 P3 Studio 集成组件。
参见 docs/design/architecture/orca_gym_euler_architecture.md 第 5.1 节。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import grpc

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub
from orca_gym import OrcaGymModel, OrcaGymData
from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimConfig
from orca_gym.environment.orca_gym_env import OrcaGymBaseEnv

_logger = get_orca_logger()


class OrcaGymEulerEnv(OrcaGymBaseEnv):
    """OrcaGym Euler 环境 Facade，组合 OrcaGymEuler 仿真核心。

    设计契约:
        - 继承 OrcaGymBaseEnv 的 Gymnasium 接口与名称空间方法。
        - 通过 _gym（OrcaGymEuler）访问仿真，不直接持有 _mjModel/_mjData。
        - env.data 返回 OrcaGymDataView（只读视图），替代直接访问 _mjData。
        - env.sim_config 返回 SimConfig，替代直接访问 _mjModel.opt。
        - 支持 gRPC 在线模式与离线短链模式（skip_grpc_load）。

    使用示例:
        ```python
        env = OrcaGymEulerEnv(
            frame_skip=5,
            orcagym_addr="localhost:50051",
            agent_names=["agent0"],
            time_step=0.002,
            model_xml_path="scene.xml",
        )
        obs, info = env.reset()
        for _ in range(100):
            ctrl = np.zeros(env.nu, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(ctrl)
            env.render()
        env.close()
        ```
    """

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        *,
        model_xml_path: Optional[str] = None,
        skip_grpc_load: bool = False,
        render_mode: str = "human",
        sync_render: bool = False,
        **kwargs,
    ) -> None:
        """初始化 Euler 环境。

        Args:
            frame_skip: 每次 gym step() 执行的物理步进次数。
            orcagym_addr: OrcaStudio gRPC 服务器地址。
            agent_names: 智能体名称列表。
            time_step: 物理仿真时间步长（秒）。
            model_xml_path: 本地 MJCF 场景 XML 路径（离线模式必需）。
            skip_grpc_load: True 时不连接 gRPC，使用本地 XML。
            render_mode: 渲染模式，"human" 渲染到 Studio，"none" 不渲染。
            sync_render: True 时每个物理步都渲染（同步渲染），False 时按 fps 节流。
            **kwargs: 传递给 OrcaGymBaseEnv 的额外参数。
        """
        self._skip_grpc_load = bool(skip_grpc_load)
        self._local_xml_path = model_xml_path
        self._xml_assets_dir = kwargs.pop("xml_assets_dir", None)
        self._render_mode = render_mode
        self._sync_render = bool(sync_render)

        # Python 3.12 兼容：确保主线程有事件循环（OrcaGymBaseEnv 依赖 asyncio.get_event_loop）
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # 渲染相关（与 OrcaGymLocalEnv 一致）
        render_fps = self.metadata.get("render_fps", 30)
        self._render_interval = 1.0 / render_fps
        self._render_time_step = time.perf_counter()
        self._render_count_interval = time_step * frame_skip * render_fps
        self.render_count = 0
        self._last_frame_index = -1

        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        self.mj_forward()

    # ------------------------------------------------------------------
    # 生命周期（实现 OrcaGymBaseEnv 抽象方法）
    # ------------------------------------------------------------------

    def initialize_grpc(self) -> None:
        """初始化 gRPC 通信通道和 OrcaGymEuler 仿真核心。"""
        if self._skip_grpc_load:
            self.channel = None
            self.stub = None
            self.gym = OrcaGymEuler(stub=None)
            # 配置离线模式的本地 XML 路径
            if self._local_xml_path:
                self.gym.studio.configure_offline(
                    self._local_xml_path, self._xml_assets_dir
                )
            return

        self.channel = grpc.aio.insecure_channel(
            self.orcagym_addr,
            options=[
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ],
        )
        self.stub = GrpcServiceStub(self.channel)
        self.gym = OrcaGymEuler(stub=self.stub)

    def pause_simulation(self) -> None:
        """暂停 OrcaStudio 仿真（被动模式）。"""
        if self._skip_grpc_load:
            return
        self.loop.run_until_complete(self.gym.pause_simulation())

    def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]:
        """初始化仿真，加载模型并构建 OrcaGymModel/OrcaGymData。"""
        _logger.info(f"Initializing simulation: Class: {self.__class__.__name__}")
        if self._skip_grpc_load and self._local_xml_path:
            # 离线模式：直接使用本地 XML
            model_xml_path = self._local_xml_path
        else:
            # 在线模式：从 OrcaStudio 加载
            model_xml_path = self.loop.run_until_complete(self.gym.load_model_xml())
        self.loop.run_until_complete(self.gym.init_simulation(model_xml_path))
        # OrcaGymBaseEnv 期望 self.model 和 self.data 为 OrcaGymModel/OrcaGymData
        model = self.gym.model
        data = self.gym._registry.build_orca_gym_data()
        return model, data

    def reset_simulation(self) -> None:
        """重置仿真到初始状态。"""
        # Euler 体系：重新加载 MJCF 重置状态
        # MuJoCoSimCore 不暴露 reset，通过重新 init 实现（与 OrcaGymLocal.load_initial_frame 等价）
        # 这里采用 mj_forward + sync_to_view 确保状态一致
        self.gym.mj_forward()
        self.gym.sync_to_view()
        self.set_time_step(self.time_step)

    def init_qpos_qvel(self) -> None:
        """初始化并保存初始关节位置和速度。"""
        self.gym.sync_to_view()
        self.init_qpos = self.gym.data.qpos.ravel().copy()
        self.init_qvel = self.gym.data.qvel.ravel().copy()

    def set_time_step(self, time_step: float) -> None:
        """设置仿真时间步长。"""
        self.time_step = time_step
        self.realtime_step = time_step * self.frame_skip
        # 通过 SimConfig 设置本地 timestep
        if self.gym._opt is not None:
            self.gym.sim_config.timestep = time_step
        # 同步到远程 OrcaStudio
        if not self._skip_grpc_load:
            self.loop.run_until_complete(self.gym.studio.set_timestep_remote(time_step))

    async def _resume_simulation(self) -> None:
        """恢复 OrcaStudio 仿真（被动模式下通常不需要）。"""
        pass

    def close(self) -> None:
        """关闭环境，清理 gRPC 资源。"""
        if self._skip_grpc_load:
            return
        if self.channel is not None:
            self.loop.run_until_complete(self.channel.close())

    # ------------------------------------------------------------------
    # 仿真控制
    # ------------------------------------------------------------------

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """执行仿真步进：设置控制并步进 n_frames 次，然后同步状态。

        Args:
            ctrl: 控制输入数组，形状 (nu,)。
            n_frames: 步进次数，通常等于 frame_skip。
        """
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, "
                f"found {np.array(ctrl).shape}"
            )
        self._step_orca_sim_simulation(ctrl, n_frames)
        # 同步状态到 DataView
        self.gym.sync_to_view()

    def _step_orca_sim_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """执行仿真步进：设置控制并步进 n_frames 次。"""
        self.set_ctrl(ctrl)
        self.mj_step(nstep=n_frames)

    def mj_step(self, nstep: int) -> None:
        """执行 MuJoCo 仿真步进。"""
        self.gym.mj_step(nstep)

    def mj_forward(self) -> None:
        """执行 MuJoCo 前向计算。"""
        self.gym.mj_forward()
        # 同步派生状态到 DataView
        if self.gym._view is not None:
            self.gym.sync_to_view()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置控制输入。"""
        self.gym.set_ctrl(ctrl)

    # ------------------------------------------------------------------
    # 状态访问（覆盖 OrcaGymBaseEnv 的 self.data 行为）
    # ------------------------------------------------------------------

    @property
    def data(self) -> OrcaGymDataView:
        """MuJoCo 状态只读视图，替代直接访问 _mjData。"""
        return self.gym.data

    @data.setter
    def data(self, value) -> None:
        # OrcaGymBaseEnv.__init__ 会赋值 self.data = ...（来自 initialize_simulation）
        # Euler 体系忽略该赋值，data 始终通过 self.gym.data 获取
        pass

    @property
    def model(self) -> OrcaGymModel:
        """OrcaGymModel 模型信息。"""
        return self.gym.model

    @model.setter
    def model(self, value) -> None:
        # OrcaGymBaseEnv.__init__ 会赋值 self.model = ...
        # Euler 体系忽略该赋值，model 始终通过 self.gym.model 获取
        pass

    @property
    def sim_config(self) -> SimConfig:
        """求解器配置。"""
        return self.gym.sim_config

    @property
    def dt(self) -> float:
        """环境时间步长（物理时间步长 × frame_skip）。"""
        return self.gym.sim_config.timestep * self.frame_skip

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    @property
    def render_mode(self) -> str:
        if hasattr(self, "_render_mode"):
            return self._render_mode
        return "human"

    @property
    def sync_render(self) -> bool:
        if hasattr(self, "_sync_render"):
            return self._sync_render
        return False

    def render(self) -> Union[NDArray[np.float64], None]:
        """渲染当前仿真状态到 OrcaStudio。

        支持两种渲染节流模式（与 OrcaGymLocalEnv 一致）：
        - sync_render=True：每个物理步累加 render_count，达到阈值时渲染。
        - sync_render=False（默认）：按 render_fps 节流，时间间隔超过阈值时渲染。
        """
        if self.render_mode not in ["human", "force"]:
            return

        if self._skip_grpc_load:
            return

        if self.sync_render:
            self.render_count += self._render_count_interval
            if self.render_count >= 1.0:
                self.loop.run_until_complete(self.gym.render())
                self.do_body_manipulation()
                self.render_count -= 1.0
        else:
            time_diff = time.perf_counter() - self._render_time_step
            if time_diff > self._render_interval:
                self._render_time_step = time.perf_counter()
                self.loop.run_until_complete(self.gym.render())
                self.do_body_manipulation()

    def do_body_manipulation(self) -> None:
        """处理 Studio UI 的 body 锚点拖拽操作。

        P3A 阶段：最小实现，仅当模型包含锚点 body 时启用。
        完整实现（anchor_actor/release_body_anchored 等）在 P4 阶段补齐。
        """
        if not hasattr(self, "_anchor_body_id") or self._anchor_body_id is None:
            return
        # P4 阶段实现完整锚点逻辑
        # 当前仅占位，避免阻塞在线渲染循环

    # ------------------------------------------------------------------
    # step / reset（Gymnasium 接口，子类应实现 step/reset_model）
    # ------------------------------------------------------------------

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError("子类必须实现 step()")

    def reset_model(self) -> tuple[dict, dict]:
        raise NotImplementedError("子类必须实现 reset_model()")

    def _get_obs(self) -> dict:
        """默认观测（子类可重写）。使用 env.data 公共 API。"""
        return {
            "qpos": np.array(self.data.qpos, dtype=np.float32),
            "qvel": np.array(self.data.qvel, dtype=np.float32),
        }
