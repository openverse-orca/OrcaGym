"""OrcaGymEulerEnv — Euler 环境 Facade（阶段二填充）。

本模块属于 OrcaGym Euler 体系阶段二（P3-Step1 骨架 + P4-Step6 生命周期与步进填充），
继承 OrcaGymBaseEnv，实现 Env 层隔离机制（K1/K2/K4/K6/K7/K8/K9/K10/K11/K12）。

阶段二 Step 6 填充生命周期方法（initialize_grpc/initialize_simulation/
reset_simulation/init_qpos_qvel/set_time_step/pause_simulation/close）、
步进方法（do_simulation/mj_step/mj_forward/set_ctrl）、状态设置方法
（set_joint_qpos/set_joint_qvel），替换骨架的 no-op / NotImplementedError。

核心设计:
    - 持有 _gym/_stub/_channel/_studio_bridge（带下划线，K1）
    - __setattr__ 屏蔽父类的 self.gym/self.model/self.data 赋值（K10 方案 A）
    - __getattr__ 拦截 _BLOCKED_ATTRS，__dir__ 只列公共 API（K2）
    - 仿真控制全部委托 self._gym 公共方法，不触私有（K4/K8）
    - env.data/env.model/env.sim_config 通过 Gym 公共属性委托（K6/K7）
    - Studio 交互通过 self._studio_bridge，不通过 gym.studio（K9）

父类和解（架构 §12.5）:
    OrcaGymBaseEnv.__init__ 执行 self.gym=None / self.model,self.data=initialize_simulation()
    / self.gym.opt.timestep —— 与 K1/K2/K6 直接冲突。
    采用方案 A: __setattr__ 屏蔽 + property 接管。
    __setattr__ 是类级方法，定义即生效，父类赋值被转发或忽略。
"""

import asyncio
import time
from typing import Any, Dict, Tuple, Union

import grpc
import numpy as np
from numpy.typing import NDArray

from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimConfig
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub
from ..orca_gym_env import OrcaGymBaseEnv


class OrcaGymEulerEnv(OrcaGymBaseEnv):
    """OrcaGym Euler 环境 Facade。

    ┌─────────────────────────────────────────────────────────────┐
    │  使用契约：用户不应直接访问 _gym/_stub/_channel/_mjData 或  │
    │  任何内部 MuJoCo 对象。                                      │
    │  读取状态 → 使用 env.data（OrcaGymDataView）                │
    │  写入外力 → 使用 env.apply_body_force()（P4 填充）          │
    │  仿真步进 → 使用 env.do_simulation(ctrl, n_frames)          │
    │  求解器配置 → 使用 env.sim_config.timestep = 0.002          │
    │  Studio 交互 → 使用 env.studio_bridge()（P4 填充）          │
    │  缺少功能时 → 扩展本类的公共方法，不要直接访问内部对象。    │
    └─────────────────────────────────────────────────────────────┘

    使用契约:
        读取状态:   env.data.qpos / env.data.body_xpos(name) / env.query_*()
        写入状态:   env.set_joint_qpos()（P4）/ env.apply_body_force()（P4）
        仿真步进:   env.do_simulation(ctrl, n_frames)
        求解器配置: env.sim_config.timestep = 0.002

    禁止:
        不要访问 env._gym._sim._mjData 或任何内部 MuJoCo 对象。
        不要访问 env._gym._studio / env._studio_bridge（内部组件）。
        缺少功能时，扩展本类的公共方法。
    """

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

    # K2: Env 层隔离机制（与 Gym 层对称）
    _BLOCKED_ATTRS = frozenset({
        # L3 引擎内部
        "_mjData", "_mjModel", "mj_data", "mj_model",
        "_mj_data", "_mj_model", "mjData", "mjModel",
        # L2 内部组件（含父类残留的公共名）
        "gym", "stub", "channel",
    })

    # K10: 父类契约屏蔽字段（方案 A，架构 §12.5）
    # 父类 __init__ 会 self.gym=X / self.stub=X / self.channel=X / self.model=Y / self.data=Z
    # __setattr__ 拦截这些赋值：gym/stub/channel 转发到带下划线，model/data 忽略（走 property）
    _SHIELDED_ATTRS = frozenset({"gym", "stub", "channel", "model", "data"})

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        *,
        model_xml_path: str | None = None,
        skip_grpc_load: bool = False,
        render_mode: str = "human",
        sync_render: bool = False,
        **kwargs,
    ) -> None:
        """初始化 Euler 环境 Facade。

        Args:
            frame_skip: 每次 Gym step() 执行的物理步进次数。
            orcagym_addr: OrcaStudio gRPC 服务器地址。
            agent_names: 环境中智能体名称列表。
            time_step: 仿真时间步长。
            model_xml_path: MuJoCo 模型 XML 文件路径（离线模式使用）。
            skip_grpc_load: 跳过 gRPC 加载（骨架测试/离线模式）。
            render_mode: 渲染模式（"human"/"none"）。
            sync_render: 是否同步渲染。
            **kwargs: 传递给父类的额外参数。

        Note:
            __setattr__ 是类级方法，定义即生效，父类 __init__ 中的
            self.gym=X / self.model=Y / self.data=Z 赋值会被自动屏蔽。
        """
        # Env 自有字段（在 super().__init__ 之前设置，供生命周期方法使用）
        self._skip_grpc_load = skip_grpc_load
        self._local_xml_path = model_xml_path
        self._render_mode = render_mode
        self._sync_render = sync_render
        self._studio_bridge = None   # 将在 initialize_grpc 中赋值
        # _time_step 缓存：set_time_step 在 initialize_simulation 前被父类调用，
        # 此时 SimConfig 未绑定 mjModel，缓存到 _time_step，
        # 在 initialize_simulation 末尾重新设置。
        self._time_step = time_step
        # 渲染节流字段（render_mode="human" 在线渲染时使用）
        self._render_count = 0.0
        self._render_count_interval = 0.0
        self._render_time_step = 0.0
        self._render_interval = 1.0 / self.metadata.get("render_fps", 30)
        self._last_frame_index = -1

        # 父类 __init__ 无条件调用 asyncio.get_event_loop()，在 Python 3.12 中
        # 若先前测试已关闭事件循环（asyncio.run / loop.close）会抛 RuntimeError。
        # 此处确保存在 current event loop，保证骨架模式在测试套件中可重复构造。
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # 调用父类 __init__（会触发 initialize_grpc / pause_simulation /
        # set_time_step / initialize_simulation / reset_simulation / init_qpos_qvel）
        # 父类中的 self.gym=X / self.model=Y / self.data=Z 被 __setattr__ 屏蔽。
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

    # --- K10: __setattr__ 屏蔽父类赋值（方案 A，架构 §12.5）---
    def __setattr__(self, name: str, value: Any) -> None:
        """屏蔽父类对 gym/stub/channel/model/data 的直接赋值。

        gym/stub/channel 转发到带下划线字段（_gym/_stub/_channel），
        model/data 忽略（始终通过 @property 从 self._gym 取）。
        """
        if name == "gym":
            object.__setattr__(self, "_gym", value)   # 转发到 _gym
            return
        if name == "stub":
            object.__setattr__(self, "_stub", value)
            return
        if name == "channel":
            object.__setattr__(self, "_channel", value)
            return
        if name == "model":
            return   # 忽略：model 始终通过 @property 从 self._gym.model 取
        if name == "data":
            return   # 忽略：data 始终通过 @property 从 self._gym.data 取
        super().__setattr__(name, value)

    # --- K2: 隔离机制 ---
    def __getattr__(self, name: str) -> Any:
        """拦截 _BLOCKED_ATTRS 的外部访问，返回引导性错误。

        __getattr__ 仅在属性查找失败时触发。由于 __setattr__ 将 gym/stub/channel
        转发到 _gym/_stub/_channel（不创建 gym/stub/channel 属性），
        访问 env.gym/env.stub/env.channel 会触发本方法。
        """
        if name.startswith("__") and name.endswith("__"):
            # dunder 方法不拦截（避免 pickle/copy 等副作用）
            raise AttributeError(name)
        blocked = type(self)._BLOCKED_ATTRS
        if name in blocked:
            raise AttributeError(
                f"'{type(self).__name__}' 对象的属性 '{name}' 被隔离。\n"
                f"  使用契约：用户不应直接访问 _gym/_stub/_channel/_mjData 或任何内部对象。\n"
                f"  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView），如 env.data.qpos\n"
                f"  写入外力 → 使用 env.apply_body_force()（P4）\n"
                f"  配置求解器 → 使用 env.sim_config\n"
                f"  仿真步进 → 使用 env.do_simulation(ctrl, n_frames)\n"
                f"  Studio 交互 → 使用 env.studio_bridge()（P4）\n"
                f"  缺少功能时 → 扩展 OrcaGymEulerEnv 公共方法，不要直接访问内部对象。"
            )
        raise AttributeError(
            f"'{type(self).__name__}' 对象没有属性 '{name}'"
        )

    def __dir__(self) -> list[str]:
        """只列出公共 API，不含内部组件或引擎内部。"""
        result = super().__dir__()
        blocked = type(self)._BLOCKED_ATTRS | {"_gym", "_stub", "_channel", "_studio_bridge"}
        return [name for name in result if name not in blocked]

    # --- 生命周期（实现 OrcaGymBaseEnv 抽象方法）---

    def initialize_grpc(self) -> None:
        """初始化 gRPC 通信管道并创建 OrcaGymEuler 实例。

        K9 合规：Studio 交互通过自持 _studio_bridge，不通过 gym.studio。
        离线模式: skip_grpc_load=True 时创建 stub=None 的 Gym，不连接 gRPC。
        在线模式: 创建 grpc.aio.insecure_channel + GrpcServiceStub。
        """
        if self._skip_grpc_load:
            # 离线模式：不创建 gRPC channel
            object.__setattr__(self, "_channel", None)
            object.__setattr__(self, "_stub", None)
            self.gym = OrcaGymEuler(stub=None)   # __setattr__ 转发到 _gym
            self._studio_bridge = self._gym.studio_bridge()   # 取一次引用
            if self._local_xml_path:
                self._studio_bridge.configure_offline(self._local_xml_path)
            return
        # 在线模式：创建 gRPC channel + stub
        object.__setattr__(self, "_channel", grpc.aio.insecure_channel(
            self.orcagym_addr,
            options=[
                ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ],
        ))
        object.__setattr__(self, "_stub", GrpcServiceStub(self._channel))
        self.gym = OrcaGymEuler(stub=self._stub)
        self._studio_bridge = self._gym.studio_bridge()

    def initialize_simulation(self) -> Tuple[Any, OrcaGymDataView]:
        """初始化仿真：加载模型 XML + init_simulation + 返回 (model, view)。

        K6 合规：返回 OrcaGymDataView 而非 OrcaGymData。
        """
        # 1. 获取模型 XML 路径（离线：本地路径；在线：从 Studio 拉取）
        if self._skip_grpc_load:
            model_xml_path = self._local_xml_path
        else:
            model_xml_path = self.loop.run_until_complete(self._gym.load_model_xml())
        # 2. 初始化仿真
        self.loop.run_until_complete(self._gym.init_simulation(model_xml_path))
        # 3. 应用缓存的 time_step（init_simulation 前设置的值需重新生效）
        self._gym.sim_config.timestep = self._time_step
        # 4. 在线模式：同步时间步到远端 OrcaStudio
        if not self._skip_grpc_load:
            self._studio_bridge.set_timestep_remote(self._time_step)
        # 5. 返回 (OrcaGymModel, OrcaGymDataView)
        return self._gym.model, self._gym.data

    def reset_simulation(self) -> None:
        """重置 MjData 到初始状态并同步 DataView。"""
        self._gym.reset_data()
        self._gym.sync_to_view()

    def init_qpos_qvel(self) -> None:
        """保存初始 qpos/qvel。"""
        self._gym.sync_to_view()
        self.init_qpos = self._gym.data.qpos.ravel().copy()
        self.init_qvel = self._gym.data.qvel.ravel().copy()

    def set_time_step(self, time_step: float) -> None:
        """设置仿真时间步长（本地缓存 + 远端同步）。

        父类 __init__ 在 initialize_simulation 前调用本方法，此时 SimConfig
        未绑定 mjModel，缓存到 self._time_step，在 initialize_simulation
        末尾重新设置。initialize_simulation 后调用时，本地立即生效，
        在线模式同步到远端 OrcaStudio。
        """
        self._time_step = time_step
        self.realtime_step = time_step * self.frame_skip
        # 本地：若 Gym 已初始化（init_simulation 已执行），直接设置
        if hasattr(self, "_gym") and self._gym is not None:
            try:
                self._gym.sim_config.timestep = time_step
            except RuntimeError:
                pass   # SimConfig 未绑定，缓存待 init_simulation
        # 远端：在线模式同步到 OrcaStudio
        if not self._skip_grpc_load and hasattr(self, "_studio_bridge"):
            self._studio_bridge.set_timestep_remote(time_step)

    def pause_simulation(self) -> None:
        """暂停仿真（离线模式 no-op）。"""
        if self._skip_grpc_load:
            return
        # 在线模式待 2.2 Step 3
        self.loop.run_until_complete(self._gym.pause_simulation())

    def close(self) -> None:
        """关闭环境（离线模式 no-op）。"""
        if self._skip_grpc_load:
            return
        # 在线模式关闭 gRPC channel
        if self._channel is not None:
            self.loop.run_until_complete(self._channel.close())

    # --- 仿真控制（K4/K8: 全部委托 self._gym 公共方法，不触私有）---

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """标准仿真步进（含 Euler 耦合，骨架阶段等价于纯 MuJoCo）。

        K4 合规: 只走 Gym 公共方法，不触 _gym._sim/_euler 等私有。
        K8 合规: 不写 if self._gym._euler is not None，通过 step_with_coupling 封装。

        Args:
            ctrl: 控制输入数组，形状 (nu,)。
            n_frames: 帧数。

        Raises:
            ValueError: ctrl 形状不匹配。
        """
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, "
                f"found {np.array(ctrl).shape}"
            )
        self._gym.step_with_coupling(ctrl, n_frames, self.dt)
        self._gym.sync_to_view()

    def mj_step(self, nstep: int) -> None:
        """纯 MuJoCo 步进（无 Euler 耦合），委托 self._gym.mj_step()。"""
        self._gym.mj_step(nstep)

    def mj_forward(self) -> None:
        """MuJoCo 前向计算（不步进，仅更新派生量），委托 self._gym.mj_forward()。"""
        self._gym.mj_forward()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置控制输入，委托 self._gym.set_ctrl()。"""
        self._gym.set_ctrl(ctrl)

    # --- 状态设置（reset_model 必需）---

    def set_joint_qpos(self, qpos: np.ndarray) -> None:
        """设置广义坐标 qpos（全量设置，reset_model 用）。

        注意：设置后需调用 mj_forward() 以更新派生量。

        Args:
            qpos: 广义坐标数组。
        """
        self._gym.set_qpos_qvel(qpos, self._gym.data.qvel)

    def set_joint_qvel(self, qvel: np.ndarray) -> None:
        """设置广义速度 qvel（全量设置，reset_model 用）。

        Args:
            qvel: 广义速度数组。
        """
        self._gym.set_qpos_qvel(self._gym.data.qpos, qvel)

    def _sync_view(self) -> None:
        """同步 DataView（子类内部使用，封装 _gym.sync_to_view()）。

        K4 合规：子类通过本方法同步 DataView，不直接触 _gym。
        """
        self._gym.sync_to_view()

    # --- K6/K7: 状态访问（通过 Gym 公共属性，不触私有）---

    @property
    def data(self) -> OrcaGymDataView:
        """返回 MuJoCo 状态只读视图（OrcaGymDataView），委托 self._gym.data。

        K6 合规: 返回 OrcaGymDataView 而非 OrcaGymData/mujoco.MjData。
        替代父类的 self.data（被 __setattr__ 屏蔽赋值）。
        """
        return self._gym.data

    @property
    def model(self) -> Any:
        """返回模型结构抽象（OrcaGymModel），委托 self._gym.model。

        K7 合规: 通过 Gym 公共属性委托。
        替代父类的 self.model（被 __setattr__ 屏蔽赋值）。
        """
        return self._gym.model

    @property
    def sim_config(self) -> SimConfig:
        """返回求解器配置（SimConfig），委托 self._gym.sim_config。

        K7 合规: 替代直接访问 _mjModel.opt.*。
        """
        return self._gym.sim_config

    @property
    def dt(self) -> float:
        """返回环境时间步长（物理时间步长 × frame_skip）。

        K7 合规: 替代父类的 self.gym.opt.timestep * self.frame_skip。
        """
        return self._gym.sim_config.timestep * self.frame_skip

    @property
    def ctrl(self) -> np.ndarray:
        """返回当前控制输入（读 actuator_force，阶段二简化实现）。"""
        return self._gym.data.actuator_force

    @ctrl.setter
    def ctrl(self, value: np.ndarray) -> None:
        """设置控制输入，委托 self._gym.set_ctrl()。"""
        self._gym.set_ctrl(value)

    # --- 渲染（K9: Studio 交互通过 self._studio_bridge / self._gym.render）---

    def render(self) -> Union[NDArray[np.float64], None]:
        """渲染当前仿真状态到 OrcaStudio。

        K9 合规: 通过 self._gym.render()（Gym 公共方法），不触 _gym.studio。

        节流策略（复用老体系）:
            - sync_render=True: 按计数器节流（每 N 物理步渲染一帧）
            - sync_render=False: 按墙钟 fps 节流（render_fps）

        离线模式（skip_grpc_load=True）: 无 OrcaStudio 可渲染，返回 None。
        render_mode 不在 ["human", "force"] 时立即返回 None。

        Returns:
            None（Euler 渲染不返回像素数组，由 OrcaStudio 负责显示）。
        """
        if self._render_mode not in ["human", "force"]:
            return None
        if self._skip_grpc_load:
            return None
        # 在线模式：节流后委托 gym.render()
        if self._sync_render:
            self._render_count += self._render_count_interval
            if self._render_count >= 1.0:
                self.loop.run_until_complete(self._gym.render())
                self.do_body_manipulation()
                self._render_count -= 1.0
        else:
            time_diff = time.perf_counter() - self._render_time_step
            if time_diff > self._render_interval:
                self._render_time_step = time.perf_counter()
                self.loop.run_until_complete(self._gym.render())
                self.do_body_manipulation()
        return None

    def do_body_manipulation(self) -> None:
        """处理 Studio UI 体操作（阶段二占位实现）。

        阶段二仅查询 Studio 的锚定/运动状态但不实际应用，
        完整的体操作力应用在 P4 实现。
        离线模式 no-op。
        """
        if self._skip_grpc_load:
            return
        # 查询状态（占位，不应用；P4 实现体操作力应用）
        self.loop.run_until_complete(self._studio_bridge.get_body_manipulation_anchored())
        self.loop.run_until_complete(self._studio_bridge.get_body_manipulation_movement())

    def studio_bridge(self):
        """返回 OrcaStudio 桥接对象（K9 方法访问模式）。

        替代 gym.studio property 式穿墙。
        """
        return self._studio_bridge

    # --- Gymnasium 接口（子类实现）---

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], float, bool, bool, Dict[str, float]]:
        """执行一个环境步进（子类实现）。"""
        raise NotImplementedError("step 待子类实现")

    def reset_model(self) -> tuple[dict, dict]:
        """重置机器人自由度（子类实现）。"""
        raise NotImplementedError("reset_model 待子类实现")

    def _get_obs(self) -> dict:
        """获取观测（子类实现）。"""
        raise NotImplementedError("_get_obs 待子类实现")
