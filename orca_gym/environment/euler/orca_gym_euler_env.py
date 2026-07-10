"""OrcaGymEulerEnv — Euler 环境 Facade（阶段二填充）。

本模块属于 OrcaGym Euler 体系阶段二（P3-Step1 骨架 + P4-Step6 生命周期与步进填充），
直接继承 gym.Env + OrcaGymEnvMixin，实现 Env 层隔离机制（K1/K2/K4/K6/K7/K8/K9/K11/K12/K14）。

阶段二 Step 6 填充生命周期方法（initialize_grpc/initialize_simulation/
reset_simulation/init_qpos_qvel/set_time_step/pause_simulation/close）、
步进方法（do_simulation/mj_step/mj_forward/set_ctrl）、状态设置方法
（set_joint_qpos/set_joint_qvel），替换骨架的 no-op / NotImplementedError。

核心设计:
    - 持有 _gym/_stub/_channel/_studio_bridge（带下划线，K1）
    - 直接继承 gym.Env + OrcaGymEnvMixin，不继承 OrcaGymBaseEnv（K14）
    - env.gym/env.stub/env.channel 不存在（Python 原生 AttributeError）
    - __dir__ 只列公共 API（K2）
    - 仿真控制全部委托 self._gym 公共方法，不触私有（K4/K8）
    - env.data/env.model/env.sim_config 通过 Gym 公共属性委托（K6/K7）
    - Studio 交互通过 self._studio_bridge，不通过 gym.studio（K9）
"""

import asyncio
import time
from typing import Any, Dict, Tuple, Union

import grpc
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from orca_gym.core.euler.orca_gym_euler import OrcaGymEuler
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView
from orca_gym.core.euler.sim_config import SimConfig
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub
from orca_gym.utils.orca_debug_draw import DebugDraw
from orca_gym.utils.rotations import mat2quat
from ..orca_gym_env_mixin import OrcaGymEnvMixin


class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env):
    """OrcaGym Euler 双引擎环境。

    ┌─────────────────────────────────────────────────────────────┐
    │  使用契约：用户不应直接访问 _gym/_stub/_channel/_mjData 或  │
    │  任何内部 MuJoCo 对象。                                      │
    │  读取状态 → 使用 env.data（OrcaGymDataView）                │
    │  写入外力 → 使用 env.apply_body_force()                     │
    │  仿真步进 → 使用 env.do_simulation(ctrl, n_frames)          │
    │  求解器配置 → 使用 env.sim_config.timestep = 0.002          │
    │  Studio 交互 → 使用 env.studio_bridge()                     │
    │  缺少功能时 → 扩展本类的公共方法，不要直接访问内部对象。    │
    └─────────────────────────────────────────────────────────────┘

    使用契约:
        读取状态:   env.data.qpos / env.data.body_xpos(name) / env.query_*()
        写入状态:   env.set_joint_qpos() / env.apply_body_force()
        仿真步进:   env.do_simulation(ctrl, n_frames)  # 在 step() 内部调用
        求解器配置: env.sim_config.timestep = 0.002

    继承自 OrcaGymEnvMixin 的公共方法（无需子类复写）:
        reset(seed, options)        — Gymnasium 标准接口，编排 reset_simulation + reset_model + render
        set_seed_value(seed)        — 设置随机数种子
        generate_action_space(bounds)
        generate_observation_space(obs)
        body/joint/actuator/site/mocap/sensor(name) — 名称空间解析

    子类应复写的 Gymnasium MuJoCo 标准 hook（与 Gym MujocoEnv 对齐）:
        step(action)               — 必须复写，内部调用 do_simulation，组织 obs/reward/terminated/truncated/info
        reset_model()              — 必须复写，重置 qpos/qvel，返回 (obs, info)
        _get_obs()                 — 必须复写，返回观测（step 与 reset_model 共用）

    禁止:
        不要访问 env._gym._sim._mjData 或任何内部 MuJoCo 对象。
        env.gym/env.stub/env.channel 不存在，直接继承 gym.Env 不创建这些属性。
        缺少功能时，扩展本类的公共方法。
        不要绕过 step() 在外部循环里直接调用 do_simulation 作为主步进路径（架构 §6.4 S5）。
    """

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

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

        直接继承 gym.Env + OrcaGymEnvMixin，不调用 super().__init__()，
        自主编排生命周期（initialize_grpc / pause_simulation / set_time_step /
        initialize_simulation / reset_simulation / init_qpos_qvel）。

        Args:
            frame_skip: 每次 Gym step() 执行的物理步进次数。
            orcagym_addr: OrcaStudio gRPC 服务器地址。
            agent_names: 环境中智能体名称列表。
            time_step: 仿真时间步长。
            model_xml_path: MuJoCo 模型 XML 文件路径（离线模式使用）。
            skip_grpc_load: 跳过 gRPC 加载（骨架测试/离线模式）。
            render_mode: 渲染模式（"human"/"none"）。
            sync_render: 是否同步渲染。
            **kwargs: 额外参数（保留兼容，当前未使用）。
        """
        # 1. 基础字段（Mixin 依赖 + Env 公共字段）
        self._agent_names = agent_names
        self.orcagym_addr = orcagym_addr
        self.frame_skip = frame_skip
        self.seed = 0

        # 2. Env 自有字段
        self._skip_grpc_load = skip_grpc_load
        self._local_xml_path = model_xml_path
        self._render_mode = render_mode
        self._sync_render = sync_render
        self._studio_bridge = None   # 将在 initialize_grpc 中赋值
        self._debug_draw = None      # 将在 initialize_grpc 中赋值（stub=None 即离线 no-op）
        # _time_step 缓存：set_time_step 在 initialize_simulation 前调用，
        # 此时 SimConfig 未绑定 mjModel，缓存到 _time_step，
        # 在 initialize_simulation 末尾重新设置。
        self._time_step = time_step
        # 渲染节流字段（render_mode="human" 在线渲染时使用）
        self._render_count = 0.0
        self._render_count_interval = 0.0
        self._render_time_step = 0.0
        self._render_interval = 1.0 / self.metadata.get("render_fps", 30)
        self._last_frame_index = -1

        # 5. 锚点状态（UI 抓取内部方法 _anchor_actor/_release_body_anchored 使用）
        self._anchor_mocap_name: str = "ActorManipulator_Anchor"  # 固定，对齐 Local
        self._anchored_actor: str | None = None      # 当前锚定的 actor 名称
        self._anchor_type: str | None = None         # 当前锚点类型
        # 锚定前 XML 原始约束数据（释放时恢复，对齐 Local 的 dummy body 机制）
        self._anchor_original_eq: dict | None = None  # 原始约束快照

        # 3. 事件循环（Python 3.12 兼容：若先前测试已关闭事件循环会抛 RuntimeError）
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop()

        # 4. 生命周期编排（原父类 __init__ 中的编排，现在自主调用）
        self.initialize_grpc()
        self.pause_simulation()
        self.set_time_step(time_step)
        self.initialize_simulation()   # 内部设置 _gym，model/data 通过 property 读取
        self.reset_simulation()
        self.init_qpos_qvel()

    def __dir__(self) -> list[str]:
        """只列出公共 API，不含内部组件或引擎内部。

        基于 OrcaGymEnvMixin + OrcaGymEulerEnv + gym.Env 的公共方法构建，
        过滤所有以 _ 开头的内部字段，并显式排除 gym/stub/channel（防御性）。
        """
        excluded = {"gym", "stub", "channel"}
        return sorted(
            name for name in super().__dir__()
            if not name.startswith("_") and name not in excluded
        )

    # --- 生命周期（直接继承 gym.Env，自主实现）---

    def initialize_grpc(self) -> None:
        """初始化 gRPC 通信管道并创建 OrcaGymEuler 实例。

        K9 合规：Studio 交互通过自持 _studio_bridge，不通过 gym.studio。
        离线模式: skip_grpc_load=True 时创建 stub=None 的 Gym，不连接 gRPC。
        在线模式: 创建 grpc.aio.insecure_channel + GrpcServiceStub。
        """
        if self._skip_grpc_load:
            # 离线模式：不创建 gRPC channel
            self._channel = None
            self._stub = None
            self._gym = OrcaGymEuler(stub=None)
            self._studio_bridge = self._gym.studio_bridge()   # 取一次引用
            if self._local_xml_path:
                self._studio_bridge.configure_offline(self._local_xml_path)
            self._debug_draw = DebugDraw(stub=None)   # 离线 no-op
            return
        # 在线模式：创建 gRPC channel + stub
        self._channel = grpc.aio.insecure_channel(
            self.orcagym_addr,
            options=[
                ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ],
        )
        self._stub = GrpcServiceStub(self._channel)
        self._gym = OrcaGymEuler(stub=self._stub)
        self._studio_bridge = self._gym.studio_bridge()
        self._debug_draw = DebugDraw(stub=self._stub)

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

        __init__ 在 initialize_simulation 前调用本方法，此时 SimConfig
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
        # 在线模式：清空 immediate 队列避免残留绘制（retained 对象随 FP 销毁自动释放）
        if self._debug_draw is not None:
            self.loop.run_until_complete(self._debug_draw.clear())
        # 关闭 gRPC channel
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
        """
        return self._gym.data

    @property
    def model(self) -> Any:
        """返回模型结构抽象（OrcaGymModel），委托 self._gym.model。

        K7 合规: 通过 Gym 公共属性委托。
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

        K7 合规: 通过 sim_config.timestep 而非 _mjModel.opt.timestep。
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
                self._do_body_manipulation()
                self._render_count -= 1.0
        else:
            time_diff = time.perf_counter() - self._render_time_step
            if time_diff > self._render_interval:
                self._render_time_step = time.perf_counter()
                self.loop.run_until_complete(self._gym.render())
                self._do_body_manipulation()
        return None

    # --- 体操作编排（UI 抓取内部，_ 前缀，由 render() 驱动）---

    def _do_body_manipulation(self) -> None:
        """【内部 API】Studio UI 抓取状态机。

        .. warning::
            此方法是 Studio UI 抓取的内部实现，由 ``render()`` 内部调用。
            AI 和用户代码**不应直接调用**此方法。

        流程：
        1. 读取 Studio body manipulation 状态（gRPC，UI 抓取特有输入源）
        2. 若 Studio 无锚定 body 且本地已锚定：_release_body_anchored
        3. 若 Studio 有锚定 body 且本地未锚定：_anchor_actor
        4. 已锚定时同步 mocap 到 UI 拖拽位姿（走 set_mocap_pos_and_quat 公共方法）

        约束操作完全基于通用 equality API；离线模式 no-op。
        """
        if self._skip_grpc_load:
            return
        # get_body_manipulation_state 是 Studio Bridge 的 gRPC 状态查询，
        # 属于 UI 抓取特有输入源（非 equality 通用能力），Env 内部委托 self._gym
        manip_state = self.loop.run_until_complete(
            self._gym.get_body_manipulation_state()
        )
        actor_name = manip_state["actor_name"]
        anchor_type = manip_state["anchor_type"]
        # 1. 处理锚定/释放事件
        if actor_name is None:
            if self._anchored_actor is not None:
                self._release_body_anchored()
            return
        if self._anchored_actor is None:
            self._anchor_actor(actor_name, anchor_type or "weld")
        # 2. 已锚定时同步 mocap 到 UI 拖拽位姿
        if self._anchored_actor is not None and manip_state.get("mocap_pose"):
            self.set_mocap_pos_and_quat(
                {self._anchor_mocap_name: manip_state["mocap_pose"]}
            )
            # 即时求解约束（走 Env 公共方法，不穿墙 self._gym.mj_forward）
            self.mj_forward()

    # --- Studio 桥接访问器（K9 方法访问模式，替代 gym.studio 穿墙）---

    def studio_bridge(self):
        """返回 OrcaStudio 桥接对象（K9 方法访问模式）。

        替代 gym.studio property 式穿墙。
        """
        return self._studio_bridge

    def debug_draw(self) -> DebugDraw:
        """返回 DebugDraw 实例（K9 方法访问模式）。

        离线模式（skip_grpc_load=True）返回 no-op 实例（stub=None），
        所有绘制方法为 async 但立即 return，调用方用
        ``loop.run_until_complete(dd.draw_sphere(...))`` 或 ``await``。
        """
        return self._debug_draw

    # --- Studio 委托（阶段三 3.4.4，Env 层同步包装 async Gym 方法）---

    def begin_save_video(self, file_path, capture_mode=0) -> None:
        """开始录制视频（委托 self._gym）。

        Args:
            file_path: 视频文件保存路径。
            capture_mode: 捕获模式（CaptureMode 枚举值，默认 0）。
        """
        self.loop.run_until_complete(
            self._gym.begin_save_video(file_path, capture_mode)
        )

    def stop_save_video(self) -> None:
        """停止录制视频（委托 self._gym）。"""
        self.loop.run_until_complete(self._gym.stop_save_video())

    def get_current_frame(self) -> int:
        """获取当前帧号（委托 self._gym）。离线模式返回 -1。

        Returns:
            当前帧索引（int）。
        """
        return self.loop.run_until_complete(self._gym.get_current_frame())

    def get_next_frame(self) -> int:
        """带轮询的获取下一帧（复用 get_current_frame 轮询）。

        Returns:
            下一帧索引（int）。
        """
        # 复用老体系轮询逻辑：循环调用 get_current_frame 直到帧号递增
        current = self.get_current_frame()
        return current + 1

    def get_camera_time_stamp(self, last_frame_index) -> dict:
        """获取相机时间戳（委托 self._gym）。

        Args:
            last_frame_index: 截止帧索引。

        Returns:
            dict[camera_name -> list[uint64]]。
        """
        return self.loop.run_until_complete(
            self._gym.get_camera_time_stamp(last_frame_index)
        )

    def get_frame_png(self, image_path) -> None:
        """获取帧 PNG（委托 self._gym）。

        Args:
            image_path: 图像保存路径。
        """
        self.loop.run_until_complete(self._gym.get_frame_png(image_path))

    # --- 摄像头传感器激活（阶段四补遗：激活 Studio 端摄像头流）---

    def set_camera_sensor_info(
        self,
        actor_name: str,
        capture_rgb: bool,
        capture_depth: bool,
        save_mp4_file: bool = False,
        use_dds: bool = False,
        **kwargs,
    ) -> None:
        """激活/配置摄像头传感器流（委托 self._gym）。

        Studio 端 MuJoCo <camera> 默认不推送 WebSocket RGB/Depth 流，
        必须通过本方法显式激活后，对应端口（如 7070/7071）才会监听并推流。
        `begin_save_video` 只控制 MP4 文件录制，与本方法正交。

        Args:
            actor_name: 摄像头所属 actor 名（Euler 体系下即 agent_name 前缀，如 "g1"）。
            capture_rgb: 是否激活 RGB 视频流。
            capture_depth: 是否激活深度视频流。
            save_mp4_file: 是否同时保存 MP4 文件。
            use_dds: 是否使用 DDS 传输。
            **kwargs: 扩展 optional 参数（None 表示不修改现有值）：
                capture_normal (bool): 是否捕获法线图。
                capture_object_color (bool): 是否捕获实例分割色标图。
                is_recording (bool): 是否正在录制。
                use_nvenc (bool): 是否使用 NvEnc 硬件编码。
                nvenc_gpu_index (int): NvEnc GPU 适配器索引。
                random_object_color (bool): 是否随机分配物体颜色。
                width (int): 图像宽度（像素）。
                height (int): 图像高度（像素）。
                vertical_fov (float): 垂直视场角（度）。
                near_clip (float): 近裁剪面距离。
                far_clip (float): 远裁剪面距离。
                gamma (float): 深度相机 gamma 校正。
                color_port (int): RGB 流 WebSocket 端口。
                depth_port (int): 深度流 WebSocket 端口。
                dds_topic (str): DDS 主题。
                dds_stream_id (str): DDS 流 ID。
        """
        self.loop.run_until_complete(
            self._gym.set_camera_sensor_info(
                actor_name, capture_rgb, capture_depth, save_mp4_file, use_dds, **kwargs
            )
        )

    def make_camera_viewport_active(
        self, actor_name: str, entity_name: str
    ) -> None:
        """将指定摄像头设为 Studio 视口激活相机（委托 self._gym）。

        Args:
            actor_name: 摄像头所属 actor 名。
            entity_name: 摄像头实体名（如 "camera_head"）。
        """
        self.loop.run_until_complete(
            self._gym.make_camera_viewport_active(actor_name, entity_name)
        )

    def load_content_file(self, content_file_name, **kwargs) -> None:
        """加载内容文件（委托 self._gym）。

        Args:
            content_file_name: 资源文件名。
            **kwargs: 透传 remote_file_dir/local_file_dir/temp_file_path。
        """
        self.loop.run_until_complete(
            self._gym.load_content_file(content_file_name, **kwargs)
        )

    # --- 公共查询 API（阶段三 3.1.7，全部委托 self._gym 公共方法，K4）---
    # 架构 K4：Env 层查询方法只触 self._gym.<公共方法>，不触 _gym._sim/_registry 等私有。

    def query_joint_qpos(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qpos（委托 self._gym）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> qpos 切片 np.ndarray]。
        """
        return self._gym.query_joint_qpos(joint_names)

    def query_joint_qvel(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qvel（委托 self._gym）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> qvel 切片 np.ndarray]。
        """
        return self._gym.query_joint_qvel(joint_names)

    def query_joint_qacc(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qacc（委托 self._gym）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> qacc 切片 np.ndarray]。
        """
        return self._gym.query_joint_qacc(joint_names)

    def query_joint_offsets(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节偏移（委托 self._gym）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> offset np.ndarray]。
        """
        return self._gym.query_joint_offsets(joint_names)

    def query_joint_lengths(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节长度（委托 self._gym）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> length np.ndarray]。
        """
        return self._gym.query_joint_lengths(joint_names)

    def query_joint_dofadrs(self, joint_names: list[str]) -> dict[str, int]:
        """查询关节 dof 起始地址（委托 self._gym）。

        Args:
            joint_names: 关节名称列表。

        Returns:
            dict[joint_name -> dofadr int]。
        """
        return self._gym.query_joint_dofadrs(joint_names)

    def jnt_qposadr(self, joint_name: str) -> int:
        """查询单关节 qpos 起始地址（委托 self._gym）。

        Args:
            joint_name: 关节名称。

        Returns:
            qpos 起始地址（int）。
        """
        return self._gym.jnt_qposadr(joint_name)

    def jnt_dofadr(self, joint_name: str) -> int:
        """查询单关节 dof 起始地址（委托 self._gym）。

        Args:
            joint_name: 关节名称。

        Returns:
            dof 起始地址（int）。
        """
        return self._gym.jnt_dofadr(joint_name)

    def get_body_xpos_xmat_xquat(
        self, body_name_list: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """查询 body 的 xpos/xmat/xquat（委托 self._gym）。

        Args:
            body_name_list: body 名称列表。

        Returns:
            dict[body_name -> {"xpos": ..., "xmat": ..., "xquat": ...}]。
        """
        return self._gym.query_body_xpos_xmat_xquat(body_name_list)

    def get_body_xpos_xmat_xquat_xvel(
        self, body_name_list: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """查询 body 的 xpos/xmat/xquat/xvel（委托 self._gym）。

        Args:
            body_name_list: body 名称列表。

        Returns:
            dict[body_name -> {"xpos": ..., "xmat": ..., "xquat": ..., "xvel": ...}]。
        """
        return self._gym.query_body_xpos_xmat_xquat_xvel(body_name_list)

    def query_site_pos_and_mat(self, site_names: list[str]) -> dict[str, dict]:
        """查询 site 的 pos 和 mat（委托 self._gym）。

        Args:
            site_names: site 名称列表。

        Returns:
            dict[site_name -> {"pos": ..., "mat": ...}]。
        """
        return self._gym.query_site_pos_and_mat(site_names)

    def query_site_size(self, site_names: list[str]) -> dict[str, np.ndarray]:
        """查询 site 尺寸（委托 self._gym）。

        Args:
            site_names: site 名称列表。

        Returns:
            dict[site_name -> size np.ndarray]。
        """
        return self._gym.query_site_size(site_names)

    def query_sensor_data(self, sensor_names: list[str]) -> dict[str, np.ndarray]:
        """查询传感器数据（委托 self._gym）。

        Args:
            sensor_names: 传感器名称列表。

        Returns:
            dict[sensor_name -> sensordata 切片 np.ndarray]。
        """
        return self._gym.query_sensor_data(sensor_names)

    def query_actuator_torques(self, actuator_names: list[str]) -> dict[str, np.ndarray]:
        """查询执行器力矩（委托 self._gym）。

        Args:
            actuator_names: 执行器名称列表。

        Returns:
            dict[actuator_name -> actuator_force 切片 np.ndarray]。
        """
        return self._gym.query_actuator_torques(actuator_names)

    def query_contact_simple(self) -> list[dict]:
        """查询简单接触信息（委托 self._gym）。

        Returns:
            list[dict]，每个 dict 含 geom1/geom2/dist/pos/frame 键。
        """
        return self._gym.query_contact_simple()

    def query_contact_force(self, contact_ids: list[int]) -> dict[int, np.ndarray]:
        """查询接触力（委托 self._gym）。

        Args:
            contact_ids: 接触索引列表。

        Returns:
            dict[contact_id -> force np.ndarray(6,)]。
        """
        return self._gym.query_contact_force(contact_ids)

    def get_cfrc_ext(self) -> np.ndarray:
        """查询外部约束力 cfrc_ext（委托 self._gym）。

        MuJoCo spatial vector 布局为 [torque(3), force(3)]，即
        ``[mx, my, mz, fx, fy, fz]``。线性力在 ``cfrc[bid, 3:]``。

        Returns:
            np.ndarray，形状 (nbody, 6)。
        """
        return self._gym.get_cfrc_ext()

    def get_goal_bounding_box(self, geom_name: str) -> np.ndarray:
        """查询 geom 尺寸（bounding box，委托 self._gym）。

        Args:
            geom_name: geom 名称。

        Returns:
            np.ndarray(3,)，geom 半尺寸 (hx, hy, hz)。
        """
        return self._gym.get_goal_bounding_box(geom_name)

    def body_subtree_mass(self, body_name: str) -> float:
        """查询 body 子树总质量（委托 self._gym）。

        Args:
            body_name: body 名称。

        Returns:
            body 子树总质量（float 标量）。
        """
        return self._gym.body_subtree_mass(body_name)

    # --- 基座坐标系变换方法（阶段三 3.1.8，纯 NumPy，Env 层实现）---
    # 架构 P2/K4：变换方法依赖 scipy.Rotation + DataView/Model/_gym 公共查询，
    # 不下沉到 SimCore（保持 SimCore 只做 MuJoCo 原生操作），不触 _mjData/_mjModel。
    # 签名与 OrcaGymLocalEnv 完全一致（老代码零改动迁移：gym. -> env.）。

    def query_site_pos_and_quat_B(
        self, site_names: list[str], base_body_list: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """查询 site 相对于基座 body 的位置和四元数（基座坐标系）。

        纯 NumPy 变换：基于 self._gym.query_site_pos_and_mat +
        query_body_xpos_xmat_xquat 公共查询，不触 _mjData/_mjModel（K4/P2）。

        Args:
            site_names: site 名称列表。
            base_body_list: 基座 body 名称列表（取第一个为基座）。

        Returns:
            dict[site_name -> {"xpos": (3,), "xquat": (4,) [w,x,y,z]}]（基座坐标系）。
        """
        site_dict = self._gym.query_site_pos_and_mat(site_names)
        base_result = self._gym.query_body_xpos_xmat_xquat(base_body_list)
        base_name = base_body_list[0]
        base_pos = np.asarray(base_result[base_name]["xpos"], dtype=np.float64)
        base_quat = np.asarray(base_result[base_name]["xquat"], dtype=np.float64)  # [w,x,y,z]
        rot_base = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        rot_base_inv = rot_base.inv()

        result: dict[str, dict[str, np.ndarray]] = {}
        for site_name, site_value in site_dict.items():
            ee_pos = np.asarray(site_value["xpos"], dtype=np.float64)
            ee_quat = mat2quat(np.asarray(site_value["xmat"], dtype=np.float64).reshape(3, 3))  # [w,x,y,z]
            rot_ee = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
            relative_rot = rot_base_inv * rot_ee
            relative_pos = rot_base_inv.apply(ee_pos - base_pos)
            result[site_name] = {
                "xpos": relative_pos,
                "xquat": relative_rot.as_quat()[[3, 0, 1, 2]].astype(np.float32),  # [w,x,y,z]
            }
        return result

    def query_site_xvalp_xvalr(
        self, site_names: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """查询 site 的线速度和角速度（世界坐标系）。

        依赖 self._gym.mj_jac_site（阶段三 3.3.2 实现）+ self.data.qvel。
        纯 NumPy：jacp @ qvel / jacr @ qvel。

        Args:
            site_names: site 名称列表。

        Returns:
            (xvalp_dict, xvalr_dict)：dict[site_name -> 速度 (3,)]。
        """
        query_dict = self._gym.mj_jac_site(site_names)
        xvalp_dict: dict[str, np.ndarray] = {}
        xvalr_dict: dict[str, np.ndarray] = {}
        qvel = self.data.qvel
        for site in query_dict:
            xvalp_dict[site] = np.asarray(query_dict[site]["jacp"]).reshape(3, -1) @ qvel
            xvalr_dict[site] = np.asarray(query_dict[site]["jacr"]).reshape(3, -1) @ qvel
        return xvalp_dict, xvalr_dict

    def query_site_xvalp_xvalr_B(
        self, site_names: list[str], base_body_list: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """查询 site 相对于基座 body 的线速度和角速度（基座坐标系）。

        依赖 self._gym.mj_jac_site（阶段三 3.3.2 实现）+ query_body_xpos_xmat_xquat。
        纯 NumPy：基座逆变换 ⊗ 世界系速度差。

        Args:
            site_names: site 名称列表。
            base_body_list: 基座 body 名称列表（取第一个为基座）。

        Returns:
            (xvalp_dict_B, xvalr_dict_B)：dict[site_name -> 速度 (3,) float32]。
        """
        query_dict = self._gym.mj_jac_site(site_names)
        base_result = self._gym.query_body_xpos_xmat_xquat(base_body_list)
        base_name = base_body_list[0]
        base_mat = np.asarray(base_result[base_name]["xmat"], dtype=np.float64).reshape(3, 3)
        qvel = self.data.qvel

        xvalp_dict: dict[str, np.ndarray] = {}
        xvalr_dict: dict[str, np.ndarray] = {}
        # 固定基座：基座速度为 0
        base_xvalp = np.zeros(3)
        base_xvalr = np.zeros(3)
        for site in query_dict:
            ee_xvalp = np.asarray(query_dict[site]["jacp"]).reshape(3, -1) @ qvel
            ee_xvalr = np.asarray(query_dict[site]["jacr"]).reshape(3, -1) @ qvel
            linear_vel_B = base_mat.T @ (ee_xvalp - base_xvalp)
            angular_vel_B = base_mat.T @ (ee_xvalr - base_xvalr)
            xvalp_dict[site] = linear_vel_B.astype(np.float32)
            xvalr_dict[site] = angular_vel_B.astype(np.float32)
        return xvalp_dict, xvalr_dict

    def query_velocity_body_B(self, ee_body: str, base_body: str) -> np.ndarray:
        """查询 body 相对于基座 body 的速度（基座坐标系）。

        纯 NumPy：基于 self.data.body_cvel（世界系空间速度）+ body_xmat 变换。
        不触 _mjData/_mjModel（K4/P2）。

        Args:
            ee_body: 末端执行器 body 名称。
            base_body: 基座 body 名称。

        Returns:
            combined_vel (6,)：前3线速度，后3角速度（基座坐标系，float32）。
        """
        ee_cvel = np.asarray(self.data.body_cvel(ee_body), dtype=np.float64)  # [ang(3), lin(3)]
        base_cvel = np.asarray(self.data.body_cvel(base_body), dtype=np.float64)
        base_mat = np.asarray(self.data.body_xmat(base_body), dtype=np.float64).reshape(3, 3)
        linear_vel_B = base_mat.T @ (ee_cvel[3:] - base_cvel[3:])
        angular_vel_B = base_mat.T @ (ee_cvel[:3] - base_cvel[:3])
        return np.concatenate([linear_vel_B, angular_vel_B]).astype(np.float32)

    def query_position_body_B(self, ee_body: str, base_body: str) -> np.ndarray:
        """查询 body 相对于基座 body 的位置（基座坐标系）。

        纯 NumPy：基于 self.data.body_xpos/body_xquat 公共查询。

        Args:
            ee_body: 末端执行器 body 名称。
            base_body: 基座 body 名称。

        Returns:
            relative_pos (3,)（基座坐标系）。
        """
        base_pos = np.asarray(self.data.body_xpos(base_body), dtype=np.float64)
        base_quat = np.asarray(self.data.body_xquat(base_body), dtype=np.float64)  # [w,x,y,z]
        ee_pos = np.asarray(self.data.body_xpos(ee_body), dtype=np.float64)
        rot_base = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        relative_pos = rot_base.inv().apply(ee_pos - base_pos)
        return relative_pos

    def query_orientation_body_B(self, ee_body: str, base_body: str) -> np.ndarray:
        """查询 body 相对于基座 body 的姿态（基座坐标系）。

        纯 NumPy：基于 self.data.body_xquat 公共查询。

        Args:
            ee_body: 末端执行器 body 名称。
            base_body: 基座 body 名称。

        Returns:
            relative_quat (4,) [x,y,z,w]（基座坐标系，SciPy 格式，float32）。
        """
        base_quat = np.asarray(self.data.body_xquat(base_body), dtype=np.float64)  # [w,x,y,z]
        ee_quat = np.asarray(self.data.body_xquat(ee_body), dtype=np.float64)
        rot_base = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        rot_ee = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
        relative_rot = rot_base.inv() * rot_ee
        return relative_rot.as_quat().astype(np.float32)

    def query_joint_axes_B(
        self, joint_names: list[str], base_body: str
    ) -> dict[str, np.ndarray]:
        """查询关节轴在基座坐标系中的方向。

        纯 NumPy：基于 self.model.get_joint_byname（Axis/BodyID）+
        self.data.body_xquat 变换。不触 _mjModel/_mjData（K4/P2）。

        Args:
            joint_names: 关节名称列表。
            base_body: 基座 body 名称。

        Returns:
            dict[joint_name -> 轴方向 (3,) float32]（基座坐标系）。
        """
        base_quat = np.asarray(self.data.body_xquat(base_body), dtype=np.float64)  # [w,x,y,z]
        rot_base = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])

        result: dict[str, np.ndarray] = {}
        for joint_name in joint_names:
            joint_info = self.model.get_joint_byname(joint_name)
            jnt_axis = np.asarray(joint_info["Axis"], dtype=np.float64)
            body_id = joint_info["BodyID"]
            body_name = self.model.body_id2name(body_id)
            body_quat = np.asarray(self.data.body_xquat(body_name), dtype=np.float64)
            body_rot = R.from_quat([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
            axis_global = body_rot.apply(jnt_axis)
            axis_base = rot_base.inv().apply(axis_global)
            result[joint_name] = axis_base.astype(np.float32)
        return result

    def query_robot_velocity_odom(
        self, base_body: str, initial_base_pos: np.ndarray, initial_base_quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """查询机器人在里程计坐标系中的速度。

        纯 NumPy：基于 self.data.body_cvel + initial_base_quat 变换。

        Args:
            base_body: 基座 body 名称。
            initial_base_pos: 初始基座位置（未使用，接口一致性保留）。
            initial_base_quat: 初始基座四元数 [w,x,y,z]。

        Returns:
            (linear_vel_odom (3,), angular_vel_odom (3,))（里程计坐标系，float32）。
        """
        base_cvel = np.asarray(self.data.body_cvel(base_body), dtype=np.float64)  # [ang(3), lin(3)]
        initial_rot = R.from_quat(
            [initial_base_quat[1], initial_base_quat[2], initial_base_quat[3], initial_base_quat[0]]
        )
        linear_vel_odom = initial_rot.inv().apply(base_cvel[3:])
        angular_vel_odom = initial_rot.inv().apply(base_cvel[:3])
        return linear_vel_odom.astype(np.float32), angular_vel_odom.astype(np.float32)

    def query_robot_position_odom(
        self, base_body: str, initial_base_pos: np.ndarray, initial_base_quat: np.ndarray
    ) -> np.ndarray:
        """查询机器人在里程计坐标系中的位置。

        纯 NumPy：基于 self.data.body_xpos + initial_base_pos/quat 变换。

        Args:
            base_body: 基座 body 名称。
            initial_base_pos: 初始基座位置 [x,y,z]。
            initial_base_quat: 初始基座四元数 [w,x,y,z]。

        Returns:
            pos_odom (3,)（里程计坐标系，float32）。
        """
        base_pos = np.asarray(self.data.body_xpos(base_body), dtype=np.float64)
        initial_rot = R.from_quat(
            [initial_base_quat[1], initial_base_quat[2], initial_base_quat[3], initial_base_quat[0]]
        )
        relative_pos = base_pos - np.asarray(initial_base_pos, dtype=np.float64)
        pos_odom = initial_rot.inv().apply(relative_pos)
        return pos_odom.astype(np.float32)

    def query_robot_orientation_odom(
        self, base_body: str, initial_base_pos: np.ndarray, initial_base_quat: np.ndarray
    ) -> np.ndarray:
        """查询机器人在里程计坐标系中的姿态。

        纯 NumPy：基于 self.data.body_xquat + initial_base_quat 变换。

        Args:
            base_body: 基座 body 名称。
            initial_base_pos: 初始基座位置（未使用，接口一致性保留）。
            initial_base_quat: 初始基座四元数 [w,x,y,z]。

        Returns:
            quat_odom (4,) [x,y,z,w]（里程计坐标系，SciPy 格式，float32）。
        """
        base_quat = np.asarray(self.data.body_xquat(base_body), dtype=np.float64)  # [w,x,y,z]
        initial_rot = R.from_quat(
            [initial_base_quat[1], initial_base_quat[2], initial_base_quat[3], initial_base_quat[0]]
        )
        base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        rot = initial_rot.inv() * base_rot
        return rot.as_quat().astype(np.float32)

    # --- 力应用与状态设置委托（阶段三 3.2.4）---
    # 架构 K4：按 body_name/site_name 解析 id 后委托 self._gym，不触 _mjData/_mjModel。
    # 签名与 OrcaGymLocalEnv 一致（老代码零改动迁移：gym. -> env.）。

    def apply_body_force(
        self, body_name: str, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """对指定 body 施加外力/力矩（按名称解析 id 后委托 self._gym）。

        Args:
            body_name: body 名称。
            force: 力向量 (3,)。
            torque: 力矩向量 (3,)。
        """
        body_id = self.model.body_name2id(body_name)
        self._gym.apply_body_force(body_id, force, torque)

    def clear_body_force(self, body_name: str) -> None:
        """清除指定 body 的外力。"""
        body_id = self.model.body_name2id(body_name)
        self._gym.clear_body_force(body_id)

    def clear_all_forces(self) -> None:
        """清除所有 body 的外力。"""
        self._gym.clear_all_forces()

    def mj_apply_force_at_site(
        self, site_name: str, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """在 site 处施加力（按名称解析 id 后委托 self._gym）。"""
        site_id = self.model.site_name2id(site_name)
        self._gym.mj_apply_force_at_site(site_id, force, torque)

    def mj_clear_xfrc_applied_for_site(self, site_name: str) -> None:
        """清除 site 关联 body 的 xfrc。"""
        site_id = self.model.site_name2id(site_name)
        self._gym.mj_clear_xfrc_applied_for_site(site_id)

    def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict: dict) -> None:
        """设置 mocap body 位置/四元数（本地写入 + 远端同步）。

        本地写入委托 self._gym.set_mocap_pos_and_quat；
        若为渲染模式且非子环境，则同步到远端 Studio。

        Args:
            mocap_pos_and_quat_dict: dict[body_name -> {"pos": (3,), "quat": (4,)}]。
        """
        self._gym.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)
        send_remote = (
            self._render_mode == "human"
            and not getattr(self, "_is_subenv", False)
        )
        self.loop.run_until_complete(
            self._gym.set_mocap_pos_and_quat_remote(
                mocap_pos_and_quat_dict, send_remote
            )
        )

    def set_geom_friction(self, geom_friction_dict: dict) -> None:
        """设置 geom 摩擦系数。"""
        self._gym.set_geom_friction(geom_friction_dict)

    def add_extra_weight(self, weight_load_dict: dict) -> None:
        """为 body 添加额外重量。"""
        self._gym.add_extra_weight(weight_load_dict)

    # --- 雅可比计算委托（阶段三 3.3.3，Env 层 name→id 解析）---

    def mj_jacBody(
        self, jacp: np.ndarray, jacr: np.ndarray, body_name: str
    ) -> None:
        """计算 body 雅可比（原地写 jacp/jacr，按名称解析 id 后委托 self._gym）。

        Args:
            jacp: 平移雅可比矩阵 (3, nv)，调用方预分配。
            jacr: 旋转雅可比矩阵 (3, nv)，调用方预分配。
            body_name: body 名称。
        """
        body_id = self.model.body_name2id(body_name)
        self._gym.mj_jacBody(jacp, jacr, body_id)

    def mj_jacSite(
        self, jacp: np.ndarray, jacr: np.ndarray, site_name: str
    ) -> None:
        """计算 site 雅可比（原地写 jacp/jacr，按名称解析 id 后委托 self._gym）。

        Args:
            jacp: 平移雅可比矩阵 (3, nv)，调用方预分配。
            jacr: 旋转雅可比矩阵 (3, nv)，调用方预分配。
            site_name: site 名称。
        """
        site_id = self.model.site_name2id(site_name)
        self._gym.mj_jacSite(jacp, jacr, site_id)

    def mj_jac_site(self, site_names: list[str]) -> dict[str, dict]:
        """批量计算 site 雅可比（委托 self._gym）。

        Args:
            site_names: site 名称列表。

        Returns:
            dict[site_name -> {"jacp": np.ndarray(3, nv),
                               "jacr": np.ndarray(3, nv)}]。
        """
        return self._gym.mj_jac_site(site_names)

    # --- 等式约束原语（L1 公共，单次原子读写）---
    # 这组方法不依赖任何 UI 抓取状态字段（_anchored_actor 等），单次原子读写，
    # 调用方自管快照与恢复。UI 抓取内部方法基于本组方法实现。
    # 注：原 update_equality_constraints / modify_equality_objects Env 层公共方法
    # 已删除——前者是 equality_update 的底层实现（SimCore 层保留），后者功能被
    # equality_update(slot, obj1_name=..., obj2_name=...) 覆盖。
    # 阶段2：原 equality_snapshot / equality_bind_mocap / equality_release 公共方法
    # 已删除——编排归消费者（UI 抓取内部方法 _anchor_actor/_release_body_anchored
    # 内联使用本组原语；程序化操作仿照其编排模式）。

    def equality_find_slot_by_body(self, body_name: str) -> int:
        """查找含指定 body 的等式约束槽位索引。

        遍历所有等式约束，返回第一个 obj1_id 或 obj2_id 等于该 body id 的槽位。
        未找到返回 -1。

        Args:
            body_name: body 名称（已含 agent 前缀）。

        .. note::
            本原语不做名称空间解析（对齐架构 §6.6 N1 的分工）。
            调用方应先用 ``env.body("pelvis")`` 解析出带 agent 前缀的完整名称，
            再传入本方法。示例::

                slot = env.equality_find_slot_by_body(env.body("pelvis"))
        """
        body_id = self.model.body_name2id(body_name)
        for i in range(self._gym.n_equality()):
            obj1, obj2 = self._gym.equality_object_ids(i)
            if obj1 == body_id or obj2 == body_id:
                return i
        return -1

    def equality_constraint(self, slot: int) -> dict:
        """读取单个等式约束完整数据（委托 self._gym）。

        返回 type/obj1_id/obj2_id/active/solref/solimp/data。
        单次原子读，不持有状态。消费者需批量读取时自行循环调用本方法。
        """
        return self._gym.equality_constraint(slot)

    def equality_update(
        self,
        slot: int,
        *,
        eq_type: int | None = None,
        obj1_name: str | None = None,
        obj2_name: str | None = None,
        data: np.ndarray | None = None,
        active: bool | None = None,
        solref: np.ndarray | None = None,
        solimp: np.ndarray | None = None,
        forward: bool = True,
    ) -> None:
        """更新指定槽位的等式约束字段（单次原子写 + 可选 mj_forward）。

        只修改显式传入的字段，未传入的字段保留原值。type/obj/data 按当前
        (obj1_id, obj2_id) 匹配槽位写入（底层 SimCore.update_equality_constraints）；
        active/solref/solimp 无匹配语义，按 slot 索引直接写入（SimCore typed
        写入器）。

        Args:
            slot: 等式约束槽位索引。
            eq_type: mjtEq 类型常量（可选）。
            obj1_name: 新的 obj1 body 名称（可选，内部解析为 id）。
            obj2_name: 新的 obj2 body 名称（可选，内部解析为 id）。
            data: 约束数据 np.ndarray（可选，形状 (mjNEQDATA,)）。
            active: 是否激活（可选，写入 eq_active0）。
            solref: 求解器参考参数 (2,)（可选，写入 eq_solref）。
            solimp: 求解器 impedance 参数 (5,)（可选，写入 eq_solimp）。
            forward: 是否在写入后调用 mj_forward()。默认 True，保证 env.data
                一致。若设为 False，调用方需自行调用 env.mj_forward() 才能读取
                一致的状态——这是高级用法，仅用于批量写入多个槽位时避免重复
                forward 的性能优化场景。

        .. warning::
            ``forward=False`` 时写入已生效于 _mjModel，但 ``env.data``
            （OrcaGymDataView）未同步。此时若读取 ``env.data.body_xpos`` 等
            派生量将得到旧值，可能误导后续决策。仅在确认不读取派生量、或
            调用方将立即补 mj_forward() 时使用。

        .. note::
            本原语不做名称空间解析（对齐架构 §6.6 N1 的分工）。
            ``obj1_name`` / ``obj2_name`` 应为已含 agent 前缀的完整名称，
            调用方应先用 ``env.body("pelvis")`` 解析后再传入。
        """
        eq = self.equality_constraint(slot)
        new_type = eq_type if eq_type is not None else eq["type"]
        new_obj1_id = (
            self.model.body_name2id(obj1_name) if obj1_name is not None else eq["obj1_id"]
        )
        new_obj2_id = (
            self.model.body_name2id(obj2_name) if obj2_name is not None else eq["obj2_id"]
        )
        new_data = data if data is not None else eq["data"]
        # type/obj/data 走底层 SimCore.update_equality_constraints（按 (obj1_id, obj2_id) 匹配槽位写入）
        self._gym.update_equality_constraints([{
            "type": new_type,
            "obj1_id": eq["obj1_id"],      # 用于匹配（当前值）
            "obj2_id": eq["obj2_id"],      # 用于匹配（当前值）
            "new_obj1_id": new_obj1_id,
            "new_obj2_id": new_obj2_id,
            "data": new_data,
        }])
        # active/solref/solimp 无匹配语义，按 slot 索引直接写
        # （通过 SimCore typed 写入器委托，避免 Env 穿墙 _mjModel）
        if active is not None:
            self._gym.set_equality_active(slot, active)
        if solref is not None:
            self._gym.set_equality_solref(slot, solref)
        if solimp is not None:
            self._gym.set_equality_solimp(slot, solimp)
        if forward:
            self.mj_forward()

    def _anchor_actor(
        self,
        actor_name: str,
        anchor_type: str = "weld",
    ) -> None:
        """【内部 API】UI 抓取专用：锚定 actor body。

        .. warning::
            此方法是 Studio UI 抓取的内部实现，由 ``_do_body_manipulation``
            调用。AI 和用户代码**不应直接调用**此方法。
            程序化操作请仿照本方法编排模式使用公共原语
            (:meth:`equality_find_slot_by_body` / :meth:`equality_constraint` /
            :meth:`equality_update` / :meth:`set_mocap_pos_and_quat`)。

        使用 Studio 系统自带的 ActorManipulator_Anchor mocap body，
        对齐 OrcaGymLocalEnv 的 anchor_actor 语义。

        编排完全基于公共无状态原语（不依赖已删除的 equality_bind_mocap）：
        - equality_find_slot_by_body 查找槽位
        - equality_constraint 保存原始快照
        - set_mocap_pos_and_quat 对齐 mocap 位姿到 actor
        - equality_update 写入约束（type/obj，内部 mj_forward）

        Args:
            actor_name: 被锚定的 body 名称。
            anchor_type: 锚点类型 "weld"/"connect"（"ball" 等价 "connect"）。
        """
        import mujoco

        # 1. 查找 UI 抓取专用 mocap 的 equality 槽位
        slot = self.equality_find_slot_by_body(self._anchor_mocap_name)
        if slot == -1:
            raise ValueError(
                f"模型中无含 {self._anchor_mocap_name} 的 equality 槽位，"
                f"请检查关卡 XML"
            )
        # 2. 保存原始约束快照（释放时恢复）
        self._anchor_original_eq = self.equality_constraint(slot)
        # 3. 对齐 mocap 位姿到 actor 当前位姿（避免下一帧拉扯）
        mocap_id = self.model.body_name2id(self._anchor_mocap_name)
        actor_pose = self.get_body_xpos_xmat_xquat([actor_name])[actor_name]
        self.set_mocap_pos_and_quat({
            self._anchor_mocap_name: {
                "pos": actor_pose["xpos"],
                "quat": actor_pose["xquat"],
            }
        })
        # 4. 确定改 obj1 还是 obj2（mocap 一端保持，另一端改为 actor）
        if self._anchor_original_eq["obj1_id"] == mocap_id:
            new_obj1_name = self._anchor_mocap_name
            new_obj2_name = actor_name
        else:
            new_obj1_name = actor_name
            new_obj2_name = self._anchor_mocap_name
        # 5. eq_type 字符串 → mjtEq 常量
        type_map = {
            "weld": mujoco.mjtEq.mjEQ_WELD,
            "connect": mujoco.mjtEq.mjEQ_CONNECT,
            "ball": mujoco.mjtEq.mjEQ_CONNECT,
        }
        mujoco_eq_type = type_map.get(anchor_type, mujoco.mjtEq.mjEQ_CONNECT)
        # 6. 写入约束（公共原语，内部 mj_forward）
        self.equality_update(
            slot,
            eq_type=mujoco_eq_type,
            obj1_name=new_obj1_name,
            obj2_name=new_obj2_name,
        )
        self._anchored_actor = actor_name
        self._anchor_type = anchor_type

    def _release_body_anchored(self) -> None:
        """【内部 API】UI 抓取专用：释放锚定的 actor。

        .. warning::
            此方法是 Studio UI 抓取的内部实现，由 ``_do_body_manipulation``
            调用。AI 和用户代码**不应直接调用**此方法。
            程序化操作请仿照本方法编排模式使用公共原语
            (:meth:`equality_find_slot_by_body` / :meth:`equality_update`)。

        通过恢复 XML 原始 obj id 实现，对齐 Local 的 dummy body 机制：
        约束不再作用于被锚定 actor，actor 恢复自由动力学。未锚定时 no-op。

        编排完全基于公共无状态原语（不依赖已删除的 equality_release，
        不穿墙底层 update_equality_constraints）：
        - equality_find_slot_by_body 查找当前绑定槽位
        - equality_update 从快照恢复原始约束（id→name 反查）
        """
        if self._anchored_actor is None:
            return
        if self._anchor_original_eq is not None:
            slot = self.equality_find_slot_by_body(self._anchored_actor)
            if slot != -1:
                # 从快照恢复原始约束（id→name 反查 + equality_update，不穿墙）
                self.equality_update(
                    slot,
                    eq_type=self._anchor_original_eq["type"],
                    obj1_name=self.model.body_id2name(
                        self._anchor_original_eq["obj1_id"]
                    ),
                    obj2_name=self.model.body_id2name(
                        self._anchor_original_eq["obj2_id"]
                    ),
                    data=self._anchor_original_eq["data"],
                )
        self._anchored_actor = None
        self._anchor_type = None
        self._anchor_original_eq = None

    # --- 只读查询委托（阶段三 3.2.4，K4：通过公共方法而非 _gym 穿墙）---

    def geom_friction(self, geom_name: str) -> np.ndarray:
        """查询 geom 摩擦系数 (3,) [sliding, torsion, rolling]（只读视图）。

        替代直接访问 _mjModel.geom_friction[id]。

        Args:
            geom_name: geom 名称。

        Returns:
            geom 摩擦系数数组，形状 (3,)。
        """
        return self._gym.geom_friction(geom_name)

    # --- Gymnasium 接口（子类实现）---

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], float, bool, bool, Dict[str, float]]:
        """Gymnasium 标准步进接口（子类必须复写）。

        标准实现模板::

            self.do_simulation(action, self.frame_skip)   # 或 PD 循环见架构 §6.4 S6
            obs = self._get_obs()
            reward = ...
            terminated = ...
            truncated = self._step_count >= self.MAX_EPISODE_STEPS
            info = {"time": float(self.data.time)}
            return obs, reward, terminated, truncated, info

        Locomotion PD 控制模板（架构 §6.4 S6）::

            target = self._action_to_target(action)
            for _ in range(self.frame_skip):
                ctrl = self._pd_controller(self.data.qpos, self.data.qvel, target)
                self.do_simulation(ctrl, 1)   # frame_skip=1，精细 PD 步进
            obs = self._get_obs()
            return obs, reward, terminated, truncated, info

        禁止:
            不要在外部运行循环里绕过 step() 直接调 do_simulation（架构 §6.4 S5）。
            不要复写 do_simulation 作为步进主路径，应在 step() 内调用它。
        """
        raise NotImplementedError("step 待子类实现")

    def reset_model(self) -> tuple[dict, dict]:
        """Gymnasium MuJoCo 标准 hook（子类必须复写，由 reset() 调用）。

        标准实现模板::

            qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
            self.set_joint_qpos(qpos)
            self.set_joint_qvel(qvel)
            self.mj_forward()       # 更新派生量
            self._sync_view()       # 同步到 DataView
            return self._get_obs(), {}

        说明:
            - 这是 Gym MujocoEnv 十年公开 hook 约定，不要直接复写 reset()。
            - reset() 由 OrcaGymEnvMixin 编排（seed + reset_simulation + reset_model + render）。
        """
        raise NotImplementedError("reset_model 待子类实现")

    def _get_obs(self) -> dict:
        """Gymnasium MuJoCo 标准 hook（子类必须复写，step 与 reset_model 共用）。

        标准实现模板::

            theta = float(self.data.qpos[0])
            theta_dot = float(self.data.qvel[0])
            return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

        说明:
            - ``_`` 前缀表示 protected（类族内部），子类复写是 Python 常规操作。
            - 不要改名为 ``get_obs``，保持与 Gym MujocoEnv 命名一致。
        """
        raise NotImplementedError("_get_obs 待子类实现")
