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
from orca_gym.utils.rotations import mat2quat
from ..orca_gym_env_mixin import OrcaGymEnvMixin


class OrcaGymEulerEnv(OrcaGymEnvMixin, gym.Env):
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
        env.gym/env.stub/env.channel 不存在（直接继承 gym.Env，无此属性）。
        缺少功能时，扩展本类的公共方法。
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

        # 5. 锚点状态（阶段三 3.5.4 anchor_actor/release_body_anchored 使用）
        self._anchor_mocap_name: str | None = None  # 锚点 mocap body 名称
        self._anchored_actor: str | None = None      # 当前锚定的 actor 名称
        self._anchor_type: str | None = None         # 当前锚点类型

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
                self.do_body_manipulation()
                self._render_count -= 1.0
        else:
            time_diff = time.perf_counter() - self._render_time_step
            if time_diff > self._render_interval:
                self._render_time_step = time.perf_counter()
                self.loop.run_until_complete(self._gym.render())
                self.do_body_manipulation()
        return None

    # --- 体操作编排（阶段三 3.5.6，锚定 + mocap 移动 + 释放编排）---

    def do_body_manipulation(self) -> None:
        """Studio UI 体操作编排：根据 UI 状态执行锚定/移动/释放。

        完整流程（基于 Studio body manipulation 状态）：
        1. 读取 body manipulation 状态（self._gym.get_body_manipulation_state）
        2. 若 Studio 无锚定 body 且 Env 已锚定：release_body_anchored
        3. 若 Studio 有锚定 body 且 Env 未锚定：anchor_actor
        4. 若已锚定且有 UI 拖拽位姿：set_mocap_pos_and_quat（跟随 UI 拖拽）

        走合规 API：anchor_actor / release_body_anchored / set_mocap_pos_and_quat
        / get_body_manipulation_state。离线模式 no-op。
        """
        if self._skip_grpc_load:
            return
        manip_state = self.loop.run_until_complete(
            self._gym.get_body_manipulation_state()
        )
        actor_name = manip_state["actor_name"]
        anchor_type = manip_state["anchor_type"]
        # 1. 处理锚定/释放事件
        if actor_name is None:
            if self._anchored_actor is not None:
                self.release_body_anchored()
            return
        if self._anchored_actor is None:
            self.anchor_actor(actor_name, anchor_type or "weld")
        # 2. 已锚定时同步 mocap 到 UI 拖拽位姿
        if self._anchored_actor is not None and manip_state.get("mocap_pose"):
            self.set_mocap_pos_and_quat(
                {self._anchor_mocap_name: manip_state["mocap_pose"]}
            )

    # --- Studio 桥接访问器（K9 方法访问模式，替代 gym.studio 穿墙）---

    def studio_bridge(self):
        """返回 OrcaStudio 桥接对象（K9 方法访问模式）。

        替代 gym.studio property 式穿墙。
        """
        return self._studio_bridge

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

    # --- 等式约束委托（阶段三 3.5.3，Env 层 name→id 解析）---

    def update_equality_constraints(self, eq_list: list[dict]) -> None:
        """更新等式约束（Env 层 name→id 解析后委托 self._gym）。

        Args:
            eq_list: 等式约束列表，每项可含 obj1_name/obj2_name（Env 层解析为 id）
                或 obj1_id/obj2_id（直接使用）。type/data 字段原样透传。
        """
        resolved = []
        for eq in eq_list:
            eq_r = dict(eq)
            if "obj1_name" in eq_r:
                eq_r["obj1_id"] = self.model.body_name2id(eq_r.pop("obj1_name"))
            if "obj2_name" in eq_r:
                eq_r["obj2_id"] = self.model.body_name2id(eq_r.pop("obj2_name"))
            resolved.append(eq_r)
        self._gym.update_equality_constraints(resolved)

    def modify_equality_objects(
        self,
        eq_ids: list[int],
        obj1_names=None,
        obj2_names=None,
    ) -> None:
        """修改等式约束关联对象（Env 层 name→id 解析后委托 self._gym）。

        Args:
            eq_ids: 等式约束索引列表。
            obj1_names: 新的 obj1 body 名称列表（None 不修改）。
            obj2_names: 新的 obj2 body 名称列表（None 不修改）。
        """
        obj1_ids = (
            [self.model.body_name2id(n) for n in obj1_names] if obj1_names else None
        )
        obj2_ids = (
            [self.model.body_name2id(n) for n in obj2_names] if obj2_names else None
        )
        self._gym.modify_equality_objects(eq_ids, obj1_ids, obj2_ids)

    def update_anchor_equality_constraints(
        self, actor_name: str, anchor_type: str = "weld"
    ) -> None:
        """锚点约束更新（connect/weld 联动 actor 与 mocap body）。

        组装 eq_list（含 actor_id、mocap_id、anchor_type），委托 self._gym。
        anchor_type: "weld"（焊接）/ "connect"（球关节）/ "none"（释放）。

        Args:
            actor_name: 被锚定的 body 名称。
            anchor_type: 锚点类型 "weld"/"connect"/"none"。
        """
        import mujoco

        actor_id = self.model.body_name2id(actor_name)
        # 查找锚点 mocap body（第一个 mocap body）
        mocap_names = self._gym.mocap_body_names()
        if not mocap_names:
            raise ValueError("模型中无 mocap body，无法锚定")
        mocap_name = getattr(self, "_anchor_mocap_name", None) or mocap_names[0]
        mocap_id = self.model.body_name2id(mocap_name)
        # 映射 anchor_type → mujoco eq type
        if anchor_type == "weld":
            eq_type = mujoco.mjtEq.mjEQ_WELD
        elif anchor_type in ("connect", "ball"):
            eq_type = mujoco.mjtEq.mjEQ_CONNECT
        else:
            eq_type = mujoco.mjtEq.mjEQ_CONNECT
        # 组装 eq_list（写入 eq[0]）
        eq_data = np.zeros(mujoco.mjNEQDATA)
        eq_list = [
            {
                "type": eq_type,
                "obj1_id": mocap_id,
                "obj2_id": actor_id,
                "data": eq_data,
            }
        ]
        self._gym.update_equality_constraints(eq_list)

    # --- 体操作（阶段三 3.5.4，mocap + equality 联动）---

    def anchor_actor(self, actor_name: str, anchor_type: str = "weld") -> None:
        """锚定 actor body：设置 mocap 位姿 + 建立 weld/connect 等式约束。

        走合规 API：
        - get_body_xpos_xmat_xquat（查询 actor 当前位姿）
        - set_mocap_pos_and_quat（设置 mocap 位姿到 actor 当前位姿）
        - update_anchor_equality_constraints（建立约束）

        Args:
            actor_name: 被锚定的 body 名称。
            anchor_type: 锚点类型 "weld"/"connect"。
        """
        # 1. 查询 actor 当前位姿
        actor_pose = self.get_body_xpos_xmat_xquat([actor_name])[actor_name]
        # 2. 查找锚点 mocap body 名称（缓存到 _anchor_mocap_name）
        if self._anchor_mocap_name is None:
            mocap_names = self._gym.mocap_body_names()
            if not mocap_names:
                raise ValueError("模型中无 mocap body，无法锚定")
            self._anchor_mocap_name = mocap_names[0]
        # 3. 设置 mocap body 到 actor 当前位姿
        mocap_dict = {
            self._anchor_mocap_name: {
                "pos": actor_pose["xpos"],
                "quat": actor_pose["xquat"],
            }
        }
        self.set_mocap_pos_and_quat(mocap_dict)
        # 4. 建立 weld/connect 等式约束（actor ↔ mocap）
        self.update_anchor_equality_constraints(actor_name, anchor_type)
        # 5. 记录锚定状态
        self._anchored_actor = actor_name
        self._anchor_type = anchor_type

    # --- 体操作（阶段三 3.5.5，释放锚定 actor）---

    def release_body_anchored(self) -> None:
        """释放锚定的 actor：清除锚点等式约束 + 清除锚定状态。

        走合规 API：
        - self._gym.update_equality_constraints（将锚点约束 type 清零）

        未锚定时调用为 no-op。
        """
        if self._anchored_actor is None:
            return
        import mujoco

        # 1. 清除锚点等式约束（type 清零 + 数据清零）
        n_eq = self._gym.n_equality()
        if n_eq > 0:
            release_list = [
                {
                    "type": 0,
                    "obj1_id": -1,
                    "obj2_id": -1,
                    "data": np.zeros(mujoco.mjNEQDATA),
                }
                for _ in range(n_eq)
            ]
            self._gym.update_equality_constraints(release_list)
        # 2. 清除锚定状态
        self._anchored_actor = None
        self._anchor_type = None

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
        """执行一个环境步进（子类实现）。"""
        raise NotImplementedError("step 待子类实现")

    def reset_model(self) -> tuple[dict, dict]:
        """重置机器人自由度（子类实现）。"""
        raise NotImplementedError("reset_model 待子类实现")

    def _get_obs(self) -> dict:
        """获取观测（子类实现）。"""
        raise NotImplementedError("_get_obs 待子类实现")
