"""OrcaGymEuler — 仿真核心 Facade，组合子组件（阶段二填充）。

本模块属于 OrcaGym Euler 体系阶段二（P2-Step4 骨架 + P4-Step5 委托填充），
组合 MuJoCoSimCore/OrcaStudioBridge/ModelRegistry/SimConfig/OrcaGymDataView
子组件，实现隔离机制（K3/K5/K8/K9）。

阶段二 Step 5 填充委托方法，将 Env 调用转发到
MuJoCoSimCore/ModelRegistry/SimConfig，并实现 model property 返回缓存的
OrcaGymModel。

核心设计:
    - 不暴露 _mjModel/_mjData（K3）—— 通过 __getattribute__ 拦截
    - 不暴露子组件对象（K5）—— _sim/_studio 等带下划线，__getattribute__ 拦截
    - Studio 交互通过方法 studio_bridge() 而非 property（K9）
    - Euler 耦合查询通过 has_euler()/step_with_coupling()（K8）

隔离机制说明:
    架构 §7.2 使用 __getattr__ 拦截 _mjData/_mjModel（不存储在 Gym 上）。
    本类将 _BLOCKED_ATTRS 扩展到子组件名（_sim/_studio 等，存储在 __dict__），
    因此使用 __getattribute__ 拦截（__getattr__ 仅在属性查找失败时触发，
    无法拦截已存在于 __dict__ 的属性）。内部访问通过 object.__getattribute__
    绕过拦截。
"""

import numpy as np

from orca_gym.core.euler.mujoco_sim_core import MuJoCoSimCore
from orca_gym.core.euler.orca_studio_bridge import AnchorType, OrcaStudioBridge
from orca_gym.core.euler.model_registry import ModelRegistry
from orca_gym.core.euler.sim_config import SimConfig
from orca_gym.core.euler.orca_gym_data_view import OrcaGymDataView


class OrcaGymEuler:
    """仿真核心 Facade，组合子组件，不暴露 _mjModel/_mjData，不暴露子组件对象。

    ┌─────────────────────────────────────────────────────────────┐
    │  API 契约：用户不应直接访问 _mjData / _mjModel / 任何子组件。│
    │  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView）        │
    │  写入外力 → 使用 env.apply_body_force()                     │
    │  配置求解器 → 使用 env.sim_config                           │
    │  缺少功能时 → 扩展 OrcaGymEulerEnv 公共方法                 │
    └─────────────────────────────────────────────────────────────┘

    使用契约:
        读状态:     env.data.qpos / env.data.body_xpos("link1")
        写外力:     env.apply_body_force("link1", force, torque)
        配置:       env.sim_config.timestep = 0.002
        步进:       env.mj_step(nstep=1)
        Studio:     env.studio_bridge().render(...)  # 方法，非 property

    禁止:
        不要访问 gym._mjData / gym._mjModel / gym._sim / gym._studio 等。
        不要通过 @property 暴露 studio/sim/opt/view/euler 子组件。
    """

    # K3/K5: 隔离机制 — 拦截引擎内部和子组件对象
    _BLOCKED_ATTRS = frozenset({
        # L3 引擎内部
        "_mjData", "_mjModel", "mj_data", "mj_model",
        "_mj_data", "_mj_model", "mjData", "mjModel",
        # K5: 子组件对象也不对外暴露
        "_sim", "_studio", "_registry", "_opt", "_view", "_euler",
        "sim", "studio", "registry", "opt", "view", "euler",
    })

    def __init__(self, stub=None) -> None:
        """初始化仿真核心 Facade。

        组合所有子组件，全部带下划线（不在 __dir__ 暴露，被 __getattribute__ 拦截）。

        Args:
            stub: OrcaStudio gRPC stub，传递给 OrcaStudioBridge。
        """
        # 内部组件（全部带下划线，不在 __dir__ 暴露，访问被 __getattribute__ 拦截）
        self._sim = MuJoCoSimCore()
        self._studio = OrcaStudioBridge(stub=stub)
        self._registry = ModelRegistry()
        self._opt = SimConfig()
        self._view = OrcaGymDataView()
        self._euler = None    # EulerOrchestrator | None（骨架阶段恒为 None）
        self._orca_model = None  # OrcaGymModel | None（init_simulation 后填充，model property 返回缓存）

    # --- K3/K5: 隔离机制 ---

    def __getattribute__(self, name: str):
        """拦截 _BLOCKED_ATTRS 的外部访问，返回引导性错误。

        使用 __getattribute__（而非 __getattr__）是因为子组件名（_sim 等）
        存储在 __dict__ 中，__getattr__ 仅在属性查找失败时触发，无法拦截。
        内部访问通过 object.__getattribute__ 绕过本拦截。
        """
        blocked = object.__getattribute__(self, "_BLOCKED_ATTRS")
        if name in blocked:
            # 针对不同违规类型给出精准引导
            if name in ("_euler", "euler"):
                euler_hint = (
                    "  Euler 耦合查询 → 使用 env.has_euler() / env.step_with_coupling()\n"
                )
            elif name in ("_studio", "studio"):
                euler_hint = "  Studio 交互 → 使用 env.studio_bridge()\n"
            elif name in ("_sim", "sim"):
                euler_hint = (
                    "  仿真步进 → 使用 env.mj_step() / env.mj_forward() / env.do_simulation()\n"
                )
            elif name in ("_opt", "opt"):
                euler_hint = "  求解器配置 → 使用 env.sim_config\n"
            elif name in ("_view", "view"):
                euler_hint = "  状态读取 → 使用 env.data（OrcaGymDataView）\n"
            else:
                # L3 引擎内部 _mjData/_mjModel 等
                euler_hint = ""
            raise AttributeError(
                f"'{type(self).__name__}' 对象的属性 '{name}' 被隔离。\n"
                f"  API 契约：用户不应直接访问 _mjData / _mjModel / 任何子组件。\n"
                f"  读取 MuJoCo 状态 → 使用 env.data（OrcaGymDataView），如 env.data.qpos\n"
                f"  写入外力 → 使用 env.apply_body_force()\n"
                f"  配置求解器 → 使用 env.sim_config\n"
                f"{euler_hint}"
                f"  缺少功能时 → 扩展 OrcaGymEulerEnv 公共方法，不要直接访问内部对象。"
            )
        return object.__getattribute__(self, name)

    def __dir__(self) -> list[str]:
        """只列出公共 API，不含子组件对象或引擎内部。"""
        result = super().__dir__()
        blocked = self._BLOCKED_ATTRS
        return [name for name in result if name not in blocked]

    # --- 生命周期 ---

    async def init_simulation(self, model_xml_path: str) -> None:
        """初始化仿真：加载模型、绑定 SimConfig/ModelRegistry、同步 DataView。

        Args:
            model_xml_path: MuJoCo 模型 XML 文件路径。
        """
        sim = object.__getattribute__(self, "_sim")
        opt = object.__getattribute__(self, "_opt")
        registry = object.__getattribute__(self, "_registry")
        view = object.__getattribute__(self, "_view")

        sim.init_simulation(model_xml_path)
        # 绑定 SimConfig/ModelRegistry 到真实 mjModel
        opt._bind(sim._mjModel)           # noqa: SLF001  core 层组件编排：Euler 绑定 SimConfig
        registry._bind(sim._mjModel)      # noqa: SLF001  core 层组件编排：Euler 绑定 ModelRegistry
        # 缓存 OrcaGymModel（构建一次，后续 model property 返回缓存）
        object.__setattr__(self, "_orca_model", registry.build_orca_gym_model())
        # 首次同步 DataView
        sim.sync_to_view(view)

    async def load_model_xml(self) -> str:
        """加载模型 XML（在线模式从 Studio 拉取，离线模式返回本地路径）。

        Returns:
            MuJoCo 模型 XML 字符串。
        """
        studio = object.__getattribute__(self, "_studio")
        return await studio.load_model_xml()

    # --- 仿真控制（委托 _sim）---

    def mj_step(self, nstep: int) -> None:
        """执行 nstep 步 MuJoCo 仿真。

        Args:
            nstep: 步进次数。
        """
        object.__getattribute__(self, "_sim").step(nstep)

    def mj_forward(self) -> None:
        """执行 MuJoCo 前向计算（不步进，仅更新派生量）。"""
        object.__getattribute__(self, "_sim").forward()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """设置控制输入，应用 override_ctrls（如果存在）。

        override 逻辑在 Gym 层：从 Bridge 取 override_ctrls，应用到 ctrl 后
        再传给 SimCore。保持 MuJoCoSimCore.set_ctrl 的纯净（只写 _mjData.ctrl）。

        Args:
            ctrl: 控制输入数组。
        """
        studio = object.__getattribute__(self, "_studio")
        overrides = studio.get_override_ctrls()
        if overrides:
            ctrl = ctrl.copy()
            for idx, value in overrides.items():
                if 0 <= idx < len(ctrl):
                    ctrl[idx] = value
        object.__getattribute__(self, "_sim").set_ctrl(ctrl)

    def set_qpos_qvel(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """设置广义坐标和速度（供 set_joint_qpos/qvel 使用）。

        Args:
            qpos: 广义坐标数组。
            qvel: 广义速度数组。
        """
        object.__getattribute__(self, "_sim").set_qpos_qvel(qpos, qvel)

    def reset_data(self) -> None:
        """重置 MjData 到初始状态。"""
        object.__getattribute__(self, "_sim").reset_data()

    # --- 状态同步 ---

    def sync_to_view(self) -> None:
        """将 MuJoCo 状态同步到 OrcaGymDataView（env.data）。"""
        object.__getattribute__(self, "_sim").sync_to_view(
            object.__getattribute__(self, "_view")
        )

    # --- K5/K6: 状态访问（返回 typed 对象，不返回子组件引用）---

    @property
    def data(self) -> OrcaGymDataView:
        """返回 MuJoCo 状态只读视图（OrcaGymDataView）。

        替代直接访问 _mjData。
        """
        return object.__getattribute__(self, "_view")

    @property
    def model(self):
        """返回缓存的 OrcaGymModel。

        init_simulation 后构建一次并缓存，后续访问直接返回缓存。
        """
        return object.__getattribute__(self, "_orca_model")

    @property
    def nq(self) -> int:
        """广义坐标维度（qpos 维度）。"""
        return object.__getattribute__(self, "_sim").nq

    @property
    def nu(self) -> int:
        """控制输入维度（ctrl 维度）。"""
        return object.__getattribute__(self, "_sim").nu

    @property
    def sim_config(self) -> SimConfig:
        """返回求解器配置（SimConfig）。

        替代直接访问 _mjModel.opt.*。
        """
        return object.__getattribute__(self, "_opt")

    # --- K9: Studio 桥接访问（方法而非 property）---

    def studio_bridge(self) -> OrcaStudioBridge:
        """返回 OrcaStudio 桥接对象。

        K9: 通过方法访问而非 @property，防止 gym.studio 式穿墙。
        禁止: 不提供 studio 的 property 定义。
        """
        return object.__getattribute__(self, "_studio")

    # --- Studio 委托（骨架最小集）---

    async def render(self) -> None:
        """渲染当前仿真状态到 OrcaStudio。

        从 DataView 读取 qpos/time（不直接触 _mjData），委托到 studio.render。
        """
        view = object.__getattribute__(self, "_view")
        studio = object.__getattribute__(self, "_studio")
        await studio.render(view.qpos, view.time)

    async def pause_simulation(self) -> None:
        """通知 OrcaStudio 暂停仿真。"""
        await object.__getattribute__(self, "_studio").pause_simulation()

    # --- Studio 委托（阶段三 3.4.4，委托 _studio bridge）---

    async def begin_save_video(self, file_path: str, capture_mode) -> None:
        """开始录制视频（委托 bridge）。"""
        await object.__getattribute__(self, "_studio").begin_save_video(
            file_path, capture_mode
        )

    async def stop_save_video(self) -> None:
        """停止录制视频（委托 bridge）。"""
        await object.__getattribute__(self, "_studio").stop_save_video()

    async def get_current_frame(self) -> int:
        """获取当前帧号（委托 bridge）。"""
        return await object.__getattribute__(self, "_studio").get_current_frame()

    async def get_camera_time_stamp(self, last_frame_index: int) -> dict:
        """获取相机时间戳（委托 bridge）。"""
        return await object.__getattribute__(
            self, "_studio"
        ).get_camera_time_stamp(last_frame_index)

    async def get_frame_png(self, image_path: str) -> None:
        """获取帧 PNG（委托 bridge）。"""
        await object.__getattribute__(self, "_studio").get_frame_png(image_path)

    # --- 摄像头传感器激活（阶段四补遗，委托 _studio bridge）---

    async def set_camera_sensor_info(
        self,
        actor_name: str,
        capture_rgb: bool,
        capture_depth: bool,
        save_mp4_file: bool = False,
        use_dds: bool = False,
        **kwargs,
    ) -> None:
        """激活/配置摄像头传感器流（委托 bridge）。

        Args:
            actor_name: 摄像头所属 actor 名。
            capture_rgb: 是否激活 RGB 视频流。
            capture_depth: 是否激活深度视频流。
            save_mp4_file: 是否同时保存 MP4 文件。
            use_dds: 是否使用 DDS 传输。
            **kwargs: 扩展 optional 参数（capture_normal/capture_object_color/
                is_recording/use_nvenc/nvenc_gpu_index/random_object_color/
                width/height/vertical_fov/near_clip/far_clip/gamma/
                color_port/depth_port/dds_topic/dds_stream_id），
                None 表示不修改现有值。
        """
        await object.__getattribute__(self, "_studio").set_camera_sensor_info(
            actor_name, capture_rgb, capture_depth, save_mp4_file, use_dds, **kwargs
        )

    async def make_camera_viewport_active(
        self, actor_name: str, entity_name: str
    ) -> None:
        """将指定摄像头设为 Studio 视口激活相机（委托 bridge）。

        Args:
            actor_name: 摄像头所属 actor 名。
            entity_name: 摄像头实体名。
        """
        await object.__getattribute__(
            self, "_studio"
        ).make_camera_viewport_active(actor_name, entity_name)

    async def load_content_file(
        self,
        content_file_name: str,
        remote_file_dir: str = "",
        local_file_dir: str = "",
        temp_file_path: str | None = None,
    ) -> None:
        """加载内容文件（委托 bridge）。"""
        await object.__getattribute__(self, "_studio").load_content_file(
            content_file_name,
            remote_file_dir=remote_file_dir,
            local_file_dir=local_file_dir,
            temp_file_path=temp_file_path,
        )

    # --- Studio 体操作状态查询（阶段三 3.5.6，委托 _studio bridge）---

    async def get_body_manipulation_state(self) -> dict:
        """查询 Studio 体操作状态（委托 bridge），组装为结构化 dict。

        依赖反转：bridge 返回 (body_name, anchor_type) 与 movement dict，
        本方法组装为 Env 编排可直接消费的结构。

        Returns:
            dict 含键：
                - actor_name: str | None（Studio 当前锚定的 body 名，无则 None）。
                - anchor_type: str | None（"weld"/"connect"/None）。
                - mocap_pose: {"pos": np.ndarray(3,), "quat": np.ndarray(4,)}。
                  UI 拖拽目标位姿（bridge 的 movement 字段为绝对目标位姿）。
        """
        studio = object.__getattribute__(self, "_studio")
        body_name, anchor_type = await studio.get_body_manipulation_anchored()
        movement = await studio.get_body_manipulation_movement()
        if anchor_type == AnchorType.WELD:
            anchor_type_str = "weld"
        elif anchor_type == AnchorType.BALL:
            anchor_type_str = "connect"
        else:
            anchor_type_str = None
        actor_name = body_name if body_name else None
        return {
            "actor_name": actor_name,
            "anchor_type": anchor_type_str,
            "mocap_pose": {
                "pos": movement["delta_pos"],
                "quat": movement["delta_quat"],
            },
        }

    # --- K8: 步进耦合查询（供 do_simulation 使用，不暴露 _euler）---

    def has_euler(self) -> bool:
        """查询是否存在 Euler 耦合编排器。

        骨架阶段恒返回 False（_euler 为 None）。

        Returns:
            False（骨架阶段无 Euler）。
        """
        return object.__getattribute__(self, "_euler") is not None

    def step_with_coupling(self, ctrl: np.ndarray, n_frames: int, dt: float) -> None:
        """带 Euler 耦合的步进（骨架阶段无 Euler，等价于纯 MuJoCo 步进）。

        供 do_simulation 使用，替代 do_simulation 内部直接读 self._gym._euler。
        has_euler()=False 时等价于 set_ctrl + step。后续 Euler 耦合实现时扩展。

        Args:
            ctrl: 控制输入数组。
            n_frames: 帧数。
            dt: 时间步长。
        """
        sim = object.__getattribute__(self, "_sim")
        sim.set_ctrl(ctrl)
        sim.step(n_frames)

    # --- 查询委托（阶段三 3.1.6，全部经 object.__getattribute__ 访问子组件）---
    # 架构 K3：委托方法必须用 object.__getattribute__(self, "_sim"/...) 访问
    # 子组件，不得直接 self._sim（被 __getattribute__ 拦截）。

    def query_joint_qpos(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qpos（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_joint_qpos(joint_names)

    def query_joint_qvel(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qvel（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_joint_qvel(joint_names)

    def query_joint_qacc(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节 qacc（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_joint_qacc(joint_names)

    def query_joint_offsets(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节偏移（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_joint_offsets(joint_names)

    def query_joint_lengths(self, joint_names: list[str]) -> dict[str, np.ndarray]:
        """查询关节长度（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_joint_lengths(joint_names)

    def query_joint_dofadrs(self, joint_names: list[str]) -> dict[str, int]:
        """查询关节 dof 起始地址（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_joint_dofadrs(joint_names)

    def jnt_qposadr(self, joint_name: str) -> int:
        """查询单关节 qpos 起始地址（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").jnt_qposadr(joint_name)

    def jnt_dofadr(self, joint_name: str) -> int:
        """查询单关节 dof 起始地址（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").jnt_dofadr(joint_name)

    def query_body_xpos_xmat_xquat(
        self, body_name_list: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """查询 body 的 xpos/xmat/xquat（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_body_xpos_xmat_xquat(
            body_name_list
        )

    def query_body_xpos_xmat_xquat_xvel(
        self, body_name_list: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """查询 body 的 xpos/xmat/xquat/xvel（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_body_xpos_xmat_xquat_xvel(
            body_name_list
        )

    def query_site_pos_and_mat(self, site_names: list[str]) -> dict[str, dict]:
        """查询 site 的 pos 和 mat（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_site_pos_and_mat(site_names)

    def query_site_size(self, site_names: list[str]) -> dict[str, np.ndarray]:
        """查询 site 尺寸（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_site_size(site_names)

    def query_sensor_data(
        self, sensor_names: list[str]
    ) -> dict[str, np.ndarray]:
        """查询传感器数据（委托 SimCore，从 _orca_model 拼装 sensor_info）。

        K3：SimCore 不持有 OrcaGymModel，sensor_info 由 Gym 从 _orca_model
        拼装后传入。
        """
        sim = object.__getattribute__(self, "_sim")
        model = object.__getattribute__(self, "_orca_model")
        sensor_info = {name: model.get_sensor(name) for name in sensor_names}
        return sim.query_sensor_data(sensor_names, sensor_info)

    def query_actuator_torques(
        self, actuator_names: list[str]
    ) -> dict[str, np.ndarray]:
        """查询执行器力矩（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_actuator_torques(
            actuator_names
        )

    def query_contact_simple(self) -> list[dict]:
        """查询简单接触信息（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_contact_simple()

    def query_contact_force(self, contact_ids: list[int]) -> dict[int, np.ndarray]:
        """查询接触力（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").query_contact_force(contact_ids)

    def get_cfrc_ext(self) -> np.ndarray:
        """查询外部约束力 cfrc_ext（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").get_cfrc_ext()

    def get_goal_bounding_box(self, geom_name: str) -> np.ndarray:
        """查询 geom 尺寸（委托 SimCore）。"""
        return object.__getattribute__(self, "_sim").get_goal_bounding_box(geom_name)

    def body_subtree_mass(self, body_name: str) -> float:
        """查询 body 子树总质量（委托 ModelRegistry）。"""
        return object.__getattribute__(self, "_registry").body_subtree_mass(body_name)

    def geom_friction(self, geom_name: str) -> np.ndarray:
        """查询 geom 摩擦系数 (3,)（委托 ModelRegistry，只读视图）。"""
        return object.__getattribute__(self, "_registry").geom_friction(geom_name)

    # --- 力应用方法（阶段三 3.2.1，委托 SimCore）---

    def apply_body_force(
        self, body_id: int, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """对指定 body 施加外力/力矩（委托 SimCore）。

        Args:
            body_id: MuJoCo body id。
            force: 力向量 (3,)。
            torque: 力矩向量 (3,)。
        """
        object.__getattribute__(self, "_sim").apply_body_force(body_id, force, torque)

    def clear_body_force(self, body_id: int) -> None:
        """清除指定 body 的外力（委托 SimCore）。"""
        object.__getattribute__(self, "_sim").clear_body_force(body_id)

    def clear_all_forces(self) -> None:
        """清除所有 body 的外力（委托 SimCore）。"""
        object.__getattribute__(self, "_sim").clear_all_forces()

    def mj_apply_force_at_site(
        self, site_id: int, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """在 site 处施加力（委托 SimCore）。

        Args:
            site_id: MuJoCo site id。
            force: 力向量 (3,)（世界坐标系）。
            torque: 力矩向量 (3,)（世界坐标系）。
        """
        object.__getattribute__(self, "_sim").mj_apply_force_at_site(site_id, force, torque)

    def mj_clear_xfrc_applied_for_site(self, site_id: int) -> None:
        """清除 site 关联 body 的 xfrc（委托 SimCore）。"""
        object.__getattribute__(self, "_sim").mj_clear_xfrc_applied_for_site(site_id)

    # --- 状态设置方法（阶段三 3.2.4，委托 SimCore/Bridge）---

    def set_mocap_pos_and_quat(self, mocap_dict: dict[str, dict]) -> None:
        """设置 mocap body 位置/四元数（本地写入，委托 SimCore）。

        远端同步由 env.set_mocap_pos_and_quat 调用 set_mocap_pos_and_quat_remote 完成。

        Args:
            mocap_dict: dict[body_name -> {"pos": (3,), "quat": (4,) [w,x,y,z]}]。
        """
        object.__getattribute__(self, "_sim").set_mocap_pos_and_quat(mocap_dict)

    async def set_mocap_pos_and_quat_remote(
        self, mocap_data: dict, send_remote: bool = False
    ) -> None:
        """远端同步 mocap 位姿到 OrcaStudio（委托 Bridge）。

        Args:
            mocap_data: dict[body_name -> {"pos": (3,), "quat": (4,)}]。
            send_remote: 是否真正发送到远端。
        """
        bridge = object.__getattribute__(self, "_studio")
        await bridge.set_mocap_pos_and_quat(mocap_data, send_remote)

    def set_geom_friction(self, geom_friction_dict: dict[str, np.ndarray]) -> None:
        """设置 geom 摩擦系数（委托 SimCore）。"""
        object.__getattribute__(self, "_sim").set_geom_friction(geom_friction_dict)

    def add_extra_weight(self, weight_load_dict: dict) -> None:
        """为 body 添加额外重量（委托 SimCore）。"""
        object.__getattribute__(self, "_sim").add_extra_weight(weight_load_dict)

    # --- 雅可比计算方法（阶段三 3.3.3，委托 SimCore）---

    def mj_jacBody(
        self, jacp: np.ndarray, jacr: np.ndarray, body_id: int
    ) -> None:
        """计算 body 雅可比（原地写 jacp/jacr，委托 SimCore）。

        Args:
            jacp: 平移雅可比矩阵 (3, nv)，调用方预分配。
            jacr: 旋转雅可比矩阵 (3, nv)，调用方预分配。
            body_id: MuJoCo body id。
        """
        object.__getattribute__(self, "_sim").mj_jacBody(jacp, jacr, body_id)

    def mj_jacSite(
        self, jacp: np.ndarray, jacr: np.ndarray, site_id: int
    ) -> None:
        """计算 site 雅可比（原地写 jacp/jacr，委托 SimCore）。

        Args:
            jacp: 平移雅可比矩阵 (3, nv)，调用方预分配。
            jacr: 旋转雅可比矩阵 (3, nv)，调用方预分配。
            site_id: MuJoCo site id。
        """
        object.__getattribute__(self, "_sim").mj_jacSite(jacp, jacr, site_id)

    def mj_jac_site(self, site_names: list[str]) -> dict[str, dict]:
        """批量计算 site 雅可比（委托 SimCore）。

        Args:
            site_names: site 名称列表。

        Returns:
            dict[site_name -> {"jacp": np.ndarray(3, nv),
                               "jacr": np.ndarray(3, nv)}]。
        """
        return object.__getattribute__(self, "_sim").mj_jac_site(site_names)

    def equality_data_width(self) -> int:
        """查询等式约束数据宽度（委托 ModelRegistry）。"""
        return object.__getattribute__(self, "_registry").equality_data_width()

    def equality_object_ids(self, eq_idx: int) -> tuple[int, int]:
        """查询等式约束关联对象 id（委托 ModelRegistry）。"""
        return object.__getattribute__(self, "_registry").equality_object_ids(eq_idx)

    def equality_constraint(self, eq_idx: int) -> dict:
        """读取单个等式约束完整数据（委托 ModelRegistry）。

        返回 type/obj1_id/obj2_id/active/solref/solimp/data 完整字段，
        用于体操作时读取 XML 预定义约束的原始值，修改后回写。
        """
        return object.__getattribute__(self, "_registry").equality_constraint(eq_idx)

    def n_equality(self) -> int:
        """查询等式约束数量（委托 ModelRegistry）。"""
        return object.__getattribute__(self, "_registry").n_equality()

    def mocap_body_names(self) -> list[str]:
        """查询所有 mocap body 名称（委托 ModelRegistry）。"""
        return object.__getattribute__(self, "_registry").mocap_body_names()

    # --- 等式约束委托（阶段三 3.5.3，委托 SimCore）---

    def update_equality_constraints(self, eq_list: list[dict]) -> None:
        """更新等式约束（委托 SimCore，写 _mjModel.eq_*）。

        Args:
            eq_list: 等式约束列表，每项含 type/obj1_id/obj2_id/data。
        """
        object.__getattribute__(self, "_sim").update_equality_constraints(eq_list)

    def modify_equality_objects(
        self,
        eq_ids: list[int],
        obj1_ids=None,
        obj2_ids=None,
    ) -> None:
        """修改等式约束关联对象（委托 SimCore）。

        Args:
            eq_ids: 等式约束索引列表。
            obj1_ids: 新的 obj1 id 列表（None 不修改）。
            obj2_ids: 新的 obj2 id 列表（None 不修改）。
        """
        object.__getattribute__(self, "_sim").modify_equality_objects(
            eq_ids, obj1_ids, obj2_ids
        )

    def set_equality_active(self, eq_idx: int, active: bool) -> None:
        """设置等式约束激活状态（委托 SimCore，写 _mjModel.eq_active0）。

        作为 Env.equality_update 的实现细节，不进入 Env 公共 API。
        """
        object.__getattribute__(self, "_sim").set_equality_active(eq_idx, active)

    def set_equality_solref(self, eq_idx: int, solref) -> None:
        """设置等式约束 solver reference 参数（委托 SimCore）。"""
        object.__getattribute__(self, "_sim").set_equality_solref(eq_idx, solref)

    def set_equality_solimp(self, eq_idx: int, solimp) -> None:
        """设置等式约束 solver impedance 参数（委托 SimCore）。"""
        object.__getattribute__(self, "_sim").set_equality_solimp(eq_idx, solimp)

    # --- AR-001：拖拽代理碰撞掩码关闭（委托 SimCore，不暴露 _mjModel）---

    def disable_actor_manipulator_collision(self) -> int:
        """关闭 ActorManipulator 拖拽代理几何体的碰撞掩码（委托 SimCore）。

        模型加载时（init_simulation）已自动执行；本方法供环境在需要时重断言。

        Returns:
            本次被修改的 geom 数量。
        """
        return object.__getattribute__(self, "_sim").disable_actor_manipulator_collision()
