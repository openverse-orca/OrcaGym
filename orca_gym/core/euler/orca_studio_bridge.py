"""OrcaStudioBridge — OrcaStudio gRPC 集成桥接。

本模块属于 OrcaGym Euler 体系（阶段二 Step 1），处理与 OrcaStudio
的 gRPC 交互（架构 §5.4）。

核心设计：依赖反转 — 不持有 _mjData/_mjModel，通过接收数据参数实现
与仿真核心的解耦。只负责通信和场景同步，不碰 mj_step。

离线模式（stub=None）所有 gRPC 方法 no-op，不抛异常。
在线模式通过 gRPC stub 与 OrcaStudio 通信。
"""

import os
import re
import shutil
import tempfile
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from orca_gym.protos import mjc_message_pb2
from orca_gym.utils.dir_utils import file_lock


class AnchorType:
    """锚点类型枚举（与 proto GetBodyManipulationAnchoredResponse.AnchorType 对应）。"""
    NONE = 0
    WELD = 1
    BALL = 2


class OrcaStudioBridge:
    """OrcaStudio gRPC 集成桥接。

    依赖反转：不持有 _mjData，通过接收数据参数实现解耦。
    只负责通信和场景同步，不碰 mj_step。

    使用契约:
        渲染:       await bridge.render(qpos, sim_time)
        加载模型:   xml = await bridge.load_model_xml()
        暂停:       await bridge.pause_simulation()
        离线配置:   bridge.configure_offline(xml_path, assets_dir)
        远端时间步: bridge.set_timestep_remote(0.002)
        体操作:     anchored, movement = await bridge.get_body_manipulation_*
        控制覆盖:   ctrls = bridge.get_override_ctrls()

    禁止:
        不要通过本类访问 MuJoCo 内部数据结构。
    """

    def __init__(self, stub=None) -> None:
        """初始化 OrcaStudio 桥接。

        Args:
            stub: OrcaStudio gRPC stub。None 表示离线模式。
        """
        self._stub = stub
        self._local_xml_path: str | None = None
        self._xml_assets_dir: str | None = None
        self._override_ctrls: dict[int, float] = {}

    # --- 离线配置 ---

    def configure_offline(self, xml_path: str, assets_dir: str | None = None) -> None:
        """离线模式配置本地 XML 路径。

        Args:
            xml_path: 模型 XML 文件路径。
            assets_dir: 资源目录路径（可选，默认取 XML 所在目录）。
        """
        self._local_xml_path = os.path.abspath(os.path.expanduser(xml_path))
        if assets_dir is None:
            self._xml_assets_dir = os.path.dirname(self._local_xml_path)
        else:
            self._xml_assets_dir = os.path.abspath(os.path.expanduser(assets_dir))

    @property
    def xml_file_dir(self) -> str:
        """XML 文件缓存目录（在线模式下载、离线模式本地资源）。"""
        if self._xml_assets_dir:
            assets = os.path.abspath(os.path.expanduser(self._xml_assets_dir))
            os.makedirs(assets, exist_ok=True)
            return assets
        user_home = os.path.expanduser('~')
        save_dir = os.path.join(user_home, '.orcagym', 'tmp')
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    # --- 模型加载 ---

    async def load_model_xml(self) -> str:
        """加载模型 XML（离线返回本地路径，在线从 Studio 拉取）。

        两分支返回路径后统一调用 `process_xml_file` 检查并补全 mesh/hfield
        资源：在线模式自动从 Studio 下载缺失文件，离线模式缺失即抛
        `FileNotFoundError`（早于 `MjModel.from_xml_path` 的底层错误）。

        Returns:
            模型 XML 文件本地路径（此时 mesh 资源已就位）。

        Raises:
            RuntimeError: 离线模式未配置 local_xml_path。
            FileNotFoundError: 离线模式 mesh 资源缺失。
        """
        if self._stub is None:
            if self._local_xml_path is None:
                raise RuntimeError("Offline mode but no local_xml_path configured")
            if not os.path.isfile(self._local_xml_path):
                raise FileNotFoundError(
                    f"local_xml_path not found: {self._local_xml_path}"
                )
            xml_path = self._local_xml_path
        else:
            xml_path = await self._load_model_xml_online()
        # 统一在此处检查/补全 mesh 资源（在线下载，离线校验）
        await self.process_xml_file(xml_path)
        return xml_path

    async def process_xml_file(self, file_path: str) -> None:
        """解析 XML 文件，下载缺失的 mesh/hfield 资源。

        读取 XML 文件，解析根节点，递归调用 `process_xml_node` 检查所有
        `mesh`/`hfield` 节点的 `file` 属性，缺失文件在线模式自动下载，
        离线模式抛 `FileNotFoundError`。

        Args:
            file_path: XML 文件路径。
        """
        with open(file_path, 'r') as f:
            xml_content = f.read()
        root = ET.fromstring(xml_content)
        await self.process_xml_node(root)

    async def process_xml_node(self, node) -> None:
        """递归处理 XML 节点，下载缺失的 mesh/hfield 资源。

        遇到 `mesh`/`hfield` 节点时检查 `file` 属性指向的文件是否存在，
        缺失则调用 `_download_asset_to_cache` 下载。其他节点递归处理子节点。

        Args:
            node: XML 元素节点（ElementTree.Element）。
        """
        if node.tag in ('mesh', 'hfield'):
            content_file_name = node.get('file')
            if content_file_name is not None:
                content_file_path = os.path.join(self.xml_file_dir, content_file_name)
                if not os.path.exists(content_file_path):
                    await self._download_asset_to_cache(content_file_name)
        else:
            for child in node:
                await self.process_xml_node(child)

    async def _load_model_xml_online(self) -> str:
        """在线模式：从 Studio 拉取模型 XML 文件。"""
        # 第一步：获取文件名
        request = mjc_message_pb2.LoadLocalEnvRequest()
        request.req_type = mjc_message_pb2.LoadLocalEnvRequest.XML_FILE_NAME
        response = await self._stub.LoadLocalEnv(request)

        if response.status != mjc_message_pb2.LoadLocalEnvResponse.SUCCESS:
            raise Exception(
                f"LoadLocalEnv XML_FILE_NAME failed: {response.error_message}"
            )

        file_name = response.file_name
        file_path = os.path.join(self.xml_file_dir, file_name)

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                # 第二步：获取文件内容
                request = mjc_message_pb2.LoadLocalEnvRequest()
                request.req_type = mjc_message_pb2.LoadLocalEnvRequest.XML_FILE_CONTENT
                response = await self._stub.LoadLocalEnv(request)

                if response.status != mjc_message_pb2.LoadLocalEnvResponse.SUCCESS:
                    raise Exception(
                        f"LoadLocalEnv XML_FILE_CONTENT failed: {response.error_message}"
                    )

                xml_content = response.xml_content
                xml_content = self._sanitize_xml_for_newer_mujoco(xml_content, file_name)

                # 原子化保存：先写临时文件，再移动
                temp_file = tempfile.NamedTemporaryFile(
                    mode='wb',
                    dir=self.xml_file_dir,
                    delete=False,
                    prefix=f"{file_name}_",
                    suffix=".tmp",
                )
                try:
                    temp_file.write(xml_content)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_file.close()
                    shutil.move(temp_file.name, file_path)
                except Exception:
                    try:
                        os.unlink(temp_file.name)
                    except OSError:
                        pass
                    raise

        return os.path.abspath(file_path)

    @staticmethod
    def _sanitize_xml_for_newer_mujoco(xml_content: bytes, file_name: str) -> bytes:
        """兼容清洗：移除新版本 MuJoCo 已废弃的 XML 属性（vertcollide）。"""
        sanitized = re.sub(rb'\svertcollide="[^"]*"', b'', xml_content)
        if sanitized != xml_content:
            pass
        return sanitized

    # --- 渲染 ---

    async def render(
        self,
        qpos: np.ndarray,
        sim_time: float,
        simulate_index: int = -1,
        request_idr: bool = False,
        contacts: list[dict] | None = None,
    ) -> None:
        """渲染当前仿真状态到 OrcaStudio（依赖反转：接收 qpos/sim_time）。

        离线模式 no-op；在线模式将 qpos/time/simulate_index 推送到 Studio，
        接收 override_ctrls。

        Args:
            qpos: 广义坐标位置数组。
            sim_time: 仿真时间。
            simulate_index: 物理仿真步索引，透传到相机管线用于帧对齐。
                ``-1`` 表示由服务端自增（向后兼容，默认值）。
            request_idr: 是否请求引擎在本次渲染输出一个 IDR 关键帧。
                默认 ``False``。
        """
        if self._stub is None:
            return
        request = mjc_message_pb2.UpdateLocalEnvRequest(
            qpos=qpos.tolist(),
            time=float(sim_time),
            simulate_index=simulate_index,
            request_idr=request_idr,
        )
        if contacts:
            for con in contacts:
                cs = request.contacts.add()
                cs.pos.extend(con["pos"])
                cs.force.extend(con["force"])
        response = await self._stub.UpdateLocalEnv(request)
        # 更新 override_ctrls 缓存
        self._override_ctrls.clear()
        for ctrl in response.override_ctrls:
            self._override_ctrls[ctrl.index] = ctrl.value

    def get_override_ctrls(self) -> dict[int, float]:
        """返回当前的 override 控制覆盖值。

        Returns:
            dict[int, float]: 索引到值的映射（副本）。
        """
        return dict(self._override_ctrls)

    # --- 暂停 ---

    async def pause_simulation(self) -> None:
        """通知 OrcaStudio 暂停仿真。离线模式 no-op。"""
        if self._stub is None:
            return
        request = mjc_message_pb2.SetSimulationStateRequest(
            state=mjc_message_pb2.PAUSED
        )
        await self._stub.SetSimulationState(request)

    # --- 远端时间步 ---

    def set_timestep_remote(self, timestep: float) -> None:
        """设置远端 OrcaStudio 的仿真时间步（同步签名）。离线模式 no-op。

        Args:
            timestep: 时间步长（秒）。
        """
        if self._stub is None:
            return
        request = mjc_message_pb2.SetOptTimestepRequest(timestep=timestep)
        # 同步方法内部通过 event_loop 调度 gRPC（调用方需在 asyncio 上下文中）
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在运行中的 loop：创建 task 但无法 await（同步方法）
                # 阶段二简化：fire-and-forget，实际由调用方负责同步
                asyncio.ensure_future(self._stub.SetOptTimestep(request), loop=loop)
            else:
                loop.run_until_complete(self._stub.SetOptTimestep(request))
        except RuntimeError:
            # 无 event loop，创建临时 loop
            asyncio.run(self._stub.SetOptTimestep(request))

    # --- 体操作 ---

    async def get_body_manipulation_anchored(self) -> tuple:
        """查询体操作的锚定状态。

        离线模式返回 (None, AnchorType.NONE)。

        Returns:
            (body_name, anchor_type) 元组。
        """
        if self._stub is None:
            return (None, AnchorType.NONE)
        request = mjc_message_pb2.GetBodyManipulationAnchoredRequest()
        response = await self._stub.GetBodyManipulationAnchored(request)
        body_name = response.body_name
        if body_name is None or len(body_name) == 0:
            return (None, AnchorType.NONE)
        return (body_name, response.anchor_type)

    async def get_body_manipulation_movement(self) -> dict:
        """查询体操作的运动状态。

        离线模式返回零增量。

        Returns:
            包含 'delta_pos' 和 'delta_quat' 的字典。
        """
        if self._stub is None:
            return {
                "delta_pos": np.zeros(3),
                "delta_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            }
        request = mjc_message_pb2.GetBodyManipulationMovementRequest()
        response = await self._stub.GetBodyManipulationMovement(request)
        return {
            "delta_pos": np.array(response.delta_pos),
            "delta_quat": np.array(response.delta_quat),
        }

    # --- mocap 远端同步（阶段三 3.2.3）---

    async def set_mocap_pos_and_quat(
        self, mocap_data: dict, send_remote: bool = False
    ) -> None:
        """设置 mocap 位姿并同步到远端 Studio（依赖反转：接收 mocap_data）。

        离线模式（_stub is None）直接 return，不抛错。
        在线模式通过 gRPC stub.SetMocapPosAndQuat 同步到远端。

        Args:
            mocap_data: dict[body_name -> {"pos": (3,), "quat": (4,) [w,x,y,z]}]。
                        本方法不写入 _mjData，仅做远端同步；
                        本地写入由 SimCore.set_mocap_pos_and_quat 完成。
            send_remote: 是否真正发送到远端。False 时 no-op（仅本地已写入）。
        """
        if self._stub is None:
            return
        if not send_remote:
            return
        mocap_infos = []
        for body_name, pose in mocap_data.items():
            pos = np.asarray(pose["pos"], dtype=np.float64).reshape(3).tolist()
            quat = np.asarray(pose["quat"], dtype=np.float64).reshape(4).tolist()
            mocap_infos.append(
                mjc_message_pb2.SetMocapPosAndQuatRequest.MocapBodyInfo(
                    mocap_body_name=body_name, pos=pos, quat=quat
                )
            )
        request = mjc_message_pb2.SetMocapPosAndQuatRequest(
            mocap_body_info=mocap_infos
        )
        await self._stub.SetMocapPosAndQuat(request)

    # --- 视频录制 / 帧捕获（已废弃）---
    # 引擎侧 BeginSaveMp4File / StopSaveMp4File / GetCurrentFrameIndex /
    # GetTimeStamp 四个 RPC 已从 proto 中删除。
    # 录制能力已迁移到客户端 PyAV remux（见 orca_gym/recorder/），
    # 通过 OrcaGymEulerEnv.save_streaming 使用。
    # 以下方法保留为 no-op + DeprecationWarning，向后兼容旧调用方。

    async def begin_save_video(self, file_path: str, capture_mode) -> None:
        """[Deprecated] 开始录制视频（原 gRPC BeginSaveMp4File）。

        .. deprecated::
            引擎侧 MP4 录制 RPC 已删除。请使用
            ``OrcaGymEulerEnv.save_streaming(camera_name, file_path,
            start_simulate_index, end_simulate_index, color_port)``
            进行客户端 PyAV remux 录制。
            此方法为 no-op 并发出 ``DeprecationWarning``。

        Args:
            file_path: 视频文件保存路径（已忽略）。
            capture_mode: 捕获模式（已忽略）。
        """
        warnings.warn(
            "OrcaStudioBridge.begin_save_video is deprecated. "
            "Engine-side MP4 recording RPC has been removed. "
            "Use OrcaGymEulerEnv.save_streaming for client-side PyAV remux recording.",
            DeprecationWarning,
            stacklevel=2,
        )

    async def stop_save_video(self) -> None:
        """[Deprecated] 停止录制视频（原 gRPC StopSaveMp4File）。

        .. deprecated::
            引擎侧 MP4 录制 RPC 已删除。请使用
            ``OrcaGymEulerEnv.save_streaming(camera_name, file_path,
            start_simulate_index, end_simulate_index, color_port)``。
            此方法为 no-op 并发出 ``DeprecationWarning``。
        """
        warnings.warn(
            "OrcaStudioBridge.stop_save_video is deprecated. "
            "Engine-side MP4 recording RPC has been removed. "
            "Use OrcaGymEulerEnv.save_streaming instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    async def get_current_frame(self) -> int:
        """[Deprecated] 获取当前帧号（原 gRPC GetCurrentFrameIndex）。

        .. deprecated::
            引擎侧帧索引 RPC 已删除。客户端录制通过 ``simulate_index``
            在 WebSocket 帧头中对齐，无需查询远端帧号。
            此方法返回 -1 并发出 ``DeprecationWarning``。

        Returns:
            -1（始终）。
        """
        warnings.warn(
            "OrcaStudioBridge.get_current_frame is deprecated. "
            "Engine-side frame index RPC has been removed. "
            "Use simulate_index in render() for frame alignment.",
            DeprecationWarning,
            stacklevel=2,
        )
        return -1

    async def get_camera_time_stamp(self, last_frame_index: int) -> dict:
        """[Deprecated] 获取相机时间戳（原 gRPC GetTimeStamp）。

        .. deprecated::
            引擎侧时间戳 RPC 已删除。客户端录制通过 WebSocket 帧头
            携带 timestamp（uint64 LE）对齐，无需查询远端时间戳。
            此方法返回 ``{}`` 并发出 ``DeprecationWarning``。

        Args:
            last_frame_index: 截止帧索引（已忽略）。

        Returns:
            ``{}``（始终）。
        """
        warnings.warn(
            "OrcaStudioBridge.get_camera_time_stamp is deprecated. "
            "Engine-side timestamp RPC has been removed. "
            "WebSocket frame header carries uint64 timestamp for alignment.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {}

    async def get_frame_png(self, image_path: str) -> None:
        """获取帧 PNG（gRPC GetCameraFramePNG）。离线 no-op。

        Args:
            image_path: 图像保存路径。
        """
        if self._stub is None:
            return
        request = mjc_message_pb2.GetCameraFramePNGRequest(image_path=image_path)
        await self._stub.GetCameraFramePNG(request)

    # --- 相机属性查询/设置 + 推流状态机（Phase 2 新增）---
    # 状态机：Idle <-- SetStreamingEnabled(true) --> Streaming（InitCameraSensor）
    #         Streaming <-- SetStreamingEnabled(false) --> Idle（UninitCameraSensor）
    #         属性 Set 仅在 Idle 状态允许（streaming=true 时禁止所有属性设置）
    # MP4 录制由 begin_save_video/stop_save_video 控制，与本组接口正交。

    async def get_camera_names(self) -> list[str]:
        """获取所有已注册相机名称列表（gRPC GetCameraNames）。离线返回空列表。

        Returns:
            已注册相机名称列表（含 uuid 后缀的 registered name）。
            离线模式返回空列表。
        """
        if self._stub is None:
            return []
        request = mjc_message_pb2.GetCameraNamesRequest()
        resp = await self._stub.GetCameraNames(request)
        if resp.status != mjc_message_pb2.GetCameraNamesResponse.SUCCESS:
            raise RuntimeError(
                f"GetCameraNames failed: {resp.error_message}"
            )
        return list(resp.camera_names)

    async def get_camera_properties(
        self,
        camera_name: str,
    ) -> mjc_message_pb2.GetCameraPropertiesResponse:
        """查询相机属性 + 推流状态（gRPC GetCameraProperties）。离线返回空响应。

        一次性获取所有相机属性与 streaming_enabled 状态，适合 UI 渲染。

        Args:
            camera_name: 相机名称（对应 Studio 端注册的 camera name，
                可通过 get_camera_names 枚举获取）。

        Returns:
            GetCameraPropertiesResponse：包含 streaming_enabled、传感器开关、
            图像参数、编码器、推流端口、DDS 等全部字段。离线模式返回默认实例。
        """
        if self._stub is None:
            return mjc_message_pb2.GetCameraPropertiesResponse()
        request = mjc_message_pb2.GetCameraPropertiesRequest(camera_name=camera_name)
        resp = await self._stub.GetCameraProperties(request)
        if resp.status != mjc_message_pb2.GetCameraPropertiesResponse.SUCCESS:
            raise RuntimeError(
                f"GetCameraProperties failed: {resp.error_message}"
            )
        return resp

    async def set_camera_properties(
        self,
        camera_name: str,
        *,
        capture_rgb: bool | None = None,
        capture_depth: bool | None = None,
        capture_normal: bool | None = None,
        capture_object_color: bool | None = None,
        random_object_color: bool | None = None,
        use_nvenc: bool | None = None,
        nvenc_gpu_index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        vertical_fov: float | None = None,
        near_clip: float | None = None,
        far_clip: float | None = None,
        gamma: float | None = None,
        color_port: int | None = None,
        depth_port: int | None = None,
        use_dds: bool | None = None,
        dds_topic: str | None = None,
        dds_stream_id: str | None = None,
    ) -> None:
        """批量设置相机属性（gRPC SetCameraProperties）。离线 no-op。

        仅设置显式传参（非 None）的字段，未传字段保持 server 现有值。
        状态机约束：属性 Set 仅在 Idle 状态允许；若当前为 Streaming 状态，
        需先调用 set_streaming_enabled(False) 回到 Idle 再设置属性。

        Args:
            camera_name: 相机名称。
            capture_rgb: 是否激活 RGB 视频流。
            capture_depth: 是否激活深度视频流。
            capture_normal: 是否捕获法线图。
            capture_object_color: 是否捕获实例分割色标图。
            random_object_color: 是否随机分配物体颜色。
            use_nvenc: 是否使用 NvEnc 硬件编码。
            nvenc_gpu_index: NvEnc GPU 适配器索引。
            width: 图像宽度（像素）。
            height: 图像高度（像素）。
            vertical_fov: 垂直视场角（度）。
            near_clip: 近裁剪面距离。
            far_clip: 远裁剪面距离。
            gamma: 深度相机 gamma 校正。
            color_port: RGB 流 WebSocket 端口。
            depth_port: 深度流 WebSocket 端口。
            use_dds: 是否启用 DDS。
            dds_topic: DDS 主题。
            dds_stream_id: DDS 流 ID。
        """
        if self._stub is None:
            return
        # optional 字段：仅当显式传参（非 None）时才设置，对应 proto optional 语义
        property_kwargs: dict = {}
        _optional_fields = [
            "capture_rgb", "capture_depth", "capture_normal", "capture_object_color",
            "random_object_color", "use_nvenc", "nvenc_gpu_index",
            "width", "height", "vertical_fov", "near_clip", "far_clip", "gamma",
            "color_port", "depth_port",
            "use_dds", "dds_topic", "dds_stream_id",
        ]
        for fname in _optional_fields:
            val = locals()[fname]
            if val is not None:
                property_kwargs[fname] = val
        request = mjc_message_pb2.SetCameraPropertiesRequest(
            camera_name=camera_name,
            property=mjc_message_pb2.CameraProperty(**property_kwargs),
        )
        resp = await self._stub.SetCameraProperties(request)
        if resp.status != mjc_message_pb2.SetCameraPropertiesResponse.SUCCESS:
            raise RuntimeError(
                f"SetCameraProperties failed: {resp.error_message}"
            )

    async def set_streaming_enabled(
        self,
        camera_name: str,
        enabled: bool,
    ) -> None:
        """显式切换推流状态（gRPC SetStreamingEnabled）。离线 no-op。

        状态机：
            - enabled=True：Idle → Streaming（Studio 端调用 InitCameraSensor，
              7070/7071 等端口开始监听并推流）
            - enabled=False：Streaming → Idle（Studio 端调用 UninitCameraSensor，
              停止推流并释放资源）

        Args:
            camera_name: 相机名称。
            enabled: True 启动推流，False 停止推流。
        """
        if self._stub is None:
            return
        request = mjc_message_pb2.SetStreamingEnabledRequest(
            camera_name=camera_name,
            enabled=enabled,
        )
        resp = await self._stub.SetStreamingEnabled(request)
        if resp.status != mjc_message_pb2.SetStreamingEnabledResponse.SUCCESS:
            raise RuntimeError(
                f"SetStreamingEnabled failed: {resp.error_message}"
            )

    async def make_camera_viewport_active(
        self, actor_name: str, entity_name: str
    ) -> None:
        """将指定摄像头设为 Studio 视口激活相机（gRPC MakeCameraViewportActive）。

        离线 no-op。用于让 Studio 3D 视口以指定相机视角渲染。

        Args:
            actor_name: 摄像头所属 actor 名。
            entity_name: 摄像头实体名（如 "camera_head"）。
        """
        if self._stub is None:
            return
        request = mjc_message_pb2.MakeCameraViewportActiveRequest(
            actor_name=actor_name, entity_name=entity_name
        )
        resp = await self._stub.MakeCameraViewportActive(request)
        if resp.status != mjc_message_pb2.MakeCameraViewportActiveResponse.SUCCESS:
            raise RuntimeError(
                f"MakeCameraViewportActive failed: {resp.error_message}"
            )

    # --- 内容文件（阶段三 3.4.3）---

    async def load_content_file(
        self,
        content_file_name: str,
        remote_file_dir: str = "",
        local_file_dir: str = "",
        temp_file_path: str | None = None,
    ) -> None:
        """加载内容文件（gRPC LoadContentFile）。离线 no-op。

        Bridge 层为薄 gRPC 包装：仅发起请求，文件落盘由上层处理。
        离线模式（_stub is None）直接 return。

        Args:
            content_file_name: 资源文件名（如 mesh.obj）。
            remote_file_dir: 服务器端文件目录（对应 proto file_dir 字段）。
            local_file_dir: 本地存储目录（Bridge 层不使用，由上层处理）。
            temp_file_path: 临时文件路径（Bridge 层不使用，由上层处理）。
        """
        if self._stub is None:
            return
        request = mjc_message_pb2.LoadContentFileRequest(
            file_name=content_file_name, file_dir=remote_file_dir
        )
        await self._stub.LoadContentFile(request)

    async def _download_asset_to_cache(self, content_file_name: str) -> str:
        """从 Studio 下载资源文件并原子落盘到 xml_file_dir。

        与 `load_content_file`（薄 gRPC 包装）的区别：本方法捕获响应、原子
        落盘到 `xml_file_dir`，供 `process_xml_node` 在线补全 mesh/hfield
        资源时调用。离线模式（_stub is None）抛 `FileNotFoundError`。

        多进程安全：`file_lock` + 存在性二次检查，并发调用同一文件时
        仅首个进程发起 gRPC 请求。

        Args:
            content_file_name: 资源文件名（可含子目录，如 "g1/foot.stl"）。

        Returns:
            本地缓存文件绝对路径。

        Raises:
            FileNotFoundError: 离线模式（_stub is None）资源缺失。
            Exception: gRPC 请求失败或返回空内容。
        """
        if self._stub is None:
            raise FileNotFoundError(
                f"Offline mode: missing mesh/asset '{content_file_name}' "
                f"(place file under xml assets dir: {self.xml_file_dir})"
            )
        content_file_path = os.path.join(self.xml_file_dir, content_file_name)
        # 先确保目标目录存在（file_lock 会在同目录创建 .lock 文件，
        # 子目录场景如 "g1/foot.stl" 需先 makedirs）
        os.makedirs(os.path.dirname(content_file_path), exist_ok=True)
        async with file_lock(content_file_path, timeout=30):
            # 二次检查：可能在等锁期间已被其他进程创建
            if os.path.exists(content_file_path):
                return content_file_path
            request = mjc_message_pb2.LoadContentFileRequest(
                file_name=content_file_name, file_dir=""
            )
            response = await self._stub.LoadContentFile(request)
            if response.status != mjc_message_pb2.LoadContentFileResponse.SUCCESS:
                raise Exception(
                    f"LoadContentFile failed for '{content_file_name}'"
                )
            content = response.content
            if not content:
                raise Exception(
                    f"LoadContentFile returned empty content for '{content_file_name}'"
                )
            # 原子化保存：先写临时文件，再 move
            temp_file = tempfile.NamedTemporaryFile(
                mode='wb',
                dir=os.path.dirname(content_file_path),
                delete=False,
                prefix=f"{os.path.basename(content_file_name)}_",
                suffix=".tmp",
            )
            try:
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()
                shutil.move(temp_file.name, content_file_path)
            except Exception:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass
                raise
        return content_file_path
