"""OrcaStudioBridge — OrcaStudio gRPC 集成，依赖反转，不持有 _mjData。

将 OrcaGymLocal 中与 OrcaStudio 通信的方法迁移到此处，但采用依赖反转：
- render(qpos, sim_time) 接收数据参数，不直接访问 _mjData
- 所有方法委托 gRPC stub，stub=None 时跳过（离线模式）

属于 OrcaGymEuler 体系的 P3 Studio 集成组件。
参见 docs/design/architecture/orca_gym_euler_architecture.md 第 5.4 节。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import aiofiles
from typing import Optional

import numpy as np

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.protos import mjc_message_pb2
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub
from orca_gym.core.orca_gym_local import AnchorType, CaptureMode
from orca_gym.utils.dir_utils import file_lock

_logger = get_orca_logger()


class OrcaStudioBridge:
    """OrcaStudio gRPC 集成，依赖反转，不持有 _mjData。

    设计契约:
        - stub=None 时所有方法安全跳过（离线短链模式）。
        - render(qpos, sim_time) 接收数据参数，不访问 _mjData。
        - 所有方法委托 gRPC stub，与 OrcaGymLocal 对应方法逻辑一致。

    使用示例:
        ```python
        bridge = OrcaStudioBridge(stub)
        model_xml_path = await bridge.load_model_xml()
        await bridge.render(qpos, sim_time)
        ```
    """

    def __init__(self, stub: Optional[GrpcServiceStub]) -> None:
        self._stub = stub
        # 离线模式下的本地 XML 路径与资源目录（与 OrcaGymLocal 一致）
        self._local_xml_path: Optional[str] = None
        self._xml_assets_dir: Optional[str] = None
        # 控制覆盖（由 render/UpdateLocalEnv 返回，供 set_ctrl 应用）
        self._override_ctrls: dict[int, float] = {}

    @property
    def stub(self) -> Optional[GrpcServiceStub]:
        return self._stub

    @property
    def is_offline(self) -> bool:
        """是否为离线模式（stub=None）。"""
        return self._stub is None

    @property
    def override_ctrls(self) -> dict[int, float]:
        """控制覆盖字典（由 render 返回，供外部 set_ctrl 应用）。"""
        return self._override_ctrls

    def configure_offline(
        self,
        local_xml_path: str,
        xml_assets_dir: Optional[str] = None,
    ) -> None:
        """配置离线模式的本地 XML 路径与资源目录。

        Args:
            local_xml_path: 本地 MJCF XML 路径。
            xml_assets_dir: mesh/hfield 资源目录；默认取 local_xml_path 所在目录。
        """
        self._local_xml_path = os.path.abspath(os.path.expanduser(local_xml_path))
        if xml_assets_dir is None:
            self._xml_assets_dir = os.path.dirname(self._local_xml_path)
        else:
            self._xml_assets_dir = xml_assets_dir

    # --- 模型加载 ---

    async def load_model_xml(self) -> str:
        """加载模型 XML 文件，返回本地路径。

        离线模式（stub=None 且配置了 local_xml_path）直接返回本地路径；
        在线模式从 OrcaStudio 下载 XML 与资源。
        """
        if self._local_xml_path:
            path = self._local_xml_path
            if not os.path.isfile(path):
                raise FileNotFoundError(f"local_xml_path not found: {path}")
            await self._process_xml_file(path)
            return path

        if self.is_offline:
            raise RuntimeError(
                "离线模式未配置 local_xml_path，请调用 configure_offline() 或提供 stub"
            )

        xml_path = await self._load_local_env()
        await self._process_xml_file(xml_path)
        return xml_path

    async def _load_local_env(self) -> str:
        """从 OrcaStudio 下载 XML 文件到本地缓存目录。"""
        xml_file_dir = self._xml_cache_dir()

        # 第一步：获取文件名
        request = mjc_message_pb2.LoadLocalEnvRequest()
        request.req_type = mjc_message_pb2.LoadLocalEnvRequest.XML_FILE_NAME
        response = await self._stub.LoadLocalEnv(request)
        if response.status != mjc_message_pb2.LoadLocalEnvResponse.SUCCESS:
            raise Exception(
                f"LoadLocalEnv(XML_FILE_NAME) failed: {response.error_message}"
            )

        file_name = response.file_name
        file_path = os.path.join(xml_file_dir, file_name)

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                request = mjc_message_pb2.LoadLocalEnvRequest()
                request.req_type = mjc_message_pb2.LoadLocalEnvRequest.XML_FILE_CONTENT
                response = await self._stub.LoadLocalEnv(request)
                if response.status != mjc_message_pb2.LoadLocalEnvResponse.SUCCESS:
                    raise Exception(
                        f"LoadLocalEnv(XML_FILE_CONTENT) failed: {response.error_message}"
                    )
                xml_content = response.xml_content
                # 原子写入
                self._atomic_write(file_path, xml_content)

        return os.path.abspath(file_path)

    async def _process_xml_file(self, file_path: str) -> None:
        """处理 XML 文件中引用的资源（mesh/hfield），按需下载。

        离线模式跳过（资源已在本地）。
        """
        if self.is_offline:
            return
        # 在线模式：解析 XML，下载缺失资源
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            _logger.warning(f"解析 XML 失败，跳过资源下载: {e}")
            return

        xml_dir = os.path.dirname(os.path.abspath(file_path))
        meshdir = ""
        compiler = root.find("compiler")
        if compiler is not None:
            meshdir = compiler.get("meshdir", "")

        for mesh_elem in root.findall(".//asset/mesh"):
            file_attr = mesh_elem.get("file", "")
            if not file_attr:
                continue
            local_path = (
                os.path.join(xml_dir, meshdir, file_attr)
                if meshdir
                else os.path.join(xml_dir, file_attr)
            )
            if not os.path.exists(local_path):
                try:
                    await self.load_content_file(
                        content_file_name=os.path.basename(file_attr),
                        remote_file_dir=meshdir,
                        local_file_dir=os.path.dirname(local_path),
                    )
                except Exception as e:
                    _logger.warning(f"下载资源失败 {file_attr}: {e}")

    # --- 渲染与仿真状态 ---

    async def render(self, qpos: np.ndarray, sim_time: float) -> None:
        """将 qpos 和 sim_time 发送到 OrcaStudio 进行渲染。

        依赖反转：接收数据参数，不访问 _mjData。
        离线模式跳过。

        Args:
            qpos: 广义坐标数组。
            sim_time: 仿真时间。
        """
        if self.is_offline:
            return
        request = mjc_message_pb2.UpdateLocalEnvRequest(qpos=qpos, time=sim_time)
        response = await self._stub.UpdateLocalEnv(request)
        # 接收控制覆盖值
        self._override_ctrls.clear()
        override_ctrls = response.override_ctrls
        if override_ctrls:
            for ctrl in override_ctrls:
                if ctrl.index < 0:
                    continue
                self._override_ctrls[ctrl.index] = ctrl.value

    async def pause_simulation(self) -> None:
        """将 OrcaStudio 仿真状态设置为 PAUSED（被动模式）。"""
        if self.is_offline:
            return
        request = mjc_message_pb2.SetSimulationStateRequest(
            state=mjc_message_pb2.PAUSED
        )
        await self._stub.SetSimulationState(request)

    async def set_timestep_remote(self, timestep: float) -> None:
        """同步时间步长到 OrcaStudio。"""
        if self.is_offline:
            return
        request = mjc_message_pb2.SetTimestepRequest(timestep=timestep)
        await self._stub.SetTimestep(request)

    # --- 视频捕获 ---

    async def begin_save_video(
        self, path: str, mode: CaptureMode = CaptureMode.ASYNC
    ) -> None:
        """开始保存视频到指定路径。"""
        if self.is_offline:
            return
        request = mjc_message_pb2.BeginSaveMp4FileRequest(
            file_path=path, capture_mode=mode
        )
        response = await self._stub.BeginSaveMp4File(request)
        if response.status != mjc_message_pb2.BeginSaveMp4FileResponse.Status.SUCCESS:
            _logger.error(f"Failed to start video saving: {response.error_message}")

    async def stop_save_video(self) -> None:
        """停止保存视频。"""
        if self.is_offline:
            return
        request = mjc_message_pb2.StopSaveMp4FileRequest()
        await self._stub.StopSaveMp4File(request)

    async def get_current_frame(self) -> int:
        """获取当前相机帧索引。"""
        if self.is_offline:
            return -1
        request = mjc_message_pb2.GetCurrentFrameIndexRequest()
        response = await self._stub.GetCurrentFrameIndex(request)
        return response.current_frame

    async def get_camera_time_stamp(self, last_frame: int) -> dict:
        """获取相机时间戳。

        Args:
            last_frame: 上次查询的帧索引。

        Returns:
            字典，键为相机名称，值为时间戳列表。
        """
        if self.is_offline:
            return {}
        request = mjc_message_pb2.GetTimeStampRequest()
        request.last_frame_index = last_frame
        response = await self._stub.GetTimeStamp(request)
        if response.error_message != "":
            _logger.error(f"Get time stamp failed: {response.error_message}")
        return {
            camera_name: time_stamp_list.time_stamps
            for camera_name, time_stamp_list in response.time_stamp_map.items()
        }

    async def get_frame_png(self, image_path: str) -> dict:
        """获取相机帧 PNG 图像及位姿信息。

        Args:
            image_path: 图像保存路径。

        Returns:
            字典，键为相机名称，值为包含 'pos' 和 'quat' 的字典。
        """
        if self.is_offline:
            return {}
        request = mjc_message_pb2.GetCameraFramePNGRequest()
        request.image_path = image_path
        response = await self._stub.GetCameraFramePNG(request)
        result = {}
        for name_transform in response.name_transform:
            result[name_transform.name] = {
                "pos": list(name_transform.pos),
                "quat": list(name_transform.quat),
            }
        return result

    # --- 物体操作（UI 拖拽）---

    async def get_body_manipulation_anchored(self) -> tuple:
        """获取当前被锚定的 body 信息。

        Returns:
            (body_name, anchor_type) 元组；无锚定时返回 (None, AnchorType.NONE)。
        """
        if self.is_offline:
            return None, AnchorType.NONE
        request = mjc_message_pb2.GetBodyManipulationAnchoredRequest()
        response = await self._stub.GetBodyManipulationAnchored(request)
        body_anchored = response.body_name
        anchor_type = response.anchor_type
        if not body_anchored:
            return None, AnchorType.NONE
        return body_anchored, anchor_type

    async def get_body_manipulation_movement(self) -> dict:
        """获取 body 操作移动增量。

        Returns:
            字典，包含 'delta_pos' 和 'delta_quat'。
        """
        if self.is_offline:
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

    # --- 资源文件下载 ---

    async def load_content_file(
        self,
        content_file_name: str,
        remote_file_dir: str = "",
        local_file_dir: str = "",
        temp_file_path: Optional[str] = None,
    ) -> str:
        """从 OrcaStudio 下载资源文件（mesh/hfield 等）。

        Args:
            content_file_name: 资源文件名。
            remote_file_dir: 服务器端文件目录。
            local_file_dir: 本地存储目录。
            temp_file_path: 临时文件路径（特殊场景）。

        Returns:
            本地文件绝对路径。
        """
        if self.is_offline:
            raise FileNotFoundError(
                f"离线模式无法下载资源文件: {content_file_name}"
            )

        request = mjc_message_pb2.LoadContentFileRequest(
            file_name=content_file_name, file_dir=remote_file_dir
        )
        response = await self._stub.LoadContentFile(request)
        if response.status != mjc_message_pb2.LoadContentFileResponse.SUCCESS:
            raise Exception(f"Load content file failed: {content_file_name}")
        content = response.content
        if not content:
            raise Exception(f"Content is empty: {content_file_name}")

        if temp_file_path is not None:
            async with aiofiles.open(temp_file_path, "wb") as f:
                await f.write(content)
            return temp_file_path

        if not local_file_dir:
            local_file_dir = self._xml_cache_dir()
        content_file_path = os.path.join(local_file_dir, content_file_name)

        async with file_lock(content_file_path, timeout=30):
            if not os.path.exists(content_file_path):
                self._atomic_write(content_file_path, content)

        return content_file_path

    # --- 辅助方法 ---

    def _xml_cache_dir(self) -> str:
        """XML 与资源缓存目录（默认 ~/.orcagym/tmp）。"""
        cache_dir = os.path.expanduser("~/.orcagym/tmp")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _atomic_write(self, path: str, content: bytes) -> None:
        """原子写入：先写临时文件，再移动到最终位置。"""
        temp_file = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=os.path.dirname(path),
            delete=False,
            prefix=f"{os.path.basename(path)}_",
            suffix=".tmp",
        )
        try:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            shutil.move(temp_file.name, path)
        except Exception:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
            raise
