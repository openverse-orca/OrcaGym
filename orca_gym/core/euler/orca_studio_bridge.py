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

        Returns:
            模型 XML 文件本地路径。

        Raises:
            RuntimeError: 离线模式未配置 local_xml_path。
        """
        if self._stub is None:
            if self._local_xml_path is None:
                raise RuntimeError("Offline mode but no local_xml_path configured")
            if not os.path.isfile(self._local_xml_path):
                raise FileNotFoundError(
                    f"local_xml_path not found: {self._local_xml_path}"
                )
            return self._local_xml_path
        return await self._load_model_xml_online()

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

    async def render(self, qpos: np.ndarray, sim_time: float) -> None:
        """渲染当前仿真状态到 OrcaStudio（依赖反转：接收 qpos/sim_time）。

        离线模式 no-op；在线模式将 qpos/time 推送到 Studio，接收 override_ctrls。

        Args:
            qpos: 广义坐标位置数组。
            sim_time: 仿真时间。
        """
        if self._stub is None:
            return
        request = mjc_message_pb2.UpdateLocalEnvRequest(
            qpos=qpos.tolist(), time=float(sim_time)
        )
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
