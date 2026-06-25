"""OrcaStudioBridge — OrcaStudio gRPC 集成桥接（骨架）。

本模块属于 OrcaGym Euler 体系骨架阶段（P2-Step3），处理与 OrcaStudio
的 gRPC 交互（架构 §5.4）。

核心设计：依赖反转 — 不持有 _mjData/_mjModel，通过接收数据参数实现
与仿真核心的解耦。只负责通信和场景同步，不碰 mj_step。

骨架阶段不执行真实 gRPC 通信，方法体 `raise NotImplementedError`。
P4 填充阶段将填入真实 gRPC 逻辑。
"""

import numpy as np


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

    禁止:
        不要通过本类访问 MuJoCo 内部数据结构。
    """

    def __init__(self, stub=None) -> None:
        """初始化 OrcaStudio 桥接。

        Args:
            stub: OrcaStudio gRPC stub。骨架阶段不依赖真实 stub，
                仅存储引用供 P4 填充阶段使用。
        """
        self._stub = stub

    # --- 骨架最小集（架构 §12.2）---

    async def render(self, qpos: np.ndarray, sim_time: float) -> None:
        """渲染当前仿真状态到 OrcaStudio。

        依赖反转：接收 qpos/sim_time 而非访问 _mjData。

        Args:
            qpos: 广义坐标位置数组。
            sim_time: 仿真时间。

        Raises:
            NotImplementedError: 骨架阶段未实现真实 gRPC 通信。
        """
        raise NotImplementedError("render 待 P4 填充")

    async def load_model_xml(self) -> str:
        """从 OrcaStudio 加载模型 XML 字符串。

        Returns:
            MuJoCo 模型 XML 字符串。

        Raises:
            NotImplementedError: 骨架阶段未实现真实 gRPC 通信。
        """
        raise NotImplementedError("load_model_xml 待 P4 填充")

    async def pause_simulation(self) -> None:
        """通知 OrcaStudio 暂停仿真。

        Raises:
            NotImplementedError: 骨架阶段未实现真实 gRPC 通信。
        """
        raise NotImplementedError("pause_simulation 待 P4 填充")

    def configure_offline(self, xml_path: str, assets_dir: str | None = None) -> None:
        """离线模式配置（不连接 OrcaStudio）。

        Args:
            xml_path: 模型 XML 文件路径。
            assets_dir: 资源目录路径（可选）。

        Raises:
            NotImplementedError: 骨架阶段未实现真实配置。
        """
        raise NotImplementedError("configure_offline 待 P4 填充")

    def set_timestep_remote(self, timestep: float) -> None:
        """设置远端 OrcaStudio 的仿真时间步。

        Args:
            timestep: 时间步长（秒）。

        Raises:
            NotImplementedError: 骨架阶段未实现真实 gRPC 通信。
        """
        raise NotImplementedError("set_timestep_remote 待 P4 填充")

    async def get_body_manipulation_anchored(self) -> tuple:
        """查询体操作的锚定状态。

        Returns:
            体操作锚定状态元组。

        Raises:
            NotImplementedError: 骨架阶段未实现真实 gRPC 通信。
        """
        raise NotImplementedError("get_body_manipulation_anchored 待 P4 填充")

    async def get_body_manipulation_movement(self) -> dict:
        """查询体操作的运动状态。

        Returns:
            体操作运动状态字典。

        Raises:
            NotImplementedError: 骨架阶段未实现真实 gRPC 通信。
        """
        raise NotImplementedError("get_body_manipulation_movement 待 P4 填充")
