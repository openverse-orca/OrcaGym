"""灰度深度相机录制器。

深度流由引擎在 ``depth_port`` 上推送，编码为 **NV12 灰度** H.264：每个像素
值即归一化后的深度（服务端 ``VideoEncoder::encode`` 中 ``log`` 归一 →
``gamma`` 校正 → 存 0-1）。本类继承 [[CameraRecorder]] 复用全部录制/缓存/
保存逻辑，仅额外携带近远裁剪面与 gamma 参数，并提供：
    - ``grayscale_to_depth``：将解码出的灰度帧还原为**米制真实深度**。
    - ``depth_to_visualization``：将米制深度映射为便于目视的伪彩色图。

与 color 相机完全解耦（独立实现类），由 [[VideoRecorderManager]] 以与
[[VideoRecorder]] 一致的接口统一管理。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from orca_gym.recorder.camera_recorder import CameraRecorder

if TYPE_CHECKING:
    import numpy.typing as npt


class DepthRecorder(CameraRecorder):
    """灰度深度相机录制器。

    Args:
        camera_name: 相机名称（与 color 录制器共用同一名称）。
        depth_port: 深度流 WebSocket 推流端口。
        near_clip: 近裁剪面距离（米）。用于逆变换还原真实深度。
        far_clip: 远裁剪面距离（米）。
        gamma: 深度相机 gamma 校正值。
        max_buffer_frames: 滚动缓存最大帧数。
    """

    stream_kind = "depth"

    def __init__(
        self,
        camera_name: str,
        depth_port: int,
        near_clip: float,
        far_clip: float,
        gamma: float,
        max_buffer_frames: int = 36000,
    ) -> None:
        super().__init__(
            camera_name=camera_name,
            port=depth_port,
            max_buffer_frames=max_buffer_frames,
        )
        self._near_clip = float(near_clip)
        self._far_clip = float(far_clip)
        self._gamma = float(gamma)

    # ------------------------------------------------------------------
    # 深度参数
    # ------------------------------------------------------------------

    @property
    def depth_port(self) -> int:
        """深度流 WebSocket 推流端口。"""
        return self._port

    @property
    def near_clip(self) -> float:
        """近裁剪面距离（米）。"""
        return self._near_clip

    @property
    def far_clip(self) -> float:
        """远裁剪面距离（米）。"""
        return self._far_clip

    @property
    def gamma(self) -> float:
        """深度相机 gamma 校正值。"""
        return self._gamma

    def update_params(self, **kwargs) -> None:
        """更新深度还原参数（覆盖基类空操作）。

        Args:
            near_clip: 近裁剪面距离（米），None 则保持不变。
            far_clip: 远裁剪面距离（米），None 则保持不变。
            gamma: gamma 校正值，None 则保持不变。
        """
        near_clip = kwargs.get("near_clip")
        far_clip = kwargs.get("far_clip")
        gamma = kwargs.get("gamma")
        if near_clip is not None:
            self._near_clip = float(near_clip)
        if far_clip is not None:
            self._far_clip = float(far_clip)
        if gamma is not None:
            self._gamma = float(gamma)

    # frame_transform 不覆盖：深度流显示时直接展示原始灰度帧，
    # 不做伪彩色转换。grayscale_to_depth / depth_to_visualization
    # 保留供程序化使用（如离线分析、保存深度数据）。

    # ------------------------------------------------------------------
    # 深度还原
    # ------------------------------------------------------------------

    def grayscale_to_depth(
        self, gray_frame: "npt.NDArray"
    ) -> "npt.NDArray[np.float32]":
        """将解码出的灰度深度帧还原为米制真实深度（float32，单通道）。

        服务端编码（``VideoEncoder::encode`` 的灰度分支）：
            ``pixel = powf(log(d - near + 1) / log(far - near + 1), gamma)``，
            归一化到 [0, 1] 后作为灰度值（>= 0 存入 NV12 Y 平面）。

        逆变换：
            ``d = near - 1 + (far - near + 1) ** (pixel ** (1 / gamma))``

        Args:
            gray_frame: 解码出的灰度帧。可为 ``(H, W)`` 单通道，或
                ``(H, W, 3)`` 的 BGR 灰度图（取任一通道）。

        Returns:
            ``(H, W)`` 的 float32 米制深度图。范围约为
            ``[near_clip, far_clip]``。
        """
        if np is None:
            raise RuntimeError(
                "numpy is not installed. Install with: pip install numpy"
            )
        if gray_frame.ndim == 3:
            gray = gray_frame[..., 0]
        else:
            gray = gray_frame

        gray_f = gray.astype(np.float32) / 255.0
        gray_f = np.clip(gray_f, 0.0, 1.0)

        inv_gamma = 1.0 / self._gamma
        log_range = np.log(self._far_clip - self._near_clip + 1.0)
        normalized = np.power(gray_f, inv_gamma)
        depth = self._near_clip - 1.0 + np.exp(normalized * log_range)
        return depth

    def depth_to_visualization(
        self, gray_frame: "npt.NDArray"
    ) -> "npt.NDArray":
        """将灰度深度帧转换为便于目视的伪彩色图（通道 BGR，uint8）。

        先还原米制深度，再线性映射到 ``[near_clip, far_clip]`` 并应用
        OpenCV ``COLORMAP_JET`` 伪彩色，便于人眼分辨远近。

        Args:
            gray_frame: 解码出的灰度深度帧（``(H, W)`` 或 ``(H, W, 3)``）。

        Returns:
            ``(H, W, 3)`` uint8 BGR 伪彩色图。
        """
        if cv2 is None:
            raise RuntimeError(
                "OpenCV (opencv-python) is not installed. "
                "Install with: pip install opencv-python"
            )
        depth = self.grayscale_to_depth(gray_frame)
        norm = np.clip(
            (depth - self._near_clip) / (self._far_clip - self._near_clip),
            0.0,
            1.0,
        )
        norm_u8 = (norm * 255.0).astype(np.uint8)
        return cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)

    def get_stats(self) -> dict:
        """返回深度录制器状态统计（含深度还原参数）。"""
        stats = super().get_stats()
        stats.update(
            {
                "near_clip": self._near_clip,
                "far_clip": self._far_clip,
                "gamma": self._gamma,
            }
        )
        return stats