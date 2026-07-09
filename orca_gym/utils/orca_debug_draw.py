"""OrcaDebugDraw — DebugMesh 调试绘制封装（独立顶层模块）。

依赖约束：
    本模块为顶层工具模块，禁止依赖 orca_gym.core / orca_gym.environment。
    只依赖 numpy + protos 生成产物 + 标准库。

离线模式（stub=None）所有方法 no-op，不抛异常，便于在无 OrcaStudio
连接的环境下复用上层逻辑（如离线渲染、单元测试）。

坐标系约定：
    与 MuJoCo / ORCA 一致，世界坐标系为右手系，单位米。
    位置 position=[x,y,z]，旋转 rotation=[x,y,z,w]（四元数，与 AZ::Quaternion 构造顺序一致），
    缩放 scale=[sx,sy,sz]（per-axis 非均匀缩放），颜色 color=[r,g,b,a]（0..1 浮点）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from orca_gym.protos import mjc_message_pb2


class InstanceFlags:
    """与 OrcaDebugMesh::InstanceFlags 严格对应（OrcaDebugMeshInstanceData.h）。"""

    NONE = 0
    EDGE_HIGHLIGHT = 1 << 0


class DebugMeshType:
    """与 OrcaDebugMesh::MeshType / proto DebugMeshType 严格对应。"""

    SPHERE = mjc_message_pb2.DEBUG_MESH_SPHERE
    CYLINDER = mjc_message_pb2.DEBUG_MESH_CYLINDER
    CONE = mjc_message_pb2.DEBUG_MESH_CONE
    BOX = mjc_message_pb2.DEBUG_MESH_BOX
    QUAD = mjc_message_pb2.DEBUG_MESH_QUAD
    ARROW = mjc_message_pb2.DEBUG_MESH_ARROW


def _make_instance(
    position: Sequence[float],
    rotation: Sequence[float],
    scale: Sequence[float],
    color: Sequence[float],
    flags: int = InstanceFlags.NONE,
) -> mjc_message_pb2.DebugMeshInstance:
    """构造 DebugMeshInstance proto。position/rotation/scale/color 支持 list/ndarray。"""
    inst = mjc_message_pb2.DebugMeshInstance()
    inst.position.extend(np.asarray(position, dtype=np.float32).tolist())
    inst.rotation.extend(np.asarray(rotation, dtype=np.float32).tolist())
    inst.scale.extend(np.asarray(scale, dtype=np.float32).tolist())
    inst.color.extend(np.asarray(color, dtype=np.float32).tolist())
    inst.flags = int(flags)
    return inst


def _direction_to_quat(direction: np.ndarray) -> np.ndarray:
    """单位方向向量 → 四元数 [x,y,z,w]。

    将 +Z 轴对齐到给定方向（圆柱/箭头沿 +Z 建模）。
    direction 会被归一化；零向量返回单位四元数（无旋转）。
    """
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    d = d / norm

    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # 旋转轴 = z × d，旋转角 = arccos(z·d)
    axis = np.cross(z, d)
    cos_angle = float(np.clip(np.dot(z, d), -1.0, 1.0))
    sin_half = np.linalg.norm(axis) * 0.5
    # 当 d ≈ +z 时无旋转；当 d ≈ -z 时绕任意垂直轴转 180°
    if np.linalg.norm(axis) < 1e-12:
        if cos_angle > 0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        # 反向：绕 X 轴 180°
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(cos_angle)
    cos_half = np.cos(angle * 0.5)
    sin_half = np.sin(angle * 0.5)
    return np.array(
        [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half],
        dtype=np.float32,
    )


def _make_cylinder_instance(
    p_from: Sequence[float],
    p_to: Sequence[float],
    radius: float,
    color: Sequence[float],
    flags: int = InstanceFlags.NONE,
) -> mjc_message_pb2.DebugMeshInstance:
    """从 from→to 构造圆柱实例的 transform。

    圆柱沿 +Z 轴建模，高度 = |to - from|，半径 = radius。
    位置取中点，旋转将 +Z 对齐到 (to - from) 方向，scale = [radius, radius, height]。
    """
    a = np.asarray(p_from, dtype=np.float64)
    b = np.asarray(p_to, dtype=np.float64)
    delta = b - a
    height = float(np.linalg.norm(delta))
    center = (a + b) * 0.5
    quat = _direction_to_quat(delta)
    scale = np.array([radius, radius, height], dtype=np.float32)
    return _make_instance(center.tolist(), quat.tolist(), scale.tolist(), color, flags)


def _make_arrow_instance(
    p_from: Sequence[float],
    p_to: Sequence[float],
    shaft_radius: float,
    color: Sequence[float],
    flags: int = InstanceFlags.NONE,
) -> mjc_message_pb2.DebugMeshInstance:
    """从 from→to 构造箭头实例的 transform。

    箭头沿 +Z 轴建模，总长 = |to - from|。
    C++ 侧 Arrow 网格内部已处理箭头头/杆比例，Python 侧只需提供
    总长度作为 scale.z、杆半径作为 scale.x/y，方向旋转同圆柱。
    """
    a = np.asarray(p_from, dtype=np.float64)
    b = np.asarray(p_to, dtype=np.float64)
    delta = b - a
    length = float(np.linalg.norm(delta))
    center = (a + b) * 0.5
    quat = _direction_to_quat(delta)
    scale = np.array([shaft_radius, shaft_radius, length], dtype=np.float32)
    return _make_instance(center.tolist(), quat.tolist(), scale.tolist(), color, flags)


def _handle_to_dict(handle: Any) -> Dict[str, Any]:
    """proto DebugMeshHandle → dict（便于 Python 侧传递与序列化）。

    generation==0 表示 C++ 侧返回的无效句柄（CreateMeshBatch 中 slot 已满等）。
    """
    return {
        "index": int(handle.index),
        "generation": int(handle.generation),
        "valid": int(handle.generation) != 0,
    }


class DebugDraw:
    """DebugMesh 调试绘制封装。stub=None 时离线模式，所有方法 no-op。

    stub 约定：
        OrcaStudio 生产环境使用 grpc.aio.insecure_channel 构造的 GrpcServiceStub，
        其 RPC 方法为 async（需 await）。故本类所有 RPC 方法均为 async def，
        调用方在同步上下文用 loop.run_until_complete(dd.draw_sphere(...))，
        在 async 上下文用 await dd.draw_sphere(...)。
        离线模式（stub=None）方法仍为 async 但立即 return，便于统一 await。

    线程安全：本类不做同步，调用方需保证单线程访问（OrcaGymEnv 主循环内调用即可）。
    """

    def __init__(self, stub: Any = None):
        self._stub = stub

    @property
    def is_online(self) -> bool:
        """是否连接到 OrcaStudio（stub != None）。"""
        return self._stub is not None

    # ============= Immediate（单帧，下一帧 Simulate flush 后消失）=============
    async def draw_sphere(
        self,
        center: Sequence[float],
        radius: float,
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
    ) -> None:
        inst = _make_instance(center, [0, 0, 0, 1], [radius, radius, radius], color, flags)
        await self.draw_batch(DebugMeshType.SPHERE, [inst])

    async def draw_box(
        self,
        center: Sequence[float],
        size: Sequence[float],
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
    ) -> None:
        inst = _make_instance(center, [0, 0, 0, 1], size, color, flags)
        await self.draw_batch(DebugMeshType.BOX, [inst])

    async def draw_cylinder(
        self,
        p_from: Sequence[float],
        p_to: Sequence[float],
        radius: float,
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
    ) -> None:
        inst = _make_cylinder_instance(p_from, p_to, radius, color, flags)
        await self.draw_batch(DebugMeshType.CYLINDER, [inst])

    async def draw_arrow(
        self,
        p_from: Sequence[float],
        p_to: Sequence[float],
        shaft_radius: float,
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
    ) -> None:
        inst = _make_arrow_instance(p_from, p_to, shaft_radius, color, flags)
        await self.draw_batch(DebugMeshType.ARROW, [inst])

    async def draw_quad(
        self,
        center: Sequence[float],
        size: Sequence[float],
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
    ) -> None:
        inst = _make_instance(center, [0, 0, 0, 1], size, color, flags)
        await self.draw_batch(DebugMeshType.QUAD, [inst])

    async def draw_batch(self, mesh_type: int, instances: Sequence[mjc_message_pb2.DebugMeshInstance]) -> None:
        """批量绘制同类型实例（immediate 模式）。离线模式直接 return。"""
        if self._stub is None:
            return
        req = mjc_message_pb2.DrawDebugMeshBatchRequest()
        req.mesh_type = int(mesh_type)
        req.instances.extend(instances)
        await self._stub.DrawDebugMeshBatch(req)
        # 调用方可自行包一层捕获响应；此处不返回，绘制失败（如无 handler）
        # 通常不应中断仿真流程。

    async def clear(self) -> None:
        """清空 immediate 队列。离线模式 no-op。"""
        if self._stub is None:
            return
        await self._stub.ClearDebugMesh(mjc_message_pb2.ClearDebugMeshRequest())

    # ============= Retained（跨帧持久对象，返回句柄供后续更新/销毁）=============
    async def create_objects(
        self, mesh_type: int, instances: Sequence[mjc_message_pb2.DebugMeshInstance]
    ) -> List[Dict[str, Any]]:
        """创建持久调试网格对象，返回句柄列表。

        离线模式返回空列表。每个句柄为
        {'index': int, 'generation': int, 'valid': bool}，
        generation==0 表示 C++ 侧分配失败（slot 已满等），调用方应忽略。
        """
        if self._stub is None:
            return []
        req = mjc_message_pb2.CreateDebugMeshObjectsRequest()
        req.mesh_type = int(mesh_type)
        req.instances.extend(instances)
        resp = await self._stub.CreateDebugMeshObjects(req)
        return [_handle_to_dict(h) for h in resp.handles]

    async def update_transforms(
        self,
        handles: Sequence[Dict[str, Any]],
        instances: Sequence[mjc_message_pb2.DebugMeshInstance],
    ) -> None:
        """更新持久对象的 transform（全量替换 instance 数据）。离线模式 no-op。

        handles 与 instances 须等长且顺序对应。
        """
        if self._stub is None:
            return
        req = mjc_message_pb2.UpdateDebugMeshTransformsRequest()
        for h in handles:
            ph = req.handles.add()
            ph.index = int(h["index"])
            ph.generation = int(h["generation"])
        req.instances.extend(instances)
        await self._stub.UpdateDebugMeshTransforms(req)

    async def destroy_objects(self, handles: Sequence[Dict[str, Any]]) -> None:
        """销毁持久对象，释放 slot。离线模式 no-op。"""
        if self._stub is None:
            return
        req = mjc_message_pb2.DestroyDebugMeshObjectsRequest()
        for h in handles:
            ph = req.handles.add()
            ph.index = int(h["index"])
            ph.generation = int(h["generation"])
        await self._stub.DestroyDebugMeshObjects(req)

    async def query_count(self) -> Dict[str, int]:
        """查询当前 retained / immediate 对象计数。离线模式返回全零。"""
        if self._stub is None:
            return {"retained": 0, "immediate": 0}
        resp = await self._stub.QueryDebugMeshCount(mjc_message_pb2.QueryDebugMeshCountRequest())
        return {"retained": int(resp.retained_count), "immediate": int(resp.immediate_count)}
