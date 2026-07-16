"""OrcaDebugDraw — DebugMesh 调试绘制封装（独立顶层模块）。

依赖约束：
    本模块为顶层工具模块，禁止依赖 orca_gym.core / orca_gym.environment。
    只依赖 numpy + protos 生成产物 + 标准库。

离线模式（stub=None）所有方法 no-op，不抛异常，便于在无 OrcaStudio
连接的环境下复用上层逻辑（如离线渲染、单元测试）。

坐标系约定（见设计文档 §2.3）：
    全链路统一：右手系、Z-up（X=右，Y=前，Z=上）、单位=米。
    各层 API 之间不做坐标系转换，唯一需要处理的是四元数存储顺序：
        - MuJoCo data.xquat 返回 [w,x,y,z]
        - 本模块 / gRPC proto / AZ::Quaternion 均使用 [x,y,z,w]
    从 MuJoCo 取位姿数据时需调用 quat_mujoco_to_grpc() 重排四元数顺序。

    字段定义：
        position = [x, y, z]     Z-up 世界坐标，单位米
        rotation = [x, y, z, w]  四元数（非 MuJoCo 的 [w,x,y,z]）
        scale    = [sx, sy, sz]  per-axis 非均匀缩放
        color    = [r, g, b, a]  0..1 浮点

    方向性基元（Cylinder/Cone/Arrow）模型空间沿 +Y（前方），
    _direction_to_quat 将 +Y 对齐到目标方向，详见各函数文档。
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from orca_gym.protos import mjc_message_pb2


class InstanceFlags:
    """与 OrcaDebugMesh::InstanceFlags 严格对应（OrcaDebugMeshInstanceData.h）。"""

    NONE = 0
    EDGE_HIGHLIGHT = 1 << 0
    WIREFRAME = 1 << 1  # W2: render as LineList wireframe (o_wireframe PSO variant)


class DebugMeshType:
    """与 OrcaDebugMesh::MeshType / proto DebugMeshType 严格对应。"""

    SPHERE = mjc_message_pb2.DEBUG_MESH_SPHERE
    CYLINDER = mjc_message_pb2.DEBUG_MESH_CYLINDER
    CONE = mjc_message_pb2.DEBUG_MESH_CONE
    BOX = mjc_message_pb2.DEBUG_MESH_BOX
    QUAD = mjc_message_pb2.DEBUG_MESH_QUAD
    ARROW = mjc_message_pb2.DEBUG_MESH_ARROW


# Arrow 网格在 GeometryGenerator 中 baked-in 的固定比例（非单位半径）：
#   shaft radius = 0.05, tip radius = 0.15, total length = 1.0（沿 +Y）。
# 而 Cylinder/Cone 是单位半径(1.0)。故 _make_arrow_instance 需将用户传入的
# 绝对杆半径除以此 baked-in 值，得到正确的 scale.x/z。
_ARROW_BAKED_SHAFT_RADIUS = 0.05


def quat_mujoco_to_grpc(wxyz: Sequence[float]) -> List[float]:
    """将 MuJoCo 四元数 [w,x,y,z] 转为 gRPC/O3DE 四元数 [x,y,z,w]。

    这是存储布局重排，非坐标系转换（见设计文档 §2.3.3）。
    MuJoCo data.body_xquat() 返回 [w,x,y,z]，需用本函数转换后传入
    DebugDraw 的 rotation 字段或 _make_instance。

    Args:
        wxyz: [w, x, y, z] MuJoCo 四元数

    Returns:
        [x, y, z, w] gRPC/O3DE 四元数
    """
    return [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]


def _make_instance(
    position: Sequence[float],
    rotation: Sequence[float],
    scale: Sequence[float],
    color: Sequence[float],
    flags: int = InstanceFlags.NONE,
) -> mjc_message_pb2.DebugMeshInstance:
    """构造 DebugMeshInstance proto。position/rotation/scale/color 支持 list/ndarray。

    Args:
        position: [x, y, z] Z-up 世界坐标锚点（单位米）。锚点含义随图形而异，
            详见 DebugDraw 类 docstring 的 "Position 锚点约定" 表。
        rotation: [x, y, z, w] O3DE 四元数（非 MuJoCo 的 [w,x,y,z]）。
            作用于模型空间基元：
              - 无方向性基元（Sphere/Box/Quad）旋转不改变外观对称性
              - 方向性基元（Cylinder/Cone/Arrow）默认沿 +Y，rotation 旋转 +Y 到目标方向
            从 MuJoCo 取四元数需先 quat_mujoco_to_grpc() 重排。
            要将方向性基元对齐到任意方向 d，可用 _direction_to_quat(d) 生成。
        scale:    [sx, sy, sz] per-axis 缩放（米）。方向性基元的 sy 为长度，
            sx/sz 为半径；Arrow 的 sx/sz 需除以 baked-in shaft radius，见 _make_arrow_instance。
        color:    [r, g, b, a] 0..1
        flags:    InstanceFlags bitmask
    """
    inst = mjc_message_pb2.DebugMeshInstance()
    inst.position.extend(np.asarray(position, dtype=np.float32).tolist())
    inst.rotation.extend(np.asarray(rotation, dtype=np.float32).tolist())
    inst.scale.extend(np.asarray(scale, dtype=np.float32).tolist())
    inst.color.extend(np.asarray(color, dtype=np.float32).tolist())
    inst.flags = int(flags)
    return inst


def _direction_to_quat(direction: np.ndarray) -> np.ndarray:
    """单位方向向量 → 四元数 [x,y,z,w]。

    语义：将方向性基元的模型空间 +Y 轴对齐到 ``direction``。
      - Cylinder/Cone/Arrow 沿 +Y 建模（见 OrcaDebugMeshGeometryGenerator.h）
      - direction = (to - from)，结果四元数使基元从 from 指向 to
    direction 会被归一化；零向量返回单位四元数（无旋转，保持 +Y 朝向）。

    注意：O3DE/AZ::Quaternion 使用 Hamilton 共轭形式 (q*·v·q)，旋转方向与
    标准约定 (q·v·q*) 相反，故内部使用 ``cross(d, y)`` 而非 ``cross(y, d)``。
    """
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    d = d / norm

    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    # 旋转轴 = d × y（非 y × d）。O3DE/AZ::Quaternion 使用 Hamilton 约定的
    # 共轭形式 (q*·v·q)，与标准数学约定 (q·v·q*) 相比旋转方向相反。
    # 故此处用 reversed cross 使最终视觉效果为 "将 +Y 轴对齐到 direction"。
    axis = np.cross(d, y)
    cos_angle = float(np.clip(np.dot(y, d), -1.0, 1.0))
    # 当 d ≈ +y 时无旋转；当 d ≈ -y 时绕任意垂直轴转 180°
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

    方向语义：圆柱底面中心贴 p_from，顶面中心贴 p_to，轴向 = (p_to - p_from)。

    实现细节（与 C++ 锚点一致）：
        C++ 侧 Cylinder 模型空间原点在底面中心（y=0，见
        OrcaDebugMeshGeometryGenerator.h），shader 执行
        worldPos = rot(modelPos * scale) + position。
        故 position = p_from（底面中心），旋转将 +Y 对齐到 (to-from)，
        scale.y = height 使顶面落在 p_to。
        scale = [radius, height, radius]，height = |to - from|。

    坐标系：p_from / p_to 为 Z-up 右手系世界坐标，单位米。
    """
    a = np.asarray(p_from, dtype=np.float64)
    b = np.asarray(p_to, dtype=np.float64)
    delta = b - a
    height = float(np.linalg.norm(delta))
    quat = _direction_to_quat(delta)
    scale = np.array([radius, height, radius], dtype=np.float32)
    return _make_instance(a.tolist(), quat.tolist(), scale.tolist(), color, flags)


def _make_arrow_instance(
    p_from: Sequence[float],
    p_to: Sequence[float],
    shaft_radius: float,
    color: Sequence[float],
    flags: int = InstanceFlags.NONE,
) -> mjc_message_pb2.DebugMeshInstance:
    """从 from→to 构造箭头实例的 transform。

    方向语义：箭头杆底（尾部）贴 p_from，箭尖贴 p_to，方向 = (p_to - p_from)。

    实现细节（与 C++ 锚点一致，同 _make_cylinder_instance）：
        C++ 侧 Arrow 模型空间原点在杆底（y=0，见
        OrcaDebugMeshGeometryGenerator.h），shader 执行
        worldPos = rot(modelPos * scale) + position。
        故 position = p_from（杆底），旋转将 +Y 对齐到 (to-from)，
        scale.y = length 使箭尖落在 p_to。
        Arrow 网格 baked-in shaft radius=0.05 / tip radius=0.15（非单位半径），
        故 scale.x/z = shaft_radius / 0.05 使 shaft_radius 为绝对米值；
        tip 半径自动按 3:1 比例缩放。scale.y = 总长度。

    坐标系：p_from / p_to 为 Z-up 右手系世界坐标，单位米。
    """
    a = np.asarray(p_from, dtype=np.float64)
    b = np.asarray(p_to, dtype=np.float64)
    delta = b - a
    length = float(np.linalg.norm(delta))
    quat = _direction_to_quat(delta)
    r_scale = shaft_radius / _ARROW_BAKED_SHAFT_RADIUS
    scale = np.array([r_scale, length, r_scale], dtype=np.float32)
    return _make_instance(a.tolist(), quat.tolist(), scale.tolist(), color, flags)


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

    坐标系约定（见设计文档 §2.3）：
        - 所有 position / center / p_from / p_to 参数均为 Z-up 右手系世界坐标，单位米
        - 所有 rotation 参数为 [x, y, z, w] 四元数（O3DE 原生，非 MuJoCo 的 [w,x,y,z]）
        - 从 MuJoCo 获取的位姿需用 quat_mujoco_to_grpc() 重排四元数顺序

    方向性基元约定（Cylinder/Cone/Arrow）：
        模型空间沿 +Y（见 OrcaDebugMeshGeometryGenerator.h）：
          - Cylinder：底面 y=0，顶面 y=1（position 锚点=底面中心）
          - Cone    ：底面 y=0，apex y=1（position 锚点=底面中心）
          - Arrow   ：杆底 y=0，箭尖 y=1（position 锚点=杆底）

        旋转 API 语义（rotation 字段 / _make_instance / create_objects / update_transforms）：
          rotation 为 O3DE [x,y,z,w] 四元数，将模型空间 +Y 轴旋转到目标方向。
          position 直接作为 C++ 锚点（底面/杆底），无偏移补偿。
          要将方向性基元对齐到任意方向 d，用 _direction_to_quat(d) 生成四元数。
          例：圆锥朝上(+Z) → _direction_to_quat([0,0,1])；
              圆锥朝 -Y → _direction_to_quat([0,-1,0])。
          若希望视觉上从 from 到 to：
              position = from，rotation = _direction_to_quat(to - from)，scale.y = |to - from|。
          O3DE/AZ::Quaternion 使用 Hamilton 共轭形式，旋转方向与标准约定相反，
          _direction_to_quat 内部已用 cross(d, y) 补偿。

        指向 API 语义（draw_cylinder/draw_arrow 的 p_from→p_to）：
          方向 = (p_to - p_from)，_direction_to_quat 自动生成旋转。
          position = p_from（底面/杆底），与 C++ 锚点一致，
          底面/杆底贴 p_from，顶面/箭尖贴 p_to。
          （immediate 和 retained 语义完全一致）

    Position 锚点约定（C++ 侧 OrcaDebugMeshGeometryGenerator 模型空间原点）：
        无方向性基元的 position = 几何中心；方向性基元的 position = 底部/尾部。
        immediate（draw_*）和 retained（create_objects/update_transforms）共用此约定。

        ┌──────────┬─────────────────────┬────────────────────┐
        │ 图形     │ position 含义       │ 模型空间描述         │
        ├──────────┼─────────────────────┼────────────────────┤
        │ Sphere   │ 球心（几何中心）    │ 半径 1，原点在球心   │
        │ Box      │ 盒心（几何中心）    │ 半宽 0.5，原点在盒心 │
        │ Quad     │ 四边形中心          │ 半宽 0.5，XY 平面    │
        │ Cylinder │ **底面中心**        │ 沿 +Y，底面在原点   │
        │ Cone     │ **底面中心**        │ 沿 +Y，底面在原点   │
        │ Arrow    │ **尾部（杆底）**    │ 沿 +Y，总长 1       │
        └──────────┴─────────────────────┴────────────────────┘

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

    # ============= Immediate（单帧 / TTL）=============
    async def draw_sphere(
        self,
        center: Sequence[float],
        radius: float,
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
        wireframe: bool = False,
        duration: float = 0.0,
    ) -> None:
        """绘制球体（immediate）。

        Args:
            center: [x, y, z] 球心位置（Z-up 世界坐标，单位米）
            radius: 球体半径，米
            color:  [r, g, b, a] 0..1
            flags:  InstanceFlags bitmask
            wireframe: W5 便捷参数，True 时 OR 入 InstanceFlags.WIREFRAME（线框渲染）
            duration: TTL 持续时长（秒）。0=单帧（默认），>0 跨帧存活至仿真时间到期，防闪烁
        """
        if wireframe:
            flags |= InstanceFlags.WIREFRAME
        inst = _make_instance(center, [0, 0, 0, 1], [radius, radius, radius], color, flags)
        await self.draw_batch(DebugMeshType.SPHERE, [inst], duration=duration)

    async def draw_box(
        self,
        center: Sequence[float],
        size: Sequence[float],
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
        wireframe: bool = False,
        duration: float = 0.0,
    ) -> None:
        """绘制立方体（immediate）。

        Args:
            center: [x, y, z] 盒心位置（Z-up 世界坐标，单位米）
            size:   [sx, sy, sz] 三轴边长，米
            color:  [r, g, b, a] 0..1
            flags:  InstanceFlags bitmask
            wireframe: W5 便捷参数，True 时 OR 入 InstanceFlags.WIREFRAME（线框渲染）
            duration: TTL 持续时长（秒）。0=单帧（默认），>0 跨帧存活至仿真时间到期，防闪烁
        """
        if wireframe:
            flags |= InstanceFlags.WIREFRAME
        inst = _make_instance(center, [0, 0, 0, 1], size, color, flags)
        await self.draw_batch(DebugMeshType.BOX, [inst], duration=duration)

    async def draw_cylinder(
        self,
        p_from: Sequence[float],
        p_to: Sequence[float],
        radius: float,
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
        wireframe: bool = False,
        duration: float = 0.0,
    ) -> None:
        """绘制圆柱（immediate）。圆柱从 p_from 延伸到 p_to。

        Args:
            p_from: [x, y, z] 起点（对应模型空间底面），Z-up 世界坐标，米
            p_to:   [x, y, z] 终点，Z-up 世界坐标，米
            radius: 圆柱半径，米
            color:  [r, g, b, a] 0..1
            flags:  InstanceFlags bitmask
            wireframe: W5 便捷参数，True 时 OR 入 InstanceFlags.WIREFRAME（线框渲染）
            duration: TTL 持续时长（秒）。0=单帧（默认），>0 跨帧存活至仿真时间到期，防闪烁
        """
        if wireframe:
            flags |= InstanceFlags.WIREFRAME
        inst = _make_cylinder_instance(p_from, p_to, radius, color, flags)
        await self.draw_batch(DebugMeshType.CYLINDER, [inst], duration=duration)

    async def draw_arrow(
        self,
        p_from: Sequence[float],
        p_to: Sequence[float],
        shaft_radius: float,
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
        wireframe: bool = False,
        duration: float = 0.0,
    ) -> None:
        """绘制箭头（immediate）。箭头从 p_from 指向 p_to。

        Args:
            p_from: [x, y, z] 尾部位置（对应模型空间杆底），Z-up 世界坐标，米
            p_to:   [x, y, z] 头部（箭尖）位置，Z-up 世界坐标，米
            shaft_radius: 箭杆半径，米
            color:  [r, g, b, a] 0..1
            flags:  InstanceFlags bitmask
            wireframe: W5 便捷参数，True 时 OR 入 InstanceFlags.WIREFRAME（线框渲染）
            duration: TTL 持续时长（秒）。0=单帧（默认），>0 跨帧存活至仿真时间到期，防闪烁
        """
        if wireframe:
            flags |= InstanceFlags.WIREFRAME
        inst = _make_arrow_instance(p_from, p_to, shaft_radius, color, flags)
        await self.draw_batch(DebugMeshType.ARROW, [inst], duration=duration)

    async def draw_quad(
        self,
        center: Sequence[float],
        size: Sequence[float],
        color: Sequence[float],
        flags: int = InstanceFlags.NONE,
        wireframe: bool = False,
        duration: float = 0.0,
    ) -> None:
        """绘制双面 Quad（immediate）。Quad 在 XY 平面，法线沿 +Z。

        Args:
            center: [x, y, z] 四边形中心位置（Z-up 世界坐标，单位米）
            size:   [sx, sy, sz] 三轴缩放，米（sz 通常不影响扁平 quad）
            color:  [r, g, b, a] 0..1
            flags:  InstanceFlags bitmask
            wireframe: W5 便捷参数，True 时 OR 入 InstanceFlags.WIREFRAME（线框渲染）
            duration: TTL 持续时长（秒）。0=单帧（默认），>0 跨帧存活至仿真时间到期，防闪烁
        """
        if wireframe:
            flags |= InstanceFlags.WIREFRAME
        inst = _make_instance(center, [0, 0, 0, 1], size, color, flags)
        await self.draw_batch(DebugMeshType.QUAD, [inst], duration=duration)

    async def draw_batch(
        self,
        mesh_type: int,
        instances: Sequence[mjc_message_pb2.DebugMeshInstance],
        duration: float = 0.0,
    ) -> None:
        """批量绘制同类型实例（immediate 模式）。离线模式直接 return。

        Args:
            mesh_type: DebugMeshType 枚举值
            instances: DebugMeshInstance proto 列表
            duration:  TTL 持续时长（秒）。0=单帧（默认），>0 跨帧存活至仿真时间到期，防闪烁
        """
        if self._stub is None:
            return
        req = mjc_message_pb2.DrawDebugMeshBatchRequest()
        req.mesh_type = int(mesh_type)
        req.instances.extend(instances)
        req.duration_seconds = float(duration)  # TTL：0=单帧，>0 跨帧存活
        await self._stub.DrawDebugMeshBatch(req)
        # 调用方可自行包一层捕获响应；此处不返回，绘制失败（如无 handler）
        # 通常不应中断仿真流程。

    async def clear(self) -> None:
        """清空 immediate 队列。离线模式 no-op。"""
        if self._stub is None:
            return
        await self._stub.ClearDebugMesh(mjc_message_pb2.ClearDebugMeshRequest())

    # ============= Retained（跨帧持久对象 + Keepalive，返回句柄供后续更新/销毁）=============
    async def create_objects(
        self,
        mesh_type: int,
        instances: Sequence[mjc_message_pb2.DebugMeshInstance],
        keepalive: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """创建持久调试网格对象，返回句柄列表。

        离线模式返回空列表。每个句柄为
        {'index': int, 'generation': int, 'valid': bool}，
        generation==0 表示 C++ 侧分配失败（slot 已满等），调用方应忽略。

        方向性基元（Cylinder/Cone/Arrow）：
            instances 中的 rotation 直接用于旋转模型空间 +Y 到目标方向，
            position 直接作为 C++ 锚点（底面/杆底），与 draw_cylinder/draw_arrow 语义一致。
            要对齐到方向 d，用 _direction_to_quat(d) 生成 rotation。
            若希望视觉上从 from 到 to，position = from，
            rotation=_direction_to_quat(to-from)，scale.y=|to-from|。

        Args:
            mesh_type:  DebugMeshType 枚举值
            instances:  DebugMeshInstance proto 列表
            keepalive:  保活时长（秒，默认 1.0）。超时未收到 update/keepalive 则自动销毁。
                        用于断连/崩溃等异常场景的残留清理。
        """
        if self._stub is None:
            return []
        req = mjc_message_pb2.CreateDebugMeshObjectsRequest()
        req.mesh_type = int(mesh_type)
        req.instances.extend(instances)
        req.keepalive_seconds = float(keepalive)  # 默认 1.0s，断连自清
        resp = await self._stub.CreateDebugMeshObjects(req)
        return [_handle_to_dict(h) for h in resp.handles]

    async def update_transforms(
        self,
        handles: Sequence[Dict[str, Any]],
        instances: Sequence[mjc_message_pb2.DebugMeshInstance],
    ) -> None:
        """更新持久对象的 transform（全量替换 instance 数据）。离线模式 no-op。

        handles 与 instances 须等长且顺序对应。
        方向性基元的 rotation/position 语义同 create_objects。
        每次调用自动刷新 keepalive 计时器（C++ UpdateInstance 传入当前 sim time）。
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

    async def keepalive_objects(self, handles: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """保活心跳：刷新 handle 的保活计时器，不修改 transform。

        用于"对象位置不变但仍需保活"的场景（如静态标注点）。建议在达到 1/2
        保活时长时触发一次。开销极低（无 GPU 操作）。

        离线模式返回 {'alive': [], 'expired': []}。

        Args:
            handles: 句柄列表，每个为 {'index': int, 'generation': int}

        Returns:
            {'alive': [...], 'expired': [...]} —— alive 为仍存活的句柄，
            expired 为已过期/无效的句柄，调用方应从本地列表中移除。
        """
        if self._stub is None:
            return {"alive": [], "expired": []}
        req = mjc_message_pb2.KeepAliveDebugMeshObjectsRequest()
        for h in handles:
            ph = req.handles.add()
            ph.index = int(h["index"])
            ph.generation = int(h["generation"])
        resp = await self._stub.KeepAliveDebugMeshObjects(req)
        return {
            "alive": [_handle_to_dict(h) for h in resp.alive_handles],
            "expired": [_handle_to_dict(h) for h in resp.expired_handles],
        }

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
