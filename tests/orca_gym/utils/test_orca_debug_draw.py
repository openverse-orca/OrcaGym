"""orca_debug_draw 模块单元测试。

覆盖：
- 离线模式（stub=None）所有方法 no-op，不抛异常
- proto 构造正确性（字段顺序、四元数顺序）
- 在线模式（mock stub）验证 stub 被正确调用
- cylinder/arrow 几何计算（方向→四元数、长度、半径）

不依赖真实 gRPC，全部 CPU 纯测试。

注：DebugDraw 的 RPC 方法为 async（生产环境使用 grpc.aio stub），
离线 no-op 方法也保持 async 签名，测试用 asyncio.run 驱动。
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from orca_gym.protos import mjc_message_pb2
from orca_gym.utils.orca_debug_draw import (
    DebugDraw,
    DebugMeshType,
    InstanceFlags,
    _direction_to_quat,
    _handle_to_dict,
    _make_arrow_instance,
    _make_cylinder_instance,
    _make_instance,
    quat_mujoco_to_grpc,
)


# ============================================================
# 离线模式（stub=None）
# ============================================================
class TestOfflineMode:
    """离线模式所有方法 no-op，不抛异常。"""

    def test_offline_draw_sphere_noop(self):
        dd = DebugDraw(stub=None)
        asyncio.run(dd.draw_sphere([0, 0, 0], 1.0, [1, 0, 0, 1]))  # 不抛异常

    def test_offline_draw_arrow_noop(self):
        dd = DebugDraw(stub=None)
        asyncio.run(dd.draw_arrow([0, 0, 0], [1, 0, 0], 0.05, [1, 1, 0, 1]))

    def test_offline_clear_noop(self):
        dd = DebugDraw(stub=None)
        asyncio.run(dd.clear())

    def test_offline_create_returns_empty(self):
        dd = DebugDraw(stub=None)
        result = asyncio.run(dd.create_objects(DebugMeshType.SPHERE, []))
        assert result == []

    def test_offline_update_transforms_noop(self):
        dd = DebugDraw(stub=None)
        asyncio.run(dd.update_transforms([{"index": 0, "generation": 1}], []))

    def test_offline_destroy_objects_noop(self):
        dd = DebugDraw(stub=None)
        asyncio.run(dd.destroy_objects([{"index": 0, "generation": 1}]))

    def test_offline_query_count_returns_zero(self):
        dd = DebugDraw(stub=None)
        result = asyncio.run(dd.query_count())
        assert result == {"retained": 0, "immediate": 0}

    def test_offline_is_online_false(self):
        assert DebugDraw(stub=None).is_online is False


# ============================================================
# proto 构造
# ============================================================
class TestProtoConstruction:
    """验证 proto 构造正确性（不依赖真实 gRPC）。"""

    def test_make_instance_fields(self):
        inst = _make_instance([1, 2, 3], [0, 0, 0, 1], [1, 1, 1], [1, 0, 0, 1])
        assert list(inst.position) == [1, 2, 3]
        assert list(inst.rotation) == [0, 0, 0, 1]
        assert list(inst.scale) == [1, 1, 1]
        assert list(inst.color) == [1, 0, 0, 1]
        assert inst.flags == InstanceFlags.NONE

    def test_make_instance_with_edge_highlight(self):
        inst = _make_instance(
            [0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1],
            [1, 1, 1, 1],
            flags=InstanceFlags.EDGE_HIGHLIGHT,
        )
        assert inst.flags == InstanceFlags.EDGE_HIGHLIGHT

    def test_make_instance_accepts_ndarray(self):
        inst = _make_instance(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([0.5, 0.5, 0.5, 1.0]),
        )
        assert list(inst.position) == [1, 2, 3]
        assert list(inst.scale) == [2, 3, 4]

    def test_make_instance_flags_default_none(self):
        inst = _make_instance([0, 0, 0], [0, 0, 0, 1], [1, 1, 1], [1, 1, 1, 1])
        assert inst.flags == 0

    def test_debug_mesh_type_values_match_proto(self):
        """DebugMeshType 常量必须与 proto 枚举值严格对应。"""
        assert DebugMeshType.SPHERE == mjc_message_pb2.DEBUG_MESH_SPHERE
        assert DebugMeshType.CYLINDER == mjc_message_pb2.DEBUG_MESH_CYLINDER
        assert DebugMeshType.CONE == mjc_message_pb2.DEBUG_MESH_CONE
        assert DebugMeshType.BOX == mjc_message_pb2.DEBUG_MESH_BOX
        assert DebugMeshType.QUAD == mjc_message_pb2.DEBUG_MESH_QUAD
        assert DebugMeshType.ARROW == mjc_message_pb2.DEBUG_MESH_ARROW

    def test_instance_flags_values(self):
        assert InstanceFlags.NONE == 0
        assert InstanceFlags.EDGE_HIGHLIGHT == 1
        assert InstanceFlags.WIREFRAME == 2


# ============================================================
# 句柄转换
# ============================================================
class TestHandleConversion:
    def test_handle_to_dict_valid(self):
        h = mjc_message_pb2.DebugMeshHandle()
        h.index = 3
        h.generation = 5
        d = _handle_to_dict(h)
        assert d == {"index": 3, "generation": 5, "valid": True}

    def test_handle_to_dict_invalid_generation_zero(self):
        """generation==0 表示 C++ 侧返回的无效句柄。"""
        h = mjc_message_pb2.DebugMeshHandle()
        h.index = 0
        h.generation = 0
        d = _handle_to_dict(h)
        assert d == {"index": 0, "generation": 0, "valid": False}


# ============================================================
# 几何计算：方向→四元数（+Y 对齐到给定方向）
# ============================================================
class TestDirectionToQuat:
    def test_identity_direction_returns_identity_quat(self):
        """+Y（mesh 默认轴）→ 无旋转。"""
        q = _direction_to_quat([0, 1, 0])
        np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-6)

    def test_zero_direction_returns_identity_quat(self):
        q = _direction_to_quat([0, 0, 0])
        np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-6)

    def test_negative_y_returns_180_around_x(self):
        """+Y → -Y 应绕 X 轴转 180°，四元数 [1,0,0,0]。"""
        q = _direction_to_quat([0, -1, 0])
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-6)

    def test_z_axis_rotation(self):
        """+Y → +Z：axis = y×d = +X，angle = +90°，四元数 [sin45, 0, 0, cos45]。"""
        q = _direction_to_quat([0, 0, 1])
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        np.testing.assert_allclose(q, [s, 0, 0, c], atol=1e-6)

    def test_x_axis_rotation(self):
        """+Y → +X：axis = y×d = -Z，angle = +90°，四元数 [0, 0, -sin45, cos45]。"""
        q = _direction_to_quat([1, 0, 0])
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        np.testing.assert_allclose(q, [0, 0, -s, c], atol=1e-6)

    def test_unit_vector_input_unchanged_by_magnitude(self):
        """方向相同、长度不同应得到相同四元数。"""
        q1 = _direction_to_quat([0, 1, 2])
        q2 = _direction_to_quat([0, 2, 4])
        np.testing.assert_allclose(q1, q2, atol=1e-6)

    def test_quaternion_is_unit(self):
        for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-1, 2, 3]):
            q = _direction_to_quat(d)
            np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-6)


# ============================================================
# 几何计算：cylinder / arrow 实例构造
# Cylinder/Cone/Arrow 沿 +Y 建模（见 OrcaDebugMeshGeometryGenerator.h），
# scale = [radius, height, radius]，_direction_to_quat 将 +Y 对齐到方向。
# ============================================================
class TestCylinderArrowGeometry:
    def test_cylinder_along_y(self):
        """沿 +Y（mesh 默认轴）→ 无旋转。"""
        inst = _make_cylinder_instance([0, 0, 0], [0, 2, 0], radius=0.5, color=[1, 0, 0, 1])
        assert list(inst.position) == [0, 1, 0]
        np.testing.assert_allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(list(inst.scale), [0.5, 2.0, 0.5], atol=1e-6)

    def test_cylinder_along_z(self):
        """沿 +Z → 旋转 +Y→+Z，scale=[r, h, r]。"""
        inst = _make_cylinder_instance([0, 0, 0], [0, 0, 2], radius=0.5, color=[1, 0, 0, 1])
        assert list(inst.position) == [0, 0, 1]
        # 非单位旋转（+Y→+Z）
        assert not np.allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(list(inst.scale), [0.5, 2.0, 0.5], atol=1e-6)

    def test_cylinder_along_x(self):
        inst = _make_cylinder_instance([0, 0, 0], [2, 0, 0], radius=0.3, color=[1, 0, 0, 1])
        assert list(inst.position) == [1, 0, 0]
        np.testing.assert_allclose(list(inst.scale), [0.3, 2.0, 0.3], atol=1e-6)

    def test_cylinder_zero_length(self):
        """from==to 时高度=0，方向为单位四元数（不抛异常）。"""
        inst = _make_cylinder_instance([1, 2, 3], [1, 2, 3], radius=0.5, color=[1, 0, 0, 1])
        assert list(inst.position) == [1, 2, 3]
        np.testing.assert_allclose(list(inst.scale), [0.5, 0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)

    def test_arrow_along_y(self):
        """沿 +Y（mesh 默认轴）→ 无旋转。Arrow baked-in shaft r=0.05，
        shaft_radius=0.1 → scale.x/z = 0.1/0.05 = 2.0。"""
        inst = _make_arrow_instance([0, 0, 0], [0, 3, 0], shaft_radius=0.1, color=[1, 1, 0, 1])
        assert list(inst.position) == [0, 1.5, 0]
        np.testing.assert_allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(list(inst.scale), [2.0, 3.0, 2.0], atol=1e-6)

    def test_arrow_along_z(self):
        inst = _make_arrow_instance([0, 0, 0], [0, 0, 3], shaft_radius=0.1, color=[1, 1, 0, 1])
        assert list(inst.position) == [0, 0, 1.5]
        assert not np.allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(list(inst.scale), [2.0, 3.0, 2.0], atol=1e-6)

    def test_cylinder_flags_propagated(self):
        inst = _make_cylinder_instance(
            [0, 0, 0], [0, 1, 0], radius=0.5, color=[1, 0, 0, 1], flags=InstanceFlags.EDGE_HIGHLIGHT
        )
        assert inst.flags == InstanceFlags.EDGE_HIGHLIGHT


# ============================================================
# 在线模式（AsyncMock stub，模拟 grpc.aio stub）
# ============================================================
class TestOnlineWithMockStub:
    """在线模式（AsyncMock stub）验证 async stub 被正确 await。"""

    def test_draw_batch_calls_stub(self):
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        inst = _make_instance([0, 0, 0], [0, 0, 0, 1], [1, 1, 1], [1, 0, 0, 1])
        asyncio.run(dd.draw_batch(DebugMeshType.SPHERE, [inst]))
        stub.DrawDebugMeshBatch.assert_awaited_once()
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.mesh_type == DebugMeshType.SPHERE
        assert len(req.instances) == 1

    def test_draw_sphere_calls_stub_with_sphere_type(self):
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.draw_sphere([1, 2, 3], 2.0, [1, 0, 0, 1]))
        stub.DrawDebugMeshBatch.assert_awaited_once()
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.mesh_type == DebugMeshType.SPHERE
        assert list(req.instances[0].position) == [1, 2, 3]
        # sphere scale = [r, r, r]
        np.testing.assert_allclose(list(req.instances[0].scale), [2.0, 2.0, 2.0], atol=1e-6)

    def test_draw_sphere_wireframe_sets_flag(self):
        # W5: wireframe=True convenience param must OR in InstanceFlags.WIREFRAME
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.draw_sphere([0, 0, 0], 1.0, [1, 1, 1, 1], wireframe=True))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.instances[0].flags & InstanceFlags.WIREFRAME

    def test_draw_sphere_wireframe_combines_with_flags(self):
        # W5: wireframe=True must combine with caller-supplied flags (not overwrite)
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(
            dd.draw_sphere(
                [0, 0, 0], 1.0, [1, 1, 1, 1],
                flags=InstanceFlags.EDGE_HIGHLIGHT,
                wireframe=True,
            )
        )
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.instances[0].flags == (InstanceFlags.EDGE_HIGHLIGHT | InstanceFlags.WIREFRAME)

    def test_draw_box_wireframe_sets_flag(self):
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.draw_box([0, 0, 0], [1, 1, 1], [1, 1, 1, 1], wireframe=True))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.instances[0].flags & InstanceFlags.WIREFRAME

    def test_draw_sphere_wireframe_false_no_flag(self):
        # W5: wireframe=False (default) must NOT set the WIREFRAME bit
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.draw_sphere([0, 0, 0], 1.0, [1, 1, 1, 1]))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert not (req.instances[0].flags & InstanceFlags.WIREFRAME)

    def test_clear_calls_stub(self):
        stub = MagicMock()
        stub.ClearDebugMesh = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.clear())
        stub.ClearDebugMesh.assert_awaited_once()

    def test_create_objects_returns_handles(self):
        stub = MagicMock()
        resp = mjc_message_pb2.CreateDebugMeshObjectsResponse()
        h1 = resp.handles.add()
        h1.index = 0
        h1.generation = 1
        h2 = resp.handles.add()
        h2.index = 1
        h2.generation = 2
        stub.CreateDebugMeshObjects = AsyncMock(return_value=resp)

        dd = DebugDraw(stub=stub)
        result = asyncio.run(dd.create_objects(DebugMeshType.BOX, []))
        assert len(result) == 2
        assert result[0] == {"index": 0, "generation": 1, "valid": True}
        assert result[1] == {"index": 1, "generation": 2, "valid": True}

    def test_update_transforms_calls_stub(self):
        stub = MagicMock()
        stub.UpdateDebugMeshTransforms = AsyncMock()
        dd = DebugDraw(stub=stub)
        handles = [{"index": 0, "generation": 1}, {"index": 1, "generation": 2}]
        inst = _make_instance([0, 0, 0], [0, 0, 0, 1], [1, 1, 1], [1, 0, 0, 1])
        asyncio.run(dd.update_transforms(handles, [inst, inst]))
        stub.UpdateDebugMeshTransforms.assert_awaited_once()
        req = stub.UpdateDebugMeshTransforms.await_args[0][0]
        assert len(req.handles) == 2
        assert req.handles[0].index == 0 and req.handles[0].generation == 1
        assert len(req.instances) == 2

    def test_destroy_objects_calls_stub(self):
        stub = MagicMock()
        stub.DestroyDebugMeshObjects = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.destroy_objects([{"index": 5, "generation": 3}]))
        stub.DestroyDebugMeshObjects.assert_awaited_once()
        req = stub.DestroyDebugMeshObjects.await_args[0][0]
        assert req.handles[0].index == 5 and req.handles[0].generation == 3

    def test_query_count_returns_response(self):
        stub = MagicMock()
        resp = mjc_message_pb2.QueryDebugMeshCountResponse()
        resp.retained_count = 7
        resp.immediate_count = 3
        stub.QueryDebugMeshCount = AsyncMock(return_value=resp)
        dd = DebugDraw(stub=stub)
        result = asyncio.run(dd.query_count())
        assert result == {"retained": 7, "immediate": 3}

    def test_is_online_true_when_stub_set(self):
        assert DebugDraw(stub=MagicMock()).is_online is True


# ============================================================
# 坐标系与单位规范（对应设计文档 §2.3）
# 验证全链路 Z-up 右手系、单位米、四元数 [x,y,z,w] 约定。
# 关键检查点：
#   1. MuJoCo [w,x,y,z] → gRPC [x,y,z,w] 重排（非坐标系转换）
#   2. _direction_to_quat 的 +Y 模型空间约定
#   3. 方向性基元（cylinder/arrow）的旋转、长度、半径正确性
#   4. Z-up 坐标透传（无轴交换）
# ============================================================
class TestCoordinateSystem:
    """坐标系与单位规范正确性测试（对应设计文档 §2.3）。"""

    # ---------- 四元数顺序重排（MuJoCo → gRPC）----------
    def test_quat_mujoco_to_grpc_identity(self):
        """MuJoCo identity [1,0,0,0] → gRPC identity [0,0,0,1]。"""
        result = quat_mujoco_to_grpc([1.0, 0.0, 0.0, 0.0])
        assert result == [0.0, 0.0, 0.0, 1.0]

    def test_quat_mujoco_to_grpc_reorder(self):
        """MuJoCo [w,x,y,z] → gRPC [x,y,z,w] 重排：wxyz → yzwx? 不，是 wxyz → xyzw。"""
        # MuJoCo [w=0.707, x=0, y=0.707, z=0]（绕 Y 轴 90°）
        result = quat_mujoco_to_grpc([0.707, 0.0, 0.707, 0.0])
        assert result == [0.0, 0.707, 0.0, 0.707]

    def test_quat_mujoco_to_grpc_nontrivial(self):
        """非平凡四元数重排：[w,a,b,c] → [a,b,c,w]。"""
        result = quat_mujoco_to_grpc([0.5, 0.1, 0.2, 0.3])
        assert result == [0.1, 0.2, 0.3, 0.5]

    def test_quat_mujoco_to_grpc_roundtrip(self):
        """重排是双射：两次重排恢复原值。"""
        wxyz = [0.3, 0.4, 0.5, 0.6]
        xyzw = quat_mujoco_to_grpc(wxyz)
        # 注意：xyzw→wxyz 的重排是 [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
        back = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
        assert back == wxyz

    # ---------- +Y 模型空间约定（_direction_to_quat）----------
    def test_direction_to_quat_y_axis_identity(self):
        """+Y（mesh 默认轴）→ 单位旋转 [0,0,0,1]。"""
        q = _direction_to_quat([0, 1, 0])
        np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-6)

    def test_direction_to_quat_z_axis_rotation(self):
        """+Y → +Z（竖直向上）：绕 X 轴 +90°。

        Z-up 系统中 +Z 是上方向。+Y→+Z 旋转轴 = y×z = +x，角 = +90°。
        四元数 [x,y,z,w] = [sin45, 0, 0, cos45]。
        """
        q = _direction_to_quat([0, 0, 1])
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        np.testing.assert_allclose(q, [s, 0, 0, c], atol=1e-6)

    def test_direction_to_quat_negative_y_180(self):
        """+Y → -Y（反向）：绕 X 轴 180°，四元数 [1,0,0,0]。"""
        q = _direction_to_quat([0, -1, 0])
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-6)

    def test_direction_to_quat_x_axis_rotation(self):
        """+Y → +X（向右）：旋转轴 = y×x = -z，角 = +90°。

        四元数 [x,y,z,w] = [0, 0, -sin45, cos45]。
        """
        q = _direction_to_quat([1, 0, 0])
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        np.testing.assert_allclose(q, [0, 0, -s, c], atol=1e-6)

    def test_direction_to_quat_unit_norm(self):
        """任意方向 → 单位四元数。"""
        for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-1, 2, 3], [0, -1, 0]):
            q = _direction_to_quat(d)
            np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-6)

    # ---------- 方向性基元：cylinder（Z-up 世界坐标 + +Y 模型空间）----------
    def test_cylinder_along_y_no_rotation(self):
        """from→to 沿 +Y（前方）：单位旋转，长度编码到 scale.y。"""
        inst = _make_cylinder_instance([0, 0, 0], [0, 2, 0], radius=0.5, color=[1, 0, 0, 1])
        # 中点 = [0, 1, 0]
        np.testing.assert_allclose(list(inst.position), [0, 1, 0], atol=1e-6)
        # 单位旋转
        np.testing.assert_allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        # scale = [radius, height, radius] = [0.5, 2.0, 0.5]
        np.testing.assert_allclose(list(inst.scale), [0.5, 2.0, 0.5], atol=1e-6)

    def test_cylinder_along_z_vertical(self):
        """from→to 沿 +Z（竖直向上）：+Y→+Z 旋转，长度 2.0 米。

        这是 Z-up 坐标系验证的关键用例：圆柱应竖直朝上。
        """
        inst = _make_cylinder_instance([0, 0, 0], [0, 0, 2], radius=0.5, color=[1, 0, 0, 1])
        # 中点 = [0, 0, 1]（高度 1 米处）
        np.testing.assert_allclose(list(inst.position), [0, 0, 1], atol=1e-6)
        # 旋转非单位（+Y→+Z）
        assert not np.allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        # scale.y = 长度 = 2.0
        np.testing.assert_allclose(list(inst.scale), [0.5, 2.0, 0.5], atol=1e-6)

    def test_cylinder_along_x_horizontal(self):
        """from→to 沿 +X（向右）：+Y→+X 旋转。"""
        inst = _make_cylinder_instance([0, 0, 0], [2, 0, 0], radius=0.3, color=[1, 0, 0, 1])
        np.testing.assert_allclose(list(inst.position), [1, 0, 0], atol=1e-6)
        assert not np.allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(list(inst.scale), [0.3, 2.0, 0.3], atol=1e-6)

    def test_cylinder_zup_coords_passthrough(self):
        """Z-up 坐标透传：from=[1,2,3]（z=3 是高度）应原样使用。"""
        inst = _make_cylinder_instance([1, 2, 3], [1, 2, 5], radius=0.3, color=[1, 0, 0, 1])
        # 中点 = [1, 2, 4]，长度 = 2（沿 +Z）
        np.testing.assert_allclose(list(inst.position), [1, 2, 4], atol=1e-6)
        np.testing.assert_allclose(list(inst.scale), [0.3, 2.0, 0.3], atol=1e-6)

    def test_cylinder_zero_length(self):
        """from==to：长度 0，方向为单位四元数（不抛异常）。"""
        inst = _make_cylinder_instance([1, 2, 3], [1, 2, 3], radius=0.5, color=[1, 0, 0, 1])
        np.testing.assert_allclose(list(inst.position), [1, 2, 3], atol=1e-6)
        np.testing.assert_allclose(list(inst.scale), [0.5, 0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)

    # ---------- 方向性基元：arrow（baked-in 半径补偿）----------
    def test_arrow_along_y_no_rotation(self):
        """from→to 沿 +Y：单位旋转。

        Arrow baked-in shaft r=0.05，shaft_radius=0.1 → scale.x/z = 0.1/0.05 = 2.0。
        scale.y = 长度 = 3.0。
        """
        inst = _make_arrow_instance([0, 0, 0], [0, 3, 0], shaft_radius=0.1, color=[1, 1, 0, 1])
        np.testing.assert_allclose(list(inst.position), [0, 1.5, 0], atol=1e-6)
        np.testing.assert_allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(list(inst.scale), [2.0, 3.0, 2.0], atol=1e-6)

    def test_arrow_along_z_vertical(self):
        """from→to 沿 +Z：+Y→+Z 旋转，长度 3.0 米。"""
        inst = _make_arrow_instance([0, 0, 0], [0, 0, 3], shaft_radius=0.1, color=[1, 1, 0, 1])
        np.testing.assert_allclose(list(inst.position), [0, 0, 1.5], atol=1e-6)
        assert not np.allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(list(inst.scale), [2.0, 3.0, 2.0], atol=1e-6)

    def test_arrow_radius_scaling(self):
        """不同 shaft_radius 对应不同 scale.x/z（baked-in 0.05 补偿）。"""
        inst1 = _make_arrow_instance([0, 0, 0], [0, 1, 0], shaft_radius=0.05, color=[1, 0, 0, 1])
        inst2 = _make_arrow_instance([0, 0, 0], [0, 1, 0], shaft_radius=0.10, color=[1, 0, 0, 1])
        # shaft_radius=0.05 → scale.x/z = 1.0；shaft_radius=0.10 → scale.x/z = 2.0
        np.testing.assert_allclose(list(inst1.scale), [1.0, 1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(list(inst2.scale), [2.0, 1.0, 2.0], atol=1e-6)

    # ---------- Z-up 坐标透传（无轴交换）----------
    def test_make_instance_preserves_zup_coords(self):
        """_make_instance 透传 Z-up 坐标，z=3 是高度而非深度。"""
        inst = _make_instance([1.0, 2.0, 3.0], [0, 0, 0, 1], [1, 1, 1], [1, 0, 0, 1])
        np.testing.assert_allclose(list(inst.position), [1.0, 2.0, 3.0], atol=1e-6)

    def test_sphere_zup_position(self):
        """球体 center=[0,0,1]：高度 1 米（Z-up），而非 y=1（Y-up）。"""
        dd = DebugDraw(stub=None)
        # 离线模式 no-op，仅验证不抛异常（坐标透传在 _make_instance 中验证）
        asyncio.run(dd.draw_sphere([0, 0, 1], 0.2, [1, 0, 0, 1]))

    # ---------- 四元数顺序回归保护 ----------
    def test_quat_order_not_swapped_in_make_instance(self):
        """_make_instance 的 rotation 字段必须保持 [x,y,z,w] 顺序，不得被误转为 [w,x,y,z]。"""
        # AZ::Quaternion 构造顺序为 (x, y, z, w)，proto 也必须用此顺序
        inst = _make_instance([0, 0, 0], [0.1, 0.2, 0.3, 0.4], [1, 1, 1], [1, 0, 0, 1])
        # float32 精度，用 allclose 而非严格相等
        np.testing.assert_allclose(list(inst.rotation), [0.1, 0.2, 0.3, 0.4], atol=1e-6)

    def test_cylinder_rotation_is_xyzw_not_wxyz(self):
        """cylinder 旋转结果必须是 [x,y,z,w]，验证 w 在末位。"""
        # 沿 +Z：四元数 = [sin45, 0, 0, cos45]，w=cos45≈0.707 在末位
        inst = _make_cylinder_instance([0, 0, 0], [0, 0, 1], radius=0.5, color=[1, 0, 0, 1])
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        # [x, y, z, w] = [s, 0, 0, c]，w=c 在 index 3
        np.testing.assert_allclose(list(inst.rotation), [s, 0, 0, c], atol=1e-6)
        # 若误用 [w,x,y,z]，w=c 会在 index 0，此处会失败


# ============================================================
# TTL（immediate 模式 duration）—— 对应设计文档 §3.1 / 实现指南 §3.2.4
# ============================================================
class TestTtlDuration:
    """immediate 模式 TTL（duration_seconds）字段正确性。"""

    def test_draw_batch_default_duration_zero(self):
        """默认 duration=0.0 → 单帧（向后兼容）。"""
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        inst = _make_instance([0, 0, 0], [0, 0, 0, 1], [1, 1, 1], [1, 0, 0, 1])
        asyncio.run(dd.draw_batch(DebugMeshType.SPHERE, [inst]))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.duration_seconds == 0.0

    def test_draw_batch_with_duration(self):
        """duration=0.5 → req.duration_seconds == 0.5。"""
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        inst = _make_instance([0, 0, 0], [0, 0, 0, 1], [1, 1, 1], [1, 0, 0, 1])
        asyncio.run(dd.draw_batch(DebugMeshType.SPHERE, [inst], duration=0.5))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.duration_seconds == 0.5

    def test_draw_sphere_propagates_duration(self):
        """draw_sphere(duration=1.0) → req.duration_seconds == 1.0。"""
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.draw_sphere([0, 0, 0], 1.0, [1, 0, 0, 1], duration=1.0))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.duration_seconds == 1.0

    def test_draw_box_propagates_duration(self):
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.draw_box([0, 0, 0], [1, 1, 1], [1, 0, 0, 1], duration=2.0))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.duration_seconds == 2.0

    def test_draw_cylinder_propagates_duration(self):
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.draw_cylinder([0, 0, 0], [0, 0, 1], 0.1, [1, 0, 0, 1], duration=0.3))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.duration_seconds == pytest.approx(0.3)

    def test_draw_arrow_propagates_duration(self):
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.draw_arrow([0, 0, 0], [0, 0, 1], 0.05, [1, 1, 0, 1], duration=0.1))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.duration_seconds == pytest.approx(0.1)

    def test_draw_quad_propagates_duration(self):
        stub = MagicMock()
        stub.DrawDebugMeshBatch = AsyncMock()
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.draw_quad([0, 0, 0], [1, 1, 1], [1, 0, 0, 1], duration=0.05))
        req = stub.DrawDebugMeshBatch.await_args[0][0]
        assert req.duration_seconds == pytest.approx(0.05)


# ============================================================
# Keepalive（retained 模式保活心跳）—— 对应设计文档 §3.2 / 实现指南 §3.3
# ============================================================
class TestKeepalive:
    """retained 模式 keepalive_seconds 与 keepalive_objects 正确性。"""

    def test_create_objects_default_keepalive(self):
        """create_objects 默认 keepalive=1.0s。"""
        stub = MagicMock()
        resp = mjc_message_pb2.CreateDebugMeshObjectsResponse()
        stub.CreateDebugMeshObjects = AsyncMock(return_value=resp)
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.create_objects(DebugMeshType.BOX, []))
        req = stub.CreateDebugMeshObjects.await_args[0][0]
        assert req.keepalive_seconds == 1.0

    def test_create_objects_custom_keepalive(self):
        """create_objects(keepalive=5.0) → req.keepalive_seconds == 5.0。"""
        stub = MagicMock()
        resp = mjc_message_pb2.CreateDebugMeshObjectsResponse()
        stub.CreateDebugMeshObjects = AsyncMock(return_value=resp)
        dd = DebugDraw(stub=stub)
        asyncio.run(dd.create_objects(DebugMeshType.BOX, [], keepalive=5.0))
        req = stub.CreateDebugMeshObjects.await_args[0][0]
        assert req.keepalive_seconds == 5.0

    def test_keepalive_objects_calls_stub(self):
        """keepalive_objects 调用 KeepAliveDebugMeshObjects RPC。"""
        stub = MagicMock()
        resp = mjc_message_pb2.KeepAliveDebugMeshObjectsResponse()
        resp.success = True
        alive = resp.alive_handles.add()
        alive.index = 0
        alive.generation = 1
        expired = resp.expired_handles.add()
        expired.index = 2
        expired.generation = 1
        stub.KeepAliveDebugMeshObjects = AsyncMock(return_value=resp)

        dd = DebugDraw(stub=stub)
        handles = [
            {"index": 0, "generation": 1, "valid": True},
            {"index": 2, "generation": 1, "valid": True},
        ]
        result = asyncio.run(dd.keepalive_objects(handles))
        stub.KeepAliveDebugMeshObjects.assert_awaited_once()
        req = stub.KeepAliveDebugMeshObjects.await_args[0][0]
        assert len(req.handles) == 2
        assert req.handles[0].index == 0 and req.handles[0].generation == 1
        assert req.handles[1].index == 2 and req.handles[1].generation == 1
        # 返回结构
        assert len(result["alive"]) == 1
        assert result["alive"][0] == {"index": 0, "generation": 1, "valid": True}
        assert len(result["expired"]) == 1
        assert result["expired"][0] == {"index": 2, "generation": 1, "valid": True}

    def test_keepalive_objects_empty_handles(self):
        """空 handle 列表也能调用（不抛异常）。"""
        stub = MagicMock()
        resp = mjc_message_pb2.KeepAliveDebugMeshObjectsResponse()
        resp.success = True
        stub.KeepAliveDebugMeshObjects = AsyncMock(return_value=resp)
        dd = DebugDraw(stub=stub)
        result = asyncio.run(dd.keepalive_objects([]))
        assert result == {"alive": [], "expired": []}

    def test_keepalive_objects_offline_returns_empty(self):
        """离线模式返回 {'alive': [], 'expired': []}。"""
        dd = DebugDraw(stub=None)
        result = asyncio.run(dd.keepalive_objects([{"index": 0, "generation": 1}]))
        assert result == {"alive": [], "expired": []}

    def test_keepalive_objects_all_expired(self):
        """全部过期的场景：alive 为空，expired 含所有 handle。"""
        stub = MagicMock()
        resp = mjc_message_pb2.KeepAliveDebugMeshObjectsResponse()
        resp.success = True
        e1 = resp.expired_handles.add()
        e1.index = 0
        e1.generation = 1
        e2 = resp.expired_handles.add()
        e2.index = 1
        e2.generation = 2
        stub.KeepAliveDebugMeshObjects = AsyncMock(return_value=resp)
        dd = DebugDraw(stub=stub)
        handles = [
            {"index": 0, "generation": 1, "valid": True},
            {"index": 1, "generation": 2, "valid": True},
        ]
        result = asyncio.run(dd.keepalive_objects(handles))
        assert result["alive"] == []
        assert len(result["expired"]) == 2
