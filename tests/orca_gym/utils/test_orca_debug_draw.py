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
# 几何计算：方向→四元数
# ============================================================
class TestDirectionToQuat:
    def test_identity_direction_returns_identity_quat(self):
        q = _direction_to_quat([0, 0, 1])
        np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-6)

    def test_zero_direction_returns_identity_quat(self):
        q = _direction_to_quat([0, 0, 0])
        np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-6)

    def test_negative_z_returns_180_around_x(self):
        """+Z → -Z 应绕 X 轴转 180°，四元数 [1,0,0,0]。"""
        q = _direction_to_quat([0, 0, -1])
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-6)

    def test_x_axis_rotation(self):
        """+Z → +X：axis = z×d = +Y，angle = +90°，四元数 [0, sin45, 0, cos45]。"""
        q = _direction_to_quat([1, 0, 0])
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        np.testing.assert_allclose(q, [0, s, 0, c], atol=1e-6)

    def test_unit_vector_input_unchanged_by_magnitude(self):
        """方向相同、长度不同应得到相同四元数。"""
        q1 = _direction_to_quat([0, 0, 2])
        q2 = _direction_to_quat([0, 0, 5])
        np.testing.assert_allclose(q1, q2, atol=1e-6)

    def test_quaternion_is_unit(self):
        for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-1, 2, 3]):
            q = _direction_to_quat(d)
            np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-6)


# ============================================================
# 几何计算：cylinder / arrow 实例构造
# ============================================================
class TestCylinderArrowGeometry:
    def test_cylinder_along_z(self):
        inst = _make_cylinder_instance([0, 0, 0], [0, 0, 2], radius=0.5, color=[1, 0, 0, 1])
        # 中点
        assert list(inst.position) == [0, 0, 1]
        # 无旋转（沿 +Z）
        np.testing.assert_allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)
        # scale = [radius, radius, height]
        np.testing.assert_allclose(list(inst.scale), [0.5, 0.5, 2.0], atol=1e-6)

    def test_cylinder_along_x(self):
        inst = _make_cylinder_instance([0, 0, 0], [2, 0, 0], radius=0.3, color=[1, 0, 0, 1])
        assert list(inst.position) == [1, 0, 0]
        np.testing.assert_allclose(list(inst.scale), [0.3, 0.3, 2.0], atol=1e-6)

    def test_cylinder_zero_length(self):
        """from==to 时高度=0，方向为单位四元数（不抛异常）。"""
        inst = _make_cylinder_instance([1, 2, 3], [1, 2, 3], radius=0.5, color=[1, 0, 0, 1])
        assert list(inst.position) == [1, 2, 3]
        np.testing.assert_allclose(list(inst.scale), [0.5, 0.5, 0.0], atol=1e-6)
        np.testing.assert_allclose(list(inst.rotation), [0, 0, 0, 1], atol=1e-6)

    def test_arrow_along_z(self):
        inst = _make_arrow_instance([0, 0, 0], [0, 0, 3], shaft_radius=0.1, color=[1, 1, 0, 1])
        assert list(inst.position) == [0, 0, 1.5]
        np.testing.assert_allclose(list(inst.scale), [0.1, 0.1, 3.0], atol=1e-6)

    def test_cylinder_flags_propagated(self):
        inst = _make_cylinder_instance(
            [0, 0, 0], [0, 0, 1], radius=0.5, color=[1, 0, 0, 1], flags=InstanceFlags.EDGE_HIGHLIGHT
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
