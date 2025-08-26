import grpc
from orca_gym.orca_lab.math import Transform
import orca_gym.orca_lab.protos.edit_service_pb2_grpc as edit_service_pb2_grpc
import orca_gym.orca_lab.protos.edit_service_pb2 as edit_service_pb2
import orca_gym.protos.mjc_message_pb2_grpc as mjc_message_pb2_grpc
import orca_gym.protos.mjc_message_pb2 as mjc_message_pb2

from orca_gym.orca_lab.path import Path
from orca_gym.orca_lab.actor import AssetActor, BaseActor, GroupActor
import asyncio

from typing import Tuple, Dict


Success = edit_service_pb2.StatusCode.Success
Error = edit_service_pb2.StatusCode.Error


# 由于Qt是异步的，所以这里只提供异步接口。
class OrcaLabScene:

    def __init__(self, edit_grpc_addr: str, sim_grpc_addr: str):
        self.edit_grpc_addr = edit_grpc_addr
        self.sim_grpc_addr = sim_grpc_addr

        # 作为根节点，不可见， 路径是"/"。下面挂着所有的顶层Actor。
        self.root_actor = GroupActor(name="root", parent=None)
        self.actors: Dict[Path, BaseActor] = {}
        self.actors[Path.root_path()] = self.root_actor

    async def init_grpc(self):
        options = [
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
        ]
        self.edit_channel = grpc.aio.insecure_channel(
            self.edit_grpc_addr,
            options=options,
        )
        self.sim_channel = grpc.aio.insecure_channel(
            self.sim_grpc_addr,
            options=options,
        )
        self.edit_stub = edit_service_pb2_grpc.GrpcServiceStub(self.edit_channel)
        self.sim_stub = mjc_message_pb2_grpc.GrpcServiceStub(self.sim_channel)

        self.timeout = 3

        self.c = 0

        if not await self.aloha():
            raise Exception("Failed to connect to server.")

        self._query_pending_operation_lock = asyncio.Lock()
        self._query_pending_operation_running = False
        await self._start_query_pending_operation_loop()

    async def destroy_grpc(self):
        await self._stop_query_pending_operation_loop()

        if self.edit_channel:
            await self.edit_channel.close()
        self.edit_stub = None
        self.edit_channel = None

        if self.sim_channel:
            await self.sim_channel.close()
        self.sim_stub = None
        self.sim_channel = None

    def _create_transform_message(self, transform: Transform):
        msg = edit_service_pb2.Transform(
            pos=transform.position,
            quat=transform.rotation,
            scale=1.0,
        )
        return msg

    def _get_transform_from_message(self, msg) -> Transform:
        transform = Transform()
        transform.position = msg.pos
        transform.rotation = msg.quat
        return transform

    def _check_response(self, response):
        if response.status_code != Success:
            raise Exception(f"Request failed. {response.error_message}")

    async def aloha(self) -> bool:
        request = edit_service_pb2.AlohaRequest(value=1)
        response = await self.edit_stub.Aloha(request)
        self._check_response(response)
        return response.value == 2

    async def _start_query_pending_operation_loop(self):
        async with self._query_pending_operation_lock:
            if self._query_pending_operation_running:
                return
            asyncio.create_task(self._query_pending_operation_loop())
            self._query_pending_operation_running = True

    async def _stop_query_pending_operation_loop(self):
        async with self._query_pending_operation_lock:
            self._query_pending_operation_running = False

    async def _query_pending_operation_loop(self):
        async with self._query_pending_operation_lock:
            if not self._query_pending_operation_running:
                return

            # print(f"query {self.c}")
            # self.c += 1
            request = edit_service_pb2.GetPendingOperationsRequest()
            response = await self.edit_stub.GetPendingOperations(request)
            self._check_response(response)

            operations = response.operations
            for op in operations:
                await self._process_pending_operation(op)

        # self.pump_qt_loop()
        frequency = 10  # Hz
        asyncio.sleep(1 / frequency)
        asyncio.create_task(self._query_pending_operation_loop())

    async def _process_pending_operation(self, op: str):
        local_transform_change = "local_transform_change:"
        if op.startswith(local_transform_change):
            actor_path = Path(op[len(local_transform_change) :])

            if not actor_path in self.actors:
                raise Exception(f"actor not exist")

            transform = await self.get_pending_actor_transform(actor_path, True)

            await self.set_actor_transform(actor_path, transform, True)

            actor = self.actors[actor_path]
            actor.transform = transform

        world_transform_change = "world_transform_change:"
        if op.startswith(world_transform_change):
            actor_path = op[len(world_transform_change) :]

            if not actor_path in self.actors:
                raise Exception(f"actor not exist")

            transform = await self.get_pending_actor_transform(actor_path, False)

            await self.set_actor_transform(actor_path, transform, False)

            actor = self.actors[actor_path]
            actor.transform = transform

    async def get_pending_actor_transform(self, path: Path, local: bool) -> Transform:
        space = edit_service_pb2.Space.Local if local else edit_service_pb2.Space.World

        request = edit_service_pb2.GetPendingActorTransformRequest(
            actor_path=path.string(),
            space=space,
        )

        response = await self.edit_stub.GetPendingActorTransform(request)
        self._check_response(response)

        transform = response.transform
        return self._get_transform_from_message(transform)

    async def add_actor(self, actor: BaseActor, parent_path: Path):
        if not parent_path.is_valid():
            raise Exception("Invalid path")

        if not parent_path in self.actors:
            raise Exception("parent not exist.")

        path = parent_path / actor.name

        transform_msg = self._create_transform_message(actor.transform)
        request = edit_service_pb2.AddActorRequest(
            actor_name=actor.name,
            spawnable_name=actor.spawnable_name,
            parent_actor_path=parent_path.string(),
            transform=transform_msg,
            space=edit_service_pb2.Space.Local,
        )

        response = await self.edit_stub.AddActor(request)
        self._check_response(response)

        actor.parent = self.actors[parent_path]
        self.actors[path] = actor

    async def set_actor_transform(self, path: Path, transform: Transform, local: bool):
        transform_msg = self._create_transform_message(transform)
        space = edit_service_pb2.Space.Local if local else edit_service_pb2.Space.World

        request = edit_service_pb2.SetActorTransformRequest(
            actor_path=path.string(),
            transform=transform_msg,
            space=space,
        )

        response = await self.edit_stub.SetActorTransform(request, timeout=self.timeout)
        self._check_response(response)

    async def publish_scene(self):
        print(f"publish_scene")
        request = mjc_message_pb2.PublishSceneRequest()
        response = await self.sim_stub.PublishScene(request)
        if response.status != mjc_message_pb2.PublishSceneResponse.SUCCESS:
            print("Publish scene failed: ", response.error_message)
            raise Exception("Publish scene failed.")
        print("done")

    async def forward_scene(self):
        print(f"forward_scene")
        request = mjc_message_pb2.MJ_ForwardRequest()
        response = await self.sim_stub.MJ_Forward(request)
        print("done")

    async def get_sync_from_mujoco_to_scene(self) -> bool:
        request = edit_service_pb2.GetSyncFromMujocoToSceneRequest()
        response = await self.edit_stub.GetSyncFromMujocoToScene(request)
        self._check_response(response)
        return response.value

    async def set_sync_from_mujoco_to_scene(self, value: bool):
        print(f"set_sync_from_mujoco_to_scene {value}")
        request = edit_service_pb2.SetSyncFromMujocoToSceneRequest(value=value)
        response = await self.edit_stub.SetSyncFromMujocoToScene(request)
        self._check_response(response)
        print("done")

    async def clear_scene(self):
        request = edit_service_pb2.ClearSceneRequest()
        response = await self.edit_stub.ClearScene(request)
        self._check_response(response)

    async def get_actor_assets(self):
        request = edit_service_pb2.GetActorAssetsRequest()
        response = await self.edit_stub.GetActorAssets(request)
        self._check_response(response)
        return response.actor_asset_names

    async def save_body_transform(self):
        request = edit_service_pb2.SaveBodyTransformRequest()
        response = await self.edit_stub.SaveBodyTransform(request)
        self._check_response(response)

    async def restore_body_transform(self):
        request = edit_service_pb2.RestoreBodyTransformRequest()
        response = await self.edit_stub.RestoreBodyTransform(request)
        self._check_response(response)
