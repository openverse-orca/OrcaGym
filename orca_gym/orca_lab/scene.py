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
import time


Success = edit_service_pb2.StatusCode.Success
Error = edit_service_pb2.StatusCode.Error


class OrcaLabScene:

    def __init__(self, edit_grpc_addr: str, sim_grpc_addr: str):
        self.edit_grpc_addr = edit_grpc_addr
        self.sim_grpc_addr = sim_grpc_addr

        self.loop = asyncio.get_event_loop()

        self.last_query_time = None
        self.query_frequency = 30

        # 作为根节点，不可见， 路径是"/"。下面挂着所有的顶层Actor。
        self.root_actor = GroupActor(name="root", parent=None)
        self.actors: Dict[Path, BaseActor] = {}
        self.actors[Path.root_path()] = self.root_actor

        self.initialize_grpc()

    def initialize_grpc(self):
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

        if not self.aloha():
            raise Exception("Failed to connect to server.")

        self.loop.create_task(self._query_pending_operation_loop())

        self.running = True

    async def close_grpc_async(self):
        self.running = False
        if self.edit_channel:
            await self.edit_channel.close()
        self.edit_stub = None
        self.edit_channel = None

    def close_grpc(self):
        self.loop.run_until_complete(self.close_grpc_async())

    def start_sim(self):
        self.publish_scene()

    def create_transform_message(self, transform: Transform):
        msg = edit_service_pb2.Transform(
            pos=transform.position,
            quat=transform.rotation,
            scale=1.0,
        )
        return msg

    def get_transform_from_message(self, msg) -> Transform:
        transform = Transform()
        transform.position = msg.pos
        transform.rotation = msg.quat
        return transform

    def _check_response(self, response):
        if response.status_code != Success:
            raise Exception(f"Request failed. {response.error_message}")

    async def aloha_async(self) -> bool:
        request = edit_service_pb2.AlohaRequest(value=1)
        response = await self.edit_stub.Aloha(request)
        self._check_response(response)
        return response.value == 2

    def aloha(self) -> bool:
        return self.loop.run_until_complete(self.aloha_async())

    async def _query_pending_operation_loop(self):
        while True:
            if self.running:
                if self.last_query_time is not None:
                    now = time.time()
                    delta = 1 / self.query_frequency
                    if (now - self.last_query_time) < delta:
                        continue

                # print("query")
                request = edit_service_pb2.GetPendingOperationsRequest()
                response = await self.edit_stub.GetPendingOperations(request)
                self._check_response(response)

                self.last_query_time = time.time()

                operations = response.operations
                for op in operations:
                    await self._process_pending_operation(op)

    async def _process_pending_operation(self, op: str):
        local_transform_change = "local_transform_change:"
        if op.startswith(local_transform_change):
            actor_path = Path(op[len(local_transform_change) :])

            if not actor_path in self.actors:
                raise Exception(f"actor not exist")

            transform = await self.get_pending_actor_transform_async(actor_path, True)

            await self.set_actor_transform_async(actor_path, transform, True)

            actor = self.actors[actor_path]
            actor.transform = transform

        world_transform_change = "world_transform_change:"
        if op.startswith(world_transform_change):
            actor_path = op[len(world_transform_change) :]

            if not actor_path in self.actors:
                raise Exception(f"actor not exist")

            transform = await self.get_pending_actor_transform_async(actor_path, False)

            await self.set_actor_transform_async(actor_path, transform, False)

            actor = self.actors[actor_path]
            actor.transform = transform

    async def get_pending_actor_transform_async(
        self, path: Path, local: bool
    ) -> Transform:
        space = edit_service_pb2.Space.Local if local else edit_service_pb2.Space.World

        request = edit_service_pb2.GetPendingActorTransformRequest(
            actor_path=path.string(),
            space=space,
        )

        response = await self.edit_stub.GetPendingActorTransform(request)
        self._check_response(response)

        transform = response.transform
        return self.get_transform_from_message(transform)

    async def add_actor_async(self, actor: BaseActor, parent_path: Path):
        if not parent_path.is_valid():
            raise Exception("Invalid path")

        if not parent_path in self.actors:
            raise Exception("parent not exist.")

        path = parent_path / actor.name

        transform_msg = self.create_transform_message(actor.transform)
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

    def add_actor(self, actor: BaseActor, parent_path: Path):
        self.loop.run_until_complete(self.add_actor_async(actor, parent_path))

    async def set_actor_transform_async(
        self, path: Path, transform: Transform, local: bool
    ):

        transform_msg = self.create_transform_message(transform)
        space = edit_service_pb2.Space.Local if local else edit_service_pb2.Space.World

        request = edit_service_pb2.SetActorTransformRequest(
            actor_path=path.string(),
            transform=transform_msg,
            space=space,
        )

        response = await self.edit_stub.SetActorTransform(request, timeout=self.timeout)
        self._check_response(response)

    def set_actor_transform(self, path: str, transform: Transform, local: bool):
        self.loop.run_until_complete(
            self.set_actor_transform_async(path, transform, local)
        )

    async def publish_scene_async(self):
        print(f"publish_scene_async")
        request = mjc_message_pb2.PublishSceneRequest()
        response = await self.sim_stub.PublishScene(request)
        if response.status != mjc_message_pb2.PublishSceneResponse.SUCCESS:
            print("Publish scene failed: ", response.error_message)
            raise Exception("Publish scene failed.")
        print("done")

    def publish_scene(self):
        self.publish_scene_async()

    async def get_sync_from_mujoco_to_scene_async(self) -> bool:
        request = edit_service_pb2.GetSyncFromMujocoToSceneRequest()
        response = await self.edit_stub.GetSyncFromMujocoToScene(request)
        self._check_response(response)
        return response.value

    def get_sync_from_mujoco_to_scene(self) -> bool:
        return self.loop.run_until_complete(self.get_sync_from_mujoco_to_scene_async())

    async def set_sync_from_mujoco_to_scene_async(self, value: bool):
        print(f"set_sync_from_mujoco_to_scene_async {value}")
        request = edit_service_pb2.SetSyncFromMujocoToSceneRequest(value=value)
        response = await self.edit_stub.SetSyncFromMujocoToScene(request)
        self._check_response(response)
        print("done")
        return response

    def set_sync_from_mujoco_to_scene(self, value: bool):
        return self.loop.run_until_complete(
            self.set_sync_from_mujoco_to_scene_async(value)
        )

    async def clear_scene_async(self):
        request = edit_service_pb2.ClearSceneRequest()
        response = await self.edit_stub.ClearScene(request)
        self._check_response(response)
        return response

    def clear_scene(self):
        return self.loop.run_until_complete(self.clear_scene_async())

    async def get_actor_assets_async(self):
        request = edit_service_pb2.GetActorAssetsRequest()
        response = await self.edit_stub.GetActorAssets(request)
        self._check_response(response)
        return response

    def get_actor_assets(self):
        return self.loop.run_until_complete(self.get_actor_assets_async())
