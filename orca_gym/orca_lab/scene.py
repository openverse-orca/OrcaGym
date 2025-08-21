import grpc
import orca_gym.orca_lab.protos.edit_service_pb2_grpc as edit_service_pb2_grpc
import orca_gym.orca_lab.protos.edit_service_pb2 as edit_service_pb2
import orca_gym.protos.mjc_message_pb2_grpc as mjc_message_pb2_grpc
import orca_gym.protos.mjc_message_pb2 as mjc_message_pb2

from orca_gym.orca_lab.path import Path
import asyncio
import numpy as np

from typing import Tuple, Dict
from datetime import datetime
import time


class Actor:
    """
    A class to represent an actor in the ORCA Gym environment.
    """

    def __init__(
        self,
        name: str,
        spawnable_name: str,
        position: np.ndarray,
        rotation: np.ndarray,
        scale: float,
        path: Path = Path(),
    ):
        self.name = name
        self.spawnable_name = spawnable_name
        self.position = position
        self.rotation = rotation
        self.scale = float(scale)
        self.path = path
        self._check_actor()

    def _check_actor(self):
        if self.name is None or self.name == "":
            raise ValueError("Actor name cannot be None or empty.")
        if self.spawnable_name is None or self.spawnable_name == "":
            raise ValueError("Actor spawnable name cannot be None or empty.")
        if self.position is None or len(self.position) != 3:
            raise ValueError("Actor position must be a 3D vector.")
        if self.rotation is None or len(self.rotation) != 4:
            raise ValueError("Actor rotation must be a quaternion (4D vector).")
        if self.scale is None or not isinstance(self.scale, float):
            raise ValueError("Actor scale must be a float.")


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
        self.root_actor = Actor(
            name="root",
            spawnable_name="root",
            position=np.array([0, 0, 0]),
            rotation=np.array([1, 0, 0, 0]),
            scale=1.0,
        )
        self.actors: Dict[Path, Actor] = {}
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

            [pos, quat, scale] = await self.get_pending_actor_transform_async(
                actor_path, True
            )

            await self.set_actor_transform_async(actor_path, pos, quat, scale, True)

            actor = self.actors[actor_path]
            actor.position = pos
            actor.rotation = quat
            actor.scale = scale

        world_transform_change = "world_transform_change:"
        if op.startswith(world_transform_change):
            actor_path = op[len(world_transform_change) :]

            if not actor_path in self.actors:
                raise Exception(f"actor not exist")

            [pos, quat, scale] = await self.get_pending_actor_transform_async(
                actor_path, False
            )

            await self.set_actor_transform_async(actor_path, pos, quat, scale, False)

            actor = self.actors[actor_path]
            actor.position = pos
            actor.rotation = quat
            actor.scale = scale

    async def get_pending_actor_transform_async(self, path: Path, local: bool):
        space = edit_service_pb2.Space.Local if local else edit_service_pb2.Space.World

        request = edit_service_pb2.GetPendingActorTransformRequest(
            actor_path=path.string(),
            space=space,
        )

        response = await self.edit_stub.GetPendingActorTransform(request)
        self._check_response(response)

        transform = response.transform
        return transform.pos, transform.quat, transform.scale

    async def add_actor_async(self, actor: Actor, parent_path: Path):
        if not parent_path.is_valid():
            raise Exception("Invalid path")

        if not parent_path in self.actors:
            raise Exception("parent not exist.")

        path = parent_path / actor.name

        transform = edit_service_pb2.Transform(
            pos=actor.position,
            quat=actor.rotation,
            scale=actor.scale,
        )

        request = edit_service_pb2.AddActorRequest(
            actor_name=actor.name,
            spawnable_name=actor.spawnable_name,
            parent_actor_path=parent_path.string(),
            transform=transform,
            space=edit_service_pb2.Space.Local,
        )

        response = await self.edit_stub.AddActor(request)
        self._check_response(response)

        actor.path = path
        self.actors[path] = actor

    def add_actor(self, actor: Actor, parent_path: Path):
        self.loop.run_until_complete(self.add_actor_async(actor, parent_path))

    async def set_actor_transform_async(
        self, path: Path, pos, quat, scale, local: bool
    ):

        transform = edit_service_pb2.Transform(pos=pos, quat=quat, scale=scale)
        space = edit_service_pb2.Space.Local if local else edit_service_pb2.Space.World

        request = edit_service_pb2.SetActorTransformRequest(
            actor_path=path.string(),
            transform=transform,
            space=space,
        )

        response = await self.edit_stub.SetActorTransform(request)
        self._check_response(response)

    def set_actor_transform(self, path: str, pos, rot, scale, local: bool):
        self.loop.run_until_complete(
            self.set_actor_transform_async(path, pos, rot, scale, local)
        )

    async def publish_scene_async(self):
        request = mjc_message_pb2.PublishSceneRequest()
        response = await self.sim_stub.PublishScene(request)
        if response.status != mjc_message_pb2.PublishSceneResponse.SUCCESS:
            print("Publish scene failed: ", response.error_message)
            raise Exception("Publish scene failed.")

    def publish_scene(self):
        self.loop.run_until_complete(self.publish_scene_async())

    async def get_sync_from_mujoco_to_scene_async(self) -> bool:
        request = edit_service_pb2.GetSyncFromMujocoToSceneRequest()
        response = await self.edit_stub.GetSyncFromMujocoToScene(request)
        self._check_response(response)
        return response.value

    def get_sync_from_mujoco_to_scene(self) -> bool:
        return self.loop.run_until_complete(self.get_sync_from_mujoco_to_scene_async())

    async def set_sync_from_mujoco_to_scene_async(self, value: bool):
        request = edit_service_pb2.SetSyncFromMujocoToSceneRequest(value=value)
        response = await self.edit_stub.SetSyncFromMujocoToScene(request)
        self._check_response(response)
        return response

    def set_sync_from_mujoco_to_scene(self, value: bool):
        return self.loop.run_until_complete(
            self.set_sync_from_mujoco_to_scene_async(value)
        )
