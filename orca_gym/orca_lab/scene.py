import grpc
import orca_gym.orca_lab.protos.edit_service_pb2 as edit_service_pb2
from orca_gym.orca_lab.protos.edit_service_pb2_grpc import GrpcServiceStub
from orca_gym.orca_lab.path import Path
import asyncio
import numpy as np

from typing import Tuple, Dict


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


class LightInfo:
    """
    A class to represent light information in the ORCA Gym environment.
    """

    def __init__(
        self,
        color: np.ndarray,
        intensity: float,
    ):
        self.color = color
        self.intensity = intensity
        self._check_light_info()

    def _check_light_info(self):
        if self.color is None or len(self.color) != 3:
            raise ValueError("Light color must be a 3D vector.")
        if self.intensity is None or not isinstance(self.intensity, float):
            raise ValueError("Light intensity must be a float.")


class CameraSensorInfo:
    """
    A class to represent camera sensor information in the ORCA Gym environment.
    """

    def __init__(
        self,
        capture_rgb: bool,
        capture_depth: bool,
        save_mp4_file: bool,
        use_dds: bool,
    ):
        self.capture_rgb = capture_rgb
        self.capture_depth = capture_depth
        self.save_mp4_file = save_mp4_file
        self.use_dds = use_dds
        self._check_camera_sensor_info()

    def _check_camera_sensor_info(self):
        if self.capture_rgb is None or not isinstance(self.capture_rgb, bool):
            raise ValueError("Capture RGB must be a boolean.")
        if self.capture_depth is None or not isinstance(self.capture_depth, bool):
            raise ValueError("Capture depth must be a boolean.")
        if self.save_mp4_file is None or not isinstance(self.save_mp4_file, bool):
            raise ValueError("Save MP4 file must be a boolean.")
        if self.use_dds is None or not isinstance(self.use_dds, bool):
            raise ValueError("Use DDS must be a boolean.")


class MaterialInfo:
    """
    A class to represent material information in the ORCA Gym environment.
    """

    def __init__(
        self,
        base_color: np.ndarray,
    ):
        self.base_color = base_color
        self._check_material_info()

    def _check_material_info(self):
        if self.base_color is None or len(self.base_color) != 4:
            raise ValueError("Base color must be a 4D vector.")


Success = edit_service_pb2.StatusCode.Success
Error = edit_service_pb2.StatusCode.Error


class OrcaLabScene:
    """
    A class to represent a scene in the ORCA lab environment.
    """

    def __init__(
        self,
        grpc_addr: str,
    ):
        """
        Initialize the ORCA lab scene.

        Args:
            scene_id (int): The ID of the scene.
            scene_name (str): The name of the scene.
        """
        self.grpc_addr = grpc_addr
        self.loop = asyncio.get_event_loop()
        self.actors: Dict[Path, Actor] = {}
        self.initialize_grpc()

        self.root_actor = Actor(
            name="root",
            spawnable_name="root",
            position=np.array([0, 0, 0]),
            rotation=np.array([1, 0, 0, 0]),
            scale=1.0,
        )
        self.actors[Path.root_path()] = self.root_actor

    def initialize_grpc(self):
        self.channel = grpc.aio.insecure_channel(
            self.grpc_addr,
            options=[
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ],
        )
        self.stub = GrpcServiceStub(self.channel)
        if not self.aloha():
            raise Exception("Failed to connect to server.")

        self.loop.create_task(self._query_pending_operation_loop())

        self.running = True

    async def close_grpc_async(self):
        self.running = False
        if self.channel:
            await self.channel.close()
        self.stub = None
        self.channel = None

    def close_grpc(self):
        self.loop.run_until_complete(self.close_grpc_async())

    def start_simulation():
        pass

    def stop_simulation():
        pass

    def pause_simulation():
        pass

    def resume_simulation():
        pass

    async def aloha_async(self) -> bool:
        request = edit_service_pb2.AlohaRequest(value=1)
        response = await self.stub.Aloha(request)
        return response.value == 2

    def aloha(self) -> bool:
        return self.loop.run_until_complete(self.aloha_async())

    async def _query_pending_operation_loop(self):
        while True:
            if self.running:
                request = edit_service_pb2.GetPendingOperationsRequest()
                response = await self.stub.GetPendingOperations(request)

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

        response = await self.stub.GetPendingActorTransform(request)
        if response.status_code != Success:
            print("GetPendingActorTransform failed")
            raise Exception("GetPendingActorTransform failed.")

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

        response = await self.stub.AddActor(request)
        if response.status_code != Success:
            print("Add actor failed: ", response.error_message)
            raise Exception("Add actor failed.")

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

        response = await self.stub.SetActorTransform(request)
        if response.status_code != Success:
            print("Set actor transform failed: ", response.error_message)
            raise Exception("Set actor transform failed.")

    def set_actor_transform(self, path: str, pos, rot, scale, local: bool):
        self.loop.run_until_complete(
            self.set_actor_transform_async(path, pos, rot, scale, local)
        )
