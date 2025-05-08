import grpc
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
import orca_gym.protos.mjc_message_pb2 as mjc_message_pb2
import asyncio
import numpy as np

class Actor:
    """
    A class to represent an actor in the ORCA Gym environment.
    """

    def __init__(self,
                 name: str,
                 spawnable_name: str,
                 position: np.ndarray,
                 rotation: np.ndarray,
                 scale: float,):
        self.name = name
        self.spawnable_name = spawnable_name
        self.position = position
        self.rotation = rotation
        self.scale = float(scale)
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




class OrcaGymScene:
    """
    A class to represent a scene in the ORCA Gym environment.
    """

    def __init__(self,
                 grpc_addr: str,):
        """
        Initialize the ORCA Gym scene.

        Args:
            scene_id (int): The ID of the scene.
            scene_name (str): The name of the scene.
        """
        self.grpc_addr = grpc_addr
        self.loop = asyncio.get_event_loop()
        self.initialize_grpc()

    def initialize_grpc(self):
        self.channel = grpc.aio.insecure_channel(
            self.grpc_addr,
            options=[
                ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ]
        )
        self.stub = GrpcServiceStub(self.channel)

    async def _close_grpc(self):
        if self.channel:
            await self.channel.close()

    def close(self):
        self.loop.run_until_complete(self._close_grpc())

    async def _publish_scene(self):
        request = mjc_message_pb2.PublishSceneRequest()
        response = await self.stub.PublishScene(request)
        if response.status != mjc_message_pb2.PublishSceneResponse.SUCCESS:
            raise Exception("Publish scene failed.")

    def publish_scene(self):
        self.loop.run_until_complete(self._publish_scene())

    async def _add_actor(self, actor : Actor):
        request = mjc_message_pb2.AddActorRequest(
            name = actor.name,
            spawnable_name = actor.spawnable_name,
            pos = actor.position,
            quat = actor.rotation,
            scale = actor.scale)
        
        response = await self.stub.AddActor(request)
        if response.status != mjc_message_pb2.AddActorResponse.SUCCESS:
            raise Exception("Add actor failed.")
        
    def add_actor(self, actor: Actor):
        self.loop.run_until_complete(self._add_actor(actor))