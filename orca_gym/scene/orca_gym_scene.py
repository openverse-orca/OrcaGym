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
                 scale: float,
                 base_color: np.ndarray,):
        self.name = name
        self.spawnable_name = spawnable_name
        self.position = position
        self.rotation = rotation
        self.scale = float(scale)
        self.base_color = base_color
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

    def __init__(self,
                 color: np.ndarray,
                 intensity: float,):
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

    def __init__(self,
                 capture_rgb : bool,
                 capture_depth : bool,
                 save_mp4_file : bool,
                 use_dds : bool,):
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
        self.lock = asyncio.Lock()  # 新增一个异步锁
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
        async with self.lock:  # 加锁保证串行
            request = mjc_message_pb2.PublishSceneRequest()
            response = await self.stub.PublishScene(request)
            if response.status != mjc_message_pb2.PublishSceneResponse.SUCCESS:
                print("Publish scene failed: ", response.error_message)
                raise Exception("Publish scene failed.")

    def publish_scene(self):
        self.loop.run_until_complete(self._publish_scene())

    async def _add_actor(self, actor : Actor):
        async with self.lock:  # 加锁保证串行
            request = mjc_message_pb2.AddActorRequest(
                name = actor.name,
                spawnable_name = actor.spawnable_name,
                pos = actor.position,
                quat = actor.rotation,
                scale = actor.scale,
                base_color = actor.base_color,)
        
        response = await self.stub.AddActor(request)
        if response.status != mjc_message_pb2.AddActorResponse.SUCCESS:
            print("Add actor failed: ", response.error_message)
            raise Exception("Add actor failed.")
        
    def add_actor(self, actor: Actor):
        self.loop.run_until_complete(self._add_actor(actor))

    async def _set_light_info(self, actor_name : str, light_info: LightInfo):
        async with self.lock:  # 加锁保证串行
            request = mjc_message_pb2.SetLightInfoRequest(
                actor_name = actor_name,
                light_color = light_info.color,
                light_intensity = light_info.intensity,)
            
            response = await self.stub.SetLightInfo(request)
            if response.status != mjc_message_pb2.SetLightInfoResponse.SUCCESS:
                print("Set light info failed: ", response.error_message)
                raise Exception("Set light info failed.")
        
    def set_light_info(self, actor_name : str, light_info: LightInfo):
        self.loop.run_until_complete(self._set_light_info(actor_name, light_info))


    async def _set_camera_sensor_info(self, actor_name: str, camera_sensor_info: CameraSensorInfo):
        async with self.lock:
            request = mjc_message_pb2.SetCameraSensorInfoRequest(
                actor_name = actor_name,
                capture_rgb = camera_sensor_info.capture_rgb,
                capture_depth = camera_sensor_info.capture_depth,
                save_mp4_file = camera_sensor_info.save_mp4_file,
                use_dds = camera_sensor_info.use_dds,)
            
            response = await self.stub.SetCameraSensorInfo(request)
            if response.status != mjc_message_pb2.SetCameraSensorInfoResponse.SUCCESS:
                print("Set camera sensor info failed: ", response.error_message)
                raise Exception("Set camera sensor info failed.")
            
    def set_camera_sensor_info(self, actor_name: str, camera_sensor_info: CameraSensorInfo):
        self.loop.run_until_complete(self._set_camera_sensor_info(actor_name, camera_sensor_info))

    async def _make_camera_viewport_active(self, actor_name: str, entity_name: str):
        async with self.lock:
            request = mjc_message_pb2.MakeCameraViewportActiveRequest(
                actor_name = actor_name,
                entity_name = entity_name,)
            
            response = await self.stub.MakeCameraViewportActive(request)
            if response.status != mjc_message_pb2.MakeCameraViewportActiveResponse.SUCCESS:
                print("Make camera viewport activate failed: ", response.error_message)
                raise Exception("Make camera viewport activate failed.")
            
    def make_camera_viewport_active(self, actor_name: str, entity_name: str):
        self.loop.run_until_complete(self._make_camera_viewport_active(actor_name, entity_name))