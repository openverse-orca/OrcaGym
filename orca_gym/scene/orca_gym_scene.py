import grpc
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
import orca_gym.protos.mjc_message_pb2 as mjc_message_pb2
import asyncio
import numpy as np

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class Actor:
    """
    A class to represent an actor in the ORCA Gym environment.
    """

    def __init__(self,
                 name: str,
                 asset_path: str,
                 position: np.ndarray,
                 rotation: np.ndarray,
                 scale: float,):
        self.name = name
        self.asset_path = asset_path
        self.position = position
        self.rotation = rotation
        self.scale = float(scale)
        self._check_actor()

    def _check_actor(self):
        if self.name is None or self.name == "":
            raise ValueError("Actor name cannot be None or empty.")
        if self.asset_path is None or self.asset_path == "":
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
        
class MaterialInfo:
    """
    A class to represent material information in the ORCA Gym environment.
    """

    def __init__(self,
                 base_color: np.ndarray,):
        self.base_color = base_color
        self._check_material_info()

    def _check_material_info(self):
        if self.base_color is None or len(self.base_color) != 4:
            raise ValueError("Base color must be a 4D vector.")


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
                _logger.error(f"Publish scene failed:  {response.error_message}")
                raise Exception("Publish scene failed.")

    def publish_scene(self):
        self.loop.run_until_complete(self._publish_scene())

    async def _add_actor(self, actor : Actor):
        async with self.lock:  # 加锁保证串行
            request = mjc_message_pb2.AddActorRequest(
                name = actor.name,
                spawnable_name = actor.asset_path,
                pos = actor.position,
                quat = actor.rotation,
                scale = actor.scale,)
        
        _logger.info(f"尝试添加 Actor: name='{actor.name}', spawnable_name='{actor.asset_path}', position={actor.position}, scale={actor.scale}")
        response = await self.stub.AddActor(request)
        if response.status != mjc_message_pb2.AddActorResponse.SUCCESS:
            _logger.error("=" * 80)
            _logger.error("添加 Actor 失败！详细信息：")
            _logger.error(f"  Actor 名称 (name): '{actor.name}'")
            _logger.error(f"  Spawnable 路径 (asset_path): '{actor.asset_path}'")
            _logger.error(f"  位置 (position): {actor.position}")
            _logger.error(f"  旋转 (rotation): {actor.rotation}")
            _logger.error(f"  缩放 (scale): {actor.scale}")
            _logger.error(f"  错误消息: {response.error_message}")
            _logger.error("=" * 80)
            _logger.error("可能的原因：")
            if "already exists" in response.error_message.lower() or "name already" in response.error_message.lower():
                _logger.error("  ⚠️  Actor 名称已存在！")
                _logger.error("  - 场景中已经存在同名的 actor")
                _logger.error("  - 解决方案：在添加 actor 之前先清空场景（调用 publish_scene() 发布空场景）")
                _logger.error("  - 或者使用不同的 actor 名称")
            elif "spawnable" in response.error_message.lower() and "not found" in response.error_message.lower():
                _logger.error("  ⚠️  Spawnable 路径不存在！")
                _logger.error("  1. Spawnable 路径不存在或拼写错误")
                _logger.error("  2. 资产文件未正确导入到 OrcaStudio")
                _logger.error("  3. 路径格式不正确（检查是否有双斜杠、路径分隔符等）")
            else:
                _logger.error("  1. Spawnable 路径不存在或拼写错误")
                _logger.error("  2. 资产文件未正确导入到 OrcaStudio")
                _logger.error("  3. 路径格式不正确（检查是否有双斜杠、路径分隔符等）")
                _logger.error("  4. Actor 名称冲突（场景中已存在同名 actor）")
            _logger.error("=" * 80)
            raise Exception(f"Add actor failed. Actor name: '{actor.name}', Spawnable path: '{actor.asset_path}', Error: {response.error_message}")
        
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
                _logger.error(f"Set light info failed:  {response.error_message}")
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
                _logger.error(f"Set camera sensor info failed:  {response.error_message}")
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
                _logger.error(f"Make camera viewport activate failed:  {response.error_message}")
                raise Exception("Make camera viewport activate failed.")
            
    def make_camera_viewport_active(self, actor_name: str, entity_name: str):
        self.loop.run_until_complete(self._make_camera_viewport_active(actor_name, entity_name))

    async def _set_material_info(self, actor_name: str, material_info: MaterialInfo):
        async with self.lock:
            request = mjc_message_pb2.SetMaterialInfoRequest(
                actor_name = actor_name,
                base_color = material_info.base_color,)
            
            response = await self.stub.SetMaterialInfo(request)
            if response.status != mjc_message_pb2.SetMaterialInfoResponse.SUCCESS:
                _logger.error(f"Set material info failed:  {response.error_message}")
                raise Exception("Set material info failed.")
            
    def set_material_info(self, actor_name: str, material_info: MaterialInfo):
        self.loop.run_until_complete(self._set_material_info(actor_name, material_info))


    async def _set_actor_anim_param_number(self, actor_name: str, param_name: str, value: float):
        async with self.lock:
            request = mjc_message_pb2.SetActorAnimParamNumberRequest(
                actor_name = actor_name,
                param_name = param_name,
                value = value,)
            
            response = await self.stub.SetActorAnimParamNumber(request)
            if response.status != mjc_message_pb2.SetActorAnimParamNumberResponse.SUCCESS:
                _logger.error(f"Set actor anim param number failed:  {response.error_message}")
                raise Exception("Set actor anim param number failed.")
            
    def set_actor_anim_param_number(self, actor_name: str, param_name: str, value: float):
        self.loop.run_until_complete(self._set_actor_anim_param_number(actor_name, param_name, value))

    async def _set_actor_anim_param_bool(self, actor_name: str, param_name: str, value: bool):
        async with self.lock:
            request = mjc_message_pb2.SetActorAnimParamBoolRequest(
                actor_name = actor_name,
                param_name = param_name,
                value = value,)
            
            response = await self.stub.SetActorAnimParamBool(request)
            if response.status != mjc_message_pb2.SetActorAnimParamBoolResponse.SUCCESS:
                _logger.error(f"Set actor anim param bool failed:  {response.error_message}")
                raise Exception("Set actor anim param bool failed.")
            
    def set_actor_anim_param_bool(self, actor_name: str, param_name: str, value: bool):
        self.loop.run_until_complete(self._set_actor_anim_param_bool(actor_name, param_name, value))

    async def _set_actor_anim_param_string(self, actor_name: str, param_name: str, value: str):
        async with self.lock:
            request = mjc_message_pb2.SetActorAnimParamStringRequest(
                actor_name = actor_name,
                param_name = param_name,
                value = value,)
            
            response = await self.stub.SetActorAnimParamString(request)
            if response.status != mjc_message_pb2.SetActorAnimParamStringResponse.SUCCESS:
                _logger.error(f"Set actor anim param string failed:  {response.error_message}")
                raise Exception("Set actor anim param string failed.")
            
    def set_actor_anim_param_string(self, actor_name: str, param_name: str, value: str):
        self.loop.run_until_complete(self._set_actor_anim_param_string(actor_name, param_name, value))