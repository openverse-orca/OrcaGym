import numpy as np
from orca_gym.scene.orca_gym_scene import OrcaGymScene
from orca_gym.scene.orca_gym_scene import Actor, LightInfo, CameraSensorInfo, MaterialInfo

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class OrcaGymSceneRuntime():
    """
    将 OrcaGymScene 的功能封装为一个运行时类
    Env 对象可以持有Runtime类的实例，实现部分控制Scene的能力。

    - 允许的Scene操作：
        例如 调整 Light 属性， 切换 Camera。这类操作不涉及场景资源销毁和仿真对象的变更

    - 禁止的Scene操作：
        例如 添加 Actor， Publish Scene。这类操作涉及资源销毁和仿真对象的变更，无法在仿真运行过程中使用
    """

    def __init__(
        self,
        scene: OrcaGymScene
    ):
        # Initialize the scene
        self.scene = scene

    def set_light_info(self, actor_name : str, light_info: LightInfo):
        try:
            self.scene.set_light_info(actor_name, light_info)
        except Exception as e:
            _logger.error(f"Failed to set light info for {actor_name}: {e}")


    def make_camera_viewport_active(self, actor_name: str, entity_name: str):
        try:
            self.scene.make_camera_viewport_active(actor_name, entity_name)
        except Exception as e:
            _logger.error(f"Failed to make camera viewport active for {actor_name}: {e}")

    def set_actor_anim_param_number(self, actor_name: str, param_name: str, value: float):
        try:
            self.scene.set_actor_anim_param_number(actor_name, param_name, value)
        except Exception as e:
            _logger.error(f"Failed to set actor anim param number for {actor_name}: {e}")

    def set_actor_anim_param_bool(self, actor_name: str, param_name: str, value: bool):
        try:
            self.scene.set_actor_anim_param_bool(actor_name, param_name, value)
        except Exception as e:
            _logger.error(f"Failed to set actor anim param bool for {actor_name}: {e}")

    def set_actor_anim_param_string(self, actor_name: str, param_name: str, value: str):
        try:
            self.scene.set_actor_anim_param_string(actor_name, param_name, value)
        except Exception as e:
            _logger.error(f"Failed to set actor anim param string for {actor_name}: {e}")