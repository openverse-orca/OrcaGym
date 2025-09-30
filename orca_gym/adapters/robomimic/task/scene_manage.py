import time
import warnings

import numpy as np

from orca_gym.scene.orca_gym_scene import Actor, MaterialInfo, LightInfo, OrcaGymScene

class SceneManager:
    def __init__(self, grpc_addr: str, init_env_callback = None):
        self._scene = OrcaGymScene(grpc_addr)
        self._init_env_callback = init_env_callback

    def register_init_env_callback(self, callback):
        self._init_env_callback = callback

    def add_actor(self, actor_name: str, asset_path: str, position: np.ndarray, rotation: np.ndarray, scale: float = 1.0):
        actor = Actor(actor_name, asset_path, position, rotation, scale)
        self._scene.add_actor(actor)

    def add_light(self, light_name: str, asset_path: str, position: np.ndarray, rotation: np.ndarray, scale: float = 1.0):
        actor= Actor(light_name, asset_path, position, rotation, 1.0)
        self._scene.add_actor(actor)

    def set_light_info(self, light_name: str, color = None, intensity = None):
        light_info = LightInfo(color, intensity)
        self._scene.set_light_info(light_name, light_info)

    def set_material_info(self, actor_name: str, base_color = None):
        material_info = MaterialInfo(base_color)
        self._scene.set_material_info(actor_name, material_info)

    def publish_scene(self):
        """
        Publish the scene to the ORCA Gym environment.
        """
        self._scene.publish_scene()
        if self._init_env_callback is None:
            warnings.warn("The ORCA Gym environment does not have an init_env_callback method.")

        time.sleep(3)
        self._init_env_callback()

    def publish_scene_without_init_env(self):
        '''
        只添加视觉效果，无需初始化环境。
        '''
        self._scene.publish_scene()

    def destory_scene(self):
        self._scene.publish_scene()
        self._scene.close()

