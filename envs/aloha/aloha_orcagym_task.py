import collections

from gym_aloha.tasks.sim import TransferCubeTask
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.sensor.rgbd_camera import CameraWrapper
from dm_control.mujoco.engine import Physics

class TransferCubeTask_OrcaGym(TransferCubeTask):
    def __init__(self, 
                 random,
                 orcagym_env : OrcaGymLocalEnv,
                 camera_config : dict,
                 ):
        super().__init__(random=random)
        self._orcagym_env = orcagym_env
        
        cameras = [CameraWrapper(name=camera_name, port=camera_port) for camera_name, camera_port in camera_config.items()]
        self._top_camera = cameras[0]
        self._top_camera.start()
        
    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["top"] = self._top_camera.get_frame(format='rgb24')
        # obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        # obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")

        return obs
    
    def after_step(self, physics : Physics):
        """
        之前由dm_env运行了mujoco环境的仿真，这里不再需要运行step，
        而是需要将mujoco的数据传递给orcagym_env，然后渲染orcagym_env
        """
        super().after_step(physics)
        data = physics.data
        self._orcagym_env.gym.update_data_external(data.qpos, data.qvel, data.qacc, data.qfrc_bias, data.time)
        self._orcagym_env.render()
        return