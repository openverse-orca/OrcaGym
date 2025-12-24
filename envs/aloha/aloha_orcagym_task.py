import collections

from gym_aloha.tasks.sim import TransferCubeTask, START_ARM_POSE, BOX_POSE
import numpy as np
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.sensor.rgbd_camera import CameraWrapper
from dm_control.mujoco.engine import Physics
import time

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

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
        obs["images"]["top"], _ = self._top_camera.get_frame(format='rgb24')
        # obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        # obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")

        return obs
    
    def after_step(self, physics : Physics):
        """
        之前由dm_env运行了mujoco环境的仿真，这里不再需要运行step，
        而是需要将mujoco的数据传递给orcagym_env，然后渲染orcagym_env
        """
        super().after_step(physics)
        
        _logger.performance(f"Physics step, time={physics.time()}")
        
        data = physics.data
        self._orcagym_env.gym.update_data_external(data.qpos, data.qvel, data.qacc, data.qfrc_bias, data.time)
        self._orcagym_env.render()
        
        # 等待渲染结果串流到orcagym的客户端，最长等待时间不超过最大帧率
        time.sleep(0.02) # max_hz=50
        
        
        return
    
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None and BOX_POSE[1] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            physics.named.data.qpos[-7 * 2 : -7] = BOX_POSE[1]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)    