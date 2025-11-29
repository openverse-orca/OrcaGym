import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import orca_gym.utils.rotations as rotations

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class LightsEnv(OrcaGymLocalEnv):
    """
    A class to represent the ORCA Gym environment for the Replicator scene.
    """

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        **kwargs,
    ):
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        self._light_config = [
            {"name" : "light_with_random_color_scale_intensity_0", "body_name" : "light_with_random_color_scale_intensity_0_SpotLight"},
            {"name" : "light_with_random_color_scale_intensity_1", "body_name" : "light_with_random_color_scale_intensity_1_SpotLight"},
            {"name" : "light_with_random_color_scale_intensity_2", "body_name" : "light_with_random_color_scale_intensity_2_SpotLight"},
            {"name" : "light_with_random_color_scale_intensity_3", "body_name" : "light_with_random_color_scale_intensity_3_SpotLight"},
            {"name" : "light_with_random_color_scale_intensity_4", "body_name" : "light_with_random_color_scale_intensity_4_SpotLight"},
            {"name" : "light_with_random_color_scale_intensity_5", "body_name" : "light_with_random_color_scale_intensity_5_SpotLight"},
            {"name" : "light_with_random_color_scale_intensity_6", "body_name" : "light_with_random_color_scale_intensity_6_SpotLight"},
            {"name" : "light_with_random_color_scale_intensity_7", "body_name" : "light_with_random_color_scale_intensity_7_SpotLight"},
            {"name" : "light_with_random_color_scale_intensity_8", "body_name" : "light_with_random_color_scale_intensity_8_SpotLight"},
            {"name" : "light_with_random_color_scale_intensity_9", "body_name" : "light_with_random_color_scale_intensity_9_SpotLight"},
            
        ]
        self._light_rotation_delta = np.zeros((len(self._light_config), 3), dtype=np.float32)
        _logger.info(f"Light rotation delta:  {self._light_rotation_delta}")
        self._light_rotation_update_phrase = 0


        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        self.action_space = self.generate_action_space(np.ones(self.nu, dtype=np.float32))

    
    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def step(self, action) -> tuple:

        ctrl = np.zeros(self.nu, dtype=np.float32)

        # print("runmode: ", self._run_mode, "no_scaled_action: ", noscaled_action, "scaled_action: ", scaled_action, "ctrl: ", ctrl)
        
        # step the simulation with original action space
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0.0

        self._rotate_lights()


        return obs, reward, terminated, truncated, info
    

    def _get_obs(self) -> dict:
           
        obs = {
            "joint_pos": self.data.qpos[:self.nq].copy(),
            "joint_vel": self.data.qvel[:self.nv].copy(),
            "joint_acc": self.data.qacc[:self.nv].copy(),
        }
        return obs


    def reset_model(self) -> tuple[dict, dict]:
        """
        Reset the environment, return observation
        """

        self.ctrl = np.zeros(self.nu, dtype=np.float32)

        obs = self._get_obs().copy()
        return obs, {}
    


    def get_observation(self, obs=None) -> dict:
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()
        
    def set_scene_runtime(self, scene_runtime: OrcaGymSceneRuntime) -> None:
        self.scene_runtime = scene_runtime
        _logger.performance("Scene runtime is set.")

    def _timer_now(self) -> float:
        """
        Get the current time in seconds.
        """
        return self.gym.data.time
    
    def _rotate_lights(self) -> None:
        """
        Rotate the lights in the scene.
        """
        for i, light in enumerate(self._light_config):
            if self._light_rotation_update_phrase == 0:
                self._light_rotation_delta[i] = np.array([
                    np.random.uniform(-np.pi * 0.01, np.pi * 0.01),
                    np.random.uniform(-np.pi * 0.01, np.pi * 0.01),
                    np.random.uniform(-np.pi * 0.01, np.pi * 0.01)])
            rotation_delta = rotations.euler2quat(self._light_rotation_delta[i])

            body_name = light["body_name"]
            current_xpos, _, current_rotation = self.get_body_xpos_xmat_xquat([body_name])

            new_rotation = rotations.quat_mul(current_rotation, rotation_delta)
            self.set_mocap_pos_and_quat(
                {body_name : {"pos": current_xpos, "quat": new_rotation}}
            )

        self._light_rotation_update_phrase = (self._light_rotation_update_phrase + 1) % 1000