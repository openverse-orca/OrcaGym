import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import orca_gym.utils.rotations as rotations

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class CamerasEnv(OrcaGymLocalEnv):
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

        self._camera_config = [
            {"name" : "default_camera","duration" : 5, "body_name" : "default_camera_CameraViewport"}, 
            # {"name" : "top_camera", "duration" : 5, "body_name" : "top_camera_CameraViewport"},
            # {"name" : "side_camera", "duration" : 5, "body_name" : "side_camera_CameraViewport"},
        ]
        self._current_camera = 0
        self._camera_switch_time = 0.0

        self._default_camera_body_name = self._camera_config[0]["body_name"]
        # self._top_camera_body_name = self._camera_config[1]["body_name"]
        # self._side_camera_body_name = self._camera_config[2]["body_name"]
        # self._side_camera_init_pos, _, self._side_camera_init_quat = self.get_body_xpos_xmat_xquat([self._side_camera_body_name])
        # self._side_camera_rotate_angle = 0.0
        self._camera_move_phrase = 0



        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        self.action_space = self.generate_action_space(np.array([self.nu, self.nu], dtype=np.float32))

    
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


        self._move_default_camera()
        # self._rotate_top_camera()
        # self._slide_side_camera()
        self._switch_camera()

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
    
    def _switch_camera(self) -> None:
        """
        Switch the camera view.
        """
        if not hasattr(self, "scene_runtime") or self.scene_runtime is None:
            _logger.performance("Scene runtime is not set.")
            return
        
        if self._timer_now() - self._camera_switch_time > self._camera_config[self._current_camera]["duration"]:
            self._current_camera = (self._current_camera + 1) % len(self._camera_config)
            camera_name = self._camera_config[self._current_camera]["name"]
            entity_name = "CameraViewport"
            self.scene_runtime.make_camera_viewport_active(camera_name, entity_name)
            _logger.info(f"Switched to camera: {camera_name}")
            self._camera_switch_time = self._timer_now()
        else:
            # print(f"Current camera: {self._switch_camera_config[self._current_camera]['name']}, time left: {self._switch_camera_config[self._current_camera]['duration'] - (self._timer_now() - self._camera_switch_time)}")
            pass
        
    def _move_default_camera(self) -> None:
        self._camera_move_phrase = (self._camera_move_phrase + 1) % 1000

        default_camera_pos, _, default_camera_quat = self.get_body_xpos_xmat_xquat([self._default_camera_body_name])

        y_delta = -0.002 if self._camera_move_phrase < 500 else 0.002
        default_camera_local_y_delta = np.array([0.0, y_delta, 0.0])
        default_camera_global_y_delta = rotations.quat_rot_vec(default_camera_quat, default_camera_local_y_delta)
        default_camera_pos += default_camera_global_y_delta
        self.set_mocap_pos_and_quat({self._default_camera_body_name: {"pos": default_camera_pos, "quat": default_camera_quat}})
        
    def _rotate_top_camera(self) -> None:
        top_camera_pos, _, top_camera_quat = self.get_body_xpos_xmat_xquat([self._top_camera_body_name])
        top_camera_roll_angle = 0.002
        top_camera_local_roll_delta = np.array([0.0, 0.0, top_camera_roll_angle])
        top_camera_quat_new = rotations.quat_mul(rotations.euler2quat(top_camera_local_roll_delta), top_camera_quat)
        self.set_mocap_pos_and_quat({self._top_camera_body_name: {"pos": top_camera_pos, "quat": top_camera_quat_new}})

    def _slide_side_camera(self) -> None:
        self._side_camera_rotate_angle = (self._side_camera_rotate_angle + 0.002) % (np.pi * 2)
        side_camera_rotate_quat = rotations.euler2quat(np.array([0.0, 0.0, self._side_camera_rotate_angle]))
        side_camera_quat = rotations.quat_mul(self._side_camera_init_quat, side_camera_rotate_quat)
        side_camera_pos = rotations.quat_rot_vec(side_camera_rotate_quat, self._side_camera_init_pos)
        self.set_mocap_pos_and_quat({self._side_camera_body_name: {"pos": side_camera_pos, "quat": side_camera_quat}})

