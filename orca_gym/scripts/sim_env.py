import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class SimEnv(OrcaGymLocalEnv):
    """
    A class to represent the ORCA Gym environment for the Replicator scene.
    """

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        max_steps: Optional[int] = None,
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

        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        # 归一化到 [-1, 1]区间
        if (self.nu > 0):
            scaled_action_range = np.concatenate([[[-1.0, 1.0]] for _ in range(self.nu)])
            # print("Scaled action range: ", scaled_action_range)
            self.action_space = self.generate_action_space(scaled_action_range)
        else:
            self.action_space = spaces.Box(
                low=np.array([]),
                high=np.array([]),
                dtype=np.float32
            )
            _logger.info(f"No action space defined, nu is 0.")

    
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
