import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
import orca_gym.adapters.robosuite.utils.transform_utils as transform_utils
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


class WheeledChassisEnv(OrcaGymLocalEnv):
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

        self._base_name = self.body("base_link")
        self._actuator_names = [self.actuator("M_wheel_r"), self.actuator("M_wheel_l")]
        self._right_wheel_name = self.actuator("M_wheel_r")
        self._left_wheel_name = self.actuator("M_wheel_l")
        self._ctrl_index = self._get_ctrl_index()
        self._actuator_forcerange = self._get_actuator_forcerange()
        self._actuator_dir = {self.actuator("M_wheel_r"): 1.0, self.actuator("M_wheel_l"): -1.0}
        # print("Actuator ctrl range: ", self._actuator_forcerange)
        _, _, self._last_xquat= self.get_body_xpos_xmat_xquat([self._base_name])

        self._keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)

        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        # 归一化到 [-1, 1]区间
        scaled_action_range = np.concatenate([[[-1.0, 1.0]] for _ in range(self.nu)])
        # print("Scaled action range: ", scaled_action_range)
        self.action_space = self.generate_action_space(scaled_action_range)

    
    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def step(self, action) -> tuple:

        ctrl = self._process_input()

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
        
    def _get_ctrl_index(self):
        """
        Get the index of the control in the actuator list.
        """
        ctrl_index = {}
        for actuator in self._actuator_names:
            ctrl_index[actuator] = self.model.actuator_name2id(actuator)
        return ctrl_index
    
    def _get_actuator_forcerange(self):
        """
        Get the actuator force range.
        """
        all_ctrlrange = self.model.get_actuator_ctrlrange()
        # print("Actuator ctrl range: ", all_ctrlrange)
        actuator_forcerange = {}
        for actuator in self._actuator_names:
            actuator_forcerange[actuator] = all_ctrlrange[self._ctrl_index[actuator]]
        return actuator_forcerange
    
    def _action2ctrl(self, action: dict[str, float]) -> np.ndarray:
        """
        Convert the action to control.
        action is normalized to [-1, 1]
        ctrl is in range of actuator force
        """
        ctrl = np.zeros(self.nu, dtype=np.float32)
        for actuator in self._actuator_names:
            actuator_index = self._ctrl_index[actuator]
            actuator_forcerange = self._actuator_forcerange[actuator]
            actuator_dir = self._actuator_dir[actuator]
            ctrl[actuator_index] = actuator_dir * action[actuator] * (actuator_forcerange[1] - actuator_forcerange[0]) / 2.0 + (actuator_forcerange[1] + actuator_forcerange[0]) / 2.0
        return ctrl

    def _process_input(self):
        """
        Process the input from the keyboard and joystick.
        """
        MOVE_SPEED = 0.2
        TURN_SPEED = 0.2

        self._keyboard.update()
        state = self._keyboard.get_state()

        move_forward = state["W"] - state["S"]
        turn_left = state["A"] - state["D"]
        move_backward = state["S"] - state["W"]
        turn_right = state["D"] - state["A"]

        if move_forward > 0 and (turn_left == 0 and turn_right == 0):
            turn_left = self._drift_correction(1.0)
        elif move_backward > 0 and (turn_left == 0 and turn_right == 0):
            turn_left = self._drift_correction(1.0)

        move = (move_forward - move_backward) * MOVE_SPEED / 2
        turn = (turn_left - turn_right) * TURN_SPEED / 2

        # Create a dictionary to hold the action
        action = {self._right_wheel_name: 0.0, self._left_wheel_name: 0.0}
        action[self._right_wheel_name] = move + turn
        action[self._left_wheel_name] = move - turn

        # Normalize the action to be between -1 and 1
        action[self._right_wheel_name] = np.clip(action[self._right_wheel_name], -1.0, 1.0)
        action[self._left_wheel_name] = np.clip(action[self._left_wheel_name], -1.0, 1.0)

        # convert the action to control
        ctrl = self._action2ctrl(action)

        return ctrl

    def _drift_correction(self, move_dir: float):
        """
        Drift correction for the wheeled chassis.
        """
        DRIFT_THRESHOLD = 0.002 * self.dt

        _, _, current_xquat = self.get_body_xpos_xmat_xquat([self._base_name])
        current_angle_z = rotations.quat2euler(current_xquat)[2]
        last_angle_z = rotations.quat2euler(self._last_xquat)[2]
        self._last_xquat = current_xquat

        # Calculate the angle difference
        angle_diff = current_angle_z - last_angle_z
        if abs(angle_diff) < DRIFT_THRESHOLD:
            # If the angle difference is small, no correction needed
            # print("diff: ", angle_diff)
            return 0.0
        else:
            # If the angle difference is large, apply correction
            correction = -angle_diff * move_dir * 10000000
            correction = np.clip(correction, -1.0, 1.0)
            return correction