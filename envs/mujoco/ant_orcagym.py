import numpy as np
from gymnasium.core import ObsType
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat, Tuple, Union
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
import orca_gym.adapters.robosuite.utils.transform_utils as transform_utils
from orca_gym.environment.orca_gym_env import RewardType
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
import time
import gymnasium as gym

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

class AntOrcaGymEnv(OrcaGymLocalEnv):
    """
    A class to represent the ORCA Gym environment for the Replicator scene.
    """

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        render_mode: str,
        env_id: Optional[str] = None,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        self._render_mode = render_mode
        self._env_id = env_id

        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        self._base_name = self.body("torso")
        self._base_joint_name = self.joint("root")
        self._init_body_geom_id()

        self._leg_joint_names = [
            self.joint("hip_1"), self.joint("ankle_1"), self.joint("hip_2"), self.joint("ankle_2"), self.joint("hip_3"), self.joint("ankle_3"), self.joint("hip_4"), self.joint("ankle_4")
        ]
        self._actuator_names = [self.actuator("M_hip_4"), self.actuator("M_ankle_4"), self.actuator("M_hip_1"), self.actuator("M_ankle_1")
                                , self.actuator("M_hip_2"), self.actuator("M_ankle_2"), self.actuator("M_hip_3"), self.actuator("M_ankle_3")]
        self._ctrl_index = self._get_ctrl_index()
        self._actuator_forcerange = self._get_actuator_forcerange()

        self._forward_reward_weight: float = 1
        self._ctrl_cost_weight: float = 5e-3
        self._contact_cost_weight: float = 5e-4

        self._healthy_reward: float = 1e-3
        self._terminated_reward: float = -10.0
        self._terminate_when_unhealthy: bool = True
        self._healthy_z_range: Tuple[float, float] = (0.2, 1.0)
        self._reset_noise_scale: float = 0.1

        self._contact_force_range : Tuple[float, float] = (-1.0, 1.0)

        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        # 归一化到 [-1, 1]区间
        scaled_action_range = np.concatenate([[[-1.0, 1.0]] for _ in range(self.nu)])
        # print("Scaled action range: ", scaled_action_range)
        self.action_space = self.generate_action_space(scaled_action_range)


    @property
    def contact_forces(self):
        raw_contact_forces = self.get_cfrc_ext()
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        qpos = self.query_joint_qpos([self._base_joint_name])[self._base_joint_name]
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z <= qpos[2] <= max_z
        return is_healthy
    
    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward


    def _init_body_geom_id(self):
        self._body_geom_id = []
        geom_dict = self.model.get_geom_dict()
        for geom_name, geom in geom_dict.items():
            body_name = self.model.get_geom_body_name(geom["GeomId"])
            if body_name == self._base_name:
                self._body_geom_id.append(geom["GeomId"])

        _logger.info(f"body_geom_id:  {self._body_geom_id}")

    @property
    def is_terminated(self):
        if not self._terminate_when_unhealthy:
            return False
        
        simple_contact = self.query_contact_simple()
        for contact in simple_contact:
            if contact["Geom1"] in self._body_geom_id or contact["Geom2"] in self._body_geom_id:
                return True
        return False

    @property
    def terminated_reward(self):
        return self.is_terminated * self._terminated_reward

    
    def render_callback(self, mode='human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action) -> tuple:
        action_dict = {self._actuator_names[i]: action[i] for i in range(len(self._actuator_names))}
        ctrl = self._action2ctrl(action_dict)
        # print("ctrl: ", ctrl)

        # step the simulation with original action space
        xy_position_before = self.get_body_xpos_xmat_xquat([self._base_name])[0][:2].copy()
        self.do_simulation(ctrl, self.frame_skip)
        xy_position_after = self.get_body_xpos_xmat_xquat([self._base_name])[0][:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        obs = self._get_obs().copy()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = self.is_terminated
        qpos = self.query_joint_qpos([self._base_joint_name])[self._base_joint_name]
        info = {
            "x_position": qpos[0],
            "y_position": qpos[1],
            "distance_from_origin": np.linalg.norm(qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        # if not self.is_healthy:
        #     print("Is healthy: ", self.is_healthy)

        # print("x_velocity: ", x_velocity, " y_velocity: ", y_velocity, " reward: ", reward, "reward info", reward_info, " terminated: ", terminated)

        # 正常来说不需要在这里调用render函数，但是由于APPO没有内置渲染回掉，如果需要在训练时查看，就需要在这里调用
        self.render()

        return obs, reward, terminated, False, info
    

    def _get_obs(self) -> np.ndarray:
        base_qpos = self.query_joint_qpos([self._base_joint_name])[self._base_joint_name]
        joint_qpos_dict = self.query_joint_qpos(self._leg_joint_names)
        joint_qpos = np.concatenate([joint_qpos_dict[joint] for joint in self._leg_joint_names]).flatten()

        base_qvel = self.query_joint_qvel([self._base_joint_name])[self._base_joint_name]
        joint_qvel_dict = self.query_joint_qvel(self._leg_joint_names)
        joint_qvel = np.concatenate([joint_qvel_dict[joint] for joint in self._leg_joint_names]).flatten()
        # obs = {
        #     "position": base_qpos[2:].copy(),
        #     "velocity": base_qvel.copy(),
        #     "joint_position": joint_qpos.copy(),
        #     "joint_velocity": joint_qvel.copy(),
        #     # "contact_force": self.contact_forces[1:].flatten(),
        # }

        obs = np.concatenate(
            [
                base_qpos[2:].copy(),  # x, y position
                base_qvel.copy(),  # x, y velocity
                joint_qpos.copy(),  # joint positions
                joint_qvel.copy(),  # joint velocities
            ],
            dtype=np.float32,
        ).flatten()

        return obs

    def reset_model(self) -> tuple[dict, dict]:
        """
        Reset the environment, return observation
        """
        
        body_qpos = self.query_joint_qpos([self._base_joint_name])[self._base_joint_name]
        body_xyz = body_qpos[:2]
        random_xyz = self._reset_noise_scale * self.np_random.uniform(-1, 1, (2))
        body_xyz += random_xyz

        self.set_joint_qpos({self._base_joint_name: body_qpos})
        self.mj_forward()

        self.ctrl = np.zeros(self.nu, dtype=np.float32)

        obs = self._get_obs().copy()
        return obs, self._get_reset_info()
    
    def _get_reset_info(self):
        qpos = self.query_joint_qpos([self._base_joint_name])[self._base_joint_name]
        return {
            "x_position": qpos[0],
            "y_position": qpos[1],
            "distance_from_origin": np.linalg.norm(qpos[0:2], ord=2),
        }

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
            ctrl[actuator_index] = action[actuator] * (actuator_forcerange[1] - actuator_forcerange[0]) / 2.0 + (actuator_forcerange[1] + actuator_forcerange[0]) / 2.0
        return ctrl

    def _get_rew(self, x_velocity: float, action):
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        terminated_reward = self.terminated_reward
        rewards = forward_reward + healthy_reward + terminated_reward

        ctrl_cost = self.control_cost(action)
        # contact_cost = self.contact_cost
        contact_cost = 0
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info
