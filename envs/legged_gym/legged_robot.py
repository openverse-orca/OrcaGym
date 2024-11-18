import numpy as np
from gymnasium.core import ObsType
from envs import OrcaGymLocalEnv, OrcaGymRemoteEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces



class LeggedRobot:
    def __init__(self, 
                 agent_name: str, 
                 task: str,
                 max_episode_steps: int):
        
        self._agent_name = agent_name
        self._task = task
        self._max_episode_steps = max_episode_steps
        self._current_episode_step = 0

        self._leg_joint_names = None
        self._neutral_joint_angles = None
        self._leg_actuator_names = None
        self._ctrl = None


    @property
    def name(self) -> str:
        return self._agent_name

    def name_space(self, name : str) -> str:
        return f"{self._agent_name}_{name}"
    
    def name_space_list(self, names : list[str]) -> list[str]:
        return [self.name_space(name) for name in names]
    
    @property
    def leg_joint_names(self) -> list[str]:
        return self._leg_joint_names
    
    @property
    def neutral_joint_angles(self) -> np.ndarray:
        return np.array([self._neutral_joint_angles.values()])
    

    @property
    def truncated(self) -> bool:
        return self._current_episode_step >= self._max_episode_steps


    def get_joint_neutral(self) -> np.ndarray:
        joint_qpos = {}
        for name, value in zip(self.leg_joint_names, self.neutral_joint_angles):
            joint_qpos[name] = np.array([value])
        return joint_qpos


    def get_obs(self, site_pos_quat, site_pos_mat, site_xvalp, site_xvalr, joint_qpos, dt) -> dict:
        if self._task == "reach":
            achieved_goal = ee_position.copy()
            desired_goal = self.goal.copy()
            obs = np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                    ]).copy()   
        elif self._task == "pick_and_place":
            achieved_goal = np.concatenate([object_position, ee_position])
            desired_goal = np.concatenate([self.goal, self.goal])
            obs = np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        fingers_qpos,
                        object_position,
                        object_rotation,
                        object_velp,
                        object_velr,
                    ]).copy()                 
        else:
            raise ValueError("Unsupport task type: ", self._task)   

        result = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

        # print("Agent Obs: ", result)

        return result

    def set_acutator_ctrl_range(self, actuator_dict) -> None:
        if not hasattr(self, "_ctrl_range"):
            self._ctrl_range = []
            
        for actuator_name in self._leg_actuator_names:
            # matain the order of actuators
            self._ctrl_range.append(actuator_dict[actuator_name])

    def set_action(self, action):
        assert len(action) == len(self._ctrl_range)

        for i in range(len(action)):
            # 线性变换到 ctrl range 空间
            self._ctrl[i] = np.interp(action[i], self._action_space_range, self._ctrl_range[i])

        return
    
    def set_action_space(self, action_space : spaces) -> None:
        self._action_space = action_space
        self._action_space_range = [action_space.low, action_space.high]
    
    def step(self, action):
        self._current_episode_step += 1
        self.set_action(action)
        return self._ctrl

    def reset(self, np_random) -> dict:
        self._current_episode_step = 0
        joint_neutral_qpos = self.get_joint_neutral()
        return joint_neutral_qpos

    def is_success(self, achieved_goal, desired_goal, env_id) -> np.float32:
        d = self._goal_distance(achieved_goal, desired_goal)
        # if d < self._distance_threshold:
        #     print(f"{env_id} Agent {self.name} Task Sussecced: achieved goal: ", achieved_goal, "desired goal: ", desired_goal)
        return (d < self._distance_threshold).astype(np.float32)

    def _compute_reward_ndim1(self, achieved_goal, desired_goal) -> SupportsFloat:
        d = self._goal_distance(achieved_goal, desired_goal)
        if self._task == "reach":
            if d < self._distance_threshold:
                reward = 1.0    # is_success
            else:
                # reward = -1   # sparse reward
                reward = -d     # dense reward
            return reward
        elif self._task == "pick_and_place":
            if d < self._distance_threshold:
                reward = 1.0
            else:
                reward = self._compute_pick_and_place_reward(achieved_goal, desired_goal)

            return reward
        else:
            raise ValueError("Unsupport task type: ", self._task)
        
    def _compute_reward_ndim2(self, achieved_goal, desired_goal) -> SupportsFloat:
        d = self._goal_distance(achieved_goal, desired_goal)
        if self._task == "reach":
            rewards = np.zeros(len(achieved_goal))
            for i in range(len(achieved_goal)):
                if d[i] < self._distance_threshold:
                    rewards[i] = 1.0
                else:
                    # rewards[i] = -1.0
                    rewards[i] = -d[i]
            return rewards
        elif self._task == "pick_and_place":
            rewards = np.zeros(len(achieved_goal))
            for i in range(len(achieved_goal)):
                if d[i] < self._distance_threshold:
                    rewards[i] = 1.0
                else:
                    rewards[i] = self._compute_pick_and_place_reward(achieved_goal[i], desired_goal[i])

            return rewards
        else:
            raise ValueError("Unsupport task type: ", self._task)
        
    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        if achieved_goal.ndim == 1:
            return self._compute_reward_ndim1(achieved_goal, desired_goal)
        else:
            return self._compute_reward_ndim2(achieved_goal, desired_goal)
        

    def _compute_pick_and_place_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        assert achieved_goal.shape == desired_goal.shape
        ee_position = achieved_goal[:3]
        object_position = achieved_goal[3:6]
        goal_position = desired_goal[:3]

        reward = 0

        # 1. ee to object distance
        ee_to_obj_distance = np.linalg.norm(ee_position - object_position)
        reward += -ee_to_obj_distance

        # 2. object to goal distance
        obj_to_goal_distance = np.linalg.norm(object_position - goal_position)
        reward += -obj_to_goal_distance

        return reward