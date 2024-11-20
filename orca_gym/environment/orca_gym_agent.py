import numpy as np
from gymnasium.core import ObsType
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces



class OrcaGymAgent:
    def __init__(self, 
                 env_id: str,
                 agent_name: str, 
                 task: str,
                 max_episode_steps: int):
        
        self._env_id = env_id
        self._agent_name = agent_name
        self._task = task
        self._max_episode_steps = max_episode_steps
        self._current_episode_step = 0

        self._joint_names = None
        self._actuator_names = None
        self._site_names = None
        self._sensor_names = None

        self._ctrl = None
        self._nu = None
        self._nq = None
        self._nv = None


    @property
    def name(self) -> str:
        return self._agent_name

    def name_space(self, name : str) -> str:
        return f"{self._agent_name}_{name}"
    
    def name_space_list(self, names : list[str]) -> list[str]:
        return [self.name_space(name) for name in names]
    
    @property
    def joint_names(self) -> list[str]:
        return self._joint_names
    
    @property
    def actuator_names(self) -> list[str]:
        return self._actuator_names
    
    @property
    def site_names(self) -> list[str]:
        return self._site_names
    
    @property
    def sensor_names(self) -> list[str]:
        return self._sensor_names

    @property
    def nu(self) -> int:
        return self._nu
    
    @property
    def nq(self) -> int:
        return self._nq
    
    @property
    def nv(self) -> int:
        return self._nv

    @property
    def truncated(self) -> bool:
        return self._current_episode_step >= self._max_episode_steps

    @property
    def ctrl_start(self) -> int:
        return self._ctrl_start

    def get_obs(self, **kwargs):
        raise NotImplementedError

    def set_ctrl_info(self, actuator_dict) -> None:
        if not hasattr(self, "_ctrl_range"):
            self._ctrl_range = []
        if not hasattr(self, "_ctrl_offset"):
            self._ctrl_start = 0
            
        for i, actuator_name in enumerate(self._actuator_names):
            # matain the order of actuators
            self._ctrl_range.append(np.array(actuator_dict[actuator_name]['CtrlRange']).flatten())
            if i == 0:
                self._ctrl_start = actuator_dict[actuator_name]['ActuatorId']

    def set_action(self, action):
        assert len(action) == len(self._ctrl_range)

        for i in range(len(action)):
            # 线性变换到 ctrl range 空间
            # print("action: ", action[i])
            # print("action_space_range: ", self._action_space_range)
            # print("ctrl_range: ", self._ctrl_range[i])
            self._ctrl[i] = np.interp(action[i], self._action_space_range, self._ctrl_range[i])

        # print("Agent: ", self.name, "Ctrl: ", self._ctrl)

        return
    
    def set_action_space(self, action_space : spaces) -> None:
        self._action_space = action_space
        self._action_space_range = [action_space.low[0], action_space.high[0]]
    
    def on_step(self):
        """
        Called after each step in the environment.
        Implement this method in the subclass to perform additional operations.
        """
        pass

    def step(self, action) -> np.ndarray:
        self._current_episode_step += 1
        self.set_action(action)
        self.on_step()
        return self._ctrl

    def on_reset(self, np_random : np.random.Generator):
        """
        Called after each reset in the environment.
        Implement this method in the subclass to perform additional operations.
        """
        pass

    def reset(self, np_random : np.random.Generator):
        self._current_episode_step = 0
        reset_info = self.on_reset(np_random)
        return reset_info


    def is_success(self, achieved_goal, desired_goal) -> np.float32:
        raise NotImplementedError

    def is_terminated(self, achieved_goal, desired_goal) -> bool:
        raise NotImplementedError
        
    def compute_reward(self, achieved_goal, desired_goal) -> SupportsFloat:
        raise NotImplementedError
        
    def set_init_state(self, joint_qpos: dict):
        raise NotImplementedError
    
    def get_action_size(self) -> int:
        """
        Action size can be overridden in the subclass.
        In most of cases, this is the number of actuators in the robot.
        But in some cases, the action size may be different.
        """        
        return self._nu