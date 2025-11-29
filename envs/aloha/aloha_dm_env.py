import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask
from gym_aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose

from envs.aloha.aloha_env import AlohaEnv
import time
from envs.aloha.aloha_orcagym_task import TransferCubeTask_OrcaGym
from dm_control.mujoco import wrapper
import imageio

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class AlohaDMEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        task,    
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,        
        camera_config: dict,    
        obs_type="pixels",
        render_mode="human",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
        **kwargs,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        

        self._aloha_env = self._make_orca_gym_local_env(frame_skip, orcagym_addr, agent_names, time_step, **kwargs)
        self._aloha_env.reset()
        _logger.info(f"Init aloha orca gym env, reseted.")
        
        self._dm_env = self._make_dm_env_task(
            task_name=self.task,
            camera_config=camera_config,
        )

        if self.obs_type == "state":
            raise NotImplementedError()
            self.observation_space = spaces.Box(
                low=np.array([0] * len(JOINTS)),  # ???
                high=np.array([255] * len(JOINTS)),  # ???
                dtype=np.float64,
            )
        elif self.obs_type == "pixels":
            _logger.info("obs_type: pixels")
            self.observation_space = spaces.Dict(
                {
                    "top": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            _logger.info("obs_type: pixels_agent_pos")
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8, 
                            )
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        # assert self.render_mode == "human"
        # width, height = (
        #     (self.visualization_width, self.visualization_height)
        #     if visualize
        #     else (self.observation_width, self.observation_height)
        # )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        # image = self._dm_env.physics.render(height=height, width=width, camera_id="top")
        
        # self._aloha_env.render()
        return

    def _make_orca_gym_local_env(self, 
                                 frame_skip: int,
                                 orcagym_addr: str,
                                 agent_names: list[str],
                                 time_step: float,
                                 **kwargs) -> AlohaEnv:
        return AlohaEnv(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

    def _make_dm_env_task(self, 
                          task_name: str,
                          camera_config: dict,
                          ) -> control.Environment:
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")
        
        mj_model = wrapper.MjModel(self._aloha_env.gym._mjModel)
        physics = mujoco.Physics.from_model(mj_model)

        if task_name == "transfer_cube":
            # xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            # physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask_OrcaGym(
                random=None,
                orcagym_env=self._aloha_env,
                camera_config=camera_config,
            )
        elif task_name == "insertion":
            # xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            # physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "end_effector_transfer_cube":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif task_name == "end_effector_insertion":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        dm_env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return dm_env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {"top": raw_obs["images"]["top"].copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {"top": raw_obs["images"]["top"].copy()},
                "agent_pos": raw_obs["qpos"],
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._dm_env.task.random.seed(seed)
            self._dm_env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
            BOX_POSE[1] = sample_box_pose(seed)  # used in sim reset
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._dm_env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        _, reward, _, raw_obs = self._dm_env.step(action)

        # TODO(rcadene): add an enum
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)
        
        # img_data = observation["pixels"]["top"]
        # time = self._dm_env.physics.data.time
        # imageio.imwrite(f"camera_frame_top_{time}.png", img_data)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
