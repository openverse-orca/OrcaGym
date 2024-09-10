from os import path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space

import asyncio
import sys
from orca_gym import orca_gym
from orca_gym.orca_gym import OrcaGym
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
from orca_gym.utils.rotations import mat2quat, quat2mat

from orca_gym.orca_gym import OrcaGymModel
from orca_gym.orca_gym import OrcaGymData

class ActionSpaceType:
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"

class BaseOrcaGymEnv(gym.Env[NDArray[np.float64], NDArray[np.float32]]):
    """Superclass for all OrcaSim environments."""

    def __init__(
        self,
        frame_skip: int,
        grpc_address: str,
        agent_names: list[str],
        time_step: float,
        observation_space: Space,
        action_space_type: Optional[ActionSpaceType],
        action_step_count: Optional[float],
        **kwargs
    ):
        """Base abstract class for OrcaSim based environments.

        Args:
            frame_skip: Number of MuJoCo simulation steps per gym `step()`.
            observation_space: The observation space of the environment.
            grpc_address: The address of the gRPC server.
            agent_names: The names of the agents in the environment.
            time_step: The time step of the simulation.

        Raises:
        """

        # 初始化GRPC通信管道，采用异步通信
        self.grpc_address = grpc_address
        self.channel = None
        self.stub = None
        self.gym = None
        self._agent_names = agent_names
        self.seed = 0
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._initialize_grpc())
        self.loop.run_until_complete(self._pause_simulation())  # 暂停仿真，Gym 采用被动模式，OrcaSim侧不执行周期循环
        self.loop.run_until_complete(self._set_time_step(time_step))  # 设置仿真时间步长

        # may use width and height
        self.model, self.data = self._initialize_simulation()
        print("Agent Names: ", self._agent_names)
        
        self.reset_simulation() # 重置仿真环境

        self._init_qpos_qvel()

        self.frame_skip = frame_skip

        if observation_space is not None:
            self.observation_space = observation_space

        if action_space_type is None:
            self.action_space_type = ActionSpaceType.CONTINUOUS
        else:    
            self.action_space_type = action_space_type

        if self.action_space_type == ActionSpaceType.CONTINUOUS:
            self._set_continuous_action_space()
        elif self.action_space_type == ActionSpaceType.DISCRETE:
            if action_step_count is None:
                raise ValueError("Action step count must be provided for discrete action space")
            self._set_discrete_action_space(action_step_count)
        else:
            raise ValueError("Invalid action space type")

    def _set_continuous_action_space(self):
        bounds = self.model.get_actuator_ctrlrange().copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    
    def _set_discrete_action_space(self, action_step_count):
        # 将连续动作空间转换为离散动作空间
        self.ctrl = self.get_default_ctrl()
        self.ctrl_step = []
        self.ctrl_bounds = self.model.get_actuator_ctrlrange().copy().astype(np.float32)
        for b in self.ctrl_bounds:
            self.ctrl_step.append((b[1] - b[0]) / action_step_count)  # 每次操作的步长
        self.ctrl_step = np.array(self.ctrl_step)

        print("Ctrl Bounds: ", self.ctrl_bounds)
        print("Ctrl Step: ", self.ctrl_step)
        print("Ctrl: ", self.ctrl)

        self.action_space = spaces.MultiDiscrete([3] * len(self.ctrl_bounds))
        return self.action_space

    def _discrete_to_continuous(self, action):
        # 将离散动作 0, 1, 2 映射到 -1, 0, 1
        mapped_action = action - 1

        for i in range(len(mapped_action)):
            self.ctrl[i] += mapped_action[i] * self.ctrl_step[i]

        # 确保控制值在范围内
        self.ctrl = np.clip(self.ctrl, self.ctrl_bounds[:, 0], self.ctrl_bounds[:, 1])
        return self.ctrl

    def apply_action(self, action):
        if self.action_space_type == ActionSpaceType.CONTINUOUS:
            return action
        elif self.action_space_type == ActionSpaceType.DISCRETE:
            return self._discrete_to_continuous(action)

    # methods to override:
    # ----------------------------
    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]:
        raise NotImplementedError

    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def _initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]:
        """
        Initialize MuJoCo simulation data structures mjModel and mjData.
        """
        raise NotImplementedError

    def _step_orca_sim_simulation(self, ctrl, n_frames) -> None:
        """
        Step over the MuJoCo simulation.
        """
        raise NotImplementedError

    def render(self) -> Union[NDArray[np.float64], None]:
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        raise NotImplementedError

    # -----------------------------
    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.set_seed_value(seed)

        # mujoco.mj_resetData(self.model, self.data)
        self.reset_simulation()

        ob = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    def set_seed_value(self, seed=None):
        self.seed_value = seed
        self.np_random = np.random.RandomState(seed)
        return [seed]

    @property
    def dt(self) -> float:
        # return self.model.opt.timestep * self.frame_skip
        return self.gym.opt_config['timestep'] * self.frame_skip

    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_orca_sim_simulation(ctrl, n_frames)
        self.loop.run_until_complete(self.gym.update_data())

    def close(self):
        """Close all processes like rendering contexts"""
        raise NotImplementedError

    async def _initialize_grpc(self):
        """Initialize the GRPC communication channel."""
        raise NotImplementedError
    
    async def _pause_simulation(self):
        """Pause the simulation."""
        raise NotImplementedError

    async def _resume_simulation(self):
        """Resume the simulation."""
        raise NotImplementedError
    
    def _init_qpos_qvel(self):
        """Init qpos and qvel of the model."""
        raise NotImplementedError
    
    async def _reset_simulation(self):
        """Reset the simulation."""
        raise NotImplementedError
    
    def reset_simulation(self):
        """Reset the simulation."""
        raise NotImplementedError
    
    async def _query_actuator_ctrlrange(self):
        """Query the actuator control range."""
        raise NotImplementedError
    
    async def _set_time_step(self, time_step):
        """Set the time step of the simulation."""
        raise NotImplementedError

class OrcaGymEnv(BaseOrcaGymEnv):
    """Superclass for OrcaSim environments."""

    def __init__(
        self,
        frame_skip: int,
        grpc_address: str,
        agent_names: list[str],
        time_step: float,        
        observation_space: Space,
        action_space_type: Optional[ActionSpaceType],
        action_step_count: Optional[float],
        **kwargs        
    ):
        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,
            time_step = time_step,            
            observation_space = observation_space,
            action_space_type = action_space_type,
            action_step_count = action_step_count,
            **kwargs
        )

    def _initialize_simulation(
        self,
    ) -> Tuple[OrcaGymModel, OrcaGymData]:
        print(f"Initializing simulation: Class: {self.__class__.__name__}")
        self.loop.run_until_complete(self._initialize_orca_sim())
        model = self.gym.model
        data = self.gym.data
        return model, data

    def set_qpos_qvel(self, qpos, qvel):
        """Set the joints position qpos and velocity qvel of the model.

        Note: `qpos` and `qvel` is not the full physics state for all mujoco models/environments https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.loop.run_until_complete(self._set_qpos(qpos))
        self.loop.run_until_complete(self._set_qvel(qvel))
        self.loop.run_until_complete(self._mj_forward())

        # self.data.qpos[:] = np.copy(qpos)
        # self.data.qvel[:] = np.copy(qvel)
        # if self.model.na == 0:
        #     self.data.act[:] = None
        # mujoco.mj_forward(self.model, self.data)

    def _step_orca_sim_simulation(self, ctrl, n_frames):
        # self.data.ctrl[:] = ctrl

        self.loop.run_until_complete(self._set_ctrl(ctrl))
        self.loop.run_until_complete(self._mj_step(nstep=n_frames))
        # mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        # mujoco.mj_rnePostConstraint(self.model, self.data)

    def render(self):
        # Do nothing.
        return
    
    def body(self, name: str) -> str:
        if len(self._agent_names[0]) > 0:
            return f"{self._agent_names[0]}_{name}"
        else:
            return name
    
    def joint(self, name: str) -> str:
        if len(self._agent_names[0]) > 0:
            return f"{self._agent_names[0]}_{name}"
        else:
            return name
    
    def actuator(self, name: str) -> str:
        if len(self._agent_names[0]) > 0:
            return f"{self._agent_names[0]}_{name}"
        else:
            return name
    
    def site(self, name: str) -> str:
        if len(self._agent_names[0]) > 0:
            return f"{self._agent_names[0]}_{name}"
        else:
            return name
    
    def mocap(self, name: str) -> str:
        if len(self._agent_names[0]) > 0:
            return f"{self._agent_names[0]}_{name}"
        else:
            return name

    def sensor(self, name: str) -> str:
        return f"{self._agent_names[0]}_{name}"
    

    async def _close_grpc(self):
        if self.channel:
            await self.channel.close()

    def close(self):
        self.loop.run_until_complete(self._resume_simulation())  # 退出gym恢复仿真事件循环
        self.loop.run_until_complete(self._close_grpc())

    def get_body_com_dict(self, body_name_list) -> Dict[str, Dict[str, NDArray[np.float64]]]:
        body_com_dict = self.loop.run_until_complete(self._get_body_com_xpos_xmat(body_name_list))
        return body_com_dict

    def get_body_com_xpos_xmat(self, body_name_list) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        # return self.data.body(body_name).xpos
        body_com_dict = self.loop.run_until_complete(self._get_body_com_xpos_xmat(body_name_list))
        if len(body_com_dict) != len(body_name_list):
            raise ValueError("Some body names are not found in the simulation.")
        xpos = np.array([body_com_dict[body_name]['Pos'] for body_name in body_name_list]).flat.copy()
        xmat = np.array([body_com_dict[body_name]['Mat'] for body_name in body_name_list]).flat.copy()
        return xpos, xmat
    
    def get_body_com_xpos_xmat_list(self, body_name_list) -> Tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        body_com_dict = self.loop.run_until_complete(self._get_body_com_xpos_xmat(body_name_list))
        if len(body_com_dict) != len(body_name_list):
            raise ValueError("Some body names are not found in the simulation.")
        xpos_list = [np.array(body_com_dict[body_name]['Pos']) for body_name in body_name_list]
        xmat_list = [np.array(body_com_dict[body_name]['Mat']) for body_name in body_name_list]
        return xpos_list, xmat_list

    def get_body_xpos_xmat_xquat(self, body_name_list):
        # return self.data.body(body_name).xpos
        body_dict = self.loop.run_until_complete(self._get_body_xpos_xmat_xquat(body_name_list))
        if len(body_dict) != len(body_name_list):
            print("Body Nmae List: ", body_name_list)
            print("Body Dict: ", body_dict)
            raise ValueError("Some body names are not found in the simulation.")
        xpos = np.array([body_dict[body_name]['Pos'] for body_name in body_name_list]).flat.copy()
        xmat = np.array([body_dict[body_name]['Mat'] for body_name in body_name_list]).flat.copy()
        xquat = np.array([body_dict[body_name]['Quat'] for body_name in body_name_list]).flat.copy()
        return xpos, xmat, xquat
    
    def get_geom_xpos_xmat(self, geom_name_list):
        geom_dict = self.loop.run_until_complete(self._get_geom_xpos_xmat(geom_name_list))
        if len(geom_dict) != len(geom_name_list):
            raise ValueError("Some geom names are not found in the simulation.")
        xpos = np.array([geom_dict[geom_name]['Pos'] for geom_name in geom_name_list]).flat.copy()
        xmat = np.array([geom_dict[geom_name]['Mat'] for geom_name in geom_name_list]).flat.copy()
        return xpos, xmat

    async def _initialize_orca_sim(self):
        await self.gym.init_simulation()
        return

    async def _initialize_grpc(self):
        self.channel = orca_gym.grpc.aio.insecure_channel(self.grpc_address)
        self.stub = GrpcServiceStub(self.channel)
        self.gym = orca_gym.OrcaGym(self.stub)

    async def _pause_simulation(self):
        await self.gym.pause_simulation()

    async def _resume_simulation(self):
        await self.gym.resume_simulation()

    def _init_qpos_qvel(self):
        self.loop.run_until_complete(self.gym.update_data())
        self.init_qpos = self.gym.data.qpos.ravel().copy()
        self.init_qvel = self.gym.data.qvel.ravel().copy()

    def reset_simulation(self):
        self.loop.run_until_complete(self._reset_simulation())
        self.loop.run_until_complete(self.gym.update_data())

    async def _reset_simulation(self):
        await self.gym.load_initial_frame()

    def set_ctrl(self, ctrl):
        self.loop.run_until_complete(self._set_ctrl(ctrl))

    async def _set_ctrl(self, ctrl):
        await self.gym.set_ctrl(ctrl)

    async def _mj_step(self, nstep):
        await self.gym.mj_step(nstep)
    
    async def _set_qpos(self, qpos):
        await self.gym.set_qpos(qpos)

    async def _set_qvel(self, qvel):
        await self.gym.set_qvel(qvel)

    def mj_forward(self):
        self.loop.run_until_complete(self._mj_forward())
        
    async def _mj_forward(self):
        await self.gym.mj_forward()

    async def _get_body_com_xpos_xmat(self, body_name_list):
        body_com_dict = await self.gym.query_body_com_xpos_xmat(body_name_list)
        return body_com_dict
    
    async def _get_body_xpos_xmat_xquat(self, body_name_list):
        body_dict = await self.gym.query_body_xpos_xmat_xquat(body_name_list)
        return body_dict
    
    async def _get_geom_xpos_xmat(self, geom_name_list):
        geom_dict = await self.gym.query_geom_xpos_xmat(geom_name_list)
        return geom_dict
    
    async def _query_all_cfrc_ext(self):
        body_names = self.model.get_body_names()
        cfrc_ext_dict = await self.gym.query_cfrc_ext(body_names=body_names)
        return cfrc_ext_dict
        
    def set_time_step(self, time_step):
        self.loop.run_until_complete(self._set_time_step(time_step))
                                     
    async def _set_time_step(self, time_step):
        await self.gym.set_opt_timestep(time_step)
    
    async def _query_joint_qpos(self, joint_names):
        joint_qpos_dict = await self.gym.query_joint_qpos(joint_names)
        return joint_qpos_dict
    
    def query_joint_qpos(self, joint_names):
        joint_qpos_dict = self.loop.run_until_complete(self._query_joint_qpos(joint_names))
        return joint_qpos_dict
    
    async def _query_joint_qvel(self, joint_names):
        joint_qvel_dict = await self.gym.query_joint_qvel(joint_names)
        return joint_qvel_dict
    
    def query_joint_qvel(self, joint_names):
        joint_qvel_dict = self.loop.run_until_complete(self._query_joint_qvel(joint_names))
        return joint_qvel_dict
    
    async def _set_joint_qpos(self, joint_qpos_list):
        await self.gym.set_joint_qpos(joint_qpos_list)

    def set_joint_qpos(self, joint_qpos_list):
        self.loop.run_until_complete(self._set_joint_qpos(joint_qpos_list))

    async def _query_cfrc_ext(self, body_names):
        cfrc_ext_dict = await self.gym.query_cfrc_ext(body_names)
        return cfrc_ext_dict
    
    def query_cfrc_ext(self, body_names):
        cfrc_ext_dict = self.loop.run_until_complete(self._query_cfrc_ext(body_names))
        cfrc_ext_array = np.array([cfrc_ext_dict[body_name] for body_name in body_names])
        return cfrc_ext_dict, cfrc_ext_array
    
    async def _query_actuator_force(self):
        actuator_force = await self.gym.query_actuator_force()
        return actuator_force
    
    def query_actuator_force(self):
        actuator_force = self.loop.run_until_complete(self._query_actuator_force())
        return actuator_force
    
    def load_keyframe(self, keyframe_name):
        self.loop.run_until_complete(self._load_keyframe(keyframe_name))

    async def _load_keyframe(self, keyframe_name):
        await self.gym.load_keyframe(keyframe_name)

    def query_joint_limits(self, joint_names):
        joint_limit_dict = self.loop.run_until_complete(self._query_joint_limits(joint_names))
        return joint_limit_dict

    async def _query_joint_limits(self, joint_names):
        joint_limit_dict = await self.gym.query_joint_limits(joint_names)
        return joint_limit_dict 
    
    def query_body_velocities(self, body_names):
        body_velocity_dict = self.loop.run_until_complete(self._query_body_velocities(body_names))
        return body_velocity_dict

    async def _query_body_velocities(self, body_names):
        body_velocity_dict = await self.gym.query_body_velocities(body_names)
        return body_velocity_dict
    
    def query_actuator_gain_prm(self, actuator_names):
        actuator_gain_prm = self.loop.run_until_complete(self._query_actuator_gain_prm(actuator_names))
        return actuator_gain_prm

    async def _query_actuator_gain_prm(self, actuator_names):
        actuator_gain_prm = await self.gym.query_actuator_gain_prm(actuator_names)
        return actuator_gain_prm

    def set_actuator_gain_prm(self, gain_prm_set_list):
        self.loop.run_until_complete(self._set_actuator_gain_prm(gain_prm_set_list))

    async def _set_actuator_gain_prm(self, gain_prm_set_list):
        await self.gym.set_actuator_gain_prm(gain_prm_set_list)

    def query_actuator_bias_prm(self, actuator_names):
        actuator_bias_prm = self.loop.run_until_complete(self._query_actuator_bias_prm(actuator_names))
        return actuator_bias_prm

    async def _query_actuator_bias_prm(self, actuator_names):
        actuator_bias_prm = await self.gym.query_actuator_bias_prm(actuator_names)
        return actuator_bias_prm
    
    def set_actuator_bias_prm(self, bias_prm_set_list):
        self.loop.run_until_complete(self._set_actuator_bias_prm(bias_prm_set_list))

    async def _set_actuator_bias_prm(self, bias_prm_set_list):
        await self.gym.set_actuator_bias_prm(bias_prm_set_list)
    
    async def _query_mocap_pos_and_quat(self, mocap_body_names):
        mocap_pos_and_quat_dict = await self.gym.query_mocap_pos_and_quat(mocap_body_names)
        return mocap_pos_and_quat_dict
    
    def query_mocap_pos_and_quat(self, mocap_body_names):
        mocap_pos_and_quat_dict = self.loop.run_until_complete(self._query_mocap_pos_and_quat(mocap_body_names))
        return mocap_pos_and_quat_dict
    
    async def _set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict):
        await self.gym.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict):
        self.loop.run_until_complete(self._set_mocap_pos_and_quat(mocap_pos_and_quat_dict))

    async def _query_site_pos_and_mat(self, site_names):
        query_dict = await self.gym.query_site_pos_and_mat(site_names)
        return query_dict
    
    def query_site_pos_and_mat(self, site_names):
        query_dict = self.loop.run_until_complete(self._query_site_pos_and_mat(site_names))
        site_dict = {}
        for site in query_dict:
            site_dict[site] = {
                'xpos': np.array(query_dict[site]['xpos']),
                'xmat': np.array(query_dict[site]['xmat'])
            }
        return site_dict
    
    def query_site_pos_and_quat(self, site_names):
        query_dict = self.loop.run_until_complete(self._query_site_pos_and_mat(site_names))
        site_dict = {}
        for site in query_dict:
            site_dict[site] = {
                'xpos': np.array(query_dict[site]['xpos']),
                'xquat': mat2quat(np.array(query_dict[site]['xmat']).reshape(3, 3))
            }
        return site_dict
    
    async def _mj_jac_site(self, site_names):
        query_dict = await self.gym.mj_jac_site(site_names)
        return query_dict
    
    def query_site_xvalp_xvalr(self, site_names):
        query_dict = self.loop.run_until_complete(self._mj_jac_site(site_names))
        xvalp_dict = {}
        xvalr_dict = {}
        for site in query_dict:
            xvalp_dict[site] = np.array(query_dict[site]['jacp']).reshape(3, -1) @ self.data.qvel
            xvalr_dict[site] = np.array(query_dict[site]['jacr']).reshape(3, -1) @ self.data.qvel

        return xvalp_dict, xvalr_dict
    
    async def _update_equality_constraints(self, eq_list):
        await self.gym.update_equality_constraints(eq_list)

    def update_equality_constraints(self, eq_list):
        self.loop.run_until_complete(self._update_equality_constraints(eq_list))

    def query_all_geoms(self):
        geom_dict = self.loop.run_until_complete(self._query_all_geoms())
        return geom_dict

    async def _query_all_geoms(self):
        geom_dict = await self.gym.query_all_geoms()
        return geom_dict

    async def _query_opt_config(self):
        opt_config = await self.gym.query_opt_config()
        return opt_config
    
    def query_opt_config(self):
        opt_config = self.loop.run_until_complete(self._query_opt_config())
        return opt_config

    async def _set_opt_config(self, opt_config):
        await self.gym.set_opt_config(opt_config)

    def set_opt_config(self, opt_config):
        self.loop.run_until_complete(self._set_opt_config(opt_config))

    async def _query_contact_simple(self):
        contact_simple = await self.gym.query_contact_simple()
        return contact_simple
    
    def query_contact_simple(self):
        contact_simple = self.loop.run_until_complete(self._query_contact_simple())
        return contact_simple
    
    async def _query_contact(self):
        contact = await self.gym.query_contact()
        return contact
    
    def query_contact(self):
        contact = self.loop.run_until_complete(self._query_contact())
        return contact
    
    async def _query_contact_force(self, contact_ids):
        contact_force = await self.gym.query_contact_force(contact_ids)
        return contact_force
    
    def query_contact_force(self, contact_ids):
        contact_force = self.loop.run_until_complete(self._query_contact_force(contact_ids))
        return contact_force
    
    async def _mj_jac(self, body_point_list, compute_jacp=True, compute_jacr=True):
        jacp_list, jacr_list = await self.gym.mj_jac(body_point_list, compute_jacp, compute_jacr)
        return jacp_list, jacr_list
    
    def mj_jac(self, body_point_list, compute_jacp=True, compute_jacr=True):
        jacp_list, jacr_list = self.loop.run_until_complete(self._mj_jac(body_point_list, compute_jacp, compute_jacr))
        return jacp_list, jacr_list
    
    async def _calc_full_mass_matrix(self):
        mass_matrix = await self.gym.calc_full_mass_matrix()
        return mass_matrix
    
    def calc_full_mass_matrix(self):
        mass_matrix = self.loop.run_until_complete(self._calc_full_mass_matrix())
        return mass_matrix
    
    async def _query_qfrc_bias(self):
        qfrc_bias = await self.gym.query_qfrc_bias()
        return qfrc_bias
    
    def query_qfrc_bias(self):
        qfrc_bias = self.loop.run_until_complete(self._query_qfrc_bias())
        return qfrc_bias
    
    async def _query_subtree_com(self, body_name):
        subtree_com_dict = await self.gym.query_subtree_com(body_name)
        return subtree_com_dict
    
    def query_subtree_com(self, body_name):
        subtree_com_dict = self.loop.run_until_complete(self._query_subtree_com(body_name))
        return subtree_com_dict
    
    async def _set_geom_friction(self, geom_name_list, friction_list):
        await self.gym.set_geom_friction(geom_name_list, friction_list)

    def set_geom_friction(self, geom_name_list, friction_list):
        self.loop.run_until_complete(self._set_geom_friction(geom_name_list, friction_list))

    async def _query_sensor_data(self, sensor_names):
        sensor_data_dict = await self.gym.query_sensor_data(sensor_names)
        return sensor_data_dict        
    
    def query_sensor_data(self, sensor_names):
        sensor_data_dict = self.loop.run_until_complete(self._query_sensor_data(sensor_names))
        return sensor_data_dict
    
    async def _query_joint_offsets(self, joint_names):
        qpos_offsets, qvel_offsets, qacc_offsets = await self.gym.query_joint_offsets(joint_names)
        return qpos_offsets, qvel_offsets, qacc_offsets
    
    def query_joint_offsets(self, joint_names) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        qpos_offsets, qvel_offsets, qacc_offsets = self.loop.run_until_complete(self._query_joint_offsets(joint_names))
        return qpos_offsets, qvel_offsets, qacc_offsets