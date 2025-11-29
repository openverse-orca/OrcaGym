from os import path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space

import asyncio
import sys

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

from orca_gym.core import orca_gym
from orca_gym import OrcaGymRemote
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
from orca_gym.utils.rotations import mat2quat, quat2mat

from orca_gym import OrcaGymModel
from orca_gym import OrcaGymData
from . import OrcaGymBaseEnv

import grpc

class OrcaGymRemoteEnv(OrcaGymBaseEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}
        
    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,        
        **kwargs        
    ):
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs
        )

    def initialize_simulation(
        self,
    ) -> Tuple[OrcaGymModel, OrcaGymData]:
        _logger.info(f"Initializing simulation: Class: {self.__class__.__name__}")
        self.loop.run_until_complete(self._initialize_orca_sim())
        model = self.gym.model
        data = self.gym.data
        return model, data

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


    def get_observation(self, obs=None):
        """
        Return the current environment observation as a dictionary, unless obs is not None.
        This function should process the raw environment observation to align with the input expected by the policy model.
        For example, it should cast an image observation to float with value range 0-1 and shape format [C, H, W].
        """
        raise NotImplementedError

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
            _logger.error(f"Body Name List: {body_name_list}")
            _logger.error(f"Body Dict: {body_dict}")
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

    def initialize_grpc(self):
        self.loop.run_until_complete(self._initialize_grpc())

    async def _initialize_grpc(self):
        self.channel = grpc.aio.insecure_channel(self.orcagym_addr)
        self.stub = GrpcServiceStub(self.channel)
        self.gym = OrcaGymRemote(self.stub)

    def pause_simulation(self):
        self.loop.run_until_complete(self._pause_simulation())

    async def _pause_simulation(self):
        await self.gym.pause_simulation()

    async def _resume_simulation(self):
        await self.gym.resume_simulation()

    def init_qpos_qvel(self):
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
        self.time_step = time_step
        self.realtime_step = time_step * self.frame_skip
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
    
    async def _set_joint_qpos(self, joint_qpos):
        await self.gym.set_joint_qpos(joint_qpos)

    def set_joint_qpos(self, joint_qpos):
        self.loop.run_until_complete(self._set_joint_qpos(joint_qpos))

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

    async def _set_opt_config(self, opt_config: dict):
        await self.gym.set_opt_config(opt_config)

    def set_opt_config(self):
        opt_config = {
            "timestep": self.gym.opt.timestep,
            "apirate": self.gym.opt.apirate,
            "impratio": self.gym.opt.impratio,
            "tolerance": self.gym.opt.tolerance,
            "ls_tolerance": self.gym.opt.ls_tolerance,
            "noslip_tolerance": self.gym.opt.noslip_tolerance,
            "ccd_tolerance": self.gym.opt.ccd_tolerance,
            "gravity": self.gym.opt.gravity,
            "wind": self.gym.opt.wind,
            "magnetic": self.gym.opt.magnetic,
            "density": self.gym.opt.density,
            "viscosity": self.gym.opt.viscosity,
            "o_margin": self.gym.opt.o_margin,
            "o_solref": self.gym.opt.o_solref,
            "o_solimp": self.gym.opt.o_solimp,
            "o_friction": self.gym.opt.o_friction,
            "integrator": self.gym.opt.integrator,
            "cone": self.gym.opt.cone,
            "jacobian": self.gym.opt.jacobian,
            "solver": self.gym.opt.solver,
            "iterations": self.gym.opt.iterations,
            "ls_iterations": self.gym.opt.ls_iterations,
            "noslip_iterations": self.gym.opt.noslip_iterations,
            "ccd_iterations": self.gym.opt.ccd_iterations,
            "disableflags": self.gym.opt.disableflags,
            "enableflags": self.gym.opt.enableflags,
            "disableactuator": self.gym.opt.disableactuator,
            "sdf_initpoints": self.gym.opt.sdf_initpoints,
            "sdf_iterations": self.gym.opt.sdf_iterations
        }
        self.loop.run_until_complete(self._set_opt_config(opt_config))

    async def _query_contact_simple(self):
        contact_simple =  await self.gym.query_contact_simple()
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