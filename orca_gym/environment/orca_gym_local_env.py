from os import path
from typing import Any, Dict, Optional, Tuple, Union, SupportsFloat

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space
from gymnasium.core import ObsType

import asyncio
import sys
from orca_gym import OrcaGymLocal
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
from orca_gym.utils.rotations import mat2quat, quat2mat

from orca_gym import OrcaGymModel
from orca_gym import OrcaGymData
from . import OrcaGymBaseEnv

import grpc

from datetime import datetime
import time

class OrcaGymLocalEnv(OrcaGymBaseEnv):
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

        render_fps = 30
        self._render_interval = 1.0 / render_fps
        self._render_time_step = time.perf_counter()


    def initialize_simulation(
        self,
    ) -> Tuple[OrcaGymModel, OrcaGymData]:
        print(f"Initializing simulation: Class: {self.__class__.__name__}")
        self.loop.run_until_complete(self._initialize_orca_sim())
        model = self.gym.model
        data = self.gym.data
        return model, data

    async def _initialize_orca_sim(self):
        await self.gym.init_simulation()
        return

    def initialize_grpc(self):
        self.channel = grpc.aio.insecure_channel(self.orcagym_addr)
        self.stub = GrpcServiceStub(self.channel)
        self.gym = OrcaGymLocal(self.stub)
    
    def pause_simulation(self):
        self.loop.run_until_complete(self._pause_simulation())

    async def _pause_simulation(self):
        await self.gym.pause_simulation()

    async def _close_grpc(self):
        if self.channel:
            await self.channel.close()

    def close(self):
        self.loop.run_until_complete(self._close_grpc())

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
        self.gym.update_data()


    @property
    def render_mode(self) -> str:
        if hasattr(self, "_render_mode"):
            return self._render_mode
        else:
            return "human"
        
    @property
    def render_remote(self) -> bool:
        if hasattr(self, "_render_remote"):
            return self._render_remote
        else:
            return False

    def render(self):
        time_diff = time.perf_counter() - self._render_time_step
        if (time_diff > self._render_interval):
            self._render_time_step = time.perf_counter()
            if self.render_mode == "human" or self.render_mode == "force":
                self.loop.run_until_complete(self.gym.render())
            

    def set_ctrl(self, ctrl):
        self.gym.set_ctrl(ctrl)

    def mj_step(self, nstep):
        self.gym.mj_step(nstep)

    def mj_forward(self):
        self.gym.mj_forward()

    def _step_orca_sim_simulation(self, ctrl, n_frames):
        self.set_ctrl(ctrl)
        self.mj_step(nstep=n_frames)

    def set_time_step(self, time_step):
        self.gym.set_time_step(time_step)
        return

    def reset_simulation(self):
        self.gym.load_initial_frame()
        self.gym.update_data()

    def init_qpos_qvel(self):
        self.gym.update_data()
        self.init_qpos = self.gym.data.qpos.ravel().copy()
        self.init_qvel = self.gym.data.qvel.ravel().copy()

    def query_joint_offsets(self, joint_names) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        qpos_offsets, qvel_offsets, qacc_offsets = self.gym.query_joint_offsets(joint_names)
        return qpos_offsets, qvel_offsets, qacc_offsets
    
    def query_joint_lengths(self, joint_names):
        qpos_lengths, qvel_lengths, qacc_lengths = self.gym.query_joint_lengths(joint_names)
        return qpos_lengths, qvel_lengths, qacc_lengths
    
    def get_body_xpos_xmat_xquat(self, body_name_list):
        body_dict = self.gym.query_body_xpos_xmat_xquat(body_name_list)
        if len(body_dict) != len(body_name_list):
            print("Body Nmae List: ", body_name_list)
            print("Body Dict: ", body_dict)
            raise ValueError("Some body names are not found in the simulation.")
        xpos = np.array([body_dict[body_name]['Pos'] for body_name in body_name_list]).flat.copy()
        xmat = np.array([body_dict[body_name]['Mat'] for body_name in body_name_list]).flat.copy()
        xquat = np.array([body_dict[body_name]['Quat'] for body_name in body_name_list]).flat.copy()
        return xpos, xmat, xquat
    
    def query_sensor_data(self, sensor_names):
        sensor_data_dict = self.gym.query_sensor_data(sensor_names)
        return sensor_data_dict
    
    def query_joint_qpos(self, joint_names):
        joint_qpos_dict = self.gym.query_joint_qpos(joint_names)
        return joint_qpos_dict
    
    def query_joint_qvel(self, joint_names):
        joint_qvel_dict = self.gym.query_joint_qvel(joint_names)
        return joint_qvel_dict
    
    def query_joint_qacc(self, joint_names):
        joint_qacc_dict = self.gym.query_joint_qacc(joint_names)
        return joint_qacc_dict
    
    def jnt_qposadr(self, joint_name):
        joint_qposadr = self.gym.jnt_qposadr(joint_name)
        return joint_qposadr
    
    def jnt_dofadr(self, joint_name):
        joint_dofadr = self.gym.jnt_dofadr(joint_name)
        return joint_dofadr
        
    def query_site_pos_and_mat(self, site_names):
        query_dict = self.gym.query_site_pos_and_mat(site_names)
        site_dict = {}
        for site in query_dict:
            site_dict[site] = {
                'xpos': np.array(query_dict[site]['xpos']),
                'xmat': np.array(query_dict[site]['xmat'])
            }
        return site_dict
    
    def query_site_pos_and_quat(self, site_names):
        query_dict = self.gym.query_site_pos_and_mat(site_names)
        site_dict = {}
        for site in query_dict:
            site_dict[site] = {
                'xpos': np.array(query_dict[site]['xpos']),
                'xquat': mat2quat(np.array(query_dict[site]['xmat']).reshape(3, 3))
            }
        return site_dict

    def set_joint_qpos(self, joint_qpos):
        self.gym.set_joint_qpos(joint_qpos)

    
    def query_site_xvalp_xvalr(self, site_names):
        query_dict = self.gym.mj_jac_site(site_names)
        xvalp_dict = {}
        xvalr_dict = {}
        for site in query_dict:
            xvalp_dict[site] = np.array(query_dict[site]['jacp']).reshape(3, -1) @ self.data.qvel
            xvalr_dict[site] = np.array(query_dict[site]['jacr']).reshape(3, -1) @ self.data.qvel

        return xvalp_dict, xvalr_dict        
    
    def update_equality_constraints(self, eq_list):
        self.gym.update_equality_constraints(eq_list)

    def set_mocap_pos_and_quat(self, mocap_pos_and_quat_dict):
        send_remote = self.render_mode == "human" and self.render_remote
        self.loop.run_until_complete(self.gym.set_mocap_pos_and_quat(mocap_pos_and_quat_dict, send_remote))

    def query_contact_simple(self):
        return self.gym.query_contact_simple()