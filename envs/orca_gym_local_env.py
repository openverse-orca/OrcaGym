from os import path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space

import asyncio
import sys
from orca_gym import OrcaGymRemote, OrcaGymLocal
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
from orca_gym.utils.rotations import mat2quat, quat2mat

from orca_gym import OrcaGymModel
from orca_gym import OrcaGymData
from envs import OrcaGymBaseEnv, RewardType

import grpc

import mujoco


class OrcaGymLocalEnv(OrcaGymBaseEnv):
    def __init__(
        self,
        frame_skip: int,
        grpc_address: str,
        agent_names: list[str],
        time_step: float,        
        **kwargs        
    ):
        super().__init__(
            frame_skip = frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs
        )

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
        self.channel = grpc.aio.insecure_channel(self.grpc_address)
        self.stub = GrpcServiceStub(self.channel)
        self.gym = OrcaGymLocal(self.stub)

    def pause_simulation(self):
        self.loop.run_until_complete(self._pause_simulation())

    async def _pause_simulation(self):
        await self.gym.pause_simulation()

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