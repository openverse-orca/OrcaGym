import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import asyncio
import sys
import time
import grpc

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym import OrcaGymModel, OrcaGymData
from orca_gym.core.orca_gym_warp import OrcaGymWarp
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

from orca_gym.utils.rotations import mat2quat, quat2mat


class OrcaGymWarpEnv(OrcaGymLocalEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        nworld: int = 1,
        **kwargs
    ):
        self._nworld = nworld
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs
        )

    def initialize_grpc(self):
        self.channel = grpc.aio.insecure_channel(
            self.orcagym_addr,
            options=[
                ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ]
        )
        self.stub = GrpcServiceStub(self.channel)
        self.gym = OrcaGymWarp(self.stub, nworld=self._nworld)

    async def _load_model_xml(self):
        model_xml_path = await self.gym.load_model_xml()
        return model_xml_path

    async def _init_orca_sim(self, model_xml_path):
        await self.gym.init_simulation(model_xml_path)

    def initialize_simulation(self) -> Tuple[OrcaGymModel, OrcaGymData]:
        model_xml_path = self.loop.run_until_complete(self._load_model_xml())
        self.loop.run_until_complete(self._init_orca_sim(model_xml_path))

        model = self.gym.model
        data = self.gym.data
        return model, data

    def _step_orca_sim_simulation(self, ctrl, n_frames) -> None:
        if ctrl is not None:
            self.gym.set_ctrl(ctrl)
        self.gym.mj_step(nstep=n_frames)
        self.gym.update_data()

    def do_simulation(self, ctrl, n_frames) -> None:
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_orca_sim_simulation(ctrl, n_frames)

    def close(self):
        self.loop.run_until_complete(self._close_grpc())

    async def _close_grpc(self):
        if self.channel:
            await self.channel.close()