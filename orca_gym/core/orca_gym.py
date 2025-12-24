import sys
import os
import grpc

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proto_path = os.path.abspath(os.path.join(proj_dir, "protos"))
sys.path.append(proto_path)
import mjc_message_pb2
import mjc_message_pb2_grpc

import numpy as np
import scipy.linalg
from datetime import datetime

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class OrcaGymBase:
    """
    OrcaGymBase class
    """
    def __init__(self, stub):
        self.stub = stub
        self.model = None
        self.opt = None
        self.data = None

    async def pause_simulation(self):
        request = mjc_message_pb2.SetSimulationStateRequest(state=mjc_message_pb2.PAUSED)
        response = await self.stub.SetSimulationState(request)
        return response


    def print_opt_config(self):
        _logger.info(f"Opt config: timestep:{self.opt.timestep}, iterations:{self.opt.iterations}, noslip_iterations:{self.opt.noslip_iterations}, ccd_iterations:{self.opt.ccd_iterations}, sdf_iterations:{self.opt.sdf_iterations}, gravity:{self.opt.gravity}")
        
    def print_model_info(self, model_info):
        _logger.info(f"Model info: nq:{model_info['nq']}, nv:{model_info['nv']}, nu:{model_info['nu']}, nbody:{model_info['nbody']}, njnt:{model_info['njnt']}, ngeom:{model_info['ngeom']}, nsite:{model_info['nsite']}")

    async def set_qpos(self, qpos):
        request = mjc_message_pb2.SetQposRequest(qpos=qpos)
        response = await self.stub.SetQpos(request)
        return response

    async def mj_forward(self):
        request = mjc_message_pb2.MJ_ForwardRequest()
        response = await self.stub.MJ_Forward(request)
        return response

    async def mj_inverse(self):
        request = mjc_message_pb2.MJ_InverseRequest()
        response = await self.stub.MJ_Inverse(request)
        return response
    
    async def mj_step(self, nstep):
        request = mjc_message_pb2.MJ_StepRequest(nstep=nstep)
        response = await self.stub.MJ_Step(request)
        return response    
    
    async def set_qvel(self, qvel):
        request = mjc_message_pb2.SetQvelRequest(qvel=qvel)
        response = await self.stub.SetQvel(request)
        return response    