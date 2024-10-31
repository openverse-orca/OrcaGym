import sys
import os
import grpc

current_dir = os.path.dirname(os.path.abspath(__file__))
proto_path = os.path.abspath(os.path.join(current_dir, "protos"))
sys.path.append(proto_path)
import mjc_message_pb2
import mjc_message_pb2_grpc


import numpy as np
import scipy.linalg
from datetime import datetime


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
        print("Opt config: ", 
              f"timestep:{self.opt.timestep}", 
              f"iterations:{self.opt.iterations}", 
              f"noslip_iterations:{self.opt.noslip_iterations}",
              f"mpr_iterations:{self.opt.mpr_iterations}",
              f"sdf_iterations:{self.opt.sdf_iterations}",
              f"gravity:{self.opt.gravity}",)
        
    def print_model_info(self, model_info):
        print("Model info: ", f"nq:{model_info['nq']}",
              f"nv:{model_info['nv']}",
              f"nu:{model_info['nu']}",
              f"nbody:{model_info['nbody']}",
              f"njnt:{model_info['njnt']}",
              f"ngeom:{model_info['ngeom']}",
              f"nsite:{model_info['nsite']}",
              )
