import os
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union

current_file_path = os.path.abspath('')
project_root = os.path.dirname(current_file_path)

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)

import asyncio
import nest_asyncio
nest_asyncio.apply()

from orca_gym import orca_gym
from orca_gym.orca_gym import OrcaGym
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
from orca_gym.devices.xbox_joystick import XboxJoystick



class GameRuntime:
    def __init__(self, grpc_address: str, agent_names: list[str], time_step: float, **kwargs):
        self._running = False
        self._paused = False
        self.agent_names = agent_names
        self.grpc_address = grpc_address
        self.time_step = time_step

        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._initialize_grpc())
        self.loop.run_until_complete(self._set_time_step(self.time_step))  # 设置仿真时间步长
        self.model, self.data = self._initialize_simulation()
        self.loop.run_until_complete(self._reset_simulation()) # 重置仿真环境       
         
        
        self.joystick = XboxJoystick()
        

    def _main_loop(self):
        while self._running:
            time_start = time.time()

            if not self._paused:
                self.joystick.update()
                self._update()
            else:
                pass

            time_end = time.time()
            time_elapsed = time_end - time_start
            time.sleep(max(0, self.time_step - time_elapsed))

    def run(self):
        self._running = True
        self._paused = False
        self._main_loop()

    def _pause(self):
        self._paused = True

    def _resume(self):
        self._paused = False

    def _stop(self):
        self._running = False
        self._paused = False

    def _update(self):
        pass


    async def _initialize_grpc(self):
        self.channel = orca_gym.grpc.aio.insecure_channel(self.grpc_address)
        self.stub = GrpcServiceStub(self.channel)
        self.gym = orca_gym.OrcaGym(self.stub)

    async def _set_time_step(self, time_step):
        await self.gym.set_opt_timestep(time_step)        

    def _initialize_simulation(
        self,
    ) -> Tuple["OrcaGym.model", "OrcaGym.data"]:
        print(f"Initializing simulation: Class: OrcaGymEnv")
        self.loop.run_until_complete(self._initialize_orca_sim())
        model = self.gym.model
        data = None # data 是远端异步数据，只支持实时查询和设置，没有本地副本
        return model, data        
    
    async def _reset_simulation(self):
        await self.gym.load_initial_frame()

    async def _initialize_orca_sim(self):
        await self.gym.init_simulation()
        return        