import os
from .panda_env import FrankaEnv

class FrankaReachEnv(FrankaEnv):
    def __init__(
        self,
        frame_skip: int,                
        reward_type: str,
        grpc_address: str,
        agent_names: list[str],
        time_step: float,  
        **kwargs,
    ):
        super().__init__(
            frame_skip=frame_skip,
            grpc_address=grpc_address,
            agent_names=agent_names,
            time_step=time_step,
            reward_type=reward_type,
            has_object=False,
            block_gripper=True,
            distance_threshold=0.05,
            goal_xy_range=0.5,
            obj_xy_range=0.3,
            goal_x_offset=0.0,
            goal_z_range=0.3,
            **kwargs,
        )
