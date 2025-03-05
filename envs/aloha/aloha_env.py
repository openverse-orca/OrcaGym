from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from numpy.typing import NDArray
import numpy as np
class AlohaEnv(OrcaGymLocalEnv):
    """
    Orca Gym Local Environment for Aloha 2 hand manipulation tasks.
    """

    def __init__(
        self,
        frame_skip: int,        
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        **kwargs,
    ):
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            **kwargs,
        )
        
    def reset_model(self) -> tuple[dict, dict]:
        print("AlohaEnv.reset_model")
        return {}, {}