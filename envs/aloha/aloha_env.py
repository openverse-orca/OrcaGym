from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from numpy.typing import NDArray
import numpy as np
import time

START_ARM_POSE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.025,
    -0.025,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.025,
    -0.025,
]
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
        
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv        
        
        self._agent_name = agent_names[0]
        self._neutral_joint_values = np.array(START_ARM_POSE, dtype=np.float32).flatten()
        self._arm_gripper_joint_names = [self.joint("vx300s_left_waist"), self.joint("vx300s_left_shoulder"), self.joint("vx300s_left_elbow"), 
                                         self.joint("vx300s_left_forearm_roll"), self.joint("vx300s_left_wrist_angle"), self.joint("vx300s_left_wrist_rotate"),
                                         self.joint("vx300s_left_left_finger"), self.joint("vx300s_left_right_finger"),
                                         self.joint("vx300s_right_waist"), self.joint("vx300s_right_shoulder"), self.joint("vx300s_right_elbow"),
                                         self.joint("vx300s_right_forearm_roll"), self.joint("vx300s_right_wrist_angle"), self.joint("vx300s_right_wrist_rotate"),
                                         self.joint("vx300s_right_left_finger"), self.joint("vx300s_right_right_finger")]
        
        
    @property
    def agent_name(self):
        return self._agent_name

    def _set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self._arm_gripper_joint_names, self._neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos)
        
    def reset_model(self) -> tuple[dict, dict]:
        print("Aloha Env reset_model: ", START_ARM_POSE)
        
        self._set_joint_neutral()
        self.ctrl = np.array(self._neutral_joint_values).flatten()
        self.set_ctrl(self.ctrl)
        self.mj_forward()
        
        return {}, {}