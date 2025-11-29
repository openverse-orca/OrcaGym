from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from numpy.typing import NDArray
import numpy as np
import time

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


START_ARM_POSE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
]

JOINT_FRIC_LOSS = {
    "bimanual_viperx_transfer_cube_usda_vx300s_left_waist": 50.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_left_shoulder": 60.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_left_elbow": 60.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_left_forearm_roll": 30.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_left_wrist_angle": 30.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_left_wrist_rotate": 30.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_left_left_finger": 30.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_left_right_finger": 30.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_right_waist": 50.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_right_shoulder": 60.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_right_elbow": 60.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_right_forearm_roll": 30.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_right_wrist_angle": 30.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_right_wrist_rotate": 30.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_right_left_finger": 30.0,
    "bimanual_viperx_transfer_cube_usda_vx300s_right_right_finger": 30.0,
}
    

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
        
        self.gym._mjModel.opt.apirate = 100.0
        self.gym._mjModel.opt.ccd_iterations = 50
        self.gym._mjModel.opt.ccd_tolerance = 1e-06
        self.gym._mjModel.opt.cone = 0
        self.gym._mjModel.opt.density = 0.0
        self.gym._mjModel.opt.disableactuator = 0
        self.gym._mjModel.opt.disableflags = 0
        self.gym._mjModel.opt.enableflags = 0
        self.gym._mjModel.opt.gravity = np.array([ 0.  ,  0.  , -9.81])
        self.gym._mjModel.opt.impratio = 1.0
        self.gym._mjModel.opt.integrator = 0
        self.gym._mjModel.opt.iterations = 100
        self.gym._mjModel.opt.jacobian = 2
        self.gym._mjModel.opt.ls_iterations = 50
        self.gym._mjModel.opt.ls_tolerance = 0.01
        self.gym._mjModel.opt.magnetic = np.array([ 0. , -0.5,  0. ])
        self.gym._mjModel.opt.noslip_iterations = 0
        self.gym._mjModel.opt.noslip_tolerance = 1e-06
        self.gym._mjModel.opt.o_friction = np.array([1.e+00, 1.e+00, 5.e-03, 1.e-04, 1.e-04])
        self.gym._mjModel.opt.o_margin = 0.0
        self.gym._mjModel.opt.o_solimp = np.array([9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00])
        self.gym._mjModel.opt.o_solref = np.array([0.02, 1.  ])
        self.gym._mjModel.opt.sdf_initpoints = 40
        self.gym._mjModel.opt.sdf_iterations = 10
        self.gym._mjModel.opt.solver = 2
        self.gym._mjModel.opt.timestep = 0.002
        self.gym._mjModel.opt.tolerance = 1e-08
        self.gym._mjModel.opt.viscosity = 0.0
        self.gym._mjModel.opt.wind = np.array([0., 0., 0.])
        
        
        self._agent_name = agent_names[0]
        self._neutral_joint_values = np.array(START_ARM_POSE, dtype=np.float32).flatten()
        self._arm_gripper_joint_names = [
            self.joint("vx300s_left_waist"), self.joint("vx300s_left_shoulder"), self.joint("vx300s_left_elbow"), 
            self.joint("vx300s_left_forearm_roll"), self.joint("vx300s_left_wrist_angle"), self.joint("vx300s_left_wrist_rotate"),
            self.joint("vx300s_left_left_finger"), self.joint("vx300s_left_right_finger"),
            self.joint("vx300s_right_waist"), self.joint("vx300s_right_shoulder"), self.joint("vx300s_right_elbow"),
            self.joint("vx300s_right_forearm_roll"), self.joint("vx300s_right_wrist_angle"), self.joint("vx300s_right_wrist_rotate"),
            self.joint("vx300s_right_left_finger"), self.joint("vx300s_right_right_finger")
        ]
        
        self._arm_gripper_body_name = [
            self.body("vx300s_left"), self.body("vx300s_left_shoulder_link"), self.body("vx300s_left_upper_arm_link"),
            self.body("vx300s_left_upper_forearm_link"), self.body("vx300s_left_lower_forearm_link"), self.body("vx300s_left_wrist_link"), self.body("vx300s_left_gripper_link"),
            self.body("vx300s_left_left_finger_link"), self.body("vx300s_left_right_finger_link"),
            self.body("vx300s_right"), self.body("vx300s_right_shoulder_link"), self.body("vx300s_right_upper_arm_link"),
            self.body("vx300s_right_upper_forearm_link"), self.body("vx300s_right_lower_forearm_link"), self.body("vx300s_right_wrist_link"), self.body("vx300s_right_gripper_link"),
            self.body("vx300s_right_left_finger_link"), self.body("vx300s_right_right_finger_link")            
        ]
        
        for joint_name in self._arm_gripper_joint_names:
            joint_id = self.model.joint_name2id(joint_name)
            joint = self.gym._mjModel.joint(joint_id)
            joint.frictionloss[0] = JOINT_FRIC_LOSS[joint_name]
        
        # for joint_name in self._arm_gripper_joint_names:
        #     joint = self.model.get_joint_byname(joint_name)
        #     print("joint: ", joint)
            
        # for body_name in self._arm_gripper_body_name:
        #     body = self.model.get_body_byname(body_name)
        #     print("body: ", body)
        
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
        _logger.info(f"Aloha Env reset_model:  {START_ARM_POSE}")
        
        self._set_joint_neutral()
        self.ctrl = np.array(self._neutral_joint_values).flatten()
        self.set_ctrl(self.ctrl)
        self.mj_forward()
        
        return {}, {}