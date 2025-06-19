import numpy as np
from orca_gym.utils import rotations
from orca_gym.robosuite.controllers.controller_factory import controller_factory
import orca_gym.robosuite.controllers.controller_config as controller_config
import orca_gym.robosuite.utils.transform_utils as transform_utils
from scipy.spatial.transform import Rotation as R, Rotation
from envs.manipulation.dual_arm_env import DualArmEnv, AgentBase, RunMode, ControlDevice, ActionType, TaskStatus

from orca_gym.utils.inverse_kinematics_controller import InverseKinematicsController

from envs.manipulation.robots.configs.openloong_config import openloong_config
robot_config = {
    "openloong_hand_fix_base" : openloong_config,
    "openloong_gripper_2f85_fix_base" : openloong_config,
    "openloong_gripper_2f85_mobile_base" : openloong_config,
}
def get_robot_config(robot_name: str):
    for key in robot_config.keys():
        if key in robot_name:
            return robot_config[key]
        
    raise ValueError(f"Robot configuration for {robot_name} not found in robot_config dictionary.")

class DualArmRobot(AgentBase):
    def __init__(self, env: DualArmEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)

    def init_agent(self, id: int):
        config = get_robot_config(self._name)
        self._read_config(config, id)
        self._setup_initial_info()
        self._setup_device()
        self._setup_controller()


    def _read_config(self, config: dict, id: int) -> None:
        """
        根据配置初始化访问环境需要的名字和 ID
        注意：xpos, xquat 不能在这里读取，因为还没有执行 mj_forward，读到的是不正确的
        """
        ######## Base body and joint setup ########
        self._base_body_name = [self._env.body(config["base"]["base_body_name"], id)]

        # print("base_body_xpos: ", self._base_body_xpos)
        # print("base_body_xquat: ", self._base_body_xquat)

        dummy_joint_id = self._env.model.joint_name2id(self._env.joint(config["base"]["dummy_joint_name"], id))

        # ######## Right Arm ########
        self._r_arm_joint_names = [self._env.joint(config["right_arm"]["joint_names"][i], id) for i in range(len(config["right_arm"]["joint_names"]))]
        self._r_arm_joint_id = [self._env.model.joint_name2id(joint_name) for joint_name in self._r_arm_joint_names]
        self._r_jnt_address = [self._env.jnt_qposadr(joint_name) for joint_name in self._r_arm_joint_names]
        self._r_jnt_dof = [self._env.jnt_dofadr(joint_name) for joint_name in self._r_arm_joint_names]

        self._r_arm_motor_names = [self._env.actuator(config["right_arm"]["motor_names"][i], id) for i in range(len(config["right_arm"]["motor_names"]))]
        self._r_arm_position_names = [self._env.actuator(config["right_arm"]["position_names"][i], id) for i in range(len(config["right_arm"]["position_names"]))]
        if self._env.action_use_motor():
            self._env.disable_actuators(self._r_arm_position_names, dummy_joint_id)
            self._r_arm_actuator_id = [self._env.model.actuator_name2id(actuator_name) for actuator_name in self._r_arm_motor_names]
        else:
            self._env.disable_actuators(self._r_arm_motor_names, dummy_joint_id)
            self._r_arm_actuator_id = [self._env.model.actuator_name2id(actuator_name) for actuator_name in self._r_arm_position_names]
        self._r_neutral_joint_values = np.array(config["right_arm"]["neutral_joint_values"])
        self._ee_site_r  = self._env.site(config["right_arm"]["ee_center_site_name"], id)

        # ######## Left Arm ########
        self._l_arm_joint_names = [self._env.joint(config["left_arm"]["joint_names"][i], id) for i in range(len(config["left_arm"]["joint_names"]))]
        self._l_arm_joint_id = [self._env.model.joint_name2id(joint_name) for joint_name in self._l_arm_joint_names]
        self._l_jnt_address = [self._env.jnt_qposadr(joint_name) for joint_name in self._l_arm_joint_names]
        self._l_jnt_dof = [self._env.jnt_dofadr(joint_name) for joint_name in self._l_arm_joint_names]

        self._l_arm_motor_names = [self._env.actuator(config["left_arm"]["motor_names"][i], id) for i in range(len(config["left_arm"]["motor_names"]))]
        self._l_arm_position_names = [self._env.actuator(config["left_arm"]["position_names"][i], id) for i in range(len(config["left_arm"]["position_names"]))]
        if self._env.action_use_motor():
            self._env.disable_actuators(self._l_arm_position_names, dummy_joint_id)
            self._l_arm_actuator_id = [self._env.model.actuator_name2id(actuator_name) for actuator_name in self._l_arm_motor_names]
        else:
            self._env.disable_actuators(self._l_arm_motor_names, dummy_joint_id)
            self._l_arm_actuator_id = [self._env.model.actuator_name2id(actuator_name) for actuator_name in self._l_arm_position_names]
        self._l_neutral_joint_values = np.array(config["left_arm"]["neutral_joint_values"])
        self._ee_site_l  = self._env.site(config["left_arm"]["ee_center_site_name"], id)

        # ######## neck setup ########
        if config.get("neck", None) is None:
            self._neck_joint_names = [self._env.joint(config["neck"]["yaw_joint_name"], id), self._env.joint(config["neck"]["pitch_joint_name"], id)]
            self._neck_actuator_names = [self._env.actuator(config["neck"]["yaw_actuator_name"], id), self._env.actuator(config["neck"]["pitch_actuator_name"], id)]
            self._neck_actuator_id = [self._env.model.actuator_name2id(actuator_name) for actuator_name in self._neck_actuator_names]
            self._neck_neutral_joint_values = np.array([0, -0.7854])
            self._neck_ctrl_values = {"yaw": 0.0, "pitch": -0.7854}
            self._neck_site_name  = self._env.site(config["neck"]["neck_center_site_name"], id)
        else:
            self._neck_site_name = None
            self._neck_joint_names = []
            self._neck_actuator_names = []
            self._neck_neutral_joint_values = np.array([0, 0])

    def _setup_initial_info(self):
        """
        读取系统初始化状态
        """
        # mujoco 模型初始化
        self._env.mj_forward()
        self.set_joint_neutral()
        self._env.mj_forward()

        self._base_body_xpos, _, self._base_body_xquat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)        
        # control range
        self._all_ctrlrange = self._env.model.get_actuator_ctrlrange()

        r_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._r_arm_actuator_id]
        # print("ctrl_range: ", r_ctrl_range)

        l_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._l_arm_actuator_id]
        # print("ctrl_range: ", l_ctrl_range)

        arm_qpos_range_l = self._env.model.get_joint_qposrange(self._l_arm_joint_names)
        arm_qpos_range_r = self._env.model.get_joint_qposrange(self._r_arm_joint_names)

        self._setup_action_range(l_ctrl_range, r_ctrl_range)
        self._setup_obs_scale(arm_qpos_range_l, arm_qpos_range_r)

        site_dict = self._env.query_site_pos_and_quat([self._ee_site_l])
        self._initial_grasp_site_xpos = site_dict[self._ee_site_l]['xpos']
        self._initial_grasp_site_xquat = site_dict[self._ee_site_l]['xquat']
        self._grasp_value_l = 0.0
        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)


        site_dict = self._env.query_site_pos_and_quat([self._ee_site_r])
        self._initial_grasp_site_xpos_r = site_dict[self._ee_site_r]['xpos']
        self._initial_grasp_site_xquat_r = site_dict[self._ee_site_r]['xquat']
        self._grasp_value_r = 0.0
        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)

        if self._neck_site_name is not None:
            site_dict = self._env.query_site_pos_and_quat([self._neck_site_name])
            self._initial_neck_site_xpos = site_dict[self._neck_site_name]['xpos']
            self._initial_neck_site_xquat = site_dict[self._neck_site_name]['xquat']
            self.set_neck_mocap(self._initial_neck_site_xpos, self._initial_neck_site_xquat)
            self._mocap_neck_xpos, self._mocap_neck_xquat = self._initial_neck_site_xpos, self._initial_neck_site_xquat
            self._neck_angle_x, self._neck_angle_y = 0, 0

        # 设置各部位初始位置
        self._env.mj_forward()

    def _setup_device(self):
        if self._env.run_mode == RunMode.TELEOPERATION:
            if self._env.ctrl_device == ControlDevice.VR:
                self._pico_joystick = self._env.joystick[self.name]
            else:
                raise ValueError("Invalid control device: ", self._env.ctrl_device)
            
    def _setup_controller(self):
        # -----------------------------
        # Neck controller
        if self._neck_site_name is not None:
            neck_ctrl_range = [self._all_ctrlrange[actuator_id] for actuator_id in self._neck_actuator_id]
            # print("ctrl_range: ", neck_ctrl_range)

            self._neck_controller_config = controller_config.load_config("osc_pose")
            # print("controller_config: ", self.controller_config)

            # Add to the controller dict additional relevant params:
            #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
            #   policy (control) freq, and ndim (# joints)
            self._neck_controller_config["robot_name"] = self.name
            self._neck_controller_config["sim"] = self._env.gym
            self._neck_controller_config["eef_name"] = self._neck_site_name
            # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
            qpos_offsets, qvel_offsets, _ = self._env.query_joint_offsets(self._neck_joint_names)
            self._neck_controller_config["joint_indexes"] = {
                "joints": self._neck_joint_names,
                "qpos": qpos_offsets,
                "qvel": qvel_offsets,
            }
            self._neck_controller_config["actuator_range"] = neck_ctrl_range
            self._neck_controller_config["policy_freq"] = self._env.control_freq
            self._neck_controller_config["ndim"] = len(self._neck_joint_names)
            self._neck_controller_config["control_delta"] = False


            self._neck_controller = controller_factory(self._neck_controller_config["type"], self._neck_controller_config)
            self._neck_controller.update_initial_joints(self._neck_neutral_joint_values)
        else:
            self._neck_controller = None

        if self._env.action_use_motor():
            # -----------------------------
            # OSC controller
            # Right controller
            self._r_controller_config = controller_config.load_config("osc_pose")
            # print("controller_config: ", self._r_controller_config)

            # Add to the controller dict additional relevant params:
            #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
            #   policy (control) freq, and ndim (# joints)
            self._r_controller_config["robot_name"] = self.name
            self._r_controller_config["sim"] = self._env.gym
            self._r_controller_config["eef_name"] = self._ee_site_r
            # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
            qpos_offsets, qvel_offsets, _ = self._env.query_joint_offsets(self._r_arm_joint_names)
            self._r_controller_config["joint_indexes"] = {
                "joints": self._r_arm_joint_names,
                "qpos": qpos_offsets,
                "qvel": qvel_offsets,
            }
            self._r_controller_config["actuator_range"] = r_ctrl_range
            self._r_controller_config["policy_freq"] = self._env.control_freq
            self._r_controller_config["ndim"] = len(self._r_arm_joint_names)
            self._r_controller_config["control_delta"] = False


            self._r_controller = controller_factory(self._r_controller_config["type"], self._r_controller_config)
            self._r_controller.update_initial_joints(self._r_neutral_joint_values)

            self._r_gripper_offset_rate_clip = 0.0


            # -----------------------------
            # Left controller
            self._l_controller_config = controller_config.load_config("osc_pose")
            # print("controller_config: ", self.controller_config)

            # Add to the controller dict additional relevant params:
            #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
            #   policy (control) freq, and ndim (# joints)
            self._l_controller_config["robot_name"] = self.name
            self._l_controller_config["sim"] = self._env.gym
            self._l_controller_config["eef_name"] = self._ee_site_l
            # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
            qpos_offsets, qvel_offsets, _ = self._env.query_joint_offsets(self._l_arm_joint_names)
            self._l_controller_config["joint_indexes"] = {
                "joints": self._l_arm_joint_names,
                "qpos": qpos_offsets,
                "qvel": qvel_offsets,
            }
            self._l_controller_config["actuator_range"] = l_ctrl_range
            self._l_controller_config["policy_freq"] = self._env.control_freq
            self._l_controller_config["ndim"] = len(self._l_arm_joint_names)
            self._l_controller_config["control_delta"] = False

            self._l_controller = controller_factory(self._l_controller_config["type"], self._l_controller_config)
            self._l_controller.update_initial_joints(self._l_neutral_joint_values)

            self._l_gripper_offset_rate_clip = 0.0
        else:
            # -----------------------------
            # Inverse Kinematics controller
            self._l_inverse_kinematics_controller = InverseKinematicsController(self._env, self._env.model.site_name2id(self._ee_site_l), self._l_jnt_dof, 1e-2, 0.1)
            self._r_inverse_kinematics_controller = InverseKinematicsController(self._env, self._env.model.site_name2id(self._ee_site_r), self._r_jnt_dof, 1e-2, 0.1)


    def on_close(self):
        if hasattr(self, "_pico_joystick") and self._pico_joystick is not None:
            self._pico_joystick.close()        

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos = {}
        for name, value in zip(self._r_arm_joint_names, self._r_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        for name, value in zip(self._l_arm_joint_names, self._l_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])     
        for name, value in zip(self._neck_joint_names, self._neck_neutral_joint_values):
            arm_joint_qpos[name] = np.array([value])
        self._env.set_joint_qpos(arm_joint_qpos)        

    def set_init_ctrl(self) -> None:
        if self._env.action_use_motor():
            return
        for i in range(len(self._r_arm_actuator_id)):
            self._env.ctrl[self._r_arm_actuator_id[i]] = self._r_neutral_joint_values[i]

        for i in range(len(self._l_arm_actuator_id)):
            self._env.ctrl[self._l_arm_actuator_id[i]] = self._l_neutral_joint_values[i]

    def on_reset_model(self) -> None:
        self._reset_grasp_mocap()
        self._reset_gripper()
        self._reset_neck_mocap()

    def set_neck_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self._env.mocap("neckMocap", self.id): {'pos': position, 'quat': orientation}}
        self._env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self._env.mocap("leftHandMocap", self.id): {'pos': position, 'quat': orientation}}
        self._env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_grasp_mocap_r(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self._env.mocap("rightHandMocap", self.id): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        self._env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def _reset_gripper(self) -> None:
        self._l_gripper_offset_rate_clip = 0.0
        self._r_gripper_offset_rate_clip = 0.0

    def _reset_neck_mocap(self) -> None:
        if self._neck_controller is None:
            return
        
        self._mocap_neck_xpos, self._mocap_neck_xquat = self._initial_neck_site_xpos, self._initial_neck_site_xquat
        self.set_neck_mocap(self._mocap_neck_xpos, self._mocap_neck_xquat)
        self._neck_angle_x, self._neck_angle_y = 0, 0

    def _reset_grasp_mocap(self) -> None:
        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)
        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)

    def get_obs(self) -> dict:
        ee_sites = self._env.query_site_pos_and_quat_B([self._ee_site_l, self._ee_site_r], self._base_body_name)
        ee_xvalp, ee_xvalr = self._env.query_site_xvalp_xvalr_B([self._ee_site_l, self._ee_site_r], self._base_body_name)

        arm_joint_values_l = self._get_arm_joint_values(self._l_arm_joint_names)
        arm_joint_values_r = self._get_arm_joint_values(self._r_arm_joint_names)
        arm_joint_velocities_l = self._get_arm_joint_velocities(self._l_arm_joint_names)
        arm_joint_velocities_r = self._get_arm_joint_velocities(self._r_arm_joint_names)

        self._obs = {
            "ee_pos_l": ee_sites[self._ee_site_l]["xpos"].flatten().astype(np.float32),
            "ee_quat_l": ee_sites[self._ee_site_l]["xquat"].flatten().astype(np.float32),
            "ee_pos_r": ee_sites[self._ee_site_r]["xpos"].flatten().astype(np.float32),
            "ee_quat_r": ee_sites[self._ee_site_r]["xquat"].flatten().astype(np.float32),

            "ee_vel_linear_l": ee_xvalp[self._ee_site_l].flatten().astype(np.float32),
            "ee_vel_angular_l": ee_xvalr[self._ee_site_l].flatten().astype(np.float32),
            "ee_vel_linear_r": ee_xvalp[self._ee_site_r].flatten().astype(np.float32),
            "ee_vel_angular_r": ee_xvalr[self._ee_site_r].flatten().astype(np.float32),

            "arm_joint_qpos_l": arm_joint_values_l.flatten().astype(np.float32),
            "arm_joint_qpos_sin_l": np.sin(arm_joint_values_l).flatten().astype(np.float32),
            "arm_joint_qpos_cos_l": np.cos(arm_joint_values_l).flatten().astype(np.float32),
            "arm_joint_vel_l": arm_joint_velocities_l.flatten().astype(np.float32),

            "arm_joint_qpos_r": arm_joint_values_r.flatten().astype(np.float32),
            "arm_joint_qpos_sin_r": np.sin(arm_joint_values_r).flatten().astype(np.float32),
            "arm_joint_qpos_cos_r": np.cos(arm_joint_values_r).flatten().astype(np.float32),
            "arm_joint_vel_r": arm_joint_velocities_r.flatten().astype(np.float32),

            "grasp_value_l": np.array([self._grasp_value_l], dtype=np.float32),
            "grasp_value_r": np.array([self._grasp_value_r], dtype=np.float32),
        }
        scaled_obs = {key : self._obs[key] * self._obs_scale[key] for key in self._obs.keys()}
        return scaled_obs

    
    def _get_arm_joint_values(self, joint_names) -> np.ndarray:
        qpos_dict = self._env.query_joint_qpos(joint_names)
        return np.array([qpos_dict[joint_name] for joint_name in joint_names]).flatten()
    
    def _get_arm_joint_velocities(self, joint_names) -> np.ndarray:
        qvel_dict = self._env.query_joint_qvel(joint_names)
        return np.array([qvel_dict[joint_name] for joint_name in joint_names]).flatten()

    def _setup_obs_scale(self, arm_qpos_range_l, arm_qpos_range_r) -> None:
        # 观测空间范围
        ee_xpos_scale = np.array([max(abs(act_range[0]), abs(act_range[1])) for act_range in self._action_range[:3]], dtype=np.float32)   # 末端位置范围
        ee_xquat_scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)   # 裁剪到 -pi, pi 的单位四元数范围
        max_ee_linear_vel = 2.0  # 末端线速度范围 m/s
        max_ee_angular_vel = np.pi # 末端角速度范围 rad/s

        arm_qpos_scale_l = np.array([max(abs(qpos_range[0]), abs(qpos_range[1])) for qpos_range in arm_qpos_range_l], dtype=np.float32)  # 关节角度范围
        arm_qpos_scale_r = np.array([max(abs(qpos_range[0]), abs(qpos_range[1])) for qpos_range in arm_qpos_range_r], dtype=np.float32)  # 关节角度范围
        max_arm_joint_vel = np.pi  # 关节角速度范围 rad/s
                
        self._obs_scale = {
            "ee_pos_l": 1.0 / ee_xpos_scale,
            "ee_quat_l": 1.0 / ee_xquat_scale,
            "ee_pos_r": 1.0 / ee_xpos_scale,
            "ee_quat_r": 1.0 / ee_xquat_scale,

            "ee_vel_linear_l": np.ones(3, dtype=np.float32) / max_ee_linear_vel,
            "ee_vel_angular_l": np.ones(3, dtype=np.float32) / max_ee_angular_vel,
            "ee_vel_linear_r": np.ones(3, dtype=np.float32) / max_ee_linear_vel,
            "ee_vel_angular_r": np.ones(3, dtype=np.float32) / max_ee_angular_vel,

            "arm_joint_qpos_l": 1.0 / arm_qpos_scale_l,
            "arm_joint_qpos_sin_l": np.ones(len(arm_qpos_scale_l), dtype=np.float32),
            "arm_joint_qpos_cos_l": np.ones(len(arm_qpos_scale_l), dtype=np.float32),
            "arm_joint_vel_l": np.ones(len(arm_qpos_scale_l), dtype=np.float32) / max_arm_joint_vel,

            "arm_joint_qpos_r": 1.0 / arm_qpos_scale_r,
            "arm_joint_qpos_sin_r": np.ones(len(arm_qpos_scale_r), dtype=np.float32),
            "arm_joint_qpos_cos_r": np.ones(len(arm_qpos_scale_r), dtype=np.float32),
            "arm_joint_vel_r": np.ones(len(arm_qpos_scale_r), dtype=np.float32) / max_arm_joint_vel,

            "grasp_value_l": np.ones(1, dtype=np.float32),
            "grasp_value_r": np.ones(1, dtype=np.float32),       
        }

    def _setup_action_range(self, arm_ctrl_range_l, arm_ctrl_range_r) -> None:
        # 支持的动作范围空间，遥操作时不能超过这个范围
        # 模型接收的是 [-1, 1] 的动作空间，这里是真实的物理空间，需要进行归一化
        self._action_range =  np.concatenate(
            [
                [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], # left hand ee pos and angle euler
                arm_ctrl_range_l,                                                                           # left arm auctuator ctrl
                [[-1.0, 0.0]],                                                                                # left hand grasp value
                [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], # right hand ee pos and angle euler
                arm_ctrl_range_r,                                                                           # right arm auctuator ctrl
                [[-1.0, 0.0]],                                                                                # right hand grasp value
            ],
            dtype=np.float32,
            axis=0
        )

        self._action_range_min = self._action_range[:, 0]
        self._action_range_max = self._action_range[:, 1]
        
    def set_l_arm_position_ctrl(self, mocap_xpos, mocap_xquat) -> None:

        self._l_inverse_kinematics_controller.set_goal(mocap_xpos, mocap_xquat)
        delta = self._l_inverse_kinematics_controller.compute_inverse_kinematics()
        # print("delta: ", delta, "\n", "delta size: ", delta.size)
        # print("jnt_address: ", self._l_jnt_address)

        for i in range(len(self._l_arm_actuator_id)):
            self._env.ctrl[self._l_arm_actuator_id[i]] += delta[self._l_jnt_dof[i]]
            # print(f"ctrl {i}: ", self._env.ctrl[self._l_arm_actuator_id[i]], "error: ", delta[self._l_jnt_address[i]])

        self._env.ctrl = np.clip(self._env.ctrl, self._all_ctrlrange[:, 0], self._all_ctrlrange[:, 1])
        return self._env.ctrl[self._l_arm_actuator_id]


    def set_r_arm_position_ctrl(self, mocap_xpos, mocap_xquat):

        self._r_inverse_kinematics_controller.set_goal(mocap_xpos, mocap_xquat)
        delta = self._r_inverse_kinematics_controller.compute_inverse_kinematics()
        # print("delta: ", delta, "\n", "delta size: ", delta.size)
        # print("jnt_address: ", self._r_jnt_address)

        for i in range(len(self._r_arm_actuator_id)):
            self._env.ctrl[self._r_arm_actuator_id[i]] += delta[self._r_jnt_dof[i]]
            # print(f"ctrl {i}: ", self._env.ctrl[self._r_arm_actuator_id[i]], "error: ", delta[self._r_arm_joint_address[i]])

        self._env.ctrl = np.clip(self._env.ctrl, self._all_ctrlrange[:, 0], self._all_ctrlrange[:, 1])
        # print("ctrl: ", self._env.ctrl)
        return self._env.ctrl[self._r_arm_actuator_id]

    def on_teleoperation_action(self) -> np.ndarray:
        mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = None, None, None, None

        if self._pico_joystick is not None:
            mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = self._processe_pico_joystick_move()
            self.set_grasp_mocap(mocap_l_xpos, mocap_l_xquat)
            self.set_grasp_mocap_r(mocap_r_xpos, mocap_r_xquat)
            self._process_pico_joystick_operation()
            # print("base_body_euler: ", self._base_body_euler / np.pi * 180)
        else:
            return np.zeros(14)


        # 两个工具的quat不一样，这里将 qw, qx, qy, qz 转为 qx, qy, qz, qw
        mocap_r_axisangle = transform_utils.quat2axisangle(np.array([mocap_r_xquat[1], 
                                                                   mocap_r_xquat[2], 
                                                                   mocap_r_xquat[3], 
                                                                   mocap_r_xquat[0]]))              
        # mocap_axisangle[1] = -mocap_axisangle[1]
        action_r = np.concatenate([mocap_r_xpos, mocap_r_axisangle])

        if self._env.action_use_motor():
            # print("action r:", action_r)
            self._r_controller.set_goal(action_r)
            ctrl_r = self._r_controller.run_controller()
            # print("ctrl r: ", ctrl)
            self._set_arm_ctrl(self._r_arm_actuator_id, ctrl_r)
        else:
            ctrl_r = self.set_r_arm_position_ctrl(mocap_r_xpos, mocap_r_xquat)

        mocap_l_axisangle = transform_utils.quat2axisangle(np.array([mocap_l_xquat[1], 
                                                                   mocap_l_xquat[2], 
                                                                   mocap_l_xquat[3], 
                                                                   mocap_l_xquat[0]]))  
        action_l = np.concatenate([mocap_l_xpos, mocap_l_axisangle])

        if self._env.action_use_motor():
            # print("action l:", action_l)
            # print(action)
            self._l_controller.set_goal(action_l)
            ctrl_l = self._l_controller.run_controller()
            # print("ctrl l: ", ctrl)
            self._set_arm_ctrl(self._l_arm_actuator_id, ctrl_l)
        else:
            ctrl_l = self.set_l_arm_position_ctrl(mocap_l_xpos, mocap_l_xquat)


        action_l_B = self._action_to_action_B(action_l)
        action_r_B = self._action_to_action_B(action_r)

        action = np.concatenate([action_l_B,                # left eef pos and angle, 0-5
                                 ctrl_l,               # left arm joint pos, 6-12 (will be fill after do simulation)
                                 [self._grasp_value_l],     # left hand grasp value, 13
                                 action_r_B,                # right eef pos and angle, 14-19
                                 ctrl_r,               # right arm joint pos, 20-26 (will be fill after do simulation)
                                 [self._grasp_value_r]]     # right hand grasp value, 27
                                ).flatten()

        return action

    def fill_arm_joint_pos(self, action : np.ndarray) -> np.ndarray:
        arm_joint_values_l = self._get_arm_joint_values(self._l_arm_joint_names)
        arm_joint_values_r = self._get_arm_joint_values(self._r_arm_joint_names)
        action[6:13] = arm_joint_values_l
        action[20:27] = arm_joint_values_r
        return action

    def fill_arm_ctrl(self, action : np.ndarray) -> np.ndarray:
        ctrl_l = self._env.ctrl[self._l_arm_actuator_id]
        ctrl_r = self._env.ctrl[self._r_arm_actuator_id]
        action[6:13] = ctrl_l
        action[20:27] = ctrl_r
        return action

    def _set_arm_ctrl(self, arm_actuator_id, ctrl) -> None:
        for i in range(len(arm_actuator_id)):
            self._env.ctrl[arm_actuator_id[i]] = ctrl[i]


    def _process_pico_joystick_operation(self) -> None:
        joystick_state = self._pico_joystick.get_key_state()
        if joystick_state is None:
            return
        
        # print("Josytick state: ", joystick_state)

        self.set_gripper_ctrl_r(joystick_state)
        self.set_gripper_ctrl_l(joystick_state)
        self._set_head_ctrl(joystick_state)
        self._set_task_status(joystick_state)



    def _set_task_status(self, joystick_state) -> None:
        if self.id != 0:
            # Just for the first agent's controller
            return

        if self._env.task_status == TaskStatus.NOT_STARTED and joystick_state["leftHand"]["gripButtonPressed"]:
            self._env.set_task_status(TaskStatus.GET_READY)
        elif self._env.task_status == TaskStatus.GET_READY and not joystick_state["leftHand"]["gripButtonPressed"]:
            self._env.set_task_status(TaskStatus.BEGIN)
        elif self._env.task_status == TaskStatus.BEGIN and joystick_state["leftHand"]["gripButtonPressed"]:
            self._env.set_task_status(TaskStatus.FAILURE)
        elif self._env.task_status == TaskStatus.BEGIN and joystick_state["rightHand"]["gripButtonPressed"]:
            self._env.set_task_status(TaskStatus.SUCCESS)

    def _set_head_ctrl(self, joystick_state) -> None:
        if self._neck_controller is None:
            return

        x_axis = joystick_state["rightHand"]["joystickPosition"][0]
        if x_axis == 0:
            x_axis = joystick_state["leftHand"]["joystickPosition"][0]

        y_axis = joystick_state["rightHand"]["joystickPosition"][1]
        if y_axis == 0:
            y_axis = joystick_state["leftHand"]["joystickPosition"][1]
            
        mocap_neck_xpos, mocap_neck_xquat = self._mocap_neck_xpos, self._mocap_neck_xquat

        # 将 x_axis 和 y_axis 输入转换为旋转角度，按需要调节比例系数
        angle_x = -x_axis * np.pi / 180  # 转换为弧度，模拟绕 X 轴的旋转
        angle_y = -y_axis * np.pi / 180  # 转换为弧度，模拟绕 Y 轴的旋转

        # 设置旋转角度的限制
        self._neck_angle_x += angle_x
        if self._neck_angle_x > np.pi / 3 or self._neck_angle_x < -np.pi / 3:
            self._neck_angle_x = np.clip(self._neck_angle_x, -np.pi / 3, np.pi / 3)
            angle_x = 0
        
        self._neck_angle_y += angle_y
        if self._neck_angle_y > np.pi / 3 or self._neck_angle_y < -np.pi / 3:
            self._neck_angle_y = np.clip(self._neck_angle_y, -np.pi / 3, np.pi / 3)
            angle_y = 0

        new_neck_quat_local = rotations.euler2quat(np.array([0.0, angle_y, angle_x]))

        # 将局部坐标系的旋转转换为全局坐标系，乘以当前全局旋转四元数
        new_neck_quat_global = rotations.quat_mul(mocap_neck_xquat, new_neck_quat_local)

        # 将新的全局旋转四元数转换为轴角表示
        mocap_neck_axisangle = transform_utils.quat2axisangle(np.array([new_neck_quat_global[1], 
                                                                        new_neck_quat_global[2],
                                                                        new_neck_quat_global[3],
                                                                        new_neck_quat_global[0]]))

        # 可选：将轴角重新转换回四元数进行夹紧或其他操作
        new_neck_quat_cliped = transform_utils.axisangle2quat(mocap_neck_axisangle)

        # 将动作信息打包并发送到控制器
        action_neck = np.concatenate([mocap_neck_xpos, mocap_neck_axisangle])

        # # 更新 _mocap_neck_xquat 为新的全局旋转值
        self._mocap_neck_xquat = new_neck_quat_global

        self._neck_controller.set_goal(action_neck)
        ctrl = self._neck_controller.run_controller()
        for i in range(len(self._neck_actuator_id)):
            self._env.ctrl[self._neck_actuator_id[i]] = ctrl[i]

        # 更新头部位置
        self.set_neck_mocap(mocap_neck_xpos, self._mocap_neck_xquat)

    def _processe_pico_joystick_move(self) -> tuple:
        if self._pico_joystick.is_reset_pos():
            self._pico_joystick.set_reset_pos(False)
            # self._set_init_state()
            self._reset_gripper()
            self._reset_neck_mocap()

        transform_list = self._pico_joystick.get_transform_list()
        if transform_list is None:
            return self._initial_grasp_site_xpos, self._initial_grasp_site_xquat, self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r

        left_relative_position, left_relative_rotation = self._pico_joystick.get_left_relative_move(transform_list)
        right_relative_position, right_relative_rotation = self._pico_joystick.get_right_relative_move(transform_list)

        mocap_l_xpos = self._initial_grasp_site_xpos + rotations.quat_rot_vec(self._base_body_xquat, left_relative_position)
        mocap_r_xpos = self._initial_grasp_site_xpos_r + rotations.quat_rot_vec(self._base_body_xquat, right_relative_position)

        mocap_l_xquat = rotations.quat_mul(self._initial_grasp_site_xquat, left_relative_rotation)
        mocap_r_xquat = rotations.quat_mul(self._initial_grasp_site_xquat_r, right_relative_rotation)
        # mocap_r_xquat = (R.from_quat(self._initial_grasp_site_xquat_r, scalar_first=True) * 
        #                  R.from_quat(right_relative_rotation, scalar_first=True)).as_quat(scalar_first=True, canonical=True)
        
        return mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat

    def on_playback_action(self, action) -> None:
        assert(len(action) == self.action_range.shape[0])
        
        self._grasp_value_l = action[13]
        self.set_l_hand_actuator_ctrl(self._grasp_value_l)
        self._grasp_value_r = action[27]
        self.set_r_hand_actuator_ctrl(self._grasp_value_r)

        if self._env.action_type in [ActionType.END_EFFECTOR_OSC, ActionType.END_EFFECTOR_IK]:
            action_l = self._action_B_to_action(action[:6])
            self._l_controller.set_goal(action_l)
            ctrl = self._l_controller.run_controller()
            self._set_arm_ctrl(self._l_arm_actuator_id, ctrl)

            action_r = self._action_B_to_action(action[14:20])
            self._r_controller.set_goal(action_r)
            ctrl = self._r_controller.run_controller()
            self._set_arm_ctrl(self._r_arm_actuator_id, ctrl)

        elif self._env.action_type == ActionType.JOINT_POS:
            l_arm_joint_action = action[6:13]
            self._set_arm_ctrl(self._l_arm_actuator_id, l_arm_joint_action)

            r_arm_joint_action = action[20:27]
            self._set_arm_ctrl(self._r_arm_actuator_id, r_arm_joint_action)
        else:
            raise ValueError("Invalid action type: ", self._env._action_type)
        
        return
    
    def _action_B_to_action(self, action_B: np.ndarray) -> np.ndarray:
        ee_pos = action_B[:3]
        ee_axisangle = action_B[3:6]

        base_link_pos, base_link_xmat, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)

        # 在h5文件中的数据是B系数据，需要转换到世界坐标系

        base_link_rot = R.from_quat(base_link_quat[[1, 2, 3, 0]])
        ee_pos_global = base_link_pos + base_link_rot.apply(ee_pos)

        ee_quat = transform_utils.axisangle2quat(ee_axisangle)
        ee_rot = R.from_quat(ee_quat)
        ee_rot_global = base_link_rot * ee_rot

        ee_axisangle_global = transform_utils.quat2axisangle(ee_rot_global.as_quat())
        return np.concatenate([ee_pos_global, ee_axisangle_global], dtype=np.float32).flatten()

    def _action_to_action_B(self, action_global: np.ndarray) -> np.ndarray:
        ee_pos_global = action_global[:3]
        ee_axisangle_global = action_global[3:6]

        # 获取基础链接的全局位姿
        base_link_pos, _, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)
        
        # 处理四元数顺序（假设环境返回wxyz格式）
        quat_xyzw = base_link_quat[[1, 2, 3, 0]]  # 转换为xyzw格式
        base_rot = R.from_quat(quat_xyzw)
        base_rot_matrix = base_rot.as_matrix()

        # 位置转换（全局→局部）
        pos_local = base_rot_matrix.T @ (ee_pos_global - base_link_pos)

        # 旋转转换（全局→局部）
        global_quat = transform_utils.axisangle2quat(ee_axisangle_global)
        global_rot = R.from_quat(global_quat)
        local_rot = base_rot.inv() * global_rot  # 旋转的逆运算
        local_axisangle = transform_utils.quat2axisangle(local_rot.as_quat())

        return np.concatenate([pos_local, local_axisangle], dtype=np.float32).flatten()

    def set_gripper_ctrl_r(self, joystick_state) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")
    
    def set_gripper_ctrl_l(self, joystick_state) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")

    def update_force_feedback(self) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")

    def set_l_hand_actuator_ctrl(self, offset_rate) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")
    
    def set_r_hand_actuator_ctrl(self, offset_rate) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")    


    