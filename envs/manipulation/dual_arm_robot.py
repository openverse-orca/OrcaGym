import numpy as np
from orca_gym.utils import rotations
from orca_gym.adapters.robosuite.controllers.controller_factory import controller_factory
import orca_gym.adapters.robosuite.controllers.controller_config as controller_config
import orca_gym.adapters.robosuite.utils.transform_utils as transform_utils
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

    def _setup_initial_info(self):
        """
        读取系统初始化状态
        """
        # mujoco 模型初始化
        self.set_joint_neutral()
        self._env.mj_forward()
   
        # control range
        self._all_ctrlrange = self._env.model.get_actuator_ctrlrange()
        r_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._r_arm_actuator_id]
        l_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._l_arm_actuator_id]

        arm_qpos_range_l = self._env.model.get_joint_qposrange(self._l_arm_joint_names)
        arm_qpos_range_r = self._env.model.get_joint_qposrange(self._r_arm_joint_names)

        self._setup_action_range(l_ctrl_range, r_ctrl_range)
        self._setup_obs_scale(arm_qpos_range_l, arm_qpos_range_r)

        site_dict = self._env.query_site_pos_and_quat([self._ee_site_l])
        self._initial_grasp_site_xpos, self._initial_grasp_site_xquat = self._global_to_local(site_dict[self._ee_site_l]['xpos'], site_dict[self._ee_site_l]['xquat'])
        self._grasp_value_l = 0.0
        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)


        site_dict = self._env.query_site_pos_and_quat([self._ee_site_r])
        self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r = self._global_to_local(site_dict[self._ee_site_r]['xpos'], site_dict[self._ee_site_r]['xquat'])
        self._grasp_value_r = 0.0
        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)

        # 设置各部位初始位置
        self._env.mj_forward()

    def _setup_device(self):
        if self._env.run_mode == RunMode.TELEOPERATION:
            if self._env.ctrl_device == ControlDevice.VR:
                if self._env.joystick is None:
                    raise ValueError("VR controller is not initialized.")
                self._pico_joystick = self._env.joystick[self.name]
            else:
                raise ValueError("Invalid control device: ", self._env.ctrl_device)
            
    def _setup_controller(self):
        if self._env.action_use_motor():
            r_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._r_arm_actuator_id]
            l_ctrl_range = [self._all_ctrlrange[actoator_id] for actoator_id in self._l_arm_actuator_id]

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
            self._l_inverse_kinematics_controller = InverseKinematicsController(self._env, self._env.model.site_name2id(self._ee_site_l), self._l_jnt_dof, 2e-1, 0.075)
            self._r_inverse_kinematics_controller = InverseKinematicsController(self._env, self._env.model.site_name2id(self._ee_site_r), self._r_jnt_dof, 2e-1, 0.075)


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
        
    def set_l_arm_position_ctrl(self, mocap_xpos, mocap_xquat) -> np.ndarray:

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
        if self._pico_joystick is not None:
            grasp_l_xpos, grasp_l_xquat, grasp_r_xpos, grasp_r_xquat = self._processe_pico_joystick_move()
            self._process_pico_joystick_operation()
            # print("base_body_euler: ", self._base_body_euler / np.pi * 180)
        else:
            return np.zeros(14)
        
        # transform_utils 和 rotations 的quat不一样，这里将 qw, qx, qy, qz 转为 qx, qy, qz, qw
        grasp_r_axisangle = transform_utils.quat2axisangle(np.array([grasp_r_xquat[1], grasp_r_xquat[2], grasp_r_xquat[3], grasp_r_xquat[0]]))
        grasp_l_axisangle = transform_utils.quat2axisangle(np.array([grasp_l_xquat[1], grasp_l_xquat[2], grasp_l_xquat[3], grasp_l_xquat[0]]))

        grasp_l_xpos_global, grasp_l_xquat_global = self._local_to_global(grasp_l_xpos, grasp_l_xquat)
        grasp_r_xpos_global, grasp_r_xquat_global = self._local_to_global(grasp_r_xpos, grasp_r_xquat)
        grasp_r_axisangle_global = transform_utils.quat2axisangle(np.array([grasp_r_xquat_global[1], grasp_r_xquat_global[2], grasp_r_xquat_global[3], grasp_r_xquat_global[0]]))
        grasp_l_axisangle_global = transform_utils.quat2axisangle(np.array([grasp_l_xquat_global[1], grasp_l_xquat_global[2], grasp_l_xquat_global[3], grasp_l_xquat_global[0]]))

        # 手部控制器使用全局坐标系
        if self._env.action_use_motor():
            action_r = np.concatenate([grasp_r_xpos_global, grasp_r_axisangle_global])
            self._r_controller.set_goal(action_r)
            ctrl_r = self._r_controller.run_controller()
            self._set_arm_ctrl(self._r_arm_actuator_id, ctrl_r)

            action_l = np.concatenate([grasp_l_xpos_global, grasp_l_axisangle_global])
            self._l_controller.set_goal(action_l)
            ctrl_l = self._l_controller.run_controller()
            self._set_arm_ctrl(self._l_arm_actuator_id, ctrl_l)
        else:
            ctrl_r = self.set_r_arm_position_ctrl(grasp_r_xpos_global, grasp_r_xquat_global)
            ctrl_l = self.set_l_arm_position_ctrl(grasp_l_xpos_global, grasp_l_xquat_global)

        # 数据采集保存局部坐标系
        action_l_B = np.concatenate([grasp_l_xpos, grasp_l_axisangle])
        action_r_B = np.concatenate([grasp_r_xpos, grasp_r_axisangle])
        ctrl_l = np.asarray(ctrl_l, dtype=np.float32)
        ctrl_r = np.asarray(ctrl_r, dtype=np.float32)
        action = np.concatenate([
            np.asarray(action_l_B, dtype=np.float32),                # 0-5  : left eef pos and axisangle, normalized to [-1, 1] based on the coordinate action space [-2m, 2m] and rotation action space [-pi, pi]
            ctrl_l,                                                  # 6-12 : left arm joint pos (ik control mode) or torque (osc control mode) ,  normalized to [-1, 1] based on the pos or torque range
            np.array([self._grasp_value_l], dtype=np.float32),       # 13   : left hand grasp value, normalized to [-1, 1] based on the pos or torque range
            np.asarray(action_r_B, dtype=np.float32),                # 14-19: right eef pos and axisangle, normalized to [-1, 1] based on the coordinate action space [-2m, 2m] and rotation action space [-pi, pi]
            ctrl_r,                                                  # 20-26: right arm joint pos (ik control mode) or torque (osc control mode) ,  normalized to [-1, 1] based on the pos or torque range
            np.array([self._grasp_value_r], dtype=np.float32)        # 27   : right hand grasp value, normalized to [-1, 1] based on the pos or torque range
        ]).flatten()

        # Mocap 调试标记采用全局坐标系
        self.set_grasp_mocap(grasp_l_xpos_global, grasp_l_xquat_global)
        self.set_grasp_mocap_r(grasp_r_xpos_global, grasp_r_xquat_global)

        return action

    # def fill_arm_joint_pos(self, action : np.ndarray) -> np.ndarray:
    #     arm_joint_values_l = self._get_arm_joint_values(self._l_arm_joint_names)
    #     arm_joint_values_r = self._get_arm_joint_values(self._r_arm_joint_names)
    #     action[6:13] = arm_joint_values_l
    #     action[20:27] = arm_joint_values_r
    #     return action

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
        self.set_wheel_ctrl(joystick_state)
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


    def _processe_pico_joystick_move(self) -> tuple:
        if self._pico_joystick.is_reset_pos():
            self._pico_joystick.set_reset_pos(False)
            # self._set_init_state()
            self._reset_gripper()

        transform_list = self._pico_joystick.get_transform_list()
        if transform_list is None:
            return self._initial_grasp_site_xpos, self._initial_grasp_site_xquat, self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r

        left_relative_position, left_relative_rotation = self._pico_joystick.get_left_relative_move(transform_list)
        right_relative_position, right_relative_rotation = self._pico_joystick.get_right_relative_move(transform_list)

        grasp_l_xpos = self._initial_grasp_site_xpos + left_relative_position
        grasp_r_xpos = self._initial_grasp_site_xpos_r + right_relative_position

        grasp_l_xquat = rotations.quat_mul(self._initial_grasp_site_xquat, left_relative_rotation)
        grasp_r_xquat = rotations.quat_mul(self._initial_grasp_site_xquat_r, right_relative_rotation)

        return grasp_l_xpos, grasp_l_xquat, grasp_r_xpos, grasp_r_xquat

    def on_playback_action(self, action) -> np.ndarray:
        assert(len(action) == self.action_range.shape[0])
        
        self._grasp_value_l = action[13]
        self.set_l_hand_actuator_ctrl(self._grasp_value_l)
        self._grasp_value_r = action[27]
        self.set_r_hand_actuator_ctrl(self._grasp_value_r)

        if self._env.action_type == ActionType.END_EFFECTOR_OSC:
            action_l = self._action_B_to_action(action[:6])
            self._l_controller.set_goal(action_l)
            ctrl = self._l_controller.run_controller()
            self._set_arm_ctrl(self._l_arm_actuator_id, ctrl)

            action_r = self._action_B_to_action(action[14:20])
            self._r_controller.set_goal(action_r)
            ctrl = self._r_controller.run_controller()
            self._set_arm_ctrl(self._r_arm_actuator_id, ctrl)

            action = self.fill_arm_ctrl(action)

        elif self._env.action_type == ActionType.END_EFFECTOR_IK:
            action_l = self._action_B_to_action(action[:6])
            quat = transform_utils.axisangle2quat(action_l[3:6])
            action_l_xquat = np.array([quat[3], quat[0], quat[1], quat[2]])  # 转换为 wxyz 格式
            ctrl_l = self.set_l_arm_position_ctrl(action_l[:3], action_l_xquat)

            action_r = self._action_B_to_action(action[14:20])
            quat = transform_utils.axisangle2quat(action_r[3:6])
            action_r_xquat = np.array([quat[3], quat[0], quat[1], quat[2]])  # 转换为 wxyz 格式
            ctrl_r = self.set_r_arm_position_ctrl(action_r[:3], action_r_xquat)

            self._set_arm_ctrl(self._l_arm_actuator_id, ctrl_l)
            self._set_arm_ctrl(self._r_arm_actuator_id, ctrl_r)

            action = self.fill_arm_ctrl(action)

        elif self._env.action_type in [ActionType.JOINT_POS, ActionType.JOINT_MOTOR]:
            l_arm_joint_action = action[6:13]
            self._set_arm_ctrl(self._l_arm_actuator_id, l_arm_joint_action)

            r_arm_joint_action = action[20:27]
            self._set_arm_ctrl(self._r_arm_actuator_id, r_arm_joint_action)
        else:
            raise ValueError("Invalid action type: ", self._env._action_type)
        
        return action
    

    def _local_to_global(self, local_pos: np.ndarray, local_quat: np.ndarray) -> tuple:
        """
        将局部坐标系的位姿转换为全局坐标系
        :param local_pos: 局部位置
        :param local_quat: 局部四元数
        :return: 全局位置和全局四元数
        """
        base_link_pos, _, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)

        global_pos = base_link_pos + rotations.quat_rot_vec(base_link_quat, local_pos)
        global_quat = rotations.quat_mul(base_link_quat, local_quat)
        return global_pos, global_quat
    
    def _global_to_local(self, global_pos: np.ndarray, global_quat: np.ndarray) -> tuple:
        """
        将全局坐标系的位姿转换为局部坐标系
        :param global_pos: 全局位置
        :param global_quat: 全局四元数
        :return: 局部位置和局部四元数
        """
        base_link_pos, _, base_link_quat = self._env.get_body_xpos_xmat_xquat(self._base_body_name)

        base_link_quat_inv = rotations.quat_conjugate(base_link_quat)
        local_pos = rotations.quat_rot_vec(base_link_quat_inv, global_pos - base_link_pos)
        local_quat = rotations.quat_mul(base_link_quat_inv, global_quat)
        return local_pos, local_quat

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
    
    def set_wheel_ctrl(self, joystick_state) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")

    def set_wheel_actuator_ctrl(self, offset_rate) -> None:
        raise NotImplementedError("This method should be implemented in the derived class.")    


    