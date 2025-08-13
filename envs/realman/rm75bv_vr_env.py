import numpy as np
from gymnasium.core import ObsType
#from envs.robot_env import MujocoRobotEnv
from envs import OrcaGymRemoteEnv
from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystickManager
from scipy.spatial.transform import Rotation as R
from orca_gym.adapters.robosuite.controllers.controller_factory import controller_factory
import orca_gym.adapters.robosuite.controllers.controller_config as controller_config
import orca_gym.adapters.robosuite.utils.transform_utils as transform_utils
from orca_gym.devices.pico_joytsick import PicoJoystick
from envs.realman_rm65b.robot_controller import RobotController, DeviceType
from envs.realman_rm65b.roh_registers import *

class GripperState:
    OPENNING = "openning"
    CLOSING = "closing"
    STOPPED = "stopped"
    
class RM75BVVREnv(OrcaGymRemoteEnv):
    """
    通过xbox手柄控制机械臂
    """
    def __init__(
        self,
        frame_skip: int = 5,        
        orcagym_addr: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps        
        control_freq: int = 20,        
        **kwargs,
    ):

        action_size = 3 # 实际并不使用

        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,            
            n_actions=action_size,
            observation_space = None,
            **kwargs,
        )
        self.use_controller = False
        
        self.control_freq = control_freq
        if self.use_controller:
            self.robot_controller_l = RobotController("169.254.128.18", DeviceType.RM75)
            self.robot_controller_r = RobotController("169.254.128.19", DeviceType.RM75)
       # self._neutral_ljoint_values = np.array([1.83, 1.75, 2.95, -0.78, 1.06, 0.758,0])
       # self._neutral_rjoint_values = np.array([1.83, 1.75, 2.95, -0.78, 1.06, 0.758,0])
      #  self._neutral_ljoint_values = np.array([-0.435, -0.636, 0, 0, 0, 0, 0])
       # self._neutral_ljoint_values = np.array([-0.777, -1.34, 1.52, -1.42, -0.093, 0.535, 0])
      #  self._neutral_ljoint_values = np.array([1.87, 1.02, 2.05, -0.33, 0.591, -1,0.314])
       # self._neutral_rjoint_values = np.array([-1.87, -1.02, -2.02, 0.33, -0.591, 1,0.314])
       # self._neutral_ljoint_values = np.array([2.33, 1.68, 1.83, -0.33, 0.591, -1.11,-2.5])
        self._neutral_ljoint_values = np.array([2.27, 1.86, 1.37, -0.33, 0.591, -1.34,-2.5])
        self._neutral_rjoint_values = np.array([-2.61, -1.4, -2.05, 0.33, -0.591, 1,-0.314])
      #  self._neutral_rjoint_values = np.array([0.902, 1.36, 0.218, 1.42, 0.404, 0.558,-1.32])      
       # self._neutral_ljoint_values = np.array([0, 0, 0, 0, 0, 0.758,-1.7])
       # self._neutral_rjoint_values = np.array([0, 0, 0, 0, 0, -0.914,1.63])

        # Three auxiliary variables to understand the component of the xml document but will not be used
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        self.goal = self._sample_goal()
        
        self.gym.opt.iterations = 150
        self.gym.opt.noslip_tolerance = 50
        self.gym.opt.ccd_iterations = 100
        self.gym.opt.sdf_iterations = 50
        self.set_opt_config()

        self._base_body_name = [self.body("platform_base_link")]
        self._base_body_xpos, _, self._base_body_xquat = self.get_body_xpos_xmat_xquat(self._base_body_name)
        print("base_body_xpos: ", self._base_body_xpos)
        print("base_body_xquat: ", self._base_body_xquat)

        # index used to distinguish arm and gripper joints
        self._larm_joint_names = [self.joint("l_joint1"), self.joint("l_joint2"), self.joint("l_joint3"), self.joint("l_joint4"), self.joint("l_joint5"), self.joint("l_joint6"),self.joint("l_joint7")]
        self._rarm_joint_names = [self.joint("r_joint1"), self.joint("r_joint2"), self.joint("r_joint3"), self.joint("r_joint4"), self.joint("r_joint5"), self.joint("r_joint6"),self.joint("r_joint7")]
        self.gripper_joint_names = [self.joint("Gripper_Link1"), self.joint("Gripper_Link11"), self.joint("Gripper_Link2"), self.joint("Gripper_Link22"),]
        self.gripper_body_names = [self.body("Gripper_Link11"), self.body("Gripper_Link22")]
        self.gripper_geom_ids = []
        for geom_info in self.model.get_geom_dict().values():
            if geom_info["BodyName"] in self.gripper_body_names:
                self.gripper_geom_ids.append(geom_info["GeomId"])
                
        self._lhand_joint_names = [self.joint("l_th_immediate_link"), self.joint("l_th_distal_link"), self.joint("l_ff_proximal_link"), self.joint("l_ff_distal_link"),
                                   self.joint("l_mf_proximal_link"), self.joint("l_mf_distal_link"), self.joint("l_rf_proximal_link"), self.joint("l_rf_distal_link"),
                                   self.joint("l_lf_proximal_link"), self.joint("l_lf_distal_link"), self.joint("l_th_proximal_link")]
        self._rhand_joint_names =  [self.joint("r_th_immediate_link"), self.joint("r_th_distal_link"), self.joint("r_ff_proximal_link"), self.joint("r_ff_distal_link"),
                                   self.joint("r_mf_proximal_link"), self.joint("r_mf_distal_link"), self.joint("r_rf_proximal_link"), self.joint("r_rf_distal_link"),
                                   self.joint("r_lf_proximal_link"), self.joint("r_lf_distal_link"), self.joint("r_th_proximal_link")]

        self._larm_moto_names = [self.actuator("lactuator1"), self.actuator("lactuator2"),
                                self.actuator("lactuator3"),self.actuator("lactuator4"),
                                self.actuator("lactuator5"),self.actuator("lactuator6"),self.actuator("lactuator7")]
        self._rarm_moto_names = [self.actuator("ractuator1"), self.actuator("ractuator2"),
                                self.actuator("ractuator3"),self.actuator("ractuator4"),
                                self.actuator("ractuator5"),self.actuator("ractuator6"),self.actuator("ractuator7")]
        
        
        self._larm_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._larm_moto_names]
        self._rarm_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._rarm_moto_names]
        
        self._l_hand_moto_names = [self.actuator("lactuator8"), self.actuator("lactuator9"), self.actuator("lactuator10"),
                                   self.actuator("lactuator11"),self.actuator("lactuator12"),self.actuator("lactuator13"),
                                   self.actuator("lactuator14"),self.actuator("lactuator15"),self.actuator("lactuator16"),
                                   self.actuator("lactuator17"),self.actuator("lactuator18")]
        self._r_hand_moto_names = [self.actuator("ractuator8"), self.actuator("ractuator9"), self.actuator("ractuator10"),
                                   self.actuator("ractuator11"),self.actuator("ractuator12"),self.actuator("ractuator13"),
                                   self.actuator("ractuator14"),self.actuator("ractuator15"),self.actuator("ractuator16"),
                                   self.actuator("ractuator17"),self.actuator("ractuator18")]
        self._lhand_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._l_hand_moto_names]
        self._rhand_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._r_hand_moto_names]
        
        print("_larm_actuator_id: ", self._larm_actuator_id)
        print("_rarm_actuator_id: ", self._rarm_actuator_id)
        
        self._neutral_lhandjoint_values = np.array([0.15, 0.15, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, -1.4])
       # self._neutral_lhandjoint_values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self._neutral_rhandjoint_values = np.array([0.15, 0.15, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, -1.4])
        
        """ 临时注释
        self._gripper_actuator_names = [self.actuator("actuator_gripper1"), self.actuator("actuator_gripper2"),
                                        self.actuator("actuator_gripper11"), self.actuator("actuator_gripper22")]
        self._gripper_actuator_id = [self.model.actuator_name2id(actuator_name) for actuator_name in self._gripper_actuator_names]
        """
        # control range
        all_actuator_ctrlrange = self.model.get_actuator_ctrlrange()
        self._larm_ctrl_range = [all_actuator_ctrlrange[actoator_id] for actoator_id in self._larm_actuator_id]
        self._rarm_ctrl_range = [all_actuator_ctrlrange[actoator_id] for actoator_id in self._rarm_actuator_id]
        
        print("_rarm_ctrl_range: ", self._larm_ctrl_range)
        print("_rarm_ctrl_range: ", self._rarm_ctrl_range)
        
     #   self._gripper_ctrl_range = {actuator_name: all_actuator_ctrlrange[actuator_id] for actuator_name, actuator_id in zip(self._gripper_actuator_names, self._gripper_actuator_id)}
     #   print("gripper ctrl range: ", self._gripper_ctrl_range)
        actuators_dict = self.model.get_actuator_dict()
        self.gripper_force_limit = 1 #actuators_dict[self.actuator("actuator_gripper1")]["ForceRange"][1]
        self.gripper_state = GripperState.STOPPED
        
        self._l_gripper_pressed = False
        self._r_gripper_pressed = False

        self._handjointlimit = self.query_joint_limits(self._lhand_joint_names + self._rhand_joint_names)
        print("self.handjointlimit:" ,self._handjointlimit) 
        self._set_init_state()

        # 保存初始的抓夹位置
        EE_NAME  = self.site("ee_center_site")
        _site_dict = self.query_site_pos_and_quat([EE_NAME])
        self._initial_grasp_site_xpos = _site_dict[EE_NAME]['xpos']
        self._initial_grasp_site_xquat = _site_dict[EE_NAME]['xquat']
        self._saved_xpos = self._initial_grasp_site_xpos
        self._saved_xquat = self._initial_grasp_site_xquat

        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)

        EE_NAME_R  = self.site("ee_center_site_r")
        _site_dict_r = self.query_site_pos_and_quat([EE_NAME_R])
        self._initial_grasp_site_xpos_r = _site_dict_r[EE_NAME_R]['xpos']
        self._initial_grasp_site_xquat_r = _site_dict_r[EE_NAME_R]['xquat']
        self._saved_xpos_r = self._initial_grasp_site_xpos_r
        self._saved_xquat_r = self._initial_grasp_site_xquat_r

        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)
      
        self._pico_joystick = PicoJoystick()
        
        self._r_controller_config = controller_config.load_config("osc_pose")
        # print("controller_config: ", self.controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._r_controller_config["robot_name"] = agent_names[0]
        self._r_controller_config["sim"] = self.gym
        self._r_controller_config["eef_name"] = EE_NAME_R
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._rarm_joint_names)
        self._r_controller_config["joint_indexes"] = {
            "joints": self._rarm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        self._r_controller_config["actuator_range"] = self._rarm_ctrl_range
        self._r_controller_config["policy_freq"] = self.control_freq
        self._r_controller_config["ndim"] = len(self._rarm_joint_names)
        self._r_controller_config["control_delta"] = False
        self._r_gripper_offset_rate_clip = 0.0

        self._r_controller = controller_factory(self._r_controller_config["type"], self._r_controller_config)
        self._r_controller.update_initial_joints(self._neutral_rjoint_values)

        self._l_controller_config = controller_config.load_config("osc_pose")
        print("controller_config: ", self._l_controller_config)

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self._l_controller_config["robot_name"] = agent_names[0]
        self._l_controller_config["sim"] = self.gym
        self._l_controller_config["eef_name"] = EE_NAME
        # self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._larm_joint_names) #+self._rarm_joint_names
        print("qpos_offsets: ", qpos_offsets)
        self._l_controller_config["joint_indexes"] = {
            "joints": self._larm_joint_names,   #+self._rarm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        
      #  qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self._rarm_joint_names)
      #  self._controller_config["rarmjoint_indexes"] = {
      #$      "joints": self._rarm_joint_names,
      #      "qpos": qpos_offsets,
      #      "qvel": qvel_offsets,
      #  }
        self._l_controller_config["actuator_range"] = self._larm_ctrl_range
        self._l_controller_config["policy_freq"] = self.control_freq
        self._l_controller_config["ndim"] = len(self._larm_joint_names) #+ len(self._rarm_joint_names)
        self._l_controller_config["control_delta"] = False


        self._l_controller = controller_factory(self._l_controller_config["type"], self._l_controller_config)
        self._l_controller.update_initial_joints(self._neutral_ljoint_values)  
        self._l_gripper_offset_rate_clip = 0.0   
                        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        self.action_space = self.generate_action_space(self.model.get_actuator_ctrlrange()) 

        
    def _ConvertHandCtrl(self,ctrl):
        lnamelist = self._lhand_joint_names
        ctrlret = np.zeros(len(lnamelist) +1, dtype = np.int32)
        for index in range(len(lnamelist)):
            ctrllimitinfo = self._handjointlimit[index]
            qpos_min = ctrllimitinfo['range_min']
            qpos_max = ctrllimitinfo['range_max']  
            ctrlret[index] = round((qpos_max - ctrl[index])*255 / (qpos_max - qpos_min)) 
            print("Index Joint Result:",index, " qpos_min:", qpos_min, " qpos_max: ", qpos_max, " ctrl:", ctrl[index], " ctrlRet:", ctrlret[index])
            #if index % 2 == 0:
            #    ctrlret[index] = ctrlret[index] << 8
        print("ctrlret:", ctrlret)
        return ctrlret.tolist()   

    def _set_init_state(self) -> None:
        # print("Set initial state")
        self.set_joint_neutral()

        self.ctrl = np.concatenate((self._neutral_ljoint_values,self._neutral_lhandjoint_values,self._neutral_rjoint_values,self._neutral_lhandjoint_values))
        self.last_ctrl_l = self.ctrl[:7].copy()
        self.last_ctrl_r = self.ctrl[18:25].copy()
        
        print("initial ctrl: ", self.ctrl)
        print("initial last_ctrl_l: ", self.last_ctrl_l)
        print("initial last_ctrl_r: ", self.last_ctrl_r)
        # print(f'initial self.ctrl:{self.ctrl * 57.29577951308232}')
        if self.use_controller:
           self.robot_controller_l.init_joint_state(self.last_ctrl_l.copy())
           self.robot_controller_r.init_joint_state(self.last_ctrl_r.copy())
           
           handctrl = self._ConvertHandCtrl(self._neutral_lhandjoint_values) 
           self.robot_controller_l.roh_init()
           self.robot_controller_l.finger_move(ROH_FINGER_POS_TARGET0,6,handctrl,1.0)
           
           handctrl = self._ConvertHandCtrl(self._neutral_rhandjoint_values) 
           #
           self.robot_controller_r.roh_init()
           self.robot_controller_r.finger_move(ROH_FINGER_POS_TARGET0,6,handctrl,1.0)
           
          # handctrl = self._ConvertHandCtrl(self._neutral_rhandjoint_values)
          #self.robot_controller_r.finger_move(ROH_FINGER_POS_TARGET0,6,handctrl,1.0)
        self.set_ctrl(self.ctrl)
        print("ctrl: ", self.ctrl)

        self.mj_forward()


    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        self._set_action()
        self.do_simulation(self.ctrl, self.frame_skip)
        
        if self.use_controller:
            real_hand_qpos_dict = self.query_joint_qpos(self._larm_joint_names )
            self.last_ctrl_l = self.ctrl.copy()
            for i in range(len(self._larm_joint_names)):
                self.last_ctrl_l[i] = real_hand_qpos_dict[self._larm_joint_names[i]]
            mm = self.last_ctrl_l * 57.29577951308232
            self.robot_controller_l.sync_joint(self.last_ctrl_l.copy())
           # print("\nreal_hand_qpos_dict  leftHand: ", self.last_ctrl_l)
          #  print("\nreal_hand_qpos_dict  leftHand degree: ", mm)
            
            real_hand_qpos_dict2 = self.query_joint_qpos(self._rarm_joint_names)
            self.last_ctrl_r = self.ctrl.copy()
            for i in range(len(self._rarm_joint_names)):
                self.last_ctrl_r[i] = real_hand_qpos_dict2[self._rarm_joint_names[i]]
   
            mm = self.last_ctrl_r * 57.29577951308232
            self.robot_controller_r.sync_joint(self.last_ctrl_r.copy())          
           # print("\nreal_hand_qpos_dict  rightHand: ", self.last_ctrl_r)
          #  print("\nreal_hand_qpos_dict  rightHand degree: ", mm)
            
            if self._l_gripper_pressed:
                handctrl = self._ConvertHandCtrl(self.ctrl[7:18])
                #handctrl.append(0)
                print("handctrl:" ,handctrl)
                self.robot_controller_l.finger_move(ROH_FINGER_POS_TARGET0,6,handctrl,0.001)
            if self._r_gripper_pressed:
                handctrl = self._ConvertHandCtrl(self.ctrl[25:36])
                #handctrl.append(0)
                print("handctrl:" ,handctrl)
                self.robot_controller_r.finger_move(ROH_FINGER_POS_TARGET0,6,handctrl,0.001)

        obs = self._get_obs().copy()
        
        info = {}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info
    
    def _query_gripper_contact_force(self) -> dict:
        contact_simple_list = self.query_contact_simple()
        contact_force_query_ids = []
        for contact_simple in contact_simple_list:
            if contact_simple["Geom1"] in self.gripper_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])
            if contact_simple["Geom2"] in self.gripper_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])

        # print("Contact force query ids: ", contact_force_query_ids)
        contact_force_dict = self.query_contact_force(contact_force_query_ids)
        return contact_force_dict

    def _set_gripper_ctrl(self, joystick_state):
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.1 *30 # 10% per second
        IsPRESS = False
        if joystick_state["leftHand"]["secondaryButtonPressed"]:
            self._l_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * self.dt    
            self._l_gripper_offset_rate_clip = np.clip(self._l_gripper_offset_rate_clip, -1,0)
            IsPRESS = True
        elif joystick_state["leftHand"]["primaryButtonPressed"]:
            self._l_gripper_offset_rate_clip += offset_rate_clip_adjust_rate * self.dt
            self._l_gripper_offset_rate_clip = np.clip(self._l_gripper_offset_rate_clip, -1, 0)
            IsPRESS = True

        # Press trigger to close gripper
        # Adjust sensitivity using an exponential function
        trigger_value = joystick_state["leftHand"]["triggerValue"]  # Value in [0, 1]
        k = np.e  # Adjust 'k' to change the curvature of the exponential function
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1, self._l_gripper_offset_rate_clip)

        for i in range(8):
            ctrllimitinfo = self._handjointlimit[i+2]
            qpos_min = ctrllimitinfo['range_min']
            qpos_max = ctrllimitinfo['range_max']  

            abs_ctrlrange = qpos_max - qpos_min
            self.ctrl[i+9] = qpos_min -  self._l_gripper_offset_rate_clip * 1 * abs_ctrlrange
            self.ctrl[i+9] = np.clip(
                self.ctrl[i+9],
                qpos_min,
                qpos_max)
        if IsPRESS:
            print("offset_rate: ",  offset_rate)
            print("ctrl  aaaaaaa: ", self.ctrl[:18])
            print("ToHnadCtrl:",self._ConvertHandCtrl(self.ctrl[7:18]))
        return IsPRESS
        
        
        
    def _set_gripper_ctrl_r(self, joystick_state):
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.1 *30 # 10% per second
        IsPRESS = False
        if joystick_state["rightHand"]["secondaryButtonPressed"]:
            self._r_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * self.dt    
            self._r_gripper_offset_rate_clip = np.clip(self._r_gripper_offset_rate_clip, -1, 0)
            IsPRESS = True
        elif joystick_state["rightHand"]["primaryButtonPressed"]:
            self._r_gripper_offset_rate_clip += offset_rate_clip_adjust_rate * self.dt
            self._r_gripper_offset_rate_clip = np.clip(self._r_gripper_offset_rate_clip, -1, 0)
            IsPRESS = True

        # Press trigger to close gripper
        # Adjust sensitivity using an exponential function
        trigger_value = joystick_state["rightHand"]["triggerValue"]  # Value in [0, 1]
        k = np.e  # Adjust 'k' to change the curvature of the exponential function
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1, self._r_gripper_offset_rate_clip)

        for i in range(8):
            ctrllimitinfo = self._handjointlimit[i+2+11]
            qpos_min = ctrllimitinfo['range_min']
            qpos_max = ctrllimitinfo['range_max']  

            abs_ctrlrange = qpos_max - qpos_min
            self.ctrl[i+9+18] = qpos_min -  self._r_gripper_offset_rate_clip * 1 * abs_ctrlrange
            self.ctrl[i+9+18] = np.clip(
                self.ctrl[i+9+18],
                qpos_min,
                qpos_max)
        if IsPRESS:
            print("ctrl  aaaaaaa: ", self.ctrl[:18])
            print("ToHnadCtrl:",self._ConvertHandCtrl(self.ctrl[25:36]))
        return IsPRESS

        
    def _process_pico_joystick_operation(self):
        joystick_state = self._pico_joystick.get_key_state()
        if joystick_state is None:
            return

        self._l_gripper_pressed = self._set_gripper_ctrl(joystick_state)
        self._r_gripper_pressed = self._set_gripper_ctrl_r(joystick_state)
        #self._set_head_ctrl(joystick_state)

    def _set_action(self) -> None:
        mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = None, None, None, None

        if self._pico_joystick is not None:
                mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat = self._processe_pico_joystick_move()
                self.set_grasp_mocap(mocap_l_xpos, mocap_l_xquat)
                self.set_grasp_mocap_r(mocap_r_xpos, mocap_r_xquat)
                self._process_pico_joystick_operation()
                # print("base_body_euler: ", self._base_body_euler / np.pi * 180)
        else:
                return



        # 两个工具的quat不一样，这里将 qw, qx, qy, qz 转为 qx, qy, qz, qw
        mocap_r_axisangle = transform_utils.quat2axisangle(np.array([mocap_r_xquat[1], 
                                                                   mocap_r_xquat[2], 
                                                                   mocap_r_xquat[3], 
                                                                   mocap_r_xquat[0]]))              
        # mocap_axisangle[1] = -mocap_axisangle[1]
        action_r = np.concatenate([mocap_r_xpos, mocap_r_axisangle])
        # print("action r:", action_r)
        self._r_controller.set_goal(action_r)
        ctrl = self._r_controller.run_controller()
        # print("ctrl r: ", ctrl)
        for i in range(len(self._rarm_actuator_id)):
            self.ctrl[self._rarm_actuator_id[i]] = ctrl[i]


        mocap_l_axisangle = transform_utils.quat2axisangle(np.array([mocap_l_xquat[1], 
                                                                   mocap_l_xquat[2], 
                                                                   mocap_l_xquat[3], 
                                                                   mocap_l_xquat[0]]))  
        action_l = np.concatenate([mocap_l_xpos, mocap_l_axisangle])
        # print("action l:", action_l)        
        # print(action)
        self._l_controller.set_goal(action_l)
        ctrl = self._l_controller.run_controller()
        # print("ctrl l: ", ctrl)
        for i in range(len(self._larm_actuator_id)):
            self.ctrl[self._larm_actuator_id[i]] = ctrl[i]
    
    def _reset_gripper(self) -> None:
        self._l_gripper_offset_rate_clip = 0.0
        self._r_gripper_offset_rate_clip = 0.0
            
    def _processe_pico_joystick_move(self):
        if self._pico_joystick.is_reset_pos():
            self._pico_joystick.set_reset_pos(False)
            self._reset_gripper()
            print("Reset pos")
            self._set_init_state()

        transform_list = self._pico_joystick.get_transform_list()
        if transform_list is None:
           # print("transform_list is None")
            return self._initial_grasp_site_xpos, self._initial_grasp_site_xquat, self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r

        left_relative_position, left_relative_rotation = self._pico_joystick.get_left_relative_move(transform_list)
        right_relative_position, right_relative_rotation = self._pico_joystick.get_right_relative_move(transform_list)

        mocap_l_xpos = self._initial_grasp_site_xpos + rotations.quat_rot_vec(self._base_body_xquat, left_relative_position)
        mocap_r_xpos = self._initial_grasp_site_xpos_r + rotations.quat_rot_vec(self._base_body_xquat, right_relative_position)

        mocap_l_xquat = rotations.quat_mul(self._initial_grasp_site_xquat, left_relative_rotation)
        # mocap_r_xquat = rotations.quat_mul(self._initial_grasp_site_xquat_r, right_relative_rotation)
        mocap_r_xquat = (R.from_quat(self._initial_grasp_site_xquat_r, scalar_first=True) * 
                         R.from_quat(right_relative_rotation, scalar_first=True)).as_quat(scalar_first=True, canonical=True)
        
        return mocap_l_xpos, mocap_l_xquat, mocap_r_xpos, mocap_r_xquat

    def _process_xbox_controller(self, mocap_xpos, mocap_xquat) -> tuple[np.ndarray, np.ndarray]:
        self._joystick_manager.update()

        pos_ctrl_dict = self._joystick.capture_joystick_pos_ctrl()
        pos_ctrl = np.array([pos_ctrl_dict['z'], -pos_ctrl_dict['x'], pos_ctrl_dict['y']])
        rot_ctrl_dict = self._joystick.capture_joystick_rot_ctrl()
        rot_ctrl = np.array([rot_ctrl_dict['yaw'], rot_ctrl_dict['pitch'], rot_ctrl_dict['roll']])
        
     #   self._set_gripper_ctrl(self._joystick.get_state())

        # 考虑到手柄误差，只有输入足够的控制量，才移动mocap点
        CTRL_MIN = 0.10000000
        if np.linalg.norm(pos_ctrl) < CTRL_MIN and np.linalg.norm(rot_ctrl) < CTRL_MIN:
            return mocap_xpos, mocap_xquat

        mocap_xmat = rotations.quat2mat(mocap_xquat)

        # 平移控制
        MOVE_SPEED = self.gym.opt.timestep * 0.2
        mocap_xpos = mocap_xpos + np.dot(mocap_xmat, pos_ctrl) * MOVE_SPEED
        mocap_xpos[2] = np.max((0, mocap_xpos[2]))  # 确保在地面以上

        # 旋转控制
        ROUTE_SPEED = self.gym.opt.timestep * 0.5
        rot_offset = rot_ctrl * ROUTE_SPEED
        new_xmat = self._joystick.calc_rotate_matrix(rot_offset[0], rot_offset[1], rot_offset[2])
        mocap_xquat = rotations.mat2quat(np.dot(mocap_xmat, new_xmat))

        return mocap_xpos, mocap_xquat

    def _get_obs(self) -> dict:
        # robot

        EE_NAME = self.site("ee_center_site")
        ee_position = self.query_site_pos_and_quat([EE_NAME])[EE_NAME]['xpos'].copy()
        ee_xvalp, _ = self.query_site_xvalp_xvalr([EE_NAME])
        ee_velocity = ee_xvalp[EE_NAME].copy() * self.dt
        fingers_width = [0]


        achieved_goal = np.array([0,0,0])
        desired_goal = self.goal.copy()
        obs = np.concatenate(
                [
                    ee_position,
                    ee_velocity,
                    fingers_width
                ], dtype=np.float32).copy()            
        result = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

        return result

    def _render_callback(self) -> None:
        pass


    def reset_model(self):
        # Robot_env 统一处理，这里实现空函数就可以
        self._set_init_state()
        #self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat)
        self.mj_forward()
        obs = self._get_obs().copy()
        return obs

    def _reset_sim(self) -> bool:
        self._set_init_state()
        self.set_grasp_mocap(self._initial_grasp_site_xpos, self._initial_grasp_site_xquat) #临时注释
        self.set_grasp_mocap_r(self._initial_grasp_site_xpos_r, self._initial_grasp_site_xquat_r)
        self.mj_forward()
        return True

    # custom methods
    # -----------------------------

    def set_grasp_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("rm75bv_l_mocap"): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)
        
    
    def set_grasp_mocap_r(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {self.mocap("rm75bv_r_mocap"): {'pos': position, 'quat': orientation}}
        # print("Set grasp mocap: ", position, orientation)
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_goal_mocap(self, position, orientation) -> None:
        mocap_pos_and_quat_dict = {"goal_goal": {'pos': position, 'quat': orientation}}
        self.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        arm_joint_qpos_list = {}
        for name, value in zip(self._larm_joint_names, self._neutral_ljoint_values):
            arm_joint_qpos_list[name] = np.array([value])
        for name, value in zip(self._rarm_joint_names, self._neutral_rjoint_values):
            arm_joint_qpos_list[name] = np.array([value])
        self.set_joint_qpos(arm_joint_qpos_list)

        # assign value to finger joints
        gripper_joint_qpos_list = {}
        for name, value in zip(self.gripper_joint_names, self._neutral_ljoint_values):
            gripper_joint_qpos_list[name] = np.array([value])
        self.set_joint_qpos(gripper_joint_qpos_list)

    def _sample_goal(self) -> np.ndarray:
        # 训练reach时，任务是移动抓夹，goal以抓夹为原点采样
        goal = np.array([0, 0, 0])
        return goal

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()
