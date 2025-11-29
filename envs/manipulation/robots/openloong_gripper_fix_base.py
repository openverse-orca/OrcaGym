import numpy as np
from envs.manipulation.dual_arm_env import DualArmEnv
from envs.manipulation.dual_arm_robot import DualArmRobot
from envs.manipulation.robots.configs.gripper_2f85_config import gripper_2f85_config as config

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class OpenLoongGripperFixBase(DualArmRobot):
    def __init__(self, env: DualArmEnv, id: int, name: str) -> None:
        super().__init__(env, id, name)

        self.init_agent(id)

        
    def init_agent(self, id: int):
        _logger.info("OpenLoongGripperFixBase init_agent")

        super().init_agent(id)

        self._l_hand_actuator_names = [self._env.actuator(config["left_hand"]["actuator_names"][0], id)]
        
        self._l_hand_actuator_id = [self._env.model.actuator_name2id(actuator_name) for actuator_name in self._l_hand_actuator_names]        
        self._l_hand_body_names = [self._env.body(config["left_hand"]["body_names"][0], id), self._env.body(config["left_hand"]["body_names"][1], id)]
        self._l_hand_gemo_ids = []
        for geom_info in self._env.model.get_geom_dict().values():
            if geom_info["BodyName"] in self._l_hand_body_names:
                self._l_hand_gemo_ids.append(geom_info["GeomId"])

        self._r_hand_actuator_names = [self._env.actuator(config["right_hand"]["actuator_names"][0], id)]

        self._r_hand_actuator_id = [self._env.model.actuator_name2id(actuator_name) for actuator_name in self._r_hand_actuator_names]
        self._r_hand_body_names = [self._env.body(config["right_hand"]["body_names"][0], id), self._env.body(config["right_hand"]["body_names"][1], id)]
        self._r_hand_gemo_ids = []
        for geom_info in self._env.model.get_geom_dict().values():
            if geom_info["BodyName"] in self._r_hand_body_names:
                self._r_hand_gemo_ids.append(geom_info["GeomId"])  


    def set_l_hand_actuator_ctrl(self, offset_rate) -> None:
        for actuator_id in self._l_hand_actuator_id:
            offset_dir = -1

            abs_ctrlrange = self._all_ctrlrange[actuator_id][1] - self._all_ctrlrange[actuator_id][0]
            self._env.ctrl[actuator_id] = offset_rate * offset_dir * abs_ctrlrange
            self._env.ctrl[actuator_id] = np.clip(
                self._env.ctrl[actuator_id],
                self._all_ctrlrange[actuator_id][0],
                self._all_ctrlrange[actuator_id][1])
            
    def set_gripper_ctrl_l(self, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.5  # 10% per second
        if joystick_state["leftHand"]["secondaryButtonPressed"]:
            self._l_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * self._env.dt    
            self._l_gripper_offset_rate_clip = np.clip(self._l_gripper_offset_rate_clip, -1, 0)
        elif joystick_state["leftHand"]["primaryButtonPressed"]:
            self._l_gripper_offset_rate_clip = 0

        # Press trigger to close gripper
        # Adjust sensitivity using an exponential function
        trigger_value = joystick_state["leftHand"]["triggerValue"]  # Value in [0, 1]
        k = np.e  # Adjust 'k' to change the curvature of the exponential function
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1, self._l_gripper_offset_rate_clip)
        self.set_l_hand_actuator_ctrl(offset_rate)
        self._grasp_value_l = offset_rate
            
    def set_r_hand_actuator_ctrl(self, offset_rate) -> None:
        for actuator_id in self._r_hand_actuator_id:
            offset_dir = -1

            abs_ctrlrange = self._all_ctrlrange[actuator_id][1] - self._all_ctrlrange[actuator_id][0]
            self._env.ctrl[actuator_id] = offset_rate * offset_dir * abs_ctrlrange
            self._env.ctrl[actuator_id] = np.clip(
                self._env.ctrl[actuator_id],
                self._all_ctrlrange[actuator_id][0],
                self._all_ctrlrange[actuator_id][1])

    def set_gripper_ctrl_r(self, joystick_state) -> None:
        # Press secondary button to set gripper minimal value
        offset_rate_clip_adjust_rate = 0.5
        if joystick_state["rightHand"]["secondaryButtonPressed"]:
            self._r_gripper_offset_rate_clip -= offset_rate_clip_adjust_rate * self._env.dt
            self._r_gripper_offset_rate_clip = np.clip(self._r_gripper_offset_rate_clip, -1, 0)
        elif joystick_state["rightHand"]["primaryButtonPressed"]:
            self._r_gripper_offset_rate_clip = 0

        # Adjust sensitivity using an exponential function
        trigger_value = joystick_state["rightHand"]["triggerValue"]  # Value in [0, 1]
        k = np.e  # Adjust 'k' to change the curvature of the exponential function
        adjusted_value = (np.exp(k * trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
        offset_rate = -adjusted_value
        offset_rate = np.clip(offset_rate, -1, self._r_gripper_offset_rate_clip)
        self.set_r_hand_actuator_ctrl(offset_rate)
        self._grasp_value_r = offset_rate       
            
    def update_force_feedback(self) -> None:
        if self._pico_joystick is not None:
            r_hand_force = self._query_hand_force(self._r_hand_gemo_ids)
            l_hand_force = self._query_hand_force(self._l_hand_gemo_ids)
            self._pico_joystick.send_force_message(l_hand_force, r_hand_force)            
            

    def _query_hand_force(self, hand_geom_ids):
        contact_simple_list = self._env.query_contact_simple()
        contact_force_query_ids = []
        for contact_simple in contact_simple_list:
            if contact_simple["Geom1"] in hand_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])
            if contact_simple["Geom2"] in hand_geom_ids:
                contact_force_query_ids.append(contact_simple["ID"])

        contact_force_dict = self._env.query_contact_force(contact_force_query_ids)
        compose_force = 0
        for force in contact_force_dict.values():
            compose_force += np.linalg.norm(force[:3])
        return compose_force                
    
    def set_wheel_ctrl(self, joystick_state) -> None:
        return