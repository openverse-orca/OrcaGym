from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
import yaml
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
import os
import numpy as np

class Character():
    def __init__(self, 
        env: OrcaGymLocalEnv, 
        agent_name: str,
        agent_id: int,
        character_name: str,
    ):
        self._env = env
        self._agent_name = agent_name
        self._agent_id = agent_id
        self._model = env.model
        self._data = env.data
        self._load_config(character_name)
        self._init_character()  
        self._keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO)

    def _load_config(self, character_name: str):
        path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{path}/character_config/{character_name}.yaml", 'r') as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
            
    def _init_character(self):
        self._spawnable_name = self._config['spawnable_name']
        self._body_name = self._env.body(self._config['body_name'], self._agent_id)
        self._body_id = self._model.body_name2id(self._body_name)
        joint_name_dict = self._config['joint_names']
        self._joint_names = {
            "Move_X" : self._env.joint(joint_name_dict['Move_X'], self._agent_id),
            "Move_Y" : self._env.joint(joint_name_dict['Move_Y'], self._agent_id),
            "Move_Z" : self._env.joint(joint_name_dict['Move_Z'], self._agent_id),
            "Rotate_Z" : self._env.joint(joint_name_dict['Rotate_Z'], self._agent_id),
        }
        self._joint_ids = {
            "Move_X" : self._model.joint_name2id(self._joint_names['Move_X']),
            "Move_Y" : self._model.joint_name2id(self._joint_names['Move_Y']),
            "Move_Z" : self._model.joint_name2id(self._joint_names['Move_Z']),
            "Rotate_Z" : self._model.joint_name2id(self._joint_names['Rotate_Z']),
        }
        
        self._ctrl_joint_qvel = {
            self._joint_names["Move_X"] : [0],
            self._joint_names["Move_Y"] : [0],
            self._joint_names["Rotate_Z"] : [0],
        }

        self._speed = self._config['speed']
        self._move_speed = 0.0
        self._turn_speed = 0.0

        self._use_keyboard_control = self._config['use_keyboard_control']
        self._keyboard_control = self._config['keyboard_control']

        
    def on_step(self):
        rotate_z_pos = self._env.query_joint_qpos([self._joint_names["Rotate_Z"]])[self._joint_names["Rotate_Z"]][0]
        heading = rotate_z_pos % (2 * np.pi)

        if self._use_keyboard_control:
            self._process_keyboard_input(heading)

        self._env.set_joint_qvel(self._ctrl_joint_qvel)

    def on_reset(self):
        self._move_speed = 0
        self._turn_speed = 0
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Forward", False)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Backward", False)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnLeft", False)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnRight", False)

        ctrl_qpos = {
            self._joint_names["Rotate_Z"] : [0],
        }

        self._env.set_joint_qpos(ctrl_qpos)

    def _process_keyboard_input(self, heading : float):
        if not self._use_keyboard_control:
            return
        
        self._keyboard.update()
        keyboard_state = self._keyboard.get_state()


        # print("Speed: ", self._speed)

        if keyboard_state[self._keyboard_control['move_forward']] == 1:
            self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Forward", True)
            if self._move_speed < self._speed['Forward']:
                self._move_speed += self._speed['Forward'] * 0.1
        elif keyboard_state[self._keyboard_control['move_backward']] == 1:
            self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Backward", True)
            if self._move_speed > self._speed['Backward']:
                self._move_speed += self._speed['Backward'] * 0.1
        else:
            self._move_speed = 0
            self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Forward", False)
            self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Backward", False)


        if keyboard_state[self._keyboard_control['turn_left']] == 1:
            self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnLeft", True)
            self._turn_speed = self._speed['TurnLeft']
        elif keyboard_state[self._keyboard_control['turn_right']] == 1:
            self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnRight", True)
            self._turn_speed = self._speed['TurnRight']
        else:
            self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnLeft", False)
            self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnRight", False)
            self._turn_speed = 0

        # print("heading: ", heading, "move_speed: ", self._move_speed, "turn_speed: ", self._turn_speed)
        move_y_vel = self._move_speed * np.cos(heading)
        move_x_vel = self._move_speed * -np.sin(heading)

        self._ctrl_joint_qvel[self._joint_names["Move_X"]][0] = move_x_vel
        self._ctrl_joint_qvel[self._joint_names["Move_Y"]][0] = move_y_vel
        self._ctrl_joint_qvel[self._joint_names["Rotate_Z"]][0] = self._turn_speed


            
