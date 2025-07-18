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
        self._actor_name = self._config['actor_name']
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

        
    def on_step(self):
        rotate_z_pos = self._data.qpos[self._joint_ids['Rotate_Z']]
        heading = rotate_z_pos % (2 * np.pi)

        self._process_keyboard_input(heading)
        self._env.set_joint_qvel(self._ctrl_joint_qvel)

    def _process_keyboard_input(self, heading):
        self._keyboard.update()
        keyboard_state = self._keyboard.get_state()


        # print("Speed: ", self._speed)

        if keyboard_state['Up'] == 1:
            if self._move_speed < self._speed['Forward']:
                self._move_speed += self._speed['Forward'] * 0.1
        elif keyboard_state['Down'] == 1:
            if self._move_speed > self._speed['Backward']:
                self._move_speed += self._speed['Backward'] * 0.1
        else:
            self._move_speed = 0


        if keyboard_state['Left'] == 1:
            self._turn_speed = self._speed['TurnLeft']
        elif keyboard_state['Right'] == 1:
            self._turn_speed = self._speed['TurnRight']
        else:
            self._turn_speed = 0

        move_y_vel = self._move_speed * np.cos(-heading)
        move_x_vel = self._move_speed * np.sin(-heading)

        self._ctrl_joint_qvel[self._joint_names["Move_X"]][0] = move_x_vel
        self._ctrl_joint_qvel[self._joint_names["Move_Y"]][0] = move_y_vel
        self._ctrl_joint_qvel[self._joint_names["Rotate_Z"]][0] = self._turn_speed


            
