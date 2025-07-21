from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
import yaml
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
import os
import numpy as np
import time

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
        self._acceleration = self._speed['Acceleration']
        self._move_speed = 0.0
        self._turn_speed = 0.0

        self._use_keyboard_control = self._config['control_type'] == "keyboard"
        self._keyboard_control = self._config['keyboard_control']


        self._use_waypoint_control = self._config['control_type'] == "waypoint"
        self._waypoint_control = self._config['waypoint_control']
        self._waypoint_distance_threshold = self._config['waypoint_distance_threshold']
        self._waypoint_angle_threshold = self._config['waypoint_angle_threshold']

        body_xpos, _, _ = self._env.get_body_xpos_xmat_xquat([self._body_name])
        self._original_coordinates = body_xpos[:2]
        print("Original Coordinates: ", self._original_coordinates)

        
    def on_step(self):
        rotate_z_pos = self._env.query_joint_qpos([self._joint_names["Rotate_Z"]])[self._joint_names["Rotate_Z"]][0]
        heading = rotate_z_pos % (2 * np.pi)

        if self._use_keyboard_control:
            self._process_keyboard_input(heading)
        elif self._use_waypoint_control:
            self._process_waypoint_input(heading)

        self._env.set_joint_qvel(self._ctrl_joint_qvel)

    def on_reset(self):
        self._move_speed = 0
        self._turn_speed = 0
        self._waypoint_time = 0
        self._waypoint_index = -1
        self._moving_to_waypoint = False
        self._next_waypoint_coord = None
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Forward", False)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Backward", False)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnLeft", False)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnRight", False)

        ctrl_qpos = {
            self._joint_names["Rotate_Z"] : [0],
        }

        self._env.set_joint_qpos(ctrl_qpos)


    def _turn_left(self):
        self._turn_speed = self._speed['TurnLeft']
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnLeft", True)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnRight", False)

    def _turn_right(self):
        self._turn_speed = self._speed['TurnRight']
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnRight", True)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnLeft", False)
    

    def _stop_turning(self):
        self._turn_speed = 0
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnLeft", False)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "TurnRight", False)

    def _move_forward(self):
        if self._move_speed < self._speed['Forward']:
            self._move_speed += self._speed['Forward'] * self._acceleration
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Forward", True)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Backward", False)

    def _move_backward(self):
        if self._move_speed > self._speed['Backward']:
            self._move_speed += self._speed['Backward'] * self._acceleration
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Backward", True)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Forward", False)

    def _stop_moving(self):
        self._move_speed = 0
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Forward", False)
        self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, "Backward", False)

    def _process_move(self, move_speed : float, heading : float):
        move_y_vel = move_speed * np.cos(heading)
        move_x_vel = move_speed * -np.sin(heading)

        self._ctrl_joint_qvel[self._joint_names["Move_X"]][0] = move_x_vel
        self._ctrl_joint_qvel[self._joint_names["Move_Y"]][0] = move_y_vel
        self._ctrl_joint_qvel[self._joint_names["Rotate_Z"]][0] = self._turn_speed

    def _process_keyboard_input(self, heading : float):
        if not self._use_keyboard_control:
            return
        
        self._keyboard.update()
        keyboard_state = self._keyboard.get_state()


        # print("Speed: ", self._speed)

        if keyboard_state[self._keyboard_control['move_forward']] == 1:
            self._move_forward()
        elif keyboard_state[self._keyboard_control['move_backward']] == 1:
            self._move_backward()
        else:
            self._stop_moving()

        if keyboard_state[self._keyboard_control['turn_left']] == 1:
            self._turn_left()
        elif keyboard_state[self._keyboard_control['turn_right']] == 1:
            self._turn_right()
        else:
            self._stop_turning()

        # print("heading: ", heading, "move_speed: ", self._move_speed, "turn_speed: ", self._turn_speed)
        self._process_move(self._move_speed, heading)

    def _process_waypoint_input(self, heading : float):
        if not self._use_waypoint_control:
            return

        current_time = time.time()
        time_delta = current_time - self._waypoint_time

        if not self._moving_to_waypoint and time_delta > self._waypoint_control[self._waypoint_index]['Duration']:
            # time to go to next waypoint
            self._moving_to_waypoint = True
            self._waypoint_index += 1
            if self._waypoint_index >= len(self._waypoint_control):
                self._waypoint_index = 0
            
            next_waypoint_coord = np.array(self._waypoint_control[self._waypoint_index]['Coordinates']) + self._original_coordinates
            print("Next Waypoint Coordinates: ", next_waypoint_coord)
            self._next_waypoint_coord = next_waypoint_coord

        if self._next_waypoint_coord is not None and self._moving_to_waypoint:
            body_xpos, _, _ = self._env.get_body_xpos_xmat_xquat([self._body_name])
            current_coordinates = body_xpos[:2] 

            # calculate the distance to the next waypoint, and the direction
            distance = np.linalg.norm(self._next_waypoint_coord - current_coordinates)
            direction = (self._next_waypoint_coord - current_coordinates) / distance
            # print("Distance: ", distance, "Direction: ", direction)
                
            # check if the character has reached the waypoint
            if distance < self._waypoint_distance_threshold:
                # reached the waypoint
                self._stop_moving()
                self._stop_turning()
                self._moving_to_waypoint = False
                self._waypoint_time = current_time
                self._process_move(self._move_speed, heading)
                return

            # check the direction of the character, if not facing the waypoint, turn to the waypoint
            angle_error = np.dot(direction, [np.cos(heading), np.sin(heading)])
            if angle_error < -self._waypoint_angle_threshold:
                self._turn_left()
            elif angle_error > self._waypoint_angle_threshold:
                self._turn_right()
            else:
                self._stop_turning()
                self._move_forward()
                
            self._process_move(self._move_speed, heading)



            
