import asyncio
import json
import threading
import numpy as np
from websockets.asyncio.server import serve
import copy
from scipy.spatial.transform import Rotation as R
import threading


class PicoJoystick:
    def __init__(self):
        self.mutex= threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()
        self.first_transform = None
        self.current_transform = None
        self.current_key_state = None
        self.reset_pos = False
    
    def __del__(self):
        self.running = False
        self.thread.join()

    def _loop(self):
        asyncio.run(self._start_server())

    async def _handler(self, websocket):
        print("new connection")
        try:
            is_first_message = True
            async for message in websocket:
                # print(message)
                with self.mutex:
                    self.current_transform = self.extact_all_transform(json.loads(message))
                    self.current_key_state = self.extact_key_state(json.loads(message))
                    if is_first_message:
                        self.first_transform = copy.deepcopy(self.current_transform)
                        self.reset_pos = True
                        is_first_message = False
        except Exception as e:
            print("disconnected", e)

    async def _start_server(self):
        async with serve(self._handler, "0.0.0.0", 8001, ping_interval=None):
            await asyncio.get_running_loop().create_future()  # run forever

    def update(self):
        pass

    def is_reset_pos(self):
        if self.reset_pos:
            self.reset_pos = False
            return True
        return False

    def get_transform_list(self):
        with self.mutex:
            if self.current_transform is None:
                return None
            else:
                transform = copy.deepcopy(self.current_transform)
                return transform
            
    def get_key_state(self):
        with self.mutex:
            if self.current_key_state is None:
                return None
            else:
                key_state = copy.deepcopy(self.current_key_state)
                return key_state
            
    def _calc_rotate_matrix(self, yaw, pitch, roll) -> np.ndarray:
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        new_xmat = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        return new_xmat
        
    def extact_single_transform(self, body):
        position = body["position"]
        p_x = position["x"]
        p_y = position["y"]
        p_z = position["z"]
        rotation = body["rotation"]
        r_x = rotation["x"]
        r_y = rotation["y"]
        r_z = rotation["z"]
        # r = self._calc_rotate_matrix(r_z, -r_x, r_y)
        r = R.from_euler('xzy', [r_y, r_z, r_x], degrees=True)
        return [[p_z, -p_x, p_y], r]
        
    def extact_all_transform(self, message):
        left_hand_transform = self.extact_single_transform(message["leftHand"])
        right_hand_transform = self.extact_single_transform(message["rightHand"])
        motion_trackers_transform = []
        if "motionTrackers" in message:
            for motionTracker in message["motionTrackers"]:
                motion_trackers_transform.append(self.extact_single_transform(motionTracker))
        return [left_hand_transform, right_hand_transform, motion_trackers_transform]
    
    def extact_key_state(self, message) -> dict:
        left_hand_key_state = {"triggerValue": message["leftHand"]["triggerValue"],
                               "primaryButtonPressed": message["leftHand"]["primaryButtonPressed"],
                               "secondaryButtonPressed": message["leftHand"]["secondaryButtonPressed"],
                               "joystickPosition": [message["leftHand"]["joystickPosition"]["x"], message["leftHand"]["joystickPosition"]["y"]],
                               "joystickPressed": message["leftHand"]["joystickPressed"]}

        right_hand_key_state = {"triggerValue": message["rightHand"]["triggerValue"],
                                "primaryButtonPressed": message["rightHand"]["primaryButtonPressed"],
                                "secondaryButtonPressed": message["rightHand"]["secondaryButtonPressed"],
                                "joystickPosition": [message["rightHand"]["joystickPosition"]["x"], message["rightHand"]["joystickPosition"]["y"]],
                                "joystickPressed": message["rightHand"]["joystickPressed"]}
        return {"leftHand": left_hand_key_state, "rightHand": right_hand_key_state}

    def get_left_relative_move(self, transform):
        relative_position = [x - y for x, y in zip(transform[0][0], self.first_transform[0][0])]
        relative_rotation = self.first_transform[0][1].inv() * transform[0][1]
        # relative_rotation = np.dot(self.first_transform[0][1].T, transform[0][1])
        return [relative_position, relative_rotation]
    
    def get_right_relative_move(self, transform):
        relative_position = [x - y for x, y in zip(transform[1][0], self.first_transform[1][0])]
        relative_rotation = self.first_transform[1][1].inv() * transform[1][1]
        # relative_rotation = np.dot(self.first_transform[1][1].T, transform[1][1])
        return [relative_position, relative_rotation]
    
    def get_motion_trackers_relative_move(self, transform):
        motion_trackers_relative_move = []
        for i in range(len(transform[2])):
            relative_position = [x - y for x, y in zip(transform[2][i][0], self.first_transform[2][i][0])]
            relative_rotation = self.first_transform[2][i][1].inv() * transform[2][i][1]
            # relative_rotation = np.dot(self.first_transform[2][i][1].T, transform[2][i][1])
            motion_trackers_relative_move.append([relative_position, relative_rotation])
        return motion_trackers_relative_move
    
    def get_relative_move(self, transform):
        left_relative_move = self.get_left_relative_move(transform)
        right_relative_move = self.get_right_relative_move(transform)
        motion_trackers_relative_move = self.get_motion_trackers_relative_move(transform)
        return [left_relative_move, right_relative_move, motion_trackers_relative_move]


