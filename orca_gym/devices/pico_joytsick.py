import asyncio
import enum
import json
import threading
from typing import Callable
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def transform_position_to_mujoco(p_unity):
    x_mujoco = p_unity[2]
    y_mujoco = -p_unity[0]
    z_mujoco = p_unity[1]
    return np.array([x_mujoco, y_mujoco, z_mujoco])

def transform_quaternion_to_mujoco(q_unity):
    R_unity = R.from_quat(q_unity).as_matrix()
    T = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    R_mujoco = T @ R_unity @ T.T
    v = R.from_matrix(R_mujoco).as_rotvec()
    return R.from_rotvec(v).as_quat()

class PicoJoystickKey(enum.Enum):
    X = 0
    Y = 1
    L_TRIGGER = 2
    L_GRIPBUTTON = 3
    L_JOYSTICK_POSITION = 4
    L_JOYSTICK_PRESSED = 5
    L_TRANSFORM = 6 
    A = 7
    B = 8
    R_TRIGGER = 9
    R_GRIPBUTTON = 10
    R_JOYSTICK_POSITION = 11
    R_JOYSTICK_PRESSED = 12
    R_TRANSFORM = 13



class PicoJoystick:
    def __init__(self, port=8001):
        self.mutex = threading.Lock()
        self.running = True
        self.current_transform = None
        self.current_key_state = None
        self.suceess_AB = 0 #AB键长按次数
        self.suceess_XY = 0 #XY键长按次数
        self.key_event = {}
        self.reset_pos = False
        self.loop = asyncio.new_event_loop()
        self.clients = set()  # 初始化 self.clients
        self.port = port
        self.thread = threading.Thread(target=self._start_server_thread)
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.running = False
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()

    def _start_server_thread(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start_server())

    async def _start_server(self):
        server = await asyncio.start_server(
            self._handle_client, '0.0.0.0', self.port)
        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader, writer):
        try:
            self.clients.add(writer)  # 添加客户端 writer
            buffer = ""
            while self.running:
                data = await reader.read(1024)
                if not data:
                    break
                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    message = json.loads(line)
                    with self.mutex:
                        self.current_transform = self.extract_all_transform(message)
                        self.current_key_state = self.extract_key_state(message)
        except Exception as e:
            _logger.info(f"Client disconnected: {e}")
        finally:
            self.clients.discard(writer)  # 移除客户端 writer
            writer.close()
            await writer.wait_closed()

    def send_force_message(self, l_hand_force, r_hand_force):
        message = json.dumps({"l_hand_force": l_hand_force, "r_hand_force": r_hand_force})
        asyncio.run_coroutine_threadsafe(self._broadcast_message(message), self.loop)

    async def _broadcast_message(self, message):
        clients = list(self.clients)
        for writer in clients:
            try:
                writer.write((message + '\n').encode('utf-8'))
                await writer.drain()
            except Exception as e:
                _logger.error(f"send message error: {e}")
                self.clients.discard(writer)
                writer.close()

    def bind_key_event(self, key: str, event: Callable):
        if key in self.key_event:
            raise ValueError(f"Key {key} already bound")
        self.key_event[key] = event

    def update(self, keys: list[PicoJoystickKey]):
        transform = self.get_transform_list()
        key_state = self.get_key_state()
        for key in keys:
            if key in self.key_event:
                self.key_event[key](transform, key_state)

    def is_reset_pos(self):
        return self.reset_pos

    def set_reset_pos(self, reset_pos):
        self.reset_pos = reset_pos

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

    def extract_single_transform(self, body):
        position = body["position"]
        p_x = position["x"]
        p_y = position["y"]
        p_z = position["z"]
        pos = transform_position_to_mujoco(np.array([p_x, p_y, p_z]))

        rotation = body["rotation"]
        r_x = rotation["x"]
        r_y = rotation["y"]
        r_z = rotation["z"]
        r_w = rotation["w"]
        quat = transform_quaternion_to_mujoco([r_x, r_y, r_z, r_w])

        return [np.array([pos[0], pos[1], pos[2]]), np.array([quat[3], quat[0], quat[1], quat[2]])]

    def extract_all_transform(self, message):
        left_hand_transform = self.extract_single_transform(message["leftHand"])
        right_hand_transform = self.extract_single_transform(message["rightHand"])
        motion_trackers_transform = []
        if "motionTrackers" in message:
            for motionTracker in message["motionTrackers"]:
                motion_trackers_transform.append(self.extract_single_transform(motionTracker))
        return [left_hand_transform, right_hand_transform, motion_trackers_transform]

    def extract_key_state(self, message) -> dict:
        # unity里面是左手系， 这里的值是Unity里面的值
        left_hand_key_state = {
            "triggerValue": message["leftHand"]["triggerValue"],
            "primaryButtonPressed": message["leftHand"]["primaryButtonPressed"],
            "secondaryButtonPressed": message["leftHand"]["secondaryButtonPressed"],
            "joystickPosition": [
                message["leftHand"]["joystickPosition"]["x"],
                message["leftHand"]["joystickPosition"]["y"]
            ],
            "joystickPressed": message["leftHand"]["joystickPressed"],
            "gripButtonPressed": message["leftHand"]["gripButtonPressed"],
            "position": list(message["leftHand"]["position"].values()),
            "rotation": list(message["leftHand"]["rotation"].values())
        }

        right_hand_key_state = {
            "triggerValue": message["rightHand"]["triggerValue"],
            "primaryButtonPressed": message["rightHand"]["primaryButtonPressed"],
            "secondaryButtonPressed": message["rightHand"]["secondaryButtonPressed"],
            "joystickPosition": [
                message["rightHand"]["joystickPosition"]["x"],
                message["rightHand"]["joystickPosition"]["y"]
            ],
            "joystickPressed": message["rightHand"]["joystickPressed"],
            "gripButtonPressed": message["rightHand"]["gripButtonPressed"],
            "position": list(message["rightHand"]["position"].values()),
            "rotation": list(message["rightHand"]["rotation"].values())
        }
        return {"leftHand": left_hand_key_state, "rightHand": right_hand_key_state}

    def get_right_gripButton(self) -> bool:
        key_state = self.get_key_state()
        if key_state is None:
            return False
        right_hand_key_state = key_state["rightHand"]
        if right_hand_key_state["gripButtonPressed"]:
            return True

        return False
    
    def get_left_gripButton(self) -> bool:
        key_state = self.get_key_state()
        if key_state is None:
            return False
        left_hand_key_state = key_state["leftHand"]
        if left_hand_key_state["gripButtonPressed"]:
            return True

        return False

    def get_success_AB(self) -> bool:
        key_state = self.get_key_state()
        if key_state is None:
            return False
        right_hand_key_state = key_state["rightHand"]
        if right_hand_key_state["primaryButtonPressed"] and right_hand_key_state["secondaryButtonPressed"]:
            self.suceess_AB += 1
        else:
            self.suceess_AB = 0
        return self.suceess_AB >= 20

    def get_success_XY(self) -> bool:
        key_state = self.get_key_state()
        if key_state is None:
            return False
        left_hand_key_state = key_state["leftHand"]
        if left_hand_key_state["primaryButtonPressed"] and left_hand_key_state["secondaryButtonPressed"]:
            self.suceess_XY += 1
        else:
            self.suceess_XY = 0
        return self.suceess_XY >= 20

    def get_left_relative_move(self, transform):
        relative_position = transform[0][0]
        relative_rotation = transform[0][1]
        return [relative_position, relative_rotation]

    def get_right_relative_move(self, transform):
        relative_position = transform[1][0]
        relative_rotation = transform[1][1]
        return [relative_position, relative_rotation]

    def get_motion_trackers_relative_move(self, transform):
        motion_trackers_relative_move = []
        for i in range(len(transform[2])):
            relative_position = transform[2][i][0]
            relative_rotation = transform[2][i][1]
            motion_trackers_relative_move.append([relative_position, relative_rotation])
        return motion_trackers_relative_move

    def get_relative_move(self, transform):
        left_relative_move = self.get_left_relative_move(transform)
        right_relative_move = self.get_right_relative_move(transform)
        motion_trackers_relative_move = self.get_motion_trackers_relative_move(transform)
        return [left_relative_move, right_relative_move, motion_trackers_relative_move]

if __name__ == "__main__":
    with PicoJoystick() as pico_joystick:
        try:
            while True:
                # 主线程可以执行其他任务
                pass
        except KeyboardInterrupt:
            pass  # 上下文管理器会自动调用 close()
