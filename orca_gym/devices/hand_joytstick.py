import asyncio
import json
import threading
import numpy as np
from numpy import arccos, array
from numpy.linalg import norm
from websockets.asyncio.server import serve
import copy
from scipy.spatial.transform import Rotation as R
import threading

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


mediapipe_hand_point_count = 21
openloong_hand_point_count = 11

# line1 line2 direction
point_mapping = [
    [[0, 1], [1, 2], -1],
    [[1, 2], [2, 3], 1],
    [[2, 3], [3, 4], 1],
    [[0, 5], [5, 6], -1],
    [[5, 6], [6, 7], -1],
    [[0, 9], [9, 10], -1],
    [[9, 10], [10, 11], -1],
    [[0, 13], [13, 14], -1],
    [[13, 14], [14, 15], -1],
    [[0, 17], [17, 18], -1],
    [[17, 18], [18, 19], -1]
]

def calculate_angle(u, v):
    return arccos(u.dot(v)/(norm(u)*norm(v)))

class HandPointLocation:
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.z = 0
        self.visibility = 0

class HandInfo:
    def __init__(self) -> None:
        self.hand_index = 0  # 1 for left, 0 for right
        self.hand_points = [HandPointLocation() for i in range(mediapipe_hand_point_count)]
        self.qpos = np.zeros(openloong_hand_point_count)

    def __repr__(self) -> str:
        return f"hand_index: {self.hand_index}, qpos: {self.qpos}"


class HandJoystick:
    def __init__(self):
        self.mutex= threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()
        self.current_hand_infos = None
    
    def __del__(self):
        self.running = False
        self.thread.join()

    def _loop(self):
        asyncio.run(self._start_server())

    async def _handler(self, websocket):
        _logger.info("new connection")
        try:
            async for message in websocket:
                # print(message)
                hand_infos = self.parse_message(message)
                # print(hand_infos)
                with self.mutex:
                    self.current_hand_infos = hand_infos
        except Exception as e:
            _logger.info(f"disconnected {e}")

    async def _start_server(self):
        async with serve(self._handler, "0.0.0.0", 8787, ping_interval=None):
            await asyncio.get_running_loop().create_future()  # run forever

    def calculate_qpos(self, hand_infos: list[HandInfo]):
        for hand_info in hand_infos:
            for index, point_map in enumerate(point_mapping):
                point1 = hand_info.hand_points[point_map[0][0]]
                point2 = hand_info.hand_points[point_map[0][1]]
                point3 = hand_info.hand_points[point_map[1][0]]
                point4 = hand_info.hand_points[point_map[1][1]]
                u = np.array([point2.x - point1.x, point2.y - point1.y, point2.z - point1.z])
                v = np.array([point4.x - point3.x, point4.y - point3.y, point4.z - point3.z])
                angle = calculate_angle(u, v)
                hand_info.qpos[index] = angle * point_map[2]

    def parse_message(self, message):
        body = json.loads(message)
        handedness = body["handedness"]
        hand_count = len(handedness)
        hand_infos = [HandInfo() for i in range(hand_count)]
        for index in range(hand_count):
            hand_info = hand_infos[index]
            hand_info.hand_index = handedness[index][0]["index"]
            for point_index, point in enumerate(body["landmarks"][index]):
                hand_info.hand_points[point_index].x = point["x"]
                hand_info.hand_points[point_index].y = point["y"]
                hand_info.hand_points[point_index].z = point["z"]
                hand_info.hand_points[point_index].visibility = point["visibility"]
        self.calculate_qpos(hand_infos)
        return hand_infos

    def update(self):
        pass

    def get_hand_infos(self):
        with self.mutex:
            return copy.deepcopy(self.current_hand_infos)