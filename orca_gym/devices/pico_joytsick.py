import asyncio
import json
import threading
import numpy as np
from websockets.asyncio.server import serve
import copy
import threading
from scipy.spatial.transform import Rotation as R

   
# def quaternion_to_matrix(q):
#     # 将四元数转换为旋转矩阵
#     R_matrix = R.from_quat(q).as_matrix()
#     return R_matrix

# def apply_coordinate_transform(R_matrix):
#     # 定义坐标轴变换矩阵（交换 Y 和 Z 轴）
#     T = np.array([[1, 0, 0],
#                 [0, 0, 1],
#                 [0, 1, 0]])
#     # 应用坐标轴变换
#     R_transformed = T @ R_matrix @ T.T
#     return R_transformed

# def correct_rotation_matrix(R_matrix):
#     # 使用 SVD 分解
#     U, _, Vt = np.linalg.svd(R_matrix)
#     # 重新计算旋转矩阵
#     R_corrected = U @ Vt
#     # 确保行列式为 1
#     if np.linalg.det(R_corrected) < 0:
#         U[:, -1] *= -1
#         R_corrected = U @ Vt
#     return R_corrected

# def matrix_to_quaternion(R_matrix):
#     # 将旋转矩阵转换回四元数
#     R_obj = R.from_matrix(R_matrix)
#     q = R_obj.as_quat()
#     return q

# def transform_quaternion(q):
#     # 手系转换：对 Y 和 Z 分量取反（从左手系到右手系）
#     q_lhs = np.array([q[0], q[1], q[2], q[3]])

#     # 将四元数转换为旋转矩阵
#     R_matrix = quaternion_to_matrix(q_lhs)

#     # 应用坐标轴变换
#     R_transformed = apply_coordinate_transform(R_matrix)

#     # 修正旋转矩阵
#     R_transformed = correct_rotation_matrix(R_transformed)

#     # 将旋转矩阵转换回四元数
#     q_transformed = matrix_to_quaternion(R_transformed)

#     return q_transformed


# def transform_position(p):
#     # 手系转换：对 Z 轴取反
#     p_rhs = np.array([p[0], p[1], -p[2]])
#     # 坐标轴互换：交换 Y 和 Z 轴
#     p_transformed = np.array([p_rhs[0], p_rhs[2], p_rhs[1]])
#     return p_transformed

# def transform_quaternion(q):
#     """
#     将四元数从 Unity 左手系（Y-up，Z-forward）转换到目标右手系（Z-up，Y-forward）
#     """
#     # 将四元数转换为旋转矩阵
#     R_matrix = quaternion_to_matrix(q)

#     # 应用坐标系转换
#     R_transformed = transform_rotation_matrix(R_matrix)

#     # 将旋转矩阵转换回四元数
#     q_transformed = matrix_to_quaternion(R_transformed)
#     return q_transformed

# def quaternion_to_matrix(q):
#     R_matrix = R.from_quat(q).as_matrix()
#     return R_matrix

# def transform_rotation_matrix(R_matrix):
#     T = np.array([
#         [1, 0, 0],
#         [0, 0, 1],
#         [0, -1, 0]
#     ])
#     T_inv = T.T
#     R_transformed = T @ R_matrix @ T_inv
#     return R_transformed

# def matrix_to_quaternion(R_matrix):
#     q = R.from_matrix(R_matrix).as_quat()
#     return q

# def transform_position(p):
#     """
#     将位置向量从 Unity 左手系（Y-up，Z-forward）转换到目标右手系（Z-up，Y-forward）
#     """
#     T = np.array([
#         [1, 0, 0],
#         [0, 0, 1],
#         [0, -1, 0]
#     ])
#     p_transformed = T @ p
#     return p_transformed

def transform_position_to_mujoco(p_unity):
    x_mujoco = p_unity[2]
    y_mujoco = -p_unity[0]
    z_mujoco = p_unity[1]
    return np.array([x_mujoco, y_mujoco, z_mujoco])

def transform_quaternion_to_mujoco(q_unity):
    #q_fixed = R.from_quat(q_unity, scalar_first=False).as_quat(canonical=True)
    R_unity = R.from_quat(q_unity, scalar_first=False).as_matrix()
    T = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    R_mujoco = T @ R_unity @ T.T
    # R_mujoco = T @ R_unity 
    # R_mujoco = R_unity @ T
    # R_mujoco = R_unity 
    # q_mujoco = R.from_matrix(R_mujoco).as_quat(canonical=True)

    v = R.from_matrix(R_mujoco).as_rotvec()

    return R.from_rotvec(v).as_quat(canonical=True)
    # return R.from_quat(q_unity).as_quat(canonical=True)

    # return [ q_unity[0], q_unity[2], q_unity[1], -q_unity[3]]

class PicoJoystick:
    def __init__(self):
        self.mutex= threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()
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
        pos = transform_position_to_mujoco(np.array([p_x, p_y, p_z]))

        rotation = body["rotation"]
        r_x = rotation["x"]
        r_y = rotation["y"]
        r_z = rotation["z"]
        r_w = rotation["w"]
        quat = transform_quaternion_to_mujoco([r_x, r_y, r_z, r_w])
        
        return [np.array([pos[0], pos[1], pos[2]]), np.array([quat[3], quat[0], quat[1], quat[2]])] # 调整wxyz顺序
        # return [np.array([p_x, p_y, p_z]), np.array([r_w, r_x, r_y, r_z])]

    # def extact_single_transform_org(self, body):
    #     position = body["position"]
    #     p_x = position["x"]
    #     p_y = position["y"]
    #     p_z = position["z"]
    #     pos = transform_position_to_mujoco(np.array([p_x, p_y, p_z]))

    #     rotation = body["rotation"]
    #     r_x = rotation["x"]
    #     r_y = rotation["y"]
    #     r_z = rotation["z"]
    #     r_w = rotation["w"]
    #     quat = transform_quaternion_to_mujoco([r_x, r_y, r_z, r_w])
        
    #     # return [np.array([pos[0], pos[1], pos[2]]), np.array([quat[3], quat[0], quat[1], quat[2]])] # 调整wxyz顺序
    #     return [np.array([p_x, p_y, p_z]), np.array([r_w, r_x, r_y, r_z])]

        
    def extact_all_transform(self, message):
        left_hand_transform = self.extact_single_transform(message["leftHand"])
        # left_hand_transform_org = self.extact_single_transform_org(message["leftHand"])
        right_hand_transform = self.extact_single_transform(message["rightHand"])
        # right_hand_transform_org = self.extact_single_transform_org(message["rightHand"])
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
        relative_position = transform[0][0]
        relative_rotation = transform[0][1]
        # print("rotation: ", relative_rotation.as_euler('xzy', degrees=True))
        # relative_rotation = transform[0][1]
        # relative_rotation = np.dot(self.first_transform[0][1].T, transform[0][1])
        return [relative_position, relative_rotation]
    
    # def get_left_relative_move_org(self, transform):
    #     relative_position = transform[3][0]
    #     relative_rotation = transform[3][1]
    #     return [relative_position, relative_rotation]
    
    def get_right_relative_move(self, transform):
        relative_position = transform[1][0]
        relative_rotation = transform[1][1]
        # relative_rotation = transform[1][1]
        # relative_rotation = np.dot(self.first_transform[1][1].T, transform[1][1])
        return [relative_position, relative_rotation]
    
    # def get_right_relative_move_org(self, transform):
    #     relative_position = transform[4][0]
    #     relative_rotation = transform[4][1]
    #     return [relative_position, relative_rotation]
    
    def get_motion_trackers_relative_move(self, transform):
        motion_trackers_relative_move = []
        for i in range(len(transform[2])):
            relative_position = transform[2][i][0]
            relative_rotation = transform[2][i][1]
            # relative_rotation = np.dot(self.first_transform[2][i][1].T, transform[2][i][1])
            motion_trackers_relative_move.append([relative_position, relative_rotation])
        return motion_trackers_relative_move
    
    def get_relative_move(self, transform):
        left_relative_move = self.get_left_relative_move(transform)
        right_relative_move = self.get_right_relative_move(transform)
        motion_trackers_relative_move = self.get_motion_trackers_relative_move(transform)
        return [left_relative_move, right_relative_move, motion_trackers_relative_move]


