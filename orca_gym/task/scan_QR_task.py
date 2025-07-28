import re

import numpy as np
from colorama import Fore, Style
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.task.abstract_task import AbstractTask
import random

class ScanQRTask(AbstractTask):
    def __init__(self, config: dict):

        if config.__len__():
            super().__init__(config)
        else:
            super().__init__()
        self.target_object = None

    def get_task(self, env: OrcaGymLocalEnv):
        # 随机灯光说明是增广任务
        if self.random_light:
            self._get_augmentation_task_(env, self.data, sample_range=self.sample_range)
        else:
            self._get_teleperation_task_(env)
        print(
            f"{Fore.WHITE}level: {self.level_name}{Style.RESET_ALL}  "
            f"object: {Fore.CYAN}{Style.BRIGHT}{self.target_object}{Style.RESET_ALL}  to  "
            f"goal:   {Fore.MAGENTA}{Style.BRIGHT}{self.goal_bodys[0]}{Style.RESET_ALL}"
        )

    def _get_teleperation_task_(self, env: OrcaGymLocalEnv) -> dict[str, str]:
        """
        随机选一个 object_bodys 里的物体，配对到第一个 goal_bodys。
        """
        self.generate_object(env, 3, 5)

        self.random_objs_and_goals(env, random_rotation=True)

            # 如果没有可用的物体或目标，直接返回空
        if not self.object_bodys or not self.goal_bodys:
            return None

            # 从 object_bodys 随机选一个
        self.target_object = random.choice(self.object_bodys)
        print(f"self.target_object: {self.target_object}")
        # 只取第一个 goal
        goal_name = self.goal_bodys[0]
        return self.target_object

    def _get_augmentation_task_(self, env: OrcaGymLocalEnv, data: dict, sample_range=0.0) -> dict[str, str]:
        '''
        获取一个增广任务
        '''
        self._restore_objects_(env, data['objects'])
        self._restore_goals_(env, data['goals'])
        self._set_target_object_(env, data)
        if sample_range > 0.0:
            self._resample_objects_(env, data, sample_range)
        self.__random_count__ += 1

    def _restore_objects_(self, env: OrcaGymLocalEnv, objects_data):
        """
        恢复物体到指定位置
        :param positions: 物体位置字典
        """
        qpos_dict = {}
        arr = objects_data
        for entry in arr:
            name = entry['joint_name']
            pos = entry['position']
            quat = entry['orientation']
            qpos_dict[name] = np.concatenate([pos, quat], axis=0)

        env.set_joint_qpos(qpos_dict)

        env.mj_forward()

    def _restore_goals_(self, env: OrcaGymLocalEnv, goals_data):
        arr = goals_data
        if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
            self.goals = arr.copy()
            return

        # 2) 否则把它变为 (num_goals, 16) 的纯数值数组
        flat = np.asarray(arr, dtype=np.float32)
        if flat.ndim == 1:
            flat = flat.reshape(-1, 16)
        elif flat.ndim == 2 and flat.shape[0] > 1:
            # 如果是时序数据，取第一帧
            flat = flat[0].reshape(-1, 16)

        # joint_name 列表从旧的 self.goals 拿，如果第一次用请先跑一次 reset_model() 初始化它
        names = [entry['joint_name'] for entry in self.goals]

        # 3) 重建结构化数组
        goal_dtype = np.dtype([
            ('joint_name', 'U100'),
            ('position', 'f4', (3,)),
            ('orientation', 'f4', (4,)),
            ('min', 'f4', (3,)),
            ('max', 'f4', (3,)),
            ('size', 'f4', (3,))
        ])
        entries = []
        for idx, row in enumerate(flat):
            name = names[idx]
            pos = row[0:3].tolist()
            quat = row[3:7].tolist()
            mn = row[7:10].tolist()
            mx = row[10:13].tolist()
            sz = row[13:16].tolist()
            entries.append((name, pos, quat, mn, mx, sz))

        self.goals = np.array(entries, dtype=goal_dtype)

    def _set_target_object_(self, env: OrcaGymLocalEnv, data: dict):
        lang_instr = data.get("language_instruction", b"")
        if isinstance(lang_instr, (bytes, bytearray)):
            lang_instr = lang_instr.decode("utf-8")
        obj_match = re.search(r'object:\s*([^\s]+)', lang_instr)
        self.target_object = obj_match.group(1) if obj_match else None

    def _resample_objects_(self, env: OrcaGymLocalEnv, data: dict, sample_range: float = 0.0):
        target_obj_joint_name = env.joint(self.target_object + "_joint")
        target_obj_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        objects = data.get("objects", [])
        for obj in objects:
            joint_name = obj["joint_name"].decode('utf-8') if isinstance(obj["joint_name"], (bytes, bytearray)) else \
            obj["joint_name"]
            if target_obj_joint_name == joint_name:
                target_obj_position = np.array(obj["position"], dtype=np.float32)
                break

        resample_success = False
        for i in range(100):
            self.random_objs_and_goals(env, False, target_obj_joint_name)
            target_obj_joint_qpos = env.query_joint_qpos([target_obj_joint_name])[target_obj_joint_name]
            target_obj_position_delta = np.linalg.norm(target_obj_joint_qpos[:2] - target_obj_position[:2])
            if target_obj_position_delta < sample_range:
                resample_success = True
                break


    def is_success(self, env: OrcaGymLocalEnv):
        pos, _, quat = env.get_body_xpos_xmat_xquat([env.body(self.target_object), env.body(self.goal_bodys[0])])
        target_pos, goal_pos = pos[:3], pos[3:6]
        target_quat, goal_quat = quat[:4], quat[4:8]
        is_success = self._is_facing_(target_pos, target_quat, goal_pos, goal_quat, 120.0)
        print(f"task is success: {is_success}")
        return is_success

    def _normalize_quaternion_(self, quat: np.ndarray) -> np.ndarray:
        """归一化四元数"""
        w, x, y, z = quat
        norm = np.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
        return np.array([w / norm, x / norm, y / norm, z / norm])

    def _quaternion_forward_vector_(selt, quat: np.ndarray) -> np.ndarray:
        """计算当前前向向量，基于初始前向 (0, 1, 0)"""
        w, x, y, z = quat
        # 直接公式计算
        forward_x = 2 * (x * y + w * z)
        forward_y = w * w - x * x + y * y - z * z
        forward_z = 2 * (y * z - w * x)
        forward = np.array([forward_x, forward_y, forward_z])
        # 归一化防止浮点误差
        norm = np.linalg.norm(forward)
        if norm < 1e-10:
            return forward
        return forward / norm

    def _is_facing_(self, posA, quatA, posB, quatB, tolerance_deg=60.0) -> bool:
        """
        检查物体A是否面向物体B
        :param posA: 物体A的位置
        :param quatA: 物体A的四元数
        :param posB: 物体B的位置
        :param quatB: 物体B的四元数
        :param tolerance_deg: 面向的角度偏差
        :return: 是否面向
        """
        cos_tolerance = np.cos(np.radians(tolerance_deg))
        cos_opposite = np.cos(np.radians(180 - tolerance_deg))

        posA = np.array(posA)
        posB = np.array(posB)

        # 归一化四元数
        quatA_norm = self._normalize_quaternion_(quatA)
        quatB_norm = self._normalize_quaternion_(quatB)

        # 计算前向向量
        forwardA = self._quaternion_forward_vector_(quatA_norm)
        forwardB = self._quaternion_forward_vector_(quatB_norm)

        # 计算方向向量
        dir_A_to_B = posB - posA
        if np.linalg.norm(dir_A_to_B) < 1e-10:
            return False  # 避免除以零
        dir_A_to_B_normalized = dir_A_to_B / np.linalg.norm(dir_A_to_B)

        dir_B_to_A = posA - posB
        dir_B_to_A_normalized = dir_B_to_A / np.linalg.norm(dir_B_to_A)

        # 计算关键点积
        dot_forward = np.dot(forwardA, forwardB)
        dot_A_to_B = np.dot(forwardA, dir_A_to_B_normalized)
        dot_B_to_A = np.dot(forwardB, dir_B_to_A_normalized)

        # 判断对视条件（允许60度偏差）
        return (
                dot_forward <= cos_opposite and  # 两个前向向量夹角在120°~180°之间
                dot_A_to_B >= cos_tolerance and  # A的前向与A到B方向夹角≤60°
                dot_B_to_A >= cos_tolerance  # B的前向与B到A方向夹角≤60°
        )

    def get_language_instruction(self) -> str:
        if not self.target_object:
            return "Do something."
        obj_str = "object: " + self.target_object
        goal_str = "goal: " + self.goal_bodys[0]

        return f"level: {self.level_name}  use {goal_str} scan {obj_str} QR code"

