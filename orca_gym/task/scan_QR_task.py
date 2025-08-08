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
        self._set_target_object_(env, data)
        if sample_range > 0.0:
            self.resample_objects(env, sample_range)
        self.__random_count__ += 1

    def is_success(self, env: OrcaGymLocalEnv):
        pos, _, quat = env.get_body_xpos_xmat_xquat([env.body(self.target_object), env.body(self.goal_bodys[0])])
        target_pos, goal_pos = pos[:3], pos[3:6]
        target_quat, goal_quat = quat[:4], quat[4:8]
        is_success = self._is_facing_(target_pos, target_quat, goal_pos, goal_quat, 75, 0.3)
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
        forward_x = 2 * (x * y - w * z)
        forward_y = w * w - x * x + y * y - z * z
        forward_z = 2 * (y * z + w * x)
        forward = np.array([forward_x, forward_y, forward_z])
        norm = np.linalg.norm(forward)
        if norm < 1e-10:
            return forward
        return forward / norm

    def _is_facing_(self, posA, quatA, posB, quatB, tolerance_deg=60, tolerance_distance=0.2) -> bool:
        """
        检查物体A是否面向物体B
        :param posA: 物体A的位置
        :param quatA: 物体A的四元数
        :param posB: 物体B的位置
        :param quatB: 物体B的四元数
        :param tolerance_deg: 面向的角度偏差
        :param tolerance_distance: 面向的距离偏差
        :return: 是否面向
        """
        cos_tolerance = np.cos(np.radians(tolerance_deg ))
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

        distance = np.linalg.norm(dir_A_to_B)
        if distance < 1e-10 or distance > tolerance_distance:
            print(f"Distance too small or too large: {distance}")
            return False  # 避免除以零

        dir_A_to_B_normalized = dir_A_to_B / distance

        dir_B_to_A_normalized = -dir_A_to_B_normalized

        # 计算关键点积
        dot_forward = np.dot(forwardA, forwardB)
        dot_A_to_B = np.dot(forwardA, dir_A_to_B_normalized)
        dot_B_to_A = np.dot(forwardB, dir_B_to_A_normalized)

        return (
                dot_forward <= cos_opposite and
                dot_A_to_B >= cos_tolerance and
                dot_B_to_A >= cos_tolerance
        )

    def get_language_instruction(self) -> str:
        if not self.target_object:
            return "Do something."
        obj_str = "object: " + self.target_object
        goal_str = "goal: " + self.goal_bodys[0]

        # return f"level: {self.level_name}  use {goal_str} scan {obj_str} QR code"
        return f"In the {self.level_name} scene, pick up the {self.target_object} and scan it with the {self.goal_bodys[0]}."

