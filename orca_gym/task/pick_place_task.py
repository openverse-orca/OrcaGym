import re

import numpy as np
from colorama import Fore, Style
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.task.abstract_task import AbstractTask
import random

class PickPlaceTask(AbstractTask):
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
        while True:
            # 随机摆放
            self.random_objs_and_goals(env, random_rotation=True)

            # 如果没有可用的物体或目标，直接返回空
            if not self.object_bodys or not self.goal_bodys:
                return None

            # 从 object_bodys 随机选一个
            self.target_object = random.choice(self.object_bodys)
            print(f"self.target_object: {self.target_object}")
            # 只取第一个 goal
            goal_name = self.goal_bodys[0]

            # 记录到 task_dict 并返回

            objs = self.randomized_object_positions
            goal_body = self.goal_bodys[0]

            return self.target_object

    def _get_augmentation_task_(self, env: OrcaGymLocalEnv, data: dict, sample_range = 0.0) ->dict[str, str]:
        '''
        获取一个增广任务
        '''
        self._restore_objects_(env, data['objects'])
        self._set_target_object_(env, data)
        if sample_range > 0.0:
            self.resample_objects(env, sample_range)
        self.__random_count__ += 1

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
            joint_name = obj["joint_name"].decode('utf-8') if isinstance(obj["joint_name"], (bytes, bytearray)) else obj["joint_name"]
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

    def process_goals(self, env:OrcaGymLocalEnv):
        """
        处理目标（goals）的信息，返回目标的bounding box（最大最小坐标）。
        :param goal_positions: 目标位置字典
        :return: 目标信息的条目，目标位置数组，和bounding box数据
        """
        goal_joints = [env.joint(jn) for jn in self.goal_joints]
        goal_positions = env.query_joint_qpos(goal_joints)

        goal_entries = []
        goal_positions_list = []
        goal_bounding_boxes = {}  # 用于存储目标的bounding box信息

        for goal_joint_name, qpos in goal_positions.items():
            # 获取目标的尺寸
            goal_name = goal_joint_name.replace("_joint", "")
            info = env.get_goal_bounding_box(goal_name)

            # 如果没有尺寸信息，跳过目标
            if not info:
                print(f"Error: No geometry size information found for goal {goal_name}")
                continue

            mn = np.array(info["min"]).flatten()
            mx = np.array(info["max"]).flatten()
            sz = mx - mn

            # 添加目标位置信息
            goal_entries.append({
                "joint_name": goal_name,
                "position": qpos[:3].tolist(),
                "orientation": qpos[3:].tolist(),
                "min": mn.tolist(),
                "max": mx.tolist(),
                "size": sz.tolist()
            })

            goal_positions_list.append(qpos[:3])  # 仅记录目标位置

        goal_positions_array = np.array(goal_positions_list)

        # 返回目标数据及bounding box信息
        return goal_entries, goal_positions_array

    def is_success(self, env: OrcaGymLocalEnv):
        target_pos, _, _ = env.get_body_xpos_xmat_xquat([env.body(self.target_object)])
        pos_vec = target_pos[0] if hasattr(target_pos, 'ndim') and target_pos.ndim > 1 else target_pos
        xy = pos_vec[:2]

        goal_entries, goal_positions = self.process_goals(env)
        #只有一个goal, 只取第一个
        goal_entry, goal_position = goal_entries[0], goal_positions[0]
        # 获取目标区域边界
        gmin = goal_entry['min'][:2]
        gmax = goal_entry['max'][:2]
        if xy[0] < gmin[0] or xy[0] > gmax[0] or xy[1] < gmin[1] or xy[1] > gmax[1]:
            return False
        return True

    def get_language_instruction(self) -> str:
        if not self.target_object:
            return "Do something."
        obj_str = "object: " + self.target_object
        goal_str = "goal: " + self.goal_bodys[0]

        return f"level: {self.level_name}  {obj_str} to {goal_str}"

