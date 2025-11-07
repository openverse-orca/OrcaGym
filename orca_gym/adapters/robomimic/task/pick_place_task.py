from colorama import Fore, Style
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.adapters.robomimic.task.abstract_task import AbstractTask
import random
from typing import Optional


import json
import re

import numpy as np
from colorama import Fore, Style
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.adapters.robomimic.task.abstract_task import AbstractTask
import random

class PickPlaceTask(AbstractTask):
    def __init__(self, config: dict):
        
        if config.__len__():
            super().__init__(config)
        else:
            super().__init__()

    def get_task(self, env: OrcaGymLocalEnv):
        from envs.manipulation.dual_arm_env import RunMode
        is_augmentation_mode = (env._run_mode == RunMode.POLICY_NORMALIZED and 
                            hasattr(env, '_task') and 
                            hasattr(env._task, 'data') and 
                            env._task.data is not None)
        if is_augmentation_mode:
            self._get_augmentation_task_(env, self.data, sample_range=self.sample_range)
        else:
            self._get_teleperation_task_(env)

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


    def _resample_objects_(self, env: OrcaGymLocalEnv, data: dict, sample_range: float = 0.0):
        target_obj_joint_name = self._find_joint_name(env, self.target_object + "_joint")
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
        goal_joints = [self._find_joint_name(env, jn) for jn in self.goal_joints]
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
        target_body_name = self._find_body_name(env, self.target_object)
        target_pos, _, _ = env.get_body_xpos_xmat_xquat([target_body_name])
        pos_vec = target_pos[0] if hasattr(target_pos, 'ndim') and target_pos.ndim > 1 else target_pos
        
        goal_entries, goal_positions = self.process_goals(env)
        #只有一个goal, 只取第一个
        goal_entry, goal_position = goal_entries[0], goal_positions[0]
        # 获取目标区域边界（包括z轴）
        gmin = goal_entry['min']  # [x_min, y_min, z_min]
        gmax = goal_entry['max']  # [x_max, y_max, z_max]
        
        # 检查xyz三个维度是否都在目标区域内
        success = not (pos_vec[0] < gmin[0] or pos_vec[0] > gmax[0] or 
                       pos_vec[1] < gmin[1] or pos_vec[1] > gmax[1] or
                       pos_vec[2] < gmin[2] or pos_vec[2] > gmax[2])
        
        # 打印调试信息
        print(f"[SUCCESS CHECK] Object: {self.target_object}")
        print(f"  Position: [{pos_vec[0]:.3f}, {pos_vec[1]:.3f}, {pos_vec[2]:.3f}]")
        print(f"  Goal min: [{gmin[0]:.3f}, {gmin[1]:.3f}, {gmin[2]:.3f}]")
        print(f"  Goal max: [{gmax[0]:.3f}, {gmax[1]:.3f}, {gmax[2]:.3f}]")
        print(f"  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        return success

    def get_language_instruction(self) -> str:
        if not self.target_object:
            return "Do something."
        obj_str = "object: " + self.target_object
        goal_str = "goal: " + self.goal_bodys[0]

        return f"level: {self.level_name}  {obj_str} to {goal_str}"
        # return f"In the level {self.level_name} put the {self.target_object} into the {self.goal_bodys[0]}."
        # return f"In the {self.level_name} scene, put the {self.target_object} into the {self.goal_bodys[0]}."


class EmptyTask(AbstractTask):
    """Minimal task that keeps the scene empty for free-form teleoperation."""

    def __init__(self, grpc_addr: Optional[str] = None):
        empty_config = {
            "random_object": False,
            "object_bodys": [],
            "object_sites": [],
            "object_joints": [],
            "random_goal": False,
            "goal_bodys": [],
            "goal_sites": [],
            "goal_joints": [],
            "level_name": "empty",
            "random_cycle": 0,
            "range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": 0.0, "r": 0.0},
            "random_actor": False,
            "actors": [],
            "actors_spawnable": [],
            "description": [],
            "random_light": False,
            "lights": [],
            "lights_spawnable": [],
            "grpc_addr": grpc_addr or "localhost:50051",
        }
        super().__init__(empty_config)
        self.is_empty = True
        self.randomized_object_positions = {}
        self.randomized_goal_positions = {}

    def spawn_scene(self, env: OrcaGymLocalEnv):
        """Override to skip any spawn logic for empty scenes."""
        return

    def generate_object(self, env: OrcaGymLocalEnv, pick_min, pick_max):
        return

    def random_objs_and_goals(self, env: OrcaGymLocalEnv, random_rotation=True, target_obj_joint_name=None):
        return

    def get_task(self, env: OrcaGymLocalEnv):
        self.randomized_object_positions = {}
        self.randomized_goal_positions = {}
        return None

    def get_language_instruction(self) -> str:
        return "No task loaded."

    def is_success(self, env: OrcaGymLocalEnv):
        return False

class TaskStatus:
    """
    Enum class for task status
    """
    NOT_STARTED = "not_started"
    GET_READY = "get_ready"
    BEGIN = "begin"
    SUCCESS = "success"
    FAILURE = "failure"
    END = "end"
