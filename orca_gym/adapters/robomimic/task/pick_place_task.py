from colorama import Fore, Style
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.adapters.robomimic.task.abstract_task import AbstractTask
from orca_gym.utils import rotations
import random
from typing import Optional


import json
import re

import numpy as np

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
        # 限制选择数量不超过可用的 actors 数量
        if self.random_actor and self.actors:
            pick_max = min(5, len(self.actors))
            pick_min = min(3, len(self.actors))
            self.generate_object(env, pick_min, pick_max)
        else:
            self.generate_object(env, 3, 5)
        while True:
            # 随机摆放
            self.random_objs_and_goals(env, random_rotation=True)

            # 如果没有可用的物体或目标，直接返回空
            if not self.object_bodys or not self.goal_bodys:
                return None

            if getattr(self, "success_objects", None):
                candidates = [obj for obj in self.success_objects if obj in self.object_bodys]
            else:
                candidates = list(self.object_bodys)
            if not candidates:
                return None

            # 默认使用列表中的第一个作为语言提示用的目标
            self.target_object = candidates[0]

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
        spawn_idx = int(np.clip(self.spawn_goal_index, 0, max(len(goal_joints) - 1, 0))) if goal_joints else 0
        success_idx = int(np.clip(self.success_goal_index, 0, max(len(goal_joints) - 1, 0))) if goal_joints else 0

        for idx, goal_joint_name in enumerate(goal_joints):
            qpos = goal_positions[goal_joint_name]
            # 获取目标的尺寸
            goal_name = goal_joint_name.replace("_joint", "")
            info = env.get_goal_bounding_box(goal_name)

            bbox_valid = (
                info is not None
                and "min" in info
                and "max" in info
                and np.all(np.isfinite(info["min"]))
                and np.all(np.isfinite(info["max"]))
            )

            tol = float(getattr(self, "success_tolerance", 0.05))
            use_joint_center = bool(getattr(self, "success_use_joint_center", False)) and idx == success_idx

            if use_joint_center or not bbox_valid:
                center = qpos[:3].flatten()
                mn = center - tol
                mx = center + tol
                info = {"min": mn.copy(), "max": mx.copy(), "size": (mx - mn).copy()}
            else:
                mn = np.array(info["min"], dtype=np.float32).flatten()
                mx = np.array(info["max"], dtype=np.float32).flatten()
                mn = mn - tol
                mx = mx + tol

            sz = np.array(mx - mn, dtype=np.float32)

            if idx == success_idx:
                role = "success"
            elif idx == spawn_idx:
                role = "spawn"
            else:
                role = "aux"

            # 添加目标位置信息
            goal_entries.append({
                "joint_name": goal_name,
                "position": qpos[:3].tolist(),
                "orientation": qpos[3:].tolist(),
                "min": mn.tolist(),
                "max": mx.tolist(),
                "size": sz.tolist(),
                "role": role,
            })

            goal_positions_list.append(qpos[:3])  # 仅记录目标位置

        goal_positions_array = np.array(goal_positions_list)

        # 返回目标数据及bounding box信息
        return goal_entries, goal_positions_array

    def is_success(self, env: OrcaGymLocalEnv):
        goal_entries, goal_positions = self.process_goals(env)
        if not goal_entries or not len(goal_positions):
            return False

        success_idx = int(np.clip(self.success_goal_index, 0, len(goal_entries) - 1))
        goal_entry = goal_entries[success_idx]
        center = np.array(goal_entry["position"], dtype=np.float32)
        orientation = np.array(goal_entry["orientation"], dtype=np.float32)
        rot_mat = rotations.quat2mat(orientation)
        offset_local = getattr(self, "success_center_offset", np.zeros(3, dtype=np.float32))
        if offset_local is None:
            offset_local = np.zeros(3, dtype=np.float32)
        offset_local = np.asarray(offset_local, dtype=np.float32).flatten()
        center_adjusted = center + rot_mat @ offset_local

        bounds_cfg = getattr(self, "success_local_bounds", None) or {}
        tol = float(getattr(self, "success_tolerance", 0.05))

        size = np.array(goal_entry.get("size", [tol * 2, tol * 2, tol * 2]), dtype=np.float32)
        half_default = np.maximum(size * 0.5, tol)

        half_xy = bounds_cfg.get("half")
        if half_xy is None:
            half_x, half_y = float(half_default[0]), float(half_default[1])
        else:
            if isinstance(half_xy, (int, float)):
                half_x = half_y = float(half_xy)
            else:
                half_x = float(half_xy[0])
                half_y = float(half_xy[1] if len(half_xy) > 1 else half_xy[0])
        above = float(bounds_cfg.get("above", max(half_default[2], tol)))
        below = float(bounds_cfg.get("below", max(half_default[2], tol)))
        debug_enabled = bool(getattr(self, "success_debug", False))

        candidate_objects = []
        if getattr(self, "success_objects", None):
            candidate_objects = list(self.success_objects)
        elif self.object_bodys:
            candidate_objects = list(self.object_bodys)
        elif self.target_object:
            candidate_objects = [self.target_object]

        any_success = False
        for obj_name in candidate_objects:
            try:
                target_body_name = self._find_body_name(env, obj_name)
            except Exception:
                continue
            target_pos, _, _ = env.get_body_xpos_xmat_xquat([target_body_name])
            pos_vec = target_pos[0] if hasattr(target_pos, 'ndim') and target_pos.ndim > 1 else target_pos

            rel_world = pos_vec - center_adjusted
            local_coord = rot_mat.T @ rel_world

            within_x = -half_x <= local_coord[0] <= half_x
            within_y = -half_y <= local_coord[1] <= half_y
            within_z = -below <= local_coord[2] <= above
            success = within_x and within_y and within_z

            if debug_enabled:
                print(f"[SUCCESS CHECK] Object: {obj_name}")
                print(f"  World position: [{pos_vec[0]:.3f}, {pos_vec[1]:.3f}, {pos_vec[2]:.3f}]")
                print(f"  Local position: [{local_coord[0]:.3f}, {local_coord[1]:.3f}, {local_coord[2]:.3f}]")
                print(f"  Local bounds: x±{half_x:.3f}, y±{half_y:.3f}, z∈[-{below:.3f}, {above:.3f}]")
                print(f"  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")

            if success:
                any_success = True
                break

        return any_success

    def get_language_instruction(self) -> str:
        if getattr(self, "success_objects", None):
            if len(self.success_objects) == 1:
                obj_str = "object: " + self.success_objects[0]
            else:
                joined = ", ".join(self.success_objects)
                obj_str = f"object: any of ({joined})"
        elif self.target_object:
            obj_str = "object: " + self.target_object
        else:
            return "Do something."
        success_idx = int(np.clip(self.success_goal_index, 0, max(len(self.goal_bodys) - 1, 0))) if self.goal_bodys else 0
        goal_str = "goal: " + self.goal_bodys[success_idx] if self.goal_bodys else "goal: unknown"

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
