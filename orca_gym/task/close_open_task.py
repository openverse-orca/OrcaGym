import re

import numpy as np
from colorama import Fore, Style
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.task.abstract_task import AbstractTask
import random

class CloseOpenTask(AbstractTask):
    def __init__(self, config: dict):
        if config.__len__():
            super().__init__(config)
        else:
            super().__init__()
        self.target_object = self.object_bodys[0]
        self.mode = 0 #0: 关门, 1: 开门

    def get_task(self, env: OrcaGymLocalEnv):
        if self.random_light:
            self._get_augmentation_task_(env, self.data, sample_range=self.sample_range)
        else:
            self._get_teleperation_task_(env)
        if self.mode == 0:
            print(
                f"{Fore.WHITE}level: {self.level_name}{Style.RESET_ALL}  "
                f"{Fore.CYAN}{Style.BRIGHT}close frigertor{Style.RESET_ALL} "
            )
        else:
            print(
                f"{Fore.WHITE}level: {self.level_name}{Style.RESET_ALL}  "
                f"{Fore.CYAN}{Style.BRIGHT}open frigertor{Style.RESET_ALL} "
            )

    def _get_teleperation_task_(self, env: OrcaGymLocalEnv) -> dict[str, str]:
        """
        随机设置冰箱铰链关节的角度
        """
        self.mode = np.random.randint(0, 100) % 2
        print(f"self.mode: {self.mode}")
        if self.mode  == 0:
            # 关门
            joint_pos = np.random.uniform(0, 2.09)
        else:
            joint_pos = 0.0
        env.set_joint_qpos({env.joint(self.object_joints[0]): [joint_pos]})


    def _get_augmentation_task_(self, env: OrcaGymLocalEnv, data: dict, sample_range=0.0) -> dict[str, str]:
        '''
        获取一个增广任务
        '''
        self._restore_objects_(env, data['objects'])

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

    def is_success(self, env: OrcaGymLocalEnv):
        """
        判断任务是否成功
        """
        joints_pos = self.get_object_joints_xpos(env)
        is_success = False
        if self.mode == 0:
            # 关门
            is_success = joints_pos[env.joint(self.object_joints[0])] < 0.01
        else:
            is_success = joints_pos[env.joint(self.object_joints[0])] > 1.05
        print(f"task is_success: {is_success}")
        return is_success

    def get_language_instruction(self) -> str:
        if self.mode == 0:
            return f"level: {self.level_name} Close the refrigerator door."
        else:
            return f"level: {self.level_name} Open the refrigerator door."