import json
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
        self.mode = 0 #0: 关门, 1: 开门

    def get_task(self, env: OrcaGymLocalEnv):
        if self.random_light:
            self._get_augmentation_task_(env, self.data, sample_range=self.sample_range)
        else:
            self._get_teleperation_task_(env)
        if self.mode == 0:
            print(
                f"{Fore.WHITE}level: {self.level_name}{Style.RESET_ALL}  "
                f"{Fore.CYAN}{Style.BRIGHT}close {self.goal_bodys[0]}{Style.RESET_ALL} "
            )
        else:
            print(
                f"{Fore.WHITE}level: {self.level_name}{Style.RESET_ALL}  "
                f"{Fore.CYAN}{Style.BRIGHT}open {self.goal_bodys[0]}{Style.RESET_ALL} "
            )

    def _get_teleperation_task_(self, env: OrcaGymLocalEnv) -> dict[str, str]:
        """
        随机设置冰箱铰链关节的角度
        """
        self.mode = np.random.randint(0, 100) % 2
        print(f"self.mode: {self.mode}")
        if self.mode  == 0:
            # 关门
            joint_pos = np.random.uniform(0.785, 2.09)
        else:
            joint_pos = 0.0

        goal_idx = random.choice(range(len(self.object_bodys)))
        self.goal_bodys = [self.object_bodys[goal_idx]]
        self.goal_joints = [self.object_joints[goal_idx]]
        self.target_body = self.object_bodys[goal_idx]
        env.set_joint_qpos({env.joint(self.goal_joints[0]): [joint_pos]})


    def _get_augmentation_task_(self, env: OrcaGymLocalEnv, data: dict, sample_range=0.0) -> dict[str, str]:
        '''
        获取一个增广任务
        '''
        self._restore_goals_(env, data['goals'])
        self._set_target_object_(env, data)
        qpos = env.query_joint_qpos([env.joint(joint_name) for joint_name in self.goal_joints])
        for joint_name in self.goal_joints:
            joint_pos = qpos[env.joint(joint_name)]
            if joint_pos < 0.01:
                self.mode = 1
            else:
                self.mode = 0
        self.__random_count__ += 1

    def is_success(self, env: OrcaGymLocalEnv):
        """
        判断任务是否成功
        """
        joints_pos = self.get_object_joints_xpos(env)
        is_success = False
        if self.mode == 0:
            # 关门
            is_success = joints_pos[env.joint(self.goal_joints[0])] < 0.01
        else:
            is_success = joints_pos[env.joint(self.goal_joints[0])] > 1.05
        print(f"task is_success: {is_success}")
        return is_success

    def get_language_instruction(self) -> str:
        if self.mode == 0:
            return f"level: {self.level_name} Close the frigerator {self.goal_bodys} door."
        else:
            return f"level: {self.level_name} Open the frigerator {self.goal_bodys} door."