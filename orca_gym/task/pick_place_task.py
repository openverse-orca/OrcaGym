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
        self.task_list: list = []

    def get_task(self, env: OrcaGymLocalEnv) -> dict[str, str]:
        """
        随机选一个 object_bodys 里的物体，配对到第一个 goal_bodys。
        """
        # 每次都清空旧任务
        self.task_dict.clear()
        # 随机摆放
        self.random_objs_and_goals(env, bounds=0.1)
    
        # 如果没有可用的物体或目标，直接返回空
        if not self.object_bodys or not self.goal_bodys:
            return self.task_dict
    
        # 从 object_bodys 随机选一个
        obj_name = random.choice(self.object_bodys)
        # 只取第一个 goal
        goal_name = self.goal_bodys[0]
    
        # 记录到 task_dict 并返回
        self.task_dict[obj_name] = goal_name
        return self.task_dict




    def get_language_instruction(self) -> str:
        if not self.task_dict:
            return "Do something."

        # 拆出 objects 和 goals
        objs  = list(self.task_dict.keys())
        goals = list(self.task_dict.values())

        # 给每个 object 名和 goal 名上色
        colored_objs  = " ".join(f"{Fore.CYAN}{Style.BRIGHT}{o}{Style.RESET_ALL}" for o in objs)
        colored_goals = " ".join(f"{Fore.MAGENTA}{Style.BRIGHT}{g}{Style.RESET_ALL}" for g in goals)

        # 拼回整句
        return (
            f"{Fore.WHITE}level: {self.level_name}{Style.RESET_ALL}  "
            f"object: {colored_objs}  to  "
            f"goal: {colored_goals}"
        )

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
