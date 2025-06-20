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
        self.task_dict.clear()
        self.random_objs_and_goals(env, bounds=0.1)
    
        object_len = len(self.object_bodys)
        goal_len   = len(self.goal_bodys)
    
        if object_len == 0 or goal_len == 0:
            return self.task_dict
    
        # 随机选一个 object，goal 总是 goal_bodys[0]
        obj_idx = random.randint(0, object_len - 1)
        obj_name = self.object_bodys[obj_idx]
        goal_name = self.goal_bodys[0]
    
        self.task_dict[obj_name] = goal_name
    
        return self.task_dict


    def get_language_instruction(self) -> str:
        if not self.task_dict:
            return "Do something."
        obj_str = "object: " + " ".join(self.task_dict.keys())
        goal_str = "goal: " + " ".join(self.task_dict.values())
        return f"level: {self.level_name}  {obj_str} to {goal_str}"

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
