import numpy as np
from envs.manipulation.dual_arm_env import DualArmEnv
from envs.manipulation.dual_arm_robot import DualArmRobot

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class OpenLoongNoHandFixBase(DualArmRobot):
    """
    无手类型的双臂机器人基类
    
    该类不包含任何手部执行器，只包含手臂部分。
    适用于只需要手臂操作，不需要抓取功能的场景。
    """
    
    def __init__(self, env: DualArmEnv, id: int, name: str, robot_config_name: str = None) -> None:
        super().__init__(env, id, name, robot_config_name=robot_config_name)
        self.init_agent(id)
        
    def init_agent(self, id: int):
        _logger.info("OpenLoongNoHandFixBase init_agent")
        super().init_agent(id)
        print(f"[OpenLoongNoHandFixBase.init_agent] 机器人 name='{self._name}', robot_config_name={self._robot_config_name}")
        print(f"[OpenLoongNoHandFixBase.init_agent] 无手类型机器人，不初始化手部执行器")

    def set_gripper_ctrl_l(self, joystick_state) -> None:
        """
        无手类型：左手夹爪控制（空实现）
        """
        # 无手类型，不执行任何操作
        self._grasp_value_l = 0.0
        pass

    def set_gripper_ctrl_r(self, joystick_state) -> None:
        """
        无手类型：右手夹爪控制（空实现）
        """
        # 无手类型，不执行任何操作
        self._grasp_value_r = 0.0
        pass

    def set_l_hand_actuator_ctrl(self, offset_rate) -> None:
        """
        无手类型：左手执行器控制（空实现）
        """
        # 无手类型，不执行任何操作
        pass

    def set_r_hand_actuator_ctrl(self, offset_rate) -> None:
        """
        无手类型：右手执行器控制（空实现）
        """
        # 无手类型，不执行任何操作
        pass

    def update_force_feedback(self) -> None:
        """
        无手类型：力反馈更新（空实现）
        """
        # 无手类型，不执行任何操作
        if self._pico_joystick is not None:
            # 发送零力反馈
            self._pico_joystick.send_force_message(0.0, 0.0)

    def set_wheel_ctrl(self, joystick_state) -> None:
        """
        无手类型：轮子控制（空实现）
        """
        # 无手类型，不执行任何操作
        return

