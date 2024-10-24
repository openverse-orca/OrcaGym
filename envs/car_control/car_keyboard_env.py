# car_keyboard_env.py

from envs.robot_env import MujocoRobotEnv
from orca_gym.utils import rotations
from orca_gym.devices.keyboard import KeyboardInput
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
import numpy as np

ObsType = Any

# 手动定义 ButtonState 类来替代导入
class ButtonState:
    """
    定义一个用于存储键盘按键状态的类
    """
    def __init__(self):
        # 初始化所有按键状态为 False
        self.key_w = False
        self.key_a = False
        self.key_s = False
        self.key_d = False
        self.key_space = False
        self.key_up = False
        self.key_down = False

    def reset(self):
        """
        重置所有按键状态为 False
        """
        self.key_w = False
        self.key_a = False
        self.key_s = False
        self.key_d = False
        self.key_space = False
        self.key_up = False
        self.key_down = False

    def update(self, key, state: bool):
        """
        更新某个按键的状态
        :param key: 键值（如 'W'、'A'、'S' 等）
        :param state: 按键状态（True 为按下，False 为松开）
        """
        if hasattr(self, f"key_{key.lower()}"):
            setattr(self, f"key_{key.lower()}", state)

    def get_state(self):
        """
        返回当前所有按键的状态
        """
        return {
            "W": self.key_w,
            "A": self.key_a,
            "S": self.key_s,
            "D": self.key_d,
            "Space": self.key_space,
            "Up": self.key_up,
            "Down": self.key_down,
        }


class CarKeyboardEnv(MujocoRobotEnv):
    """
    通过键盘控制汽车模型
    """
    def __init__(
        self,
        frame_skip: int = 5,
        grpc_address: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps
        record_state: str = RecordState.NONE,
        record_file: Optional[str] = None,
        action_space_type: ActionSpaceType = ActionSpaceType.CONTINUOUS,  # 添加 action_space_type 参数
        action_step_count: int = 0,  # 添加 action_step_count 参数
        **kwargs,
    ):
        action_size = 2  # 这里的 action size 根据汽车控制的需求设置
        self.ctrl = np.zeros(action_size)  # 提前初始化self.ctrl
        self.n_actions = 2  # 示例值；根据你的动作空间进行调整
        self.record_state = record_state
        self.record_file = record_file
        self.record_pool = []
        self.RECORD_POOL_SIZE = 1000
        self.record_cursor = 0

        # 初始化父类环境
        super().__init__(
            frame_skip=frame_skip,
            grpc_address=grpc_address,
            agent_names=agent_names,
            time_step=time_step,
            n_actions=action_size,
            observation_space=None,
            action_space_type=action_space_type,  # 传递 action_space_type 参数
            action_step_count=action_step_count,  # 传递 action_step_count 参数
            **kwargs,
        )

        # 初始化键盘输入
        self.keyboard_controller = KeyboardInput()
        self.button_state = ButtonState()  # 使用手动定义的 ButtonState 类来记录键盘按键状态

        # 定义初始位置和其他状态信息
        self._set_init_state()


    def _set_init_state(self) -> None:
        # 初始化控制变量
        self.ctrl = np.zeros(self.n_actions)  # 确保与动作空间匹配
        self.mj_forward()

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._set_action()
        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info = {}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info

    def control_step(self, key_input):
        """
        根据键盘输入控制小车的左右轮
        """
        SPEED_FACTOR = 1.0  # 速度调节因子，可根据需要调整

        # 键盘的W和S键控制前进后退，A和D键控制转向
        forward_back = (key_input["W"] - key_input["S"]) * SPEED_FACTOR
        left_right = (key_input["A"] - key_input["D"]) * SPEED_FACTOR

        # 返回左右轮的控制力矩
        return np.array([forward_back + left_right, forward_back - left_right])

    def _capture_keyboard_ctrl(self) -> np.ndarray:
        """
        获取键盘输入并返回控制量
        """
        self.keyboard_controller.update()
        key_status = self.keyboard_controller.get_state()

        # 获取键盘输入的控制量
        ctrl = self.control_step(key_status)
        return ctrl

    def _set_action(self) -> None:
        """
        根据键盘输入更新小车的控制量
        """
        # 获取控制量并应用
        ctrl = self._capture_keyboard_ctrl()
        self.ctrl = ctrl

    def _get_obs(self) -> dict:
        # 这里根据你的汽车模型获取观察数据
        obs = np.concatenate([self.ctrl]).copy()
        result = {
            "observation": obs,
            "achieved_goal": np.array([0, 0]),
            "desired_goal": np.array([0, 0]),
        }
        return result

    def reset_model(self):
        self._set_init_state()

    def _reset_sim(self) -> bool:
        self._set_init_state()
        return True

    def _sample_goal(self):
        return np.zeros((self.model.nq,))  # 例如，返回一个全零的目标
