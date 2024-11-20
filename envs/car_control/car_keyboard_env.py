# car_keyboard_env.py

from orca_gym.utils import rotations
from orca_gym.environment import OrcaGymRemoteEnv
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


class CarKeyboardEnv(OrcaGymRemoteEnv):
    """
    通过键盘控制汽车模型
    """
    def __init__(
        self,
        frame_skip: int = 5,
        orcagym_addr: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps
        **kwargs,
    ):
        action_size = 2  # 这里的 action size 根据汽车控制的需求设置
        self.ctrl = np.zeros(action_size)  # 提前初始化self.ctrl
        self.n_actions = 2  # 示例值；根据你的动作空间进行调整

        # 初始化父类环境
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        # 初始化键盘输入
        self.keyboard_controller = KeyboardInput()
        self.button_state = ButtonState()  # 使用手动定义的 ButtonState 类来记录键盘按键状态

        # 定义初始位置和其他状态信息
        self._set_init_state()

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        self.action_space = self.generate_action_space(self.model.get_actuator_ctrlrange())

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
        }
        return result

    def reset_model(self):
        self._set_init_state()
        obs = self._get_obs().copy()
        return obs

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()