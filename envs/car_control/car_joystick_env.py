from orca_gym.utils import rotations
from typing import Optional, Any, SupportsFloat
from orca_gym.environment import OrcaGymRemoteEnv
from gymnasium import spaces
from orca_gym.devices.xbox_joystick import XboxJoystick, XboxJoystickManager  # 引入 XboxJoystick 和 XboxJoystickManager
import numpy as np
ObsType = Any


class CarEnv(OrcaGymRemoteEnv):
    """
    通过xbox手柄控制汽车模型
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

        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        # 初始化手柄管理器
        self.joystick_manager = XboxJoystickManager()
        # 从手柄管理器中获取第一个手柄（假设只使用一个手柄）
        self.joystick = self.joystick_manager.get_joystick(self.joystick_manager.get_joystick_names()[0])

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

    def control_step(self, joystick_input):
        """
        根据手柄输入控制小车的左右轮
        """
        SPEED_FACTOR = 1.0  # 速度调节因子，可根据需要调整
        
        # 左摇杆控制左轮，右摇杆控制右轮
        left_wheel_force = joystick_input["axes"]["LeftStickY"] * SPEED_FACTOR
        right_wheel_force = joystick_input["axes"]["RightStickY"] * SPEED_FACTOR
        
        # 返回左轮和右轮的控制力矩
        return np.array([left_wheel_force, right_wheel_force])

    def _capture_joystick_ctrl(self, joystick_state) -> np.ndarray:
        """
        获取手柄输入并返回控制量
        """
        ctrl = self.control_step(joystick_state)
        return ctrl

    def _set_action(self) -> None:
        """
        根据手柄输入更新小车的控制量
        """

        # 调用 Joystick Manager 更新手柄状态
        self.joystick_manager.update()  # 由 Joystick Manager 更新所有手柄状态

        # 获取 joystick 的状态
        joystick_state = self.joystick.get_state()  # 获取 XboxJoystick 实例的状态

        # 获取控制量并应用
        ctrl = self._capture_joystick_ctrl(joystick_state)  # 将状态传递给控制捕获函数
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
