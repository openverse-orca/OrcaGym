import gymnasium as gym
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import cv2
from orca_gym.sensor.rgbd_camera import CameraWrapper
from orca_gym.scripts.dual_arm_manipulation import RGB_SIZE, CAMERA_CONFIG
import time

class OpenLoongOpenpiEnv(_environment.Environment):
    """An environment for an OpenLoong robot in simulation."""

    def __init__(self, 
                 env_id: str, 
                 obs_type: str, 
                 prompt: str,
                 seed: int) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self._gym = gym.make(env_id, obs_type=obs_type)

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0
        self._prompt = prompt
        self._camera = CameraWrapper(name="camera_head", port=CAMERA_CONFIG["camera_head"])
        self._camera.start()

    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        joint_pos = action["actions"]
        agetn_action = np.concatenate([
            np.zeros(6),                # left hand ee pos and angle euler
            joint_pos[:7],              # left arm joint pos
            joint_pos[7:8],             # left hand grasp value
            np.zeros(6),                # right hand ee pos and angle euler
            joint_pos[8:15],             # right arm joint pos
            joint_pos[15:16],           # right hand grasp value
        ]).flatten()
        gym_obs, reward, terminated, truncated, info = self._gym.step(agetn_action)
        
        self._gym.render()
        
        # 等待渲染结果串流到orcagym的客户端，最长等待时间不超过最大帧率
        time.sleep(0.02) # max_hz=50
        
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)

    def _convert_observation(self, gym_obs: dict) -> dict:
        img, _ = self._camera.get_frame(format="rgb24", size=RGB_SIZE)
        
        # img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        img = cv2.resize(img, (224, 224))
        
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(img, (2, 0, 1))
        
        # YAO: TODO
        # joint_qpos = np.concatenate([
        #     gym_obs["arm_joint_qpos_l"], 
        #     gym_obs["grasp_value_l"],
        #     gym_obs["arm_joint_qpos_r"],
        #     gym_obs["grasp_value_r"],
        # ]).flatten()   
        joint_qpos = np.concatenate([
            gym_obs["ee_pos_l"],
            gym_obs["ee_pos_r"],
            gym_obs["arm_joint_qpos_l"], 
            gym_obs["arm_joint_qpos_r"],
            gym_obs["grasp_value_l"],
            gym_obs["grasp_value_r"],
            np.zeros(10),
        ]).flatten()         
        
        
        return {
            "state": joint_qpos,
            "images": {"cam_high": img},
            "prompt": self._prompt,
        }

