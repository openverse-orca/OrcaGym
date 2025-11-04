import gymnasium as gym
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import cv2
from orca_gym.sensor.rgbd_camera import CameraWrapper
from orca_gym.scripts.dual_arm_manipulation import RGB_SIZE, CAMERA_CONFIG
import time
import os

class DexforceW1OpenpiEnv(_environment.Environment):
    """An environment for a Dexforce W1 robot in simulation."""

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
        # from orca_gym.sensor.fake_rgbd_camera import FakeCameraWrapper
        # self._camera = FakeCameraWrapper(name="camera_head", video_path="/media/user/A7EC-9D11/shopscene_1f15/Shop-79p12GB_4294counts_88p59h/Shelf_Operation-79p12GB_4294counts_88p59h/pick_and_place-79p12GB_4294counts_88p59h/5a5c8831_a286d304/camera/video/camera_head_color.mp4", loop_video=True, fps=30)
        # self._camera.start()

        
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
        # print(f"[Debug] joint_pos: {joint_pos}")
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
        # self._last_obs = self.fake_convert_observation(gym_obs)  # type: ignore  # TODO:DELETE
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)
        
        # 记录动作
        # self._log_action_csv(joint_pos)
        # self._log_state_csv(self._last_obs["state"])
        
    def _log_action_csv(self, action: np.ndarray) -> None:
        # np.zeros(6),                # left hand ee pos and angle euler
        # joint_pos[:7],              # left arm joint pos
        # joint_pos[7:8],             # left hand grasp value
        # np.zeros(6),                # right hand ee pos and angle euler
        # joint_pos[8:15],             # right arm joint pos
        # joint_pos[15:16],           # right hand grasp value
        # draw header if file is empty
        if not os.path.exists("action.csv"):
            with open("action.csv", "w") as f:
                f.write("left_arm_joint_pos_1,left_arm_joint_pos_2,left_arm_joint_pos_3,left_arm_joint_pos_4,left_arm_joint_pos_5,left_arm_joint_pos_6,left_arm_joint_pos_7,left_hand_grasp_value,right_arm_joint_pos_1,right_arm_joint_pos_2,right_arm_joint_pos_3,right_arm_joint_pos_4,right_arm_joint_pos_5,right_arm_joint_pos_6,right_arm_joint_pos_7,right_hand_grasp_value\n")
        with open("action.csv", "a") as f:
            f.write(",".join(action.astype(str)) + "\n")
    
    def _log_state_csv(self, state: np.ndarray) -> None:
        # gym_obs["ee_pos_l"],
        # gym_obs["ee_pos_r"],
        # gym_obs["arm_joint_qpos_l"], 
        # gym_obs["arm_joint_qpos_r"],
        # gym_obs["grasp_value_l"],
        # gym_obs["grasp_value_r"],
        # np.zeros(10),
        if not os.path.exists("state.csv"):
            with open("state.csv", "w") as f:
                f.write("ee_pos_l_x,ee_pos_l_y,ee_pos_l_z,ee_pos_r_x,ee_pos_r_y,ee_pos_r_z,arm_joint_qpos_l_1,arm_joint_qpos_l_2,arm_joint_qpos_l_3,arm_joint_qpos_l_4,arm_joint_qpos_l_5,arm_joint_qpos_l_6,arm_joint_qpos_l_7,arm_joint_qpos_r_1,arm_joint_qpos_r_2,arm_joint_qpos_r_3,arm_joint_qpos_r_4,arm_joint_qpos_r_5,arm_joint_qpos_r_6,arm_joint_qpos_r_7,grasp_value_l,grasp_value_r\n")
        with open("state.csv", "a") as f:
            f.write(",".join(state[0:22].astype(str)) + "\n")
            
        

    def _convert_observation(self, gym_obs: dict) -> dict:
        img, _ = self._camera.get_frame(format="rgb24", size=RGB_SIZE)
        # save img to file
        # cv2.imwrite("img.png", img)
        
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
            gym_obs["arm_joint_qpos_l"], 
            gym_obs["grasp_value_l"],
            gym_obs["arm_joint_qpos_r"],
            gym_obs["grasp_value_r"],
            gym_obs["ee_pos_l"],
            gym_obs["ee_pos_r"],
            np.zeros(10),
        ]).flatten()
        
        
        
        return {
            "state": joint_qpos,
            "images": {"cam_high": img},
            "prompt": self._prompt,
        }

