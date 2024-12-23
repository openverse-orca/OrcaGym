import sys
import asyncio
import argparse
import os

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))
# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)

from orca_gym.utils import rotations
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.devices.keyboard import KeyboardInput
from typing import Optional, Any, SupportsFloat
from gymnasium import spaces
import numpy as np
import gymnasium as gym
import time


ObsType = Any



class HeightMapGenerater(OrcaGymLocalEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}
    
    """
    通过检测物理碰撞，生成当前环境的高程图。分辨率为 x,y,z 方向 0.1m 
    使用方法：
    1. 首先将关卡中的agents移除，避免高程图包含了 agents 的信息
    2. 导入envs/assets/terrains/height_map_helper.xml模型，放置于关卡中，运行关卡
    3. 运行此脚本，生成高程图文件
    4. 在legged_gym等任务中，使用高程图文件
    """
    def __init__(
        self,
        frame_skip: int = 1,
        orcagym_addr: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps
        height_map_size: tuple[int, int] = (100, 100), # in meters
        height_range: tuple[int, int] = (0, 10), # in meters
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

        
        self._height_map_size = np.array(height_map_size, dtype=float)
        self._height_range = np.array(height_range, dtype=float)
        
        print("Height map size: ", self._height_map_size)
        print("Height range: ", self._height_range)
        
        self._height_map = np.zeros([height_map_size[0] * 10, height_map_size[1] * 10], dtype=float)  # 0.1m resolution
        self._unit_height_map = np.zeros([10, 10], dtype=float) # 0.1m resolution, 1m x 1m
        self._height_map[:, :] = height_range[0]    # 初始化高程图到最低高程
        self._unit_height_map[:, :] = height_range[0]
        self._height_map_border = np.array([[-self._height_map_size[0] / 2, self._height_map_size[0] / 2], [-self._height_map_size[1] / 2, self._height_map_size[1] / 2]])
        print("Height border: ", self._height_map_border)
        # print("Height map: ", self._height_map)
        
        self._helper_qpos = np.array([self._height_map_border[0][0], self._height_map_border[1][0], height_range[1], 1, 0, 0, 0], dtype=float)
        self._helper_joint_name = "height_map_helper_root"
        self._build_helper_geom_offsets()

        # raise NotImplementedError("Please implement the rest of the class")

        # 定义初始位置和其他状态信息
        self._set_init_state()

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

    def _build_helper_geom_offsets(self):
        name_2_offset = [
            "height_map_helper_geom_00", "height_map_helper_geom_01", "height_map_helper_geom_02", "height_map_helper_geom_03", "height_map_helper_geom_04", "height_map_helper_geom_05", "height_map_helper_geom_06", "height_map_helper_geom_07", "height_map_helper_geom_08", "height_map_helper_geom_09",
            "height_map_helper_geom_10", "height_map_helper_geom_11", "height_map_helper_geom_12", "height_map_helper_geom_13", "height_map_helper_geom_14", "height_map_helper_geom_15", "height_map_helper_geom_16", "height_map_helper_geom_17", "height_map_helper_geom_18", "height_map_helper_geom_19",
            "height_map_helper_geom_20", "height_map_helper_geom_21", "height_map_helper_geom_22", "height_map_helper_geom_23", "height_map_helper_geom_24", "height_map_helper_geom_25", "height_map_helper_geom_26", "height_map_helper_geom_27", "height_map_helper_geom_28", "height_map_helper_geom_29",
            "height_map_helper_geom_30", "height_map_helper_geom_31", "height_map_helper_geom_32", "height_map_helper_geom_33", "height_map_helper_geom_34", "height_map_helper_geom_35", "height_map_helper_geom_36", "height_map_helper_geom_37", "height_map_helper_geom_38", "height_map_helper_geom_39",
            "height_map_helper_geom_40", "height_map_helper_geom_41", "height_map_helper_geom_42", "height_map_helper_geom_43", "height_map_helper_geom_44", "height_map_helper_geom_45", "height_map_helper_geom_46", "height_map_helper_geom_47", "height_map_helper_geom_48", "height_map_helper_geom_49",
            "height_map_helper_geom_50", "height_map_helper_geom_51", "height_map_helper_geom_52", "height_map_helper_geom_53", "height_map_helper_geom_54", "height_map_helper_geom_55", "height_map_helper_geom_56", "height_map_helper_geom_57", "height_map_helper_geom_58", "height_map_helper_geom_59",
            "height_map_helper_geom_60", "height_map_helper_geom_61", "height_map_helper_geom_62", "height_map_helper_geom_63", "height_map_helper_geom_64", "height_map_helper_geom_65", "height_map_helper_geom_66", "height_map_helper_geom_67", "height_map_helper_geom_68", "height_map_helper_geom_69",
            "height_map_helper_geom_70", "height_map_helper_geom_71", "height_map_helper_geom_72", "height_map_helper_geom_73", "height_map_helper_geom_74", "height_map_helper_geom_75", "height_map_helper_geom_76", "height_map_helper_geom_77", "height_map_helper_geom_78", "height_map_helper_geom_79",
            "height_map_helper_geom_80", "height_map_helper_geom_81", "height_map_helper_geom_82", "height_map_helper_geom_83", "height_map_helper_geom_84", "height_map_helper_geom_85", "height_map_helper_geom_86", "height_map_helper_geom_87", "height_map_helper_geom_88", "height_map_helper_geom_89",
            "height_map_helper_geom_90", "height_map_helper_geom_91", "height_map_helper_geom_92", "height_map_helper_geom_93", "height_map_helper_geom_94", "height_map_helper_geom_95", "height_map_helper_geom_96", "height_map_helper_geom_97", "height_map_helper_geom_98", "height_map_helper_geom_99",
        ]
        self._helper_geom_offsets = {}
        
        geom_dict = self.model.get_geom_dict()
        for name in geom_dict:
            for i in range(len(name_2_offset)):
                if name_2_offset[i] in name:
                    self._helper_geom_offsets[self.model.geom_name2id(name)] = i * 0.1
                    print("Geom : ", name, " offset: ", i * 0.1)
                    break
                
        print("Helper geom offsets: ", self._helper_geom_offsets)

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        self.action_space = self.generate_action_space(np.array([[-1,1], [-1,1]]))

    def _set_init_state(self) -> None:
        self.gym.opt.gravity = np.array([0, 0, 0])
        self.gym.opt.iterations = 10
        self.gym.opt.noslip_iterations = 0
        self.gym.opt.mpr_iterations = 0
        self.gym.opt.sdf_iterations = 0
        self.gym.opt.timestep = 0.001
        self.gym.set_opt_config()
        
        
        # 初始化控制变量
        self.ctrl = np.zeros(self.n_actions)  # 确保与动作空间匹配
        joint_qpos = {self._helper_joint_name: self._helper_qpos}
        self.set_joint_qpos(joint_qpos)
        self.mj_forward()

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        collision = self._process_detection()
        
        if self.render_mode == "human":
            self.do_simulation(action, self.frame_skip)
        
        obs = self._get_obs().copy()
        info = {"height_map": self._height_map, "collision": collision}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info

    def _process_detection(self): 
        self._helper_qpos[2] = self._height_range[1]
        while self._helper_qpos[2] >= self._height_range[0]:
            # print("Helper qpos: ", self._helper_qpos)
            joint_qpos = {self._helper_joint_name: self._helper_qpos}
            self.set_joint_qpos(joint_qpos)
            self.mj_forward()
            contact_dict = self.query_contact_simple()            
            collision = self._update_height_map(contact_dict)
            
            if collision:
                break
            
            self._helper_qpos[2] -= 10

        self._helper_qpos[0] += 0.1
        if self._helper_qpos[0] >= self._height_map_border[0][1]:
            self._helper_qpos[0] = self._height_map_border[0][0]
            self._helper_qpos[1] += 0.1
            if self._helper_qpos[1] >= self._height_map_border[1][1]:
                print("Height map generation completed! helper qpos: ", self._helper_qpos)
        
        return collision
    
    def _update_height_map(self, contact_dict):        
        if len(contact_dict) > 0:
            # print("Contact dict len: ", len(contact_dict))
            # print("Pos: ", self._helper_qpos[0], self._helper_qpos[1], self._helper_qpos[2])
            z = 0
            for contact in contact_dict:
                if contact["Geom1"] in self._helper_geom_offsets:
                    offset = self._helper_geom_offsets[contact["Geom1"]]
                elif contact["Geom2"] in self._helper_geom_offsets:
                    offset = self._helper_geom_offsets[contact["Geom2"]]
                else:
                    continue
                
                z = max(z, offset)
                
            x = int(self._helper_qpos[0] * 10)
            y = int(self._helper_qpos[1] * 10)
            self._height_map[x, y] = self._helper_qpos[2] + z
            # print("Update height map: ", x, y, self._height_map[x, y])
            
            return True
        
        return False

    def _get_obs(self) -> dict:
        obs = np.concatenate([self.ctrl]).copy()
        result = {
            "observation": obs,
        }
        return result

    def reset_model(self):
        self._set_init_state()
        obs = self._get_obs().copy()
        return obs, {}

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()
        
TIME_STEP = 0.005
        
def register_env(orcagym_addr, env_name, env_index, height_map_size, height_range, render_mode):
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point="orca_gym.tools.height_map_generater:HeightMapGenerater",
        kwargs={
            'frame_skip': 1,   # 1 action per frame
            'orcagym_addr': orcagym_addr,
            'agent_names': ['height_map_helper'],
            'time_step': TIME_STEP,
            'render_mode': render_mode,
            'height_map_size': height_map_size,
            'height_range': height_range,
        },
        max_episode_steps=sys.maxsize,
        reward_threshold=0.0,
    )
    return env_id        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_addresses', type=str, nargs='+', default=['localhost:50051'], help='The gRPC addresses to connect to')
    parser.add_argument('--height_map_size', type=int, nargs=2, default=[160, 160], help='The size of the height map in meters')
    parser.add_argument('--height_range', type=int, nargs=2, default=[0, 20], help='The height range of the height map in meters')
    parser.add_argument('--render_mode', type=str, default='human', help='The render mode (human or none). Set to none for faster processing')
    parser.add_argument('--output_file', type=str, default='height_map.npy', help='The output file to save the height map')
    args = parser.parse_args()
    
    orcagym_addr = args.orcagym_addresses[0]
    height_map_size = tuple(args.height_map_size)
    height_range = tuple(args.height_range)
    render_mode = args.render_mode
    output_file = args.output_file
    
    assert height_map_size[0] > 0 and height_map_size[1] > 0 and height_range[1] > height_range[0], "Invalid height map size or height range"
    assert height_range[0] % 10 == 0 and height_range[1] % 10 == 0, "Height range should be multiple of 10"
    
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_name = "HeightMapHelper-v0"
        env_index = 0
        env_id = register_env(orcagym_addr, env_name, env_index, height_map_size, height_range, render_mode)
        print("Registering environment with id: ", env_id)

        env = gym.make(env_id)
        print("Starting simulation...")
        env.reset()
        
        # raise NotImplementedError("Please implement the rest of the script")
        
        info = {}
        action = np.zeros(0)
        for y in range(height_map_size[1] * 10):
            for x in range(height_map_size[0] * 10):
                obs, reward, terminated, truncated, info = env.step(action)
                # print("decetion: ", x, y)
                if render_mode == "human":
                    env.render()
                # time.sleep(0.001)
                                    
            print("Height map generation: ", y, " / ", height_map_size[1] * 10)
            
        print("Height map generation completed!")
        height_map = info["height_map"]
        # output to the npy file
        np.save(output_file, height_map)
            
        env.close()
    except KeyboardInterrupt:
        print("Simulation stopped")
        env.close()
            
