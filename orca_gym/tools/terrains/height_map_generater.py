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
from typing import Optional, Any, SupportsFloat
import numpy as np
import gymnasium as gym
import time
import mujoco

ObsType = Any



class HeightMapGenerater(OrcaGymLocalEnv):
    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 30}
    
    """
    通过检测物理碰撞，生成当前环境的高程图。分辨率为 x,y,z 方向 0.1m 
    使用方法：
    1. 首先将关卡中的agents移除，避免高程图包含了 agents 的信息
    3. 运行此脚本，生成高程图文件
    4. 在legged_gym等任务中，使用高程图文件
    """
    def __init__(
        self,
        frame_skip: int = 1,
        orcagym_addr: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps
        render_mode: str = 'human',
        height_map_border: tuple[int, int] = (100, 100), # in meters
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

        
        height_map_border = np.array(height_map_border, dtype=float)
        height_range = np.array(height_range, dtype=float)
        self._render_mode = render_mode
        
        print("Height map border: ", height_map_border)
        print("Height range: ", height_range)
        print("Render mode: ", self.render_mode)
        
        x_range = height_map_border[2] - height_map_border[0]
        y_range = height_map_border[3] - height_map_border[1]
        z_range = height_range[1] - height_range[0]
        
        resolution = 0.1
        ruler_length = resolution * 10
        
        self._height_map = {
            "left_up_corner": np.array([height_map_border[0], height_map_border[1]]).flatten(),
            "right_down_corner": np.array([height_map_border[2], height_map_border[3]]).flatten(),
            "height_range": height_range,
            "map": {
                "width": int(x_range / resolution),
                "height": int(y_range / resolution),
                "data": np.zeros([int(x_range / resolution), int(y_range / resolution)], dtype=float),
                "resolution": resolution,
            },
            "mini_map": {
                "width": int(x_range / resolution / 10),
                "height": int(y_range / resolution / 10),
                "data": np.zeros([int(x_range / resolution / 10), int(y_range / resolution / 10)], dtype=float),
                "resolution": resolution * 10,
            },
        }
        # 先设置为最低高度
        self._height_map["map"]["data"][:, :] = height_range[0]
        self._height_map["mini_map"]["data"][:, :] = height_range[0] - 1.0
        
        print("Height map size: ", self._height_map["map"]["width"], self._height_map["map"]["height"])
        print("Height range: ", self._height_map["height_range"])        
        # print("Mini map data: ", self._height_map["mini_map"]["data"])
        
        # 默认qpos高度为最高高度
        self._ruler = {
            "x_range": resolution,
            "y_range": resolution,
            "z_range": ruler_length,
            "joint_name": "height_map_helper_usda_ruler",
            "init_qpos": np.array([self._height_map["left_up_corner"][0] + resolution / 2, 
                                    self._height_map["left_up_corner"][1] + resolution / 2,
                                    height_range[1], 1, 0, 0, 0], dtype=float).flatten(),
            "idel_qpos": np.array([0, 0, 1000, 1, 0, 0, 0]).flatten(),  # 用于将ruler移出碰撞
        }
        
        self._big_box = {
            "x_range": 1.0,
            "y_range": 1.0,
            "z_range": 1.0,
            "joint_name": "height_map_helper_usda_big_box",
            "init_qpos": np.array([self._height_map["left_up_corner"][0] + 0.5, 
                                    self._height_map["left_up_corner"][1] + 0.5, 
                                    height_range[1], 1, 0, 0, 0], dtype=float).flatten(),
            "idel_qpos": np.array([0, 0, 2000, 1, 0, 0, 0]).flatten(),  # 用于将big_box移出碰撞
        }
            
        # self._build_ruler_geom_offsets()
        # self._query_helper_qpos_offset()

        # # raise NotImplementedError("Please implement the rest of the class")

        # # 定义初始位置和其他状态信息
        # self._set_init_state()

        # Run generate_observation_space after initialization to ensure that the observation object's name is defined.
        self._set_obs_space()
        self._set_action_space()

    def _build_ruler_geom_offsets(self):
        name_2_offset = [
            "height_map_helper_usda_geom_0", "height_map_helper_usda_geom_1", "height_map_helper_usda_geom_2", "height_map_helper_usda_geom_3", "height_map_helper_usda_geom_4", 
            "height_map_helper_usda_geom_5", "height_map_helper_usda_geom_6", "height_map_helper_usda_geom_7", "height_map_helper_usda_geom_8", "height_map_helper_usda_geom_9",
        ]
        self._ruler["geom_offsets"] = {}
        
        geom_dict = self.model.get_geom_dict()
        # print("Geom dict: ", geom_dict)
        for name in geom_dict:
            for i in range(len(name_2_offset)):
                if name_2_offset[i] in name:
                    self._ruler["geom_offsets"][self.model.geom_name2id(name)] = (i + 1) * 0.1
                    print("Geom : ", name, " offset: ", self._ruler["geom_offsets"][self.model.geom_name2id(name)])
                    break
                
        print("Helper geom offsets: ", self._ruler["geom_offsets"])
        
    def _query_helper_qpos_offset(self):
        qpos_offset, _, _ = self.query_joint_offsets([self._ruler["joint_name"], self._big_box["joint_name"]])
        self._ruler["joint_offset"] = qpos_offset[0]
        self._big_box["joint_offset"] = qpos_offset[1]
        print("Helper qpos offset: ", self._ruler["joint_offset"], self._big_box["joint_offset"])

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        self.action_space = self.generate_action_space(np.array([[-1,1], [-1,1]]))

    def _write_qpos_buffer(self, qpos_offset : int, qpos : np.ndarray):
        self.gym._mjData.qpos[qpos_offset:qpos_offset + len(qpos)] = qpos.copy()

    def _set_init_state(self) -> None:
        self.gym.opt.gravity = np.array([0, 0, 0])
        self.gym.opt.iterations = 10
        self.gym.opt.noslip_iterations = 0
        self.gym.opt.ccd_iterations = 0
        self.gym.opt.sdf_iterations = 0
        self.gym.opt.timestep = 0.001
        self.gym.set_opt_config()
        
        # 初始化控制变量
        self.ctrl = np.zeros(self.n_actions)  # 确保与动作空间匹配
        self._write_qpos_buffer(self._ruler["joint_offset"], self._ruler["idel_qpos"])
        self._write_qpos_buffer(self._big_box["joint_offset"], self._big_box["idel_qpos"])
        self.mj_forward()

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # self._generate_mini_map(action)
        # self._generate_height_map(action)
        self._generate_height_map_ray_casting(action)
        obs = self._get_obs().copy()
        info = {"height_map": self._height_map["map"]["data"]}
        terminated = False
        truncated = False
        reward = 0

        return obs, reward, terminated, truncated, info
        
    def _generate_height_map_ray_casting(self, action):
        self.mj_forward()
        # 通过射线投射的方式生成高程图
        for x in range(self._height_map["map"]["width"]):
            print("\rGenerate Height map: {}/{}".format(x, self._height_map["map"]["width"]), end='', flush=True)
            for y in range(self._height_map["map"]["height"]):
                if self._height_map["map"]["data"][x, y] < self._height_map["height_range"][0]:
                    # print("Height map ", x, y, " is already the lowest height")
                    continue
                point = np.array([self._height_map["left_up_corner"][0] + x * self._height_map["map"]["resolution"],
                                 self._height_map["left_up_corner"][1] + y * self._height_map["map"]["resolution"],
                                 10.0], dtype=float)
                direction = np.array([0, 0, -1], dtype=float)
                geomgroup = np.ones(6, dtype=np.intc)
                flg_static = 1
                bodyexclude = -1
                geomid = np.zeros(1, dtype=np.intc)
                distance = mujoco.mj_ray(self.gym._mjModel, self.gym._mjData, point, direction, geomgroup, flg_static, bodyexclude, geomid)
                height = 10.0 - distance
                if distance > 0.0 and height > 0.001:
                    # print("Height map: ", x, y, height, "Geom ID: ", geomid[0])
                    self._height_map["map"]["data"][x, y] = height
        
        print("\nDone!")
        
    def _generate_height_map(self, action):
        for x in range(self._height_map["mini_map"]["width"]):
            print("Generate Height map: ", x, "/", self._height_map["mini_map"]["width"])
            for y in range(self._height_map["mini_map"]["height"]):
                if self._height_map["mini_map"]["data"][x, y] < self._height_map["height_range"][0]:
                    # print("Height map ", x, y, " is already the lowest height")
                    continue
                self._generate_unit_height_map(action, x * 10, y * 10, self._height_map["mini_map"]["data"][x, y])
                
    def _generate_unit_height_map(self, action, mini_map_x, mini_map_y, z):
        qpos = self._ruler["init_qpos"]
        for x in range(mini_map_x, mini_map_x + 10):
            for y in range(mini_map_y, mini_map_y + 10):
                while z >= self._height_map["height_range"][0]:
                    self._render(action)
                                        
                    qpos[0] = self._height_map["left_up_corner"][0] + x * self._height_map["map"]["resolution"]
                    qpos[1] = self._height_map["left_up_corner"][1] + y * self._height_map["map"]["resolution"]
                    qpos[2] = z
                    self._write_qpos_buffer(self._ruler["joint_offset"], qpos)
                    self.mj_forward()
                    contact_dict = self.query_contact_simple()
                    
                    if len(contact_dict) > 0:
                        self._update_height_map(contact_dict, x, y, z)
                        break
                    
                    z -= self._ruler["z_range"]
                    
    def _update_height_map(self, contact_dict, x, y, z):        
        # print("Contact dict len: ", len(contact_dict))
        # print("Pos: ", self._helper_qpos[0], self._helper_qpos[1], self._helper_qpos[2])
        offset = 0
        for contact in contact_dict:
            if contact["Geom1"] in self._ruler["geom_offsets"]:
                offset = max(self._ruler["geom_offsets"][contact["Geom1"]], offset)
            elif contact["Geom2"] in self._ruler["geom_offsets"]:
                offset = max(self._ruler["geom_offsets"][contact["Geom2"]], offset)
            
        self._height_map["map"]["data"][x, y] = z + offset
        
        # print("Update height map: ", x, y, self._height_map["map"]["data"][x, y])

    
    def _generate_mini_map(self, action):
        qpos = self._big_box["init_qpos"]
        for x in range(self._height_map["mini_map"]["width"]):
            print("Building mini map: ", x, "/", self._height_map["mini_map"]["width"])
            for y in range(self._height_map["mini_map"]["height"]):
                z = self._height_map["height_range"][1]
                while z >= self._height_map["height_range"][0]:
                    self._render(action)
                        
                    qpos[0] = self._height_map["left_up_corner"][0] + x * self._height_map["mini_map"]["resolution"]
                    qpos[1] = self._height_map["left_up_corner"][1] + y * self._height_map["mini_map"]["resolution"]
                    qpos[2] = z
                    self._write_qpos_buffer(self._big_box["joint_offset"], qpos)
                    self.mj_forward()
                    contact_dict = self.query_contact_simple()
                    
                    if len(contact_dict) > 0:
                        self._height_map["mini_map"]["data"][x, y] = z
                        print("Mini map: ", x, y, z)
                        break
                    
                    z -= self._height_map["mini_map"]["resolution"]
                            
        # 将big_box移出碰撞
        self._write_qpos_buffer(self._big_box["joint_offset"], self._big_box["idel_qpos"])
        self.mj_forward()
                
                
    def _render(self, action=None):
        if self.render_mode == "human":
            self.do_simulation(action, self.frame_skip)
            self.render()
            time.sleep(self.dt)
    


    def _get_obs(self) -> dict:
        obs = np.concatenate([self.ctrl], dtype=np.float32).copy()
        result = {
            "observation": obs,
        }
        return result

    def reset_model(self) -> tuple[dict, dict]:
        # self._set_init_state()
        obs = self._get_obs().copy()
        return obs, {}

    def get_observation(self, obs=None):
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()
        
TIME_STEP = 0.005

def statiscs_height_map(flat_height_map):
    """
    统计高程图的高度分布
    """
    flat_height_map = flat_height_map.flatten()
    
    print("Height map size: ", len(flat_height_map))
    print("min: ", np.min(flat_height_map))
    print("max: ", np.max(flat_height_map))
    print("mean: ", np.mean(flat_height_map))
    print("std: ", np.std(flat_height_map))
    
    height_map_positive = flat_height_map[flat_height_map > 0]    
    if len(height_map_positive) == 0:
        print("No positive height map")
    else:
        print("Height map positive size: ", len(height_map_positive))
        area_coverage = len(height_map_positive) / len(flat_height_map)
        print(f"Area coverage: {area_coverage:.2%}")
        print("min: ", np.min(height_map_positive))
        print("max: ", np.max(height_map_positive))
        print("mean: ", np.mean(height_map_positive))
        print("std: ", np.std(height_map_positive))

        
def register_env(orcagym_addr, env_name, env_index, height_map_border, height_range, render_mode):
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    gym.register(
        id=env_id,
        entry_point="orca_gym.tools.terrains.height_map_generater:HeightMapGenerater",
        kwargs={
            'frame_skip': 1,   # 1 action per frame
            'orcagym_addr': orcagym_addr,
            'agent_names': ['height_map_helper_usda'],
            'time_step': TIME_STEP,
            'render_mode': render_mode,
            'height_map_border': height_map_border,
            'height_range': height_range,
        },
        max_episode_steps=sys.maxsize,
        reward_threshold=0.0,
    )
    return env_id        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--orcagym_addresses', type=str, nargs='+', default=['localhost:50051'], help='The gRPC addresses to connect to')
    parser.add_argument('--height_map_border', type=int, nargs=4, default=[-100, -100, 100, 100], help='The left-up and right-down corner coordinates of the height map in meters')
    parser.add_argument('--height_range', type=int, nargs=2, default=[0, 100], help='The height range of the height map in meters')
    parser.add_argument('--render_mode', type=str, default='none', help='The render mode (human or none). Set to none for faster processing')
    parser.add_argument('--output_file', type=str, default='height_map.npy', help='The output file to save the height map')
    args = parser.parse_args()
    
    orcagym_addr = args.orcagym_addresses[0]
    height_map_border = tuple(args.height_map_border)
    height_range = tuple(args.height_range)
    render_mode = args.render_mode
    output_file = args.output_file
    
    assert height_map_border[0] < height_map_border[2] and height_map_border[1] < height_map_border[3], "Height map border is invalid"
    assert height_range[0] % 10 == 0 and height_range[1] % 10 == 0, "Height range should be multiple of 10"
    
    try:
        print("simulation running... , orcagym_addr: ", orcagym_addr)

        env_name = "HeightMapHelper-v0"
        env_index = 0
        env_id = register_env(orcagym_addr, env_name, env_index, height_map_border, height_range, render_mode)
        print("Registering environment with id: ", env_id)

        env = gym.make(env_id)
        print("Starting simulation...")
        env.reset()

        action = np.zeros(0)
        obs, reward, terminated, truncated, info = env.step(action)
            
        print("Height map generation completed!")
        height_map = info["height_map"]
        
        statiscs_height_map(height_map)
        # output to the npy file
        np.save(output_file, height_map)
            
        env.close()
    except KeyboardInterrupt:
        print("Simulation stopped")
        env.close()
            
