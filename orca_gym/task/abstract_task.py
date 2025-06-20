from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.utils import rotations

import orca_gym.scene.orca_gym_scene as orca_gym_scene
import numpy as np
import random
import warnings, time

DEFAULT_CONFIG = {
    "random_object": True, # 是否随机Object的位置，object应该在Agent下面
    "object_bodys": [],
    "object_sites": [],
    "object_joints": [],

    "random_goal": False, # 是否随机Goal的位置，goal应该在Agent下面
    "goal_bodys": [],
    "goal_sites": [],
    "goal_joints": [],

    "level_name": None,
    "random_actor": False, # 是否随机添加Actor，actor与Agent同级
    "random_actor_position": False, # 是否随机Actor的位置
    "random_actor_rotation": False, # 是否随机Actor的旋转
    "actors": [], # actor的命名
    "actors_spawnable": [], # actor的spawnable name, spawnable在Asset/Prefabs下面
    "center": [0, 0, 0], # actor的中心位置
    "bound": [[-1, 1], [-1, 1], [0, 2]], # 以center点为中心，距离中心的边界位置
    "description": [],

    "random_light": False, # 是否随机添加光源
    "random_light_position": False, # 是否随机光源的位置
    "random_light_rotation": False, # 是否随机光源的旋转
    "lights": [], # 光源的命名
    "lights_spawnable": [], # 光源的spawnable name, spawnable在Asset/Prefabs下面
    "light_center": [0, 0, 0], # 光源的中心位置
    "light_bound": [[-1, 1], [-1, 1], [0, 2]], # 以center点为中心，距离中心的边界位置
    "light_description": [],
    "num_lights": 1,
    "task_element": [], # 任务元素，包含object, actor

    "grpc_addr": "localhost:50051"
}

class AbstractTask:
    """
    :param prompt: The prompt for the task.
    :param config: The configuration for the task.
    """
    def __init__(self, config: dict = DEFAULT_CONFIG, init_env_callback = None):
        self.object_bodys = []
        self.object_sites = []
        self.object_joints = []
        self.goal_bodys = []
        self.goal_sites = []
        self.goal_joints = []
        self.randomized_object_positions = []
        self.level_name = None
        self.random_actor = False
        self.random_actor_position = False
        self.random_actor_rotation = False
        self.actors = []
        self.actors_spawnable = []
        self.center = []
        self.bound = []
        self.description = []
        self.grpc_addr = None

        self.random_light = False
        self.random_light_position = False
        self.random_actor_rotation = False
        self.lights = []
        self.lights_spawnable = []
        self.light_center = []
        self.light_bound = []
        self.light_description = []

        self.load_config(config)
        self.task_dict = {}

        self.scene = orca_gym_scene.OrcaGymScene(self.grpc_addr)

        self._init_env_callback = init_env_callback

    def get_object_sites(self):
        return self.object_sites

    def get_goal_sites(self):
        return self.goal_sites

    def get_object_bodys(self):
        return self.object_bodys

    def get_goal_bodys(self):
        return self.goal_bodys

    def set_objects(self, objects: list[str]):
        self.object_bodys = objects

    def set_goals(self, goals: list[str]):
        self.goal_bodys = goals

    def set_object_sites(self, object_sites: list[str]):
        self.object_sites = object_sites

    def set_goal_sites(self, goal_sites: list[str]):
        self.goal_sites = goal_sites

    def load_config(self, config: dict):
        """
        Load task configuration from a dictionary.
        """
        if config is None or len(config) == 0:
            return
        self.object_bodys = self.__get_config_setting__("object_bodys", config)
        self.object_sites = self.__get_config_setting__("object_sites", config)
        self.object_joints = self.__get_config_setting__("object_joints", config)
        self.goal_bodys = self.__get_config_setting__("goal_bodys", config)
        self.goal_sites = self.__get_config_setting__("goal_sites", config)
        self.goal_joints = self.__get_config_setting__("goal_joints", config)

        self.level_name = self.__get_config_setting__("level_name", config)
        self.random_actor = self.__get_config_setting__("random_actor", config)
        self.random_actor_position = self.__get_config_setting__("random_actor_position", config)
        self.random_actor_rotation = self.__get_config_setting__("random_actor_rotation", config)
        self.center = self.__get_config_setting__("center", config)
        self.bound = self.__get_config_setting__("bound", config)
        self.actors = self.__get_config_setting__("actors", config)
        self.actors_spawnable = self.__get_config_setting__("actors_spawnable", config)
        self.actor_description = self.__get_config_setting__("description", config)

        self.lights = self.__get_config_setting__("lights", config)
        self.lights_spawnable = self.__get_config_setting__("lights_spawnable", config)
        self.light_center = self.__get_config_setting__("light_center", config)
        self.light_bound = self.__get_config_setting__("light_bound", config)
        self.random_light = self.__get_config_setting__("random_light", config)
        self.random_light_position = self.__get_config_setting__("random_light_position", config)
        self.random_light_rotation = self.__get_config_setting__("random_light_rotation", config)
        self.light_description = self.__get_config_setting__("light_description", config)
        self.num_lights = self.__get_config_setting__("num_lights", config)
        self.grpc_addr = self.__get_config_setting__("grpc_addr", config)

    def random_objs_and_goals(self, env: OrcaGymLocalEnv, bounds = 0.1):
        """
        Randomly select objects and goals.
        """
        object_bodys_env_names = [env.body(body_name) for body_name in self.object_bodys]
        goal_bodys_env_names = [env.body(body_name) for body_name in self.goal_bodys]

        object_ids = [env.model.body_name2id(body_name) for body_name in object_bodys_env_names]
        goal_ids = [env.model.body_name2id(body_name) for body_name in goal_bodys_env_names]

        object_sites_env_names = [env.site(site_name) for site_name in self.object_sites]
        goal_sites_env_names = [env.site(site_name) for site_name in self.goal_sites]

        object_site_dict = env.query_site_pos_and_quat(object_sites_env_names)
        goal_site_dict = env.query_site_pos_and_quat(goal_sites_env_names)

        object_joints_env_names = [env.joint(joint_name) for joint_name in self.object_joints]
        goal_joints_env_names = [env.joint(joint_name) for joint_name in self.goal_joints]

        #随机化物体位置
        def generate_random_pos_quat(bounds, site_dict, sites_env_names):
            random_pos, random_xquat = [], []
            for i in range(len(sites_env_names)):
                site_pos = site_dict[sites_env_names[i]]['xpos']
                site_quat = site_dict[sites_env_names[i]]['xquat']
                site_rotation = rotations.quat2euler(site_quat)

                pos, xquat, rotation = site_pos, site_quat, site_rotation

                pos[0] = np.random.uniform(-bounds, bounds) + pos[0]
                pos[1] = np.random.uniform(-bounds, bounds) + pos[1]
                rotation[2] = np.random.uniform(-bounds * np.pi, bounds * np.pi)
                xquat = rotations.euler2quat(rotation)

                random_pos.append(pos)
                random_xquat.append(xquat)

            return random_pos, random_xquat

        random_obj_pos, random_obj_xquat = generate_random_pos_quat(bounds, object_site_dict, object_sites_env_names)
        random_goal_pos, random_goal_xquat = generate_random_pos_quat(bounds, goal_site_dict, goal_sites_env_names)

        for i in range(len(object_ids)):
            env.set_joint_qpos({object_joints_env_names[i]: np.concatenate([random_obj_pos[i], random_obj_xquat[i]])})

        #for i in range(len(goal_ids)):
            #env.set_mocap_pos_and_quat({goal_bodys_env_names[i]: {'pos': random_goal_pos[i], 'quat': random_goal_xquat[i]}})
        self.randomized_object_positions = env.query_joint_qpos(object_joints_env_names)
        self.randomized_goal_positions = env.query_joint_qpos(goal_joints_env_names)

    def generate_actors(self):
        idxs = []
        if self.random_actor:
            total = len(self.actors)
            n_select = random.randint(1, total)
            idxs = random.sample(range(total), n_select)
            for i in idxs:
                self.add_actor(self.actors[i], self.actors_spawnable[i])
        return idxs
    
    def generate_lights(self):
        """
        只在 random_light=True 时运行，
        并且固定选 self.num_lights 盏，生成朝下的灯。
        """
        idxs = []
        if not self.random_light:
            return idxs

        total = len(self.lights)
        # 限制不超过总数
        n = min(self.num_lights, total)
        idxs = random.sample(range(total), n)

        for i in idxs:
            name      = self.lights[i]
            spawnable = self.lights_spawnable[i]
            print(f"[Debug-Light] add_light → {name} at spawnable {spawnable}")
            self.add_light(name, spawnable)
        return idxs

    def set_actors(self, actors, actors_spawnable, position, rotation):
        for i in range(actors):
            self.add_actor_with_pose(actors[i], actors_spawnable[i], position[i], rotation[i])

    def add_actor(self, actor_name, spawnable_name):
        random_pos, random_xquat = self._random_position_and_rotation(self.center, self.bound)

        if self.random_actor_position:
            position = random_pos
        else:
            position = self.center

        if self.random_actor_rotation:
            rotation = random_xquat
        else:
            rotation = rotations.euler2quat([0, 0, 0])

        actor = orca_gym_scene.Actor(actor_name, spawnable_name, position, rotation, scale=1.0)
        self.scene.add_actor(actor)

    def add_light(self, light_name, spawnable_name):
        # 1) 位置：要么随机，要么固定在 center
        rand_pos, rand_quat = self._random_position_and_rotation(self.light_center, self.light_bound)
        position = rand_pos   if self.random_light_position else np.array(self.light_center)

        # 2) 旋转：要么随机，要么固定“竖直向下”
        if self.random_light_rotation:
            rotation = rand_quat
        else:
            # 绕 X 轴 -90° → 光照沿 −Z 方向（竖直向下）
            rotation = rotations.euler2quat(np.array([-np.pi/2, 0.0, 0.0]))

        actor = orca_gym_scene.Actor(
            name=light_name,
            spawnable_name=spawnable_name,
            position=position,
            rotation=rotation,
            scale=1.0
        )
        self.scene.add_actor(actor)

    def set_light_info(self, light_name, color=None, intensity=None):
        """
        Set the light information for a specific light actor.
        :param light_name: The name of the light actor.
        :param color: The color of the light (optional).
        :param intensity: The intensity of the light (optional).
        """
        print(f"[Debug-Light] set_light_info → {light_name}, color={color}, intensity={intensity}")
        if color is None:
            color = np.array([np.random.uniform(0.0, 1.0),
                              np.random.uniform(0.0, 1.0),
                              np.random.uniform(0.0, 1.0)])
        if intensity is None:
            intensity = np.random.uniform(1000, 2000.0)

        light_info = orca_gym_scene.LightInfo(color=color, intensity=intensity)
        self.scene.set_light_info(light_name, light_info)

    def set_actor_material(self, actor_name, base_color=None):
        """
        Set the material information for a specific actor.
        :param actor_name: The name of the actor.
        :param base_color: The base color of the material (optional).
        """
        if base_color is None:
            base_color = np.array([np.random.uniform(0.0, 1.0),
                                   np.random.uniform(0.0, 1.0),
                                   np.random.uniform(0.0, 1.0),
                                   1.0])
        material_info = orca_gym_scene.MaterialInfo(base_color=base_color)
        self.scene.set_material_info(actor_name, material_info)

    def add_actor_with_pose(self, actor_name, spawnable_name, position, rotation):
        actor = orca_gym_scene.Actor(actor_name, spawnable_name, position, rotation, scale = 1.0)
        self.scene.add_actor(actor)

    def publish_scene(self):
        self.scene.publish_scene()

        #OrcaStudio renew mjcf xml after publish_scene
        #env should init env because mj_model could change
        if self._init_env_callback is None:
            warnings.warn("Not register init_env_callback\n")

        time.sleep(2)
        self._init_env_callback()

    def destory_scene(self):
        self.scene.publish_scene()
        self.scene.close()

    def register_init_env_callback(self, init_env_call):
        self._init_env_callback = init_env_call

    def get_task(self, env: OrcaGymLocalEnv):
        pass

    def is_success(self, env: OrcaGymLocalEnv):
        pass

    def get_language_instruction(self) -> str:
        pass

    def __get_config_setting__(self, attr, config: dict):
        if attr in config:
            return config[attr]
        else:
            return DEFAULT_CONFIG[attr]
    
    def __check_config__(self):
        object_len = self.object_bodys.__len__()
        object_joint_len = self.object_joints.__len__()
        object_site_len = self.object_sites.__len__()

        goal_len = self.goal_bodys.__len__()
        goal_joint_len = self.goal_joints.__len__()
        goal_site_len = self.goal_sites.__len__()

    def _random_position_and_rotation(self, center, bound):
        """
        Generate a random position and rotation within the specified bounds.
        """
        position = np.array([np.random.uniform(center[0] + bound[0][0], center[0] + bound[0][1]),
                             np.random.uniform(center[1] + bound[1][0], center[1] + bound[1][1]),
                             np.random.uniform(center[2] + bound[2][0], center[2] + bound[2][1])])
        rotation = rotations.euler2quat(np.array([np.random.uniform(-np.pi, np.pi), 
                                                  np.random.uniform(-np.pi, np.pi), 
                                                  np.random.uniform(-np.pi, np.pi)]))
        return position, rotation