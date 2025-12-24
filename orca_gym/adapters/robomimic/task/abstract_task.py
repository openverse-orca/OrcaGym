import json
import re

from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.utils import rotations
from orca_gym.adapters.robomimic.task.scene_manage import SceneManager

from orca_gym.scene.orca_gym_scene import Actor, MaterialInfo, LightInfo, OrcaGymScene
import numpy as np
import random
import warnings, time

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


DEFAULT_CONFIG = {
    "random_object": True, # 是否随机Object的位置，object应该在Agent下面
    "type": "pick_and_place", # 任务类型，默认是pick_and_place
    "object_bodys": [],
    "object_sites": [],
    "object_joints": [],

    "random_goal": False, # 是否随机Goal的位置，goal应该在Agent下面
    "goal_bodys": [],
    "goal_sites": [],
    "goal_joints": [],

    "level_name": None,
    "random_cycle": 20,
    "range": [0.0, 0.0, 0.0, 0.0],
    "random_actor": False, # 是否随机添加Actor，actor与Agent同级
    "random_actor_position": False, # 是否随机Actor的位置
    "random_actor_rotation": False, # 是否随机Actor的旋转
    "random_actor_color": False, # 是否随机Actor的颜色
    "actors": [], # actor的命名
    "actors_spawnable": [], # actor的spawnable name, spawnable在Asset/Prefabs下面
    "center": [0, 0, 0], # actor的中心位置
    "infinity": [1000, 1000, 1], # actor的无限远位置
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
    "grpc_addr": "localhost:50051"
}

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
        self.type = "pick_and_place"  # 任务类型，默认是pick_and_place
        self.range = {}
        self.random_actor = False
        self.__random_count__ = 0 # 用于随机Actor的计数， 每次随机摆放Actor和灯光都要+1
        self.random_cycle = 20
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

        self.scene = SceneManager(grpc_addr=self.grpc_addr, init_env_callback=init_env_callback)

        self.first_spawn = True
        self.data = {} #来着增广的数据
        self.sample_range = 0.0 #增广采样范围
        self.target_object = None


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

    def get_object_joints_xpos(self, env: OrcaGymLocalEnv):
        env_object_joints = [env.joint(joint_name) for joint_name in self.object_joints]
        return env.query_joint_qpos(env_object_joints)

    def get_goal_joints_xpos(self, env: OrcaGymLocalEnv):
        env_goal_joints = [env.joint(joint_name) for joint_name in self.goal_joints]
        return env.query_joint_qpos(env_goal_joints)

    def get_object_xpos_xmat_xquat(self, env: OrcaGymLocalEnv):
        env_objects_body = [env.body(body_name) for body_name in self.object_bodys]
        return env.get_body_xpos_xmat_xquat(env_objects_body)

    def get_goal_xpos_xmat_xquat(self, env: OrcaGymLocalEnv):
        env_goals_body = [env.body(body_name) for body_name in self.goal_bodys]
        return env.get_body_xpos_xmat_xquat(env_goals_body)

    def generate_object(self, env:OrcaGymLocalEnv, pick_min, pick_max):
        '''
        从场景中随机挑选actors里的几个
        '''
        if self.random_actor:
            if self.__random_count__ % self.random_cycle == 0:
                # 将object_bodys放回到无限远处
                pos = self.infinity + [1, 0, 0, 0]
                qpos_dict = {env.joint(joint_name): pos for joint_name in self.object_joints}
                env.set_joint_qpos(qpos_dict)

                n_select  = random.randint(pick_min, pick_max)
                idxs      = random.sample(range(len(self.actors)), k=n_select)
                self.object_bodys = [self.actors[idx] for idx in idxs ]
                self.object_sites = [f"{self.actors[idx]}site" for idx in idxs]
                self.object_joints = [f"{self.actors[idx]}_joint" for idx in idxs]

            self.__random_count__ += 1

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
        self.range = self.__get_config_setting__("range", config)
        self.type = self.__get_config_setting__("type", config)
        self.random_cycle = self.__get_config_setting__("random_cycle", config)
        self.random_actor = self.__get_config_setting__("random_actor", config)
        self.random_actor_position = self.__get_config_setting__("random_actor_position", config)
        self.random_actor_rotation = self.__get_config_setting__("random_actor_rotation", config)
        self.random_actor_color = self.__get_config_setting__("random_actor_color", config)
        self.center = self.__get_config_setting__("center", config)
        self.infinity = self.__get_config_setting__("infinity", config)
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

    def spawn_scene(self, env: OrcaGymLocalEnv):
        # 使用 run_mode 判断是否为增广任务
        from envs.manipulation.dual_arm_env import RunMode
        is_augmentation_mode = (env._run_mode == RunMode.POLICY_NORMALIZED and 
                                hasattr(env, '_task') and 
                                hasattr(env._task, 'data') and 
                                env._task.data is not None)
        if self.first_spawn and not is_augmentation_mode:
            self.scene.publish_scene_without_init_env()
            self.generate_actors()
            self.scene.publish_scene()
            self.first_spawn = False
        elif is_augmentation_mode:
            # 增广模式下需要生成 actors（优化：减少 publish_scene 调用次数）
            self.scene.publish_scene_without_init_env()
            self.generate_actors()
            self.scene.publish_scene()
            if self.random_light and self.__random_count__ % self.random_cycle == 0:
                light_idxs = self.generate_lights()
                # 移除额外的 publish_scene_without_init_env() 调用，避免频繁发布导致崩溃
                # self.scene.publish_scene_without_init_env()  # ❌ 删除这一行

                for idx in light_idxs:
                    self.set_light_info(self.lights[idx])
                for actor in self.actors:
                    self.set_actor_material(actor)
        elif self.random_light:
        # 周期性随机添加灯光
            if self.random_light and self.__random_count__ % self.random_cycle == 0:
                self.scene.publish_scene_without_init_env()
                self.generate_actors()
                light_idxs = self.generate_lights()
                self.scene.publish_scene()

                for idx in light_idxs:
                    self.set_light_info(self.lights[idx])
                for actor in self.actors:
                    self.set_actor_material(actor)

    #todo: 需要重写这部分代码， 这里应该只需要做随机位置，位置是否合理应该由具体任务决定
    def random_objs_and_goals(self, env: OrcaGymLocalEnv, random_rotation=True, target_obj_joint_name=None):
        object_bodys = [env.body(bn) for bn in self.object_bodys]
        obj_joints  = [env.joint(jn) for jn in self.object_joints]
        goal_joints = [env.joint(jn) for jn in self.goal_joints]

        dummy = env.site("dummy_site")
        info  = env.query_site_pos_and_quat([dummy])[dummy]
        base_pos, base_quat = info["xpos"], info["xquat"]

        def _get_qpos_not_in_goal_range(goal0_pos, base_pos, base_quat, joint):
            if target_obj_joint_name is not None:
                target_obj_pos = env.query_joint_qpos([target_obj_joint_name])[target_obj_joint_name][:3]
            else:
                target_obj_pos = None

            while True:
                lx = np.random.uniform(*self.range["x"])
                ly = np.random.uniform(*self.range["y"])
                lz = self.range["z"]
                lr = self.range["r"]
                local_pos = np.array([lx, ly, lz], dtype=np.float32)

                world_pos = rotations.quat_rot_vec(base_quat, local_pos) + base_pos
                yaw = np.random.uniform(np.pi * 0.25, np.pi * 0.75)
                world_quat = rotations.euler2quat([0.0, 0.0, yaw])
                obj_qpos = np.concatenate([world_pos, world_quat])

                if np.linalg.norm(world_pos[:2] - goal0_pos[:2]) > lr:
                    if target_obj_pos is not None and target_obj_joint_name != joint:
                        if np.linalg.norm(world_pos[:2] - target_obj_pos[:2]) < lr:
                            # print(f"Too close between {joint} and {target_obj_joint_name}, distance: {np.linalg.norm(world_pos[:2] - target_obj_pos[:2])}")
                            continue
                    break
            
            return obj_qpos

        def _find_obj_place_no_contact(find_target_obj=False):
            placed = []
            goal0_pos = env.query_joint_qpos(goal_joints)[goal_joints[0]][:3]
            placed_body = []
            for joint, body in zip(obj_joints, object_bodys):
                if find_target_obj and joint != target_obj_joint_name:
                    continue

                # 跳过落进目标的
                obj_qpos = _get_qpos_not_in_goal_range(goal0_pos, base_pos, base_quat, joint)
                # print("[Debug] Placing object on joint:", joint, "at position:", obj_qpos)
                if not random_rotation:
                    org_qpos = env.query_joint_qpos([joint])[joint]
                    obj_qpos[3:] = org_qpos[3:]
                env.set_joint_qpos({joint: obj_qpos})
                env.mj_forward()
                placed_body.append(body)

                # 跳过有碰撞的
                contacts = env.query_contact_simple()
                find_contact = False
                for contact in contacts:
                    body1 = env.model.get_geom_body_name(contact["Geom1"])
                    body2 = env.model.get_geom_body_name(contact["Geom2"])
                    if body1 in placed_body and body2 in placed_body:
                        # print(f"[Debug] Found contact between {body1} and {body2}")
                        find_contact = True
                        break
                if find_contact:
                    return None

                placed.append((joint, obj_qpos))
            
            return placed

        placed = None
        for i in range(100):  # 尝试100次
            placed = _find_obj_place_no_contact(find_target_obj=True)
            other_placed = _find_obj_place_no_contact(find_target_obj=False)
            if placed is not None and other_placed is not None:
                placed.extend(other_placed)
                break

        # 容错处理
        if placed is None:
            _logger.error(f"Warning: Failed to place objects! Falling back to default positions.")
            placed = [(joint, obj_qpos) for joint, obj_qpos in zip(obj_joints, [np.array([0, 0, 0, 1, 0, 0, 0])] * len(obj_joints))]

        # 一次性写回
        qpos_dict = {jn: q for jn, q in placed}
        env.set_joint_qpos(qpos_dict)
        env.mj_forward()

        self.randomized_object_positions = env.query_joint_qpos(obj_joints)
        self.randomized_goal_positions   = env.query_joint_qpos(goal_joints)


    def generate_actors(self):
        '''
        将所有的 actors 添加到场景中, 初始化到infinity位置
        '''
        for i in range(len(self.actors)):
            self.scene.add_actor(actor_name=self.actors[i], asset_path=self.actors_spawnable[i],
                                 position=np.array(self.infinity), rotation = rotations.euler2quat([0, 0, 0]))
    
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
            # 移除调试打印，减少输出
            # print(f"[Debug-Light] add_light → {name} at spawnable {spawnable}")
            self.add_light(name, spawnable)
        return idxs

    def set_actors(self, actors, actors_spawnable, position, rotation):
        for i in range(actors):
            self.add_actor_with_pose(actors[i], actors_spawnable[i], position[i], rotation[i])

    def add_actor(self, actor_name, asset_path):
        random_pos, random_xquat = self._random_position_and_rotation(self.center, self.bound)

        if self.random_actor_position:
            position = random_pos
        else:
            position = self.infinity    # 初始化到无限远位置

        if self.random_actor_rotation:
            rotation = random_xquat
        else:
            rotation = rotations.euler2quat([0, 0, 0])

        self.scene.add_actor(actor_name, asset_path, position, rotation, scale=1.0)


    def add_light(self, light_name, asset_path):
        # 1) 位置：要么随机，要么固定在 center
        rand_pos, rand_quat = self._random_position_and_rotation(self.light_center, self.light_bound)
        position = rand_pos   if self.random_light_position else np.array(self.light_center)

        # 2) 旋转：要么随机，要么固定“竖直向下”
        if self.random_light_rotation:
            rotation = rand_quat
        else:
            # 绕 X 轴 -90° → 光照沿 −Z 方向（竖直向下）
            rotation = rotations.euler2quat(np.array([-np.pi/2, 0.0, 0.0]))

        self.scene.add_light(light_name, asset_path, position, rotation, scale=1.0)

    def set_light_info(self, light_name, color=None, intensity=None):
        """
        Set the light information for a specific light actor.
        :param light_name: The name of the light actor.
        :param color: The color of the light (optional).
        :param intensity: The intensity of the light (optional).
        """
        # 移除调试打印，减少输出
        # print(f"[Debug-Light] set_light_info → {light_name}, color={color}, intensity={intensity}")
        if color is None:
            color = np.array([np.random.uniform(0.0, 1.0),
                              np.random.uniform(0.0, 1.0),
                              np.random.uniform(0.0, 1.0)])
        if intensity is None:
            intensity = np.random.uniform(1000, 2000.0)

        self.scene.set_light_info(light_name, color, intensity)

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
        # 移除调试打印，减少输出
        # print(f"[Debug-Actor] set_actor_material → {actor_name}, base_color={base_color}")
        self.scene.set_material_info(actor_name, base_color)

    def add_actor_with_pose(self, actor_name, asset_path, position, rotation):
        self.scene.add_actor(actor_name, asset_path, position=position, rotation=rotation, scale=1.0)

    def publish_scene(self):
        self.scene.publish_scene()

    def destory_scene(self):
        self.scene.destory_scene()

    def register_init_env_callback(self, init_env_call):
        self.scene.register_init_env_callback(init_env_call)

    def get_task(self, env: OrcaGymLocalEnv):
        raise NotImplementedError("This method should be overridden by subclasses")

    def is_success(self, env: OrcaGymLocalEnv):
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_language_instruction(self) -> str:
        raise NotImplementedError("This method should be overridden by subclasses")

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

    def _restore_objects_(self, env: OrcaGymLocalEnv, objects_data):
        """
        恢复物体到指定位置
        :param positions: 物体位置字典
        """
        qpos_dict = {}
        self.object_bodys, self.object_joints = [], []
        if objects_data.shape == () and objects_data.dtype == "object":
            json_str = objects_data[()]
            json_data = json.loads(json_str)
            for object, object_info in json_data.items():
                joint_name = object_info['joint_name']
                pos = np.array(object_info['position'], dtype=np.float32)
                quat = np.array(object_info['orientation'], dtype=np.float32)
                qpos_dict[env.joint(joint_name)] = np.concatenate([pos, quat], axis=0)
                self.object_bodys.append(object)
                self.object_joints.append(joint_name)
        else:
            arr = objects_data
            empty_name = env.joint("")
            for entry in arr:
                name = entry['joint_name']
                pos = entry['position']
                quat = entry['orientation']
                qpos_dict[name] = np.concatenate([pos, quat], axis=0)
                origin_joint = str(name, encoding='utf-8').removeprefix(empty_name)
                origin_body = origin_joint.removesuffix("_joint")
                self.object_bodys.append(origin_body)
                self.object_joints.append(origin_joint)
        env.set_joint_qpos(qpos_dict)

        env.mj_forward()

    def _restore_goals_(self, env: OrcaGymLocalEnv, goals_data):
        """
        恢复目标到指定位置
        :param positions: 目标位置字典
        """
        qpos_dict = {}
        self.goal_bodys, self.goal_joints = [], []
        if goals_data.shape == () and goals_data.dtype == "object":
            json_str = goals_data[()]
            json_data = json.loads(json_str)
            for object, object_info in json_data.items():
                joint_name = object_info['joint_name']
                pos = np.array(object_info['position'], dtype=np.float32)
                quat = np.array(object_info['orientation'], dtype=np.float32)
                qpos_dict[env.joint(joint_name)] = np.concatenate([pos, quat], axis=0)
                self.goal_bodys.append(object)
                self.goal_joints.append(joint_name)
        env.set_joint_qpos(qpos_dict)
        env.mj_forward()

    def get_objects_info(self, env: OrcaGymLocalEnv) -> str:
        info = {}
        jpos_dict = self.get_object_joints_xpos(env)
        xpos, xmat, xquat = self.get_object_xpos_xmat_xquat(env)
        for i in range(len(self.object_bodys)):
            info[self.object_bodys[i]] = {
                "joint_name": self.object_joints[i],
                "position": jpos_dict[env.joint(self.object_joints[i])].tolist(),
                "orientation": xquat[i * 4:(i + 1)*4].tolist(),
                "target_body": self.target_object == self.object_bodys[i]
            }
        # 移除调试打印，减少IK模式下的输出
        # print(f"info: {info}")

        json_str = json.dumps(info)
        return json_str

    def get_goals_info(self, env: OrcaGymLocalEnv) -> str:
        info = {}
        jpos_dict = self.get_goal_joints_xpos(env)
        xpos, xmat, xquat = self.get_goal_xpos_xmat_xquat(env)
        for i in range(len(self.goal_bodys)):
            geom_size = env.get_goal_bounding_box(env.body(self.goal_bodys[i]))
            if not geom_size:
                continue
            mn = np.array(geom_size["min"]).flatten()
            mx = np.array(geom_size["max"]).flatten()
            sz = mx - mn
            info[self.goal_bodys[i]] = {
                "joint_name": self.goal_joints[i],
                "position": jpos_dict[env.joint(self.goal_joints[i])].tolist(),
                "orientation": xquat[i * 4:(i + 1)*4].tolist(),
                "min": mn.tolist(),
                "max": mx.tolist(),
                "size": sz.tolist()
            }

        json_str = json.dumps(info)
        return json_str

    def resample_objects(self, env: OrcaGymLocalEnv, sample_range: float = 0.0):
        """
        Resample objects within a specified range.
        :param sample_range: The range within which to resample objects.
        """
        if sample_range <= 0.0:
            return

        env_joints = [env.joint(joint_name) for joint_name in self.object_joints]
        joints_pos = env.query_joint_qpos(env_joints)
        for joint, qpos in joints_pos.items():
            new_pos = qpos[:2] + np.random.uniform(-sample_range, sample_range, size=2)
            new_qpos = np.concatenate([new_pos, qpos[2:]])
            joints_pos[joint] = new_qpos

        env.set_joint_qpos(joints_pos)
        env.mj_forward()

    def _set_target_object_(self, env: OrcaGymLocalEnv, data: dict):
        qpos_dict = {}
        objects_data = data['objects']
        if objects_data.shape == () and objects_data.dtype == "object":
            json_str = objects_data[()]
            json_data = json.loads(json_str)
            for object, object_info in json_data.items():
                if object_info["target_body"]:
                    self.target_object = object
        #老数据兼容
        else:
            lang_instr = data.get("language_instruction", b"")
            if isinstance(lang_instr, (bytes, bytearray)):
                lang_instr = lang_instr.decode("utf-8")
            obj_match = re.search(r'object:\s*([^\s]+)', lang_instr)
            self.target_object = obj_match.group(1) if obj_match else None
