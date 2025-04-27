from orca_gym.robomimic.robomimic_env import RobomimicEnv

from orca_gym.utils import rotations
import numpy as np

class AbstractTask:
    """
    :param prompt: The prompt for the task.
    :param config: The configuration for the task.
    """
    def __init__(self, config: dict):
        self.object_bodys = []
        self.object_sites = []
        self.object_joints = []
        self.goal_bodys = []
        self.goal_sites = []
        self.goal_joints = []
        self.randomized_object_positions = []
        self.level_name = None
        self.load_config(config)
        self.task_dict = {}

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
        self.object_bodys = config["object_bodys"]
        self.object_sites = config["object_sites"]
        self.object_joints = config["object_joints"]
        self.goal_bodys = config["goal_bodys"]
        self.goal_sites = config["goal_sites"]
        self.goal_joints = config["goal_joints"]
        self.level_name = config["level_name"]

    def random_objs_and_goals(self, env: RobomimicEnv, bounds = 0.1):
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
    def get_task(self, env: RobomimicEnv):
        pass

    def is_success(self, env: RobomimicEnv):
        pass

    def get_language_instruction(self) -> str:
        pass