import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.orca_gym_env import OrcaGymRemoteEnv, ActionSpaceType
from orca_gym.utils.low_pass_filter import LowPassFilter



class FrankaEnv(OrcaGymRemoteEnv):
    r"""
    ## Description
    "Franka" is a multi-jointed robot arm used in various manipulation tasks.
    The goal is to move a target object to a goal position using the robot's end effector.
    The robot consists of several joints and a gripper controlled by tendons.

    ## Arguments
    FrankaEnv provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.

    - `frame_skip` (int): Number of simulation steps per environment step.
    - `grpc_address` (str): Address for the gRPC server.
    - `agent_names` (list): Names of the agents in the environment.
    - `time_step` (float): Simulation time step.
    """


    def __init__(
        self,
        frame_skip: int = 5,
        grpc_address: str = 'localhost:50051',
        agent_names: list = ['Agent0'],
        time_step: float = 0.016,  # 0.016 for 60 fps        
        action_space_type = ActionSpaceType.DISCRETE,
        action_step_count = 100,        
        **kwargs,
    ):
        OrcaGymRemoteEnv.__init__(
            self,
            frame_skip,
            grpc_address = grpc_address,
            agent_names = agent_names,       
            time_step=time_step,               
            observation_space=None,
            action_space_type = action_space_type,
            action_step_count = action_step_count,
            **kwargs,
        )   


        self.alpha = kwargs.get('alpha', 0.5)  # 默认 alpha 为 0.5
        print("Alpha: ", self.alpha)
        self.update_goal_interval = kwargs.get('update_goal_interval', 5000)
        print("Update Goal Interval: ", self.update_goal_interval)
        self.update_conter = 0
        self.total_steps = 0

        # 初始信息
        self.goal_qpos = self.query_joint_qpos(['Toys_Box1'])['Toys_Box1']
        self.obs_joint_limits = self.query_obs_joint_limits()
        print("Joint Limits: ", self.obs_joint_limits)

        # 查询一次观测空间数据，获取数组大小
        qpos_dict, qpos, qvel_dict, qvel = self.query_obs_joint_qpos_qvel()
        print("qpos: ", qpos_dict)
        print("qvel: ", qvel_dict)
        actuator_force = self.query_actuator_force()
        print("actuator_force: ", actuator_force)
        xpos_dict, xpos = self.query_obs_xpos()
        print("xpos: ", xpos_dict)
        vel_dict, vel = self.query_obs_velocities()
        print("vel: ", vel_dict)
        cfrc_ext_dict, cfrc_ext = self.query_obs_cfrc_ext()
        print("cfrc_ext: ", cfrc_ext_dict)

        observation_space_size = len(qpos.flatten()) + len(qvel.flatten()) + len(actuator_force.flatten()) + len(xpos.flatten()) + len(vel.flatten()) + len(cfrc_ext.flatten())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space_size,), dtype=np.float32)


        # 初始化低通滤波器，使用初始控制值
        # dafult_ctrl = self.get_default_ctrl()
        # self.filters = [LowPassFilter(self.alpha, initial_output=ctrl) for ctrl in dafult_ctrl]

        # 初始化奖励值
        self.reset_bonus()
        return
    
    def reset_bonus(self):
        self.bonus = {'reach_0.4': 20, 'reach_0.3': 40, 'reach_0.2': 60, 'reach_0.1': 80, 'touch': -10, 'done': 160, 'angle': 50}

    def get_bonus(self, key):
        # 奖励是一次性的，取完后清零
        value = self.bonus[key]
        if value != 0:
            print("Get Bonus: ", key, value)

        self.bonus[key] = 0
        return value

    def update_goal(self):
        update_conter = self.total_steps // self.update_goal_interval
        if update_conter > self.update_conter:
            self.update_conter  = update_conter
            print("Update Goal Position in Step: ", self.total_steps)

            # 读取目标物体的位置
            self.goal_qpos = self.query_joint_qpos(['Toys_Box1'])['Toys_Box1']
            # 修改在xy平面上的位置，以当前位置为圆心，半径为0.3的圆上随机取一个点
            self.goal_qpos[:2] += np.random.uniform(-0.3, 0.3, size=2)
    
        # 设置目标物体的位置
        self.set_joint_qpos({'Toys_Box1': self.goal_qpos})
        self.mj_forward()

    def reset_model(self):
        # 已经执行过reset_simulation，所以这里不需要再次执行
        self.update_goal()

        qpos_dict, _, _, qvel = self.query_obs_joint_qpos_qvel()
        actuator_force = self.query_actuator_force()
        _, xpos = self.query_obs_xpos()
        _, vel = self.query_obs_velocities()
        _, cfrc_ext = self.query_obs_cfrc_ext()
        observation = self._get_observation(qpos_dict, qvel, actuator_force, xpos, vel, cfrc_ext)
        
        self.initialized = False
        self.reset_bonus()  # 重置奖励值

        return observation

    # def filter_actions(self, raw_actions):
    #     return np.array([self.filters[i].apply(raw_actions[i]) for i in range(len(raw_actions))])

    def reset_default_pos(self):
        self.ctrl = self.get_default_ctrl()
        # 设置机器人的默认位置
        self.do_simulation(self.ctrl, 300)

    def step(self, action):
        # 为了避免第一次的随机动作，将第一次的动作设置为默认动作
        if not self.initialized:
            action = self.get_default_ctrl()
            self.reset_default_pos()
            self.initialized = True

        self.total_steps += 1

        self.ctrl = self.apply_action(action)
        self.do_simulation(self.ctrl, self.frame_skip)

        # 查询位置、速度、力矩
        qpos_dict, _, qvel_dict, qvel = self.query_obs_joint_qpos_qvel()
        xpos_dict, xpos = self.query_obs_xpos()
        actuator_force = self.query_actuator_force()
        vel_dict, vel = self.query_obs_velocities()
        cfrc_ext_dict, cfrc_ext = self.query_obs_cfrc_ext()

        # 计算成功条件和惩罚条件
        done = self._is_done(xpos_dict)
        reward = self._compute_reward(qpos_dict, xpos_dict, actuator_force, vel_dict, cfrc_ext_dict, is_done=done)

        observation = self._get_observation(qpos_dict, qvel, actuator_force, xpos, vel, cfrc_ext)
        info = {}

        return observation, reward, done, False, info


    def normalize_joint_position(self, qpos_dict):
        qpos = []
        for limit_info in self.obs_joint_limits:
            qpos_value = qpos_dict[limit_info['joint_name']]
            qpos_min = limit_info['range_min']
            qpos_max = limit_info['range_max']            
            for value in qpos_value:
                if limit_info['has_limit'] == True:
                    # 归一化到 [0, 1]
                    qpos_norm = (value - qpos_min) / (qpos_max - qpos_min)
                    qpos.append(qpos_norm)
                else:
                    # 如果没有限制，可以直接使用原值，或者也可以选择其他归一化方法
                    qpos.append(value)
        return np.array(qpos)
    
    def normalize_joint_velocity(self, qvel):
        # 归一化到 [0, 1]，假设角速度最快为 -2π/s 到 2π/s
        qvel_norm = (qvel + 2 * np.pi) / (4 * np.pi)
        return np.array(qvel_norm)


    def _get_observation(self, qpos_dict, qvel, actuator_force, xpos, vel, cfrc_ext):
        # 归一化关节位置
        qpos = self.normalize_joint_position(qpos_dict)
        qvel = self.normalize_joint_velocity(qvel)

        # 合并成一个观测值向量
        observation = np.concatenate([qpos.flatten(), qvel.flatten(), actuator_force.flatten(), xpos.flatten(), vel.flatten(), cfrc_ext.flatten()])
        return observation.astype(np.float32)

    def _compute_reward(self, qpos_dict, xpos_dict, actuator_force, vel_dict, cfrc_ext_dict, is_done=False):
        # 距离奖励
        distance_to_goal = self._compute_distance_to_goal(xpos_dict)
        reward = 0

        # 如果距离小于一定阈值，给予额外奖励
        if distance_to_goal < 0.4:
            reward += self.get_bonus('reach_0.4')
        if distance_to_goal < 0.3:
            reward += self.get_bonus('reach_0.3')
        if distance_to_goal < 0.2:
            reward += self.get_bonus('reach_0.2')
        if distance_to_goal < 0.1:
            reward += self.get_bonus('reach_0.1')

        if is_done:
            reward += self.get_bonus('done')

        # 每帧计算速度，评估动作的平滑性
        VELOCITY_PENALTY = 0.025
        linear_velocity = np.linalg.norm(vel_dict[self.body("hand")]['linear_velocity']) + np.linalg.norm(vel_dict[self.body("hand")]['linear_velocity'])
        angular_velocity = np.linalg.norm(vel_dict[self.body("hand")]['angular_velocity']) + np.linalg.norm(vel_dict[self.body("hand")]['angular_velocity'])
        reward -= VELOCITY_PENALTY * linear_velocity + VELOCITY_PENALTY * angular_velocity

        # 每帧计算能量消耗成本
        ENERGY_COST_PENALTY = 0.005
        energy_cost = np.sum(np.abs(actuator_force))
        reward -= ENERGY_COST_PENALTY * energy_cost

        # 防止机器人进入缠绕状态，对于关节达到极限位置，持续给予惩罚
        # JOINT_LIMIT_THRESHOLD = np.pi / 180 * 1
        # JOINT_LIMIT_PENALTY = 0.1
        # for limit_info in self.obs_joint_limits:
        #     if limit_info['has_limit'] == True:
        #         qpos_value = qpos_dict[limit_info['joint_name']]
        #         qpos_min = limit_info['range_min']
        #         qpos_max = limit_info['range_max']
        #         for value in qpos_value:
        #             if value - qpos_min < JOINT_LIMIT_THRESHOLD or qpos_max - value < JOINT_LIMIT_THRESHOLD:
        #                 print("Joint Limit Reached! Penalty!")
        #                 reward -= JOINT_LIMIT_PENALTY

        # 防止机器人碰撞地面或者自己的身体
        CONTACT_FORCE_THRESHOLD = 0.0001
        CONTACT_FORCE_PENALTY = 0.1
        for cfrc_ext in cfrc_ext_dict.values():
            contact_force = np.linalg.norm(np.array(cfrc_ext[:3]).flatten())
            if contact_force > CONTACT_FORCE_THRESHOLD:
                print("Contact Force Exceeds Limit! Penalty!")
                reward -= CONTACT_FORCE_PENALTY


        return reward

    def _is_done(self, xpos_dict):
        distance_to_goal = self._compute_distance_to_goal(xpos_dict)
        done = False
        if (distance_to_goal < 0.05):
            done = True
            print("Reach the Goal! Done!")
        return done
    
    def _compute_distance_to_goal(self, xpos_dict):
        # 获取机器人进近起始点和目标点的位置
        hand_pos = np.array(xpos_dict[self.body("hand")])
        target_pos = np.array(xpos_dict["Toys_Box1"])
        distance_to_goal = np.linalg.norm(hand_pos - target_pos)
        return distance_to_goal
    
    def get_default_ctrl(self):
        ctrl = [0, -0.5,  0, -2, 0,  1.5, 1, 0,0 ]
        return np.array(ctrl)

    def query_obs_joint_limits(self):
        OBS_JOINT_NAMES = [self.joint("joint1"), self.joint("joint2"), self.joint("joint3"), self.joint("joint4"), self.joint("joint5"), 
                           self.joint("joint6"), self.joint("joint7"), self.joint("finger_joint1"), self.joint("finger_joint2")]
        return self.query_joint_limits(OBS_JOINT_NAMES)

    def query_obs_joint_qpos_qvel(self):
        OBS_JOINT_NAMES = [self.joint("joint1"), self.joint("joint2"), self.joint("joint3"), self.joint("joint4"), self.joint("joint5"), 
                           self.joint("joint6"), self.joint("joint7"), self.joint("finger_joint1"), self.joint("finger_joint2")]
        qpos_dict = self.query_joint_qpos(OBS_JOINT_NAMES)
        qpos = np.array(list(qpos_dict.values())).flat.copy()
        qvel_dict = self.query_joint_qvel(OBS_JOINT_NAMES)
        qvel = np.array(list(qvel_dict.values())).flat.copy()

        return qpos_dict, qpos, qvel_dict, qvel
    
    def query_obs_xpos(self):
        OBS_BODY_NAMES = [self.body("hand"), "Toys_Box1"]
        body_com_dict = self.get_body_com_dict(OBS_BODY_NAMES)
        xpos_dict = {body_name: body_com_dict[body_name]['Pos'] for body_name in OBS_BODY_NAMES}
        xpos = np.array(list(xpos_dict.values())).flat.copy()
        return xpos_dict, xpos
        
    def query_obs_velocities(self):
        OBS_BODY_NAMES = [self.body("hand"), self.body("hand")]
        vel_dict = self.query_body_velocities(OBS_BODY_NAMES)

        vel_list = []
        for _, v in vel_dict.items():
            vel_list.append(np.array(v['linear_velocity']))
            vel_list.append(np.array(v['angular_velocity']))
        vel = np.array(vel_list).flat.copy()

        return vel_dict, vel
        
    def query_obs_cfrc_ext(self):
        OBS_BODY_NAMES = [self.body("link7"), self.body("hand"), self.body("left_finger"), self.body("right_finger")]
        cfrc_ext_dict, cfrc_ext = self.query_cfrc_ext(OBS_BODY_NAMES)
        return cfrc_ext_dict, cfrc_ext
