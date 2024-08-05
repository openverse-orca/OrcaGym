import numpy as np
from gym import utils
from gym import spaces
from d4rl import offline_env
import os
from orca_gym.orca_gym_env import OrcaGymEnv
from orca_gym.orca_gym_env import ActionSpaceType

ADD_BONUS_REWARDS = True

class DoorEnvV0(OrcaGymEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(
        self,
        frame_skip: int = 5,
        action_space_type = ActionSpaceType.DISCRETE,
        action_step_count = 100,        
        grpc_address: str = 'localhost:50051',
        agent_names: list = ['forearm'],
        time_step: float = 0.016,  # 0.016 for 60 fps
        **kwargs,
    ):
        OrcaGymEnv.__init__(
            self,
            frame_skip,
            observation_space=None,
            action_space_type = action_space_type,
            action_step_count = action_step_count,
            grpc_address = grpc_address,
            agent_names = agent_names,       
            time_step=time_step,     
            **kwargs,
        )   

        offline_env.OfflineEnv.__init__(self, **kwargs)
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_door.xml', 5)
        
        # Override action_space to -1, 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape)
        
        # change actuator sensitivity
        gain_prm_set_list = [
            {"actuator_name": "A_WRJ1", f"{self.agent_names[0]}_gain_prm": [10, 0, 0]}, 
            {"actuator_name": "A_WRJ0", f"{self.agent_names[0]}_gain_prm": [10, 0, 0]},
            {"actuator_name": "A_FFJ3", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_FFJ2", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_FFJ1", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_FFJ0", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_MFJ4", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_MFJ3", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_MFJ2", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_MFJ1", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_MFJ0", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_RFJ4", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_RFJ3", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_RFJ2", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_RFJ1", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_RFJ0", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_LFJ4", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_LFJ3", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_LFJ2", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_LFJ1", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_LFJ0", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_THJ4", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_THJ3", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_THJ2", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_THJ1", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]},
            {"actuator_name": "A_THJ0", f"{self.agent_names[0]}_gain_prm": [1, 0, 0]}
        ]
        
        self.set_actuator_gain_prm(gain_prm_set_list)

        bias_prm_set_list = [
            {"actuator_name": "A_WRJ1", f"{self.agent_names[0]}_bias_prm": [0, -10, 0]}, 
            {"actuator_name": "A_WRJ0", f"{self.agent_names[0]}_bias_prm": [0, -10, 0]},
            {"actuator_name": "A_FFJ3", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_FFJ2", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_FFJ1", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_FFJ0", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_MFJ4", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_MFJ3", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_MFJ2", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_MFJ1", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_MFJ0", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_RFJ4", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_RFJ3", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_RFJ2", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_RFJ1", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_RFJ0", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_LFJ4", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_LFJ3", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_LFJ2", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_LFJ1", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_LFJ0", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_THJ4", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_THJ3", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_THJ2", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_THJ1", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]},
            {"actuator_name": "A_THJ0", f"{self.agent_names[0]}_bias_prm": [0, -1, 0]}
        ]
        
        self.set_actuator_bias_prm(bias_prm_set_list)

        # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        # self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        # self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        utils.EzPickle.__init__(self)
        ob = self.reset_model()
        actuator_ctrlrange = self.model.get_actuator_ctrlrange()
        self.act_mid = np.mean(actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(actuator_ctrlrange[:,1]-actuator_ctrlrange[:,0])
        # self.door_hinge_did = self.model.jnt_dofadr[self.model.joint_name2id('door_hinge')]
        self.door_hinge_name = f"{self.agent_names[0]}_door_hinge"
        self.grasp_sid = self.model.site_name2id('S_grasp')
        self.handle_sid = self.model.site_name2id('S_handle')
        self.door_bid = self.model.body_name2id('frame')

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()

        door_pos = self.query_joint_qpos([self.door_hinge_name])[self.door_hinge_name]

        # get to handle
        reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
        # open door
        reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        reward += -1e-5*np.sum(self.data.qvel**2)

        if ADD_BONUS_REWARDS:
            # Bonus
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10

        goal_achieved = True if door_pos >= 1.35 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_did]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qp[-1]
        return np.concatenate([qp[1:-2], [latch_pos], door_pos, palm_pos, handle_pos, palm_pos-handle_pos, [door_open]])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if door open for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
