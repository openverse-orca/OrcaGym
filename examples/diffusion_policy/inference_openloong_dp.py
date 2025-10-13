#!/usr/bin/env python3
"""
Diffusion Policy Inference for OpenLoong Environment
Clean and brief implementation
"""

import torch
import numpy as np
import time
from typing import Dict, Any
from openloong_openpi_env import OpenLoongOpenpiEnv
from model import *
from model_utils import *
from dataset import *

import dataclasses
import orca_gym.scripts.dual_arm_manipulation as dual_arm_manipulation
from orca_gym.scripts.dual_arm_manipulation import ActionType
from envs.manipulation.dual_arm_env import ControlDevice, RunMode
import openloong_openpi_env as _env
import yaml
import cv2
import pickle


TIME_STEP = dual_arm_manipulation.TIME_STEP
FRAME_SKIP = dual_arm_manipulation.FRAME_SKIP
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

CAMERA_CONFIG = dual_arm_manipulation.CAMERA_CONFIG
RGB_SIZE = dual_arm_manipulation.RGB_SIZE

@dataclasses.dataclass
class Args:
    orca_gym_address: str = '127.0.0.1:50051'
    env_name: str = "DualArmEnv"
    seed: int = 0
    agent_names: str = "openloong_gripper_2f85_fix_base_usda"
    record_time: int = 20 #20
    task: str = "Manipulation"
    obs_type: str = "pixels_agent_pos"
    prompt: str = "level: jiazi  object: bottle_blue to goal: shoppingtrolley_01"

    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000
    
    pico_ports: str = "8001"
    action_step: int = 1
    ctrl_device: str = "keyboard"
    sample_range: float = 0.0
    action_type: ActionType = ActionType.JOINT_POS

    display: bool = False
    
    task_config: str = "bosch_task.yaml"


def initialize_env(args: Args):
    max_episode_steps = int(args.record_time * CONTROL_FREQ)    
    env_index = 0
    camera_config = CAMERA_CONFIG
    task_config_dict = {}
    if args.task_config is not None:
        with open(args.task_config, "r") as f:
            task_config_dict = yaml.safe_load(f)
            
    env_id, kwargs = dual_arm_manipulation.register_env(
        orcagym_addr=args.orca_gym_address,
        env_name=args.env_name, 
        env_index=env_index, 
        agent_names=args.agent_names, 
        pico_ports=args.pico_ports, 
        run_mode=RunMode.POLICY_NORMALIZED, 
        action_step=args.action_step, 
        ctrl_device=args.ctrl_device, 
        max_episode_steps=max_episode_steps, 
        sample_range=args.sample_range, 
        action_type=args.action_type,
        camera_config=camera_config, 
        task_config_dict=task_config_dict,
    )
    environment=_env.OpenLoongOpenpiEnv(
        env_id=env_id,
        seed=args.seed,
        obs_type=args.obs_type,
        prompt=args.prompt,
    )
    return environment

class OpenLoongDiffusionInference:
    def __init__(self, 
                 model_path: str = "openloong_dp_model.ckpt",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Action/observation dimensions - match OpenLoong training script
        self.action_dim = 28  # OpenLoong uses 28D actions
        self.obs_horizon = 1  # OpenLoong uses 1 observation step
        self.action_horizon = 32  # OpenLoong uses 16 action steps
        self.pred_horizon = 32  # OpenLoong predicts 16 steps
        self.obs_dim = 512 + 42  # vision features + state (like OpenLoong)
        
        # Load model
        self._load_model(model_path)
        
        # Initialize environment using the proper registration
        args = Args()  # Use default arguments
        self.env = initialize_env(args)
    
        
        # Normalization stats (you'll need to compute these from your training data)
        self.stats = self._get_default_stats()


        with open('openloong_dp_action.csv', 'w') as f:
            f.write(",".join(np.array(list(range(28))).astype(str)) + "\n")
        with open('openloong_dp_action_normalized.csv', 'w') as f:
            f.write(",".join(np.array(list(range(28))).astype(str)) + "\n")
            
        self.initialize_example_action("/home/user/Desktop/manipulation/OrcaGym/examples/openpi/records_tmp/shop/0b0b05ac_9676d71b")
        
    def _load_model(self, model_path: str):
        """Load the diffusion policy model"""
        # Initialize model architecture
        self.nets = self._create_model()
        
        # Load weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # Load state dict into individual models
            for key, model in self.nets.items():
                if key in state_dict:
                    model.load_state_dict(state_dict[key])
                    print(f"Loaded {key} from {model_path}")
                else:
                    print(f"Warning: {key} not found in {model_path}")
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model not found at {model_path}, using random weights")
            
        
    def _create_model(self):
        """Create the diffusion policy model"""
        # Vision encoder - same as PushT example
        vision_encoder = get_resnet('resnet18')
        vision_encoder = replace_bn_with_gn(vision_encoder)

        # Noise prediction network - match OpenLoong training
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon  # 534 * 1 = 534
        )
        
        return {
            'vision_encoder': vision_encoder.to(self.device),
            'noise_pred_net': noise_pred_net.to(self.device)
        }
    
    def _get_default_stats(self):
        """Default normalization statistics"""
        with open('openloong_dp_stats.pkl', 'rb') as f:
            stats = pickle.load(f)
        # print(stats)
        # exit()

        return stats

    def _normalize_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize observation"""
        # Normalize state
        state = obs['state']

        # nstate = normalize_data(state.reshape(1, -1), self.stats['state']).flatten()
        nstate = (state - self.stats['state']['min']) / (self.stats['state']['max'] - self.stats['state']['min'] + 1e-8)

        
        
        # Images are already normalized to [0,1]
        nimages = {k: v for k, v in obs['images'].items()}
        
        return {
            'state': nstate,
            'images': nimages
        }
        
        
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action"""
        return (action - self.stats['action']['min']) / (self.stats['action']['max'] - self.stats['action']['min'] + 1e-8)
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action"""
        # return unnormalize_data(action, self.stats['action'])

        out = (action * (self.stats['action']['max'] - self.stats['action']['min'] + 1e-8)) + self.stats['action']['min']
        return out
    
    def predict_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """Predict action using diffusion policy"""
        # Normalize observation
        nobs = self._normalize_obs(obs)
        # print(nobs['state'])
        # exit()

        
        # Prepare inputs - same format as PushT
        img = nobs['images']['cam_high']  # Use only the high camera
        
        # # Resize from (3, 224, 224) to (3, 96, 96)
        # # cv2.resize expects HWC format, so transpose first
        if img.shape[0] == 3:  # CHW format
            img_hwc = img.transpose(1, 2, 0)  # Convert to HWC
            img_resized = cv2.resize(img_hwc, (640, 480))
            img = img_resized.transpose(2, 0, 1)  # Convert back to CHW
            # # # save image
            # print(img_resized.shape)
            # cv2.imwrite("img_inference.png", img_resized)
            # exit()
        else:
            img = cv2.resize(img, (480, 640))

        if len(img.shape) == 3:
            img = img[np.newaxis, ...]  # Add batch dimension for obs_horizon
        

        
        images = torch.from_numpy(img).to(self.device, dtype=torch.float32)

        state = torch.from_numpy(nobs['state']).to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            # Get image features - same as PushT
            image_features = self.nets['vision_encoder'](
                images)  # Flatten obs_horizon dimension
            image_features = image_features.reshape(*images.shape[:1], -1)  # (obs_horizon, 512)

            # Concatenate with state - same as PushT
            obs_features = torch.cat([image_features, state.unsqueeze(0)], dim=-1)
            obs_cond = obs_features.flatten().unsqueeze(0)  # (1, obs_horizon * (512+32))
            
            # Initialize action from noise
            noisy_action = torch.randn(
                (1, self.pred_horizon, self.action_dim), 
                device=self.device
            )
            
            # Diffusion sampling
            noise_scheduler = DDPMScheduler(num_train_timesteps=100)
            noise_scheduler.set_timesteps(100)  # Number of denoising steps
            
            for timestep in noise_scheduler.timesteps:
                # Predict noise
                noise_pred = self.nets['noise_pred_net'](
                    sample=noisy_action,
                    timestep=timestep,
                    global_cond=obs_cond
                )
                
                # Denoise
                noisy_action = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=noisy_action
                ).prev_sample
            
            # Get final action
            action = noisy_action[0].cpu().numpy()  # (pred_horizon, action_dim)
            
        # Denormalize and return first action_horizon steps
        unnormalized_action = self._denormalize_action(action)
        return unnormalized_action[:self.action_horizon], action[:self.action_horizon]  # (action_horizon, action_dim)
    
    def run_episode(self, max_steps: int = 200, render: bool = True, use_example_action: bool = False):
        """Run a single episode"""
        print("Starting OpenLoong Diffusion Policy Episode...")
        
        # Reset environment
        self.env.reset()
        obs = self.env.get_observation()
        
        step_count = 0
        total_reward = 0
        
        while step_count < max_steps and not self.env.is_episode_complete():
            # Predict action
            if step_count == 0:
                action = self.get_example_action()

                action = np.ones((action.shape[0], action.shape[1])) * action[0]
                normalized_action = None
            else:
                action, normalized_action = self.predict_action(obs)
            
            
            # Execute action
            for i in range(len(action)):
                if self.env.is_episode_complete():
                    break
                    
                # Apply action
                self.env.apply_action({"actions": action[i]})
            
                
                # Get new observation
                obs = self.env.get_observation()
                
                step_count += 1
                print(f"Step {step_count}, Action {i+1}/{len(action)}")
                
                if render:
                    time.sleep(0.1)  # Small delay for visualization
                    
                # log action to csv
                with open('openloong_dp_action.csv', 'a') as f:
                    f.write(",".join(action[i].astype(str)) + "\n")
                # log action to csv
                if normalized_action is not None:
                    with open('openloong_dp_action_normalized.csv', 'a') as f:
                        f.write(",".join(normalized_action[i].astype(str)) + "\n")
        
        print(f"Episode completed in {step_count} steps")
        return step_count
    
    def initialize_example_action(self, path: str):
        """Initialize example action"""
        import h5py
        path = os.path.join(path, "proprio_stats", "proprio_stats.hdf5")
        with h5py.File(path, 'r') as f:
            self.example_action = f["data"]["demo_00000"]['actions'][:]
            obs = f["data"]["demo_00000"]['obs']
            self.example_state = np.concatenate([
                obs['ee_pos_l'][:], 
                obs['ee_quat_l'][:],
                obs['ee_pos_r'][:], 
                obs['ee_quat_r'][:],
                
                obs['ee_vel_linear_l'][:],
                obs['ee_vel_angular_l'][:],
                obs['ee_vel_linear_r'][:],
                obs['ee_vel_angular_r'][:],
                
                obs['arm_joint_qpos_l'][:], 
                obs['arm_joint_qpos_r'][:], 
                
                obs['grasp_value_l'][:], 
                obs['grasp_value_r'][:]
                ], dtype=np.float32, axis=1)
        
        self.current_action_index = 0
    
    def get_example_action(self):
        """Get example action"""
        output = self.example_action[self.current_action_index:self.current_action_index+self.action_horizon]
        self.current_action_index += self.action_horizon
        return output

    def get_example_state(self):
        return self.example_state[0]


def main():
    """Main inference function"""
    # Initialize inference
    inference = OpenLoongDiffusionInference(model_path="/home/user/Desktop/manipulation/OrcaGym/examples/diffusion_policy/pusht_vision_100ep_200.ckpt")
    
    # Run episode
    steps = inference.run_episode(max_steps=200, render=True, use_example_action=True)
    print(f"Episode completed in {steps} steps")

if __name__ == "__main__":
    main()