import argparse
import json
import h5py
import imageio
import numpy as np
import os
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

import urllib.request
import time


def download_dataset():
    # Get pretrained checkpooint from the model zoo

    ckpt_path = "lift_ph_low_dim_epoch_1000_succ_100.pth"
    # Lift (Proficient Human)
    # urllib.request.urlretrieve(
    #     "http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/lift/bc_rnn/lift_ph_low_dim_epoch_1000_succ_100.pth",
    #     filename=ckpt_path
    # )

    assert os.path.exists(ckpt_path)

    return ckpt_path

def create_env(ckpt_path):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        render=True, # we won't do on-screen rendering in the notebook
        render_offscreen=False, # render to RGB images for video
        verbose=True,
    )

    return env, policy

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, camera_names=None, realtime_step=0.0):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.
    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
    """
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    # assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    action_step = env.env.unwrapped.get_action_step()
    print("env sample range: ", env.env.unwrapped._sample_range, "action step: ", action_step)
    
    # state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    # obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    try:
        for step_i in range(horizon):
            if realtime_step > 0.0:
                start_time = time.time()

            # get action from policy
            act = policy(ob=obs)

            # play action
            for _ in range(action_step):
                next_obs, r, done, _ = env.step(act)
                if render:
                    env.render(mode="human", camera_name=camera_names[0])

                # sleep to maintain real-time speed
                if realtime_step > 0.0:
                    elapsed_time = time.time() - start_time
                    if elapsed_time < realtime_step:
                        time.sleep(realtime_step - elapsed_time)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            # state_dict = env.get_state()
            


    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    return stats

if __name__ == "__main__":
    rollout_horizon = 300
    np.random.seed(0)
    torch.manual_seed(0)
    video_path = "rollout.mp4"
    video_writer = imageio.get_writer(video_path, fps=20)

    ckpt_path = download_dataset()
    env, policy = create_env(ckpt_path)

    for i in range(5):
        stats = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=True, 
            video_writer=video_writer, 
            video_skip=5, 
            camera_names=["agentview"]
        )
        print(stats)


    video_writer.close()

    # from IPython.display import Video
    # Video(video_path)