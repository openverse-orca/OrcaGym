#@markdown ### **Vision Encoder**
#@markdown
#@markdown Defines helper functions:
#@markdown - `get_resnet` to initialize standard ResNet vision encoder
#@markdown - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm

import torch
import torch.nn as nn
import torchvision
from typing import Callable
from dataset import *
from model_utils import *
from tqdm import tqdm
import numpy as np
import gdown
import os

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from openloong_dp_dataset import OpenLoongDPDataset


# parameters
obs_horizon = 1
action_horizon = 32
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = OpenLoongDPDataset(
    root_dir="/home/user/Desktop/manipulation/OrcaGym/examples/openpi/records_tmp/shop",
    action_horizon=action_horizon
)



# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)





#@markdown ### **Network Demo**

# construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
vision_encoder = get_resnet('resnet18')

# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder = replace_bn_with_gn(vision_encoder)

# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 42
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 28



# fully connected network

fcn = nn.Sequential(
    nn.Linear(obs_dim, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, action_dim * action_horizon),
)
nets = nn.ModuleDict({
    'fcn': fcn,
    'vision_encoder': vision_encoder
})




# device transfer
device = torch.device('cuda')
_ = nets.to(device)


#@markdown ### **Training**
#@markdown
#@markdown Takes about 2.5 hours. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights
if __name__ == "__main__":
    num_epochs = 250

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)



    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        loss_history = list()
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['camera_frames_head'][:,:obs_horizon].to(device)

                    nagent_pos = nbatch['states'][:,:obs_horizon].to(device)
                    naction = nbatch['actions'].to(device)
                    B = nagent_pos.shape[0]
                    

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2],-1)
                    # (B,obs_horizon,D)
                    # save image
                    # import cv2
                    # print(nimage)
                    # print(nimage.shape)
                    # print(nimage.flatten(end_dim=1).shape)
                    # print(image_features.shape)
                    # cv2.imwrite("img_train.png", nimage[0][0].cpu().numpy().transpose(1, 2, 0))
                    # exit()

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    actions = nets['fcn'](obs_cond)
                    actions = actions.reshape(B, action_horizon, action_dim)

                    # L2 loss
                    loss = nn.functional.mse_loss(actions, naction)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            loss_history.append(np.mean(epoch_loss))
            # Save to file every 10 epochs
            if (epoch_idx + 1) % 10 == 0:
                np.save('loss_history.npy', loss_history)

    
    # save the model
    torch.save(nets.state_dict(), "pusht_vision_100ep_fcn.ckpt")

