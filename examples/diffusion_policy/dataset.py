#@markdown ### **Dataset Demo**

#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTImageDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data ((image, agent_pos), action) from a zarr storage
#@markdown - Normalizes each dimension of agent_pos and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)

import numpy as np
import torch
import zarr
import os
import gdown

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root['data']['img'][:]
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # (N,3,96,96)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:,:2],
            'action': dataset_root['data']['action'][:]
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]

        return nsample


# # download demonstration data from Google Drive
# dataset_path = "pusht_cchi_v7_replay.zarr.zip"
# if not os.path.isfile(dataset_path):
#     id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
#     gdown.download(id=id, output=dataset_path, quiet=False)

# # parameters
# pred_horizon = 16
# obs_horizon = 2
# action_horizon = 8
# #|o|o|                             observations: 2
# #| |a|a|a|a|a|a|a|a|               actions executed: 8
# #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# # create dataset from file
# dataset = PushTImageDataset(
#     dataset_path=dataset_path,
#     pred_horizon=pred_horizon,
#     obs_horizon=obs_horizon,
#     action_horizon=action_horizon
# )
# # save training data statistics (min, max) for each dim
# stats = dataset.stats

# # create dataloader
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=64,
#     num_workers=4,
#     shuffle=True,
#     # accelerate cpu-gpu transfer
#     pin_memory=True,
#     # don't kill worker process afte each epoch
#     persistent_workers=True
# )

# visualize data in batch
if __name__ == "__main__":
    # batch = next(iter(dataloader))
    # print("batch['image'].shape:", batch['image'].shape)
    # print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
    # print("batch['action'].shape", batch['action'].shape)
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    dataset_root = zarr.open(dataset_path, 'r')
    
    # print the shape of the dataset
    print(dataset_root["data"]["action"].shape)
    print(dataset_root["data"]["state"].shape)
    print(dataset_root["data"]["img"].shape)
    print(dataset_root["meta"]["episode_ends"].shape)

    
    from pusht_env import PushTEnv
    import time

    start_idx = 0
    for episode_count in dataset_root["meta"]["episode_ends"][:]:
        print("resetting to state", start_idx)
        i = 0
        env = PushTEnv(reset_to_state=dataset_root["data"]["state"][start_idx])
        env.seed(1000)
        obs, info = env.reset()
        while True:
            action = dataset_root["data"]["action"][start_idx+i]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
            env.render(mode="human")
            time.sleep(0.1)
            i += 1
            if i == episode_count-start_idx:
                start_idx = episode_count
                break
            