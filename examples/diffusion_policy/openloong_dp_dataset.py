import torch
from torch.utils.data import Dataset
import h5py
import os
import cv2
from tqdm import tqdm
import numpy as np
import pickle

def read_mp4_file(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # change X, X, 3 to 3, X, X
        frame = frame.transpose(2, 0, 1)
        frames.append(frame)
    return frames

def normalize_data(data, min, max):
    return (data - min) / (max - min + 1e-8)

def unnormalize_data(data, min, max):
    return (data * (max - min + 1e-8)) + min
    
# give me a dataset class that reads from /home/user/Desktop/manipulation/OrcaGym/examples/openpi/records_tmp/shop
class OpenLoongDPDataset(Dataset):
    def __init__(self, root_dir, action_horizon=16):
        self.root_dir = root_dir
        self.data = {}
        folders = os.listdir(self.root_dir)
        for folder in tqdm(folders):
            try:
                if not os.path.isdir(os.path.join(self.root_dir, folder)):
                    continue
                element = {}
                # load states and actions
                with h5py.File(os.path.join(self.root_dir, folder, 'proprio_stats', 'proprio_stats.hdf5'), 'r') as f:
                    s = f['data']['demo_00000']['states'][:]
                    obs = f['data']['demo_00000']['obs']
                    # element['states'] = np.concatenate([s[:, 14:21], s[:, 82:83], s[:, 21:28], s[:, 83:84], s[:, 56:59], s[:, 59:62]], dtype=np.float32, axis=1)
                    element['states'] = np.concatenate([
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
                    element['actions'] = f['data']['demo_00000']['actions'][:]
                    element["camera_frames_index"] = f["data"]["demo_00000"]["camera_frames"][:]
                    if len(element["camera_frames_index"]) < len(element["actions"]):
                        # extend with last value
                        element["camera_frames_index"] = np.concatenate([element["camera_frames_index"], [element["camera_frames_index"][-1]] * (len(element["actions"]) - len(element["camera_frames_index"]))])
                    elif len(element["camera_frames_index"]) > len(element["actions"]):
                        # truncate
                        element["camera_frames_index"] = element["camera_frames_index"][:len(element["actions"])]
                    assert len(element["states"]) == len(element["actions"]) == len(element["camera_frames_index"])
                    
                    # load mp4 videos
                    element["camera_frames_head"] = read_mp4_file(os.path.join(self.root_dir, folder, 'camera', 'video', 'camera_head_color.mp4'))
                    element["camera_frames_wrist_l"] = read_mp4_file(os.path.join(self.root_dir, folder, 'camera', 'video', 'camera_wrist_l_color.mp4'))
                    element["camera_frames_wrist_r"] = read_mp4_file(os.path.join(self.root_dir, folder, 'camera', 'video', 'camera_wrist_r_color.mp4'))
                    self.data[folder] = element
            except:
                print(f"error loading folder: {folder}")
                continue
        
        self.indices = self.create_indices(action_horizon)
        self.get_stats()
        
                
    def create_indices(self, action_horizon):
        indices = []
        for name, element in self.data.items():
                
            assert len(element["states"]) == len(element["actions"]) == len(element["camera_frames_index"])

            start_idx = 0
            end_idx = start_idx + action_horizon
            while True:
                if end_idx > len(element["states"]):
                    break
                indices.append((name, start_idx, end_idx))
                start_idx += 1
                end_idx += 1
        return indices
    
    
    def get_stats(self):
        # get min and max of state and action
        state_min = None
        state_max = None
        action_min = None
        action_max = None
        for name, element in self.data.items():
            states = element["states"]
            actions = element["actions"]
            if state_min is None:
                state_min = states.min(axis=0)
            if state_max is None:
                state_max = states.max(axis=0)
            if action_min is None:
                action_min = actions.min(axis=0)
            if action_max is None:
                action_max = actions.max(axis=0)

            state_min = np.min(np.stack([state_min, states.min(axis=0)]), axis=0)
            state_max = np.max(np.stack([state_max, states.max(axis=0)]), axis=0)
            action_min = np.min(np.stack([action_min, actions.min(axis=0)]), axis=0)
            action_max = np.max(np.stack([action_max, actions.max(axis=0)]), axis=0)
        self.state_min = state_min
        self.state_max = state_max
        self.action_min = action_min
        self.action_max = action_max
        
        # save stats to pickle
        # with open('openloong_dp_stats.pkl', 'wb') as f:
        #     pickle.dump({'state': {"min": state_min, "max": state_max}, 'action': {"min": action_min, "max": action_max}}, f)
        # print(state_min, state_max, action_min, action_max)
        # print(state_max-state_min, action_max-action_min)
        # exit()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        name = idx[0]
        start_idx = idx[1]
        end_idx = idx[2]
        element = self.data[name]
        
        states = element["states"][start_idx:end_idx]
        actions = element["actions"][start_idx:end_idx]
        camera_frames_index = element["camera_frames_index"][start_idx:end_idx]

        # return {
        #     "states": normalize_data(torch.from_numpy(states[0]).unsqueeze(0), self.state_min, self.state_max),
        #     "actions": normalize_data(torch.from_numpy(actions), self.action_min, self.action_max),
        #     "camera_frames_head": torch.from_numpy(element["camera_frames_head"][camera_frames_index[0] if camera_frames_index[0] < len(element["camera_frames_head"]) else -1].astype(np.float32)).unsqueeze(0),
        #     # "camera_frames_wrist_l": camera_frames_wrist_l,
        #     # "camera_frames_wrist_r": camera_frames_wrist_r
        # }
        frames = element["camera_frames_head"]
        frame_list = []
        for i in camera_frames_index:
            if i < len(frames):
                frame_list.append(frames[i])
            else:
                frame_list.append(frames[-1])
                # print(f"index {i} is out of range, name: {name}, start_idx: {start_idx}, end_idx: {end_idx}, length: {len(frames)}, frames: {camera_frames_index}")
                # exit()
        frame_list = np.stack(frame_list, axis=0)
        # exit()
        
        return {
            "states": normalize_data(torch.from_numpy(states), self.state_min, self.state_max),
            "actions": normalize_data(torch.from_numpy(actions), self.action_min, self.action_max),
            "camera_frames_head": torch.from_numpy(frame_list.astype(np.float32)),
            # "camera_frames_wrist_l": camera_frames_wrist_l,
            # "camera_frames_wrist_r": camera_frames_wrist_r
        }


if __name__ == "__main__":
    dataset = OpenLoongDPDataset("/home/user/Desktop/manipulation/OrcaGym/examples/openpi/records_tmp/shop")
    print(len(dataset))
    print(dataset[0])
    for k,v in dataset[0].items():
        print(k, v.shape)
    exit()
        
    
    