import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

# Load the action data
actions = pd.read_csv('openloong_dp_action.csv').values


# load example hdf5
path = "/home/user/Desktop/manipulation/OrcaGym/examples/openpi/records_tmp/shop/0b0b05ac_9676d71b/proprio_stats/proprio_stats.hdf5"
with h5py.File(path, 'r') as f:
    data = f["data"]["demo_00000"]["actions"][:]




def plot_actions(idx):
    action_example = [i[idx] for i in data]

    # draw the traj of the first column
    plt.plot(actions[:, idx])
    plt.plot(action_example)
    plt.show()
    
def plot_loss():
    path = "/home/user/Desktop/manipulation/OrcaGym/examples/diffusion_policy/loss_history.npy"
    loss = np.load(path)
    plt.plot(loss)
    plt.show()
    
def plot_training_data(index):
    from openloong_dp_dataset import OpenLoongDPDataset
    dataset = OpenLoongDPDataset(root_dir="/home/user/Desktop/manipulation/OrcaGym/examples/openpi/records_tmp/shop")
    action_of_index = []
    for element in list(dataset.data.keys()):
        a = dataset.data[element]["actions"]
        action_of_index += [i[index] for i in a]
    plt.plot(action_of_index)
    plt.show()
    
    
if __name__ == "__main__":
    plot_loss()
    # plot_actions(17)
    # plot_training_data(17)