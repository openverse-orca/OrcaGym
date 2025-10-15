import matplotlib.pyplot as plt
# read from hdf5 file
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def get_single_demo(path: str) -> None:
    with h5py.File(path, "r") as f:
        data = f["data"]["demo_00000"]["actions"]
        # turn to pandas dataframe
        df = pd.DataFrame(data)
    return df

def get_all_demo(dir_path: str) -> None:
    df_list = []
    for demo_name in tqdm(os.listdir(dir_path)):
        file = os.path.join(dir_path, demo_name, "proprio_stats", "proprio_stats.hdf5")
        if os.path.exists(file):
            df = get_single_demo(file)
            df_list.append(df)
        else:
            print(f"File {file} does not exist")
    # concat all df
    df = pd.concat(df_list)
    # draw distribution of each column
    for col in df.columns[6:14]:
        
        plt.hist(df[col], bins=100)
        plt.savefig(f"logs/distribution_{col}.png")
        plt.clf()
    for col in df.columns[20:28]:
        plt.hist(df[col], bins=100)
        plt.savefig(f"logs/distribution_{col}.png")
        plt.clf()
    return df

if __name__ == "__main__":
    df_list = get_all_demo("/media/user/A7EC-9D11/shopscene_1f15/Shop-79p12GB_4294counts_88p59h/Shelf_Operation-79p12GB_4294counts_88p59h/pick_and_place-79p12GB_4294counts_88p59h")
