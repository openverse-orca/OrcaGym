"""
Helper script to process datasets. 
Can be used to check if a dataset is valid, combine multiple datasets into one, or merge multiple datasets into one.
"""

import os
import sys
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

if project_root not in sys.path:
    sys.path.append(project_root)


import h5py
import json
import argparse
import numpy as np
import logging as log
import glob
from datetime import datetime
from orca_gym.robomimic.dataset_util import DatasetReader, DatasetWriter
from orca_gym.utils.dir_utils import create_tmp_dir

def _check_dataset(dataset_file, verbose=False) -> bool:

    # extract demonstration list from file
    all_filter_keys = None
    f = h5py.File(dataset_file, "r")

    # use all demonstrations
    demos = sorted(list(f["data"].keys()))
    
    if len(demos) == 0:
        print("")
        print("Dataset {} has no demonstrations.".format(dataset_file))
        return True

    # extract filter key information
    if "mask" in f:
        all_filter_keys = {}
        for fk in f["mask"]:
            fk_demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])])
            all_filter_keys[fk] = fk_demos

    # put demonstration list in increasing episode order
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # extract length of each trajectory in the file
    traj_lengths = []
    action_min = np.inf
    action_max = -np.inf
    for ep in demos:
        traj_lengths.append(f["data/{}/actions".format(ep)].shape[0])
        action_min = min(action_min, np.min(f["data/{}/actions".format(ep)][()]))
        action_max = max(action_max, np.max(f["data/{}/actions".format(ep)][()]))
    traj_lengths = np.array(traj_lengths)

    # report statistics on the data
    print("")
    print("total transitions: {}".format(np.sum(traj_lengths)))
    print("total trajectories: {}".format(traj_lengths.shape[0]))
    print("traj length mean: {}".format(np.mean(traj_lengths)))
    print("traj length std: {}".format(np.std(traj_lengths)))
    print("traj length min: {}".format(np.min(traj_lengths)))
    print("traj length max: {}".format(np.max(traj_lengths)))
    print("action min: {}".format(action_min))
    print("action max: {}".format(action_max))
    print("")
    print("==== Filter Keys ====")
    if all_filter_keys is not None:
        for fk in all_filter_keys:
            print("filter key {} with {} demos".format(fk, len(all_filter_keys[fk])))
    else:
        print("no filter keys")
    print("")
    if verbose:
        if all_filter_keys is not None:
            print("==== Filter Key Contents ====")
            for fk in all_filter_keys:
                print("filter_key {} with {} demos: {}".format(fk, len(all_filter_keys[fk]), all_filter_keys[fk]))
        print("")
    env_meta = json.loads(f["data"].attrs["env_args"])
    print("==== Env Meta ====")
    print(json.dumps(env_meta, indent=4))
    print("")

    print("==== Dataset Structure ====")
    for ep in demos:
        print("episode {} with {} transitions".format(ep, f["data/{}".format(ep)].attrs["num_samples"]))
        for k in f["data/{}".format(ep)]:
            if k in ["obs", "next_obs"]:
                print("    key: {}".format(k))
                for obs_k in f["data/{}/{}".format(ep, k)]:
                    shape = f["data/{}/{}/{}".format(ep, k, obs_k)].shape
                    print("        observation key {} with shape {}".format(obs_k, shape))
            elif isinstance(f["data/{}/{}".format(ep, k)], h5py.Dataset):
                key_shape = f["data/{}/{}".format(ep, k)].shape
                print("    key: {} with shape {}".format(k, key_shape))

        if not verbose:
            break

    f.close()

    # maybe display error message
    print("")
    if (action_min < -1.) or (action_max > 1.):
        log.warning("Dataset should have actions in [-1., 1.] but got bounds [{}, {}]".format(action_min, action_max))
        return False
    else:
        print("Checking dataset {} complete. Dataset is valid.".format(dataset_file))
        
    return True

def _process_check(dataset_files):
    check_successed = True
    checked_files_count = 0
    for dataset in dataset_files:
        if _check_dataset(dataset, verbose=verbose) == False:
            print("Dataset {} is invalid.".format(dataset))
            check_successed = False
            break
        checked_files_count += 1
        
    if check_successed:
        print("")
        print("All datasets are valid. Checked {} files.".format(checked_files_count))

def _process_combine(dataset_files, output_file):
    # combine datasets
    writer = None
    for dataset in dataset_files:
        reader = DatasetReader(dataset)
        if writer is None:
            env_name = reader.get_env_name()
            env_version = reader.get_env_version()
            env_kwargs = reader.get_env_kwargs()
            writer = DatasetWriter(output_file, env_name, env_version, env_kwargs)
            
        demo_names = reader.get_demo_names()
        for demo_name in demo_names:
            demo_data = reader.get_demo_data(demo_name)
            writer.add_demo_data(demo_data)
    
    writer.shuffle_demos()
    writer.finalize()
    print("Combined datasets written to {}".format(output_file))

def _glob_dataset_filenames(dataset_files):
    matched_files = []
    for dataset_file in dataset_files:
        matched_files.extend(glob.glob(dataset_file))
    return matched_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', help="path to hdf5 dataset",)
    parser.add_argument("--task", type=str, help="task to run (check / combine / merge)", default="check")
    parser.add_argument("--verbose", type=bool, help="output more details", default=False)
    parser.add_argument("--filter_key", type=str, help="filter key to use for processing", default=None)
    parser.add_argument("--output", type=str, help="output file for combined dataset", default=None)
    
    args = parser.parse_args()
    dataset_files = args.datasets
    task = args.task
    verbose = args.verbose
    filter_key = args.filter_key
    output_file = args.output
    
    dataset_files = _glob_dataset_filenames(dataset_files)
    create_tmp_dir("processed_datasets_tmp")
    
    if task == "check":
        _process_check(dataset_files)
    elif task == "combine":
        if output_file is None:
            formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = f"./processed_datasets_tmp/combined_{formatted_time}.hdf5"            
        _process_combine(dataset_files, output_file)
    else:
        print("Process not implemented: {}".format(task))
