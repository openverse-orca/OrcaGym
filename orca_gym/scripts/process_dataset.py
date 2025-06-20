
from ast import main
import h5py
import json
import numpy as np
import logging as log
import glob
import os
from datetime import datetime

from orca_gym.robomimic.dataset_util import DatasetReader, DatasetWriter


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

def process_check(dataset_files, verbose=False):
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

def _check_combination(dataset_files):
    env_names = []
    env_versions = []
    env_kwargs = []
    for dataset in dataset_files:
        reader = DatasetReader(dataset)
        env_names.append(reader.get_env_name())
        env_versions.append(reader.get_env_version())
        env_kwargs.append(reader.get_env_kwargs())
        
    if len(set(env_names)) > 1:
        print("Different env names in datasets: {}".format(set(env_names)))
        return False
    if len(set(env_versions)) > 1:
        print("Different env versions in datasets: {}".format(set(env_versions)))
        return False
    if not all(env_kwargs_item == env_kwargs[0] for env_kwargs_item in env_kwargs):
        print("Different env kwargs in datasets!")
        return False
    
    return True

def get_dataset_prefix(dataset_file):
    reader = DatasetReader(dataset_file)
    env_name = reader.get_env_name()
    env_name = env_name.split("-OrcaGym-")[0]
    env_version = reader.get_env_version()
    env_task = reader.get_env_kwargs()["task"]
    prefix = f"{env_name}_{env_version}_{env_task}"
    prefix = prefix.replace(" ", "_")
    prefix = prefix.replace(".", "_")
    return prefix

def process_combine(dataset_files, output_file):
    if not _check_combination(dataset_files):
        print("Datasets are not compatible for combination. Cannot combine datasets.")
        exit(1)
        
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

def glob_dataset_filenames(dataset_files):
    matched_files = []
    for dataset_file in dataset_files:
        matched_files.extend(glob.glob(dataset_file))
    return matched_files

def process_update_kwargs(dataset_files, kwargs):
    for dataset in dataset_files:
        reader = DatasetReader(dataset)
        env_name = reader.get_env_name()
        env_version = reader.get_env_version()
        env_kwargs = reader.get_env_kwargs()
        
        for key in kwargs:
            if key not in env_kwargs:
                print(f"Key {key} not in env_kwargs. Skip updating.")
                continue
            env_kwargs[key] = kwargs[key]
            print(f"Update key {key} to {kwargs[key]}")
        
        
        writer = DatasetWriter(dataset, env_name, env_version, env_kwargs)
        writer.set_env_kwargs(env_kwargs)
        writer.finalize()

def split_demos(dataset_file, output_dir):
    """
    将一个 HDF5 文件中的所有 demonstrations 拆分为单独的 HDF5 文件，
    并使用自定义文件名格式：demo_<时间>。

    Args:
        dataset_file (str): 输入的 HDF5 文件路径。
        output_dir (str): 输出目录，用于保存拆分后的单个 demonstration 文件。
    """

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(dataset_file, "r") as f:

        demos = sorted(list(f["data"].keys()))

        if len(demos) == 0:
            print(f"Dataset {dataset_file} has no demonstrations.")
            return
        file_name = os.path.basename(dataset_file)
        file_name_without_extension = os.path.splitext(file_name)[0]
        try:
            timestamp_str = file_name_without_extension.split("_")[-2] + "_" + file_name_without_extension.split("_")[-1]
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        except Exception as e:
            print(f"Warning: Could not extract timestamp from file name '{file_name}'. Using current time instead.")
            timestamp = datetime.now()

        timestamp_str_for_filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        for demo in demos:
            output_file = os.path.join(output_dir, f"demo_{timestamp_str_for_filename}.hdf5")
            output_file = os.path.join(output_dir, f"demo_{timestamp_str_for_filename}_{demos.index(demo)}.hdf5")

            with h5py.File(output_file, "w") as out_f:
                out_data = out_f.create_group("data")
                demo_group = f["data"][demo]
                for key in demo_group.keys():
                    if isinstance(demo_group[key], h5py.Dataset):
                        out_data.create_dataset(key, data=demo_group[key][()])
                    else:
                        print(f"Skipping non-dataset key: {key}")

                for attr_key in demo_group.attrs.keys():
                    out_data.attrs[attr_key] = demo_group.attrs[attr_key]

            print(f"Saved demonstration {demo} to {output_file}")

    print(f"All demonstrations from {dataset_file} have been saved to {output_dir}.")

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset processing utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === check command ===
    parser_check = subparsers.add_parser("check", help="Check dataset validity")
    parser_check.add_argument("datasets", nargs="+", help="Dataset file(s) to check")
    parser_check.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # === combine command ===
    parser_combine = subparsers.add_parser("combine", help="Combine multiple datasets")
    parser_combine.add_argument("datasets", nargs="+", help="Dataset files to combine")
    parser_combine.add_argument("-o", "--output", required=True, help="Output file for combined dataset")

    # === update-kwargs command ===
    parser_update = subparsers.add_parser("update-kwargs", help="Update environment kwargs")
    parser_update.add_argument("datasets", nargs="+", help="Dataset files to update")
    parser_update.add_argument("--set", nargs=2, action="append", metavar=("KEY", "VALUE"),
                               help="Set key to value in env_kwargs")
    
    # === split command ===
    parser_split = subparsers.add_parser("split", help="Split a dataset into individual demonstrations")
    parser_split.add_argument("dataset", help="Dataset file to split (relative to the base directory)")
    parser_split.add_argument("-o", "--output_dir", required=True, help="Output directory name (relative to the base directory)")

    args = parser.parse_args()

    # 定义输入和输出的基目录
    BASE_INPUT_DIR = "/home/orcatest/Orcagym_kps/OrcaGym/examples/openpi/records_tmp/shop/"
    BASE_OUTPUT_DIR = "/home/orcatest/Orcagym_kps/OrcaGym/examples/openpi/records_tmp/shop/"

    if args.command == "check":
        files = glob_dataset_filenames(args.datasets)
        process_check(files, verbose=args.verbose)

    elif args.command == "combine":
        files = glob_dataset_filenames(args.datasets)
        process_combine(files, output_file=args.output)

    elif args.command == "update-kwargs":
        files = glob_dataset_filenames(args.datasets)
        if args.set:
            kwargs = {k: json.loads(v) for k, v in args.set}
            process_update_kwargs(files, kwargs)
        else:
            print("No key-value pairs provided for update.")
    elif args.command == "split":
            input_file = os.path.join(BASE_INPUT_DIR, args.dataset)
            output_dir = os.path.join(BASE_OUTPUT_DIR, args.output_dir)
            split_demos(input_file, output_dir)
