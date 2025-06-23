
from ast import main
import h5py
import json
import numpy as np
import logging as log
import glob
import os
import shutil
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
def split_demos(dataset_file, output_dir, custom_name="demo"):
    """
    将一个 HDF5 文件中的所有 demonstrations 拆分为单独的 HDF5 文件，
    并自动重命名输出目录为：
    自定义名字_原始文件大小GB_文件数量_所有obs/camera_head视频时长总和秒
    """
    original_file_size = os.path.getsize(dataset_file)
    original_file_size_gb = f"{original_file_size/1024/1024/1024:.2f}GB"

    temp_dir = output_dir + "_temp"
    os.makedirs(temp_dir, exist_ok=True)

    with h5py.File(dataset_file, "r") as f:
        demos = sorted(list(f["data"].keys()))

        if len(demos) == 0:
            print(f"Dataset {dataset_file} has no demonstrations.")
            return

        for i, demo in enumerate(demos):
            output_file = os.path.join(temp_dir, f"{demo}.hdf5")
            
            with h5py.File(output_file, "w") as out_f:
                out_data = out_f.create_group("data")
                demo_group = f["data"][demo]
                
                def copy_all(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        out_data.create_dataset(name, data=obj[()])
                    else:
                        print(f"Skipping non-dataset key: {name}")
                
                demo_group.visititems(copy_all)
                for attr_key in demo_group.attrs.keys():
                    out_data.attrs[attr_key] = demo_group.attrs[attr_key]

            print(f"Saved demonstration {demo} to {output_file}")

    print(f"All demonstrations from {dataset_file} have been saved to temporary directory {temp_dir}")

    total_files = len(demos)
    total_video_duration_s = 0.0  # 改为累计视频时长(秒)
   
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            if filename.endswith(".hdf5"):
                filepath = os.path.join(temp_dir, filename)
                with h5py.File(filepath, "r") as f:
                    if "data" in f and "obs" in f["data"] and "camera_head" in f["data"]["obs"]:
                        dataset = f["data"]["obs"]["camera_head"]
                        # 计算视频时长(秒) = 帧数 / 帧率(假设30fps)
                        fps = 30.0  # 默认帧率
                        video_duration_s = dataset.shape[0] / fps
                        total_video_duration_s += video_duration_s  # 累计时长(秒)

    timestamp_str = "_".join(os.path.basename(dataset_file).split("_")[-2:])
    # 修改目录名格式: 将camera_head_size_gb改为video_duration_s
    final_dir_name = f"{custom_name}_{original_file_size_gb}_{total_files}_{total_video_duration_s:.2f}s"
    final_dir = os.path.join(os.path.dirname(output_dir), final_dir_name)
    counter = 1
    while os.path.exists(final_dir):
        final_dir = os.path.join(os.path.dirname(output_dir), f"{final_dir_name}_{counter}")
        counter += 1
    shutil.move(temp_dir, final_dir)
    print(f"Renamed directory to: {final_dir}")

def merge_splits(base_dir, custom_name="demo"):
    """
    合并相同custom_name的所有拆分结果，生成一个总目录
    格式与之前相同：自定义名字_原始文件大小GB_文件数量_所有obs/camera_head视频时长总和秒
    """
    # 1. 查找所有匹配的拆分目录
    split_dirs = []
    for dirname in os.listdir(base_dir):
        if dirname.startswith(custom_name + "_") and not dirname.endswith("_merged_"):
            split_dirs.append(os.path.join(base_dir, dirname))

    if not split_dirs:
        print(f"No split directories found for {custom_name}")
        return

    # 2. 解析每个目录的元数据并累计
    accumulated_size_gb = 0.0  # 仍然累计原始文件大小GB
    accumulated_files = 0
    accumulated_video_duration_s = 0.0  # 改为累计视频时长(秒)

    for split_dir in split_dirs:
        # 首先尝试从metadata.json读取数据
        metadata_file = os.path.join(split_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                accumulated_size_gb += metadata["original_file_size_gb"]
                accumulated_files += metadata["total_files"]
                # 注意: 原始metadata中没有video_duration_s，需要从目录名解析
        else:
            # 从目录名解析数据
            dir_basename = os.path.basename(split_dir)
            parts = [p for p in dir_basename.split("_") if p]
            
            try:
                name_index = parts.index(custom_name)
                if name_index + 2 >= len(parts):
                    print(f"Warning: Directory name {dir_basename} has insufficient parts, skipping")
                    continue
                
                size_part = parts[name_index+1]
                size_gb = float(size_part[:-2])
                
                files_part = parts[name_index+2]
                files = int(files_part)
                
                accumulated_size_gb += size_gb
                accumulated_files += files
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse directory name {dir_basename}, skipping. Error: {str(e)}")
                continue

        # 重新计算video_duration_s(从文件计算，因为metadata中没有)
        if os.path.exists(split_dir):
            for filename in os.listdir(split_dir):
                if filename.endswith(".hdf5"):
                    filepath = os.path.join(split_dir, filename)
                    with h5py.File(filepath, "r") as f:
                        if "data" in f and "obs" in f["data"] and "camera_head" in f["data"]["obs"]:
                            dataset = f["data"]["obs"]["camera_head"]
                            # 计算视频时长(秒) = 帧数 / 帧率(假设30fps)
                            fps = 30.0  # 默认帧率
                            video_duration_s = dataset.shape[0] / fps
                            accumulated_video_duration_s += video_duration_s  # 累计时长(秒)

    # 3. 创建合并目录
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 修改目录名格式: 将camera_head_size_gb改为video_duration_s
    merged_dir_name = f"{custom_name}_{accumulated_size_gb:.2f}GB_{accumulated_files}_files_{accumulated_video_duration_s:.2f}s"
    merged_dir = os.path.join(base_dir, merged_dir_name)
    os.makedirs(merged_dir, exist_ok=True)

    # 4. 合并所有文件 - 解决同名demo文件冲突
    demo_file_counter = {}  # 记录每个demo文件名出现的次数

    for split_dir in split_dirs:
        for filename in os.listdir(split_dir):
            src = os.path.join(split_dir, filename)
            
            if os.path.isdir(src):
                # 直接复制整个目录
                dst = os.path.join(merged_dir, filename)
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                # 如果是文件
                if filename.endswith(".hdf5"):
                    # 处理可能的同名demo文件
                    if filename in demo_file_counter:
                        # 文件名已存在，添加计数器后缀
                        base_name = filename[:-5]
                        ext = ".hdf5"
                        new_filename = f"{base_name}_{demo_file_counter[filename]}{ext}"
                        demo_file_counter[filename] += 1
                        dst = os.path.join(merged_dir, new_filename)
                    else:
                        # 第一次遇到这个文件名
                        demo_file_counter[filename] = 1
                        dst = os.path.join(merged_dir, filename)
                    
                    # 复制文件
                    shutil.copy2(src, dst)
                else:
                    # 其他文件直接复制
                    dst = os.path.join(merged_dir, filename)
                    shutil.copy2(src, dst)

    # 5. 在合并目录中创建metadata.json
    metadata = {
        "original_file_size_gb": round(accumulated_size_gb, 2),
        "total_files": accumulated_files,
        "total_video_duration_s": round(accumulated_video_duration_s, 2),  # 修改为视频时长(秒)
        "timestamp": timestamp_str
    }
    with open(os.path.join(merged_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Merged {len(split_dirs)} splits into: {merged_dir}")
    
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
    parser_split.add_argument("--name", default="demo", help="Custom name prefix for output")

    # === merge command ===
    parser_merge = subparsers.add_parser("merge", help="Merge split directories")
    parser_merge.add_argument("output_dir", help="Base output directory containing splits")
    parser_merge.add_argument("--name", default="demo", help="Custom name prefix of splits to merge")
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
        split_demos(input_file, output_dir, args.name)

    elif args.command == "merge":
        merge_splits(args.output_dir, args.name)