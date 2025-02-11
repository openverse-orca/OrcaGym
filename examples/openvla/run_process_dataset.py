"""
Helper script to process datasets. 
Can be used to check if a dataset is valid, combine multiple datasets into one, or merge multiple datasets into one.
"""

import os
import sys
import time

# current_file_path = os.path.abspath('')
# project_root = os.path.dirname(os.path.dirname(current_file_path))

# if project_root not in sys.path:
#     sys.path.append(project_root)

import argparse
import orca_gym.scripts.process_dataset as process_dataset
from orca_gym.utils.dir_utils import create_tmp_dir
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', help="path to hdf5 dataset",)
    parser.add_argument("--proc", type=str, help="proc to run (check / combine / merge)", default="check")
    parser.add_argument("--verbose", type=bool, help="output more details", default=False)
    parser.add_argument("--filter_key", type=str, help="filter key to use for processing", default=None)
    parser.add_argument("--output", type=str, help="output file for combined dataset", default=None)
    
    args = parser.parse_args()
    dataset_files = args.datasets
    proc = args.proc
    verbose = args.verbose
    filter_key = args.filter_key
    output_file = args.output
    
    dataset_files = process_dataset.glob_dataset_filenames(dataset_files)
    create_tmp_dir("processed_datasets_tmp")
    
    if proc == "check":
        process_dataset.process_check(dataset_files, verbose=verbose)
    elif proc == "combine":
        if output_file is None:
            formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = f"./processed_datasets_tmp/combined_{process_dataset.get_dataset_prefix(dataset_files[0])}_{formatted_time}.hdf5"            
        process_dataset.process_combine(dataset_files, output_file)
    else:
        print("Process not implemented: {}".format(proc))
