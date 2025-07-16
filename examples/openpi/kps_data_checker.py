import argparse
import json
import os, sys, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymediainfo import MediaInfo
from enum import Enum
from pathlib import Path

class ErrorType(Enum):
    Qualified = 0
    MP4FPSError = 1
    MP4DurationError = 2
    MP4TrackError = 3
    MP4NotExistError = 4
    ParametersError = 5
    ProprioStatsError = 6

class BasicUnitChecker:

    def __init__(self, basic_unit_path: str, camera_name_list: [], proprio_stats: str):
        self.basic_unit_path = basic_unit_path
        self.camera_name_list = camera_name_list
        self.proprio_stats = proprio_stats
        self.duration = 0.0

    def camera_checker(self):
        for camera_name in self.camera_name_list:
            mp4_video_filepath = os.path.join(self.basic_unit_path, "camera", "video", f"{camera_name}_color.mp4")
            mp4_depth_filepath = os.path.join(self.basic_unit_path, "camera", "depth", f"{camera_name}_depth.mp4")
            if not os.path.exists(mp4_video_filepath) or not os.path.exists(mp4_depth_filepath):
                return ErrorType.MP4NotExistError

            ret = self.mp4_metadata_checker(mp4_video_filepath)
            if ret is not ErrorType.Qualified:
                return ret

            ret = self.mp4_metadata_checker(mp4_depth_filepath)
            if ret is not ErrorType.Qualified:
                return ret

        return ErrorType.Qualified

    def mp4_metadata_checker(self, media_path) :
        media = MediaInfo.parse(media_path)
        if not media.tracks:
            return ErrorType.MP4TrackError
        try:
            video_track = media.tracks[1]
        except IndexError:
            print("media file does not have a video track: ", media_path)
            return ErrorType.MP4TrackError

        fps = video_track.frame_rate
        duration = video_track.duration / 1000.0
        if duration < 8 or duration > 30:
           return ErrorType.MP4DurationError
        if float(fps) < 30:
            return ErrorType.MP4FPSError

        self.duration = duration
        return ErrorType.Qualified

    def parameters_checker(self) -> bool:
        for camera_name in self.camera_name_list:
            extrinsic_params_path = os.path.join(self.basic_unit_path, "parameters", f"{camera_name}_extrinsic_params.json")
            extrinsic_params_aligned_path = os.path.join(self.basic_unit_path, "parameters", f"{camera_name}_extrinsic_params_aligned.json")
            intrinsic_params_path = os.path.join(self.basic_unit_path, "parameters", f"{camera_name}_intrinsic_params.json")

            if (not os.path.exists(extrinsic_params_path) or
                not os.path.exists(intrinsic_params_path) or
                not os.path.exists(extrinsic_params_aligned_path)):
                return False
        return True

    def proprio_stats_checker(self) -> bool:
        if not os.path.exists(os.path.join(self.basic_unit_path, "proprio_stats", self.proprio_stats)):
            return False
        return True

    def check(self):
        ret = self.camera_checker()
        if ret is not ErrorType.Qualified:
            return ret, 0.0
        if not self.parameters_checker():
            return ErrorType.ParametersError, 0.0
        if not self.proprio_stats_checker():
            return ErrorType.ProprioStatsError, 0.0
        return ErrorType.Qualified, self.duration

class KPSDataChecker:
    def __init__(self, dataset_path: str, camera_name_list: list, proprio_stats: str):
        self.dataset_path = dataset_path
        self.subdirectories = []
        self.get_subdirectories()

        self.camera_name_list = camera_name_list
        self.duration_total = 0.0
        self.proprio_stats = proprio_stats

        self.mp4_fps_error_path = []
        self.mp4_duration_error_path = []
        self.mp4_tracks_error_path = []
        self.mp4_not_exist_error_path = []
        self.proprio_stats_error_path = []
        self.parameters_error_path = []
        self.qualified_path = []

    def get_subdirectories(self):
        with os.scandir(self.dataset_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    self.subdirectories.append(entry.name)

    def check_subdirectory(self, subdirectory):
        basicChecker =  BasicUnitChecker(os.path.join(dataset_path, subdirectory), self.camera_name_list, self.proprio_stats)
        error_type, duration = basicChecker.check()
        return error_type, duration, subdirectory

    def check(self):
        futures = []
        for subdir in self.subdirectories:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                future = executor.submit(self.check_subdirectory, subdir)
                futures.append(future)

        for future in as_completed(futures):
            error_type, duration ,subdir = future.result()
            if error_type == ErrorType.Qualified:
                self.qualified_path.append(subdir)
                self.duration_total += duration
            elif error_type == ErrorType.MP4FPSError:
                self.mp4_fps_error_path.append(subdir)
            elif error_type == ErrorType.MP4DurationError:
                self.mp4_duration_error_path.append(subdir)
            elif error_type == ErrorType.MP4TrackError:
                self.mp4_tracks_error_path.append(subdir)
            elif error_type == ErrorType.MP4NotExistError:
                self.mp4_not_exist_error_path.append(subdir)
            elif error_type == ErrorType.ParametersError:
                self.parameters_error_path.append(subdir)
            elif error_type == ErrorType.ProprioStatsError:
                self.proprio_stats_error_path.append(subdir)


class KPSDataExport:
    def __init__(self, dataset_path: str, json_file: str):
        self.dataset_path = dataset_path
        self.json_file = json_file
        self.id_to_index = {}
        self.load_json()

    def load_json(self):
        json_path = os.path.join(self.dataset_path, self.json_file)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file {self.json_file} not found in {self.dataset_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.json_data = json.loads(f.read())

        self.id_to_index = {item["episode_id"]: index for index, item in enumerate(self.json_data)}

    def export_reporter(self,
                        subdirectories: list,
                        qualified_path: list,
                        mp4_fps_error_path: list,
                        mp4_duration_error_path: list,
                        mp4_tracks_error_path: list,
                        mp4_not_exist_error_path: list,
                        parameters_error_path: list,
                        proprio_stats_error_path: list,
                        output_file: str = "reporter.md",):

        subdirectories_len = len(subdirectories)
        qualified_path_len = len(qualified_path)
        fps_error_len = len(mp4_fps_error_path)
        duration_error_len = len(mp4_duration_error_path)
        tracks_error_len = len(mp4_tracks_error_path)
        not_exist_error_len = len(mp4_not_exist_error_path)
        parameters_error_len = len(parameters_error_path)
        proprio_stats_error_len = len(proprio_stats_error_path)
        total_error_count = (fps_error_len + duration_error_len + tracks_error_len +
                             not_exist_error_len + parameters_error_len + proprio_stats_error_len)
        reporter_str = (
            "## 数据采集质检报告\n"
            f"**检测项目:{self.json_file}**\n"
            f"**检测员: 系统脚本检测**\n"
            f"**检测方式：全检测**\n"
            f"**采集数据数量: {subdirectories_len}**\n"
            f"合格数据采集数量: {qualified_path_len}\n"
            f"MP4帧率不合格数量: {fps_error_len}\n"
            f"MP4时长不合格数量: {duration_error_len}\n"
            f"MP4轨道不合格数量: {tracks_error_len}\n"
            f"MP4文件不存在数量: {not_exist_error_len}\n"
            f"相机参数文件缺失错误数量: {parameters_error_len}\n"
            f"ProprioStats文件缺失错误数量: {proprio_stats_error_len}\n"
            f"总计不合格数量： {total_error_count}"
        )
        with open(os.path.join(self.dataset_path, output_file), 'w') as f:
            f.write(reporter_str)

    def export_data(self, output_filepath: str,
                    qualified_path: list,
                    unqualified_path: list = [],
                    MP4DurationTime: float = 0.0
                    ):
        self._export_data(output_filepath, qualified_path, "qualified", MP4DurationTime)
        self._export_data(output_filepath, unqualified_path, "unqualified")

    def _export_data(self, output_filepath: str, source_paths: list, type: str, MP4DurationTime: float = 0.0):
        output_type_path = os.path.join(output_filepath, type)
        os.makedirs(output_type_path, exist_ok=True)
        output_json_path = os.path.join(output_type_path, self.json_file)

        output_json_list = self._generate_output_json_list_(source_paths)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json_list, f, ensure_ascii=False, indent=4)

        self._export_mp4_files(output_type_path, source_paths)

        # Rename the output_qualified_path
        total_bytes = 0
        for root, _, files in os.walk(output_type_path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):  # 确保是文件
                    total_bytes += os.path.getsize(fp)

        total_GB = total_bytes / (1024 * 1024 * 1024)  # Size in GB
        counts = len(source_paths)
        duration = MP4DurationTime / (60 * 60)  # Duration in hours

        new_output_filepath = f"{type}_{total_GB:.2f}GB_{counts}counts_{duration:.2f}h".replace(".", "p")
        os.rename(output_type_path, os.path.join(output_filepath, new_output_filepath))

    def _export_mp4_files(self, output_filepath: str, source_paths: list):
        for subdir in source_paths:
            subdir_path = os.path.join(self.dataset_path, subdir)
            if os.path.commonpath([subdir_path, output_filepath]) == os.path.normpath(subdir_path):
                raise ValueError("Output path must be outside the dataset path to avoid overwriting files.")
            shutil.copytree(subdir_path, os.path.join(output_filepath, subdir))

    def _generate_output_json_list_(self, paths: list) -> list:
        output_json_list = []
        for subdir in paths:
            episode_id = subdir.split("_")[0]
            if episode_id in self.id_to_index:
                index = self.id_to_index[episode_id]
                data = self.json_data[index]
                output_json_list.append(data)
        return output_json_list

if __name__ == "__main__":
    # 示例参数 --dataset_path augmented_datasets_tmp/shop --json_file Shop-Cashier_Operation.json --output_filepath /home/orca/dataset
    # 会在augmented_datasets_tmp/shop目录下生成一个名为reporter.md的报告文件，并将合格和不合格的数据分别导出到指定的output_filepath目录下

    parser = argparse.ArgumentParser(description='KPS Data Checker')
    parser.add_argument('--dataset_path', type=str, required=True, help='The dataset path to check')
    parser.add_argument('--json_file', type=str, required=True, help='The JSON file to check')
    parser.add_argument('--output_filepath', type=str, required=True, help='The output filepath')

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path).resolve()
    output_filepath = Path(args.output_filepath).resolve()
    print("output_filepath: ", output_filepath.__str__())

    json_file = args.json_file

    kpsDataChecker = KPSDataChecker(dataset_path.__str__(), camera_name_list=["camera_head", "camera_wrist_l", "camera_wrist_r"], proprio_stats="proprio_stats.hdf5")
    kpsDataChecker.check()

    kpsDataExport = KPSDataExport(dataset_path.__str__(), json_file)
    kpsDataExport.export_reporter(kpsDataChecker.subdirectories, kpsDataChecker.qualified_path,
                                  kpsDataChecker.mp4_fps_error_path,
                                    kpsDataChecker.mp4_duration_error_path,
                                    kpsDataChecker.mp4_tracks_error_path,
                                    kpsDataChecker.mp4_not_exist_error_path,
                                    kpsDataChecker.parameters_error_path,
                                    kpsDataChecker.proprio_stats_error_path,
                                    output_file="reporter.md")
    unqualified_path = kpsDataChecker.mp4_fps_error_path + kpsDataChecker.mp4_duration_error_path + \
                        kpsDataChecker.mp4_tracks_error_path + kpsDataChecker.mp4_not_exist_error_path + \
                        kpsDataChecker.parameters_error_path + kpsDataChecker.proprio_stats_error_path
    kpsDataExport.export_data(output_filepath.__str__(), kpsDataChecker.qualified_path, unqualified_path, kpsDataChecker.duration_total)