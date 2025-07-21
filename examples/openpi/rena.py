import os
import json
import subprocess
from pymediainfo import MediaInfo

def get_size(start_path):
    """递归计算目录总大小(GB)"""
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)  # 转换为GB

def count_files(start_path):
    """统计目录的直接子目录数量(不统计文件)"""
    return sum(1 for item in os.listdir(start_path) 
              if os.path.isdir(os.path.join(start_path, item)))

def get_video_duration(file_path):
    """使用 MediaInfo 获取视频时长(秒)"""
    try:
        media_info = MediaInfo.parse(file_path)
        for track in media_info.tracks:
            if track.track_type == "Video":
                duration = track.duration / 1000  # 转换为秒
                if duration <= 0:
                    print(f"警告: 获取视频 {file_path} 时长失败，返回0或负值")
                    return 0
                return duration
    except Exception as e:
        print(f"无法获取视频时长: {file_path}, 错误: {e}")
        return 0

def get_mp4_duration(file_path):
    """递归计算MP4文件总时长(小时)"""
    total_seconds = 0
    for dirpath, _, filenames in os.walk(file_path):
        for f in filenames:
            if f.lower().endswith('.mp4'):
                fp = os.path.join(dirpath, f)
                duration = get_video_duration(fp)
                if duration is not None:  # 确保duration有效
                    total_seconds += duration
                else:
                    print(f"警告: {fp} 时长获取失败")
    return total_seconds / 3600  # 转换为小时


def format_directory_name(custom_name, size_gb, count, duration_hours):
    """格式化目录名（替换.为p）"""
    size_str = f"{size_gb:.2f}".replace('.', 'p')
    duration_str = f"{duration_hours:.2f}".replace('.', 'p')
    return f"{custom_name}-{size_str}GB_{count}counts_{duration_str}h"

def merge_json_files(json_files, output_path):
    merged_scenes = []
    for json_file in json_files:
        full_json_path = os.path.join(root_path, json_file)
        print(f"正在读取文件: {full_json_path}")
        
        try:
            with open(full_json_path, 'r', encoding='utf-8') as f:
                scenes = json.load(f)
                if isinstance(scenes, list):
                    merged_scenes.extend(scenes)
                else:
                    print(f"警告: {json_file} 内容不是列表，跳过（需为场景对象列表）")
        except Exception as e:
            print(f"读取JSON文件失败: {json_file}, 错误: {e}")
    
    if not merged_scenes:
        print("错误：所有JSON文件均无效，合并失败")
        return None
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_scenes, f, indent=4, ensure_ascii=False)
        print(f"合并后的JSON文件已保存到: {output_path}")
        return output_path
    except Exception as e:
        print(f"写入合并后的JSON文件失败: {output_path}, 错误: {e}")
        return None

def rename_directories_final(root_path):
    # 1. 查找根目录下的所有JSON文件（排除task_info目录）
    all_files = os.listdir(root_path)
    json_files = [
        f for f in all_files 
        if f.endswith('.json') 
        and os.path.isfile(os.path.join(root_path, f))
        and f != "task_info.json"  # 排除已存在的task_info.json
    ]
    
    if not json_files:
        print("错误：根目录下未找到任何有效JSON文件")
        return
    
    # 2. 合并多个JSON文件（若有）
    merged_json_temp = os.path.join(root_path, "merged_json.json")

    print(f"合并后的文件路径: {merged_json_temp}")

    print(f"开始合并JSON文件: {json_files}")
    merged_json_temp = merge_json_files(json_files, merged_json_temp)
    if not merged_json_temp:
        print("合并JSON文件失败，终止处理")
        return

    # 3. 读取合并后的JSON文件并提取场景信息（用于命名）
    try:
        with open(merged_json_temp, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
        
        if not isinstance(merged_data, list) or len(merged_data) == 0:
            print("错误：合并后的JSON文件中没有有效的场景列表（或列表为空）")
            return
        
        first_scene = merged_data[0]
        scene_name = first_scene.get("scene_name", "Default_Scene")
        sub_scene_name = first_scene.get("sub_scene_name", "Default_SubScene")
        third_scene_name = first_scene.get("label_info", {}).get("action_config", [{}])[0].get("skill", "Default_Skill")
        custom_names = [scene_name, sub_scene_name, third_scene_name]

    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return

    # 4. 查找第三层子目录数量
    third_level_count = 0
    first_level_dirs = [
        d for d in os.listdir(root_path) 
        if os.path.isdir(os.path.join(root_path, d)) 
        and d != "task_info"
    ]
    
    if not first_level_dirs:
        print("错误：根目录下没有有效第一层目录（排除task_info后）")
        return
    
    for first_dir in first_level_dirs:
        first_dir_path = os.path.join(root_path, first_dir)
        second_level_dirs = [
            d for d in os.listdir(first_dir_path) 
            if os.path.isdir(os.path.join(first_dir_path, d))
        ]
        
        for second_dir in second_level_dirs:
            second_dir_path = os.path.join(first_dir_path, second_dir)
            third_level_dirs = [
                d for d in os.listdir(second_dir_path) 
                if os.path.isdir(os.path.join(second_dir_path, d))
            ]
            
            if third_level_dirs:
                third_level_sample = os.path.join(second_dir_path, third_level_dirs[0])
                third_level_count = count_files(third_level_sample)
                break
        if third_level_count > 0:
            break
    
    if third_level_count == 0:
        print("错误：未找到有效的第三层目录来计算count值")
        return

    # 5. 重命名第三层目录
    for first_dir in first_level_dirs:
        first_dir_path = os.path.join(root_path, first_dir)
        second_level_dirs = [
            d for d in os.listdir(first_dir_path) 
            if os.path.isdir(os.path.join(first_dir_path, d))
        ]
        
        for second_dir in second_level_dirs:
            second_dir_path = os.path.join(first_dir_path, second_dir)
            third_level_dirs = [
                d for d in os.listdir(second_dir_path) 
                if os.path.isdir(os.path.join(second_dir_path, d))
            ]
            
            for third_dir in third_level_dirs:
                original_third_dir_path = os.path.join(second_dir_path, third_dir)
                count = count_files(original_third_dir_path)
                duration = get_mp4_duration(original_third_dir_path)
                new_third_name = format_directory_name(custom_names[2], 
                                                      get_size(original_third_dir_path), 
                                                      count, 
                                                      duration)
                
                if new_third_name != third_dir:
                    new_third_dir_path = os.path.join(second_dir_path, new_third_name)
                    os.rename(original_third_dir_path, new_third_dir_path)
                    print(f"重命名第三层: {third_dir} -> {new_third_name} (位于 {second_dir_path})")

    # 6. 重命名第二层目录
    for first_dir in first_level_dirs:
        first_dir_path = os.path.join(root_path, first_dir)
        second_level_dirs = [
            d for d in os.listdir(first_dir_path) 
            if os.path.isdir(os.path.join(first_dir_path, d))
        ]
        
        for second_dir in second_level_dirs:
            original_second_dir_path = os.path.join(first_dir_path, second_dir)
            count = third_level_count
            duration = get_mp4_duration(original_second_dir_path)
            new_second_name = format_directory_name(custom_names[1], 
                                                   get_size(original_second_dir_path), 
                                                   count, 
                                                   duration)
            
            if new_second_name != second_dir:
                new_second_dir_path = os.path.join(first_dir_path, new_second_name)
                os.rename(original_second_dir_path, new_second_dir_path)
                print(f"重命名第二层: {second_dir} -> {new_second_name} (位于 {first_dir_path})")

    # 7. 重命名第一层目录
    for first_dir in first_level_dirs:
        original_first_dir_path = os.path.join(root_path, first_dir)
        count = third_level_count
        duration = get_mp4_duration(original_first_dir_path)
        new_first_name = format_directory_name(custom_names[0], 
                                              get_size(original_first_dir_path), 
                                              count, 
                                              duration)
        
        if new_first_name != first_dir:
            new_first_dir_path = os.path.join(root_path, new_first_name)
            os.rename(original_first_dir_path, new_first_dir_path)
            print(f"重命名第一层: {first_dir} -> {new_first_name}")

    # 8. 创建 task_info 目录并保存合并后的 JSON
    task_info_dir = os.path.join(root_path, "task_info")
    os.makedirs(task_info_dir, exist_ok=True)

    merged_json_name = f"{custom_names[0]}-{custom_names[1]}-{custom_names[2]}.json"
    if os.path.exists(merged_json_temp):
        target_json_path = os.path.join(task_info_dir, merged_json_name)
        os.rename(merged_json_temp, target_json_path)
        print(f"合并后的JSON文件已重命名为: {target_json_path}")

    print("所有目录和文件重命名完成！")

if __name__ == "__main__":
    root_path = input("请输入根目录路径: ").strip()
    rename_directories_final(root_path)
