import os
import shutil
import json
import argparse

def merge_json_arrays(json_dir, output_file):
    """合并所有JSON数组文件到一个大数组"""
    merged_array = []
    
    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(json_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    merged_array.extend(data)  # 直接合并数组
                else:
                    merged_array.append(data)  # 非数组内容作为单个元素添加
                    
        except json.JSONDecodeError:
            print(f"错误: 文件 {filename} 不是有效的JSON格式")
        except Exception as e:
            print(f"警告: 处理文件 {filename} 时出错 - {str(e)}")
    
    # 写入合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_array, f, indent=2, ensure_ascii=False)

def merge_folders_and_collect_json(root_dir, target_subdir="234", merge_json=False):
    """合并文件夹并收集/合并JSON文件"""
    # 创建存放JSON文件的新目录
    json_dir = os.path.join(root_dir, "collected_json_files")
    os.makedirs(json_dir, exist_ok=True)
    
    # 创建合并后的目录结构
    merged_root = os.path.join(root_dir, "merged_data")
    merged_dir = os.path.join(merged_root, target_subdir)
    os.makedirs(merged_dir, exist_ok=True)
    
    # 复制原始目标目录
    original_target = os.path.join(root_dir, target_subdir)
    if os.path.exists(original_target):
        shutil.copytree(original_target, merged_dir, dirs_exist_ok=True)
    
    # 处理其他子目录
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        
        if not os.path.isdir(subdir_path) or subdir in ["collected_json_files", "merged_data", target_subdir]:
            continue
            
        # 移动JSON文件（只处理包含Operation的文件）
        for root, _, files in os.walk(subdir_path):
            for file in files:
                if file.endswith('.json') and "Operation" in file:  # 添加条件判断
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(json_dir, file)
                    
                    # 处理重名文件
                    counter = 1
                    while os.path.exists(dst_path):
                        name, ext = os.path.splitext(file)
                        dst_path = os.path.join(json_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(src_path, dst_path)
        
        # 移动子目录
        for item in os.listdir(subdir_path):
            src = os.path.join(subdir_path, item)
            if os.path.isdir(src):
                dst = os.path.join(merged_dir, item)
                
                counter = 1
                while os.path.exists(dst):
                    new_name = f"{item}_{counter}"
                    dst = os.path.join(merged_dir, new_name)
                    counter += 1
                
                shutil.move(src, dst)
        
        # 删除空目录
        try:
            shutil.rmtree(subdir_path)
        except Exception as e:
            print(f"警告: 无法删除目录 {subdir_path} - {str(e)}")
    
    # 合并JSON数组
    if merge_json:
        merged_json_path = os.path.join(root_dir, "merged_json.json")
        merge_json_arrays(json_dir, merged_json_path)
        print(f"所有JSON数组合并到: {merged_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并文件夹并合并JSON数组')
    parser.add_argument('directory', help='要处理的根目录路径')
    parser.add_argument('--target', default="234", help='主合并目录名（默认为234）')
    parser.add_argument('--merge-json', action='store_true', help='合并JSON数组到单个文件')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"错误: 目录 {args.directory} 不存在")
        exit(1)
    
    print(f"正在处理目录: {args.directory}")
    print(f"所有子文件夹将合并到: merged_data/{args.target}")
    print("仅处理文件名包含'Operation'的JSON文件")
    merge_folders_and_collect_json(args.directory, args.target, args.merge_json)
    print("操作完成!")
    print(f"JSON文件已保存到: {os.path.join(args.directory, 'collected_json_files')}")
    print(f"合并的数据已保存到: {os.path.join(args.directory, 'merged_data', args.target)}")