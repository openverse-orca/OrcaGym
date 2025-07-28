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
                    merged_array.extend(data)
                else:
                    merged_array.append(data)
        except json.JSONDecodeError:
            print(f" 错误: 文件 {filename} 不是有效的JSON格式")
        except Exception as e:
            print(f" 警告: 处理文件 {filename} 时出错 - {str(e)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_array, f, indent=2, ensure_ascii=False)

def merge_folders_and_collect_json(root_dir):
    """合并文件夹并收集/合并JSON文件"""
    target_subdir = "234"
    outer_name = "run_01"

    # 收集 JSON 的目录
    json_dir = os.path.join(root_dir, "collected_json_files")
    os.makedirs(json_dir, exist_ok=True)

    # 合并输出目录
    merged_root = os.path.join(root_dir, "merged_data", outer_name)
    merged_dir = os.path.join(merged_root, target_subdir)
    os.makedirs(merged_dir, exist_ok=True)
    
    # 复制主目录
    original_target = os.path.join(root_dir, target_subdir)
    if os.path.exists(original_target):
        shutil.copytree(original_target, merged_dir, dirs_exist_ok=True)
    
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path) or subdir in ["collected_json_files", "merged_data", target_subdir]:
            continue
        
        # 拷贝 JSON 文件
        for root, _, files in os.walk(subdir_path):
            for file in files:
                if file.endswith('.json') and "Operation" in file:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(json_dir, file)

                    counter = 1
                    while os.path.exists(dst_path):
                        name, ext = os.path.splitext(file)
                        dst_path = os.path.join(json_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(src_path, dst_path)
        
        # 合并目录
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

        # 清空子目录
        try:
            shutil.rmtree(subdir_path)
        except Exception as e:
            print(f"警告: 无法删除目录 {subdir_path} - {str(e)}")

    # 合并 JSON
    merged_json_path = os.path.join(root_dir, "merged_json.json")
    # merge_json_arrays(json_dir, merged_json_path)
    # print(f" 所有JSON数组合并到: {merged_json_path}")
    # print(f" 合并目录位置: {merged_dir}")
    merge_json_arrays(json_dir, merged_json_path)

    # 删除收集的 JSON 文件目录
    try:
        shutil.rmtree(json_dir)
        print(f" 已删除临时JSON目录: {json_dir}")
    except Exception as e:
        print(f" 警告: 删除 {json_dir} 失败 - {str(e)}")

    print(f" 所有JSON数组合并到: {merged_json_path}")
    print(f" 合并目录位置: {merged_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并文件夹（参数固定）')
    parser.add_argument('directory', help='要处理的根目录路径')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f" 错误: 目录 {args.directory} 不存在")
        exit(1)

    print(f" 正在处理目录: {args.directory}")
    print(" 合并目标: merged_data/run_01/234")
    print(" 仅处理包含 'Operation' 的 JSON 文件")
    merge_folders_and_collect_json(args.directory)
    print(" 操作完成！")
