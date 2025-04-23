from pxr import Usd
import argparse
import yaml

import yaml
import glob
from pathlib import Path
from typing import Dict, List


def _load_yaml(path: str):
    """安全加载 YAML 文件"""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _find_all_usdz_files(root_dir: Path) -> List[Path]:
    """查找目录下所有 .usdz 文件"""
    return list(root_dir.rglob("*.usdz"))

def _match_pattern(pattern: str, files: List[Path]) -> List[Path]:
    """将 glob 模式转换为实际文件路径"""
    # 将 * 转换为 glob 语法（支持 ** 递归匹配）
    glob_pattern = pattern.replace("**", "*").replace("//", "/")
    return [f for f in files if f.match(glob_pattern)]

def _extract_file_specific_params(file_path: Path, config: Dict) -> Dict:
    """提取针对特定文件的覆盖参数（可选功能）"""
    # 如果需要支持文件名特定覆盖，可在此实现
    return {}


def load_file_params(config_path: str):
    """加载配置并处理通配符模式"""
    config = _load_yaml(config_path)
    file_params = []

    # usdz 文件存放在 yaml  文件同级目录
    usdz_path = Path(config_path).parent
    # print(f"Loading config from: {usdz_path}")

    all_usdz_files = _find_all_usdz_files(usdz_path)
    
    # 按优先级处理模式：具体文件 > 通配符模式
    for item in config.get("patterns", []):
        if "pattern" not in item:
            continue
        
        # 使用 glob 匹配文件
        matched_files = _match_pattern(item["pattern"], all_usdz_files)
        
        for file_path in matched_files:
            # 合并参数：默认设置 + 当前模式参数
            params = {
                **config["default_settings"],
                **item,
                **_extract_file_specific_params(file_path, config)
            }
            
            file_params.append({
                "filename": str(file_path),
                **params
            })
    
    return file_params

def create_directories(config_path : str):
    """
    Create a directory for the USDZ file if it doesn't exist.
    """
    import os

    # Get the directory of the config file
    config_dir = os.path.dirname(config_path)
    config = _load_yaml(config_path)

    directory_settings = config.get("directory_settings", {})
    directories = {}
    
    for key, directory in directory_settings.items():
        # Create the directory if it doesn't exist
        dir_path = os.path.join(config_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            # print(f"Created directory: {dir_path}")
        else:
            # print(f"Directory already exists: {dir_path}")
            pass

        directories[key] = dir_path

    return directories


def convert_usdz2usdc(usdz_path, converted_dir):
    """
    Unzip the USDZ file to a temporary directory.
    """
    import zipfile
    import os
    import tempfile

    usdz_file_name = os.path.basename(usdz_path)
    usdz_file_name_without_ext = os.path.splitext(usdz_file_name)[0]
    temp_dir = os.path.join(converted_dir, usdz_file_name_without_ext)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        # print(f"Created temporary directory: {temp_dir}")
    else:
        # print(f"Temporary directory already exists: {temp_dir}")
        # Remove existing files in the directory
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Removed existing file: {file_path}")

    # Unzip the USDZ file
    with zipfile.ZipFile(usdz_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Rename the unzipped usdc file to the same name as the usdz file
    usdc_file_name = usdz_file_name_without_ext + ".usdc"
    usdc_file_path = os.path.join(temp_dir, usdc_file_name)
    default_usdc_file_path = os.path.join(temp_dir, "scene.usdc")
    if os.path.exists(default_usdc_file_path):
        os.rename(default_usdc_file_path, usdc_file_path)
        # print(f"Renamed {default_usdc_file_path} to {usdc_file_path}")
    else:
        print(f"Default usdc file not found: {default_usdc_file_path}")
    
    return usdc_file_path

def _get_all_prim(stage):
    # 遍历所有 Prim
    def traverse_prims(prim, all_prim):
        all_prim.append({"Prim": prim.GetPath(), "Type": prim.GetTypeName()})
        for child in prim.GetChildren():
            traverse_prims(prim=child, all_prim=all_prim)

    all_prim = []
    traverse_prims(prim=stage.Load("/"), all_prim=all_prim)  # 遍历 Stage 的根 Prim
    return all_prim

def _convert_mesh_to_obj(all_prim, params):
    for prim in all_prim:
        if prim["Type"] == "Mesh":
            # 处理 Mesh 类型的 Prim
            mesh_path = prim["Prim"]
            print(f"Processing Mesh: {mesh_path}")

            # 这里可以添加转换为 OBJ 的逻辑
            # 例如，使用 Open3D 或其他库将 Mesh 转换为 OBJ 格式
            # obj_file_path = convert_mesh_to_obj(mesh_path, params)
            # print(f"Converted Mesh to OBJ: {obj_file_path}")


def process_usdc_file(usdc_path, params):
    """
    Process the USDC file and add mujoco physics geometry.
    """
    stage = Usd.Stage.Open(usdc_path)
    if not stage:
        print(f"Failed to open {usdc_path}")
        return

    all_prim = _get_all_prim(stage)
    
    if params["convert_to_obj"]:
        # Convert mesh to OBJ format
        _convert_mesh_to_obj(all_prim, params)

    # 关闭 Stage
    stage = None

def main(config_path):
    """
    Main function to process the USDZ file and add mujoco physics geometry.
    """
    file_params = load_file_params(config_path)

    # print("Loaded file parameters:")
    # for params in file_params:
    #     print(params)

    directories = create_directories(config_path)
    for params in file_params:
        usdz_path = params["filename"]
        converted_dir = directories["converted_directory"]
        usdc_path = convert_usdz2usdc(usdz_path, converted_dir)
        process_usdc_file(usdc_path, params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Porcess USDZ file and add mujoco physics geometry')
    parser.add_argument('--config', type=str, help='The config file to use')
    args = parser.parse_args()

    main(args.config)





# # 打开 Stage
# stage = Usd.Stage.Open("/home/superfhwl/workspace/3d_assets/Cup_of_Coffee.usdz")


# upAxis = stage.GetMetadata("upAxis")  # 获取 Stage 的 upAxis 元数据
# print(f"Stage upAxis: {upAxis}")

# # 遍历所有 Prim
# def traverse_prims(prim):
#     print(f"Prim: {prim.GetPath()}, Type: {prim.GetTypeName()}")
#     for child in prim.GetChildren():
#         traverse_prims(child)

# traverse_prims(stage.Load("/"))  # 遍历 Stage 的根 Prim

# # 关闭 Stage
# stage = None