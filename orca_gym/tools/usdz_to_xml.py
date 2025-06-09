import argparse
import yaml
import os

import numpy as np

import glob
from pathlib import Path
from typing import Dict, List
from pxr import Usd, UsdGeom, Gf, Sdf
import xml.etree.ElementTree as ET
from xml.dom import minidom
import subprocess
import trimesh
import orca_gym.utils.rotations as rotations
from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans

def _load_yaml(path: str):
    """安全加载 YAML 文件"""
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def _deep_merge(source, overrides):
    """
    递归合并两个字典，处理嵌套的子节点。
    """
    merged = deepcopy(source)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged

def load_file_params(config_path: str):
    config = _load_yaml(config_path)
    file_params = []
    default = config.get("default_settings", {})
    
    for file_config in config["files"]:
        filename = file_config["filename"]
        file_path = os.path.join(Path(config_path).parent, filename)
        
        # 深度合并默认配置和文件专属配置
        merged_params = _deep_merge(default, file_config)
        
        # 添加/覆盖文件路径
        merged_params["filename"] = file_path
        
        file_params.append(merged_params)
    
    return file_params

def create_converted_dir(config_path : str):
    """
    Create a directory for the USDZ file if it doesn't exist.
    """

    # Get the directory of the config file
    config_dir = os.path.dirname(config_path)

    # Create the directory if it doesn't exist
    converted_dir_path = os.path.join(config_dir, "converted_files")
    if not os.path.exists(converted_dir_path):
        os.makedirs(converted_dir_path)
        # print(f"Created directory: {dir_path}")
    else:
        # print(f"Directory already exists: {dir_path}")
        pass

    return converted_dir_path

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

def get_all_prim(stage):
    # 遍历所有 Prim
    def _traverse_prims(prim, all_prim):
        all_prim.append({"Path": prim.GetPath(), "Type": prim.GetTypeName(), "Prim": prim})
        for child in prim.GetChildren():
            _traverse_prims(prim=child, all_prim=all_prim)

    all_prim = []
    _traverse_prims(prim=stage.Load("/"), all_prim=all_prim)  # 遍历 Stage 的根 Prim
    return all_prim

def get_bounding_box(all_prim, params, bbox_cache):
    for prim in all_prim:
        prim_path = prim["Path"]
        bbox = bbox_cache.ComputeWorldBound(prim["Prim"])
        box_range = bbox.GetRange()
        if box_range.IsEmpty():
            print(f"Prim {prim_path} has an empty bounding box.")
            continue
        min_point = box_range.GetMin()
        max_point = box_range.GetMax()

        # 计算中心点和尺寸
        center = (min_point + max_point) / 2.0
        size = box_range.GetSize()

        print(f"Prim Path: {prim_path}, Type: {prim['Type']}")
        print(f"    Min: {min_point}")
        print(f"    Max: {max_point}")
        print(f"    Center: {center}")
        print(f"    Size: {size}")

def export_mesh_to_obj(prim, output_dir):
    """将 USD Mesh 导出为 OBJ 文件"""
    mesh = UsdGeom.Mesh(prim)
    
    # 获取顶点和面数据
    points = mesh.GetPointsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    
    # 生成 OBJ 内容
    obj_content = "# OBJ File\n"
    for p in points:
        obj_content += f"v {-p[0]} {p[2]} {p[1]}\n"
    
    start_idx = 0
    for count in face_vertex_counts:
        face_indices = face_vertex_indices[start_idx:start_idx + count]
        obj_content += "f " + " ".join([str(i+1) for i in face_indices]) + "\n"
        start_idx += count
    
    # 保存文件
    obj_name = f"{prim.GetName()}.obj"
    obj_path = os.path.join(output_dir, obj_name)
    with open(obj_path, "w") as f:
        f.write(obj_content)

    return obj_name

# def split_mesh(obj_name, output_dir, max_split_mesh_number):
#     # 调用cpp程序TestVHACD执行分割
#     obj_path = os.path.join(output_dir, obj_name)
#     subprocess.run(["./TestVHACD", obj_path, "-h", str(max_split_mesh_number), "-p", "true", "-o", "obj"], check=True)
#     obj_name_list = []
#     for i in range(max_split_mesh_number):
#         obj_name_base = obj_name.split(".")[0]
#         splited_obj_name = f"{obj_name_base}{i:03d}.obj"
#         splited_obj_path = os.path.join(output_dir, splited_obj_name)
#         if os.path.exists(splited_obj_path):
#             obj_name_list.append(splited_obj_name)
    
#     return obj_name_list

def split_mesh(obj_name, output_dir, max_split_mesh_number):
    # 获取当前脚本的绝对路径所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 TestVHACD 的绝对路径
    testvhacd_path = os.path.join(script_dir, "TestVHACD")
    
    # 调用 cpp 程序 TestVHACD 执行分割
    obj_path = os.path.join(output_dir, obj_name)
    subprocess.run(
        [testvhacd_path, obj_path, "-h", str(max_split_mesh_number), "-p", "true", "-o", "obj", "-g", "true"],
        check=True
    )
    
    # 收集生成的 obj 文件列表
    obj_name_list = []
    obj_base = os.path.splitext(obj_name)[0]
    for i in range(max_split_mesh_number):
        splited_obj_name = f"{obj_base}{i:03d}.obj"
        splited_obj_path = os.path.join(output_dir, splited_obj_name)
        if os.path.exists(splited_obj_path):
            obj_name_list.append(splited_obj_name)
    
    return obj_name_list

def matrix_to_pos_quat_scale(matrix: Gf.Matrix4d, scale : np.ndarray):
    """将 USD 变换矩阵转换为 MJCF 的 pos 和 quat"""


    factor = matrix.Factor()
    scale *= factor[2]
    print(f"scale: {scale}")

    translation = matrix.ExtractTranslation() * scale

    matrix.Orthonormalize()
    quat = matrix.ExtractRotationQuat()
    
    # TODO: Orca 处理USD导入的时候做了转换，这里要做适配
    pos = [-translation[0], translation[2], translation[1]]
    quat = [quat.real, -quat.imaginary[0], -quat.imaginary[1], -quat.imaginary[2]]

    return pos, quat, scale


def _add_mesh_assets(asset, mesh_file_name, scale):
    mesh_name = mesh_file_name.split(".")[0]
    mesh_elem = ET.SubElement(asset, "mesh")
    mesh_elem.set("name", mesh_name)
    mesh_elem.set("content_type", "model/obj")
    mesh_elem.set("file", f"usd:{mesh_file_name}")
    mesh_elem.set("scale", " ".join(map(str, scale)))

def _add_box_geom(body, params, collision_pos, collision_size):
    collision_geom = ET.SubElement(body, "geom")
    collision_geom.set("type", "box")
    collision_geom.set("group", "3")
    collision_geom.set("density", str(params["physics_options"]["density"]))
    collision_geom.set("pos", " ".join(map(str, collision_pos)))
    collision_size = np.clip(collision_size, 0.01, None)   # 限制最小尺寸
    collision_geom.set("size", " ".join(map(str, collision_size)))

def _add_hull_geom(body, params, obj_name):
    obj_name_base = obj_name.split(".")[0]
    collision_geom = ET.SubElement(body, "geom")
    collision_geom.set("type", "mesh")
    collision_geom.set("mesh", obj_name_base)
    collision_geom.set("group", "3")
    collision_geom.set("density", str(params["physics_options"]["density"]))

def _add_visualize_geom(body, mesh_file_name):
    mesh_name = mesh_file_name.split(".")[0]
    visual_geom = ET.SubElement(body, "geom")
    visual_geom.set("type", "mesh")
    visual_geom.set("mesh", mesh_name)
    visual_geom.set("contype", "0")
    visual_geom.set("conaffinity", "0")
    
def process_ori_mesh(obj_name, asset, body, output_dir, scale, bbox, params, pos):
    # 如果需要用到obj文件，则添加 mesh 到 asset
    if params["collision_options"]["collider_type"] == "convex_hull" or params["debug_options"]["visualize_obj"]:
        _add_mesh_assets(asset, obj_name, output_dir, scale)

    # 添加 collision geom
    if params["collision_options"]["collider_type"] == "bounding_box":
        bbox_range = bbox.GetRange()
        collision_size = bbox_range.GetSize() * scale / 2.0
        collision_size = [abs(collision_size[0]), abs(collision_size[2]), abs(collision_size[1])]
        collision_pos = bbox_range.GetMidpoint() * scale - pos
        collision_pos = [-collision_pos[0], collision_pos[2], collision_pos[1]]
        _add_box_geom(body, params, collision_pos, collision_size)

    elif params["collision_options"]["collider_type"] == "convex_hull":
        _add_hull_geom(body, params, obj_name)
    else:
        raise ValueError(f"Unknown collider type: {params['collision_options']['collider_type']}")

    # 添加 visual geom
    if params["debug_options"]["visualize_obj"]:
        _add_visualize_geom(body, obj_name)

def process_split_mesh(obj_name_list, asset, body, output_dir, scale, bbox, params, pos):
    aabb_list = []
    for obj_name in obj_name_list:
        # 如果需要用到obj文件，则添加 mesh 到 asset
        if params["collision_options"]["collider_type"] == "convex_hull" or params["debug_options"]["visualize_obj"]:
            _add_mesh_assets(asset, obj_name, scale)
        
        if params["collision_options"]["collider_type"] == "bounding_box":
            # 这里不直接添加XML对象，只是先把包围盒搜集起来
            obj_path = os.path.join(output_dir, obj_name)
            mesh = trimesh.load(obj_path)
            aabb_list.append(mesh.bounding_box)
            # print(f"Bounding box for {obj_name}: {mesh.bounding_box.bounds}")
        elif params["collision_options"]["collider_type"] == "convex_hull":
            _add_hull_geom(body, params, obj_name)
        else:
            raise ValueError(f"Unknown collider type: {params['collision_options']['collider_type']}")

        # 添加 visual geom
        if params["debug_options"]["visualize_obj"]:
            _add_visualize_geom(body, obj_name)

    if len(aabb_list) == 0:
        return

    ###########################
    # 处理包围盒
    ###########################
    # 步骤1：剔除被完全包含的包围盒
    def _is_contained(a, b):
        """判断a是否被b完全包含"""
        return (a.bounds[0] >= b.bounds[0]).all() and (a.bounds[1] <= b.bounds[1]).all()
    
    # 过滤掉被其他AABB包含的包围盒
    filtered_aabbs = []
    for i, box in enumerate(aabb_list):
        if not any(_is_contained(box, other) for j, other in enumerate(aabb_list) if i != j):
            filtered_aabbs.append(box)
        else:
            print(f"Filtered out AABB {i} because it is contained by another AABB.")
    
    # 步骤2：合并高重合度包围盒（阈值设为90%）
    merged_aabbs = []
    skip_indices = set()
    for itr in range(params["collision_options"]["merge_iterations"]):
        for i in range(len(filtered_aabbs)):
            if i in skip_indices:
                continue
            current = filtered_aabbs[i]
            merged = False
            
            # 计算与其他包围盒的交集
            for j in range(i+1, len(filtered_aabbs)):
                if j in skip_indices:
                    continue
                
                other = filtered_aabbs[j]
                # 计算交集体积占比
                inter_min = np.maximum(current.bounds[0], other.bounds[0])
                inter_max = np.minimum(current.bounds[1], other.bounds[1])
                inter_vol = np.prod(np.clip(inter_max - inter_min, 0, None))
                
                vol_current = current.volume
                vol_other = other.volume
                ratio = inter_vol / min(vol_current, vol_other) if min(vol_current, vol_other) > 0 else 0
                
                # 若交集体积占比超过阈值则合并
                if ratio >= params["collision_options"]["merge_threshold"]:
                    new_min = np.minimum(current.bounds[0], other.bounds[0])
                    new_max = np.maximum(current.bounds[1], other.bounds[1])
                    merged_box = trimesh.primitives.Box(bounds=[new_min, new_max])
                    merged_aabbs.append(merged_box)
                    skip_indices.update([i, j])
                    merged = True
                    print(f"Merged AABBs {i} and {j} into a new AABB.")
                    filtered_aabbs.append(merged_box)
                    break
            
            if not merged:
                merged_aabbs.append(current)
        
        if not merged:
            # 如果没有合并成功，说明没有更多的包围盒可以合并了
            break

        if itr < params["collision_options"]["merge_iterations"] - 1:
            merged_aabbs = []

    assert len(merged_aabbs) > 0, "No valid AABBs found after filtering and merging."
    for aabb in merged_aabbs:
        # 最终再写入到xml
        size = aabb.extents         # 尺寸 (dx, dy, dz)
        center = aabb.centroid        # 中心点坐标 (x_center, y_center, z_center)

        collision_pos = center * scale - pos
        collision_size = size * scale / 2.0
        _add_box_geom(body, params, collision_pos, collision_size)


def build_mjcf_xml(usd_file, mjcf_file, output_dir, params):
    """主函数：构建 MJCF XML"""
    print(f"Processing {usd_file}")

    # 初始化 MJCF 文档
    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    base_body = ET.SubElement(worldbody, "body")
    base_body.set("name", "base")
    if params["physics_options"]["free_joint"]:
        print("Adding free joint to base body")
        base_joint = ET.SubElement(base_body, "freejoint")
        base_joint.set("name", "base_joint")
        if params["physics_options"]["mass"] > 0:
            inertial = ET.SubElement(base_body, "inertial")
            inertial.set("pos", "0 0 0")
            inertial.set("mass", str(params["physics_options"]["mass"]))
    
    # 打开 USD 文件
    stage = Usd.Stage.Open(usd_file)
    meters_per_unit = stage.GetMetadata('metersPerUnit')
    print(f"metersPerUnit: {meters_per_unit}")
    config_scale = params["transform_options"]["scale"]
    print(f"config_scale: {config_scale}")
    config_scale = np.array([config_scale, config_scale, config_scale], dtype=np.float64)
    scale = config_scale * meters_per_unit
    print(f"scale: {scale}")


    # 添加usdz文件到asset
    usd_file_name = os.path.basename(usd_file)
    _add_mesh_assets(asset, usd_file_name, config_scale)
    _add_visualize_geom(base_body, usd_file_name)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])

    # 递归遍历 USD 节点
    def _process_prim(prim, parent_elem, bbox_cache, scale):
        bbox = bbox_cache.ComputeWorldBound(prim)
        if bbox.GetRange().IsEmpty():
            print(f"prim {prim.GetPath()} has an empty bounding box.")
            return

        # 处理变换
        # 获取当前节点的局部变换矩阵（相对于父节点）
        xform = UsdGeom.Xformable(prim)
        local_matrix = xform.GetLocalTransformation()
        pos, quat, scale = matrix_to_pos_quat_scale(local_matrix, scale.copy())

        body_hash = hash(prim.GetPath())

        # 创建当前 body
        body = ET.SubElement(parent_elem, "body")
        body.set("name", f"{prim.GetName()}_{body_hash}")
        body.set("pos", " ".join(map(str, pos)))
        body.set("quat", " ".join(map(str, quat)))
        
        # 处理 Mesh
        if prim.IsA(UsdGeom.Mesh):
            obj_name = export_mesh_to_obj(prim, output_dir)
            
            if params["collision_options"]["split_mesh"]:
                obj_name_list = split_mesh(obj_name, output_dir, params["collision_options"]["max_split_mesh_number"])
                process_split_mesh(obj_name_list, asset, body, output_dir, scale, bbox, params, pos)
            else:
                process_ori_mesh(obj_name, asset, body, output_dir, scale, bbox, params, pos)
        
        # 递归处理子节点
        for child in prim.GetChildren():
            _process_prim(child, body, bbox_cache, scale.copy())
    
    # 从根节点开始处理
    for prim in stage.GetPseudoRoot().GetChildren():
        _process_prim(prim, base_body, bbox_cache, scale.copy())

    # 美化 XML 并保存
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml()
    with open(mjcf_file, "w") as f:
        f.write(xml_str)


def main(config_path):
    """
    Main function to process the USDZ file and add mujoco physics geometry.
    """
    file_params = load_file_params(config_path)

    # print("Loaded file parameters:")
    # for params in file_params:
    #     print(params)

    converted_dir = create_converted_dir(config_path)
    for params in file_params:
        usdz_path = params["filename"]
        usdc_path = convert_usdz2usdc(usdz_path, converted_dir)
        usdc_dir = os.path.dirname(usdc_path)
        xml_path = os.path.join(usdc_dir, os.path.basename(usdz_path).replace(".usdz", ".xml"))
        build_mjcf_xml(usd_file=usdc_path, mjcf_file=xml_path, output_dir=usdc_dir, params=params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Porcess USDZ file and add mujoco physics geometry')
    parser.add_argument('--config', type=str, help='The config file to use')
    args = parser.parse_args()

    main(args.config)

