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


def _load_yaml(path: str):
    """安全加载 YAML 文件"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_file_params(config_path: str):
    config = _load_yaml(config_path)
    file_params = []
    default = config.get("default_settings", {})
    
    for file_config in config["files"]:
        filename = file_config["filename"]
        file_path = os.path.join(Path(config_path).parent, filename)
        # 合并默认参数和文件专属参数
        params = {
            **default, 
            **file_config,
            "filename": file_path,
        }
        file_params.append(params)
    
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
    
    # 打开 USD 文件
    stage = Usd.Stage.Open(usd_file)
    meters_per_unit = stage.GetMetadata('metersPerUnit')
    print(f"metersPerUnit: {meters_per_unit}")
    config_scale = params["transform_options"]["scale"]
    print(f"config_scale: {config_scale}")
    scale = np.array([config_scale, config_scale, config_scale], dtype=np.float64)
    scale = scale * meters_per_unit
    print(f"scale: {scale}")
    

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    # all_prim = get_all_prim(stage)
    # get_bounding_box(all_prim, {}, bbox_cache)

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
        
        # ET.SubElement(body, "pos").text = " ".join(map(str, pos))
        # ET.SubElement(body, "quat").text = " ".join(map(str, quat))
        
        # 处理 Mesh
        if prim.IsA(UsdGeom.Mesh):
            obj_name = export_mesh_to_obj(prim, output_dir)
            # 添加 mesh 到 asset
            mesh_elem = ET.SubElement(asset, "mesh")
            mesh_elem.set("name", prim.GetName())
            mesh_elem.set("file", obj_name)
            mesh_elem.set("scale", " ".join(map(str, scale)))
            
            # 添加 collision geom
            bbox_range = bbox.GetRange()
            collision_size = bbox_range.GetSize() * scale / 2.0
            collision_size = [abs(collision_size[0]), abs(collision_size[2]), abs(collision_size[1])]
            collision_pos = bbox_range.GetMidpoint() * scale - pos
            collision_pos = [-collision_pos[0], collision_pos[2], collision_pos[1]]
            collision_geom = ET.SubElement(body, "geom")
            collision_geom.set("type", "box")
            collision_geom.set("pos", " ".join(map(str, collision_pos)))
            collision_geom.set("size", " ".join(map(str, collision_size)))
            collision_geom.set("group", "3")

            # 添加 visual geom
            visual_geom = ET.SubElement(body, "geom")
            visual_geom.set("type", "mesh")
            visual_geom.set("mesh", prim.GetName())
            visual_geom.set("contype", "0")
            visual_geom.set("conaffinity", "0")
        
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



# def matrix_to_pos_quat(matrix: Gf.Matrix4d):
#     """将 USD 变换矩阵转换为 MJCF 的 pos 和 quat（使用旋转矩阵方法）"""
#     # Step 1: 定义 USD → MuJoCo 的坐标系旋转矩阵（绕X轴-90度）
#     # 这是一个4x4变换矩阵，仅包含旋转，无平移
#     R = Gf.Matrix4d()
#     R.SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0), -90))  # 绕X轴旋转-90度
    
#     # Step 2: 将USD的矩阵应用坐标系转换
#     converted_matrix = matrix * R  # 矩阵乘法顺序取决于USD的矩阵定义（通常是列主序）
    
#     # Step 3: 提取转换后的位置和四元数
#     translation = converted_matrix.ExtractTranslation()
#     rotation = converted_matrix.ExtractRotation()
    
#     # 位置直接按新坐标系读取（Y-up → Z-up 已完成转换）
#     pos = [translation[0], translation[1], translation[2]]
    
#     # 四元数需要调整虚部符号以匹配MuJoCo的wxyz格式
#     usd_quat = rotation.GetQuat()
#     quat = [usd_quat.real, 
#             usd_quat.imaginary[0], 
#             usd_quat.imaginary[1], 
#             usd_quat.imaginary[2]]  # 直接使用转换后的四元数
    
#     return pos, quat

# def export_mesh_to_obj(prim, output_dir):
#     """将 USD Mesh 导出为 OBJ 文件（应用旋转矩阵转换 Y-up → Z-up）"""
#     mesh = UsdGeom.Mesh(prim)
    
#     # 获取顶点和面数据
#     points = mesh.GetPointsAttr().Get()  # Y-up 原始顶点
#     face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
#     face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    
#     # --- 定义 Y-up → Z-up 的旋转矩阵（绕X轴-90度）---
#     # 矩阵形式：
#     # [1  0   0 ]
#     # [0  0   1 ]
#     # [0 -1   0 ]
#     # 等价于将原坐标 (x,y,z) 转换为 (x, z, -y)
#     R = np.array([
#         [1, 0, 0],
#         [0, 0, 1],
#         [0, -1, 0]
#     ])
    
#     # 生成 OBJ 内容
#     obj_content = "# OBJ File\n"
    
#     # 应用旋转矩阵到所有顶点
#     for p in points:
#         # 将顶点转换为 numpy 数组并应用矩阵变换
#         v = np.array([p[0], p[1], p[2]])
#         v_transformed = np.dot(R, v)
#         # 写入转换后的顶点
#         obj_content += f"v {v_transformed[0]:.6f} {v_transformed[1]:.6f} {v_transformed[2]:.6f}\n"
    
#     # 面数据保持不变（仅顶点位置变换，索引顺序不变）
#     start_idx = 0
#     for count in face_vertex_counts:
#         face_indices = face_vertex_indices[start_idx:start_idx + count]
#         obj_content += "f " + " ".join([str(i+1) for i in face_indices]) + "\n"
#         start_idx += count
    
#     # 保存文件
#     obj_name = f"{prim.GetName()}.obj"
#     obj_path = os.path.join(output_dir, obj_name)
#     with open(obj_path, "w") as f:
#         f.write(obj_content)
    
#     return obj_name
