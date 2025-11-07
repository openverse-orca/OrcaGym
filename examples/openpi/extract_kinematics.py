#!/usr/bin/env python3
"""
提取并分析两个机器人的运动学链，计算每个关节的全局旋转轴
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_matrix(quat_str):
    """将quaternion字符串(w x y z)转换为旋转矩阵"""
    if not quat_str:
        return np.eye(3)
    quat = [float(x) for x in quat_str.split()]
    # MuJoCo使用(w x y z)，scipy使用(x y z w)
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

def pos_to_vector(pos_str):
    """将position字符串转换为向量"""
    if not pos_str:
        return np.zeros(3)
    return np.array([float(x) for x in pos_str.split()])

def euler_to_matrix(euler_str):
    """将欧拉角字符串转换为旋转矩阵"""
    if not euler_str:
        return np.eye(3)
    euler = [float(x) for x in euler_str.split()]
    return R.from_euler('xyz', euler).as_matrix()

class KinematicChain:
    def __init__(self, xml_path, arm_prefix):
        """
        xml_path: 机器人XML文件路径
        arm_prefix: 手臂关节前缀（如'J_arm_l_', 'LEFT_J'）
        """
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.arm_prefix = arm_prefix
        self.joints = []
        
    def find_joint_chain(self, body_elem, parent_rot=np.eye(3), parent_pos=np.zeros(3), depth=0):
        """递归查找关节链，计算每个关节的全局旋转轴"""
        # 获取当前body的变换
        pos = pos_to_vector(body_elem.get('pos', '0 0 0'))
        quat_str = body_elem.get('quat')
        euler_str = body_elem.get('euler')
        
        if quat_str:
            local_rot = quat_to_matrix(quat_str)
        elif euler_str:
            local_rot = euler_to_matrix(euler_str)
        else:
            local_rot = np.eye(3)
        
        # 计算全局变换
        global_rot = parent_rot @ local_rot
        global_pos = parent_pos + parent_rot @ pos
        
        # 查找关节
        for joint in body_elem.findall('joint'):
            joint_name = joint.get('name', '')
            if self.arm_prefix in joint_name:
                local_axis = pos_to_vector(joint.get('axis', '0 0 1'))
                global_axis = global_rot @ local_axis
                # 归一化
                global_axis = global_axis / np.linalg.norm(global_axis)
                
                self.joints.append({
                    'name': joint_name,
                    'local_axis': local_axis,
                    'global_axis': global_axis,
                    'global_rot': global_rot.copy(),
                    'global_pos': global_pos.copy(),
                })
                
                print(f"{'  ' * depth}{joint_name}")
                print(f"{'  ' * depth}  Local axis:  [{local_axis[0]:7.3f}, {local_axis[1]:7.3f}, {local_axis[2]:7.3f}]")
                print(f"{'  ' * depth}  Global axis: [{global_axis[0]:7.3f}, {global_axis[1]:7.3f}, {global_axis[2]:7.3f}]")
        
        # 递归处理子body
        for child_body in body_elem.findall('body'):
            self.find_joint_chain(child_body, global_rot, global_pos, depth + 1)
    
    def extract(self):
        """提取关节链"""
        worldbody = self.root.find('worldbody')
        for body in worldbody.findall('.//body'):
            self.find_joint_chain(body)
        return deduplicate_joints(self.joints)


def deduplicate_joints(joint_list, limit=7):
    """按照出现顺序去重，只保留前 limit 个关节"""
    unique = {}
    ordered = []
    for joint in joint_list:
        name = joint['name']
        if name not in unique:
            unique[name] = True
            ordered.append(joint)
        if len(ordered) >= limit:
            break
    return ordered


def axis_to_axis_rotation(vec_from, vec_to):
    vec_from = vec_from / np.linalg.norm(vec_from)
    vec_to = vec_to / np.linalg.norm(vec_to)
    dot_prod = np.clip(np.dot(vec_from, vec_to), -1.0, 1.0)
    if np.isclose(dot_prod, 1.0):
        return R.from_quat([0, 0, 0, 1])
    if np.isclose(dot_prod, -1.0):
        # 180°，选择任意垂直向量作为轴
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(vec_from, axis):
            axis = np.array([0.0, 1.0, 0.0])
        rot_axis = np.cross(vec_from, axis)
        rot_axis /= np.linalg.norm(rot_axis)
        return R.from_rotvec(rot_axis * np.pi)
    rot_axis = np.cross(vec_from, vec_to)
    rot_axis /= np.linalg.norm(rot_axis)
    angle = np.arccos(dot_prod)
    return R.from_rotvec(rot_axis * angle)

def main():
    print("=" * 80)
    print("🤖 青龙机器人（OpenLoong）- 左臂关节链")
    print("=" * 80)
    openloong = KinematicChain(
        '/home/orca/OrcaWorkStation/OrcaGym_Assets/robots/openloong/models/openloong_gripper_2f85_fix_base.xml',
        'J_arm_l_'
    )
    openloong_joints = openloong.extract()
    
    print("\n" + "=" * 80)
    print("🤖 青龙机器人（OpenLoong）- 右臂关节链")
    print("=" * 80)
    openloong_r = KinematicChain(
        '/home/orca/OrcaWorkStation/OrcaGym_Assets/robots/openloong/models/openloong_gripper_2f85_fix_base.xml',
        'J_arm_r_'
    )
    openloong_r_joints = openloong_r.extract()
    
    print("\n" + "=" * 80)
    print("🤖 DexforceW1机器人 - 左臂关节链")
    print("=" * 80)
    dexforce = KinematicChain(
        '/home/orca/Assets/跨维URDF/URDF/DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M/DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M_obj.xml',
        'LEFT_J'
    )
    dexforce_joints = dexforce.extract()
    
    print("\n" + "=" * 80)
    print("🤖 DexforceW1机器人 - 右臂关节链")
    print("=" * 80)
    dexforce_r = KinematicChain(
        '/home/orca/Assets/跨维URDF/URDF/DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M/DexforceW1V020_INDUSTRIAL_DH_PGC_GRIPPER_M_obj.xml',
        'RIGHT_J'
    )
    dexforce_r_joints = dexforce_r.extract()
    
    print("\n" + "=" * 80)
    print("📊 关节轴对比分析")
    print("=" * 80)
    print("\n左臂对比：")
    print(f"{'关节':<15} {'青龙全局轴':<30} {'DexforceW1全局轴':<30} {'点积':<10}")
    print("-" * 85)
    for i, (ol_joint, df_joint) in enumerate(zip(openloong_joints, dexforce_joints), start=1):
        ol_axis = ol_joint['global_axis']
        df_axis = df_joint['global_axis']
        dot_product = np.dot(ol_axis, df_axis)
        print(f"{ol_joint['name']:<15} [{ol_axis[0]:6.3f}, {ol_axis[1]:6.3f}, {ol_axis[2]:6.3f}]  "
              f"[{df_axis[0]:6.3f}, {df_axis[1]:6.3f}, {df_axis[2]:6.3f}]  {dot_product:6.3f}")
    
    print("\n右臂对比：")
    print(f"{'关节':<15} {'青龙全局轴':<30} {'DexforceW1全局轴':<30} {'点积':<10}")
    print("-" * 85)
    for i, (ol_joint, df_joint) in enumerate(zip(openloong_r_joints, dexforce_r_joints), start=1):
        ol_axis = ol_joint['global_axis']
        df_axis = df_joint['global_axis']
        dot_product = np.dot(ol_axis, df_axis)
        print(f"{ol_joint['name']:<15} [{ol_axis[0]:6.3f}, {ol_axis[1]:6.3f}, {ol_axis[2]:6.3f}]  "
              f"[{df_axis[0]:6.3f}, {df_axis[1]:6.3f}, {df_axis[2]:6.3f}]  {dot_product:6.3f}")
    
    print("\n" + "=" * 80)
    print("🎯 映射建议")
    print("=" * 80)
    print("\n点积含义：")
    print("  1.0  → 轴方向完全相同，直接复制值")
    print(" -1.0  → 轴方向完全相反，需要取反（乘-1）")
    print("  0.0  → 轴方向垂直，需要重新映射")
    print("  其他  → 轴方向部分对齐，可能需要几何转换")
    
    # 计算映射关系
    print("\n基于青龙实际运行值 Right: [-1.900, 0.500, -0.001, 2.000, -1.570, 0.000, 0.000]")
    print("                    Left:  [1.900, -0.500, 0.001, 2.000, 1.570, 0.000, 0.000]")
    print("\n建议的DexforceW1初始值（需要根据点积调整符号）：")
    
    openloong_left_values = [1.900, -0.500, 0.001, 2.000, 1.570, 0.000, 0.000]
    openloong_right_values = [-1.900, 0.500, -0.001, 2.000, -1.570, 0.000, 0.000]
    
    print("\nLeft arm mapping:")
    for idx, (ol_joint, df_joint, base_value) in enumerate(zip(openloong_joints, dexforce_joints, openloong_left_values), start=1):
        ol_axis = ol_joint['global_axis']
        df_axis = df_joint['global_axis']
        dot_product = np.dot(ol_axis, df_axis)
        
        # 简单映射：如果点积接近-1，取反；如果接近1，保持；如果接近0，可能需要特殊处理
        if abs(dot_product - 1.0) < 0.1:
            mapped_value = base_value
            mapping = "直接复制"
        elif abs(dot_product + 1.0) < 0.1:
            mapped_value = -base_value
            mapping = "取反"
        else:
            rot = axis_to_axis_rotation(ol_axis, df_axis)
            angle_deg = np.degrees(np.linalg.norm(rot.as_rotvec()))
            mapped_value = base_value
            mapping = f"轴需补偿 ~{angle_deg:4.1f}° (dot={dot_product:.2f})"

        print(f"  {df_joint['name']:<12}: {mapped_value:7.3f}  ({mapping})")
    
    print("\nRight arm mapping:")
    for idx, (ol_joint, df_joint, base_value) in enumerate(zip(openloong_r_joints, dexforce_r_joints, openloong_right_values), start=1):
        ol_axis = ol_joint['global_axis']
        df_axis = df_joint['global_axis']
        dot_product = np.dot(ol_axis, df_axis)
        
        if abs(dot_product - 1.0) < 0.1:
            mapped_value = base_value
            mapping = "直接复制"
        elif abs(dot_product + 1.0) < 0.1:
            mapped_value = -base_value
            mapping = "取反"
        else:
            rot = axis_to_axis_rotation(ol_axis, df_axis)
            angle_deg = np.degrees(np.linalg.norm(rot.as_rotvec()))
            mapped_value = base_value
            mapping = f"轴需补偿 ~{angle_deg:4.1f}° (dot={dot_product:.2f})"

        print(f"  {df_joint['name']:<12}: {mapped_value:7.3f}  ({mapping})")

if __name__ == '__main__':
    main()

