import random
import os
import sys
import argparse

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def generate_geom_terrain(num_x, num_y, geom_type, geom_size, geom_size_cale_range, max_tilt, min_step, max_step, max_total_height, min_spacing, max_spacing, rotation_z_min, rotation_z_max):
    terrain = []
    heights = [[0 for _ in range(num_y)] for _ in range(num_x)]
    prev_color = random.uniform(0.1, 0.5)  # 初始化第一个box的颜色，取值范围在0.1到0.5之间

    for i in range(num_x):
        for j in range(num_y):
            if i == 0 and j == 0:
                heights[i][j] = 0  # 第一块高度为 0
            else:
                # 计算相邻高度差，限制在 min_step 到 max_step 之间
                if i > 0 and j > 0:
                    height = min(heights[i-1][j], heights[i][j-1])
                elif i > 0:
                    height = heights[i-1][j]
                else:
                    height = heights[i][j-1]

                if random.uniform(0, 1) < 0.5:
                    delta_h = random.uniform(-min_step * geom_size[2], -max_step * geom_size[2])  # 限制高度差在几何体高度的 min_step 到 max_step 之间
                else:
                    delta_h = random.uniform(min_step * geom_size[2], max_step * geom_size[2])  # 限制高度差在几何体高度的 min_step 到 max_step 之间
                heights[i][j] = max(0, min(max_total_height, height + delta_h))

            # 随机生成倾斜角度和旋转
            tilt_x = random.uniform(-max_tilt, max_tilt)
            tilt_y = random.uniform(-max_tilt, max_tilt)
            rotation_z = random.uniform(rotation_z_min, rotation_z_max)  # 旋转角度范围参数化

            # 生成中心间距的随机值
            spacing_x = random.uniform(min_spacing * geom_size[0], max_spacing * geom_size[0])
            spacing_y = random.uniform(min_spacing * geom_size[0], max_spacing * geom_size[0])
            
            pos_x = i * spacing_x
            pos_y = j * spacing_y
            pos_z = heights[i][j] # + geom_size[2]  # 让box稍微浮出地面一点

            # 随机生成灰度颜色，确保相邻块色差大于 0.2，灰度范围在0.1到0.5之间
            while True:
                gray_value = random.uniform(0.1, 0.5)
                if abs(gray_value - prev_color) > 0.2:
                    break
            
            prev_color = gray_value  # 更新前一个颜色值
            rgba = f"{gray_value} {gray_value} {gray_value} 1"

            # 构建地形的box geom定义
            size_x = (geom_size[0] + geom_size[0] * random.uniform(-geom_size_cale_range, geom_size_cale_range)) / 2
            size_y = (geom_size[1] + geom_size[1] * random.uniform(-geom_size_cale_range, geom_size_cale_range)) / 2
            size_z = (geom_size[2] + geom_size[2] * random.uniform(-geom_size_cale_range, geom_size_cale_range)) / 2
            terrain.append(f'<geom type="{geom_type}" pos="{pos_x} {pos_y} {pos_z}" size="{size_x} {size_y} {size_z}" euler="{tilt_x} {tilt_y} {rotation_z}" rgba="{rgba}"/>')

    return terrain

def generate_mujoco_xml(num_x, num_y, geom_type, geom_size, geom_size_cale_range, max_tilt, min_step, max_step, max_total_height, min_spacing, max_spacing, rotation_z_min, rotation_z_max, out_file):
    # 生成地形的所有geom块
    terrain_boxes = generate_geom_terrain(num_x, num_y, geom_type, geom_size, geom_size_cale_range, max_tilt, min_step, max_step, max_total_height, min_spacing, max_spacing, rotation_z_min, rotation_z_max)
    terrain_center = (num_x * -geom_size[0] / 2, num_y * -geom_size[1] / 2, 0)

    # 定义MuJoCo XML的头部和尾部
    header = f'''<mujoco model="random_box_terrain">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>

    <worldbody>
        <body name="terrain" pos="{terrain_center[0]} {terrain_center[1]} {terrain_center[2]}">
    '''

    footer = '''
        </body>
    </worldbody>

    <actuator>
        <!-- 在这里你可以定义一些控制器 -->
    </actuator>

</mujoco>'''

    # 拼接完整的XML文件
    xml_content = header + "\n".join(terrain_boxes) + footer

    # 将XML内容写入文件
    with open(f"{out_file}", "w") as f:
        f.write(xml_content)

    _logger.info(f"XML文件已生成为 {out_file}")

def parse_geom_size(s):
    try:
        # 移除括号并分割字符串
        s = s.strip('()')
        parts = s.split(',')
        if len(parts) != 3:
            raise ValueError
        return tuple(float(part) for part in parts)
    except:
        raise argparse.ArgumentTypeError("geom_size must be a tuple of three floats, e.g., (0.5,0.3,0.2)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple instances of the script with different gRPC addresses.')
    parser.add_argument('--num_x', type=int, default=16, help='Number of boxes in x direction')
    parser.add_argument('--num_y', type=int, default=16, help='Number of boxes in y direction')
    parser.add_argument('--geom_type', type=str, default='box', help='Type of geom to use for terrain generation (sphere, ellipsoid, box, cylinder, capsule)')    
    parser.add_argument('--geom_size', type=parse_geom_size, default="(1, 1, 0.2)", help='Size (x, y, z) of each geom')
    parser.add_argument('--geom_size_cale_range', type=float, default=0.5, help="The random range of geom size")
    parser.add_argument('--max_tilt', type=float, default=3, help='Max tilt of each geom')
    parser.add_argument('--min_step', type=float, default=0.2, help='Min step between geoms')
    parser.add_argument('--max_step', type=float, default=0.5, help='Max step between geoms')
    parser.add_argument('--max_total_height', type=float, default=1, help='Max total height of terrain')
    parser.add_argument('--min_spacing', type=float, default=1, help='Min spacing between geoms')
    parser.add_argument('--max_spacing', type=float, default=1.5, help='Max spacing between geoms')
    parser.add_argument('--rotation_z_min', type=float, default=0, help='Min rotation around z axis')
    parser.add_argument('--rotation_z_max', type=float, default=360, help='Max rotation around z axis')
    parser.add_argument('--output', type=str, default='terrain.xml', help='Output file name')
    args = parser.parse_args()


    num_x = args.num_x
    num_y = args.num_y
    geom_type = args.geom_type
    geom_size = args.geom_size
    geom_size_cale_range = args.geom_size_cale_range
    max_tilt = args.max_tilt
    min_step = args.min_step
    max_step = args.max_step
    max_total_height = args.max_total_height
    min_spacing = args.min_spacing
    max_spacing = args.max_spacing
    rotation_z_min = args.rotation_z_min
    rotation_z_max = args.rotation_z_max
    out_file = args.output

    generate_mujoco_xml(num_x, num_y, geom_type, geom_size, geom_size_cale_range, max_tilt, min_step, max_step, 
                        max_total_height, min_spacing, max_spacing, rotation_z_min, rotation_z_max, out_file)

    # generate_mujoco_xml(
    #     10,   # num_x: x方向的box数量
    #     10,   # num_y: y方向的box数量
    #     "ellipsoid",  # geom_type: box的类型
    #     1,    # box_size: 每个box的边长 (2米 x 2米)
    #     3,    # max_tilt: box相对于水平面的最大倾斜角 (3度)
    #     -0.05,  # min_step: 相邻box之间的最小高度差 (0.1米)
    #     0.05,  # max_step: 相邻box之间的最大高度差 (0.2米)
    #     1,    # max_total_height: 场景的总高差最大为1米
    #     0.75,  # min_spacing: box之间的最小中心间距 (0.25米)
    #     1,    # max_spacing: box之间的最大中心间距 (0.5米)
    #     0,    # rotation_z_min: box相对于垂直方向的最小旋转角度
    #     360   # rotation_z_max: box相对于垂直方向的最大旋转角度
    # )
