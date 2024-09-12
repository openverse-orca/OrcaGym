import random

def generate_box_terrain(num_x, num_y, box_size, max_tilt, min_step, max_step, max_total_height, min_spacing, max_spacing, rotation_z_min, rotation_z_max):
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

                delta_h = random.uniform(min_step, max_step)  # 限制高度差在0.1到0.2米之间
                heights[i][j] = max(0, min(max_total_height, height + delta_h))

            # 随机生成倾斜角度和旋转
            tilt_x = random.uniform(-max_tilt, max_tilt)
            tilt_y = random.uniform(-max_tilt, max_tilt)
            rotation_z = random.uniform(rotation_z_min, rotation_z_max)  # 旋转角度范围参数化

            # 生成中心间距的随机值
            spacing_x = random.uniform(min_spacing, max_spacing)
            spacing_y = random.uniform(min_spacing, max_spacing)
            
            pos_x = i * spacing_x
            pos_y = j * spacing_y
            pos_z = heights[i][j] + box_size / 10  # 让box稍微浮出地面一点

            # 随机生成灰度颜色，确保相邻块色差大于 0.2，灰度范围在0.1到0.5之间
            while True:
                gray_value = random.uniform(0.1, 0.5)
                if abs(gray_value - prev_color) > 0.2:
                    break
            
            prev_color = gray_value  # 更新前一个颜色值
            rgba = f"{gray_value} {gray_value} {gray_value} 1"

            # 构建地形的box geom定义
            terrain.append(f'<geom type="box" pos="{pos_x} {pos_y} {pos_z}" size="{box_size/2} {box_size/2} {box_size/10}" euler="{tilt_x} {tilt_y} {rotation_z}" rgba="{rgba}"/>')

    return terrain

def generate_mujoco_xml(num_x, num_y, box_size, max_tilt, min_step, max_step, max_total_height, min_spacing, max_spacing, rotation_z_min, rotation_z_max):
    # 生成地形的所有box块
    terrain_boxes = generate_box_terrain(num_x, num_y, box_size, max_tilt, min_step, max_step, max_total_height, min_spacing, max_spacing, rotation_z_min, rotation_z_max)
    
    # 定义MuJoCo XML的头部和尾部
    header = '''<mujoco model="random_box_terrain">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>

    <worldbody>
        <body name="terrain">
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
    with open("terrain.xml", "w") as f:
        f.write(xml_content)

    print("XML文件已生成为 terrain.xml")


generate_mujoco_xml(
    20,   # num_x: x方向的box数量
    20,   # num_y: y方向的box数量
    1,    # box_size: 每个box的边长 (2米 x 2米)
    3,    # max_tilt: box相对于水平面的最大倾斜角 (3度)
    -0.05,  # min_step: 相邻box之间的最小高度差 (0.1米)
    0.05,  # max_step: 相邻box之间的最大高度差 (0.2米)
    1,    # max_total_height: 场景的总高差最大为1米
    1.5,  # min_spacing: box之间的最小中心间距 (0.5米)
    2,    # max_spacing: box之间的最大中心间距 (1米)
    0,    # rotation_z_min: box相对于垂直方向的最小旋转角度
    360   # rotation_z_max: box相对于垂直方向的最大旋转角度
)
