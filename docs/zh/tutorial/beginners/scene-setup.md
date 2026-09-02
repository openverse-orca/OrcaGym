# 🎬 场景搭建 — 往场景里放东西

在上一节，我们直接连接了一个已经搭好的场景。这一节，你将学会**自己搭建场景**：加载机器人、摆放物体、设置灯光。

---

## 场景是什么？

在 OrcaGym 中，"场景"由以下元素组成：

```
场景 (Scene)
├── Actor（角色） — 机器人、桌子、方块… 任何 3D 物体
├── Light（灯光） — 点光源、方向光等
├── Camera（相机） — 渲染视角
└── Material（材质） — 物体的颜色/纹理
```

**关键概念**：`OrcaGymScene` 负责场景的**搭建**（添加/删除物体），`OrcaGymEulerEnv` 负责场景的**仿真**（物理步进）。

---

## 第一步：创建一个空场景

```python
"""
setup_my_scene.py — 搭建一个简单场景：桌子 + 机械臂 + 方块
"""

import numpy as np
from orca_gym.scene.orca_gym_scene import (
 OrcaGymScene, # 场景管理器
 Actor, # 场景中的物体
 LightInfo, # 灯光参数
 MaterialInfo, # 材质参数
)


def build_scene():
 """搭建场景并发布到仿真服务器"""

 # 1. 创建场景管理器 —— 连接到仿真服务器
 print("正在连接场景管理器...")
 scene = OrcaGymScene(grpc_addr="localhost:50051")

 # 2. 发布一个空场景（清空之前的内容）
 scene.publish_scene()
 print("✅ 空场景已发布")

 # ============================================================
 # 3. 添加 Actor（角色/物体）
 # ============================================================

 # --- 3a. 添加一张桌子 ---
 # Actor 由四个要素定义：名字、资产路径、位置、旋转、缩放
 table = Actor(
 name="table_1", # 场景中唯一的名字
 asset_path="assets/tables/table_80x80", # 服务器上的资产路径
 position=np.array([0.5, 0.0, 0.0]), # [x, y, z] 世界坐标（米）
 rotation=np.array([1.0, 0.0, 0.0, 0.0]), # 四元数 [w, x, y, z]
 scale=1.0, # 缩放比例
 )
 scene.add_actor(table)
 print(f"✅ 已添加: {table.name}")

 # --- 3b. 添加一个机械臂 ---
 robot = Actor(
 name="robot_arm",
 asset_path="robots/franka_panda/panda_arm", # 法兰卡机械臂
 position=np.array([0.0, 0.0, 0.8]), # 放在桌子上方
 rotation=np.array([1.0, 0.0, 0.0, 0.0]), # 单位四元数 = 不旋转
 scale=1.0,
 )
 scene.add_actor(robot)
 print(f"✅ 已添加: {robot.name}")

 # --- 3c. 添加一个要操作的方块 ---
 cube = Actor(
 name="target_cube",
 asset_path="assets/blocks/red_cube_5cm",
 position=np.array([0.5, 0.2, 0.82]), # 放在桌子上
 rotation=np.array([1.0, 0.0, 0.0, 0.0]),
 scale=1.0,
 )
 scene.add_actor(cube)
 print(f"✅ 已添加: {cube.name}")

 # ============================================================
 # 4. 设置灯光
 # ============================================================
 light = LightInfo(
 color=np.array([1.0, 1.0, 1.0]), # RGB 白光
 intensity=2.0, # 亮度
 )
 scene.set_light_info("light_main", light)
 print("✅ 灯光已设置")

 # ============================================================
 # 5. 可选：设置材质颜色
 # ============================================================
 # 把方块变成蓝色
 blue_material = MaterialInfo(
 base_color=np.array([0.2, 0.4, 0.9, 1.0]), # RGBA
 )
 scene.set_material_info("target_cube", blue_material)
 print("✅ 方块材质已修改为蓝色")

 # ============================================================
 # 6. 关闭场景管理器
 # ============================================================
 scene.close()
 print("\n🎉 场景搭建完成！现在可以用 gym.make() 加载这个场景了。")


if __name__ == "__main__":
 build_scene()
```

---

## 关键 API 详解

### Actor — 场景中的一切物体

```python
Actor(
 name="唯一的名字", # 场景内不能重名
 asset_path="资产路径", # 对应 OrcaStudio 中导入的资产
 position=np.array([x, y, z]), # 世界坐标位置（米）
 rotation=np.array([w, x, y, z]), # 四元数旋转
 scale=1.0, # 缩放（1.0 = 原始大小）
)
```

### 四元数入门

四元数用 4 个数表示 3D 旋转，格式 `[w, x, y, z]`：

```python
# 几个常用的旋转
no_rotation = np.array([1.0, 0.0, 0.0, 0.0]) # 不旋转
rotate_z_90 = np.array([0.707, 0.0, 0.0, 0.707]) # 绕 Z 轴 90°
rotate_y_180 = np.array([0.0, 0.0, 1.0, 0.0]) # 绕 Y 轴 180°

# 从欧拉角生成四元数
from scipy.spatial.transform import Rotation as R
quat = R.from_euler('xyz', [0, 0, 1.57]).as_quat() # [x, y, z, w]
quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]]) # 转 [w, x, y, z]
```

### 操作顺序

```
1. OrcaGymScene(grpc_addr) ← 创建场景管理器
2. scene.publish_scene() ← 发布空场景（清空已有内容）
3. scene.add_actor(...) ← 逐个添加物体（仅入队，不立即生成）
4. scene.append_scene() ← 增量发布：spawn 新增 Actor，保留已有实体
   （首次发布也可调用 publish_scene()，二者择一）
5. scene.set_light_info(...) ← 设置灯光（可选）
6. scene.set_material_info() ← 修改材质（可选）
7. scene.close() ← 关闭连接
```

> ⚠️ **`publish_scene()` 会清空场景！**
> `publish_scene()` 会清空当前场景中的所有实体后重新发布，适用于初始化或完全重置场景。
> 若运行时只想追加 Actor 而不销毁已有实体，请改用 `append_scene()`：
> 它仅 spawn `add_actor()` 新增的 Actor，保留前序已 spawn 的实体（如运行中的机器人），
> 适用于时序 spawn 场景（逐步 `add_actor` + `append_scene`）。

---

## 获取资产

场景中的 Actor 需要指定 `asset_path`，这些资产需先获取并导入到 OrcaStudio / OrcaLab 中。主要流程如下：

1. **搜索资产**：前往 [SimAssets 资产商店](https://simassets.orca3d.cn/?tab=scene&sort=name) 搜索所需的相关资产。
2. **进入资产包**：转到对应的资产包页面。
3. **获取资产**：
   - **OrcaLab 用户**：在资产包页面直接订阅。
   - **OrcaStudio 用户**：下载原始资产，解压到项目目录中。
4. **在编辑器中定位资产**：打开 OrcaLab / OrcaStudio，在资产搜索器中找到对应资产，即可获取其 `asset_path`。

> 运行时动态添加 Actor 的示例详见 [OrcaPlayground - scene_building](https://github.com/openverse-orca/OrcaPlayground/tree/main/examples/scene_building)。

---

## 常用资产路径参考

以下是一些示例资产路径（具体路径取决于 OrcaStudio 中导入的资源）：

| 类别 | 示例路径 |
|------|----------|
| 机械臂 | `robots/franka_panda/panda_arm` |
| 机械手 | `robots/franka_panda/panda_hand` |
| 桌子 | `assets/tables/table_80x80` |
| 方块 | `assets/blocks/red_cube_5cm` |
| 球体 | `assets/balls/tennis_ball` |
| 地面 | `assets/floors/checker_floor` |

> 实际可用的资产路径取决于你在 OrcaStudio 中导入的资源。请联系你的 OrcaStudio 管理员获取完整列表。

---

## 实操练习

### 练习 1：搭建一个桌面场景

在桌子 (`0.5, 0.0, 0.0`) 上放置 3 个不同颜色的方块，间距 10cm。

### 练习 2：调整物体姿态

把一个方块绕 Z 轴旋转 45° 放置。

提示：
```python
from scipy.spatial.transform import Rotation as R
quat = R.from_euler('z', 45, degrees=True).as_quat()
rotation = np.array([quat[3], quat[0], quat[1], quat[2]])
```

---

## 下一步

场景搭好了，接下来学习如何**写一个环境类**来控制这个场景：[🏗️ 第一个环境](your-first-env.md)。
