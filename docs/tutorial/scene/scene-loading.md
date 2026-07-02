# 🏞️ 场景加载

场景加载涉及从 XML 文件创建 MuJoCo 模型和初始化所有场景元素。

## 场景加载流程

```
1. gRPC: LoadLocalEnv → 获取模型 XML
2. 本地: 解析 XML → 下载 mesh/hfield 依赖
3. 本地: mujoco.MjModel.from_xml_path()
4. 本地: mujoco.MjData(model)
5. 本地: 查询并填充 Model / Opt / Data
6. 本地: 初始化所有字典（body, joint, actuator, ...）
```

## OrcaGymScene 工具

```python
from orca_scene import OrcaGymScene

# 连接场景
scene = OrcaGymScene("localhost:50051")

# 获取运行数据
scene.get_rundata(script_name="my_script", stage="beginscene")

# 显示 UI 文本
scene.set_ui_text(
 actor_name=1,
 message="仿真开始！",
 showtime=5,
 color="0xff0000",
 size=32,
)

scene.close()
```

## OrcaGymSceneRuntime

```python
from orca_scene import OrcaGymSceneRuntime

# 在环境中注入场景运行时
scene_runtime = OrcaGymSceneRuntime(...)
env.set_scene_runtime(scene_runtime)
```

## 模型 XML 资源

模型 XML 引用的 mesh 和 hfield 文件通过 gRPC 按需下载：

```
1. 获取 XML 文件内容
2. 解析 XML，查找 <mesh> 和 <hfield> 节点
3. 检查本地缓存 (~/.orcagym/tmp/)
4. 如果不存在，通过 gRPC 下载
5. 原子写入 + 文件锁避免冲突
```
