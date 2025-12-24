# 首先生成 example 要用到的资产，运行

```bash
pip install usd-core
python ../../orca_gym/tools/assets/usdz_to_xml.py --config 你的OrcaGym_Assets/scene_assets/usdz
```

# 导入资产文件

- 对于OrcaStudio
  - 使用mujoco xml import功能导入生成的xml文件
  - 制作cameraviewport.prefab,包含camera和mujococamera即可，存放在Assets/Prefabs
  - 文件路径为 `你的OrcaGym_Assets/scene_assets/usdz/converted_files/` 下的 `.xml`

- 对于OrcaSim
  - 发布时已经集成了对应的xml文件，可以直接使用


# 运行示例
- 对于OrcaStudio
  - 新建一个mujoco仿真关卡
  - 点击运行按钮
  - 运行 run_actor.py 脚本
  - 会生成一张桌子和10个杯子

