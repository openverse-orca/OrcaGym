## 在关卡中添加 Remy.prefab 
   - 引擎已经内置，在实例化预制体菜单项，直接联想就可以找到
   - 拖动 Remy. 到需要的位置
   - 如果更改了 Remy 的名称，需要修改 run_character.py 中的 agent_name

## 运行 run_character.py
   - 配置文件在 envs/character/character_config/remy.yaml
   - 配置文件中可以修改 control_type 更换控制方式，控制方式有：
     - keyboard: 键盘控制
        - A/W/S/D 控制角色移动，也可以配置为其他按键
     - waypoint: 路径点控制 （默认）
        - 可以修改路径点，路径点在 envs/character/character_config/remy.yaml 中，坐标为相对角色初始位置的坐标
        - 可以修改路径点之间的距离阈值，路径点之间的角度阈值，路径点站立时等待时间
