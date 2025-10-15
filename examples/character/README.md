## 在关卡中添加 Remy.prefab 
   - 引擎已经内置，在实例化预制体菜单项，直接联想就可以找到
   - 拖动 Remy. 到需要的位置
   - 如果更改了 Remy 的名称，需要修改 run_character.py 中的 agent_name


## 运行 run_character.py
   - 配置文件在 envs/character/character_config/remy.yaml
   - 配置文件中可以修改 control_type 更换默认控制方式，控制方式有：
     - keyboard: 键盘控制
        - A/W/S/D 控制角色移动，也可以配置为其他按键
     - waypoint: 路径点控制 （默认）
        - 可以修改路径点，路径点在 envs/character/character_config/remy.yaml 中，坐标为相对角色初始位置的坐标
        - 可以修改路径点之间的距离阈值，路径点之间的角度阈值，路径点站立时等待时间
     - 通过按键切换控制方式，默认按键为 1 和 2，可以修改配置文件中的 switch_key 来更换按键

## 在自己的python程序中添加Remy
Character 需要用到 SenenRuntime 类，用来向Orca发送动画指令，因此需要在 env 中添加 SenenRuntime 类的回调入口
   - 参考 envs/character/character_env.py 中定义的set_scene_runtime方法，在你的env中添加一个set_scene_runtime方法。
   - 参考 run_character.py 中定义的run_simulation方法，在你的程序中调用 set_scene_runtime 方法，将SenenRuntime 类的实例传入。 **注意：** 这一步需要在env.reset()之前完成
   - 参考 envs/character/character_env.py， 
      - 在你的 env 中添加Remy的实例
      - 在 step() 函数调用 Character 的 on_step() 方法。
      - 在 reset() 函数调用 Character 的 on_reset() 方法。（在character_env.py 中是 reset_model）