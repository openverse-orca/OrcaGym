# Teleoperation 模式下生成目标物体的流程分析

## 概述

本文档详细分析了在 teleoperation（遥操作）模式下，脚本执行时生成目标物体的完整流程。

## 执行流程概览

```
teleoperation_episode()
  └─> env._task.spawn_scene(env)          # 步骤1: 初始化场景
  └─> env.reset(seed=42)                  # 步骤2: 重置环境
      └─> reset_model()
          └─> self._task.spawn_scene(self) # 步骤3: 发布场景
          └─> self._task.get_task(self)    # 步骤4: 生成任务和目标物体
              └─> _get_teleperation_task_()
                  └─> generate_object()    # 步骤5: 生成物体列表
                  └─> random_objs_and_goals() # 步骤6: 随机摆放物体
                  └─> random.choice()     # 步骤7: 选择目标物体
      └─> reset_teleoperation()
          └─> safe_get_task()             # 步骤8: 验证任务有效性
          └─> get_language_instruction()   # 步骤9: 生成语言指令
          └─> update_objects_goals()       # 步骤10: 更新物体和目标信息
```

## 详细流程分析

### 步骤1: teleoperation_episode 开始执行

**位置**: `orca_gym/scripts/dual_arm_manipulation.py:327`

```python
def teleoperation_episode(env : DualArmEnv, cameras : list[CameraWrapper], 
                         dataset_writer : DatasetWriter, ...):
    env._task.spawn_scene(env)  # 初始化场景
    obs, info = env.reset(seed=42)
    # ...
```

**作用**: 开始一次遥操作 episode，首先初始化场景。

---

### 步骤2: spawn_scene - 场景初始化

**位置**: `orca_gym/scripts/dual_arm_manipulation.py:780`

```python
def spawn_scene(env: DualArmEnv, task_config: Dict[str, Any]) -> None:
    env._task.load_config(task_config)
    
    # 初次初始化，什么都没有，先把object创建出来
    if not all_object_joints_exist(task_config, env.model):
        env._task.publish_scene()   # 清空当前缓存
        env._task.generate_actors(random_actor=False)  # 生成所有actors
        env._task.publish_scene()
        time.sleep(1)  # 等待场景加载完成
```

**关键操作**:
1. **load_config**: 加载任务配置（从 task_config.yaml）
2. **检查关节存在性**: 如果配置中的 object_joints 不存在于模型中
3. **generate_actors**: 生成所有 actors（物体模板），初始位置在 infinity（无限远处）

**generate_actors 实现** (`orca_gym/adapters/robomimic/task/abstract_task.py:353`):
```python
def generate_actors(self):
    '''将所有的 actors 添加到场景中, 初始化到infinity位置'''
    for i in range(len(self.actors)):
        self.scene.add_actor(
            actor_name=self.actors[i], 
            asset_path=self.actors_spawnable[i],
            position=np.array(self.infinity), 
            rotation=rotations.euler2quat([0, 0, 0])
        )
```

**说明**: 此时所有 actors 都在 infinity 位置，还没有被选中和摆放。

---

### 步骤3: reset_model - 环境重置

**位置**: `envs/manipulation/dual_arm_env.py:506`

```python
def reset_model(self) -> tuple[dict, dict]:
    self._task.spawn_scene(self)      # 再次确保场景已发布
    self._task.get_task(self)        # 生成任务和目标物体
    
    if self._run_mode == RunMode.TELEOPERATION:
        return self.reset_teleoperation()
```

**关键**: 调用 `get_task` 生成任务，这是目标物体生成的核心步骤。

---

### 步骤4: get_task - 生成任务

**位置**: `orca_gym/adapters/robomimic/task/pick_place_task.py:24`

```python
def get_task(self, env: OrcaGymLocalEnv):
    is_augmentation_mode = (...)
    if is_augmentation_mode:
        self._get_augmentation_task_(env, env._task.data, ...)
    else:
        self._get_teleperation_task_(env)  # teleoperation 模式走这里
```

---

### 步骤5: _get_teleperation_task_ - 生成遥操作任务

**位置**: `orca_gym/adapters/robomimic/task/pick_place_task.py:37`

```python
def _get_teleperation_task_(self, env: OrcaGymLocalEnv):
    """
    随机选一个 object_bodys 里的物体，配对到第一个 goal_bodys。
    """
    # 5.1 生成物体列表（从 actors 中随机选择 3-5 个）
    self.generate_object(env, 3, 5)
    
    while True:
        # 5.2 随机摆放物体和目标位置
        self.random_objs_and_goals(env, random_rotation=True)
        
        # 5.3 检查是否有可用物体和目标
        if not self.object_bodys or not self.goal_bodys:
            return None
        
        # 5.4 从 object_bodys 随机选一个作为目标物体
        self.target_object = random.choice(self.object_bodys)
        
        # 5.5 只取第一个 goal
        goal_name = self.goal_bodys[0]
        
        return self.target_object
```

#### 5.1 generate_object - 生成物体列表

**位置**: `orca_gym/adapters/robomimic/task/abstract_task.py:149`

```python
def generate_object(self, env: OrcaGymLocalEnv, pick_min, pick_max):
    '''从场景中随机挑选actors里的几个'''
    if self.random_actor:
        if self.__random_count__ % self.random_cycle == 0:
            # 将object_bodys放回到无限远处
            pos = self.infinity + [1, 0, 0, 0]
            qpos_dict = {env.joint(joint_name): pos 
                        for joint_name in self.object_joints}
            env.set_joint_qpos(qpos_dict)
            
            # 随机选择 3-5 个 actors
            n_select = random.randint(pick_min, pick_max)  # 3-5
            idxs = random.sample(range(len(self.actors)), k=n_select)
            
            # 更新 object_bodys, object_sites, object_joints
            self.object_bodys = [self.actors[idx] for idx in idxs]
            self.object_sites = [f"{self.actors[idx]}site" for idx in idxs]
            self.object_joints = [f"{self.actors[idx]}_joint" for idx in idxs]
        
        self.__random_count__ += 1
```

**作用**:
- 如果 `random_actor=True`，每隔 `random_cycle` 次（默认20次）重新选择物体
- 从所有 `actors` 中随机选择 3-5 个物体
- 更新 `object_bodys`、`object_sites`、`object_joints` 列表

**注意**: 如果 `random_actor=False`，则使用配置文件中固定的 `object_bodys`。

#### 5.2 random_objs_and_goals - 随机摆放物体和目标

**位置**: `orca_gym/adapters/robomimic/task/abstract_task.py:247`

**核心逻辑**:
1. 获取所有 object 和 goal 的关节名称
2. 根据配置的 `center` 和 `bound` 随机生成位置
3. 检查碰撞，确保物体不重叠
4. 设置物体的位置和旋转

**关键代码片段**:
```python
def random_objs_and_goals(self, env, random_rotation=True, target_obj_joint_name=None):
    # 获取所有物体和目标的关节
    obj_joints = [env.joint(jn) for jn in self.object_joints]
    goal_joints = [env.joint(jn) for jn in self.goal_joints]
    
    # 随机生成位置和旋转
    # ... 碰撞检测 ...
    # ... 设置位置 ...
    
    # 一次性写回所有位置
    qpos_dict = {jn: q for jn, q in placed}
    env.set_joint_qpos(qpos_dict)
    env.mj_forward()
    
    # 保存随机化后的位置
    self.randomized_object_positions = env.query_joint_qpos(obj_joints)
    self.randomized_goal_positions = env.query_joint_qpos(goal_joints)
```

#### 5.3 选择目标物体

```python
self.target_object = random.choice(self.object_bodys)
```

**作用**: 从已生成的 `object_bodys` 列表中随机选择一个作为目标物体。

---

### 步骤6: safe_get_task - 验证任务有效性

**位置**: `envs/manipulation/dual_arm_env.py:418`

```python
def safe_get_task(self, env):
    iteration = 0
    max_iterations = 100
    while True:
        iteration += 1
        if iteration > max_iterations:
            break
        
        # 1) 生成一次 task（包括随机摆放）
        self._task.get_task(env)
        
        # 2) 检查目标物体是否在 goal 区域内
        objs = self._task.randomized_object_positions
        if not objs or not self._task.goal_bodys:
            break
        
        goal_body = self._task.goal_bodys[0]
        
        # 3) 获取 goal 的包围盒
        bbox = self.get_goal_bounding_box(goal_body)
        min_xy = bbox['min'][:2]
        max_xy = bbox['max'][:2]
        
        # 4) 检查是否有物体在 goal 区域内（这是不允许的）
        bad = False
        for joint_name, qpos in objs.items():
            obj_xy = qpos[:2]
            if (min_xy <= obj_xy).all() and (obj_xy <= max_xy).all():
                bad = True  # 物体已经在目标区域内，需要重新生成
                break
        
        if not bad:
            return  # 成功退出
```

**作用**: 确保生成的物体不在目标区域内，避免任务一开始就完成。

---

### 步骤7: get_language_instruction - 生成语言指令

**位置**: `orca_gym/adapters/robomimic/task/pick_place_task.py:153`

```python
def get_language_instruction(self) -> str:
    if not self.target_object:
        return "Do something."
    obj_str = "object: " + self.target_object
    goal_str = "goal: " + self.goal_bodys[0]
    
    return f"level: {self.level_name}  {obj_str} to {goal_str}"
```

**输出示例**: `"level: shop  object: bottle_blue to goal: basket"`

---

### 步骤8: update_objects_goals - 更新物体和目标信息

**位置**: `envs/manipulation/dual_arm_env.py:517`

```python
def update_objects_goals(self, object_positions, goal_positions):
    # 构建 objects 结构化数组
    obj_dtype = np.dtype([
        ("joint_name",  "U100"),
        ("position",    "f4", 3),
        ("orientation", "f4", 4),
    ])
    self.objects = np.array(
        [(jn, pos[:3].tolist(), pos[3:].tolist())
         for jn, pos in object_positions.items()],
        dtype=obj_dtype,
    )
    
    # 构建 goals 结构化数组
    # ...
```

**作用**: 将随机化后的物体和目标位置转换为结构化数组，存储在 `env.objects` 和 `env.goals` 中。

---

## 配置参数说明

### task_config.yaml 中的关键配置

```yaml
# 物体相关配置
object_bodys: []        # 物体名称列表（如果 random_actor=False 则使用此列表）
object_sites: []        # 物体 site 名称列表
object_joints: []       # 物体关节名称列表

# Actor 相关配置（用于随机生成物体）
random_actor: false     # 是否随机选择 actors
random_cycle: 20        # 每隔多少次重新选择 actors
actors: []              # 所有可用的 actor 名称列表
actors_spawnable: []    # actor 的 prefab 路径列表

# 目标相关配置
goal_bodys: []          # 目标物体名称列表
goal_sites: []          # 目标 site 名称列表
goal_joints: []         # 目标关节名称列表

# 位置相关配置
center: [0, 0, 0]       # 物体生成的中心位置
bound: [[-1, 1], [-1, 1], [0, 2]]  # 物体生成的边界范围
infinity: [1000, 1000, 1]  # 无限远位置（用于隐藏物体）

# 场景相关配置
level_name: "shop"      # 场景名称
```

---

## 关键数据结构

### target_object
- **类型**: `str`
- **位置**: `env._task.target_object`
- **说明**: 存储当前任务的目标物体名称（例如: `"bottle_blue"`）

### object_bodys
- **类型**: `list[str]`
- **位置**: `env._task.object_bodys`
- **说明**: 当前场景中所有物体的名称列表

### randomized_object_positions
- **类型**: `dict[str, np.ndarray]`
- **位置**: `env._task.randomized_object_positions`
- **说明**: 物体关节名称到位置（7维：x, y, z, qw, qx, qy, qz）的映射

---

## 总结

在 teleoperation 模式下，目标物体的生成流程如下：

1. **场景初始化**: 通过 `spawn_scene` 和 `generate_actors` 将所有 actors 添加到场景（初始在 infinity 位置）
2. **物体选择**: 通过 `generate_object` 从 actors 中随机选择 3-5 个物体（如果 `random_actor=True`）
3. **位置随机化**: 通过 `random_objs_and_goals` 在指定区域内随机摆放物体和目标
4. **目标选择**: 从已选择的物体中随机选择一个作为 `target_object`
5. **任务验证**: 通过 `safe_get_task` 确保物体不在目标区域内
6. **信息更新**: 将位置信息更新到 `env.objects` 和 `env.goals`

整个流程确保了每次 episode 都有不同的物体配置和目标物体，增加了数据集的多样性。

