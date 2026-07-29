# 🧤 软体与柔性体

OrcaGym 通过 MuJoCo 的 Flex 系统支持柔性体仿真。

## MuJoCo Flex

MuJoCo 3.0+ 引入了 Flex（柔性体）支持，OrcaGym 封装了相关接口。

## 模型查询

```python
# 通过 env.model.model_info 获取 flex 信息（OrcaGymModel 的公共属性）
model_info = env.model.model_info

# Flex 相关信息
nflex = model_info['nflex']           # 柔性体数量
nflexvert = model_info['nflexvert']   # 柔性体顶点总数
flex_vertbodyid = model_info['flex_vertbodyid']  # 顶点所属 body
flex_names = model_info['flex_names'] # 柔性体名称

if nflex > 0:
    print(f"模型包含 {nflex} 个柔性体: {flex_names}")
```

> ⚠️ **Euler 路径**：`OrcaGymEulerEnv` 没有 `_query_model_info()` 方法（该方法仅在 Local/Warp 体系）。Euler 路径通过 `env.model.model_info` 公共属性访问模型维度信息（`OrcaGymModel.init_model_info` 中赋值）。`env.model` 也直接暴露 `nq`/`nv`/`nu`/`ngeom` 等常用维度字段。

## 柔性体锚定

> ⚠️ **Euler 路径**：`OrcaGymEulerEnv` 没有 `anchor_actor` 公共方法，也没有 `_is_flex_vertex_anchored` 内部标记字段（这两者仅在 Local 体系 `OrcaGymLocalEnv` 中存在）。Euler 路径下操作柔性体应使用等式约束原语编排，与刚性 body 操作一致：

```python
# Euler 路径：用等式约束原语编排锚定 flex vertex
import mujoco

mocap_name = "ActorManipulator_Anchor"
flex_body_name = "flex_body_name"

slot = env.equality_find_slot_by_body(mocap_name)
original_eq = env.equality_constraint(slot)
# 对齐 mocap 到 flex vertex 位姿（通过 body_xpos 查询）
flex_pose = env.get_body_xpos_xmat_xquat([flex_body_name])[flex_body_name]
env.set_mocap_pos_and_quat({
    mocap_name: {"pos": flex_pose["xpos"], "quat": flex_pose["xquat"]}
})
env.equality_update(
    slot,
    eq_type=mujoco.mjtEq.mjEQ_WELD,
    obj1_name=mocap_name,
    obj2_name=flex_body_name,
)
```

> 📝 **Local 体系**：`OrcaGymLocalEnv` 提供 `env.anchor_actor(name, AnchorType.WELD)` 便捷封装，内部会检测目标是否为 flex vertex 并设置 `_is_flex_vertex_anchored` 标记。Euler 路径不提供此封装，flex vertex 与刚性 body 的锚定流程在 Euler 路径下统一走等式约束原语。

## 柔性体状态

```python
# 柔性体状态仍然通过 qpos/qvel 访问
# 每根柔性体在 qpos 中有对应维度
```

## 局限性

- 当前版本 flex 支持为实验性功能
- 柔性体操作依赖于 MuJoCo 3.0+ 的 flex 特性
- 建议在 OrcaStudio 中测试后再用于训练
