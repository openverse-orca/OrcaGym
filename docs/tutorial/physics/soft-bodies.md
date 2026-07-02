# 🧤 软体与柔性体

OrcaGym 通过 MuJoCo 的 Flex 系统支持柔性体仿真。

## MuJoCo Flex

MuJoCo 3.0+ 引入了 Flex（柔性体）支持，OrcaGym 封装了相关接口。

## 模型查询

```python
# 通过 model_info 获取 flex 信息
model_info = env._query_model_info()

# Flex 相关信息
nflex = model_info['nflex'] # 柔性体数量
nflexvert = model_info['nflexvert'] # 柔性体顶点总数
flex_vertbodyid = model_info['flex_vertbodyid'] # 顶点所属 body
flex_names = model_info['flex_names'] # 柔性体名称

if nflex > 0:
 print(f"模型包含 {nflex} 个柔性体: {flex_names}")
```

## 柔性体锚定

```python
# 锚定柔性体顶点（与刚性 body 操作类似）
env.anchor_actor("flex_body_name", AnchorType.WELD)

# 内部会检测是否锚定的是 flex vertex
# env._is_flex_vertex_anchored = True
```

## 柔性体状态

```python
# 柔性体状态仍然通过 qpos/qvel 访问
# 每根柔性体在 qpos 中有对应维度
```

## 局限性

- 当前版本 flex 支持为实验性功能
- 柔性体操作依赖于 MuJoCo 3.0+ 的 flex 特性
- 建议在 OrcaStudio 中测试后再用于训练
