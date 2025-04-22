from pxr import Usd

# 打开 Stage
stage = Usd.Stage.Open("/home/superfhwl/workspace/3d_assets/Cup_of_Coffee.usdz")


upAxis = stage.GetMetadata("upAxis")  # 获取 Stage 的 upAxis 元数据
print(f"Stage upAxis: {upAxis}")

# 遍历所有 Prim
def traverse_prims(prim):
    print(f"Prim: {prim.GetPath()}, Type: {prim.GetTypeName()}")
    for child in prim.GetChildren():
        traverse_prims(child)

traverse_prims(stage.Load("/"))  # 遍历 Stage 的根 Prim

# 关闭 Stage
stage = None