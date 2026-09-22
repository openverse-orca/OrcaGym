# 第三方传感器接入

本接入口使用已经组装完成的 MuJoCo XML。OrcaGym 绑定已有 site、采集物理量、调用厂商算法，
并返回 NumPy 结果；模型装配、传感器安装和可视化由场景或应用工具完成。

## 准备运行文件

OrcaGym 在 Linux x86_64 / glibc 2.35+ 平台随包提供 Sensor Host、契约工具及其必要依赖，
对应 SDK 1.0.0、Provider/Host ABI 2，无需另外安装 Runtime、厂商 SDK 或 C++ 编译器。

应用只需提供可信厂商包的 `provider.json` 清单。只加载可信包：算法动态库在仿真进程内执行。
如需覆盖随包运行组件，可显式设置 `sensor_host_path` 和绝对路径环境变量 `ORCA_SENSOR_TOOL`；
自定义契约工具须保留其配套依赖，并与 Host、厂商包兼容。
契约工具仅在注册、准备阶段运行，每个物理子步由 Host 直接调用算法。

## 已有场景中的实例声明

场景应已包含机器人、传感器几何和测量 site。以下元数据将一个型号绑定到已有的 `left_tip`：

```xml
<custom>
  <text name="orca.sensor.v1/left_touch/plugin" data="com.orca.examples.contact_grid"/>
  <tuple name="orca.sensor.v1/left_touch/site">
    <element objtype="site" objname="left_tip"/>
  </tuple>
  <numeric name="orca.sensor.v1/left_touch/config/gain" data="1"/>
</custom>
```

`left_touch` 是结果查询使用的实例名；`plugin` 对应厂商包的型号；`config` 是该型号支持的标量参数。
多个独立实例可以绑定同一个 site。几何可通过 `include` 引入，Orca 实例声明须位于主 XML。
XML 不指定动态库路径，也不会在未配置厂商包时触发算法执行。

当前支持 site 接触力网格、site 射线距离和采样 time/dt/index 输入；site 所属刚性焊接组决定
接触采集范围和射线自体排除。网格无接触时为零，射线未命中为 -1。
TouchGrid、SevenPad 等对象契约通过精确映射读取已装配触面。以下声明把厂商的局部角色
绑定到已有 `left_pad` body 和 `left_frame` site，不安装或修改它们：

```xml
<custom>
  <text name="orca.sensor.v1/left_touch/plugin" data="com.orca.examples.touch_grid"/>
  <tuple name="orca.sensor.v1/left_touch/object/force_f1">
    <element objtype="body" objname="left_pad"/>
  </tuple>
  <tuple name="orca.sensor.v1/left_touch/object/surface_frame">
    <element objtype="site" objname="left_frame"/>
  </tuple>
  <text name="orca.sensor.v1/left_touch/seed" data="42"/>
</custom>
```

每个实例使用 `site` 或 `object/<alias>` 两种绑定方式之一。对象契约读取厂商 `model.xml`
中的局部角色类型；场景须完整映射这些角色。接触 body 只采集直属 geom，帧须与触面刚性连接。
射线的自身排除范围默认由显式 body/geom 映射确定；需要补齐外壳时可用 `geoms` tuple 列出
完整的精确 geom 名称，不自动排除整只手。接触行容量不足会明确失败，不静默丢掉接触。
厂商展示资料不参与运行；不带对象契约的 site 插件也无需读取厂商模型资料。

## 在 EulerEnv 中启用

在任务环境的 `super().__init__()` 中传入厂商包清单，默认使用随包 Host：

```python
super().__init__(
    frame_skip=4,
    orcagym_addr="localhost:50051",
    agent_names=[],
    time_step=0.001,
    model_xml_path="/path/to/assembled_scene.xml",
    skip_grpc_load=True,
    render_mode="none",
    sensor_provider_manifests=["/path/to/trusted/provider.json"],
)
```

该接入口使用 EulerEnv 的 CPU MuJoCo 后端；GPU 后端和 RK4 积分器不支持此采样路径。
不配置厂商包和 Host 时，环境沿用普通物理仿真路径，不加载传感器运行组件。

## 步进与读取

```python
self.do_simulation(action, self.frame_skip)
values = self.query_provider_sensor_data(["left_touch"])
grid = values["left_touch"]
```

每个物理子步采样并计算一次，整个动作成功后发布最后一个子步的结果。查询返回独立数组副本，
不会重新采样；`query_sensor_data()` 仍只读取 MuJoCo 原生传感器。
任务自行决定如何将结果加入 observation，不改变现有 reward 或 step 签名。

首次有效步进前和 reset 后结果未就绪。采样或算法失败会使整批结果失效，需 reset 后恢复；
已推进的物理状态不会回滚。seed/reset、模型重载和 close 同步管理算法实例生命周期。
接触力和 site 位姿使用子步源状态；结果对应的采样时刻与已积分的 qpos/qvel 不同。
