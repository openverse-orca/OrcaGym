# 🎮 机器人控制

OrcaGym 提供丰富的机器人控制接口，从底层的关节控制到高层的逆运动学。

> 完整可运行代码见 [OrcaPlayground examples/euler/](https://github.com/OrcaGym/OrcaPlayground)（Lesson 4-9）。

## 控制层级

```
高层 API
  ├── 阻尼最小二乘 IK      ── 末端位姿 → 关节位置（含限位 clamp）
  ├── PD 控制器            ── 目标位置 → 控制力矩
  ├── ONNX 行走策略        ── 观测 → 目标关节角度（G1 行走）
  └── Mocap 控制           ── 直接设置 mocap 位姿 + WELD 约束驱动

底层 API
  ├── do_simulation()      ── 设置 ctrl + 步进 + 自动同步（推荐）
  ├── set_joint_qpos()     ── 设置关节位置
  ├── apply_body_force()   ── 施加外力
  └── mj_forward()         ── 刷新派生量
```

## 章节导航

- [📡 状态查询 API](state-queries-api.md) — 关节、Body、传感器、接触查询全套 API
- [🎯 动作空间](action-space.md) — 动作空间定义与类型
- [👁️ 观测空间](observation-space.md) — 观测构建与归一化
- [🦿 关节控制](joint-control.md) — set_ctrl、PD 控制器、低通滤波
- [🦾 逆运动学](inverse-kinematics.md) — 阻尼最小二乘 IK + 关节限位
- [🎭 Mocap 控制](mocap-control.md) — Mocap body + WELD 约束 = 物体拖拽
- [🔄 外力应用与 IK](../physics/force-apply.md) — 外力施加 + 雅可比 + IK 完整工作流
