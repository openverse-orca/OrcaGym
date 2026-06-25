# OrcaGymEuler 测试用例归档

本文档按开发阶段归档 `OrcaGymEulerEnv + OrcaGymEuler` 的单元测试用例、验证点与运行结果。
阶段划分与验收标准参见 [orca_gym_euler_development.md](./orca_gym_euler_development.md)。

测试运行环境：`orca` conda 环境（MuJoCo 3.7.0，CPU 仿真，可在 sandbox 内运行）。

## 运行方式

```bash
cd <repo-root>
<conda-base>/envs/orca/bin/python -m unittest \
    tests.orca_gym.core.euler.test_mujoco_sim_core \
    tests.orca_gym.core.euler.test_orca_gym_euler -v
```

---

## P1：基础设施骨架

### 交付物

| 文件 | 说明 |
|------|------|
| `orca_gym/core/euler/mujoco_sim_core.py` | `MuJoCoSimCore`，持有 `_mjModel`/`_mjData`，提供步进/前向/控制/外力原子操作 |
| `orca_gym/core/euler/orca_gym_euler.py` | `OrcaGymEuler` Facade，组合 `MuJoCoSimCore`，`__getattr__`/`__dir__` 实现封装隔离 |
| `tests/orca_gym/core/euler/fixtures/test_scene.xml` | 单铰链摆 MJCF 测试场景（质心偏置，重力下 qacc 非零；含 motor 执行器） |

### `tests/orca_gym/core/euler/test_mujoco_sim_core.py`

| 测试用例 | 验证点 | 结果 |
|---------|--------|------|
| `test_init_simulation_loads_model` | 加载 MJCF，`_mjModel`/`_mjData` 非 None | PASS |
| `test_step_advances_time` | `step(1)` 后 `_mjData.time` 增加 timestep | PASS |
| `test_forward_updates_derived` | `forward()` 后 `qacc` 非零 | PASS |
| `test_set_ctrl_sets_actuator` | `set_ctrl` 后 `_mjData.ctrl` 与输入一致 | PASS |
| `test_apply_body_force_writes_xfrc` | `apply_body_force` 后 `xfrc_applied` 对应位置非零 | PASS |
| `test_clear_body_force_zeros_xfrc` | `clear_body_force` 后对应位置为零 | PASS |
| `test_clear_all_forces_zeros_all` | `clear_all_forces` 后 `xfrc_applied` 全零 | PASS |

### `tests/orca_gym/core/euler/test_orca_gym_euler.py`

| 测试用例 | 验证点 | 结果 |
|---------|--------|------|
| `test_blocked_attrs_raise_guidance_error` | 访问 `_mjData`/`_mjModel` 抛出 `AttributeError` 且消息含引导文本 | PASS |
| `test_dir_only_exposes_public_api` | `dir(gym)` 不含 `_mjData`/`_mjModel`/`_sim` | PASS |
| `test_init_simulation_delegates_to_sim_core` | `gym.init_simulation(path)` 后 sim_core 已加载 | PASS |
| `test_mj_step_delegates` | `gym.mj_step(1)` 后 time 推进 | PASS |
| `test_mj_forward_delegates` | `gym.mj_forward()` 不报错 | PASS |
| `test_set_ctrl_delegates` | `gym.set_ctrl(ctrl)` 后 sim_core 的 ctrl 一致 | PASS |

### 验收标准核对

- [x] 所有 P1 单元测试通过（13/13）
- [x] `MuJoCoSimCore` 能加载 MJCF 并执行 `mj_step`/`mj_forward`
- [x] `OrcaGymEuler` 访问 `_mjData` 抛出引导性错误
- [x] `dir(OrcaGymEuler())` 不含内部组件

### 运行结果

```
Ran 13 tests in 0.009s

OK
```
