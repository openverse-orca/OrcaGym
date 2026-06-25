# OrcaGymEuler 测试用例归档

本文档按开发阶段归档 `OrcaGymEulerEnv + OrcaGymEuler` 的单元测试用例、验证点与运行结果。
阶段划分与验收标准参见 [orca_gym_euler_development.md](./orca_gym_euler_development.md)。

测试运行环境：`orca` conda 环境（MuJoCo 3.7.0，CPU 仿真，可在 sandbox 内运行）。

## 运行方式

```bash
cd <repo-root>
# 全量运行（P1 + P2）
<conda-base>/envs/orca/bin/python tests/run_tests.py -v

# 按组件运行
<conda-base>/envs/orca/bin/python tests/run_tests.py -c tests.orca_gym.core.euler.test_sim_config -v
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

---

## P2：状态视图与配置

### 交付物

| 文件 | 说明 |
|------|------|
| `orca_gym/core/euler/orca_gym_data_view.py` | `OrcaGymDataView`，MuJoCo 状态完整只读视图，替代直接访问 `_mjData` |
| `orca_gym/core/euler/sim_config.py` | `SimConfig`，封装 `_mjModel.opt` 求解器参数，提供 typed 读写接口 |
| `orca_gym/core/euler/model_registry.py` | `ModelRegistry`，构建 `OrcaGymModel`/`OrcaGymData`，提供扩展查询 |
| `orca_gym/core/euler/mujoco_sim_core.py` | 补充 `sync_to_view()` 实现，同步 `_mjData` 状态到 `OrcaGymDataView` |

### `tests/orca_gym/core/euler/test_orca_gym_data_view.py`

| 测试用例 | 验证点 | 结果 |
|---------|--------|------|
| `test_qpos_qvel_qacc_consistent_after_sync` | sync 后 `qpos`/`qvel`/`qacc`/`qfrc_bias` 与 `_mjData` 一致 | PASS |
| `test_body_xpos_by_name` | `body_xpos("world")` 返回正确位置 | PASS |
| `test_body_cvel_by_name` | `body_cvel("pendulum")` 返回正确速度 | PASS |
| `test_body_subtree_mass_by_name` | `body_subtree_mass("pendulum")` 返回正确质量 | PASS |
| `test_site_xpos_by_name` | `site_xpos("tip")` 返回正确位置 | PASS |
| `test_xfrc_applied_is_read_only_view` | `xfrc_applied` 与 `_mjData.xfrc_applied` 共享内存（只读视图） | PASS |
| `test_missing_field_raises_guidance` | 访问不存在的字段抛出引导性 `AttributeError` | PASS |
| `test_time_field` | `time` 字段与 `_mjData.time` 一致 | PASS |

### `tests/orca_gym/core/euler/test_sim_config.py`

| 测试用例 | 验证点 | 结果 |
|---------|--------|------|
| `test_timestep_get_set` | 读写 `timestep` 反映到 `_mjModel.opt.timestep` | PASS |
| `test_integrator_get_set` | 读写 `integrator` 反映到 `_mjModel.opt.integrator` | PASS |
| `test_iterations_get_set` | 读写 `iterations` 反映到 `_mjModel.opt.iterations` | PASS |
| `test_gravity_get_set` | 读写 `gravity` 反映到 `_mjModel.opt.gravity` | PASS |
| `test_load_from_dict` | `load_from_dict({...})` 批量设置多个参数 | PASS |
| `test_to_dict` | `to_dict()` 返回所有参数字典（含 `filterparent`） | PASS |
| `test_all_opt_fields_covered` | 遍历 `_mjModel.opt` 所有字段，`SimConfig` 均有对应属性 | PASS |

### `tests/orca_gym/core/euler/test_model_registry.py`

| 测试用例 | 验证点 | 结果 |
|---------|--------|------|
| `test_build_orca_gym_model` | `build_orca_gym_model()` 返回完整 `OrcaGymModel`，body/site 字典已填充 | PASS |
| `test_build_orca_gym_data` | `build_orca_gym_data()` 返回 `OrcaGymData`，`qpos`/`qvel` 形状正确 | PASS |
| `test_body_subtree_mass` | `body_subtree_mass("pendulum")` 返回正确质量 | PASS |
| `test_equality_data_width` | `equality_data_width()` 返回 `eq_data` 列数（无等式约束时为 0） | PASS |
| `test_joint_name_by_id` | `joint_name_by_id(0)` 返回正确关节名 | PASS |

### 验收标准核对

- [x] 所有 P2 单元测试通过（20/20）
- [x] `OrcaGymDataView` 覆盖基本状态字段和 body/site 属性查询
- [x] `SimConfig` 覆盖 `_mjModel.opt` 所有用户可访问字段
- [x] `ModelRegistry` 能构建完整 `OrcaGymModel`/`OrcaGymData`
- [x] `MuJoCoSimCore.sync_to_view()` 正确同步状态到 `OrcaGymDataView`
- [x] `xfrc_applied` 是只读视图（共享内存，不 copy）
- [x] 缺字段时 `__getattr__` 抛出引导性错误

### 运行结果

```
Ran 33 tests in 0.021s

OK
```
