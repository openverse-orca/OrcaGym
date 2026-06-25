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

---

## P3：Studio 集成与端到端联调

### 交付物

| 文件 | 说明 |
|------|------|
| `orca_gym/core/euler/orca_studio_bridge.py` | `OrcaStudioBridge`，gRPC 集成，依赖反转（`render(qpos, time)` 不访问 `_mjData`），支持离线模式 |
| `orca_gym/core/euler/orca_gym_euler.py` | 更新 `OrcaGymEuler`，组合 `OrcaStudioBridge`/`ModelRegistry`/`SimConfig`/`OrcaGymDataView`，异步 `init_simulation`/`render`/`pause_simulation` |
| `orca_gym/core/euler/__init__.py` | 导出 `OrcaStudioBridge` |
| `orca_gym/environment/orca_gym_euler_env.py` | `OrcaGymEulerEnv` 骨架，继承 `OrcaGymBaseEnv`，组合 `OrcaGymEuler`，提供 Gymnasium 接口 |
| `tests/orca_gym/core/euler/test_orca_studio_bridge.py` | `OrcaStudioBridge` 单元测试（12 用例） |
| `tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py` | `OrcaGymEulerEnv` 骨架单元测试（9 用例） |
| `OrcaPlayground/envs/euler/simple_env.py` | P3 最小联调 Env（`SimpleEulerEnv`），继承 `OrcaGymEulerEnv` |
| `OrcaPlayground/envs/euler/scenes/simple_pendulum.xml` | 联调场景 MJCF |
| `OrcaPlayground/examples/euler/01_hello_euler/hello_euler.py` | 第 1 课：P3 端到端联调入口脚本 |

### `tests/orca_gym/core/euler/test_orca_studio_bridge.py`

| 测试用例 | 验证点 | 结果 |
|---------|--------|------|
| `test_init_with_none_stub` | `stub=None` 时 `is_offline` 为 True | PASS |
| `test_init_with_stub` | `stub` 非 None 时 `is_offline` 为 False | PASS |
| `test_offline_render_skips_grpc` | 离线模式 `render` 安全跳过 | PASS |
| `test_offline_pause_skips_grpc` | 离线模式 `pause_simulation` 安全跳过 | PASS |
| `test_offline_video_methods_skip_grpc` | 离线模式视频方法跳过，`get_current_frame` 返回 -1 | PASS |
| `test_offline_manipulation_returns_defaults` | 离线模式操作方法返回默认值 | PASS |
| `test_offline_load_model_xml_with_local_path` | 离线模式配置 `local_xml_path` 后返回该路径 | PASS |
| `test_offline_load_model_xml_without_path_raises` | 离线模式未配置路径时抛出 `RuntimeError` | PASS |
| `test_render_passes_qpos_and_time` | 在线模式 `render(qpos, time)` 调用 `stub.UpdateLocalEnv` | PASS |
| `test_pause_simulation_calls_stub` | 在线模式 `pause_simulation` 调用 `stub.SetSimulationState` | PASS |
| `test_configure_offline_sets_paths` | `configure_offline` 正确设置本地 XML 路径和资源目录 | PASS |
| `test_override_ctrls_property` | `override_ctrls` 属性返回字典 | PASS |

### `tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py`

| 测试用例 | 验证点 | 结果 |
|---------|--------|------|
| `test_env_initializes_without_grpc` | 离线模式可成功初始化 | PASS |
| `test_data_returns_orca_gym_data_view` | `env.data` 返回 `OrcaGymDataView` 实例 | PASS |
| `test_model_returns_orca_gym_model` | `env.model` 返回 `OrcaGymModel` 实例 | PASS |
| `test_sim_config_returns_sim_config` | `env.sim_config` 返回 `SimConfig` 实例 | PASS |
| `test_init_qpos_qvel_populated` | `init_qpos`/`init_qvel` 形状正确 | PASS |
| `test_step_advances_time` | `step` 后仿真时间推进 `dt = time_step * frame_skip` | PASS |
| `test_reset_returns_obs_and_info` | `reset` 返回 `(obs, info)` 元组 | PASS |
| `test_dt_property` | `env.dt` 等于 `time_step * frame_skip` | PASS |
| `test_do_simulation_validates_action_dim` | 错误维度 `ctrl` 抛出 `ValueError` | PASS |

### 端到端联调验证

通过 `OrcaPlayground/examples/euler/01_hello_euler/hello_euler.py`（第 1 课）驱动 `SimpleEulerEnv`，验证完整链路：

```bash
cd <OrcaPlayground-root>
<conda-base>/envs/orca/bin/python examples/euler/01_hello_euler/hello_euler.py --steps 150
```

验证点：
1. 模型加载（`init_simulation` 从本地 XML 加载）
2. 状态访问（`env.data.qpos` / `env.data.qvel`）
3. 求解器配置（`env.sim_config.timestep` / `env.sim_config.integrator`）
4. 步进（`do_simulation` → `mj_step` → `sync_to_view`）
5. 渲染（离线模式跳过 gRPC）
6. reset（`reset_model` 恢复初始状态）

运行结果：
```
============================================================
P3 Euler 端到端联调
  模式: 离线
  步数: 150
============================================================
[1/6] 环境创建成功: nq=1, nv=1, nu=1
[2/6] 状态访问: qpos.shape=(1,), time=0.0000
[3/6] 求解器配置: timestep=0.002, integrator=1
[4/6] reset 成功: obs keys=['qpos', 'qvel']
[5/6] step 50/150: qpos=[1.6655518], reward=-0.0946, time=0.5000
[5/6] step 100/150: qpos=[3.0161006], reward=-0.9921, time=1.0000
[5/6] step 150/150: qpos=[1.229366], reward=0.3348, time=1.5000
[5/6] 步进完成: 总奖励=-33.3655
[6/6] 环境关闭成功
============================================================
P3 端到端联调验证通过
============================================================
```

### 验收标准核对

- [x] 所有 P3 单元测试通过（21/21）
- [x] `OrcaStudioBridge` 支持离线模式（`stub=None` 时所有方法安全跳过）
- [x] `OrcaStudioBridge.render(qpos, sim_time)` 依赖反转，不访问 `_mjData`
- [x] `OrcaGymEuler` 组合 `OrcaStudioBridge`/`ModelRegistry`/`SimConfig`/`OrcaGymDataView`
- [x] `OrcaGymEulerEnv` 骨架可实例化，提供 Gymnasium 接口
- [x] `env.data` 返回 `OrcaGymDataView`，`env.model` 返回 `OrcaGymModel`，`env.sim_config` 返回 `SimConfig`
- [x] 端到端联调通过（模型加载 → 步进 → 状态同步 → 渲染 → reset）

### 运行结果

```
Ran 54 tests in 0.049s

OK
```

> P1（13）+ P2（20）+ P3（21）= 54 用例全量通过。

---

## P3A：在线模式端到端渲染循环

### 交付物

| 文件 | 说明 |
|------|------|
| `orca_gym/core/euler/orca_gym_euler.py` | `set_ctrl` 应用 `override_ctrls`（Studio UI 手动控制生效） |
| `orca_gym/environment/orca_gym_euler_env.py` | 增加 `render_mode`/`sync_render` 参数；`render` 支持 `sync_render` 与 `do_body_manipulation` |
| `OrcaPlayground/examples/euler/02_online_render/online_render.py` | 第 2 课：在线渲染入口，支持 `--sync-render` 等命令行参数 |
| `tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py` | 新增 `TestOrcaGymEulerEnvP3A`（7 用例） |

### `tests/orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py`（P3A 新增）

| 测试用例 | 验证点 | 结果 |
|---------|--------|------|
| `test_render_mode_default_human` | 默认 `render_mode` 为 "human" | PASS |
| `test_render_mode_none` | `render_mode="none"` 时 `render()` 不抛异常 | PASS |
| `test_sync_render_default_false` | 默认 `sync_render` 为 False | PASS |
| `test_sync_render_true` | `sync_render=True` 时属性正确 | PASS |
| `test_render_skips_in_offline_mode` | 离线模式 `render()` 直接返回 | PASS |
| `test_set_ctrl_applies_override_ctrls` | `set_ctrl` 应用 `override_ctrls` | PASS |
| `test_set_ctrl_override_does_not_mutate_input` | `set_ctrl` 不修改输入数组 | PASS |

### 在线模式使用方式

```bash
# 在线模式（需 OrcaStudio gRPC 服务器运行）
cd <OrcaPlayground-root>
<conda-base>/envs/orca/bin/python examples/euler/02_online_render/online_render.py \
    --addr <ip:port> --steps 200 --render-mode human

# 同步渲染模式（每个物理步都渲染）
<conda-base>/envs/orca/bin/python examples/euler/02_online_render/online_render.py \
    --addr <ip:port> --sync-render --steps 200
```

### 验收标准核对

- [x] `OrcaGymEuler.set_ctrl` 正确应用 `override_ctrls`
- [x] `OrcaGymEulerEnv` 支持 `render_mode`/`sync_render` 参数
- [x] `OrcaGymEulerEnv.render` 支持 `sync_render` 与 `do_body_manipulation`（占位）
- [x] 所有 P3A 单元测试通过（7/7）
- [ ] `02_online_render/online_render.py` 可完成完整在线循环（需 OrcaStudio 环境，待用户验证）
- [ ] Studio 视口显示与仿真状态同步（待用户验证）
- [ ] 用户可通过 Studio UI 交互（拖拽、手动控制）（`do_body_manipulation` 完整实现在 P4）

### 运行结果

```
Ran 61 tests in 0.057s

OK
```

> P1（13）+ P2（20）+ P3（21）+ P3A（7）= 61 用例全量通过。
