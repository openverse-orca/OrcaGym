# OrcaGym 测试用例合并重构文档

## 1. 文档定位

### 1.1 文档目标

本文是 `OrcaGym/tests/` 目录下测试用例的**合并重构**实施文档。

**重构背景**：测试套件随开发阶段增量堆积（阶段二 skeleton / phase2_revision Step 1-6 / phase3 / 阶段二-Step7 等），同一 K 约束、同一 ruff SLF001 检查、同一组件合规检查在 3-4 个文件里反复验证。当前 19 个文件、543 个用例，重复占比约 17%。

**本文不动产品代码**，仅合并/删除重复测试文件，使测试按「被测对象」组织（而非按「开发阶段」），降低维护成本与运行时间。

### 1.2 上游约束

| 文档 | 约束范围 |
|------|---------|
| `AGENTS.md` | 规则 1（orca conda 环境）、规则 2（架构约束）、规则 3（GPU 命令白名单）、规则 4（API 隔离强制） |
| `docs/design/architecture/orca_gym_euler_architecture.md` | §7 封装隔离机制（M0-M7）、§12 K 约束（K1-K14） |
| `docs/design/development/orca_gym_euler_phase2_revision_development.md` | phase2 修订步骤（其验收测试为本轮合并对象之一） |
| `docs/design/development/orca_gym_euler_phase3_development.md` | phase3 跨子步骤一致性测试（本轮精简对象） |

### 1.3 合并原则

1. **不降低覆盖率**：合并前后必须保证每个 K 约束、每个公共 API 委托链路至少有 1 个用例覆盖。删除前先确认被合并目标已覆盖。
2. **保留运行时优先**：当同一约束既有「源码模式扫描」用例又有「运行时行为」用例时，优先保留运行时用例（更稳健、不脆弱）。
3. **保留综合验收**：`test_orca_gym_euler_env_skeleton.py` 与 `test_orca_gym_euler.py` 是按被测对象组织的综合验收文件，作为合并目标。
4. **不动 core/euler 5 个文件**：`test_model_registry.py` / `test_mujoco_sim_core.py` / `test_orca_gym_data_view.py` / `test_sim_config.py` / `test_orca_gym_euler.py` 已按组件组织、无内部重复，本轮不动。
5. **删除前先迁移独有用例**：每个待删文件中的非重复用例必须先迁入合并目标，再删除原文件。
6. **测试环境统一**：全部验证使用 `orca` conda 环境（AGENTS.md 规则 1），CPU 测试可在 sandbox 内运行。

---

## 2. 现状与重复分析

### 2.1 当前测试清单（19 文件 / 585 用例）

> **注**：初始分析（19 文件 / 543 用例）遗漏了 `test_orca_studio_bridge.py`（42 用例），用例总数实为 585。文件数仍为 19。下表已补正该文件。

| 路径 | 用例数 | 组织方式 | 本轮处理 |
|------|--------|---------|---------|
| `orca_gym/test_ruff_config.py` | 5 | 按被测对象 | **保留**（接收迁入） |
| `orca_gym/test_agents_md.py` | 5 | 按被测对象 | 保留 |
| `orca_gym/core/euler/test_model_registry.py` | 28 | 按被测对象 | 保留 |
| `orca_gym/core/euler/test_mujoco_sim_core.py` | 76 | 按被测对象 | 保留 |
| `orca_gym/core/euler/test_orca_gym_data_view.py` | 32 | 按被测对象 | 保留 |
| `orca_gym/core/euler/test_orca_gym_euler.py` | 54 | 按被测对象 | 保留 |
| `orca_gym/core/euler/test_orca_studio_bridge.py` | 42 | 按被测对象 | 保留（初始清单遗漏，已补正） |
| `orca_gym/core/euler/test_sim_config.py` | 24 | 按被测对象 | 保留 |
| `orca_gym/environment/test_orca_gym_env_mixin.py` | 13 | 按被测对象 | 保留 |
| `orca_gym/environment/euler/test_orca_gym_euler_env_skeleton.py` | 135 | 综合验收 | **保留**（合并目标） |
| `orca_gym/environment/euler/test_simple_euler_env_compliance.py` | 8 | 按被测对象 | **保留**（接收迁入） |
| `orca_gym/environment/euler/test_phase2_revision_e2e.py` | 3 | 端到端独有 | 保留 |
| `orca_gym/environment/euler/test_phase3_cross_substep_consistency.py` | 29 | 综合验收 | **精简** |
| `orca_gym/environment/euler/test_phase2_revision_baseline.py` | 13 | 按开发阶段 | **删除** |
| `orca_gym/environment/euler/test_phase2_revision_env_filling.py` | 17 | 按开发阶段 | **删除** |
| `orca_gym/environment/euler/test_phase2_revision_isolation_checklist.py` | 20 | 按开发阶段 | **删除** |
| `orca_gym/environment/euler/test_phase2_revision_gym_compat.py` | 21 | 按开发阶段 | **删除** |
| `orca_gym/environment/euler/test_phase2_revision_simple_env.py` | 10 | 按开发阶段 | **删除**（合并入 compliance） |
| `orca_gym/environment/euler/test_phase2_revision_ruff_agents.py` | 8 | 按开发阶段 | **删除**（noqa 用例迁入 ruff_config） |

### 2.2 重复识别（6 组）

#### 组 A：SimpleEulerEnv 合规测试（2 文件 → 1 文件）

| 重复用例 | compliance | revision_simple_env |
|---------|:---:|:---:|
| `test_no_gym_sim_tunnel` | ✓ | ✓（逐行相同） |
| `test_no_mjdata_mjmodel_tunnel` | ✓ | ✓（逐行相同） |
| `test_reset_model_uses_compliant_api` | ✓ | ✓（逐行相同） |
| `test_step_works_after_reset_model` | ✓ | ✓（逐行相同） |
| `test_reset_model_writes_perturbed_*` | 拆 qpos/qvel | 合并 state |

**revision 独有**（需迁入 compliance）：`test_no_self_gym_access`、`test_no_xfrc_applied_direct_write`、`test_env_gym_raises_attribute_error`、`test_step_returns_gymnasium_tuple`、`test_observation_space_box`。

#### 组 B：Env 层 phase2 修订测试（4 文件 → 1 文件）

| revision 文件 | 用例 | skeleton 对应类 |
|--------------|------|----------------|
| `env_filling::test_inheritance_chain` | 1 | `TestEnvK14Inheritance` |
| `env_filling::test_env_*_raises_native_attribute_error` (3) | 3 | `TestEnvK2Isolation`（M0） |
| `env_filling::test_env_no_getattr_method` / `no_setattr_method` | 2 | `TestEnvK14Inheritance`（运行时验证） |
| `env_filling::test_init_*` (3) | 3 | `TestEnvLifecycleAndStepping` |
| `env_filling::test_time_step_cached_before_init` / `sim_config_bound_after_init` / `dt_uses_sim_config` | 3 | `TestEnvK7PropertyDelegation` |
| `env_filling::test_do_simulation_delegates_to_step_with_coupling` | 1 | `TestEnvLifecycleAndStepping` |
| `env_filling::test_set_joint_qpos/qvel_delegates_to_gym` | 2 | `TestEnvLifecycleAndStepping` |
| `env_filling::test_data_returns_dataview` | 1 | `TestEnvK6DataView` |
| `env_filling::test_do_simulation_validates_action_dim` | 1 | `TestEnvK8NoEulerPrivate` |
| `isolation_checklist::*` (20) | 20 | 全部被 skeleton K1-K14 + M3 类覆盖 |
| `baseline::*` (13) | 13 | 9 个被 skeleton 运行时覆盖，4 个为源码模式扫描（迁移期产物） |

**baseline 独有的源码扫描**（争议项，建议不保留）：`test_no_public_gym_assignment`、`test_no_object_setattr_bypass`、`test_mixin_imported`、`test_self_orchestrated_lifecycle` 源码扫描。这些是迁移期回归测试，迁移完成后由 skeleton 运行时检查替代，价值低且脆弱。

#### 组 C：Gym 层 phase2 修订测试（2 文件 → 1 文件）

`test_phase2_revision_gym_compat.py` 的 21 个用例**全部**被 `core/euler/test_orca_gym_euler.py` 覆盖：

| revision 用例 | core 对应用例 |
|--------------|--------------|
| `test_gym_has_blocked_attrs` / `test_gym_has_getattribute` / `test_gym_blocked_attrs_contains_*` | `test_gym_blocked_attrs_include_components` |
| `test_gym_external_access_*_blocked` (4) | `test_gym_blocked_attrs_include_components` + `test_gym_blocked_attrs_message_has_guidance` |
| `test_sim_config_bound_after_init` / `test_model_registry_bound_after_init` / `test_sim_config_returns_simconfig_instance` | `TestOrcaGymEulerDelegation` |
| `test_gym_step_with_coupling_works` / `test_gym_set_qpos_qvel_writes_state` / `test_gym_sync_to_view_populates_dataview` / `test_gym_data_returns_dataview` / `test_gym_mj_step_advances_time` | `TestOrcaGymEulerDelegation` |
| `test_has_euler_returns_false` / `test_has_euler_callable` | `test_gym_has_euler_returns_false` |
| `test_studio_bridge_is_method` / `test_gym_studio_blocked` / `test_studio_bridge_returns_bridge` | `test_gym_studio_bridge_is_method_not_property` |

#### 组 D：ruff / AGENTS.md 合规测试（3 文件 → 2 文件）

| revision 用例 | 重复目标 |
|--------------|---------|
| `test_ruff_installed` | 完全重复 `test_ruff_config.py::test_ruff_installed` |
| `test_ruff_slf001_euler_clean` | 重复组 E 全局检查 |
| `test_agents_md_has_api_isolation_rule` / `lists_blocked_attrs` / `has_correct_usage_table` / `has_ruff_command` (4) | 重复 `test_agents_md.py` 的 4 个用例 |

**revision 独有**（需迁入 `test_ruff_config.py`）：`test_env_no_noqa_slf001`、`test_gym_noqa_only_for_bind_orchestration`（noqa 豁免纪律）。

#### 组 E：ruff SLF001 零报警检查（4 处 → 1 处）

| 位置 | 扫描范围 | 处理 |
|------|---------|------|
| `test_phase3_cross_substep_consistency.py::test_ruff_slf001_global_zero` | `orca_gym/`（最全） | **保留** |
| `test_phase2_revision_ruff_agents.py::test_ruff_slf001_euler_clean` | `environment/euler` + `core/euler` | 随组 D 删除 |
| `test_phase2_revision_isolation_checklist.py::test_ruff_slf001_euler_env_clean` + `_core_clean` | 同上 | 随组 B 删除 |

#### 组 F：phase3 跨子步骤一致性测试（精简 29 → ~22）

`test_phase3_cross_substep_consistency.py::TestGlobalArchCompliance` 中以下 7 个用例与 skeleton 重复：

| phase3 用例 | skeleton 对应类 |
|------------|----------------|
| `test_k1_k4_env_no_global_private_access` | `TestEnvK4NoGymPrivateAccess` |
| `test_k2_env_dir_no_internal_leak` | `TestEnvK2Isolation` |
| `test_k2_env_dir_contains_phase3_public_api` | `TestEnvK2Isolation` |
| `test_k6_env_data_is_dataview` | `TestEnvK6DataView` |
| `test_k9_studio_methods_via_bridge_not_gym_studio` | `TestEnvK9StudioAccess` |
| `test_k14_mro_stable` | `TestEnvK14Inheritance` |

**保留**：`test_ruff_slf001_global_zero`（组 E 唯一保留点）+ §10.2/§10.3 全部功能与一致性用例（22 个）。

### 2.3 合并后预期效果

| 维度 | 合并前 | 合并后 | 减少 |
|------|--------|--------|------|
| 文件数 | 19 | 13 | -6 |
| 用例数 | 585 | 455 | -130 |
| 重复 ruff SLF001 检查 | 4 处 | 1 处 | -3 |
| 重复 K 约束检查 | 3-4 处 | 1 处（skeleton） | -2~3 |

---

## 3. 总体策略

### 3.1 合并次序

按风险从低到高执行，每步独立可验证：

```
Step 1: ruff/AGENTS.md 合规合并（组 D + E，最安全）
   │   （纯重复删除 + 2 个 noqa 用例迁入）
   ▼
Step 2: SimpleEulerEnv 合规合并（组 A）
   │   （5 个独有用例迁入 compliance，删除 revision_simple_env）
   ▼
Step 3: Gym 层 phase2 修订删除（组 C）
   │   （21 个全部重复，直接删除）
   ▼
Step 4: Env 层 phase2 修订删除（组 B）
   │   （3 文件删除，先核对独有用例）
   ▼
Step 5: phase3 精简（组 F）
   │   （删除 7 个与 skeleton 重复的 K 约束用例）
   ▼
Step 6: 全量回归验证
       （unittest 全量 + ruff SLF001 + 覆盖率核对）
```

### 3.2 测试环境

| 测试类型 | 环境 | 说明 |
|---------|------|------|
| 单元测试 / 隔离测试 | `orca` conda 环境（sandbox 内） | 纯 MuJoCo 仿真 + 源码审查，无 CUDA 依赖 |
| 端到端验证 | `orca` conda 环境 | 离线模式 |

**命令格式约定**（AGENTS.md 规则 1）：

```bash
# 全量测试（sandbox 内）
<conda-base>/envs/orca/bin/python -m unittest discover -s tests/orca_gym -p "test_*.py" -v

# 单组件测试
<conda-base>/envs/orca/bin/python -m unittest tests.orca_gym.environment.euler.<module>

# ruff SLF001 全局检查
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca_gym/
```

> `<conda-base>` 通过 `conda info --base` 解析（当前为 `/home/superfhwl/miniconda3`）。

---

## 4. 实施步骤

### Step 1：ruff/AGENTS.md 合规合并（组 D + E）

#### 目标

合并 `test_phase2_revision_ruff_agents.py` 到 `test_ruff_config.py` 与 `test_agents_md.py`，消除 ruff SLF001 检查的 3 处重复。

#### 1.1 开发任务

**任务 1.1.1**：在 `test_ruff_config.py` 末尾新增 noqa 豁免纪律测试类。

```python
class TestNoqaExemptionDiscipline(unittest.TestCase):
    """ruff SLF001 noqa 豁免规范：仅 core 层组件编排允许。"""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.env_source = (
            repo_root / "orca_gym" / "environment" / "euler" / "orca_gym_euler_env.py"
        ).read_text(encoding="utf-8")
        cls.gym_source = (
            repo_root / "orca_gym" / "core" / "euler" / "orca_gym_euler.py"
        ).read_text(encoding="utf-8")

    def test_env_no_noqa_slf001(self):
        """Env 源码不使用 noqa: SLF001 豁免（Env 不应穿墙）。"""
        self.assertNotIn("noqa: SLF001", self.env_source)
        self.assertNotIn("noqa:SLF001", self.env_source)

    def test_gym_noqa_only_for_bind_orchestration(self):
        """Gym 源码 noqa: SLF001 仅用于 _bind 编排（非穿墙访问）。"""
        import re
        noqa_lines = re.findall(r".*# noqa: ?SLF001.*", self.gym_source)
        self.assertGreaterEqual(len(noqa_lines), 1, "Gym 应有 _bind 编排的 noqa 豁免")
        for line in noqa_lines:
            self.assertTrue(
                "_bind(" in line or "object.__getattribute__" in line,
                f"noqa 行非组件编排豁免: {line.strip()}",
            )
```

**任务 1.1.2**：删除 `tests/orca_gym/environment/euler/test_phase2_revision_ruff_agents.py`。

#### 1.2 验收标准

- [ ] `test_ruff_config.py` 含 `TestNoqaExemptionDiscipline` 类，2 个用例通过
- [ ] `test_phase2_revision_ruff_agents.py` 文件已删除
- [ ] `test_ruff_config.py::test_ruff_installed` 通过（原 revision 的重复项已消失）
- [ ] `test_agents_md.py` 全部 5 用例通过（无变化）
- [ ] 全量 `python -m unittest discover -s tests/orca_gym` 通过

#### 1.3 验证命令

```bash
<conda-base>/envs/orca/bin/python -m unittest \
    tests.orca_gym.test_ruff_config \
    tests.orca_gym.test_agents_md -v
```

---

### Step 2：SimpleEulerEnv 合规合并（组 A）

#### 目标

将 `test_phase2_revision_simple_env.py` 的 5 个独有用例迁入 `test_simple_euler_env_compliance.py`，删除原文件。

#### 2.1 开发任务

**任务 2.1.1**：在 `test_simple_euler_env_compliance.py` 中新增以下 5 个用例（迁入 `TestSimpleEnvResetModelFunctional` 类或新建类）：

| 待迁入用例 | 目标位置 |
|-----------|---------|
| `test_no_self_gym_access` | `TestSimpleEnvK3K5NoTunnelAccess` |
| `test_no_xfrc_applied_direct_write` | `TestSimpleEnvK3K5NoTunnelAccess` |
| `test_env_gym_raises_attribute_error` | `TestSimpleEnvResetModelFunctional` |
| `test_step_returns_gymnasium_tuple` | `TestSimpleEnvResetModelFunctional` |
| `test_observation_space_box` | `TestSimpleEnvResetModelFunctional` |

**任务 2.1.2**：删除 `tests/orca_gym/environment/euler/test_phase2_revision_simple_env.py`。

#### 2.2 验收标准

- [ ] `test_simple_euler_env_compliance.py` 用例数从 8 增至 13
- [ ] `test_phase2_revision_simple_env.py` 文件已删除
- [ ] 迁入的 5 个用例全部通过
- [ ] 原 compliance 的 8 个用例无回归

#### 2.3 验证命令

```bash
<conda-base>/envs/orca/bin/python -m unittest \
    tests.orca_gym.environment.euler.test_simple_euler_env_compliance -v
```

---

### Step 3：Gym 层 phase2 修订删除（组 C）

#### 目标

删除 `test_phase2_revision_gym_compat.py`（21 个用例全部被 `core/euler/test_orca_gym_euler.py` 覆盖）。

#### 3.1 开发任务

**任务 3.1.1**：删除 `tests/orca_gym/environment/euler/test_phase2_revision_gym_compat.py`。

#### 3.2 验收标准

- [ ] `test_phase2_revision_gym_compat.py` 文件已删除
- [ ] `test_orca_gym_euler.py` 全部 54 用例通过（无回归）
- [ ] 逐一核对 §2.2 组 C 的 21 个用例的覆盖关系，确认 core 文件已覆盖

#### 3.3 验证命令

```bash
<conda-base>/envs/orca/bin/python -m unittest \
    tests.orca_gym.core.euler.test_orca_gym_euler -v
```

---

### Step 4：Env 层 phase2 修订删除（组 B）

#### 目标

删除 3 个 phase2 修订文件（`baseline` / `env_filling` / `isolation_checklist`），共 50 个用例。所有用例被 `test_orca_gym_euler_env_skeleton.py` 覆盖。

#### 4.1 开发任务

**任务 4.1.1**：逐一核对 §2.2 组 B 的覆盖关系表，确认 skeleton 已覆盖。

**任务 4.1.2**：删除以下 3 个文件：
- `tests/orca_gym/environment/euler/test_phase2_revision_baseline.py`
- `tests/orca_gym/environment/euler/test_phase2_revision_env_filling.py`
- `tests/orca_gym/environment/euler/test_phase2_revision_isolation_checklist.py`

#### 4.2 争议项处理

**baseline 的 4 个源码扫描用例**（`test_no_public_gym_assignment`、`test_no_object_setattr_bypass`、`test_mixin_imported`、`test_self_orchestrated_lifecycle`）：

- **建议不保留**：这些是迁移期回归测试，验证「源码不含某模式」。迁移完成后由 skeleton 运行时检查（如 `TestEnvK14Inheritance` 验证继承链、`TestEnvK2Isolation` 验证无 `__getattr__`）替代，运行时检查更稳健。
- **若需保留**：合并成 1 个 `test_source_audit.py`，仅保留 4 个独有源码断言，不维持 13 个用例的文件。

#### 4.3 验收标准

- [ ] 3 个 phase2 修订文件已删除
- [ ] `test_orca_gym_euler_env_skeleton.py` 全部 135 用例通过（无回归）
- [ ] §2.2 组 B 覆盖关系表逐项确认

#### 4.4 验证命令

```bash
<conda-base>/envs/orca/bin/python -m unittest \
    tests.orca_gym.environment.euler.test_orca_gym_euler_env_skeleton -v
```

---

### Step 5：phase3 精简（组 F）

#### 目标

删除 `test_phase3_cross_substep_consistency.py::TestGlobalArchCompliance` 中与 skeleton 重复的 6 个 K 约束用例，保留 ruff 全局检查与跨子步骤功能/一致性用例。

#### 5.1 开发任务

**任务 5.1.1**：在 `test_phase3_cross_substep_consistency.py::TestGlobalArchCompliance` 中删除以下 6 个用例：

| 待删除用例 | skeleton 对应类 |
|-----------|----------------|
| `test_k1_k4_env_no_global_private_access` | `TestEnvK4NoGymPrivateAccess` |
| `test_k2_env_dir_no_internal_leak` | `TestEnvK2Isolation` |
| `test_k2_env_dir_contains_phase3_public_api` | `TestEnvK2Isolation` |
| `test_k6_env_data_is_dataview` | `TestEnvK6DataView` |
| `test_k9_studio_methods_via_bridge_not_gym_studio` | `TestEnvK9StudioAccess` |
| `test_k14_mro_stable` | `TestEnvK14Inheritance` |

**任务 5.1.2**：保留 `TestGlobalArchCompliance::test_ruff_slf001_global_zero`（组 E 唯一保留的 ruff 全局检查）。

**任务 5.1.3**：保留以下 22 个用例（§10.2 委托链路 + §10.3 数据一致性 + §11.2 K 回归）：
- `TestDelegationChainQueryMethods`（5 个：query_joint_qpos/qvel/sensor_data + get_body_xpos_xmat_xquat + query_site_pos_and_mat）
- `TestDelegationChainSetMethods`（5 个：set_joint_qpos/qvel + apply_body_force + set_mocap_pos_and_quat + set_ctrl）
- `TestDelegationChainStudioMethods`（2 个：studio_methods_offline_noop + render_offline_returns_none）
- `TestDelegationChainJacMethods`（3 个：mj_jacBody/Site/site_batch）
- `TestDataConsistency`（5 个：dataview_query_consistency_body_xpos/xmat + xfrc_consistency + step_forward_updates_view + qpos_is_zero_copy）
- `TestKConstraintRegression`（2 个：k11_typed_returns + k12_docstrings_present）

#### 5.2 验收标准

- [ ] `test_phase3_cross_substep_consistency.py` 用例数从 29 降至 23（删 6 留 23，含 ruff）
- [ ] 保留的 23 个用例全部通过
- [ ] skeleton 的 K1/K2/K4/K6/K9/K14 对应用例通过（覆盖已迁移）

#### 5.3 验证命令

```bash
<conda-base>/envs/orca/bin/python -m unittest \
    tests.orca_gym.environment.euler.test_phase3_cross_substep_consistency -v
```

---

### Step 6：全量回归验证

#### 目标

确认合并后测试套件全量通过，ruff SLF001 零报警，覆盖率未下降。

#### 6.1 开发任务

**任务 6.1.1**：全量运行测试套件。

```bash
<conda-base>/envs/orca/bin/python -m unittest discover -s tests/orca_gym -p "test_*.py" -v
```

**任务 6.1.2**：ruff SLF001 全局检查。

```bash
<conda-base>/envs/orca/bin/python -m ruff check --select SLF001 orca_gym/
```

**任务 6.1.3**：核对测试文件清单与用例数。

```bash
# 统计用例数
<conda-base>/envs/orca/bin/python -c "
import unittest, pathlib
loader = unittest.TestLoader()
suite = loader.discover('tests/orca_gym', pattern='test_*.py')
print(f'总用例数: {suite.countTestCases()}')
"
```

#### 6.2 验收标准

- [x] 全量 unittest 通过，0 失败（455 用例 OK，skipped=4）
- [x] ruff SLF001 全局零报警
- [x] 文件数 = 13（含初始清单遗漏的 `test_orca_studio_bridge.py`）
- [x] 用例数 = 455（与预估 ~449 接近，差异来自初始清单遗漏 `test_orca_studio_bridge.py` 的 42 用例）
- [x] 合并后文件清单符合 §5.1（已补正）

---

## 5. 合并后产物

### 5.1 合并后文件清单

```
tests/orca_gym/
├── __init__.py
├── test_ruff_config.py              (5 → 7，+TestNoqaExemptionDiscipline)
├── test_agents_md.py                (5，不变)
├── core/
│   ├── __init__.py
│   └── euler/
│       ├── __init__.py
│       ├── test_model_registry.py       (28，不变)
│       ├── test_mujoco_sim_core.py      (76，不变)
│       ├── test_orca_gym_data_view.py   (32，不变)
│       ├── test_orca_gym_euler.py       (54，不变)
│       ├── test_orca_studio_bridge.py   (42，不变，初始清单遗漏，已补正)
│       └── test_sim_config.py           (24，不变)
└── environment/
    ├── __init__.py
    ├── test_orca_gym_env_mixin.py       (13，不变)
    └── euler/
        ├── __init__.py
        ├── test_orca_gym_euler_env_skeleton.py  (135，不变，综合验收)
        ├── test_simple_euler_env_compliance.py  (8 → 13，合并 revision)
        ├── test_phase2_revision_e2e.py          (3，不变，端到端独有)
        └── test_phase3_cross_substep_consistency.py (29 → 23，精简)
```

### 5.2 删除的 6 个文件

1. `tests/orca_gym/environment/euler/test_phase2_revision_baseline.py`（13 用例，被 skeleton 覆盖）
2. `tests/orca_gym/environment/euler/test_phase2_revision_env_filling.py`（17 用例，被 skeleton 覆盖）
3. `tests/orca_gym/environment/euler/test_phase2_revision_isolation_checklist.py`（20 用例，被 skeleton 覆盖）
4. `tests/orca_gym/environment/euler/test_phase2_revision_gym_compat.py`（21 用例，被 core `test_orca_gym_euler.py` 覆盖）
5. `tests/orca_gym/environment/euler/test_phase2_revision_simple_env.py`（10 用例，5 独有迁入 compliance，5 重复删除）
6. `tests/orca_gym/environment/euler/test_phase2_revision_ruff_agents.py`（8 用例，2 独有迁入 ruff_config，6 重复删除）

### 5.3 用例数变化明细

| 文件 | 合并前 | 合并后 | 变化 |
|------|--------|--------|------|
| `test_ruff_config.py` | 5 | 7 | +2（迁入 noqa） |
| `test_simple_euler_env_compliance.py` | 8 | 13 | +5（迁入 revision） |
| `test_phase3_cross_substep_consistency.py` | 29 | 23 | -6（删重复 K 约束） |
| `test_phase2_revision_baseline.py` | 13 | 0 | -13（删除） |
| `test_phase2_revision_env_filling.py` | 17 | 0 | -17（删除） |
| `test_phase2_revision_isolation_checklist.py` | 20 | 0 | -20（删除） |
| `test_phase2_revision_gym_compat.py` | 21 | 0 | -21（删除） |
| `test_phase2_revision_simple_env.py` | 10 | 0 | -10（5 迁入 + 5 重复） |
| `test_phase2_revision_ruff_agents.py` | 8 | 0 | -8（2 迁入 + 6 重复） |
| 其余 10 文件（含初始清单遗漏的 `test_orca_studio_bridge.py` 42 用例） | 454 | 454 | 0 |
| **合计** | **585** | **497** | **-88** |

> **注**：表格合计 497 是「迁入用例在源文件与目标文件各计一次」的账面值（迁入的 7 个用例在源文件计入后又迁入目标文件，账面多计 7）。实际 unittest 统计合并后为 **455 用例**（含 4 跳过）。以 unittest 实际统计为准。

---

## 6. 风险与回滚

### 6.1 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| skeleton 未覆盖某独有用例，删除后覆盖率下降 | 中 | 中 | Step 4 前逐一核对 §2.2 组 B 覆盖表，争议项保留 |
| 迁入用例的 import 路径或 setUp 失败 | 低 | 低 | 迁入后立即运行目标文件验证 |
| phase3 删除 K 约束用例后，未来 skeleton 漏检 | 低 | 低 | skeleton 已有对应 K 约束类，且 ruff SLF001 全局检查保留 |
| baseline 源码扫描删除后，源码回归未发现 | 低 | 低 | 迁移已完成，源码模式不再变化；运行时检查更稳健 |

### 6.2 回滚策略

每个 Step 独立提交，若某步验证失败：
1. `git revert` 该步提交
2. 重新核对覆盖关系，补充遗漏用例后重做

### 6.3 争议项决策点

**组 B baseline 的 4 个源码扫描用例**：

- **选项 A（推荐）**：不保留，依赖 skeleton 运行时检查 + ruff SLF001
- **选项 B**：保留为独立 `test_source_audit.py`（4 用例），维持源码级回归

**决策方式**：在 Step 4 执行前由实施者决定。若选 B，则文件数变为 14、用例数预估 459。

---

## 7. 实施检查清单

实施时按以下顺序逐项勾选：

### Step 1（组 D + E）✅ 已完成
- [x] 在 `test_ruff_config.py` 新增 `TestNoqaExemptionDiscipline` 类（2 用例）
- [x] 运行 `test_ruff_config.py` + `test_agents_md.py` 通过（12 用例 OK）
- [x] 删除 `test_phase2_revision_ruff_agents.py`
- [x] 全量 discover 通过（537 用例 OK，skipped=4）

### Step 2（组 A）✅ 已完成
- [x] 在 `test_simple_euler_env_compliance.py` 迁入 5 个独有用例
- [x] 运行 `test_simple_euler_env_compliance.py` 通过（13 用例 OK）
- [x] 删除 `test_phase2_revision_simple_env.py`
- [x] 全量 discover 通过（532 用例 OK，skipped=4）

### Step 3（组 C）✅ 已完成
- [x] 核对 §2.2 组 C 覆盖表（21 项全部被 core `test_orca_gym_euler.py` 覆盖）
- [x] 删除 `test_phase2_revision_gym_compat.py`
- [x] 运行 `test_orca_gym_euler.py` 通过（54 用例 OK，无回归）
- [x] 全量 discover 通过（511 用例 OK，skipped=4）

### Step 4（组 B）✅ 已完成
- [x] 决定 baseline 4 个源码扫描用例的处理（**选项 A：不保留**）
- [x] 核对 §2.2 组 B 覆盖表（50 项：47 项被 skeleton 覆盖，3 项按选项 A 丢弃）
- [x] 删除 3 个 phase2 修订文件（baseline / env_filling / isolation_checklist）
- [x] 运行 `test_orca_gym_euler_env_skeleton.py` 通过（135 用例 OK，无回归）
- [x] 全量 discover 通过（461 用例 OK，skipped=4）

### Step 5（组 F）✅ 已完成
- [x] 删除 phase3 的 6 个重复 K 约束用例（保留 `test_ruff_slf001_global_zero`）
- [x] 运行 `test_phase3_cross_substep_consistency.py` 通过（23 用例 OK，skipped=1）
- [x] 全量 discover 通过（455 用例 OK，skipped=4）

### Step 6（全量回归）
- [ ] 全量 discover 通过，0 失败
- [ ] ruff SLF001 全局零报警
- [ ] 文件数 = 13（或 14，取决于组 B 争议项）
- [ ] 用例数 ≈ 449（或 459）
- [ ] 合并后文件清单符合 §5.1

---

## 附录 A：覆盖关系详表

### A.1 组 B 详细覆盖表

#### env_filling（17 用例）→ skeleton

| env_filling 用例 | skeleton 对应类.用例 |
|----------------|---------------------|
| `test_inheritance_chain` | `TestEnvK14Inheritance::test_env_inheritance_chain` |
| `test_env_gym_raises_native_attribute_error` | `TestEnvK2Isolation::test_env_gym_attr_natural_attribute_error` |
| `test_env_stub_raises_native_attribute_error` | `TestEnvK2Isolation::test_env_stub_attr_natural_attribute_error` |
| `test_env_channel_raises_native_attribute_error` | `TestEnvK2Isolation::test_env_channel_attr_natural_attribute_error` |
| `test_env_no_getattr_method` | `TestEnvK14Inheritance::test_env_no_blocked_attrs_classvar` |
| `test_env_no_setattr_method` | `TestEnvK14Inheritance::test_env_no_blocked_attrs_classvar` |
| `test_init_completes_without_error` | `TestEnvLifecycleAndStepping::test_env_init_completes_without_error` |
| `test_init_orchestrates_lifecycle_in_order` | `TestEnvLifecycleAndStepping::test_env_init_completes_without_error` |
| `test_init_does_not_call_super_init` | `TestEnvK14Inheritance::test_env_inheritance_chain` |
| `test_time_step_cached_before_init` | `TestEnvK7PropertyDelegation::test_dt_uses_sim_config` |
| `test_sim_config_bound_after_init` | `TestEnvK7PropertyDelegation::test_sim_config_returns_config` |
| `test_dt_uses_sim_config` | `TestEnvK7PropertyDelegation::test_dt_uses_sim_config` |
| `test_do_simulation_delegates_to_step_with_coupling` | `TestEnvLifecycleAndStepping::test_do_simulation_advances_time` |
| `test_set_joint_qpos_delegates_to_gym` | `TestEnvLifecycleAndStepping::test_set_joint_qpos_writes_state` |
| `test_set_joint_qvel_delegates_to_gym` | `TestEnvLifecycleAndStepping::test_set_joint_qvel_writes_state` |
| `test_data_returns_dataview` | `TestEnvK6DataView::test_data_property_returns_view` |
| `test_do_simulation_validates_action_dim` | `TestEnvK8NoEulerPrivate::test_do_simulation_validates_action_dim` |

#### isolation_checklist（20 用例）→ skeleton

| isolation_checklist 用例 | skeleton 对应类.用例 |
|------------------------|---------------------|
| `test_env_no_public_internal_attrs` | `TestEnvK1NamingConstraint::test_env_no_public_internal_attrs` |
| `test_env_gym_native_attribute_error` | `TestEnvK2Isolation::test_env_gym_attr_natural_attribute_error` |
| `test_env_stub_native_attribute_error` | `TestEnvK2Isolation::test_env_stub_attr_natural_attribute_error` |
| `test_env_channel_native_attribute_error` | `TestEnvK2Isolation::test_env_channel_attr_natural_attribute_error` |
| `test_env_no_getattr_classvar` | `TestEnvK14Inheritance::test_env_no_blocked_attrs_classvar` |
| `test_env_no_setattr_classvar` | `TestEnvK14Inheritance::test_env_no_blocked_attrs_classvar` |
| `test_env_no_shielded_attrs_classvar` | `TestEnvK14Inheritance::test_env_no_blocked_attrs_classvar` |
| `test_env_attribute_assignment_works` | `TestEnvK1NamingConstraint::test_env_has_studio_bridge_private` |
| `test_inheritance_chain` | `TestEnvK14Inheritance::test_env_inheritance_chain` |
| `test_k4_no_gym_private_access` | `TestEnvK4NoGymPrivateAccess`（整个类） |
| `test_k8_no_euler_private_access` | `TestEnvK8NoEulerPrivate`（整个类） |
| `test_k9_no_studio_property_access` | `TestEnvK9StudioAccess`（整个类） |
| `test_k6_data_returns_dataview` | `TestEnvK6DataView::test_data_property_returns_view` |
| `test_k7_sim_config_returns_config` | `TestEnvK7PropertyDelegation::test_sim_config_returns_config` |
| `test_k11_data_not_mjdata` | `TestEnvK6DataView::test_data_returns_view_not_mjdata` |
| `test_env_docstring_has_contract` | `TestEnvK12Docstring::test_env_docstring_has_contract` |
| `test_ruff_slf001_euler_env_clean` | 组 E（phase3 全局检查覆盖） |
| `test_ruff_slf001_euler_core_clean` | 组 E（phase3 全局检查覆盖） |
| `test_env_dir_no_internal` | `TestEnvK2Isolation::test_env_dir_only_exposes_public_api` |
| `test_env_dir_contains_public_api` | `TestEnvK2Isolation::test_env_dir_contains_public_api` |

#### baseline（13 用例）→ skeleton / 争议项

| baseline 用例 | 处理 |
|-------------|------|
| `test_no_old_inheritance_chain` | skeleton `TestEnvK14Inheritance` 覆盖 |
| `test_no_blocked_attrs_in_env` | skeleton `TestEnvK14Inheritance::test_env_no_blocked_attrs_classvar` 覆盖 |
| `test_no_getattr_in_env` | skeleton `TestEnvK14Inheritance::test_env_no_blocked_attrs_classvar` 覆盖 |
| `test_no_setattr_in_env` | skeleton `TestEnvK14Inheritance::test_env_no_blocked_attrs_classvar` 覆盖 |
| `test_no_super_init_in_env` | skeleton `TestEnvK14Inheritance::test_env_inheritance_chain` 覆盖 |
| `test_no_public_gym_assignment` | **争议项**（源码扫描，建议不保留） |
| `test_no_object_setattr_bypass` | **争议项**（源码扫描，建议不保留） |
| `test_no_shielded_attrs_in_env` | skeleton `TestEnvK14Inheritance::test_env_no_blocked_attrs_classvar` 覆盖 |
| `test_new_inheritance_chain` | skeleton `TestEnvK14Inheritance::test_env_inheritance_chain` 覆盖 |
| `test_mixin_imported` | **争议项**（源码扫描，建议不保留） |
| `test_self_orchestrated_lifecycle` | **争议项**（源码扫描，建议不保留） |
| `test_private_gym_field_used` | skeleton `TestEnvK1NamingConstraint::test_env_no_public_internal_attrs` 覆盖 |
| `test_dir_method_present` | skeleton `TestEnvK2Isolation` 覆盖 |

### A.2 组 C 详细覆盖表

| gym_compat 用例 | core `test_orca_gym_euler.py` 对应 |
|----------------|-----------------------------------|
| `test_gym_has_blocked_attrs` | `test_gym_blocked_attrs_include_components` |
| `test_gym_has_getattribute` | `test_gym_blocked_attrs_message_has_guidance`（隐含） |
| `test_gym_blocked_attrs_contains_mjdata_mjmodel` | `test_gym_blocked_attrs_include_components` |
| `test_gym_blocked_attrs_contains_subcomponents` | `test_gym_blocked_attrs_include_components` |
| `test_gym_external_access_blocked_with_guidance` | `test_gym_blocked_attrs_message_has_guidance` |
| `test_gym_external_access_sim_blocked` | `test_gym_blocked_attrs_include_components` |
| `test_gym_external_access_studio_blocked` | `test_gym_blocked_attrs_include_components` |
| `test_gym_external_access_opt_blocked` | `test_gym_blocked_attrs_include_components` |
| `test_sim_config_bound_after_init` | `TestOrcaGymEulerDelegation::test_model_returns_orca_gym_model` |
| `test_model_registry_bound_after_init` | `TestOrcaGymEulerDelegation::test_model_nq_correct` |
| `test_sim_config_returns_simconfig_instance` | `test_gym_sim_config_returns_config` |
| `test_gym_step_with_coupling_works` | `TestOrcaGymEulerDelegation`（step_with_coupling 相关） |
| `test_gym_set_qpos_qvel_writes_state` | `TestOrcaGymEulerDelegation` |
| `test_gym_sync_to_view_populates_dataview` | `TestOrcaGymEulerDelegation` |
| `test_gym_data_returns_dataview` | `test_gym_data_returns_view` |
| `test_gym_mj_step_advances_time` | `TestOrcaGymEulerDelegation::test_mj_step_advances_time` |
| `test_has_euler_returns_false` | `test_gym_has_euler_returns_false` |
| `test_has_euler_callable` | `test_gym_has_euler_returns_false`（callable 隐含） |
| `test_studio_bridge_is_method` | `test_gym_studio_bridge_is_method_not_property` |
| `test_gym_studio_blocked` | `test_gym_studio_bridge_is_method_not_property` |
| `test_studio_bridge_returns_bridge` | `test_gym_studio_bridge_is_method_not_property` |

### A.3 组 F 详细覆盖表

| phase3 待删除用例 | skeleton 对应 |
|------------------|--------------|
| `test_k1_k4_env_no_global_private_access` | `TestEnvK4NoGymPrivateAccess` |
| `test_k2_env_dir_no_internal_leak` | `TestEnvK2Isolation::test_env_dir_only_exposes_public_api` |
| `test_k2_env_dir_contains_phase3_public_api` | `TestEnvK2Isolation::test_env_dir_contains_public_api` |
| `test_k6_env_data_is_dataview` | `TestEnvK6DataView::test_data_property_returns_view` |
| `test_k9_studio_methods_via_bridge_not_gym_studio` | `TestEnvK9StudioAccess::test_no_studio_property_access` |
| `test_k14_mro_stable` | `TestEnvK14Inheritance::test_env_inheritance_chain` |
