# Release Scripts

这个目录包含了 OrcaGym Core 发布到 PyPI 的所有脚本。

## 📁 脚本列表

| 脚本 | 说明 | 用途 |
|------|------|------|
| `clean.sh` | 清理构建产物 | 清除 dist/, build/, *.egg-info 等 |
| `build.sh` | 构建分发包 | 生成 wheel 和 tar.gz 包 |
| `check.sh` | 检查包质量 | 使用 twine 验证包 |
| `upload_test.sh` | 上传到 TestPyPI | 测试环境发布 |
| `upload_prod.sh` | 上传到正式 PyPI | 生产环境发布 |
| `release.sh` | 完整发布流程 | 整合所有步骤 |
| `bump_version.sh` | 更新版本号 | 修改 pyproject.toml 中的版本 |
| `test_install.sh` | 测试安装 | 验证包安装和导入 |

## 🚀 快速开始

### 1. 首次发布到 TestPyPI

```bash
# 完整流程（推荐）
./scripts/release/release.sh test

# 或者分步执行
./scripts/release/clean.sh
./scripts/release/build.sh
./scripts/release/check.sh
./scripts/release/upload_test.sh
```

### 2. 测试安装

```bash
# 从本地 dist/ 测试
./scripts/release/test_install.sh local

# 从 TestPyPI 测试
./scripts/release/test_install.sh test

# 从正式 PyPI 测试
./scripts/release/test_install.sh prod
```

### 3. 发布到正式 PyPI

```bash
./scripts/release/release.sh prod
```

## 📋 详细使用指南

### 清理构建产物

```bash
./scripts/release/clean.sh
```

清理所有构建产生的文件和目录：
- `build/`
- `dist/`
- `*.egg-info/`
- `__pycache__/`
- `*.pyc`, `*.pyo` 文件

### 构建分发包

```bash
./scripts/release/build.sh
```

生成两种格式的分发包：
- `orca_gym-{version}-py3-none-any.whl` - wheel 格式（推荐）
- `orca_gym-{version}.tar.gz` - 源码包

构建后的文件在 `dist/` 目录。

### 检查包质量

```bash
./scripts/release/check.sh
```

使用 `twine check` 验证：
- README 格式
- 元数据完整性
- 包结构正确性

### 更新版本号

```bash
./scripts/release/bump_version.sh 25.10.1
```

更新 `pyproject.toml` 中的版本号，并提示后续步骤：
1. 查看变更
2. 提交变更
3. 创建 Git tag
4. 推送到仓库

### 上传到 TestPyPI

```bash
./scripts/release/upload_test.sh
```

上传到测试环境，用于验证包的正确性。需要：
- TestPyPI 账号（https://test.pypi.org/account/register/）
- API Token（https://test.pypi.org/manage/account/token/）

### 上传到正式 PyPI

```bash
./scripts/release/upload_prod.sh
```

⚠️ **警告**：此操作不可撤销！发布前请确保：
- 已在 TestPyPI 测试
- 版本号正确
- 代码已推送到 GitHub

需要：
- PyPI 账号（https://pypi.org/account/register/）
- API Token（https://pypi.org/manage/account/token/）

### 完整发布流程

```bash
# 发布到 TestPyPI
./scripts/release/release.sh test

# 发布到正式 PyPI
./scripts/release/release.sh prod
```

自动执行完整流程：
1. 清理旧文件
2. 构建新包
3. 检查包质量
4. 上传到指定环境

### 测试安装

```bash
# 从本地 wheel 文件测试
./scripts/release/test_install.sh local

# 从 TestPyPI 测试
./scripts/release/test_install.sh test

# 从正式 PyPI 测试
./scripts/release/test_install.sh prod
```

自动创建临时虚拟环境，安装包并测试导入。

## 🔐 配置 API Token

### ⚠️ 重要：配置文件位置

`.pypirc` 文件**必须**放在用户 home 目录下：

```bash
~/.pypirc    ✅ 正确位置
```

**不是**项目目录下：
```bash
scripts/release/.pypirc    ❌ 错误位置
```

`twine` 只会读取 `~/.pypirc`，项目目录下的配置文件不会生效！

### 配置步骤

#### 方式 1: 使用示例文件（推荐）

```bash
# 1. 复制示例文件到正确位置
cp scripts/release/.pypirc.example ~/.pypirc

# 2. 编辑文件，填入你的 API token
vim ~/.pypirc

# 3. 设置正确的权限
chmod 600 ~/.pypirc
```

#### 方式 2: 手动创建

```bash
# 创建配置文件
vim ~/.pypirc
```

### TestPyPI 配置

1. **注册账号**（如果还没有）
   - 访问 https://test.pypi.org/account/register/
   - 填写用户名、邮箱、密码
   - 验证邮箱

2. **生成 API Token**
   - 访问 https://test.pypi.org/manage/account/token/
   - 点击 "Add API token"
   - Token name: 例如 "OrcaGym Upload"
   - Scope: 选择 "Entire account (all projects)"
   - 点击 "Create token"
   - ⚠️ 立即复制 token（只显示一次！）

3. **配置 `~/.pypirc`**

```ini
[distutils]
index-servers =
    pypi
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9y...你的token...
```

### 正式 PyPI 配置

1. **注册账号**（如果还没有）
   - 访问 https://pypi.org/account/register/

2. **生成 API Token**
   - 访问 https://pypi.org/manage/account/token/
   - 按照与 TestPyPI 相同的步骤创建 token

3. **更新 `~/.pypirc`**

```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...你的token...
```

### 完整配置示例

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...你的正式PyPI token...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9y...你的TestPyPI token...
```

### 设置文件权限

⚠️ **重要**：必须设置正确的文件权限，否则可能被拒绝访问：

```bash
chmod 600 ~/.pypirc
```

### 验证配置

```bash
# 检查文件是否存在
ls -la ~/.pypirc

# 检查文件权限（应该显示 -rw-------）
ls -l ~/.pypirc

# 查看配置内容
cat ~/.pypirc
```

### 常见问题

**问题 1**: 配置了还要求输入密码

**原因**: `.pypirc` 文件位置错误

**解决**:
```bash
# 确保文件在 home 目录
cp scripts/release/.pypirc ~/.pypirc
chmod 600 ~/.pypirc
```

**问题 2**: 403 Forbidden 错误

**原因**: 
- Token 作用域不正确（应该选择 "Entire account"）
- 项目名已被占用

**解决**:
```bash
# 重新生成 token，确保选择 "Entire account" 作用域
# 或者修改项目名称（首次上传时）
```

**问题 3**: 401 Unauthorized 错误

**原因**: Token 无效或格式错误

**解决**:
```bash
# 检查配置格式
# username 必须是 "__token__" (两个下划线)
# password 是完整的 token 字符串（以 pypi- 开头）
```

## 📝 完整发布工作流

### 发布新版本的完整步骤

1. **准备工作**
   ```bash
   # 确保代码已提交
   git status
   
   # 更新版本号
   ./scripts/release/bump_version.sh 25.10.1
   
   # 提交变更
   git add pyproject.toml
   git commit -m "Bump version to 25.10.1"
   ```

2. **测试发布**
   ```bash
   # 发布到 TestPyPI
   ./scripts/release/release.sh test
   
   # 测试安装
   ./scripts/release/test_install.sh test
   ```

3. **正式发布**
   ```bash
   # 确认测试通过后发布到 PyPI
   ./scripts/release/release.sh prod
   
   # 验证安装
   ./scripts/release/test_install.sh prod
   ```

4. **创建 Git Tag**
   ```bash
   git tag -a v25.10.1 -m "Release version 25.10.1"
   git push origin main
   git push origin v25.10.1
   ```

5. **创建 GitHub Release**
   - 访问 https://github.com/openverse-orca/OrcaGym/releases/new
   - 选择刚创建的 tag
   - 填写 Release Notes

## 🛠️ 故障排查

### 构建失败

**问题**: `pyproject.toml` 格式错误

**解决**:
```bash
# 验证 TOML 语法
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

### 上传失败

**问题**: 版本号已存在

**解决**: PyPI 不允许覆盖已发布的版本，必须更新版本号：
```bash
./scripts/release/bump_version.sh 25.10.2
```

**问题**: 认证失败

**解决**: 检查 `~/.pypirc` 配置是否正确，或直接输入凭据。

### 安装失败

**问题**: 依赖冲突

**解决**: 在新的虚拟环境中测试：
```bash
python -m venv test_env
source test_env/bin/activate
pip install orca-gym
```

## 📚 参考资源

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

## 💡 最佳实践

1. **始终先发布到 TestPyPI** 进行验证
2. **使用语义化版本号** (MAJOR.MINOR.PATCH)
3. **创建 Git tag** 对应每个发布版本
4. **使用 API Token** 而不是密码，更安全
5. **自动化 CI/CD** 可以使用 GitHub Actions

## 🔄 持续集成

可以使用 GitHub Actions 自动化发布流程。参考 `.github/workflows/publish.yml`（如果存在）。

## ⚠️ 注意事项

- PyPI 发布是**永久性**的，不能删除或覆盖
- TestPyPI 会定期清理旧包，不适合生产环境
- 确保所有测试通过后再发布到正式 PyPI
- 版本号递增，不要回退
- 保持包的向后兼容性

