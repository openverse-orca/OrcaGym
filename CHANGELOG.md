# Changelog

All notable changes to the OrcaGym project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [25.11.1] - 2025-11-06

### Changed - PyPI Package Preparation

#### Package Name
- Changed package name from `OrcaGym` to `orca-gym` for PyPI publication
- This reflects the core-only nature of the PyPI package

#### Dependencies
- **Reduced core dependencies from 25 to 8 packages**
- Moved 17 optional dependencies to separate groups: `[rl]`, `[imitation]`, `[devices]`, `[sensors]`, `[tools]`, `[robomimic]`, `[robosuite]`, `[all]`, `[dev]`
- Core dependencies now include only:
  - numpy>=2.0.0
  - scipy
  - scipy-stubs
  - grpcio==1.66.1
  - grpcio-tools==1.66.1
  - gymnasium>=1.0.0
  - mujoco>=3.3.0
  - aiofiles

#### Documentation
- Updated README.md to focus on core package functionality
- Removed CUDA/PyTorch installation instructions from main README
- Added installation examples for optional dependency groups
- Added usage examples for remote and local environments
- Created `examples/INSTALLATION_GUIDE.md` for example-specific dependencies
- Created `PYPI_RELEASE.md` with complete PyPI publishing guide
- Created `PACKAGE_CHANGES.md` documenting all changes

#### Package Configuration
- Updated `pyproject.toml` with comprehensive metadata:
  - Added license, authors, keywords, classifiers
  - Added project URLs (homepage, documentation, repository, bug tracker)
  - Configured package inclusion/exclusion rules
- Created `MANIFEST.in` for proper file packaging
- Created `orca_gym/py.typed` for PEP 561 type hints support

#### Package Structure
- Package now includes only `orca_gym/` directory with core modules:
  - `core/` - Core simulation interfaces
  - `environment/` - Gymnasium-compatible environments
  - `protos/` - gRPC protocol definitions
  - `scene/` - Scene management
  - `utils/` - Utility functions
- Excluded from package: `examples/`, `envs/`, `doc/`, `3rd_party/`, adapters, devices, scripts, sensors, tools
- Users can access full functionality by cloning the repository and installing optional dependencies

### Added
- `MANIFEST.in` - Package manifest file
- `orca_gym/py.typed` - Type hints marker file
- `PYPI_RELEASE.md` - PyPI publishing guide
- `examples/INSTALLATION_GUIDE.md` - Example-specific installation guide
- `PACKAGE_CHANGES.md` - Detailed change documentation
- `CHANGELOG.md` - Version history (this file)

### Migration Guide

For users upgrading from the full repository installation:

**Before:**
```bash
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym
pip install -e .
```

**After (PyPI):**
```bash
pip install orca-gym[all]  # or specific groups like [rl], [imitation]
```

**After (Development):**
```bash
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym
pip install -e ".[all,dev]"
```

### Compatibility
- ✅ API remains unchanged - no breaking changes
- ✅ Import paths remain the same
- ✅ Fully compatible with Gymnasium API
- ✅ All core functionality preserved

### Notes
- This is the first version prepared for PyPI release
- The core package is lightweight and focused on essential functionality
- Optional features require additional dependency installation
- Full examples and environments remain available in the GitHub repository

---

## [Previous Versions]

For changes prior to the PyPI package preparation, please refer to the Git commit history.

---

## Release Types

- **Major version (X.0.0)**: Incompatible API changes
- **Minor version (0.X.0)**: Backwards-compatible new features
- **Patch version (0.0.X)**: Backwards-compatible bug fixes

## Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

