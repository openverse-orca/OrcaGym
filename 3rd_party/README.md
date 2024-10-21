# 3rd_party Directory

This directory contains third-party libraries that are used in this project. Each library is included as a Git submodule, allowing us to easily integrate, update, and manage external dependencies while keeping the main repository lightweight.

## Purpose

The `3rd_party` directory serves the following purposes:

- **Modular Integration**: Each third-party library is maintained separately as a Git submodule, enabling easy updates and version control while avoiding unnecessary duplication of code.
- **Seamless Dependency Management**: By using submodules, the external libraries are kept outside of the main project codebase, which helps to maintain a clean project structure.
- **Compliance with Original Licenses**: Each library included here retains its original license, and we respect and comply with those terms. The original license files for each project can be found in their respective submodule directories.

## License Information

All libraries included in the `3rd_party` directory are external open-source projects, and each submodule retains its original license agreement. Please refer to the license files located within the corresponding submodule directories for more information about the specific terms and conditions of each library.

## Cloning the Repository

If you've already cloned the OrcaGym repository without submodules, you can initialize and update them using:

```bash
git submodule update --init --recursive
```

## Disclaimer

The usage of third-party libraries is subject to the terms and conditions set forth by the original authors. Please review each library's license before using the project.