#!/bin/bash
# 安装 RLLib 和 SB3 所需的依赖

echo "安装 RLLib 依赖..."
pip install pydantic

echo "安装 SB3 依赖..."
pip install stable-baselines3>=2.3.2 sb3_contrib>=2.3.0

echo "安装 RLLib 完整依赖..."
pip install "ray[rllib]==2.49.0"

echo "依赖安装完成！"

