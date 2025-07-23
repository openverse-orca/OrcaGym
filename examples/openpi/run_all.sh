#!/bin/bash

# 输入路径作为参数
read -p "请输入路径: " input_path

# 调用第一个脚本，传入路径作为参数
python sccc.py "$input_path"

# 调用第二个脚本，并把路径通过标准输入传入
echo "$input_path" | python rena.py
