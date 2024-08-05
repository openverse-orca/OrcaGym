import os
from grpc_tools import protoc

# 获取当前脚本文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 proto 文件的相对路径
proto_file = os.path.join(current_dir, "./mjc_message.proto")

# 定义输出目录的绝对路径
output_dir = os.path.join(current_dir, "./")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

protoc.main((
    '',
    f'-I{os.path.dirname(proto_file)}',
    f'--python_out={output_dir}',
    f'--grpc_python_out={output_dir}',
    proto_file,
))
