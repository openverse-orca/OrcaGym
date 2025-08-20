import yaml
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import os

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# if project_root not in sys.path:
#     sys.path.append(project_root)


import orca_gym.scripts.dual_arm_manipulation as dual_arm_manipulation

import argparse
import subprocess
import threading
import time
import sys

class ConfigArgumentParser:
    """支持配置文件的参数解析器"""
    
    def __init__(self, description: str = None):
        self.parser = argparse.ArgumentParser(description=description)
        self.config_data = {}
        
        # 添加配置文件参数
        self.parser.add_argument('--config', type=str, 
                                help='Configuration file path (YAML or JSON)')
    
    def add_argument(self, *args, **kwargs):
        """添加参数定义"""
        return self.parser.add_argument(*args, **kwargs)
    
    def load_config(self, config_file: str):
        """加载配置文件"""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.load(f, Loader=yaml.FullLoader)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        # 扁平化配置
        self.config_data = {}
        for section, params in config.items():
            if isinstance(params, dict):
                self.config_data.update(params)
            else:
                self.config_data[section] = params
    
    def parse_args(self, args=None):
        """解析参数，支持配置文件"""
        # 先解析命令行参数以获取配置文件路径
        temp_args, remaining_args = self.parser.parse_known_args(args)
        
        # 如果指定了配置文件，先加载配置
        if hasattr(temp_args, 'config') and temp_args.config:
            self.load_config(temp_args.config)
        elif Path('config.yaml').exists():
            # 默认加载 config.yaml
            self.load_config('config.yaml')
        
        # 根据配置文件设置默认值
        for action in self.parser._actions:
            if action.dest in self.config_data:
                action.default = self.config_data[action.dest]
        
        # 重新解析所有参数
        return self.parser.parse_args(args)
    
    def create_config_template(self, output_file: str = 'config_template.yaml'):
        """创建配置文件模板"""
        config_template = {
            'basic': {},
            'connection': {},
            'mode': {},
            'task': {},
            'advanced': {}
        }
        
        # 根据参数定义创建模板
        section_mapping = {
            'parallelism_num': 'basic',
            'datalink_auth_config': 'basic',
            'orcasim_path': 'basic',
            'level': 'basic',
            'levelorca': 'basic',
            'orcagym_address': 'connection',
            'agent_names': 'connection',
            'pico_ports': 'connection',
            'run_mode': 'mode',
            'action_type': 'mode',
            'action_step': 'mode',
            'task_config': 'task',
            'algo': 'task',
            'dataset': 'task',
            'model_file': 'task',
        }
        
        for action in self.parser._actions:
            if action.dest != 'help' and action.dest != 'config':
                section = section_mapping.get(action.dest, 'advanced')
                config_template[section][action.dest] = action.default
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_template, f, default_flow_style=False, allow_unicode=True)
        
        print(f"配置文件模板已创建: {output_file}")

# 使用示例
def main():
    # 创建配置解析器
    parser = ConfigArgumentParser(description='Run parallelism augmenta of the script')
    
    # 添加所有参数
    parser.add_argument('--parallelism_num', type=int, default=4, help='The parallelism number')
    parser.add_argument('--datalink_auth_config', type=str,  help='The datalink auth config abs path')
    parser.add_argument('--orcasim_path', type=str,  help='The orcasim processor path')
    parser.add_argument('--level', type=str, default='shopscene', help='The storage level')
    parser.add_argument('--levelorca', type=str, default='shopscene', help='The Orcagym data store directory')
    parser.add_argument('--orcagym_address', type=str, default='localhost:50051', help='The gRPC addresses')
    parser.add_argument('--agent_names', type=str, default='openloong_gripper_2f85_fix_base_usda', help='The agent names')
    parser.add_argument('--pico_ports', type=str, default='8001', help='The pico server port')
    parser.add_argument('--run_mode', type=str, default='teleoperation', help='The run mode')
    parser.add_argument('--action_type', type=str, default='joint_pos', help='The action type')
    parser.add_argument('--action_step', type=int, default=1, help='Simulation steps per action')
    parser.add_argument('--task_config', type=str, default='ashop_task.yaml', help='Task config file')
    parser.add_argument('--algo', type=str, default='bc', help='Algorithm for training')
    parser.add_argument('--dataset', type=str, help='Dataset file path')
    parser.add_argument('--model_file', type=str, help='Model file for rollout')
    parser.add_argument('--record_length', type=int, default=1200, help='Recording time length')
    parser.add_argument('--ctrl_device', type=str, default='vr', help='Control device')
    parser.add_argument('--playback_mode', type=str, default='random', help='Playback mode')
    parser.add_argument('--rollout_times', type=int, default=10, help='Rollout times')
    parser.add_argument('--augmented_noise', type=float, default=0.01, help='Augmentation noise')
    parser.add_argument('--augmented_rounds', type=int, default=3, help='Augmentation rounds')
    parser.add_argument('--teleoperation_rounds', type=int, default=100, help='Teleoperation rounds')
    parser.add_argument('--sample_range', type=float, default=0.0, help='Sampling range')
    parser.add_argument('--realtime_playback', type=str, default='True', help='Real-time playback flag')
    parser.add_argument('--withvideo', type=str, default='True', help='Video flag')
    parser.add_argument('--index', type=int, default=0, help='Index')
    parser.add_argument('--sync_codec', type=str, default='True', help='Sync codec flag')
    parser.add_argument('--useNvenc', type=int, default=1, help='Use Nvenc')
    
    # 创建配置模板（可选）
    # parser.create_config_template()
    
    # 解析参数

    args = parser.parse_args()
    print("解析成功!")
    print("\n参数列表:")
    for arg_name, arg_value in vars(args).items():
        print(f"  {arg_name}: {arg_value}")
    
    


    process  = []

    def DoStartProcess(args,proslist):
        # print("1111111111111111111111111111111111")
      #  global process,orcagym_address_list,port, level
        global process

        parallelism_num = args.parallelism_num
        datalink_auth_config = args.datalink_auth_config
        orcasim_path = args.orcasim_path
        level = args.level

        orcagym_address = args.orcagym_address
        host, port = orcagym_address.split(':')
        port = int(port)
        orcagym_address_list = []
        
    #  for i in range(parallelism_num):
        orcagym_address_list.append(f"0.0.0.0:{port + args.index}")
        print("orcagym_address_list:",orcagym_address_list[0])
        i = args.index
        adapterIndex = i % 2
        print("adapterIndex:...........", adapterIndex)
        if args.useNvenc == 1:
            p = subprocess.Popen([orcasim_path, "--datalink_auth_config", datalink_auth_config,
                    "--mj_grpc_server",  orcagym_address_list[0],
                    "--forceAdapter", " NVIDIA GeForce RTX 4090",
                    "--adapterIndex", str(adapterIndex),
                    "--r_width", "128", "--r_height", "128",
                    "--useNvenc", "1",
                    "--lockFps30",
                    f"--regset=\"/O3DE/Autoexec/ConsoleCommands/LoadLevel={level}\""], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            p = subprocess.Popen([orcasim_path, "--datalink_auth_config", datalink_auth_config,
                    "--mj_grpc_server",  orcagym_address_list[0],
                    "--forceAdapter", " NVIDIA GeForce RTX 4090",
                    "--adapterIndex", str(adapterIndex),
                    "--r_width", "128", "--r_height", "128",
                    "--lockFps30",
                    f"--regset=\"/O3DE/Autoexec/ConsoleCommands/LoadLevel={level}\""], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        proslist.append(p)

        time.sleep(15)


        #调用增广脚本，如果需要额外参数自行添加
        threads = []
    
        scriptp = subprocess.Popen(["python", "run_dual_arm_sim.py", 
                        "--orcagym_address", f"localhost:{port + i}",
                        "--run_mode", "augmentation",
                        "--dataset", args.dataset,
                        "--action_type",args.action_type,
                        "--task_config",args.task_config,
                        "--sample_range",str(args.sample_range),
                        "--augmented_noise",str(args.augmented_noise),
                        "--augmented_rounds",str(args.augmented_rounds),
                        "--realtime_playback",args.realtime_playback,
                        "--withvideo",args.withvideo,
                        "--level",args.levelorca,
                        "--sync_codec",args.sync_codec


                        ],  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        


            
        proslist.append(scriptp)
    
    DoStartProcess(args,process)

    scriptp = process[-1]


      #  监测scriptp 进程的输出，如果超过30秒没有输出，则认为脚本执行失败
        
    start_time = time.time()
    while True:
        output = scriptp.stdout.readline()
        #if output == '' and scriptp.poll() is not None:
            #  print("ccccccccccccccccccccccccccccc")
            # break
        if output:
            #print(output.strip())
            start_time = time.time()
            print(output.strip())
        if time.time() - start_time > 30:  # 超过30秒没有输出
            print("Script execution seems to be stuck, terminating...")
            for p in process:
                p.stdout.close()
                p.stderr.close()
                p.terminate()
            process.clear()

            print("Restarting the process...")  
            time.sleep(5) # 等待一会儿以确保进程终止
            DoStartProcess(args,process)
            scriptp = process[-1]
            start_time = time.time()


            

if __name__ == "__main__":
    main()