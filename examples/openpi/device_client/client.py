#!/usr/bin/env python3
"""
数据采集设备客户端
运行在每个采集设备上，监控本地目录并上报数据到中央服务器
"""

import os
import sys
import json
import time
import glob
import socket
import requests
import threading
from datetime import datetime
from typing import Dict, Optional
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

import psutil # 导入 psutil 库  

import argparse
import subprocess
import threading
import time
import sys

import logging
import requests
import websocket
import yaml

class DeviceClient:
    """
    采集设备客户端主类
    """
    
    def __init__(self, config_file: str = "client_config.json"):
        # 获取当前脚本的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取上层目录
        parent_dir = os.path.dirname(current_dir)

        print("parent_dir:",parent_dir)





        self.config = self._load_config(config_file)
        self.device_id = self.config.get('device_id')
        self.server_url = self.config.get('server_url', 'http://localhost:8000')
        self.server_ws_url = self.config.get('server_ws_url', 'ws://localhost:8000/ws')
        self.monitor_directory = self.config.get('monitor_directory', '/data')
        self.report_interval = self.config.get('report_interval', 30)
        self.heartbeat_interval = self.config.get('heartbeat_interval', 60)
        self.nodata_timeout = self.config.get('nodata_timeout', 300)
        self.auto_restart = self.config.get('auto_restart', False)
        self.start_processscripts = self.config.get('start_processscripts', "parallelism_config_test.py")
        self.start_processsparam = self.config.get('start_processsparam', "runconfig.yaml")

        if not self.device_id or self.device_id.strip() == "":
            print("Device ID is not set.")
            sys.exit(1)
        config_file_path = os.path.join(parent_dir, self.start_processsparam)
        scanpath = None
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
            # 检查 'basic' 键是否存在，以及 'levelorca' 键是否存在于 'basic' 下
            if 'basic' in config and 'levelorca' in config['basic']:
                scanpath =  config['basic']['levelorca']
            else:
                print(f"键 'basic' 或 'levelorca' 在文件中不存在。")

        if scanpath:
            print("scanpath :",scanpath)
            self.monitor_directory = os.path.join(parent_dir, "augmented_datasets_tmp", scanpath)
            print("self.monitor_directory :",self.monitor_directory)

        self.running = False
        self.last_subdir_count = 0
        self.last_scan_time = datetime.now()

        self.last_have_data= datetime.now()

        self.is_starttingdevice = False

        print("self.device_id:",  self.device_id)

        self.scripmonitor = None
        
        print(f"📱 设备客户端初始化完成")
        print(f"🆔 设备ID: {self.device_id}")
        print(f"🌐 服务器地址: {self.server_url}")
        print(f"📁 监控目录: {self.monitor_directory}")
    
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ 配置文件 {config_file} 不存在，使用默认配置")
            return self._create_default_config(config_file)
        except json.JSONDecodeError as e:
            print(f"❌ 配置文件格式错误: {e}")
            sys.exit(1)
    
    def _create_default_config(self, config_file: str) -> Dict:
        """创建默认配置文件"""
        default_config = {
            "device_id": f"device-{socket.gethostname()}",
            "server_url": "http://localhost:8000",
            "monitor_directory": "/data",
            "report_interval": 30,
            "heartbeat_interval": 60,
            "file_extensions": [".txt", ".csv", ".json", ".log", ".dat"]
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"✅ 已创建默认配置文件: {config_file}")
        except Exception as e:
            print(f"⚠️  无法创建配置文件: {e}")
        
        return default_config
    
    def start(self):
        """启动客户端"""
        print(f"🚀 启动设备客户端...")
        
        # 注册设备
        if not self._register_device():
           print("❌ 设备注册失败，退出程序")
           return
        
        self.running = True
        
        # 启动文件监控线程
        monitor_thread = threading.Thread(target=self._start_file_monitoring, daemon=True)
        monitor_thread.start()
        
        # 启动定时上报线程
        report_thread = threading.Thread(target=self._start_periodic_reporting, daemon=True)
        report_thread.start()
        
        # 启动心跳线程
        heartbeat_thread = threading.Thread(target=self._start_heartbeat, daemon=True)
        heartbeat_thread.start()

        self.connect_websocket()



        self.DoStartProcess()

        
        print("✅ 设备客户端启动成功")
        print("⏹️  按 Ctrl+C 停止客户端")
        
        try:
            while self.running:
                # 保持主线程运行，等待停止信号
                if self.auto_restart:
                    if self._check_devicestatus() == "error"  and self.is_starttingdevice == False:
                        print("❌ 设备状态异常，准备重启")
                        self.is_starttingdevice = True
                        self.DoStopSubProcess()
                        time.sleep(15)
                        self.DoStartProcess()
                        time.sleep(30)
                        self.last_have_data = datetime.now()
                        #self.last_scan_time = datetime.now()
                        self.is_starttingdevice = False

                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n🛑 收到停止信号，正在关闭客户端...")
           
            self.stop()

    def DoStopSubProcess(self):
         if self.scripmonitor is not None:
            pid = self.scripmonitor.pid

            # 将父进程也添加到要终止的列表中（如果它还在运行） 
            try: 
                parent_process = psutil.Process(self.scripmonitor.pid)
                children_processes = parent_process.children(recursive=False) 
                processes_to_terminate = [parent_process] + children_processes  

                for proc in processes_to_terminate:  
                    print(f"  Attempting to terminate PID {proc.pid} (Name: {proc.name()})...")  
                    try:  
                        proc.terminate() # 先尝试温柔地终止  
                    except psutil.NoSuchProcess:  
                        print(f"  Warning: PID {proc.pid} already gone during terminate attempt.")  
                        pass # Process might have already terminated  
                

                gone, alive = psutil.wait_procs(processes_to_terminate, timeout=10)  

                for p in alive:  
                    # 如果超时后仍然存活，则强制杀死  
                    print(f"  PID {p.pid} did not terminate gracefully, killing...")  
                    try:  
                        p.kill() # 强制杀死  
                    except psutil.NoSuchProcess:  
                        print(f"  Warning: PID {p.pid} already gone during kill attempt.")  
                        pass  
            
            except psutil.NoSuchProcess:  
                print(f"Process with PID {pid} not found, likely already terminated.")  
            except Exception as e:  
                print(f"Error terminating process tree for PID {pid}: {e}") 
        

    def DoStartProcess(self):
        self.scripmonitor = subprocess.Popen(["python", self.start_processscripts, 
                                              "--config", self.start_processsparam
                                              ],  stdout=sys.stdout, stderr=sys.stderr, text=True)

    def stop(self):
        """停止客户端"""
        self.running = False
        self.DoStopSubProcess()
        print("✅ 设备客户端已停止")
    
    def _register_device(self) -> bool:
        """向服务器注册设备"""
        try:
            client_info = {
                "ip_address": self._get_local_ip(),
                "hostname": socket.gethostname(),
                "version": "1.0.0",
                "monitor_directory": self.monitor_directory
            }
            
            print("client_info:",client_info)
            response = requests.post(
                f"{self.server_url}/api/collection/devices/{self.device_id}/register",
                json=client_info,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 设备注册成功: {result.get('message')}")
                
                # 更新配置
                device_config = result.get('device_config', {})
                self.report_interval = device_config.get('report_interval', self.report_interval)
                self.heartbeat_interval = device_config.get('heartbeat_interval', self.heartbeat_interval)
                
                return True
            else:
                print(f"❌ 设备注册失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 设备注册异常: {str(e)}")
            return False
    
    def _get_local_ip(self) -> str:
        """获取本机IP地址"""
        try:
            # 连接到远程地址来获取本机IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "127.0.0.1"
 
    # def _scan_directory(self) -> Dict:
    #     """扫描监控目录"""
    #     try:
    #         if not os.path.exists(self.monitor_directory):
    #             return {
    #                 'file_count': 0,
    #                 'total_files': 0,
    #                 'data_size': 0.0,
    #                 'production_rate': 0.0,
    #                 'directory_status': 'missing',
    #                 'error': f"目录不存在: {self.monitor_directory}"
    #             }
            
    #         # 获取允许的文件扩展名
    #         allowed_extensions = self.config.get('file_extensions', ['.txt', '.csv', '.json', '.log', '.dat'])
            
    #         # 扫描文件 
    #         all_files = []
    #         for ext in allowed_extensions:
    #             pattern = os.path.join(self.monitor_directory, f"**/*{ext}")
    #             all_files.extend(glob.glob(pattern, recursive=True))
            
    #         # 过滤出文件（排除目录）
    #         files = [f for f in all_files if os.path.isfile(f)]
    #         file_count = len(files)
            
    #         # 计算文件总大小
    #         total_size = 0
    #         for file_path in files:
    #             try:
    #                 total_size += os.path.getsize(file_path)
    #             except:
    #                 continue
            
    #         data_size_mb = total_size / (1024 * 1024)  # 转换为MB
            
    #         # 计算生产效率（文件/小时）
    #         current_time = datetime.now()
    #         time_diff = (current_time - self.last_scan_time).total_seconds() / 3600
    #         new_files = file_count - self.last_file_count
    #         production_rate = new_files / time_diff if time_diff > 0 else 0
            
    #         # 更新记录
    #         self.last_file_count = file_count
    #         self.last_scan_time = current_time
            
    #         return {
    #             'file_count': new_files,  # 新增文件数
    #             'total_files': file_count,  # 总文件数
    #             'data_size': data_size_mb,
    #             'production_rate': max(0, production_rate),
    #             'directory_status': 'normal'
    #         }
            
    #     except Exception as e:
    #         return {
    #             'file_count': 0,
    #             'total_files': 0,
    #             'data_size': 0.0,
    #             'production_rate': 0.0,
    #             'directory_status': 'error',
    #             'error': str(e)
    #         }
    


    def _get_start_time(self):
        try:
            with open("file_stats.json", 'r') as f:
                data = json.load(f)
                #获取上次运行结束时间，比较上次运行结束时间日期和当前时间日期，如果相同，则返回上次运行结束时间，否则将日期设置为当天的AM 08:00
                current_time = datetime.now()
                datetime_obj = datetime.strptime(data['last_end'], "%Y-%m-%d %H:%M:%S")
                return datetime_obj
                #获取上次运行结束时间
                #return datetime.strptime(data['last_end'], "%Y-%m-%d %H:%M:%S")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
                return None
        
    def _save_last_run(self,end_time):
        data = {'last_end': end_time.strftime("%Y-%m-%d %H:%M:%S")}
        with open("file_stats.json", 'w') as f:
            json.dump(data, f)



    def _scan_directory(self) -> Dict:
        """扫描监控目录下的第一层子目录"""
        try:
            if not os.path.exists(self.monitor_directory):
                print("Error monitor_directory:",self.monitor_directory)
                return {
                    'subdir_count': 0,
                    'directory_status': 'missing',
                    'error': f"目录不存在: {self.monitor_directory}"
                }
            
            # 获取第一层子目录
            subdirs = [d for d in os.listdir(self.monitor_directory) if os.path.isdir(os.path.join(self.monitor_directory, d))]
            subdir_count = len(subdirs)

            start_time = self._get_start_time()
            addnum = 0

            for subdir in subdirs:
                #print("Found subdirectory:", subdir)
                createtime = os.path.getctime(os.path.join(self.monitor_directory, subdir))
                #print("Creation time:", datetime.fromtimestamp(createtime))
                if start_time and createtime < start_time.timestamp():
                   # print("Skipping subdirectory:", subdir)
                    continue
                  
                addnum += 1
                #print("Adding subdirectory:", subdir)

            # 计算生产效率（子目录/小时）
            current_time = datetime.now()
            time_diff = (current_time - self.last_scan_time).total_seconds() / 3600
            new_subdirs = addnum  #subdir_count - self.last_subdir_count
            production_rate = new_subdirs / time_diff if time_diff > 0 else 0
            
            # 更新记录
            self.last_subdir_count = subdir_count
            self.last_scan_time = current_time
            if(new_subdirs > 0):
                self.last_have_data = current_time
            print("new_subdirs.................",new_subdirs)
            
            self._save_last_run(current_time)

            return {
                'subdir_count': new_subdirs,  # 新增子目录数
                'total_subdirs': subdir_count,  # 总子目录数
                'data_size': 0,
                'production_rate': max(0, production_rate),
                'directory_status': 'normal'
            }
            
        except Exception as e:
            return {
                'subdir_count': 0,
                'total_subdirs': 0,
                'data_size': 0,
                'production_rate': 0.0,
                'directory_status': 'error',
                'error': str(e)
            }
    
    def _report_production_data(self):
        """上报产量数据到服务器"""
        try:
            scan_result = self._scan_directory()
            print("scan_result:",scan_result)
            
            report_data = {
                "device_id": self.device_id,
                "file_count": scan_result['subdir_count'],
                "total_files": scan_result['total_subdirs'],
                "data_size": scan_result['data_size'],
                "production_rate": scan_result['production_rate'],
                "timestamp": datetime.now().isoformat(),
                "directory_status": scan_result['directory_status']
            }
            

            print("report_data:",report_data)
            response = requests.post(
                f"{self.server_url}/api/collection/report",
                json=report_data,
                timeout=10
            )


            
            if response.status_code == 200:
                result = response.json()
                print(f"📊 数据上报成功: 文件{scan_result['total_subdirs']}个, 新增{scan_result['subdir_count']}个, 大小{scan_result['data_size']:.2f}MB")
            else:
                print(f"❌ 数据上报失败: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 数据上报异常  2222222222222222: {str(e)}")
    
    def _check_devicestatus(self):
        """检查设备状态"""

        diff_have_data = datetime.now() - self.last_have_data

        if diff_have_data.total_seconds() > self.nodata_timeout:
            print("❌ 设备长时间没有数据，检查设备状态")
            return "error"
        else:
            return "online"

    def _send_heartbeat(self):
        """发送心跳到服务器"""
        try:
            heartbeat_data = {
                "device_id": self.device_id,
                "status": self._check_devicestatus(),
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "hostname": socket.gethostname(),
                    "ip_address": self._get_local_ip(),
                    "directory_exists": os.path.exists(self.monitor_directory)
                }
            }
            
            response = requests.post(
                f"{self.server_url}/api/collection/heartbeat",
                json=heartbeat_data,
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"💓 心跳发送成功")
            else:
                print(f"❌ 心跳发送失败: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 心跳发送异常: {str(e)}")
    
    def _start_file_monitoring(self):
        """启动文件监控"""
        print(f"👁️  开始监控目录: {self.monitor_directory}")
        
        if not os.path.exists(self.monitor_directory):
            print(f"⚠️  监控目录不存在，创建目录: {self.monitor_directory}")
            try:
                os.makedirs(self.monitor_directory, exist_ok=True)
            except Exception as e:
                print(f"❌ 无法创建目录: {e}")
                return
        
        # 使用watchdog监控文件变化
        event_handler = FileChangeHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.monitor_directory, recursive=True)
        observer.start()
        
        try:
            while self.running:
                time.sleep(1)
        except:
            pass
        finally:
            observer.stop()
            observer.join()
    
    def _start_periodic_reporting(self):
        """启动定时上报"""
        print(f"⏰ 开始定时上报，间隔 {self.report_interval} 秒")
        
        while self.running:
            try:
                self._report_production_data()
                time.sleep(self.report_interval)
            except Exception as e:
                print(f"❌ 定时上报异常: {e}")
                time.sleep(self.report_interval)
    
    def _start_heartbeat(self):
        """启动心跳发送"""
        print(f"💓 开始心跳发送，间隔 {self.heartbeat_interval} 秒")
        
        while self.running:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                print(f"❌ 心跳发送异常: {e}")
                time.sleep(self.heartbeat_interval)

    def connect_websocket(self):
        """连接WebSocket"""
        try:
            print(f"连接WebSocket: {self.server_ws_url}")
            self.ws = websocket.WebSocketApp(
                self.server_ws_url,
                on_open=self._on_websocket_open,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close
            )
            
            # 在后台线程中运行WebSocket
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()
            
        except Exception as e:
            print(f"WebSocket连接失败: {str(e)}")
    
    def _on_websocket_open(self, ws):
        """WebSocket连接打开"""
        self.connected = True
        self.retry_count = 0
        print("WebSocket连接成功")
        
        # 发送设备身份认证
        auth_message = {
            "type": "auth",
            "device_id":   self.device_id,
            "device_name":   self.device_id,#settings.DEVICE_NAME,
            "timestamp": datetime.now().isoformat()
        }
        ws.send(json.dumps(auth_message))
    
    def _on_websocket_message(self, ws, message):
        """接收WebSocket消息"""
        try:
            data = json.loads(message)
            print(f"收到服务器消息: {data}")
            
            # 处理服务器指令
            if data.get("type") == "command":
                self._handle_server_command(data)
            elif data.get("type") == "config_update":
                self._handle_config_update(data)
                
        except Exception as e:
            print(f"处理WebSocket消息失败: {str(e)}")

    def _on_websocket_error(self, ws, error):
        """WebSocket错误"""
        print(f"WebSocket错误: {error}")
        self.connected = False
    
    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """WebSocket连接关闭"""
        self.connected = False
        print("WebSocket连接关闭，尝试重连...")
        
        # 重连逻辑
        if self.retry_count < 3:
            self.retry_count += 1
            time.sleep(10)
            self.connect_websocket()
    
    def _handle_server_command(self, command_data: Dict):
        """处理服务器指令"""
        command = command_data.get("command")
        
        if command == "restart_collection":
            print("收到重启采集指令")
            self.DoStopSubProcess()
            time.sleep(15)
            self.DoStartProcess()

            # if self.data_callback:
            #     self.data_callback("restart")
        elif command == "update_config":
            print("收到配置更新指令")
            # 处理配置更新逻辑
        else:
            print(f"未知服务器指令: {command}")

    def _handle_config_update(self, config_data: Dict):
        """处理配置更新"""
        print("处理配置更新")
        # 实现配置更新逻辑
    
    def send_websocket_message(self, message_data: Dict):
        """发送WebSocket消息"""
        if self.ws and self.connected:
            try:
                self.ws.send(json.dumps(message_data))
                return True
            except Exception as e:
                print(f"发送WebSocket消息失败: {str(e)}")
                return False
        return False
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected and self.registered
    

class FileChangeHandler(FileSystemEventHandler):
    """文件变化处理器"""
    
    def __init__(self, client: DeviceClient):
        self.client = client
        self.last_report = time.time()
    
    def on_created(self, event):
        """文件创建事件"""
        if not event.is_directory:
            print(f"📁 新文件: {event.src_path}")
            self._trigger_report()
    
    def on_modified(self, event):
        """文件修改事件"""
        if not event.is_directory:
            self._trigger_report()
    
    def _trigger_report(self):
        """触发数据上报（限制频率）"""
        current_time = time.time()
        if current_time - self.last_report > 10:  # 10秒内最多上报一次
            self.last_report = current_time
            threading.Thread(target=self.client._report_production_data, daemon=True).start()

    

def main():
    """主函数"""
    print("🚀 数据采集设备客户端启动")
    parser = argparse.ArgumentParser(description='Run parallelism augmenta of the script ')
    
    # 解析命令行参数
  #  config_file = sys.argv[1] if len(sys.argv) > 1 else "client_config.json"

   # parser.add_argument('--monitorpath', type=str, 
    #                            help='Configuration file path (YAML or JSON)')
    parser.add_argument('--configfile', type=str, 
                                help='Configuration file path (YAML or JSON)')
    
    args = parser.parse_args()

    print("client args:",args)
    
    # 创建并启动客户端
    client = DeviceClient(args.configfile)

  #  client.monitor_directory = args.monitorpath
    client.start()


        # 3. 连接WebSocket
 
   # client.data_callback = self._handle_server_callback



    #augmented_path = f"{current_file_path}/augmented_datasets_tmp/{args.levelorca}"
   # print("augmented_path:",augmented_path)

if __name__ == "__main__":
    main()
