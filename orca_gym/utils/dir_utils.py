import os
import sys
import time

def create_tmp_dir(dir_name):
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 设置要检查的目录路径
    dir_path = os.path.join(current_dir, dir_name)

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录 '{dir_path}' 已创建。")


def formate_now():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


import fcntl
import os
import atexit
import psutil
import sys

def safe_remove_lockfile():
    lock_file = "/tmp/.first_process_lock"
    """安全删除锁文件（如果存在且未被任何进程使用）"""
    try:
        # 检查文件是否被其他进程打开
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for item in proc.open_files():
                    if item.path == lock_file:
                        return  # 有其他进程在使用，不要删除
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # 安全删除锁文件
        if os.path.exists(lock_file):
            os.unlink(lock_file)
            print(f"🧹 清理残留锁文件: {lock_file}")
    except Exception as e:
        print(f"⚠️ 清理锁文件时出错: {e}")

def is_first_process():
    """检查当前进程是否是第一个初始化的进程"""
    lock_file = "/tmp/.first_process_lock"
    
    # 在所有进程启动前尝试清理残留锁文件
    if not hasattr(is_first_process, '_cleanup_done'):
        safe_remove_lockfile(lock_file)
        is_first_process._cleanup_done = True
    
    # 打开锁文件（自动创建）
    lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    
    try:
        # 尝试获取非阻塞的独占锁
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        # 获取锁成功，说明是第一个进程
        # 注册退出时的清理函数
        def cleanup():
            try:
                os.close(lock_fd)
                if os.path.exists(lock_file):
                    os.unlink(lock_file)
                    print(f"🧹 进程退出，清理锁文件")
            except Exception as e:
                pass  # 忽略退出时的错误
        
        atexit.register(cleanup)
        return True
    except (IOError, BlockingIOError):
        # 获取锁失败，说明已有其他进程持有锁
        os.close(lock_fd)
        return False