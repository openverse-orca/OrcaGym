import os
import sys
import time
import fcntl
import contextlib
import signal

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


# 全局锁跟踪，用于处理同一进程内的重入
_lock_tracking = {}


def cleanup_zombie_locks(directory_path):
    """
    清理指定目录下的僵尸锁文件
    
    Args:
        directory_path: 要清理的目录路径
    """
    if not os.path.exists(directory_path):
        return
    
    cleaned_count = 0
    for filename in os.listdir(directory_path):
        if filename.endswith('.lock'):
            lock_path = os.path.join(directory_path, filename)
            try:
                with open(lock_path, 'r') as f:
                    pid_str = f.read().strip()
                    if pid_str.isdigit():
                        pid = int(pid_str)
                        # 检查进程是否还在运行
                        try:
                            os.kill(pid, 0)  # 发送信号0检查进程是否存在
                        except (OSError, ProcessLookupError):
                            # 进程不存在，删除僵尸锁文件
                            os.unlink(lock_path)
                            cleaned_count += 1
                            print(f"清理僵尸锁文件: {lock_path}")
            except (OSError, ValueError):
                # 锁文件损坏，删除它
                try:
                    os.unlink(lock_path)
                    cleaned_count += 1
                    print(f"清理损坏的锁文件: {lock_path}")
                except OSError:
                    pass
    
    if cleaned_count > 0:
        print(f"共清理了 {cleaned_count} 个僵尸锁文件")


@contextlib.asynccontextmanager
async def file_lock(file_path, timeout=30, non_blocking=False):
    """
    文件锁上下文管理器，防止多进程同时访问同一个文件
    
    Args:
        file_path: 要锁定的文件路径
        timeout: 锁获取超时时间（秒），默认30秒
        non_blocking: 是否使用非阻塞模式，默认False
    """
    lock_path = file_path + '.lock'
    lock_file = None
    current_pid = os.getpid()
    
    # 检查是否已经在当前进程中持有此锁（重入检查）
    if file_path in _lock_tracking:
        if _lock_tracking[file_path] == current_pid:
            # 重入情况，直接返回
            yield
            return
        else:
            # 其他进程持有锁，需要等待
            pass
    
    try:
        # 清理可能存在的僵尸锁文件
        if os.path.exists(lock_path):
            try:
                with open(lock_path, 'r') as f:
                    pid_str = f.read().strip()
                    if pid_str.isdigit():
                        pid = int(pid_str)
                        # 检查进程是否还在运行
                        try:
                            os.kill(pid, 0)  # 发送信号0检查进程是否存在
                        except (OSError, ProcessLookupError):
                            # 进程不存在，删除僵尸锁文件
                            os.unlink(lock_path)
            except (OSError, ValueError):
                # 锁文件损坏，删除它
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
        
        # 打开锁文件
        lock_file = open(lock_path, 'w')
        lock_file.write(str(current_pid))
        lock_file.flush()
        
        # 获取文件锁
        if non_blocking:
            # 非阻塞模式
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TimeoutError(f"无法获取文件锁 {file_path}，文件可能被其他进程占用")
        else:
            # 阻塞模式，但使用信号处理超时
            def timeout_handler(signum, frame):
                raise TimeoutError(f"获取文件锁超时 {file_path}")
            
            # 设置超时信号
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            finally:
                # 取消超时信号
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        # 记录当前进程持有此锁
        _lock_tracking[file_path] = current_pid
        
        try:
            yield
        finally:
            # 释放锁
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            lock_file = None
            
            # 从跟踪中移除
            if file_path in _lock_tracking:
                del _lock_tracking[file_path]
            
            # 清理锁文件
            try:
                os.unlink(lock_path)
            except OSError:
                pass
                
    except Exception:
        # 发生异常时确保清理
        if lock_file:
            try:
                lock_file.close()
            except:
                pass
        if file_path in _lock_tracking:
            del _lock_tracking[file_path]
        try:
            os.unlink(lock_path)
        except OSError:
            pass
        raise

