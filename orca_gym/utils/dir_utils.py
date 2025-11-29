import os
import sys
import time
import contextlib
import signal

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


# 平台检测和条件导入
if sys.platform == 'win32':
    import msvcrt
    HAS_FCNTL = False
else:
    import fcntl
    HAS_FCNTL = True

def create_tmp_dir(dir_name):
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 设置要检查的目录路径
    dir_path = os.path.join(current_dir, dir_name)

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        _logger.info(f"目录 '{dir_path}' 已创建。")


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
                            _logger.info(f"清理僵尸锁文件: {lock_path}")
            except (OSError, ValueError):
                # 锁文件损坏，删除它
                try:
                    os.unlink(lock_path)
                    cleaned_count += 1
                    _logger.info(f"清理损坏的锁文件: {lock_path}")
                except OSError:
                    pass
    
    if cleaned_count > 0:
        _logger.info(f"共清理了 {cleaned_count} 个僵尸锁文件")


def _windows_file_lock(lock_file, non_blocking=False, timeout=30):
    """
    Windows平台的文件锁实现
    
    Args:
        lock_file: 文件对象
        non_blocking: 是否使用非阻塞模式
        timeout: 超时时间（秒）
    
    Returns:
        bool: 是否成功获取锁
    """
    if non_blocking:
        # 非阻塞模式
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    else:
        # 阻塞模式，使用轮询实现超时
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                time.sleep(0.1)  # 等待100ms后重试
        return False


def _windows_file_unlock(lock_file):
    """
    Windows平台的文件锁释放
    
    Args:
        lock_file: 文件对象
    """
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


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
        if HAS_FCNTL:
            # Linux/Unix平台使用fcntl
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
        else:
            # Windows平台使用msvcrt
            if not _windows_file_lock(lock_file, non_blocking, timeout):
                raise TimeoutError(f"无法获取文件锁 {file_path}，文件可能被其他进程占用")
        
        # 记录当前进程持有此锁
        _lock_tracking[file_path] = current_pid
        
        try:
            yield
        finally:
            # 释放锁
            if HAS_FCNTL:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            else:
                _windows_file_unlock(lock_file)
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

