from ftplib import FTP, all_errors
import os
import tarfile
from tqdm import tqdm
import datetime
import re

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def connect_to_ftp(ftp_host, ftp_user, ftp_password, ftp_port=21):
    """连接到 FTP 服务器并返回 FTP 对象。"""
    ftp = FTP()
    try:
        ftp.connect(ftp_host, ftp_port, timeout=10)
        ftp.login(user=ftp_user, passwd=ftp_password)
        _logger.info(f"成功连接到 {ftp_host}")
    except all_errors as e:
        _logger.info(f"连接失败：{e}")
        return None
    return ftp

def list_directory(ftp, directory):
    """列出远程 FTP 目录内容。"""
    try:
        files = ftp.nlst(directory)
        _logger.info(f"目录 {directory} 内容: {files}")
        return files
    except all_errors as e:
        _logger.info(f"列出目录错误：{e}")
        return []

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建。"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        _logger.info(f"创建目录: {directory}")

def download_file(ftp, remote_file_path, local_file_path):
    """带有进度条的文件下载。"""
    try:
        ensure_directory_exists(os.path.dirname(local_file_path))

        # 获取文件大小
        total_size = ftp.size(remote_file_path)

        # 打开文件并开始下载
        with open(local_file_path, 'wb') as local_file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_file_path) as pbar:
                def callback(data):
                    local_file.write(data)
                    pbar.update(len(data))  # 更新进度条

                # 下载文件
                ftp.retrbinary(f'RETR {remote_file_path}', callback)

        _logger.info(f"文件已下载到 {local_file_path}")

    except all_errors as e:
        _logger.info(f"下载过程中发生错误：{e}")

def extract_tar_xz(file_path, extract_to):
    """解压 tar.xz 文件到指定目录。"""
    try:
        ensure_directory_exists(extract_to)
        with tarfile.open(file_path, "r:xz") as tar:
            tar.extractall(path=extract_to)
        _logger.info(f"文件已解压到 {extract_to}")

    except Exception as e:
        _logger.info(f"解压错误：{e}")

def find_latest_version(files, year, month):
    """根据当前年月找到最接近的版本。"""
    pattern = re.compile(r'v(\d{4})_(\d{1,2})_x\.tar\.xz')
    versions = []

    # 提取文件名并匹配符合格式的版本
    for file in files:
        filename = os.path.basename(file)  # 去除路径，保留文件名
        match = pattern.match(filename)
        if match:
            file_year, file_month = int(match.group(1)), int(match.group(2))
            versions.append((file_year, file_month, filename))

    # 根据年月排序，找到最接近的版本
    versions.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # 找到当前年月或最近的版本
    for v_year, v_month, filename in versions:
        if (v_year < year) or (v_year == year and v_month <= month):
            _logger.info(f"找到匹配的版本: {filename}")
            return filename

    _logger.info("未找到匹配的版本")
    return None

def delete_file(file_path):
    """删除指定文件。"""
    try:
        os.remove(file_path)
        _logger.info(f"压缩包 {file_path} 已删除")
    except Exception as e:
        _logger.info(f"删除文件时出错：{e}")

def main():
    # 获取当前年月
    now = datetime.datetime.now()
    year, month = now.year, now.month

    # FTP 服务器信息
    ftp_host = "47.116.64.88"
    ftp_user = "Download"  # FTP 用户名（对外客户使用）
    ftp_password = "openverse"  # FTP 密码
    ftp_port = 20080  # FTP 端口
    remote_directory = "/orca-studio-projects"  # 远程目录路径

    # 获取当前脚本的运行目录
    current_working_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义解压目录为当前目录的同级目录：orca-studio-projects
    parent_dir = os.path.dirname(current_working_dir)  # 获取当前目录的上一级目录
    extract_folder = os.path.join(parent_dir, "orca-studio-projects")
    ensure_directory_exists(extract_folder)  # 确保解压目录存在

    _logger.info(f"本地下载目录: {extract_folder}")
    _logger.info(f"解压目录: {extract_folder}")

    # 连接到 FTP 服务器
    ftp = connect_to_ftp(ftp_host, ftp_user, ftp_password, ftp_port)
    if not ftp:
        return

    # 列出远程目录内容
    files = list_directory(ftp, remote_directory)

    # 找到最新的版本
    version = find_latest_version(files, year, month)
    if not version:
        _logger.info("没有找到符合条件的版本，程序退出。")
        ftp.quit()
        return

    # 下载文件
    local_path = os.path.join(extract_folder, version)  # 下载到解压目录
    remote_path = f"{remote_directory}/{version}"
    download_file(ftp, remote_path, local_path)

    # 解压文件
    extract_tar_xz(local_path, extract_folder)

    # 删除压缩包
    delete_file(local_path)

    # 关闭 FTP 连接
    try:
        ftp.quit()
        _logger.info("FTP 连接已关闭")
    except OSError as e:
        _logger.info(f"关闭 FTP 连接时出错：{e}")

if __name__ == "__main__":
    main()
