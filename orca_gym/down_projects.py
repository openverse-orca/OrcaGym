from ftplib import FTP, all_errors
import os
import tarfile
from tqdm import tqdm
import datetime
import re

def connect_to_ftp(ftp_host, ftp_user, ftp_password, ftp_port=21):
    """连接到 FTP 服务器并返回 FTP 对象。"""
    ftp = FTP()
    try:
        ftp.connect(ftp_host, ftp_port, timeout=10)
        ftp.login(user=ftp_user, passwd=ftp_password)
        print(f"成功连接到 {ftp_host}")
    except all_errors as e:
        print(f"连接失败：{e}")
        return None
    return ftp

def list_directory(ftp, directory):
    """列出远程 FTP 目录内容。"""
    try:
        files = ftp.nlst(directory)
        print(f"目录 {directory} 内容: {files}")
        return files
    except all_errors as e:
        print(f"列出目录错误：{e}")
        return []

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建。"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

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

        print(f"文件已下载到 {local_file_path}")

    except all_errors as e:
        print(f"下载过程中发生错误：{e}")

def extract_tar_xz(file_path, extract_to):
    """解压 tar.xz 文件到指定目录。"""
    try:
        ensure_directory_exists(extract_to)
        with tarfile.open(file_path, "r:xz") as tar:
            tar.extractall(path=extract_to)
        print(f"文件已解压到 {extract_to}")

    except Exception as e:
        print(f"解压错误：{e}")

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
            print(f"找到匹配的版本: {filename}")
            return filename

    print("未找到匹配的版本")
    return None

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

    # 本地路径
    local_folder = os.path.abspath("/home/orcatest/OrcaWorkPath/orca-studio-projects")
    extract_folder = local_folder  # 解压目录与下载目录一致

    print(f"本地下载目录: {local_folder}")
    print(f"解压目录: {extract_folder}")

    # 连接到 FTP 服务器
    ftp = connect_to_ftp(ftp_host, ftp_user, ftp_password, ftp_port)
    if not ftp:
        return

    # 列出远程目录内容
    files = list_directory(ftp, remote_directory)

    # 找到最新的版本
    version = find_latest_version(files, year, month)
    if not version:
        print("没有找到符合条件的版本，程序退出。")
        ftp.quit()
        return

    # 下载文件
    local_path = os.path.join(local_folder, version)
    remote_path = f"{remote_directory}/{version}"
    download_file(ftp, remote_path, local_path)

    # 解压文件
    extract_tar_xz(local_path, extract_folder)

    # 关闭 FTP 连接
    try:
        ftp.quit()
        print("FTP 连接已关闭")
    except OSError as e:
        print(f"关闭 FTP 连接时出错：{e}")

if __name__ == "__main__":
    main()
