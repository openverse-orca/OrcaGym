from PIL import Image
import argparse

import sys
import os

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))
# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)

from orca_gym.utils.dir_utils import create_tmp_dir, formate_now

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



def split_channel(input_image_path, save_dir):
    # 打开原始图片
    image = Image.open(input_image_path)

    # 分离出RGB通道
    r, g, b = image.split()

    # 保存分离后的图片
    r.save(save_dir + input_image_path.split('/')[-1].split('.')[0] + '_r.png')
    g.save(save_dir + input_image_path.split('/')[-1].split('.')[0] + '_g.png')
    b.save(save_dir + input_image_path.split('/')[-1].split('.')[0] + '_b.png')
    _logger.info("Split channel done!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper to Process texture image')
    parser.add_argument('--image', type=str, help='The path to the input image')
    parser.add_argument('--task', type=str, help='The task to run (split_channel / reverse_channel)', default='split_channel')
    args = parser.parse_args()
    
    input_image_path = args.image
    task = args.task
    
    create_tmp_dir('texture_processer_tmp')
    tmp_dir = current_file_path + '/texture_processer_tmp/'
    
    if input_image_path is None:
        _logger.info("Please provide the path to the input image")
        exit(1)
    
    if task == 'split_channel':    
        split_channel(input_image_path, tmp_dir)