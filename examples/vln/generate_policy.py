import os, sys

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(project_root)
project_root = project_root+'/vln_policy'
# project_root = '/home/orca3d/OrcaGym'
# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


# os.chdir(project_root+'/vln_policy')
os.chdir(project_root)
from vlfm.utils import generate_dummy_policy as gene
# import vlfm.utils.generate_dummy_policy as gene_po
assert os.path.isdir("data"), "Missing 'data/' directory!"

gene.save_dummy_policy("data/dummy_policy.pth")