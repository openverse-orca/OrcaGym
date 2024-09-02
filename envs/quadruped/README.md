# Quadruped Envirement

这是一个四足机器狗行走的例子，采用MPC控制算法。

* **Ported form** https://github.com/iit-DLSLab/Quadruped-PyMPC
* **仅支持Linux平台**

## How To Install

### 安装依赖包。

``` bash
cd $(you-path-to-orcagym)/envs/quadruped
conda activate orca_gym_test
pip install -r requirements.txt
```

### 编译安装acados库

acados 是一个用于快速解决非线性优化问题的库，特别适用于非线性模型预测控制 (Nonlinear Model Predictive Control, NMPC) 和非线性优化问题。它基于高度优化的数值算法，如Interior Point Method和Sequential Quadratic Programming，旨在提供高效且可靠的解决方案。

``` bash
cd $(you-worksapce)
git clone https://github.com/acados/acados.git
cd acados
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make install -j4
pip install -e ./../interfaces/acados_template
```

### 配置环境变量

1. 你可以选择将路径添加到bash环境变量：
``` bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${path_to_acados}/lib"
export ACADOS_SOURCE_DIR="${path_to_acados}"
```

2. 相对的，你也可以在example中修改py文件中的环境变量，适配你自己的安装路径 

* 在 `examples/run_quadruped_ctrl.py` 文件中，修改下面的路径适配你的acados安装路径：

``` python
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ":${path_to_acados}/lib"
    os.environ['ACADOS_SOURCE_DIR'] = "${path_to_acados}"
```

### 运行示例

在 `examples/quadruped` 下，运行：

```bash
python run_quadruped_ctrl.py --grpc_address localhost
```

