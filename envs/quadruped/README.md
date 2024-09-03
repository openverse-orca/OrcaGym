# Quadruped Environment

This is an example of a quadruped robot dog walking, using the MPC control algorithm.

* **Ported from** https://github.com/iit-DLSLab/Quadruped-PyMPC
* **Linux platform only**

## How To Install

### Install Dependencies

```bash
cd $(your-path-to-orcagym)/envs/quadruped
conda activate orca_gym_test
pip install -r requirements.txt

# Install cuda+jax
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


### Compile and Install acados Library

acados is a library for solving nonlinear optimization problems quickly, particularly suited for Nonlinear Model Predictive Control (NMPC) and nonlinear optimization problems. It is based on highly optimized numerical algorithms such as Interior Point Method and Sequential Quadratic Programming, designed to provide efficient and reliable solutions.

``` bash
cd $(your-workspace)
git clone https://github.com/acados/acados.git
cd acados
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make install -j4
pip install -e ./../interfaces/acados_template
```

### Set Environment Variables

1. You can choose to add the paths to the bash environment variables:
``` bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${path_to_acados}/lib"
export ACADOS_SOURCE_DIR="${path_to_acados}"
```

2. Alternatively, you can modify the environment variables in the example's Python file to fit your own installation path.

* In the `examples/run_quadruped_ctrl.py` file, modify the following paths to match your acados installation path:

``` python
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ":${path_to_acados}/lib"
    os.environ['ACADOS_SOURCE_DIR'] = "${path_to_acados}"
```

### Run the Example

In the `examples/quadruped` directory, run:

```bash
python run_quadruped_ctrl.py --grpc_address localhost
```

### Keyboard Controls

* W/S: Move forward/backward
* A/D: Turn left/right
* Space: Stop

