# Quadruped Environment

This is an example of a quadruped robot dog walking, using the MPC control algorithm.

* **Ported from** https://github.com/iit-DLSLab/Quadruped-PyMPC
* **Linux platform only**

## How To Install
If you are an **installation version user**, please skip the following steps and directly run the simulation（Run the Example）.
### Install Dependencies

This guide assumes that you have already installed the basic dependencies as described in the [Readme](https://github.com/openverse-orca/OrcaGym/blob/main/README.md) document. Next, we will install additional dependencies needed for the quadruped robot example:

```bash
cd $(your-path-to-orcagym)
cd envs/quadruped
conda activate orca_gym_test
pip install -r requirements.txt

# Install cuda+jax
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Compile and Install acados Library

acados is a library for solving nonlinear optimization problems quickly, particularly suited for Nonlinear Model Predictive Control (NMPC) and nonlinear optimization problems. It is based on highly optimized numerical algorithms such as Interior Point Method and Sequential Quadratic Programming, designed to provide efficient and reliable solutions.

``` bash
cd $(your-path-to-orcagym)
git submodule update --init --recursive
cd 3rd_party/acados
mkdir build
cd build
cmake ..
make install -j10
pip install -e ./../interfaces/acados_template
cp ./../../bins/t_renderer-v0.0.34-linux ./../bin/t_renderer
chmod +x ./../bin/t_renderer
```
