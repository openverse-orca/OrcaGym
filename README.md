[中文版](doc/README-CN.md)
[中文版](doc/README-CN.md)
# OrcaGym Project

[![PyPI version](https://img.shields.io/pypi/v/orca-gym)](https://pypi.org/project/orca-gym/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

Welcome to the OrcaGym project! OrcaGym is a high-performance simulation environment designed to be compatible with the OpenAI Gym/Gymnasium interface, enabling seamless integration with existing reinforcement learning algorithms. Developed in conjunction with Songying Technology's OrcaStudio platform, OrcaGym provides robust support for multi-physics engines and ray-traced rendering while maintaining programming interface compatibility with popular RL frameworks.

## Background
Robotic simulation serves as a critical platform for embodied intelligence training, requiring both physical accuracy and scalable infrastructure. Traditional solutions often face trade-offs between fidelity and computational efficiency, particularly when scaling to distributed systems. OrcaGym bridges this gap by combining real-time physics simulation with cloud-native architecture, enabling researchers to prototype algorithms in high-fidelity environments and deploy them at scale.

## Purpose
OrcaGym aims to:

1. Provide a GPU-accelerated simulation environment compatible with OpenAI Gym/Gymnasium APIs
2. Support multiple physics backends (Mujoco, PhysX, ODE) through OrcaStudio integration
3. Enable distributed training scenarios across heterogeneous computing nodes
4. Deliver photorealistic rendering via ray tracing for vision-based RL tasks

## Key Features
- 🎮 **Gym/Gymnasium API Compatibility** - Seamless integration with existing RL algorithms
- ⚡ **Multi-Physics Backends** - Simultaneous Mujoco/PhysX/ODE simulations
- 🌐 **Distributed Deployment** - Hybrid local/remote operation via gRPC
- 🔍 **Ray-Traced Rendering** - Photorealistic visual observations
- 🤖 **Multi-Agent Support** - Native heterogeneous agent management

## Installation
```bash
# Clone repository
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym

# Initialize assets
git lfs install
git lfs pull

# Create conda environment
conda create -n orca python=3.12
conda activate orca

# Install core package
pip install -e .
```


## OrcaStudio Configuration

Download and install OrcaStudio from [official portal](http://orca3d.cn/)

## Performance Considerations
* Performance: High-fidelity rendering and complex physics simulations can be computationally intensive. Ensure your hardware meets the requirements for running OrcaStudio effectively.
* Configuration: Properly configure OrcaStudio to match your simulation needs. Refer to the OrcaStudio documentation for detailed configuration options.
* Compatibility: While OrcaGym aims for compatibility with OpenAI Gym, some advanced features may require additional configuration or modification of existing Gym environments.

## Contributing
We welcome contributions to the OrcaGym project. If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request on our GitHub repository.

## Citation
```bibtex
@software{OrcaGym2024,  
  author = {Songying Technology},  
  title = {OrcaGym: Cloud-Native Robotics Simulation Platform},  
  year = {2024},  
  publisher = {GitHub},  
  journal = {GitHub repository},  
  howpublished = {\url{https://github.com/openverse-orca/OrcaGym}}  
}  
```

## License
Distributed under MIT License. See **LICENSE** for details.

## Contact
For any inquiries or support, please contact us at huangwei@openverse.com.cn

---

We hope you find OrcaGym a valuable tool for your robotics and reinforcement learning research. Happy simulating!
