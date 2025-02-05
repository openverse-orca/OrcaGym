[中文版](doc/README-CN.md)
# OrcaGym Project
Welcome to the OrcaGym project! OrcaGym is an enhanced simulation environment based on the OpenAI Gymnasium framework, designed for seamless integration with existing OpenAI Gym simulation environments and algorithms. Developed by Songying Technology, OrcaStudio offers robust support for various physics engines and ray-tracing rendering, delivering both physical and visual precision. This document serves as an introduction to OrcaGym, its background, purpose, usage, and important considerations.

## Background
In the realm of robotics simulation, having a versatile and accurate environment is crucial for developing and testing algorithms. OpenAI Gym has been a cornerstone in this space, providing a standardized interface for reinforcement learning (RL) tasks. However, the need for more advanced features, such as support for multiple physics engines and high-fidelity rendering, led to the development of OrcaStudio. OrcaGym bridges the gap between OpenAI Gym and OrcaStudio, enabling researchers and developers to leverage the advanced capabilities of OrcaStudio while maintaining compatibility with OpenAI Gym environments and algorithms.

## Purpose
The primary goal of OrcaGym is to enhance the capabilities of OpenAI Gym by integrating it with the OrcaStudio simulation platform. This integration allows users to:

1. Leverage Multiple Physics Engines: OrcaStudio supports Mujoco, PhysX, ODE, and more, providing users with flexibility in choosing the most suitable physics engine for their tasks.
2. Achieve High-Fidelity Rendering: With ray-tracing support, OrcaStudio offers visually precise simulations, essential for tasks requiring accurate visual feedback.
3. Enable Distributed Deployment: OrcaGym and OrcaStudio can run on the same node or across different nodes, facilitating distributed deployment and large-scale AI cluster training.

## Features
* **Compatibility with OpenAI Gym:** Seamless integration with existing OpenAI Gym environments and algorithms.
* **Multi-Physics Engine Support:** Choose from Mujoco, PhysX, ODE, and more.
* **High-Fidelity Rendering:** Ray-tracing support for precise visual simulations.
* **Distributed Deployment:** Run simulations on the same or different nodes, supporting large-scale AI training.
* **Ease of Use:** Simple interface to transition from OpenAI Gym to OrcaGym.

## Installation
To install OrcaGym, follow these steps:

1. **Clone the Repository:**
If you are a release version user, please skip the following steps. The release version will automatically configure the environment for you, and you won't need to manually download and configure dependencies.
```bash
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym
git lfs install
git lfs pull
```

2. **Install Dependencies:**

To facilitate quick installation, we can create a new Conda environment: (If you do not have anaconda or miniconda installed, please go to the [official website](https://www.anaconda.com/) to install it)
```bash
conda create --name orca python=3.12
conda activate orca
```

Then install the dependencies in the newly created environment:
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
```

## Set Up OrcaStudio:

Follow the instructions provided in the [OrcaStudio documentation](URL:http://orca3d.cn/) to install and configure OrcaStudio on your system.


## Important Considerations
* Performance: High-fidelity rendering and complex physics simulations can be computationally intensive. Ensure your hardware meets the requirements for running OrcaStudio effectively.
* Configuration: Properly configure OrcaStudio to match your simulation needs. Refer to the OrcaStudio documentation for detailed configuration options.
* Compatibility: While OrcaGym aims for compatibility with OpenAI Gym, some advanced features may require additional configuration or modification of existing Gym environments.

## Contributing
We welcome contributions to the OrcaGym project. If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request on our GitHub repository.

## License
OrcaGym is licensed under the MIT License. See the LICENSE file for more information.

## Contact
For any inquiries or support, please contact us at huangwei@openverse.com.cn

---

We hope you find OrcaGym a valuable tool for your robotics and reinforcement learning research. Happy simulating!
