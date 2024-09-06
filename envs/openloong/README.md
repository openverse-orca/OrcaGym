# 青龙机器人环境

* OpenLoong开源项目是由人形机器人（上海）有限公司、上海人形机器人制造业创新中心与开放原子开源基金会（OpenAtom Foundation）共同运营的开源项目。
* 本环境适配青龙机器人行运动控制功能，基于上海人形机器人创新中心“青龙”机器人模型，提供行走、跳跃、盲踩障碍物三种运动示例。参看[OpenLoong Dynamics Control](https://atomgit.com/openloong/openloong-dyn-control)项目获取更多信息。

## 如何安装

1. **获取青龙机器人仓库源代码**
    
    ```bash
    git submodule update --init --recursive
    ```

2. **编译python绑定。**

    青龙机器人运动控制基于MPC和WBC实现，对算力实时性有较高要求，需要仿真频率在 500Hz以上，推荐1000Hz。因此算法实现采用C++方案实现。OrcaGym框架基于Python实现，因此需要将MPC和WBC算法库进行Python绑定封装。
    ``` bash
    conda activate orca_gym_test
    pip install pybind11
    mkdir build
    cd build
    cmake ..
    make -j20
    ```
