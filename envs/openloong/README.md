# 青龙机器人环境

* OpenLoong开源项目是由人形机器人（上海）有限公司、上海人形机器人制造业创新中心与开放原子开源基金会（OpenAtom Foundation）共同运营的开源项目。
* 本环境适配青龙机器人行运动控制功能，基于上海人形机器人创新中心“青龙”机器人模型，提供行走、跳跃、盲踩障碍物三种运动示例。参看[OpenLoong Dynamics Control](https://atomgit.com/openloong/openloong-dyn-control)项目获取更多信息。

## 如何安装
如果您是 **安装版** 用户，请跳过以下步骤。直接运行仿真即可
1. **获取青龙机器人仓库源代码**
    
    Clone 青龙机器人源码，并安装编译依赖库。
    
    **注意** 青龙机器人源码托管在[OPENATOM社区](https://atomgit.com/OpenAtomFoundation/)，您需要首先注册该平台账户（免费），并**添加 SSH 公钥**，才能顺利clone代码。[点击查看如何生成 SSH KEY](https://docs.atomgit.com/user/sshkey/)


    ```bash
    cd openloong/external
    git clone git@atomgit.com:openloong/openloong-dyn-control.git
    sudo apt-get update
    sudo apt install git cmake gcc-11 g++-11
    sudo apt install libglu1-mesa-dev freeglut3-dev    
    ```

2. **编译python绑定。**

    青龙机器人运动控制基于MPC和WBC实现，对算力实时性有较高要求，需要仿真频率在 500Hz以上，推荐1000Hz。因此算法实现采用C++方案实现。OrcaGym框架基于Python实现，因此需要将MPC和WBC算法库进行Python绑定封装。

    ``` bash
    conda activate orca_gym_test
    pip install pybind11
    cd ..  #注意此路径为envs/openloong
    mkdir build
    cd build
    cmake .. -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir)
    make -j20
    ```
3. **测试安装是否正常**

    运行测试脚本，输出 `Run test successfully!` 说明算法库加载正常。

    ```bash
    cd ..
    python test_libs.py
    ```

## 运行青龙机器人行走仿真

    首先在OrcaStuido中加载并运行`multy_robot`关卡，然后在 `examples/openloong` 目录下，运行 `openloong_wbc_joystick_multy_agents.py`

    ```bash
    python openloong_wbc_joystick_multy_agents.py
    ```

    **操作方式** 空格键：开始原地踏步、停止原地踏步； W/S 进入，退出前进模式； A/D 进入，退出转向模式； 注意，前进模式和转向模式可以同时进行。
    **关于弹窗** 键盘捕捉采用pygame包，该包不能在命令行模式下捕捉按键，所以当启动程序时，会弹出一个小窗口，并自动聚焦到这个小窗口。当屏幕焦点在这个小窗口上时才能响应按键输入。


