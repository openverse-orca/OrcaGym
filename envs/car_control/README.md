## 运行小车行走仿真

    首先在OrcaStuido中加载并运行`Car`关卡，然后在 `examples/car` 目录下，运行 `Car-Keyboard-Control.py`

    ```bash
    python Car-Keyboard-Control.py --grpc localhost
    ```

    **操作方式** 空格键：开始原地踏步、停止原地踏步； W/S 进入，退出前进模式； A/D 进入，退出转向模式； 注意，前进模式和转向模式可以同时进行。
    **关于弹窗** 键盘捕捉采用pygame包，该包不能在命令行模式下捕捉按键，所以当启动程序时，会弹出一个小窗口，并自动聚焦到这个小窗口。当屏幕焦点在这个小窗口上时才能响应按键输入。