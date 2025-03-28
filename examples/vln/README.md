## 1、首先打开orca engine，在机器狗上上安装ros2的camera组件

参数：

entity name：camera，放在go2_000/mujoco_wrap/rootPrim/mocap/mocap下

tf(Translate):0.278,0,0

tf(rotation)：-90,90,0

camera参数：

Far clip distance: 15.0

Frequency: 30

四个QoS均为：可靠

#### 可在rviz2中查看 rgb 和 depth 图像的topic是否分别为：/camera/camera_image_color 和 /camera/camera_image_depth

## 2、然后打开orcagym的虚拟环境，进行机械狗的仿真操控，示例命令：

```bash
python run_legged_rl.py --run_mode nav --model_file pre_trained_models/go2_ppo.zip --nav_ip 192.168.110.135
```

## 3、编译这个目录下的ros2 package，并且运行：

修改 `～/OrcaGym/examples/vln/ros2_ws/src/mujoco_image_viewer/mujoco_image_viewer/image_viewer.py` 中的 `ImageSender`为本机ip，然后运行此ros2 package（此package可能需要在 ros2 安装的环境下运行，比如`/bin/python3`或者`/usr/bin/python3`，确保安装了Flask和requests库）：

```bash
cd ～/OrcaGym/examples/vln/ros2_ws
colcon build && source install/setup.bash && ros2 run mujoco_image_viewer image_viewer
```

## 4、按照`～/vln_policy/README.md`的提示创建vln_policy虚拟环境，并且进入此虚拟环境

## 5、启动policy server

进入`～/vln_policy`目录下

运行脚本，启动服务器，等待所有模型加载完毕后进入下一步

```
./scripts/launch_vlm_servers.sh
```

## 6、启动导航

修改goal和ip：

修改`～/vln_policy/config/experiments/reality.yaml`文件中的`goal`和`nav_ip`即可；nav_ip为本机ip；因为采用yolov7进行目标检测，所以goal应该在COCO数据集的类别中。

然后切换到`～/OrcaGym/examples/vln`目录下运行

```
conda activate vln_policy
python navigation.py
```

若卡在 `BLIP2ITMClient.cosine`这里，检查`～/vln_policy/lockfiles`下有无文件，如果有，删除后再运行。
