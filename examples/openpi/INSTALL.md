首先获取 openpi 相关的第三方代码：

```bash
git clone git@github.com:openverse-orca/OrcaGym_Openpi.git
cd OrcaGym_Openpi
git submodule update --init --recursive
conda activate orca
export PROJECT_HOME=`pwd`
```

然后以开发模式安装 openpi 相关的第三方代码：


```bash
cd $PROJECT_HOME/3rd_party/openpi
pip install -e .

cd $PROJECT_HOME/3rd_party/openpi/packages/openpi-client
pip install -e .

cd $PROJECT_HOME/3rd_party/lerobot
pip install -e .

cd $PROJECT_HOME/3rd_party/gym-aloha
pip install -e .
```

测试安装是否成功：

```bash
cd $PROJECT_HOME/3rd_party/openpi
python scripts/serve_policy.py --env ALOHA_SIM
```

此时应该会看到类似如下的输出：

```bash
INFO:websockets.server:server listening on 0.0.0.0:8000
```

然后在另一个终端中运行：

```bash
cd OrcaGym_Openpi
export PROJECT_HOME=`pwd`
conda activate orca
cd $PROJECT_HOME/3rd_party/openpi
MUJOCO_GL=egl python examples/aloha_sim/main.py
``` 

此时应该会看到类似如下的输出：

```bash
INFO:absl:MuJoCo library version is: 3.3.0
INFO:root:Waiting for server at ws://0.0.0.0:8000...
INFO:root:Starting episode...
INFO:root:Episode completed.
INFO:root:Saving video to data/aloha_sim/videos/out_0.mp4
``` 

最后，在 $PROJECT_HOME/3rd_party/openpi/data 目录下会生成一个名为 `aloha_sim` 的目录，里面存放了运行仿真生成的视频，例如 out_0.mp4。
