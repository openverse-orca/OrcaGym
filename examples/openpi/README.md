## 安装详见 INSTALL.md

## AzureLoong 环境

### 数据采集

```bash
python run_openloong_sim.py
```

### 数据格式转化

```bash
python convert_orca_data_to_lerobot.py --data_dir /YOUR/HDF5/FILE/PATH
```

转化后的数据在`~/.cache/huggingface/lerobot/orca_gym/AzureLoong/data`目录下

### 数据预处理

设置训练配置：修改`OrcaGym_Openpi/3rd_party/openpi/src/openpi/training/config.py`下`TrainConfig`类的配置，此处配置名为`pi0_orca_azureloong_lora`

切换至`OrcaGym_Openpi/3rd_party/openpi`目录下，激活orca虚拟环境，然后按照INSTALL.md安装所有的packages，注意`pydantic==2.7.0`，否则会报错

数据正则化：`python scripts/compute_norm_stats.py --config-name pi0_orca_azureloong_lora`

微调模型，进行训练：`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_orca_azureloong_lora --exp-name=my_experiment --overwrite`

### 部署

server:进入`OrcaGym_Openpi/3rd_party/openpi` 目录，运行`python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_orca_azureloong_lora --policy.dir=checkpoints/pi0_orca_azureloong_lora/my_experiment/<train epochs>`命令，启动服务器

client:运行`python run_openloong_openpi_client.py --agent_names <robot name>`
   
### 评估

