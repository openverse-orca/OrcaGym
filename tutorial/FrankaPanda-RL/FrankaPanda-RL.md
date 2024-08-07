## Validate Training of a Single Agent

1. Open the `FrankaPanda_RL` level and click run.
2. Open a console window, navigate to the current directory, and execute the following command:

```bash
conda activate orca_gym_test
python ./RunFrankaMocapMultiAgents.py --ip_addr localhost --agent_num 1 --task reach --model_type tqc --run_mode training --total_timesteps 10000
```
3. You will see the Franka robot arm start moving quickly, and the console will print out training information. In this task, the agent's goal is to control the robot arm's end effector to reach the transparent red cube marker. This is a relatively simple task that can achieve a high success rate within 10,000 steps.
4. When the training is complete, run the following command to validate the training results:

```bash
python ./RunFrankaMocapMultiAgents.py --ip_addr localhost --agent_num 1 --task reach --model_type tqc --run_mode testing --total_timesteps 10000
```

## Accelerating Training with Multi-Agent Parallelism

You might have noticed that when training a single agent for 10,000 steps, it takes several minutes. This depends on your machine configuration; on my PC (i7-13700, 64GB DRAM, Nvidia 3090 GPU), it takes about 8 minutes. Next, we will use multi-agent parallel training to achieve faster training speeds.

1. Move the camera in OrcaStudio using the AWSD keys and mouse. Hold down the 'S' key to pull the camera back, and you will see 16 Franka robotic arms (impressive, isn't it?).
2. Run the following command, specifying 16 agents for training simultaneously (Note: In the previous training, a model file named `panda_mocap_reach_tqc_10000_model.zip` was generated. You can choose to delete this file and start training from scratch or continue training based on this file. Refer to `envs/panda_mocap/panda_env.py` for details).
```bash
python ./RunFrankaMocapMultiAgents.py --ip_addr localhost --agent_num 16 --task reach --model_type tqc --run_mode training --total_timesteps 10000
```

The training ended in just a few seconds. Unfortunately, the agents did not seem to learn the reach task. Performing 10 tests all failed. You can move the camera closer to the first robotic arm (the furthest one) and run the previous testing command to observe the execution result. The reason for this issue is that with multiple agents training in parallel, the number of steps grows quickly, and there may not have been enough sample information collected for the agent to learn. Therefore, we will set `total_timesteps` to 30,000 to ensure sufficient samples.
```bash
python ./RunFrankaMocapMultiAgents.py --ip_addr localhost --agent_num 16 --task reach --model_type tqc --run_mode training --total_timesteps 30000
```
This time, the training ended in about 1 minute, and the agent has learned the reach task.

Next, you can try other more challenging tasks. You can also expand the training cluster to achieve higher training performance. For how to edit levels in OrcaStudio's editor mode, please refer to other documentation tutorials.