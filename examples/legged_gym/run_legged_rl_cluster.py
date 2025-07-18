import ray
from ray import air
from ray import tune
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
import gymnasium as gym
import threading
import os


ENV_ENTRY_POINT = {
    "Ant_OrcaGymEnv": "envs.mujoco.ant_orcagym:AntOrcaGymEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

def get_orca_gym_register_info(
        orcagym_addr : str,
        env_name : str, 
        agent_name : str, 
        render_mode: str,
        worker_idx: int,
        vector_idx: int,
    ) -> tuple[ str, dict ]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{worker_idx:03d}-{vector_idx:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {
        'frame_skip': FRAME_SKIP,
        'orcagym_addr': orcagym_addr,
        'agent_names': agent_names,
        'time_step': TIME_STEP,
        'render_mode': render_mode,
        'env_id': env_id,
        'worker_idx': worker_idx,
        'vector_idx': vector_idx,
    }

    return env_id, kwargs


def env_creator(
        env_context,
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int,
    ):

    if env_context is None:
        worker_idx = 1
        vector_idx = 0
        print("Creating environment in main thread, no env_context provided.")
    else:
        worker_idx = env_context.worker_index
        vector_idx = env_context.vector_index
        print(f"Creating environment in worker {worker_idx}, vector {vector_idx}.")

    # 只有第一个worker的第一个环境渲染
    render_mode = 'human' if worker_idx == 1 and vector_idx == 0 else 'none'
    
    env_id, kwargs = get_orca_gym_register_info(
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        render_mode=render_mode,
        worker_idx=worker_idx,
        vector_idx=vector_idx,
    )

    # if vector_idx == 0:
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        reward_threshold=0.0,
    )
    print(f"Registered environment: {env_id} with kwargs: {kwargs}, in thread {threading.get_ident()}")

    env = gym.make(env_id, **kwargs)
    print(f"Creating environment: {env_name} with kwargs: {kwargs}, in thread {threading.get_ident()}")

    return env

def test_model(
        checkpoint_path,
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int
    ):

    
    algo = Algorithm.from_checkpoint(checkpoint_path)
    _, kwargs = get_orca_gym_register_info(
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        render_mode='human',  # 测试时渲染
        worker_idx=1,  # 测试时只使用一个worker
        vector_idx=0,  # 测试时只使用一个vector
    )
    env = env_creator(
        env_context=None,  # 测试时不需要env_context
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        max_episode_steps=max_episode_steps,
    )

    for _ in range(5):  # 运行5个测试episode
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = algo.compute_single_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            env.render()  # 渲染环境
        
        print(f"Test episode reward: {episode_reward}")
    
    env.close()

def config_appo_tunner() -> tune.Tuner:
    # 完全修正的APPO配置
    config = (
        APPOConfig()
        .environment(
            env="OrcaGymEnv",
            env_config={},
            disable_env_checking=True 
        )
        .env_runners(
            num_env_runners=30,          
            num_envs_per_env_runner=64,   
            num_cpus_per_env_runner=1,  
            rollout_fragment_length=64,
            create_env_on_local_worker=True,
            gym_env_vectorize_mode="ASYNC"
        )
        .training(
            train_batch_size=4096,
            gamma=0.99,
            lr=0.0003,
            model={
                "fcnet_hiddens": [256, 256],  
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [128],
                "post_fcnet_activation": "relu",
                "free_log_std": True,
            }
        )
        .resources(
            num_cpus_for_main_process=1,
            num_gpus=1, 
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
    )
    
    print(f"总环境数: {config.num_env_runners * config.num_envs_per_env_runner}")
    print(f"总CPU需求: {config.num_cpus_for_main_process + config.num_env_runners * config.num_cpus_per_env_runner}")
    
    return tune.Tuner(
        "APPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            name="APPO_OrcaGym_Training",
            stop={"training_iteration": 100},
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_at_end=True
            ),
            verbose=3,  # 更详细输出
        ),
    )



def main(
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int
    ):
    import numpy as np

    register_env(
        "OrcaGymEnv", 
        lambda env_context: env_creator(
            env_context=env_context,
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
        )
    )

    # 优化Ray初始化参数
    ray.init(
        num_cpus=32,  # 减少到实际可用资源
        num_gpus=1,   # 启用GPU
        object_store_memory=20 * 1024**3,  # 20GB
        include_dashboard=False,  # 启用仪表盘
        ignore_reinit_error=True
    )

    print("\nStarting training...")
    tuner = config_appo_tunner()
    results = tuner.fit()
    
    # if results:
    #     print("\nTraining completed. Best results:")
    #     best_result = results.get_best_result()
    #     if best_result.checkpoint:
    #         print(f"Best checkpoint: {best_result.checkpoint}")
    #         print("Starting testing...")
    #         test_model(
    #             best_result.checkpoint,
    #             orcagym_addr=orcagym_addr,
    #             env_name=env_name,
    #             agent_name=agent_name,
    #             max_episode_steps=max_episode_steps
    #         )

    ray.shutdown()
    print("Process completed.")

if __name__ == "__main__":
    main(
        orcagym_addr="localhost:50051",
        env_name="Ant_OrcaGymEnv",
        agent_name="ant_usda",
        max_episode_steps=1000
    )