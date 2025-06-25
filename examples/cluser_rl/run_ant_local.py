import ray
from ray import air
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
import numpy as np
import gymnasium as gym
import threading
import argparse
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import time
from ray.rllib.policy.policy import Policy

ENV_ENTRY_POINT = {
    "Ant_OrcaGymEnv": "envs.mujoco.ant_orcagym:AntOrcaGymEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)



# 自定义回调（添加环境数和采样速度）
class CustomMetricsCallback(DefaultCallbacks):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_time = None

    def on_train_result(self, *, algorithm, result, **kwargs):
        # 获取正确的worker配置
            # 首先尝试新的属性名称
        num_workers = algorithm.config.num_env_runners
        num_envs_per_worker = algorithm.config.num_envs_per_env_runner
        total_envs = num_workers * num_envs_per_worker

        if self._last_time is None:
            self._last_time = time.time()
            if "num_env_steps_sampled_lifetime" in result:
                self._num_env_steps_sampled_lifetime_pre = result["num_env_steps_sampled_lifetime"]
            return

        print("============================= ORCA =============================")
        print("Total environments:", total_envs)
        if "num_env_steps_sampled_lifetime" in result:
            num_env_steps_sampled_lifetime = result["num_env_steps_sampled_lifetime"]
            steps_per_second = int((num_env_steps_sampled_lifetime - self._num_env_steps_sampled_lifetime_pre) / (time.time() - self._last_time))
            self._num_env_steps_sampled_lifetime_pre = num_env_steps_sampled_lifetime
            print("Steps per second:", steps_per_second)
        print("================================================================")

        
        self._last_time = time.time()

def get_orca_gym_register_info(
        orcagym_addr : str,
        env_name : str, 
        agent_name : str, 
        render_mode: str,
    ) -> tuple[ str, dict ]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str
    agent_names = [f"{agent_name}"]
    kwargs = {
        'frame_skip': FRAME_SKIP,
        'orcagym_addr': orcagym_addr,
        'agent_names': agent_names,
        'time_step': TIME_STEP,
        'render_mode': render_mode,
    }

    return env_id, kwargs


def env_creator(
        env_context,
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int,
        render_mode: str
    ):

    if env_context is None:
        worker_idx = 1
        vector_idx = 0
        # print("Creating environment in main thread, no env_context provided.")
    else:
        worker_idx = env_context.worker_index
        vector_idx = env_context.vector_index
        # print(f"Creating environment in worker {worker_idx}, vector {vector_idx}.")

    # 只有第一个worker的第一个环境渲染
    if render_mode == 'human':
        render_mode = 'human' if worker_idx == 1 and vector_idx == 0 else 'none'
    else:
        render_mode = 'none'
    
    env_id, kwargs = get_orca_gym_register_info(
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        render_mode=render_mode,
    )

    if vector_idx == 0:
        gym.register(
            id=env_id,
            entry_point=ENV_ENTRY_POINT[env_name],
            kwargs=kwargs,
            max_episode_steps=max_episode_steps,
            reward_threshold=0.0,
        )

    env = gym.make(env_id, **kwargs)

    return env

def test_model(
        checkpoint_path,
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int
    ):
    
    # 1. 首先注册训练时使用的环境
    register_env(
        "OrcaGymEnv", 
        lambda env_context: env_creator(
            env_context=env_context,
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
            render_mode='human',
        )
    )
    
    # 2. 使用注册的环境创建实例
    from ray.rllib.env import EnvContext
    env = env_creator(
        env_context=None,  # 空上下文
        orcagym_addr=orcagym_addr,
        env_name=env_name,
        agent_name=agent_name,
        max_episode_steps=max_episode_steps,
        render_mode='human',
    )
    
    # 3. 加载策略
    from ray.rllib.policy.policy import Policy
    policies = Policy.from_checkpoint(checkpoint_path)
    default_policy_id = "default_policy"
    if default_policy_id not in policies:
        default_policy_id = next(iter(policies.keys()))
    policy = policies[default_policy_id]
    
    # 4. 验证观察空间一致性
    train_obs_space = policy.observation_space
    test_obs_space = env.observation_space
    print(f"训练模型观察空间: {train_obs_space}")
    print(f"测试环境观察空间: {test_obs_space}")
    
    # 如果空间不一致，尝试转换测试环境的观测
    if test_obs_space != train_obs_space:
        print("警告：训练和测试环境观察空间不匹配！尝试转换...")
        # 创建一个转换器（这需要根据具体情况实现）
        from ray.rllib.models import ModelCatalog
        from ray.rllib.models.preprocessors import get_preprocessor
        
        prep = get_preprocessor(train_obs_space)(train_obs_space)
        def transform_obs(obs):
            # 添加批次维度
            obs = np.expand_dims(obs, 0)
            # 应用与训练相同的预处理
            return prep.transform(obs)[0]  # 移除批次维度
        
        transform_required = True
    else:
        transform_required = False
    
    for _ in range(5):  # 运行5个测试episode
        state_dict, _ = env.reset()
        
        # 5. 获取并转换观测（如果需要）
        if transform_required:
            # 如果测试环境返回字典，获取值
            if isinstance(state_dict, dict):
                state = next(iter(state_dict.values()))
            else:
                state = state_dict
                
            state = transform_obs(state)
        else:
            state = state_dict
        
        episode_reward = 0
        done = False
        
        while not done:
            time_start = time.time()
            
            # 6. 确保格式正确（添加批次维度）
            batch_state = np.expand_dims(state, axis=0)
            
            # 7. 计算动作
            action = policy.compute_single_action(batch_state, explore=False)
            
            # 8. 处理动作（根据策略输出格式）
            if isinstance(action, tuple):
                # 如果是元组 (action, state, ...)
                action = action[0]
            
            # 环境步进
            next_state_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # 9. 转换下一个状态
            if transform_required:
                if isinstance(next_state_dict, dict):
                    state = next(iter(next_state_dict.values()))
                else:
                    state = next_state_dict
                state = transform_obs(state)
            else:
                state = next_state_dict
                
            # 10. 渲染环境（保持实时速度）
            env.render()
            time_end = time.time()
            elapsed_time = time_end - time_start
            if elapsed_time < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time)
        
        print(f"Test episode reward: {episode_reward}")
    
    env.close()

def config_appo_tunner(
        cpu_num: int, 
        iter: int
    ) -> tune.Tuner:
    # 完全修正的APPO配置
    config = (
        APPOConfig()
        .environment(
            env="OrcaGymEnv",
            env_config={},
            disable_env_checking=True 
        )
        .env_runners(
            num_env_runners=cpu_num,          
            num_envs_per_env_runner=1,   
            num_cpus_per_env_runner=1,  
            rollout_fragment_length=64,
            gym_env_vectorize_mode="ASYNC"
        )
        .rl_module(
            model_config=DefaultModelConfig(
                fcnet_hiddens=[256, 256],
                vf_share_layers=True,
            ),
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .training(
            train_batch_size_per_learner=4096,
        )
        .resources(
            num_cpus_for_main_process=1,
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .callbacks(
            CustomMetricsCallback
        )
    )
    
    print(f"总环境数: {config.num_env_runners * config.num_envs_per_env_runner}")
    print(f"总CPU需求: {config.num_cpus_for_main_process + config.num_env_runners * config.num_cpus_per_env_runner}")
    

    return tune.Tuner(
        "APPO",
        param_space=config.to_dict(),
        run_config=RunConfig(
            name="APPO_OrcaGym_Training",
            stop={"training_iteration": iter},
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_at_end=True
            ),
            verbose=3,  # 更详细输出
        ),
    )

def run_training(
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int,
        cpu_num: int,
        iter: int,
        render_mode: str
    ):
    register_env(
        "OrcaGymEnv", 
        lambda env_context: env_creator(
            env_context=env_context,
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )
    )

    # 优化Ray初始化参数
    ray.init(
        num_cpus=cpu_num + 1, 
        num_gpus=1,
        object_store_memory=cpu_num * 1024**3, 
        include_dashboard=False, 
        ignore_reinit_error=True
    )

    print("\nStarting training...")
    tuner = config_appo_tunner(
        cpu_num=cpu_num,
        iter=iter
    )
    results = tuner.fit()
    
    if results:
        print("\nTraining completed. Best results:")
        best_result = results.get_best_result()
        if best_result.checkpoint:
            print(f"Best checkpoint: {best_result.checkpoint}")

    ray.shutdown()
    print("Process completed.")

def main(
        orcagym_addr: str,
        env_name: str,
        agent_name: str,
        max_episode_steps: int,
        cpu_num: int,
        run_mode: str,
        iter: int,
        checkpoint_path: str,
        render_mode: str
    ):

    if run_mode == 'training':
        run_training(
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps,
            cpu_num=cpu_num,
            iter=iter,
            render_mode=render_mode
        )
    elif run_mode == 'testing':
        if not checkpoint_path:
            raise ValueError("Checkpoint path must be provided for testing.")
        test_model(
            checkpoint_path=checkpoint_path,
            orcagym_addr=orcagym_addr,
            env_name=env_name,
            agent_name=agent_name,
            max_episode_steps=max_episode_steps
        )
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ant OrcaGym environment with APPO training.')
    parser.add_argument('--cpu_num', type=int, help='The number of CPUs to use, default is 16, must less than YourSystemCPUNumber - 1')
    parser.add_argument('--run_mode', type=str, default='training', choices=['training', 'testing'], help='The mode to run (training / testing)')
    parser.add_argument('--iter', type=int, default=100, help='The number of iterations to run')
    parser.add_argument('--checkpoint_path', type=str, help='The path to the checkpoint file for testing, if not provided, will run training')
    parser.add_argument('--render_mode', type=str, default='human', choices=['human', 'none'], help='The render mode (human / none)')
    args = parser.parse_args()

    main(
        orcagym_addr="localhost:50051",
        env_name="Ant_OrcaGymEnv",
        agent_name="ant_usda",
        max_episode_steps=1000,
        cpu_num=args.cpu_num if args.cpu_num else 16,
        run_mode=args.run_mode,
        iter=args.iter,
        checkpoint_path=args.checkpoint_path if args.run_mode == 'testing' else None,
        render_mode=args.render_mode
    )