
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import time

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class OrcaMetricsCallback(DefaultCallbacks):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_sampled_update_time = None
        self._num_env_steps_sampled_lifetime_pre = 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        # 基础指标（环境数、采样速度等）
        num_workers = algorithm.config.num_env_runners
        num_envs_per_worker = algorithm.config.num_envs_per_env_runner
        total_envs = num_workers * num_envs_per_worker
        
        if self._last_sampled_update_time is None:
            self._last_sampled_update_time = time.time()
            return
        
        # 计算每秒步数
        current_time = time.time()
        num_env_steps_sampled_lifetime = result["num_env_steps_sampled_lifetime"]
        sampled_steps_per_second = int(
            (num_env_steps_sampled_lifetime - self._num_env_steps_sampled_lifetime_pre) / 
            (current_time - self._last_sampled_update_time)
        )

        # 添加新指标到结果中
        result["custom_metrics"] = result.get("custom_metrics", {})
        result["custom_metrics"].update({
            "sampled_steps_per_second": sampled_steps_per_second,
            "total_envs": total_envs,
        })

        # print(f"result: {result}")

        episode_return_mean = result.get('env_runners', {}).get('episode_return_mean', 0)
        episode_len_mean = result.get('env_runners', {}).get('episode_len_mean', 0)
        total_loss = result.get('learners', {}).get('default_policy', {}).get('total_loss', 0)
        policy_loss = result.get('learners', {}).get('default_policy', {}).get('policy_loss', 0)
        vf_loss = result.get('learners', {}).get('default_policy', {}).get('vf_loss', 0)
        entropy = result.get('learners', {}).get('default_policy', {}).get('entropy', 0)
        mean_kl_loss = result.get('learners', {}).get('default_policy', {}).get('mean_kl_loss', 0)
        grad_norm = result.get('learners', {}).get('default_policy', {}).get('gradients_default_optimizer_global_norm', 0)
        lr = result.get('learners', {}).get('default_policy', {}).get('default_optimizer_learning_rate', 0)
        diff_num_grad = result.get('learners', {}).get('default_policy', {}).get('diff_num_grad_updates_vs_sampler_policy', 0)

        return_mean = episode_return_mean / episode_len_mean if episode_len_mean > 0 else 0
        
        _logger.info(f"=========== ORCA METRICS ===========")
        _logger.info(f"Total environments: {total_envs}")
        _logger.info(f"Sampled steps per second: {sampled_steps_per_second}")
        _logger.info(f"return_mean: {return_mean}")
        _logger.info(f"episode_return_mean: {episode_return_mean}")
        _logger.info(f"episode_len_mean: {episode_len_mean}")
        _logger.info(f"total_loss: {total_loss}")
        _logger.info(f"policy_loss: {policy_loss}")
        _logger.info(f"vf_loss: {vf_loss}")
        _logger.info(f"entropy: {entropy}")
        _logger.info(f"mean_kl_loss: {mean_kl_loss}")
        _logger.info(f"grad_norm: {grad_norm}")
        _logger.info(f"lr: {lr}")
        _logger.info(f"diff_num_grad: {diff_num_grad}")
        _logger.info("======================================")
        
        self._last_sampled_update_time = time.time()
        self._num_env_steps_sampled_lifetime_pre = num_env_steps_sampled_lifetime