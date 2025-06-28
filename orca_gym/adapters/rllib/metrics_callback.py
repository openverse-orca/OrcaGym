
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import time


class OrcaMetricsCallback(DefaultCallbacks):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_time = None
        self._num_env_steps_sampled_lifetime_pre = 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        # 基础指标（环境数、采样速度等）
        num_workers = algorithm.config.num_env_runners
        num_envs_per_worker = algorithm.config.num_envs_per_env_runner
        total_envs = num_workers * num_envs_per_worker
        
        if self._last_time is None:
            self._last_time = time.time()
            return
        
        # 计算每秒步数
        num_env_steps_sampled_lifetime = result["num_env_steps_sampled_lifetime"]
        steps_per_second = int(
            (num_env_steps_sampled_lifetime - self._num_env_steps_sampled_lifetime_pre) / 
            (time.time() - self._last_time)
        )
        
        # 添加新指标到结果中
        result["custom_metrics"] = result.get("custom_metrics", {})
        result["custom_metrics"].update({
            "steps_per_second": steps_per_second,
            "total_envs": total_envs,
        })
        
        print(f"============================= ORCA METRICS =============================")
        print(f"Total environments: {total_envs}")
        print(f"Steps per second: {steps_per_second}")
        print("==========================================================================")
        
        self._last_time = time.time()
        self._num_env_steps_sampled_lifetime_pre = num_env_steps_sampled_lifetime