import torch as th
from typing import Tuple
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import obs_as_tensor

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)


class CustomEnv(gym.Env):
  def __init__(self, model: PPO):
    super().__init__()
    self.observation_space = model.observation_space
    self.action_space = model.action_space

  def reset(self, seed=None):
    return self.observation_space.sample(), {}

  def step(self, action):
    return self.observation_space.sample(), 0.0, False, False, {}


def convert_sb3_to_onnx(model_path: str, output_path: str):
    # Example: model = PPO("MlpPolicy", "Pendulum-v1")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = PPO.load(model_path, device=device)

    env = CustomEnv(model)
    obs, info = env.reset()
    _logger.info(f"obs:  {obs}, info: {info}")

    # 修复：添加batch维度
    batch_obs = {}
    for key, value in obs.items():
        # 添加batch维度 (1, ...)
        if isinstance(value, np.ndarray):
            batch_obs[key] = np.expand_dims(value, axis=0)
        else:
            batch_obs[key] = np.array([value])
    
    _logger.info(f"Fixed obs:  {batch_obs}, info: {info}")


    # Convert to ONNX
    onnx_policy = OnnxableSB3Policy(model.policy)
    obs_tensor = obs_as_tensor(batch_obs, model.policy.device)

    model_input = {
        "observation": obs_tensor
    }

    onnx_program = th.onnx.export(
        onnx_policy,
        args=(model_input,),
        f=output_path,
        dynamo=True
    )

def check_onnx_model(onnx_path: str, model_path: str):
    ##### Load and test with onnx
    import onnx
    import onnxruntime as ort
    import numpy as np

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = PPO.load(model_path, device=device)
    env = CustomEnv(model)
    obs, info = env.reset()

    # 修复：添加batch维度
    batch_obs = {}
    for key, value in obs.items():
        # 添加batch维度 (1, ...)
        if isinstance(value, np.ndarray):
            obs_value = np.expand_dims(value, axis=0)
        else:
            obs_value = np.array([value])

        # 随机采样到 [-10, 10]区间
        obs_value = np.array(np.random.uniform(-10, 10, obs_value.shape), dtype=np.float32)
        batch_obs[key] = obs_value
    
    _logger.info(f"Fixed obs:  {batch_obs}, info: {info}")

    # Load ONNX model and run
    ort_session = ort.InferenceSession(onnx_path)
    onnxruntime_input = {k.name: v for k, v in zip(ort_session.get_inputs(), batch_obs.values())}
    onnxruntime_outputs, _, _ = ort_session.run(None, onnxruntime_input)
    # 剪切到[-1, 1]
    onnxruntime_outputs = np.clip(onnxruntime_outputs, -1, 1)


    # Run PyTorch model and compare
    torch_outputs, _ = model.predict(batch_obs, deterministic=True)


    _logger.performance(f"onnxruntime_outputs:  {onnxruntime_outputs}")
    _logger.info(f"torch_outputs:  {torch_outputs}")

    th.testing.assert_close(torch_outputs[0], onnxruntime_outputs[0], rtol=1e-06, atol=1e-6)


    # Debug
    _logger.info("Observations:")
    for key, value in obs.items():
        _logger.info(f"{key}: {value.shape}")

    _logger.info("\nONNX input:")
    for key, value in onnxruntime_input.items():
        _logger.info(f"{key}: {value.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    convert_sb3_to_onnx(args.model_path, args.output_path)
    check_onnx_model(args.output_path, args.model_path)