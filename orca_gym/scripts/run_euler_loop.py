"""空白 Euler 仿真循环（在线模式，连接 OrcaStudio）。

与 run_sim_loop.py 对应，但使用 OrcaGymEulerEnv 架构（EulerSimEnv）。
用于 orcagym-loop --euler 启动。

用法：
    orcagym-loop --euler [--addr localhost:50051] [--agent NoAgent]

特性：
    - 在线模式：连接 OrcaStudio，模型 XML 由 Studio 端 scene 提供
    - 空白环境：零控输入，仅做基础步进 + 渲染
    - 实时节流：按 realtime_step 对齐墙钟时间
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Optional

import gymnasium as gym

from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


ENV_ENTRY_POINT = {
    "EulerSimulationLoop": "orca_gym.scripts.sim_euler_env:EulerSimEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP


def register_env(
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_name: str,
    max_episode_steps: int,
) -> tuple[str, dict]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {
        "frame_skip": FRAME_SKIP,
        "orcagym_addr": orcagym_addr,
        "agent_names": agent_names,
        "time_step": TIME_STEP,
    }
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def run_simulation(
    orcagym_addr: str,
    agent_name: str,
    env_name: str = "EulerSimulationLoop",
) -> None:
    env = None
    try:
        _logger.info(f"Euler simulation running..., orcagym_addr: {orcagym_addr}")

        env_index = 0
        env_id, kwargs = register_env(
            orcagym_addr, env_name, env_index, agent_name, sys.maxsize
        )
        _logger.info(f"Registered environment: {env_id}")

        env = gym.make(env_id)
        u = env.unwrapped
        _logger.info(
            f"[Euler] time_step={kwargs['time_step']}, frame_skip={u.frame_skip}, "
            f"dt={u.dt}, realtime_step={REALTIME_STEP}"
        )
        _logger.info("Starting Euler simulation...")

        obs, info = env.reset()
        _logger.info("Euler simulation started. Move camera with mouse/keyboard.")

        while True:
            start_time = datetime.now()

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            elapsed = datetime.now() - start_time
            if elapsed.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed.total_seconds())

    except KeyboardInterrupt:
        print("Euler simulation stopped")
    finally:
        if env is not None:
            env.close()


def main(argv: Optional[list[str]] = None) -> None:
    """命令行入口（被 run_sim_loop.main 通过 --euler 调用）。"""
    import argparse

    parser = argparse.ArgumentParser(description="OrcaGym Euler 空白仿真循环")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--agent", default="NoAgent", help="智能体名称")
    args = parser.parse_args(argv)

    run_simulation(orcagym_addr=args.addr, agent_name=args.agent)


if __name__ == "__main__":
    main()
