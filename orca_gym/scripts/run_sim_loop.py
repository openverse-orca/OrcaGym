from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo, CameraSensorInfo, MaterialInfo
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import argparse
import numpy as np
import orca_gym.utils.rotations as rotations
import time
import random
import gymnasium as gym
import sys
from datetime import datetime
import os
from typing import Optional

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


ENV_ENTRY_POINT = {
    "SimulationLoop": "orca_gym.scripts.sim_env:SimEnv",
    "DoubleGripperAuto": "doubleGripper_towel.envs.double_gripper_orcagym_env:DoubleGripperOrcaGymEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str, 
                 max_episode_steps : int,
                 frame_skip: int = FRAME_SKIP,
                 time_step: float = TIME_STEP,
                 controller_min_steps: int = 6) -> tuple[ str, dict ]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': frame_skip,
              'orcagym_addr': orcagym_addr,
              'agent_names': agent_names,
              'time_step': time_step}
    if env_name == "DoubleGripperAuto":
        kwargs["controller_min_steps"] = controller_min_steps

    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs



def sceneinfo(
    scene,
    stage: str,
    orcagym_address,
):
    toclose = False
    if scene is None:
        toclose = True
        import importlib
        OrcaGymScene = importlib.import_module("orca_gym.scene.orca_gym_scene").OrcaGymScene
        scene = OrcaGymScene(orcagym_address)
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        scene.get_rundata(script_name,stage)
        if stage == "beginscene":
            mess = f"开始仿真,可操作鼠标键盘移动镜头"
            scene.set_ui_text(actor_name=1, message=mess, showtime=5, color="0xff0000", size=32)

        elif stage == "endscene":
            mess = f"运行结束"
            scene.set_ui_text(actor_name=1, message=mess, showtime=30, color="0xff0000", size=32)
        
    finally:
        if toclose:
            scene.close()

def run_simulation(orcagym_addr : str, 
                agent_name : str,
                env_name : str,
                frame_skip: int = FRAME_SKIP,
                time_step: float = TIME_STEP,
                controller_min_steps: int = 6,
                max_steps: Optional[int] = None,
                scene_runtime: Optional[OrcaGymSceneRuntime] = None) -> None:
    env = None  # Initialize env to None
    try:
        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addr}")
        realtime_step = time_step * frame_skip

        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      agent_name, 
                                      sys.maxsize,
                                      frame_skip=frame_skip,
                                      time_step=time_step,
                                      controller_min_steps=controller_min_steps)
        _logger.info(f"Registered environment:  {env_id}")

        env = gym.make(env_id)        
        _logger.info("Starting simulation...")

        if scene_runtime is not None:
            if hasattr(env, "set_scene_runtime"):
                _logger.performance("Setting scene runtime...")
                env.set_scene_runtime(scene_runtime)
            elif hasattr(env.unwrapped, "set_scene_runtime"):
                _logger.performance("Setting scene runtime...")
                env.unwrapped.set_scene_runtime(scene_runtime)

        obs = env.reset()
        sceneinfo(
		scene=None,
		stage="beginscene",
		orcagym_address=orcagym_addr,
    	)
        steps = 0
        while True:
            if max_steps is not None and steps >= max_steps:
                _logger.info(f"Reached max_steps={max_steps}, exit simulation loop.")
                break

            start_time = datetime.now()
            if env_name == "DoubleGripperAuto":
                action = None
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            env.render()

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < realtime_step:
                time.sleep(realtime_step - elapsed_time.total_seconds())

    except KeyboardInterrupt:
        print("Simulation stopped")
    finally:
        if env is not None:
            env.close()


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="Run OrcaGym local simulation loop.")
    parser.add_argument("--orcagym-addr", default="localhost:50051", help="OrcaGym gRPC address.")
    parser.add_argument("--agent-name", default="NoRobot", help="Agent name used by OrcaGym.")
    parser.add_argument(
        "--control-mode",
        choices=["simloop", "auto_step"],
        default="simloop",
        help="simloop: legacy zero-control loop; auto_step: double gripper auto-step controller.",
    )
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP, help="Frame skip.")
    parser.add_argument("--time-step", type=float, default=TIME_STEP, help="Base simulation timestep.")
    parser.add_argument(
        "--controller-min-steps",
        type=int,
        default=6,
        help="Minimum auto_step keyframes (only for --control-mode auto_step).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after N steps; 0 means infinite.",
    )
    args = parser.parse_args()

    env_name = "SimulationLoop" if args.control_mode == "simloop" else "DoubleGripperAuto"
    run_simulation(
        args.orcagym_addr,
        args.agent_name,
        env_name,
        frame_skip=args.frame_skip,
        time_step=args.time_step,
        controller_min_steps=args.controller_min_steps,
        max_steps=(None if args.max_steps <= 0 else args.max_steps),
    )


if __name__ == "__main__":
    main()
