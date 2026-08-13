import sys

from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo, CameraSensorInfo, MaterialInfo
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
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
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str, 
                 max_episode_steps : int) -> tuple[ str, dict ]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP}           
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
                scene_runtime: Optional[OrcaGymSceneRuntime] = None) -> None:
    env = None  # Initialize env to None
    try:
        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addr}")

        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      agent_name, 
                                      sys.maxsize)
        _logger.info(f"Registered environment:  {env_id}")

        env = gym.make(env_id)
        u = env.unwrapped
        mj_ts = u.gym._mjModel.opt.timestep if getattr(u.gym, "_mjModel", None) is not None else float("nan")
        print(
            f"[MuJoCo] script TIME_STEP={TIME_STEP}, kwarg={kwargs['time_step']}, "
            f"opt.timestep={u.gym.opt.timestep}, mjModel.timestep={mj_ts}, "
            f"frame_skip={u.frame_skip}, env.dt={u.dt}, REALTIME_STEP={REALTIME_STEP}",
            flush=True,
        )
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

        env.unwrapped.begin_save_video("C:/workspace/dev/orca/Record")
        _logger.info("开始录制视频")

        _t = datetime.now()
        recording = True

        while True:
            start_time = datetime.now()

            action = env.action_space.sample()
    
            obs, reward, terminated, truncated, info = env.step(action)

            env.render()

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())

            dt = datetime.now() - _t
            if recording and dt.total_seconds() >= 5.0:
                env.unwrapped.stop_save_video()
                recording = False


    except KeyboardInterrupt:
        print("Simulation stopped")        
        if env is not None:
            env.close()


def main():
    """命令行入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="OrcaGym 仿真循环")
    parser.add_argument(
        "--euler",
        action="store_true",
        help="使用 Euler 架构（OrcaGymEulerEnv）启动空白仿真循环",
    )
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--agent", default="NoRobot", help="智能体名称")
    args, _ = parser.parse_known_args()

    if args.euler:
        # 转发到 Euler loop（agent 名称默认对齐 Euler 约定）
        from orca_gym.scripts.run_euler_loop import main as euler_main

        euler_argv = ["--addr", args.addr, "--agent", args.agent if args.agent != "NoRobot" else "NoAgent"]
        euler_main(euler_argv)
        return

    orcagym_addr = args.addr
    agent_name = args.agent
    env_name = "SimulationLoop"
    run_simulation(orcagym_addr, agent_name, env_name)


if __name__ == "__main__":
    main()
