"""端到端冒烟：Studio level → ESDF 自动发现 → 全 GPU 耦合 → 50451 推流 → 视口随动。

链路：
    Studio 自动导出 XML/ESDF（同目录同名）
      → OrcaGymEulerEnv(esdf_path="auto", device="cuda:0")
      → Euler CoupledGpuSim（MuJoCoFlow 刚体 + XPBD/SemiImplicit 柔体）
      → CouplingOrchestrator GPU 直拷（Gym 不感知位姿/力交换）
      → render() 节拍内 skinning → 通道 sync → gRPC 推流
      → Studio DeformableChannelComponent 接收 → CPU MLS 蒙皮 → 视口随动

前置条件：
    1. OrcaStudio 打开含可变形体的 level（如 Euler_xpbdCloth）
    2. Studio 已启动仿真（导出 XML/ESDF 到 ~/Orca/OrcaStudio/<proj>/tmp/）
    3. level 内 DeformableChannelComponent 实体监听 50451
    4. 本机 GPU 可用（device=cpu 会立刻报错）

用法：
    python -m orca_gym.scripts.smoke_esdf_render \
        [--addr localhost:50051] [--target 127.0.0.1:50451] \
        [--device cuda:0] [--steps 600] [--graph]
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import numpy as np

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scripts.sim_euler_env import EulerSimEnv

_logger = get_orca_logger()

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP


def run_smoke(
    orcagym_addr: str,
    render_target: str,
    device: str,
    max_steps: Optional[int],
    graph: bool,
) -> int:
    """跑冒烟循环，返回实际执行的步数。

    ``graph=True`` 时与隔离课 ``robot_cloth_latest.py --graph`` 对齐：
    刚体内层图关掉，reset 后把一个耦合窗录成外层 CUDA Graph。
    """
    env = EulerSimEnv(
        frame_skip=FRAME_SKIP,
        orcagym_addr=orcagym_addr,
        agent_names=["NoAgent"],
        time_step=TIME_STEP,
        esdf_path="auto",                    # 方案 A：XML 同名推导 ESDF
        euler_render_target=render_target,   # P5 改动点 2：渲染流推流
        device=device,
        render_mode="human",
        cycle_graph=graph,
    )

    gym = env.unwrapped._gym  # noqa: SLF001  冒烟诊断：仅调公共查询
    if not gym.has_soft_sim():
        _logger.error(
            "ESDF 柔体未注入（esdf_path='auto' 推导失败？）。"
            "请确认 Studio 已启动仿真且场景含可变形体。"
        )
        env.close()
        return 0
    print("[SMOKE] ESDF 柔体注入成功（CoupledGpuSim 就绪）")
    if graph:
        print("[SMOKE] 外层耦合窗图已开启（刚体内层图关闭，与隔离课 --graph 对齐）")
    if gym.has_render_stream():
        print(f"[SMOKE] 渲染流已连接（render() 将推流到 {render_target}）")
    else:
        _logger.warning(f"渲染流未连接（target={render_target}）")
        print(f"[SMOKE][WARN] 渲染流未连接（target={render_target}）")

    obs, info = env.reset()
    print("[SMOKE] 环境已 reset，开始步进。在 Studio 视口观察可变形体（如 Bear_Euler）。")

    step = 0
    try:
        while max_steps is None or step < max_steps:
            start = datetime.now()
            action = np.zeros(env.unwrapped.nu, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step += 1
            if step % 100 == 0:
                print(f"[SMOKE] step={step}, time={float(env.unwrapped.data.time):.3f}s")
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed)
    except KeyboardInterrupt:
        print(f"[SMOKE] 冒烟中断（step={step}）")
    finally:
        env.close()
    return step


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ESDF 柔体渲染端到端冒烟")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--target", default="127.0.0.1:50451", help="渲染流 gRPC 地址")
    parser.add_argument("--device", default="cuda:0", help="后端设备（必须 cuda:0 / hip:0；cpu 会报错）")
    parser.add_argument("--steps", type=int, default=600, help="步数（默认 600 ≈ 12s）")
    parser.add_argument(
        "--graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="打开外层耦合窗 CUDA Graph（关刚体内层图，reset 后录一整窗；与隔离课 --graph 相同）",
    )
    args = parser.parse_args(argv)

    print(
        f"[SMOKE] 冒烟参数: addr={args.addr}, target={args.target}, "
        f"device={args.device}, steps={args.steps}, graph={args.graph}"
    )
    n = run_smoke(args.addr, args.target, args.device, args.steps, args.graph)
    print(f"[SMOKE] 冒烟结束，共 {n} 步")


if __name__ == "__main__":
    main()
