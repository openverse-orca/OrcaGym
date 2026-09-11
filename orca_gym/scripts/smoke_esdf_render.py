"""端到端冒烟：Studio level → ESDF 自动发现 → 柔体仿真 → 50451 推流 → 视口随动。

链路（P4/P5 验证）：
    Studio 自动导出 XML/ESDF（同目录同名）
      → OrcaGymEulerEnv(esdf_path="auto") 同名推导 ESDF
      → EulerSoftSim 双文件注入（semi_implicit shell_gas）
      → step 推进柔体（非耦合四步时序）
      → render() 节拍内 skinning → 通道 sync → gRPC 推流
      → Studio DeformableChannelComponent 接收 → CPU MLS 蒙皮 → 视口随动

前置条件：
    1. OrcaStudio 打开含可变形体的 level（如 water_1Cup）
    2. Studio 已启动仿真（导出 XML/ESDF 到 ~/Orca/OrcaStudio/<proj>/tmp/）
    3. level 内 DeformableChannelComponent 实体监听 50451

用法：
    python -m orca_gym.scripts.smoke_esdf_render \
        [--addr localhost:50051] [--target 127.0.0.1:50451] \
        [--device cpu|cuda:0] [--steps 600]
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
) -> int:
    """跑冒烟循环，返回实际执行的步数。"""
    env = EulerSimEnv(
        frame_skip=FRAME_SKIP,
        orcagym_addr=orcagym_addr,
        agent_names=["NoAgent"],
        time_step=TIME_STEP,
        esdf_path="auto",                    # 方案 A：XML 同名推导 ESDF
        euler_render_target=render_target,   # P5 改动点 2：渲染流推流
        device=device,
        render_mode="human",
    )

    gym = env.unwrapped._gym  # noqa: SLF001  冒烟诊断：仅调公共查询
    if not gym.has_soft_sim():
        _logger.error(
            "ESDF 柔体未注入（esdf_path='auto' 推导失败？）。"
            "请确认 Studio 已启动仿真且场景含可变形体。"
        )
        env.close()
        return 0
    print("[SMOKE] ESDF 柔体注入成功（EulerSoftSim 就绪）")
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
    parser.add_argument("--device", default="cpu", help="后端设备（cpu | cuda:0）")
    parser.add_argument("--steps", type=int, default=600, help="步数（默认 600 ≈ 12s）")
    args = parser.parse_args(argv)

    print(
        f"[SMOKE] 冒烟参数: addr={args.addr}, target={args.target}, "
        f"device={args.device}, steps={args.steps}"
    )
    n = run_smoke(args.addr, args.target, args.device, args.steps)
    print(f"[SMOKE] 冒烟结束，共 {n} 步")


if __name__ == "__main__":
    main()
