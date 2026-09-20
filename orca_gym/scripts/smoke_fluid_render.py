"""端到端冒烟（流体）：Studio level → ESDF 自动发现 → FluidGpuSim → 50452 推流 → 水面随动。

链路：
    Studio 自动导出 XML/ESDF（同目录同名，ESDF 含顶层 fluid.fluid_blocks）
      → OrcaGymEulerEnv(esdf_path="auto", fluid_render_target=..., device="cuda:0")
      → OrcaGymEuler._fluid = FluidGpuSim（SPH/DFSPH 独立流体槽）
      → do_simulation 内部 fluid.advance（按仿真时长，内部拆子步）
      → render() 节拍内 冻结读槽 → 密度归一化 → particle7 → gRPC
      → Studio FluidParticlesChannelComponent（50452）→ EulerFluidFP 水面

前置条件：
    1. OrcaStudio 打开含 Euler Fluid Block 组件的 level
    2. Studio 已启动仿真（导出 XML/ESDF 到 ~/Orca/OrcaStudio/<proj>/tmp/）
    3. level 内 FluidParticlesChannelComponent 实体监听 50452
    4. 本机 GPU 可用（device=cpu 会立刻报错）

成功判据：
    Studio 视口水面随仿真演进（涌动/沉降/静水面）；脚本结束后 T3 心跳
    超时（3s）恢复无水面是预期清理，不是失败。

用法：
    python -m orca_gym.scripts.smoke_fluid_render \
        [--addr localhost:50051] [--target 127.0.0.1:50452] \
        [--device cuda:0] [--steps 600]
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
    fluid_render_target: str,
    device: str,
    max_steps: Optional[int],
    render_mode: Optional[str] = None,
    render_param_pairs: Optional[list[tuple[str, str]]] = None,
    show_aabb: bool = False,
) -> int:
    """跑流体冒烟循环，返回实际执行的步数。"""
    env = EulerSimEnv(
        frame_skip=FRAME_SKIP,
        orcagym_addr=orcagym_addr,
        agent_names=["NoAgent"],
        time_step=TIME_STEP,
        esdf_path="auto",                       # XML 同名推导 ESDF（含 fluid 块）
        fluid_render_target=fluid_render_target,  # 流体粒子渲染流（50452）
        device=device,
        render_mode="human",
    )

    gym = env.unwrapped._gym  # noqa: SLF001  冒烟诊断：仅调公共查询
    if not gym.has_fluid_sim():
        _logger.error(
            "流体槽未注入（esdf_path='auto' 推导失败或 ESDF 无 fluid.fluid_blocks？）。"
            "请确认 Studio 已启动仿真且场景含 Euler Fluid Block 组件。"
        )
        env.close()
        return 0
    n_particles = gym.fluid_particle_count()
    solver_kind = gym.fluid_solver_kind()
    print(
        f"[SMOKE] 流体槽注入成功（FluidGpuSim 就绪，粒子数 {n_particles}，"
        f"求解器={solver_kind}）"
    )
    if gym.has_fluid_render_stream():
        print(f"[SMOKE] 流体渲染流已连接（render() 将推流 {n_particles} 粒子到 {fluid_render_target}）")
    else:
        _logger.warning(f"流体渲染流未连接（target={fluid_render_target}）")
        print(f"[SMOKE][WARN] 流体渲染流未连接（target={fluid_render_target}）")

    if render_mode is not None:
        gym.set_fluid_debug_mode(render_mode, show_aabb=show_aabb)
        mode_desc = {
            "debug": "粒子球视图（水面停更；查空中悬浮/嵌入地面）",
            "overlay": "球+数据叠加（水面继续更新）",
            "normal": "恢复水面",
        }.get(render_mode, render_mode)
        print(f"[SMOKE] 已切换渲染分支: {render_mode} = {mode_desc}")
        if show_aabb:
            print("[SMOKE] 已请求绘制通道 AABB 线框（对照薄层是否贴盒子顶面）")
    if render_param_pairs:
        params = dict(render_param_pairs)
        gym.set_fluid_render_params(params)
        print(f"[SMOKE] 已下发渲染参数: {params}")

    obs, info = env.reset()
    print("[SMOKE] 环境已 reset，开始步进。在 Studio 视口观察水面（涌动→沉降→静水）。")

    step = 0
    try:
        while max_steps is None or step < max_steps:
            start = datetime.now()
            action = np.zeros(env.unwrapped.nu, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step += 1
            if step % 100 == 0:
                print(
                    f"[SMOKE] step={step}, time={float(env.unwrapped.data.time):.3f}s, "
                    f"粒子数={n_particles}, 已推流帧号={gym.fluid_render_sequence()}"
                )
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed)
    except KeyboardInterrupt:
        print(f"[SMOKE] 冒烟中断（step={step}）")
    finally:
        # 正常退出清理：UnregisterChannel 同步销毁 debug 球 + 清空水面。
        # （异常断连时引擎侧 T3 心跳超时 3s 兜底清理。）
        gym.disable_fluid_render_stream()
        env.close()
    return step


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ESDF 流体渲染端到端冒烟")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--target", default="127.0.0.1:50452", help="流体渲染流 gRPC 地址")
    parser.add_argument("--device", default="cuda:0", help="后端设备（必须 cuda:0 / hip:0；cpu 会报错）")
    parser.add_argument("--steps", type=int, default=600, help="步数（默认 600 ≈ 12s）")
    parser.add_argument(
        "--render-mode",
        choices=["normal", "debug", "overlay"],
        default=None,
        help="渲染分支：debug=粒子球（水面停更，查悬浮/嵌入）；normal=恢复水面。默认不切换",
    )
    parser.add_argument(
        "--set-param",
        action="append",
        default=[],
        metavar="K=V",
        help="运行时调渲染参数（可多次）。如 --set-param anisotropy_mode=0 "
             "--set-param render_droplets=false",
    )
    parser.add_argument(
        "--show-aabb",
        action="store_true",
        help="绘制通道 AABB 线框（需配合 --render-mode debug/overlay；对照薄层是否贴盒子顶面）",
    )
    args = parser.parse_args(argv)

    pairs: list[tuple[str, str]] = []
    for item in args.set_param:
        if "=" not in item:
            parser.error(f"--set-param 需要 K=V 形式，收到 {item!r}")
        k, v = item.split("=", 1)
        pairs.append((k.strip(), v.strip()))

    print(
        f"[SMOKE] 冒烟参数: addr={args.addr}, target={args.target}, "
        f"device={args.device}, steps={args.steps}, "
        f"render_mode={args.render_mode}, set_param={pairs}"
    )
    n = run_smoke(
        args.addr, args.target, args.device, args.steps,
        render_mode=args.render_mode,
        render_param_pairs=pairs or None,
        show_aabb=args.show_aabb,
    )
    print(f"[SMOKE] 冒烟结束，共 {n} 步")


if __name__ == "__main__":
    main()
