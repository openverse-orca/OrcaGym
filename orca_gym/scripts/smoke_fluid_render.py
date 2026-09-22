"""端到端冒烟（流体）：Studio level → ESDF 自动发现 → CoupledGpuSim(FluidPhase) → 50452 推流 → 水面随动。

链路：
    Studio 自动导出 XML/ESDF（同目录同名，ESDF 含顶层 fluid.fluid_blocks）
      → OrcaGymEulerEnv(esdf_path="auto", fluid_render_target=..., device="cuda:0")
      → OrcaGymEuler._euler = CoupledGpuSim（只挂 FluidPhase，内部持有 FluidGpuSim）
      → do_simulation 只调 euler.step（内部 FluidPhase → FluidGpuSim.advance）
      → render() 节拍内 CoupledGpuSim.render_frame → 冻结读槽 → particle7 → gRPC
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
        [--device cuda:0] [--sim-time 12] [--steps 600] \
        [--render-mode debug] [--eulerviewer]

不传 --sim-time 且不传 --steps 时一直跑，按 Ctrl+C 结束。
--sim-time 的单位是仿真秒（MuJoCo data.time），不是墙钟。
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

import numpy as np

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scripts.sim_euler_env import EulerSimEnv

_logger = get_orca_logger()

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
WATER_COLOR = (0.20, 0.45, 0.85)
BOX_COLOR = (0.55, 0.55, 0.60)
DOMAIN_COLOR = (1.0, 0.60, 0.10)


def euler_xyz_matrix(degrees: list[float]) -> np.ndarray:
    """把 MuJoCo 的 xyz 欧拉角（度，内旋）变成 3x3 旋转矩阵。

    做什么：R = Rx @ Ry @ Rz，与场景 XML 里 body euler 的约定一致。
    为什么：Polyscope 里的碰撞盒必须和 Studio 里的板子朝向相同。
    """
    ax, ay, az = np.radians(degrees)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz


def read_collision_boxes(xml_path: str) -> list[dict]:
    """读 XML 里 type=box 的碰撞盒（世界系中心、旋转、半长）。

    做什么：用 body 的 pos/euler 和 geom 的 size 拼世界系盒子。
    为什么：Polyscope 没有 Studio 的容器网格，线框用来对照粒子有没有穿板。
    """
    root = ET.parse(xml_path).getroot()
    boxes: list[dict] = []
    for body in root.iter("body"):
        body_pos = np.array([float(v) for v in body.get("pos", "0 0 0").split()])
        rotation = euler_xyz_matrix(
            [float(v) for v in body.get("euler", "0 0 0").split()]
        )
        for geom in body.findall("geom"):
            if geom.get("type") != "box" or "size" not in geom.attrib:
                continue
            geom_pos = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
            half = np.array([float(v) for v in geom.get("size").split()][:3])
            boxes.append(
                {
                    "name": geom.get("name") or body.get("name", "box"),
                    "center": body_pos + geom_pos,
                    "rotation": rotation,
                    "half": half,
                }
            )
    return boxes


def box_wireframe(center: np.ndarray, rotation: np.ndarray, half: np.ndarray):
    """盒子 8 个角点和 12 条边，坐标已经变到世界系。"""
    hx, hy, hz = half
    corners = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    nodes = corners @ rotation.T + center
    edges = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    return nodes.astype(np.float32), edges


def run_studio_loop(env, gym, n_particles: int, max_steps: Optional[int], sim_time_limit: Optional[float]) -> int:
    """按原来的节拍步进，并把粒子推给 Studio。

    做什么：每步 env.step 再 env.render，墙钟不够 REALTIME_STEP 时补睡。
    为什么：不开 Polyscope 时保持原冒烟节奏，不把显示窗口嵌进循环。
    """
    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            print(f"[SMOKE] 已到步数上限 {max_steps}")
            break
        start = datetime.now()
        action = np.zeros(env.unwrapped.nu, dtype=np.float32)
        env.step(action)
        env.render()
        step += 1
        sim_time = float(env.unwrapped.data.time)
        if step % 100 == 0:
            print(
                f"[SMOKE] step={step}, time={sim_time:.3f}s, "
                f"粒子数={n_particles}, 已推流帧号={gym.fluid_render_sequence()}"
            )
        if sim_time_limit is not None and sim_time >= sim_time_limit:
            print(f"[SMOKE] 已到仿真时间上限 {sim_time_limit:g}s（当前 {sim_time:.3f}s）")
            break
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed)
    return step


def run_euler_viewer(env, gym, n_particles: int, max_steps: Optional[int], sim_time_limit: Optional[float]) -> int:
    """打开 Polyscope，每帧推进一步，同时继续把同一帧推给 Studio。

    做什么：窗口回调里 env.step、env.render，再用流体粒子坐标刷新点云。
    为什么：Polyscope 的 show() 占住主线程，步进必须放进它的帧回调，
    否则 Studio 循环和窗口循环不能一起跑。
    """
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("[SMOKE] polyscope 未安装，改为只推 Studio。")
        return run_studio_loop(env, gym, n_particles, max_steps, sim_time_limit)

    positions = gym.fluid_particle_positions()
    ps.init()
    ps.set_program_name("Euler viewer")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("tile")
    ps.set_ground_plane_height_factor(0.0)
    if len(positions) > 0:
        center = positions.mean(axis=0)
        span = max(float(np.ptp(positions, axis=0).max()), 1.0)
        ps.look_at(
            camera_location=(
                float(center[0] + 1.8 * span),
                float(center[1] - 2.2 * span),
                float(center[2] + 1.2 * span),
            ),
            target=(float(center[0]), float(center[1]), float(center[2])),
        )

    xml_path = gym.fluid_scene_xml()
    if xml_path:
        for box in read_collision_boxes(xml_path):
            nodes, edges = box_wireframe(box["center"], box["rotation"], box["half"])
            curve = ps.register_curve_network(f"box/{box['name']}", nodes, edges)
            curve.set_radius(0.008, relative=False)
            curve.set_color(BOX_COLOR)

    bounds = gym.fluid_bounds()
    if bounds is not None:
        low, high = bounds
        center = np.array([(low[i] + high[i]) / 2.0 for i in range(3)])
        half = np.array([(high[i] - low[i]) / 2.0 for i in range(3)])
        nodes, edges = box_wireframe(center, np.eye(3), half)
        domain = ps.register_curve_network("physics_bounds", nodes, edges)
        domain.set_radius(0.006, relative=False)
        domain.set_color(DOMAIN_COLOR)

    cloud = ps.register_point_cloud("water", positions, color=WATER_COLOR)
    cloud.set_radius(float(gym.fluid_particle_radius()), relative=False)

    ui = {"paused": False, "step": 0}

    def on_frame() -> None:
        sim_time = float(env.unwrapped.data.time)
        psim.TextUnformatted(
            f"step = {ui['step']}   t = {sim_time:.3f} s   particles = {n_particles}"
        )
        changed, paused = psim.Checkbox("paused", ui["paused"])
        if changed:
            ui["paused"] = paused
        if ui["paused"]:
            return
        if max_steps is not None and ui["step"] >= max_steps:
            print(f"[SMOKE] 已到步数上限 {max_steps}")
            ps.unshow()
            return
        if sim_time_limit is not None and sim_time >= sim_time_limit:
            print(f"[SMOKE] 已到仿真时间上限 {sim_time_limit:g}s（当前 {sim_time:.3f}s）")
            ps.unshow()
            return
        action = np.zeros(env.unwrapped.nu, dtype=np.float32)
        env.step(action)
        env.render()
        ui["step"] += 1
        cloud.update_point_positions(gym.fluid_particle_positions())
        if ui["step"] % 100 == 0:
            print(
                f"[SMOKE] step={ui['step']}, time={float(env.unwrapped.data.time):.3f}s, "
                f"粒子数={n_particles}, 已推流帧号={gym.fluid_render_sequence()}"
            )

    ps.set_user_callback(on_frame)
    print("[SMOKE] Polyscope 已打开。Studio 仍接收同一帧粒子。")
    ps.show()
    return int(ui["step"])


def run_smoke(
    orcagym_addr: str,
    fluid_render_target: str,
    device: str,
    max_steps: Optional[int],
    sim_time_limit: Optional[float] = None,
    render_mode: Optional[str] = None,
    render_param_pairs: Optional[list[tuple[str, str]]] = None,
    show_aabb: bool = False,
    euler_viewer: bool = False,
) -> int:
    """跑流体冒烟循环，返回实际执行的步数。

    max_steps 是环境步数上限；sim_time_limit 是仿真时钟上限（秒，看 data.time）。
    两者都为空时不自动停止。两者都有值时，先到达的条件先停。
    """
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
            "流体相未注入（esdf_path='auto' 推导失败或 ESDF 无 fluid.fluid_blocks？）。"
            "请确认 Studio 已启动仿真且场景含 Euler Fluid Block 组件。"
        )
        env.close()
        return 0
    n_particles = gym.fluid_particle_count()
    solver_kind = gym.fluid_solver_kind()
    print(
        f"[SMOKE] 流体相注入成功（CoupledGpuSim FluidPhase 就绪，粒子数 {n_particles}，"
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

    if euler_viewer:
        print("[SMOKE] 将同时打开 Polyscope。关窗或 Ctrl+C 结束。勾选 paused 时 Studio 也不再推进。")
    elif sim_time_limit is None and max_steps is None:
        print("[SMOKE] 未指定仿真时间或步数，将一直运行，按 Ctrl+C 结束。")
    elif sim_time_limit is not None:
        print(f"[SMOKE] 仿真时间上限 {sim_time_limit:g}s（按 data.time）。")

    step = 0
    try:
        if euler_viewer:
            step = run_euler_viewer(env, gym, n_particles, max_steps, sim_time_limit)
        else:
            step = run_studio_loop(env, gym, n_particles, max_steps, sim_time_limit)
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
    parser.add_argument(
        "--eulerviewer",
        action="store_true",
        help="同时打开 Polyscope，和 Studio 看同一份流体粒子",
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
        f"render_mode={args.render_mode}, eulerviewer={args.eulerviewer}, "
        f"set_param={pairs}"
    )
    n = run_smoke(
        args.addr, args.target, args.device, args.steps,
        render_mode=args.render_mode,
        render_param_pairs=pairs or None,
        show_aabb=args.show_aabb,
        euler_viewer=args.eulerviewer,
    )
    print(f"[SMOKE] 冒烟结束，共 {n} 步")


if __name__ == "__main__":
    main()
