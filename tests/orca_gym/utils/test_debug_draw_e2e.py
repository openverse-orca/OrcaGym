"""OrcaDebugMesh 路径 A 端到端视觉验收脚本（人工运行）。

前置条件：
    1. OrcaStudio 已启动并加载含 Mujoco 场景的关卡
    2. OrcaDebugMesh Gem 已编译并加载
    3. 仿真已进入“运行”状态（Editor 仿真播放）
    4. 使用 orca conda 环境运行（${ORCA_PYTHON}）

运行：
    ${ORCA_PYTHON} tests/orca_gym/utils/test_debug_draw_e2e.py [--addr localhost:50051]
    ${ORCA_PYTHON} tests/orca_gym/utils/test_debug_draw_e2e.py --phase 1 3 5   # 只跑指定阶段

交互方式：
    本脚本连接在线 OrcaStudio，分阶段调用 debug_draw() 接口绘制
    OrcaDebugMesh 支持的全部图形，供人工在视口中观察验收。
    每个阶段/子步骤绘制完成后持续渲染，**按空格键进入下一步**，
    无超时限制，方便仔细观察。按 Ctrl+C 中断退出。

    Immediate（单帧）图形在等待期间每帧重新提交；Retained（持久）
    图形创建一次即跨帧存在。EulerSimEnv 内部使用零控制，不会让场景
    机器人乱动，可放心 step + render。

验收清单（对照 orca_debug_mesh_path_a_implementation_guide.md）：
    [Phase 1] Immediate 模式 6 种网格类型（Sphere/Cylinder/Cone/Box/Quad/Arrow）
    [Phase 2] InstanceFlags.EdgeHighlight 边缘高亮
    [Phase 3] 透明度（alpha < 1）混合
    [Phase 4] clear() 清空 immediate 队列
    [Phase 5] Retained create_objects + query_count 持久化
    [Phase 6] Retained update_transforms 动画
    [Phase 7] Retained destroy_objects 销毁
    [Phase 8] 坐标系与单位规范目视验收（设计文档 §2.3）：
             Z-up 轴向、单位米、+Y 模型空间、非均匀缩放、箭头方向
    [Phase 9] Immediate TTL（duration>0）跨帧存活 + 到期自动消失
    [Phase 10] Retained keepalive 保活心跳：过期销毁、心跳续期、断连自清
    [Phase 11] TTL=0 闪烁效果演示（为何 immediate 推荐 TTL>=0.1）
    [Phase 12] Wireframe 线框渲染（W1-W5）：solid/wire 对照、半透明线框、retained 线框
    [Phase 13] Custom Mesh（Phase C/O）：顶点+索引注册、OBJ 加载、批量绘制、注销
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import tty
import termios
from typing import Callable, List

import gymnasium as gym
import numpy as np

from orca_gym.scripts.sim_euler_env import EulerSimEnv  # noqa: F401  (确保 entry_point 可导入)
from orca_gym.utils.orca_debug_draw import (
    DebugMeshType,
    InstanceFlags,
    _direction_to_quat,
    _make_instance,
)

# ============================================================
# 常量
# ============================================================
TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP  # 0.02s，单帧墙钟步长

# Immediate 模式默认 TTL（秒）。>=0.1 跨帧存活，规避渲染节流（30Hz render
# vs 50Hz step）导致的闪烁。TTL=0 为单帧，仅用于 Phase 11 闪烁效果演示。
IMMEDIATE_TTL = 0.1

# 颜色 RGBA，范围 0..1
RED = [1.0, 0.0, 0.0, 1.0]
GREEN = [0.0, 1.0, 0.0, 1.0]
BLUE = [0.0, 0.2, 1.0, 1.0]
YELLOW = [1.0, 1.0, 0.0, 1.0]
MAGENTA = [1.0, 0.0, 1.0, 1.0]
CYAN = [0.0, 1.0, 1.0, 1.0]

Z_ROW = 1.0  # 一排图形的统一高度


# ============================================================
# 按键监听（后台线程，空格键触发下一步）
# ============================================================
class KeyListener:
    """后台线程监听空格键。主线程在 wait_for_space() 中等待事件。

    仅在终端（TTY）下工作；非 TTY 环境（如 IDE 输出重定向）会退化为
    行输入（按回车继续）。

    实现要点（控制台模式管理）：
      - 使用 ``tty.setcbreak`` 而非 ``tty.setraw``。cbreak 保留 OPOST
        （\n → \r\n 输出后处理，保证 print 格式正常）和 ISIG
        （Ctrl+C 产生 SIGINT 正常中断）；只关闭 ICANON（行缓冲）和 ECHO
        （按键不回显，适合空格监听）。setraw 会关闭这两者，导致输出呈
        阶梯状且 Ctrl+C 失效。
      - termios 的保存/恢复由主线程在 start()/stop() 完成，不依赖子线程
        的 finally。子线程阻塞在 read(1) 上，进程退出时被强杀，其 finally
        未必执行 → 终端停留在 cbreak → 退出后输入不回显。主线程 stop()
        一定执行，确保恢复。
    """

    def __init__(self):
        self._event = threading.Event()
        self._stop_flag = False
        self._thread: threading.Thread | None = None
        self._is_tty = sys.stdin.isatty()
        self._old_settings = None

    def start(self) -> None:
        if not self._is_tty:
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag = True
        self._event.set()  # 唤醒可能阻塞的等待
        # 主线程恢复 termios：不依赖子线程的 finally（子线程可能阻塞在
        # read(1) 上，进程退出时被强杀，finally 未必执行 → 终端停留在
        # cbreak 模式 → 退出后输入不回显）。
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except OSError:
                pass
            self._old_settings = None

    def wait_for_space(self, prompt: str = "按空格键继续...") -> None:
        """阻塞直到空格键被按下。期间打印提示。"""
        print(f"\n  >>> {prompt}")
        self._event.wait()
        self._event.clear()

    def was_pressed(self) -> bool:
        """非阻塞检查空格键是否已被按下。若已按下则清除事件并返回 True。"""
        if self._event.is_set():
            self._event.clear()
            return True
        return False

    def _loop(self) -> None:
        """后台读取按键，空格触发事件。"""
        try:
            while not self._stop_flag:
                ch = sys.stdin.read(1)
                if ch == " ":
                    self._event.set()
                elif ch == "\x03":  # Ctrl+C（cbreak 下 ISIG 保留，通常已由 SIGINT 处理）
                    self._stop_flag = True
                    self._event.set()
                    break
        except (OSError, ValueError):
            # stdin 关闭或被中断时退出
            self._event.set()


# ============================================================
# 辅助
# ============================================================
def run(env, coro):
    """在 env 的事件循环上同步执行 async 协程。"""
    return env.loop.run_until_complete(coro)


def force_render(env) -> None:
    """绕过 env.render() 的 30Hz 墙钟节流，强制立即刷新一帧到 OrcaStudio。

    immediate 模式下，绘制的图形在 Studio 端每次 Simulate 后被清空。
    若 env.render() 节流跳过本帧（render_fps=30，而 hold 节拍为 50Hz），
    则提交的 immediate 图形不会触发刷新就被下次 Simulate 清掉 → 闪烁。
    immediate 阶段每帧用 force_render 保证提交与刷新严格 1:1。
    """
    env.loop.run_until_complete(env._gym.render())


def render_until_key(env, key_listener: KeyListener, redraw: Callable[[], None] | None = None,
                     prompt: str = "按空格键继续...", force: bool = True) -> None:
    """持续渲染直到空格键被按下。redraw 非 None 时每帧重新提交（immediate 模式）。

    每帧：[redraw] -> step(零控制) -> render -> sleep(剩余时间)。
    计时对齐墙钟（扣除 step/render/gRPC 耗时），保证物理 time_step=0.001
    即 1000 Hz 子步进、Gym step 以 50 Hz (REALTIME_STEP=0.02s) 节流，
    使仿真进度与实时 1:1。EulerSimEnv.step 内部用零控制，场景不会乱动。

    immediate 模式（redraw 非 None）：默认用 force_render 绕过 env.render() 的
    30Hz 节流，确保每帧提交的图形都触发刷新（否则被节流跳过的帧图形会被
    下次 Simulate 清空 → 闪烁）。设 force=False 可保留 30Hz 节流，用于
    Phase 11 演示 TTL=0 的闪烁效果（TTL>=0.1 则节流下仍稳定）。

    与旧 hold() 的区别：不再按固定秒数退出，而是等待空格键。
    """
    action = env.action_space.sample()
    print(f"\n  >>> {prompt}")
    while not key_listener.was_pressed():  # 非阻塞检查，未按键则继续渲染
        frame_start = time.time()
        if redraw is not None:
            redraw()
        env.step(action)
        # immediate 模式需每帧刷新；retained 模式普通 render 即可
        if redraw is not None and force:
            force_render(env)
        else:
            env.render()
        elapsed = time.time() - frame_start
        if elapsed < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed)


def hold(env, seconds: float, redraw: Callable[[], None] | None = None) -> None:
    """持续渲染 `seconds` 秒（保留给 Phase 6 动画用，基于墙钟计时）。

    Phase 6 的圆周动画需要基于时间连续更新 transform，不适合按键切换，
    故保留此基于时间的版本。
    """
    action = env.action_space.sample()
    end = time.time() + seconds
    while time.time() < end:
        frame_start = time.time()
        if redraw is not None:
            redraw()
        env.step(action)
        if redraw is not None:
            force_render(env)
        else:
            env.render()
        elapsed = time.time() - frame_start
        if elapsed < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed)


def make_env(orcagym_addr: str):
    """注册并构造 EulerSimEnv（在线模式，连接 OrcaStudio）。"""
    env_id = "DebugDrawE2E-OrcaGym-" + orcagym_addr.replace(":", "-") + "-000"
    gym.register(
        id=env_id,
        entry_point="orca_gym.scripts.sim_euler_env:EulerSimEnv",
        kwargs={
            "frame_skip": FRAME_SKIP,
            "orcagym_addr": orcagym_addr,
            "agent_names": ["NoAgent"],
            "time_step": TIME_STEP,
        },
        max_episode_steps=sys.maxsize,
    )
    return gym.make(env_id)


# ============================================================
# Phase 1：Immediate 模式 6 种网格类型
# ============================================================
def phase1_all_mesh_types(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 1] Immediate 模式：6 种网格类型（3 组对照）")
    print("  第 1 组（原始方向，z=%.1f, y=0）：所有图形视觉居中于 z" % Z_ROW)
    print("    红球 | 绿圆柱(竖) | 蓝圆锥(朝上) | 黄方块 | 品红双面 Quad | 青箭头(斜向上)")
    print("  第 2 组（锚点对齐，z=0, y=2）：所有图形 position=[x,2,0]")
    print("    无方向性基元居中于 z=0；方向性基元底面/杆底在 z=0，向上延伸至 z=0.6")
    print("  第 3 组（绕 X 轴旋转 90°，z=1, y=-2）：方向性基元指向 -Y")
    z = Z_ROW

    # 朝上四元数（+Y → +Z）
    up_quat = _direction_to_quat(np.array([0.0, 0.0, 1.0]))
    # 绕 X 轴旋转 90° 的四元数（+Y → -Y）
    rot_x90 = _direction_to_quat(np.array([0.0, -1.0, 0.0])).tolist()

    # 方向性基元的半高（scale.y / 2），用于视觉居中补偿
    H_HALF = 0.3  # scale.y=0.6 → 半高 0.3

    def redraw():
        # ========== 第 1 组：原始方向 (z=Z_ROW, y=0) — 视觉居中 ==========
        # 无方向性：position = 中心
        run(env, dd.draw_sphere([-2.5, 0, z], 0.3, RED, duration=IMMEDIATE_TTL))
        run(env, dd.draw_box([0.5, 0, z], [0.4, 0.4, 0.4], YELLOW, duration=IMMEDIATE_TTL))
        run(env, dd.draw_quad([1.5, 0, z], [0.5, 0.5, 1.0], MAGENTA, duration=IMMEDIATE_TTL))
        # 方向性：from=z-H_HALF, to=z+H_HALF 使底面在 z-H_HALF、顶面在 z+H_HALF（视觉居中）
        run(env, dd.draw_cylinder([-1.5, 0, z - H_HALF], [-1.5, 0, z + H_HALF], 0.2, GREEN, duration=IMMEDIATE_TTL))
        # Cone：position = 底面中心 = z-H_HALF，rotation +Y→+Z，scale=[r, h, r]
        cone_inst = _make_instance([-0.5, 0, z - H_HALF], up_quat.tolist(), [0.25, 0.6, 0.25], BLUE)
        run(env, dd.draw_batch(DebugMeshType.CONE, [cone_inst], duration=IMMEDIATE_TTL))
        run(env, dd.draw_arrow([2.3, 0, z - H_HALF], [2.3, 0, z + H_HALF], 0.05, CYAN, duration=IMMEDIATE_TTL))

        # ========== 第 2 组：锚点对齐 (z=0, y=2) — 演示 position 锚点语义 ==========
        # 所有图形 position = [x, 2, 0]，观察锚点差异：
        #   无方向性 → 几何中心在 z=0
        #   方向性   → 底面/杆底在 z=0，向上延伸至 z=0.6
        run(env, dd.draw_sphere([-2.5, 2, 0.0], 0.3, RED, duration=IMMEDIATE_TTL))
        run(env, dd.draw_box([0.5, 2, 0.0], [0.4, 0.4, 0.4], YELLOW, duration=IMMEDIATE_TTL))
        run(env, dd.draw_quad([1.5, 2, 0.0], [0.5, 0.5, 1.0], MAGENTA, duration=IMMEDIATE_TTL))
        # 方向性：position = [x, 2, 0]（底面/杆底），延伸方向 +Z
        run(env, dd.draw_cylinder([-1.5, 2, 0.0], [-1.5, 2, 0.6], 0.2, GREEN, duration=IMMEDIATE_TTL))
        cone_inst2 = _make_instance([-0.5, 2, 0.0], up_quat.tolist(), [0.25, 0.6, 0.25], BLUE)
        run(env, dd.draw_batch(DebugMeshType.CONE, [cone_inst2], duration=IMMEDIATE_TTL))
        run(env, dd.draw_arrow([2.3, 2, 0.0], [2.3, 2, 0.6], 0.05, CYAN, duration=IMMEDIATE_TTL))

        # ========== 第 3 组：绕 X 轴旋转 90° (z=1, y=-2) — 方向性基元指向 -Y ==========
        # Sphere/Box/Quad 无方向性，旋转后外观不变
        run(env, dd.draw_sphere([-2.5, -2, 1.0], 0.3, RED, duration=IMMEDIATE_TTL))
        run(env, dd.draw_box([0.5, -2, 1.0], [0.4, 0.4, 0.4], YELLOW, duration=IMMEDIATE_TTL))
        run(env, dd.draw_quad([1.5, -2, 1.0], [0.5, 0.5, 1.0], MAGENTA, duration=IMMEDIATE_TTL))
        # 方向性：from = y+H_HALF（底面），to = y-H_HALF（顶面/尖端），方向 = -Y
        run(env, dd.draw_cylinder([-1.5, -1.7, 1.0], [-1.5, -2.3, 1.0], 0.2, GREEN, duration=IMMEDIATE_TTL))
        # Cone：position = 底面中心 = [-0.5, -1.7, 1.0]，rotation +Y→-Y
        cone_inst3 = _make_instance([-0.5, -1.7, 1.0], rot_x90, [0.25, 0.6, 0.25], BLUE)
        run(env, dd.draw_batch(DebugMeshType.CONE, [cone_inst3], duration=IMMEDIATE_TTL))
        run(env, dd.draw_arrow([2.3, -1.7, 1.0], [2.3, -2.3, 1.0], 0.05, CYAN, duration=IMMEDIATE_TTL))

    render_until_key(env, key_listener, redraw,
                     "观察 3 组图形：第1组居中 / 第2组锚点对齐 / 第3组旋转，按空格键进入 Phase 2")


# ============================================================
# Phase 2：EdgeHighlight 边缘高亮
# ============================================================
def phase2_edge_highlight(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 2] InstanceFlags.EdgeHighlight 边缘高亮")
    print("  预期：每对图形中，左侧无高亮、右侧带高亮（轮廓/边缘发亮）")
    z = Z_ROW

    def redraw():
        # 球：左无 / 右高亮
        run(env, dd.draw_sphere([-1.5, 0, z], 0.3, RED, flags=InstanceFlags.NONE, duration=IMMEDIATE_TTL))
        run(env, dd.draw_sphere([-0.5, 0, z], 0.3, RED, flags=InstanceFlags.EDGE_HIGHLIGHT, duration=IMMEDIATE_TTL))
        # 盒：左无 / 右高亮（盒子边缘最明显）
        run(env, dd.draw_box([0.5, 0, z], [0.4, 0.4, 0.4], YELLOW,
                             flags=InstanceFlags.NONE, duration=IMMEDIATE_TTL))
        run(env, dd.draw_box([1.5, 0, z], [0.4, 0.4, 0.4], YELLOW,
                             flags=InstanceFlags.EDGE_HIGHLIGHT, duration=IMMEDIATE_TTL))

    render_until_key(env, key_listener, redraw, "观察边缘高亮对比，按空格键进入 Phase 3")


# ============================================================
# Phase 3：透明度（alpha < 1）混合
# ============================================================
def phase3_transparency(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 3] 透明度（alpha=0.5）混合")
    print("  预期：三个半透明球相互重叠，透过前面可见后面（SrcAlpha/InvSrcAlpha blend）")
    z = Z_ROW

    def redraw():
        run(env, dd.draw_sphere([-0.3, -0.3, z], 0.4, [1.0, 0.0, 0.0, 0.5], duration=IMMEDIATE_TTL))
        run(env, dd.draw_sphere([0.3, -0.3, z], 0.4, [0.0, 1.0, 0.0, 0.5], duration=IMMEDIATE_TTL))
        run(env, dd.draw_sphere([0.0, 0.3, z], 0.4, [0.0, 0.2, 1.0, 0.5], duration=IMMEDIATE_TTL))

    render_until_key(env, key_listener, redraw, "观察半透明混合，按空格键进入 Phase 4")


# ============================================================
# Phase 4：clear() 清空 immediate 队列（两步：显示 → 清空）
# ============================================================
def phase4_clear(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 4] clear() 清空 immediate 队列")
    print("  预期：先显示一球一盒，按空格键后调用 clear()，图形消失")
    z = Z_ROW

    def draw_shapes():
        run(env, dd.draw_sphere([-1.0, 0, z], 0.3, RED, duration=IMMEDIATE_TTL))
        run(env, dd.draw_box([1.0, 0, z], [0.4, 0.4, 0.4], GREEN, duration=IMMEDIATE_TTL))

    # 步骤 1：显示图形
    render_until_key(env, key_listener, draw_shapes, "观察图形显示，按空格键触发 clear()")

    # 步骤 2：调用 clear() 后持续渲染（无提交，应消失）
    print("  >> 调用 clear()")
    run(env, dd.clear())
    # clear 后立即强制刷新一帧，确保清空效果可见
    env.step(env.action_space.sample())
    force_render(env)
    render_until_key(env, key_listener, None, "观察图形已消失，按空格键进入 Phase 5")


# ============================================================
# Phase 5：Retained create_objects + query_count 持久化
# ============================================================
def phase5_retained_create(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 5] Retained create_objects + query_count")
    print("  预期：创建 6 个持久对象（每种类型一个），按空格键前跨帧持续显示（无需重提交）")
    z = Z_ROW
    # Cylinder/Cone/Arrow 沿 +Y 建模，position = 底面/杆底（C++ 锚点）。
    # 为视觉居中于 z，position.z = z - scale.y/2（底面在下，顶面在上）
    up_quat = _direction_to_quat(np.array([0.0, 0.0, 1.0])).tolist()
    z_cyl = z - 0.3  # scale.y=0.6 → 底面在 z-0.3，顶面在 z+0.3
    z_cone = z - 0.3
    z_arrow = z - 0.3
    specs = [
        (DebugMeshType.SPHERE,   _make_instance([-2.5, 0, z], [0, 0, 0, 1], [0.3, 0.3, 0.3], RED),     "Sphere"),
        (DebugMeshType.CYLINDER, _make_instance([-1.5, 0, z_cyl], up_quat, [0.2, 0.6, 0.2], GREEN),    "Cylinder"),
        (DebugMeshType.CONE,     _make_instance([-0.5, 0, z_cone], up_quat, [0.25, 0.6, 0.25], BLUE),  "Cone"),
        (DebugMeshType.BOX,      _make_instance([0.5, 0, z],  [0, 0, 0, 1], [0.4, 0.4, 0.4], YELLOW),  "Box"),
        (DebugMeshType.QUAD,     _make_instance([1.5, 0, z],  [0, 0, 0, 1], [0.5, 0.5, 1.0], MAGENTA), "Quad"),
        (DebugMeshType.ARROW,    _make_instance([2.5, 0, z_arrow], up_quat, [0.05, 0.6, 0.05], CYAN),  "Arrow"),
    ]
    handles: List[dict] = []
    for mtype, inst, name in specs:
        hs = run(env, dd.create_objects(mtype, [inst]))
        valid = [h for h in hs if h["valid"]]
        handles.extend(valid)
        print(f"  create {name}: {hs}")
    count = run(env, dd.query_count())
    print(f"  query_count = {count}（retained 应 >= {len(handles)}）")

    render_until_key(env, key_listener, None, "观察 6 个持久对象跨帧显示，按空格键销毁并进入 Phase 6")

    run(env, dd.destroy_objects(handles))
    count2 = run(env, dd.query_count())
    print(f"  销毁 {len(handles)} 个对象后 query_count = {count2}")


# ============================================================
# Phase 6：Retained update_transforms 动画（基于时间，动画连续）
# ============================================================
def phase6_retained_animate(env, key_listener: KeyListener, duration: float = 6.0) -> None:  # noqa: ARG001
    dd = env.debug_draw()
    print("\n[Phase 6] Retained update_transforms 动画")
    print("  预期：4 个球绕中心做圆周运动（验证 update_transforms 实时更新）")
    print(f"  动画运行 {duration:.0f} 秒后自动进入下一步（动画需基于时间连续更新）")
    n = 4
    radius = 1.0
    init_insts = [
        _make_instance([radius, 0, Z_ROW], [0, 0, 0, 1], [0.2, 0.2, 0.2], CYAN)
        for _ in range(n)
    ]
    hs = run(env, dd.create_objects(DebugMeshType.SPHERE, init_insts))
    handles = [h for h in hs if h["valid"]]
    print(f"  创建 {len(handles)} 个球用于动画")

    action = env.action_space.sample()
    end = time.time() + duration
    t = 0.0
    while time.time() < end:
        new_insts = []
        for i in range(len(handles)):
            angle = t * 1.5 + i * (2 * np.pi / max(len(handles), 1))
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            new_insts.append(
                _make_instance([x, y, Z_ROW], [0, 0, 0, 1], [0.2, 0.2, 0.2], CYAN)
            )
        run(env, dd.update_transforms(handles, new_insts))
        env.step(action)
        env.render()
        t += REALTIME_STEP
        time.sleep(REALTIME_STEP)

    run(env, dd.destroy_objects(handles))
    print("  动画球已销毁")


# ============================================================
# Phase 7：Retained destroy_objects 销毁（两步：显示 → 销毁）
# ============================================================
def phase7_retained_destroy(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 7] Retained destroy_objects 销毁")
    print("  预期：先显示 3 个球，按空格键后调用 destroy，球消失，query_count retained 减少")
    z = Z_ROW
    insts = [
        _make_instance([-1.0, 0, z], [0, 0, 0, 1], [0.3, 0.3, 0.3], RED),
        _make_instance([0.0, 0, z],  [0, 0, 0, 1], [0.3, 0.3, 0.3], GREEN),
        _make_instance([1.0, 0, z],  [0, 0, 0, 1], [0.3, 0.3, 0.3], BLUE),
    ]
    hs = run(env, dd.create_objects(DebugMeshType.SPHERE, insts))
    handles = [h for h in hs if h["valid"]]
    before = run(env, dd.query_count())
    print(f"  创建 {len(handles)} 个球，query_count={before}")

    render_until_key(env, key_listener, None, "观察 3 个球显示，按空格键触发 destroy_objects()")

    print("  >> 调用 destroy_objects()")
    run(env, dd.destroy_objects(handles))
    after = run(env, dd.query_count())
    print(f"  销毁后 query_count={after}（retained 应减少）")
    render_until_key(env, key_listener, None, "观察球已消失，按空格键进入 Phase 8")


# ============================================================
# Phase 8：坐标系与单位规范目视验收（对应设计文档 §2.3）
# 逐项验证 Z-up 右手系、单位米、+Y 模型空间约定、四元数顺序。
# 每个子阶段绘制已知几何，按空格键切换到下一个子阶段，
# 人工对照检查清单核对方向/尺寸/缩放。
# ============================================================
def phase8_coordinate_system(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 8] 坐标系与单位规范目视验收（设计文档 §2.3）")
    print("  本阶段分 5 组，每组绘制已知几何并打印检查清单，按空格键切换下一组。")
    print("  坐标系：Z-up 右手系（X=右，Y=前，Z=上），单位米。")

    # ---------- 8.1 Z-up 轴向验证：球体在三个轴上 ----------
    print("\n  [8.1] Z-up 轴向验证：三球分别在 X/Y/Z 轴 1 米处")
    print("    预期：红球在右(+X)、绿球在前(+Y)、蓝球在上(+Z)")
    print("    若红/绿球出现在高度方向，说明坐标系被误转为 Y-up")

    def redraw_81():
        run(env, dd.draw_sphere([1.0, 0, 0], 0.15, RED, duration=IMMEDIATE_TTL))    # +X 右
        run(env, dd.draw_sphere([0.0, 1.0, 0], 0.15, GREEN, duration=IMMEDIATE_TTL))  # +Y 前
        run(env, dd.draw_sphere([0.0, 0, 1.0], 0.15, BLUE, duration=IMMEDIATE_TTL))   # +Z 上
    render_until_key(env, key_listener, redraw_81, "[8.1] 观察三球轴向，按空格键进入 8.2")

    # ---------- 8.2 单位验证：1 米参考立方体 ----------
    print("\n  [8.2] 单位验证：1×1×1 米立方体在原点")
    print("    预期：立方体边长 1 米，底面贴地（z=0..1），居中在 [0,0,0.5]")
    print("    size=[1,1,1] 对应三轴各 1 米（非半边长）")

    def redraw_82():
        run(env, dd.draw_box([0, 0, 0.5], [1.0, 1.0, 1.0], YELLOW, duration=IMMEDIATE_TTL))
    render_until_key(env, key_listener, redraw_82, "[8.2] 观察 1 米参考立方体，按空格键进入 8.3")

    # ---------- 8.3 方向性基元 +Y 模型空间：圆柱三轴方向 ----------
    print("\n  [8.3] 圆柱 +Y 模型空间验证：三圆柱分别沿 +X/+Y/+Z")
    print("    预期（position=底面中心，from=起点，to=终点）：")
    print("      红圆柱：from [0,0,1] → to [2,0,1]，沿 +X，底面在原点侧")
    print("      绿圆柱：from [0,0,1] → to [0,2,1]，沿 +Y，底面在原点侧")
    print("      蓝圆柱：from [0,0,1] → to [0,0,3]，沿 +Z，底面在 z=1")
    print("    若蓝圆柱水平而非竖直，说明 +Y→+Z 旋转方向错误")

    z0 = 1.0

    def redraw_83():
        run(env, dd.draw_cylinder([0, 0, z0], [2.0, 0, z0], 0.08, RED, duration=IMMEDIATE_TTL))    # +X
        run(env, dd.draw_cylinder([0, 0, z0], [0.0, 2.0, z0], 0.08, GREEN, duration=IMMEDIATE_TTL))  # +Y
        run(env, dd.draw_cylinder([0, 0, z0], [0.0, 0.0, 2.0 + z0], 0.08, BLUE, duration=IMMEDIATE_TTL))  # +Z
    render_until_key(env, key_listener, redraw_83, "[8.3] 观察圆柱三轴方向，按空格键进入 8.4")

    # ---------- 8.4 箭头方向 + 缩放验证 ----------
    print("\n  [8.4] 箭头方向与缩放验证")
    print("    预期：")
    print("      红箭头：原点→+Z(2米)，竖直向上，杆半径 0.05，长 2 米")
    print("      绿箭头：原点→+Y(1.5米)，水平向前，杆半径 0.10（更粗），长 1.5 米")
    print("      蓝箭头：原点→+X(1米)，水平向右，杆半径 0.15（最粗），长 1 米")
    print("    检查：箭尖方向 = to 方向；杆粗细 = shaft_radius（米）；长度 = |to-from|")

    def redraw_84():
        run(env, dd.draw_arrow([0, 0, 0], [0, 0, 2.0], 0.05, RED, duration=IMMEDIATE_TTL))    # +Z 竖直
        run(env, dd.draw_arrow([0, 0, 0], [0, 1.5, 0], 0.10, GREEN, duration=IMMEDIATE_TTL))  # +Y 前
        run(env, dd.draw_arrow([0, 0, 0], [1.0, 0, 0], 0.15, BLUE, duration=IMMEDIATE_TTL))   # +X 右
    render_until_key(env, key_listener, redraw_84, "[8.4] 观察箭头方向与缩放，按空格键进入 8.5")

    # ---------- 8.5 非均匀缩放验证 ----------
    print("\n  [8.5] 非均匀缩放验证：三个立方体不同 xyz 缩放")
    print("    预期：")
    print("      红盒 [2,0.5,0.5]：X 方向长 2 米，Y/Z 方向 0.5 米（长条形，沿 X）")
    print("      绿盒 [0.5,2,0.5]：Y 方向长 2 米，X/Z 方向 0.5 米（长条形，沿 Y）")
    print("      蓝盒 [0.5,0.5,2]：Z 方向长 2 米，X/Y 方向 0.5 米（长条形，沿 Z/竖直）")
    print("    若蓝盒非竖直长条，说明 Z 轴缩放被误映射到其他轴")

    def redraw_85():
        run(env, dd.draw_box([-1.5, 0, 1.0], [2.0, 0.5, 0.5], RED, duration=IMMEDIATE_TTL))    # 沿 X 长条
        run(env, dd.draw_box([0, 0, 1.0], [0.5, 2.0, 0.5], GREEN, duration=IMMEDIATE_TTL))     # 沿 Y 长条
        run(env, dd.draw_box([1.5, 0, 1.0], [0.5, 0.5, 2.0], BLUE, duration=IMMEDIATE_TTL))    # 沿 Z 长条（竖直）
    render_until_key(env, key_listener, redraw_85, "[8.5] 观察非均匀缩放，按空格键完成 Phase 8")

    print("\n  [Phase 8 完成] 请对照上述检查清单逐项确认视口中的几何方向/尺寸/缩放。")


# ============================================================
# Phase 9：Immediate TTL（duration>0）跨帧存活 + 到期自动消失
# 对照实现指南 §3.2.4 / 设计文档 §3.1。
# 三步：
#   9.1 对照：左球 duration=0（单帧，需每帧重提交），右球 duration=2.0（TTL，停止重提交后仍存活）
#   9.2 停止重提交：左球消失（单帧），右球仍存活（TTL 未到期）—— 验证 TTL 跨帧
#   9.3 等待到期：2 秒后右球自动消失 —— 验证 TTL 到期清理
# ============================================================
def phase9_ttl_duration(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 9] Immediate TTL（duration>0）跨帧存活 + 到期自动消失")
    print("  机制：duration=0 为单帧（每帧 Simulate 后清空）；duration>0 跨帧存活至仿真时间到期。")
    print("  本阶段用于防闪烁：低频提交的图形无需每帧重画。")

    ttl = 2.0  # TTL 时长（秒）

    # ---------- 9.1 对照：两球同时绘制（左单帧 / 右 TTL）----------
    print(f"\n  [9.1] 对照：左球 duration=0（单帧），右球 duration={ttl}s（TTL）")
    print("    预期：两球均显示（此阶段每帧都重提交，两者都可见）")

    def redraw_91():
        run(env, dd.draw_sphere([-1.0, 0, Z_ROW], 0.3, RED, duration=0.0))      # 单帧
        run(env, dd.draw_sphere([1.0, 0, Z_ROW], 0.3, CYAN, duration=ttl))      # TTL
    render_until_key(env, key_listener, redraw_91, "[9.1] 观察两球均显示，按空格键停止重提交")

    # ---------- 9.2 停止重提交：左球消失，右球存活 ----------
    print("\n  [9.2] 停止重提交（redraw=None），仅持续渲染")
    print("    预期：左球（单帧）立即消失；右球（TTL）仍持续显示（跨帧存活）")
    print("    若右球也消失，说明 duration_seconds 未被 C++ 侧正确写入 m_duration")
    render_until_key(env, key_listener, None, f"[9.2] 观察左消失/右存活，按空格键等待 TTL 到期（{ttl}s）")

    # ---------- 9.3 等待到期：右球自动消失 ----------
    print(f"\n  [9.3] 持续渲染等待 TTL 到期（约 {ttl} 秒）")
    print(f"    预期：{ttl} 秒后右球自动消失（C++ 侧按仿真时间到期清理）")
    print("    注意：到期基于仿真时间（time_step×step），非墙钟；若仿真暂停则不会到期")
    hold(env, ttl + 0.5, redraw=None)
    render_until_key(env, key_listener, None, "[9.3] 观察右球已消失，按空格键完成 Phase 9")


# ============================================================
# Phase 10：Retained keepalive 保活心跳
# 对照设计文档 §3.2 / 实现指南 §3.3。
# 三步：
#   10.1 过期销毁：创建 keepalive=1.0s 对象，不发送心跳，1s 后自动消失
#   10.2 心跳续期：创建 keepalive=1.0s 对象，每 0.5s 发送 keepalive_objects，持续存活
#   10.3 断连自清：停止心跳后，1s 内对象自动消失（模拟 Python 崩溃/gRPC 断连）
# ============================================================
def phase10_keepalive(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 10] Retained keepalive 保活心跳")
    print("  机制：create_objects(keepalive=N) 后，对象在 N 秒内未收到 update/keepalive 则自动销毁。")
    print("  用于断连/崩溃/句柄泄漏等异常场景的残留清理（设计文档 §11.9）。")

    keepalive = 1.0  # 保活时长（秒）

    # ---------- 10.1 过期销毁：不心跳，到期自清 ----------
    print(f"\n  [10.1] 过期销毁：创建 keepalive={keepalive}s 球，不发送心跳")
    print(f"    预期：{keepalive} 秒后球自动消失（C++ 侧保活计时器到期）")
    inst = _make_instance([-1.0, 0, Z_ROW], [0, 0, 0, 1], [0.3, 0.3, 0.3], RED)
    hs = run(env, dd.create_objects(DebugMeshType.SPHERE, [inst], keepalive=keepalive))
    handles = [h for h in hs if h["valid"]]
    print(f"    创建 {len(handles)} 个球，句柄={handles}")
    print(f"    持续渲染 {keepalive + 0.5}s（不发送 update/keepalive）...")
    hold(env, keepalive + 0.5, redraw=None)
    count = run(env, dd.query_count())
    print(f"    query_count={count}（retained 应为 0，对象已自动销毁）")
    render_until_key(env, key_listener, None, "[10.1] 观察红球已消失，按空格键进入 10.2")

    # ---------- 10.2 心跳续期：每 0.5s 发送 keepalive，持续存活 ----------
    print(f"\n  [10.2] 心跳续期：创建 keepalive={keepalive}s 球，每 0.5s 发送 keepalive_objects")
    print("    预期：球持续显示（心跳刷新保活计时器，不会过期）")
    print("    建议心跳间隔 = 1/2 保活时长（设计文档 §8）：keepalive=1.0s → 0.5s 心跳")
    inst2 = _make_instance([0.0, 0, Z_ROW], [0, 0, 0, 1], [0.3, 0.3, 0.3], GREEN)
    hs2 = run(env, dd.create_objects(DebugMeshType.SPHERE, [inst2], keepalive=keepalive))
    handles2 = [h for h in hs2 if h["valid"]]
    print(f"    创建 {len(handles2)} 个球，句柄={handles2}")

    heartbeat_interval = keepalive / 2  # 0.5s
    heartbeat_duration = 4.0  # 持续 4 秒（远超 keepalive，验证心跳确实在续期）
    print(f"    持续 {heartbeat_duration}s，每 {heartbeat_interval}s 心跳一次...")
    action = env.action_space.sample()
    end = time.time() + heartbeat_duration
    last_heartbeat = 0.0
    heartbeat_count = 0
    while time.time() < end:
        frame_start = time.time()
        if time.time() - last_heartbeat >= heartbeat_interval:
            result = run(env, dd.keepalive_objects(handles2))
            alive_n = len(result["alive"])
            expired_n = len(result["expired"])
            heartbeat_count += 1
            print(f"    心跳 #{heartbeat_count}: alive={alive_n}, expired={expired_n}")
            if expired_n > 0:
                print(f"    ⚠️ 出现过期句柄：{result['expired']}（不应发生）")
            last_heartbeat = time.time()
        env.step(action)
        env.render()
        elapsed = time.time() - frame_start
        if elapsed < REALTIME_STEP:
            time.sleep(REALTIME_STEP - elapsed)
    count2 = run(env, dd.query_count())
    print(f"    {heartbeat_duration}s 后 query_count={count2}（retained 应 >= 1，心跳续期成功）")
    render_until_key(env, key_listener, None, "[10.2] 观察绿球持续显示，按空格键进入 10.3")

    # ---------- 10.3 断连自清：停止心跳，1s 内自动消失 ----------
    print("\n  [10.3] 断连自清：停止心跳，模拟 Python 崩溃/gRPC 断连")
    print(f"    预期：{keepalive} 秒内绿球自动消失（保活计时器到期，C++ 侧自动清理）")
    print("    这正是 keepalive 机制的核心价值：调用方异常退出后无残留对象")
    print(f"    持续渲染 {keepalive + 0.5}s（不发送心跳）...")
    hold(env, keepalive + 0.5, redraw=None)
    count3 = run(env, dd.query_count())
    print(f"    query_count={count3}（retained 应为 0，断连自清成功）")
    render_until_key(env, key_listener, None, "[10.3] 观察绿球已消失，按空格键完成 Phase 10")


# ============================================================
# Phase 11：TTL=0 闪烁效果演示（为何 immediate 推荐 TTL>=0.1）
# 对照 force_render 注释 / 设计文档 §3.1。
# 机制：immediate（TTL=0）图形在 Studio 端每次 Simulate 后被清空。
#   env.render() 默认 30Hz 节流，而 step 以 50Hz (REALTIME_STEP=0.02s) 运行，
#   被节流跳过的帧 → 提交的图形未触发刷新即被下次 Simulate 清掉 → 闪烁。
#   TTL>=0.1 使图形跨帧存活，即使某帧未刷新，下一帧仍可见 → 稳定。
# 本阶段用 force=False（保留 30Hz 节流）对比 TTL=0（闪烁）与 TTL=0.1（稳定）。
# ============================================================
def phase11_flicker_demo(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 11] TTL=0 闪烁效果演示（为何 immediate 推荐 TTL>=0.1）")
    print("  机制：TTL=0 图形每次 Simulate 后清空；env.render() 默认 30Hz 节流，")
    print("        而 step 以 50Hz 运行 → 被节流跳过的帧图形未刷新即被清掉 → 闪烁。")
    print("        TTL>=0.1 跨帧存活，即使某帧未刷新仍可见 → 稳定。")
    print("  本阶段使用 force=False（保留 30Hz 节流），对比两球：")

    # ---------- 11.1 节流渲染下：左 TTL=0 闪烁 / 右 TTL=0.1 稳定 ----------
    print("\n  [11.1] 30Hz 节流渲染（force=False）：左球 TTL=0，右球 TTL=0.1")
    print("    预期：左球闪烁/时隐时现（部分帧未刷新即被清空）；右球稳定显示")
    print("    这正是其余阶段（Phase 1/2/3/4/8）使用 IMMEDIATE_TTL=0.1 的原因")

    def redraw_111():
        run(env, dd.draw_sphere([-1.0, 0, Z_ROW], 0.3, RED, duration=0.0))        # TTL=0 → 闪烁
        run(env, dd.draw_sphere([1.0, 0, Z_ROW], 0.3, CYAN, duration=IMMEDIATE_TTL))  # TTL=0.1 → 稳定
    render_until_key(env, key_listener, redraw_111,
                     "[11.1] 观察左闪/右稳，按空格键进入 11.2", force=False)

    # ---------- 11.2 对照：force_render（每帧刷新）下两球均稳定 ----------
    print("\n  [11.2] force_render（每帧刷新，绕过节流）：左球 TTL=0，右球 TTL=0.1")
    print("    预期：两球均稳定（每帧都刷新，TTL=0 也不再闪烁）")
    print("    结论：TTL>=0.1 与 force_render 都能消除闪烁；TTL 更稳健（不依赖渲染节流）")
    render_until_key(env, key_listener, redraw_111,
                     "[11.2] 观察两球均稳定，按空格键完成 Phase 11", force=True)


# ============================================================
# Phase 12：Wireframe 线框渲染（W1-W5）
# 对照 wireframe_custom_mesh_implementation_guide.md。
# 验证 W3/W4：wireframe=True 走 LineList + o_wireframe PSO + 四桶拆分。
# 三步：
#   12.1 6 种基元 solid vs wireframe 对照（左 solid / 右 wireframe）
#   12.2 wireframe + 透明度（WireTransparent 桶）
#   12.3 retained 线框对象（create_objects + flags=WIREFRAME）
# ============================================================
def phase12_wireframe(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 12] Wireframe 线框渲染（W1-W5）")
    print("  机制：wireframe=True 或 flags|=InstanceFlags.WIREFRAME 的实例被路由到 wire 桶，")
    print("        走 LineList 拓扑 + o_wireframe PSO（短路光照，直接输出顶点色×1.1）。")
    print("        W1 边去重 → W2 实例拆分 → W3 wire PSO → W4 四桶渲染。")

    up_quat = _direction_to_quat(np.array([0.0, 0.0, 1.0])).tolist()
    H_HALF = 0.3  # 方向性基元半高（scale.y=0.6），用于视觉居中

    # ---------- 12.1 6 种基元 solid vs wireframe 对照 ----------
    print("\n  [12.1] 6 种基元 solid vs wireframe 对照（左排 y=0 solid / 右排 y=2 wireframe）")
    print("    预期：左排实心着色（带光照），右排仅线框边（去重 LineList，无光照，略亮）")
    print("    验证 W1（三角形边去重）+ W3（o_wireframe PSO 短路光照/法线）")

    def redraw_121():
        # Sphere
        run(env, dd.draw_sphere([-2.5, 0, Z_ROW], 0.3, RED, duration=IMMEDIATE_TTL))
        run(env, dd.draw_sphere([-2.5, 2, Z_ROW], 0.3, RED, wireframe=True, duration=IMMEDIATE_TTL))
        # Cylinder（竖直，视觉居中）
        run(env, dd.draw_cylinder([-1.5, 0, Z_ROW - H_HALF], [-1.5, 0, Z_ROW + H_HALF],
                                  0.2, GREEN, duration=IMMEDIATE_TTL))
        run(env, dd.draw_cylinder([-1.5, 2, Z_ROW - H_HALF], [-1.5, 2, Z_ROW + H_HALF],
                                  0.2, GREEN, wireframe=True, duration=IMMEDIATE_TTL))
        # Cone（朝上）— 无 draw_cone 便捷方法，用 draw_batch + flags
        cone_solid = _make_instance([-0.5, 0, Z_ROW - H_HALF], up_quat, [0.25, 0.6, 0.25], BLUE)
        cone_wire = _make_instance([-0.5, 2, Z_ROW - H_HALF], up_quat,
                                   [0.25, 0.6, 0.25], BLUE, flags=InstanceFlags.WIREFRAME)
        run(env, dd.draw_batch(DebugMeshType.CONE, [cone_solid], duration=IMMEDIATE_TTL))
        run(env, dd.draw_batch(DebugMeshType.CONE, [cone_wire], duration=IMMEDIATE_TTL))
        # Box
        run(env, dd.draw_box([0.5, 0, Z_ROW], [0.4, 0.4, 0.4], YELLOW, duration=IMMEDIATE_TTL))
        run(env, dd.draw_box([0.5, 2, Z_ROW], [0.4, 0.4, 0.4], YELLOW, wireframe=True, duration=IMMEDIATE_TTL))
        # Quad
        run(env, dd.draw_quad([1.5, 0, Z_ROW], [0.5, 0.5, 1.0], MAGENTA, duration=IMMEDIATE_TTL))
        run(env, dd.draw_quad([1.5, 2, Z_ROW], [0.5, 0.5, 1.0], MAGENTA, wireframe=True, duration=IMMEDIATE_TTL))
        # Arrow（竖直）
        run(env, dd.draw_arrow([2.5, 0, Z_ROW - H_HALF], [2.5, 0, Z_ROW + H_HALF],
                               0.05, CYAN, duration=IMMEDIATE_TTL))
        run(env, dd.draw_arrow([2.5, 2, Z_ROW - H_HALF], [2.5, 2, Z_ROW + H_HALF],
                               0.05, CYAN, wireframe=True, duration=IMMEDIATE_TTL))

    render_until_key(env, key_listener, redraw_121,
                     "[12.1] 观察左 solid / 右 wireframe，按空格键进入 12.2")

    # ---------- 12.2 wireframe + 透明度（WireTransparent 桶）----------
    print("\n  [12.2] wireframe + 透明度（alpha=0.5）→ WireTransparent 桶")
    print("    预期：三个半透明线框球重叠，边缘可见内部结构（SrcAlpha 混合 + DepthWrite=Zero）")
    print("    验证 W4 四桶拆分：wire + transparent 走独立 PSO + 独立实例缓冲（无 ghosting）")

    def redraw_122():
        run(env, dd.draw_sphere([-0.3, -0.3, Z_ROW], 0.4, [1.0, 0.0, 0.0, 0.5], wireframe=True, duration=IMMEDIATE_TTL))
        run(env, dd.draw_sphere([0.3, -0.3, Z_ROW], 0.4, [0.0, 1.0, 0.0, 0.5], wireframe=True, duration=IMMEDIATE_TTL))
        run(env, dd.draw_sphere([0.0, 0.3, Z_ROW], 0.4, [0.0, 0.2, 1.0, 0.5], wireframe=True, duration=IMMEDIATE_TTL))

    render_until_key(env, key_listener, redraw_122,
                     "[12.2] 观察半透明线框混合，按空格键进入 12.3")

    # ---------- 12.3 retained 线框对象 ----------
    print("\n  [12.3] Retained 线框对象（create_objects + flags=WIREFRAME）")
    print("    预期：创建 6 个持久线框对象，跨帧持续显示（无需重提交）")
    print("    验证 W2 实例拆分（SplitByWireframe）对 retained 路径同样生效")

    z_cyl = Z_ROW - 0.3  # 方向性基元底面位置（视觉居中）
    wf = InstanceFlags.WIREFRAME
    specs = [
        (DebugMeshType.SPHERE,   _make_instance([-2.5, 0, Z_ROW], [0, 0, 0, 1], [0.3, 0.3, 0.3], RED, flags=wf)),
        (DebugMeshType.CYLINDER, _make_instance([-1.5, 0, z_cyl], up_quat, [0.2, 0.6, 0.2], GREEN, flags=wf)),
        (DebugMeshType.CONE,     _make_instance([-0.5, 0, z_cyl], up_quat, [0.25, 0.6, 0.25], BLUE, flags=wf)),
        (DebugMeshType.BOX,      _make_instance([0.5, 0, Z_ROW], [0, 0, 0, 1], [0.4, 0.4, 0.4], YELLOW, flags=wf)),
        (DebugMeshType.QUAD,     _make_instance([1.5, 0, Z_ROW], [0, 0, 0, 1], [0.5, 0.5, 1.0], MAGENTA, flags=wf)),
        (DebugMeshType.ARROW,    _make_instance([2.5, 0, z_cyl], up_quat, [0.05, 0.6, 0.05], CYAN, flags=wf)),
    ]
    handles: List[dict] = []
    for mtype, inst in specs:
        hs = run(env, dd.create_objects(mtype, [inst]))
        valid = [h for h in hs if h["valid"]]
        handles.extend(valid)
    count = run(env, dd.query_count())
    print(f"  创建 6 个线框对象，query_count={count}（retained 应 >= 6）")

    render_until_key(env, key_listener, None, "观察 6 个持久线框对象，按空格键销毁并完成 Phase 12")

    run(env, dd.destroy_objects(handles))
    count2 = run(env, dd.query_count())
    print(f"  销毁后 query_count={count2}")


def _make_tetrahedron_positions_indices():
    """构造一个正四面体的顶点+索引（程序化，不依赖 OBJ 文件）。

    顶点：4 个，边长 sqrt(2)，中心在原点。
        v0 = ( 1,  1,  1)
        v1 = (-1, -1,  1)
        v2 = (-1,  1, -1)
        v3 = ( 1, -1, -1)
    面：4 个三角形（CCW 绕序，法线朝外）：
        (0,1,2) (0,3,1) (0,2,3) (1,3,2)

    返回 (positions_flat, indices_flat)，positions 为 [x,y,z,...]，indices 为 [i0,i1,i2,...]。
    用于验证 register_custom_mesh（顶点+索引路径）。
    """
    positions = [
        1.0,  1.0,  1.0,
        -1.0, -1.0,  1.0,
        -1.0,  1.0, -1.0,
        1.0, -1.0, -1.0,
    ]
    # CCW 从外部观察；若 C++ CullMode=Back 剔除反向，可传 flip_winding
    indices = [
        0, 1, 2,
        0, 3, 1,
        0, 2, 3,
        1, 3, 2,
    ]
    return positions, indices


_MESHS_DIR = "/home/superfhwl/repo/OrcaGym/tests/orca_gym/utils/meshs"
_OBJ_CUBE = f"{_MESHS_DIR}/simple_mesh/cube.obj"
_OBJ_TORUS = f"{_MESHS_DIR}/simple_mesh/torus.obj"
_OBJ_BUNNY = f"{_MESHS_DIR}/bunny/bunny.obj"


def phase13_custom_mesh(env, key_listener: KeyListener) -> None:
    dd = env.debug_draw()
    print("\n[Phase 13] Custom Mesh（Phase C/O：用户提供的网格）")
    print("  机制：register_custom_mesh(顶点+索引) 或 register_custom_mesh_from_obj(OBJ 文本)")
    print("        注册返回句柄，draw_custom_mesh_batch 绘制实例，unregister_custom_mesh 释放。")
    print("        C++ 侧 GPU 资源延迟到下一帧 Simulate 创建；未就绪时绘制被静默丢弃。")

    # TTL=0 + 每帧 redraw：instance 单帧存活，下一帧 Swap 自动过期。
    # 之前用 0.5s 规避闪烁（无 TTL 支持），但这导致每帧累积 ~2 个 instance，
    # 0.5s 内堆积 50+ 个完全重叠的 instance → z-fighting → 视觉"转动"假象。
    # TTL 支持修复后，duration=0 即可稳定（每帧提交→Swap→渲染→过期）。
    CUSTOM_TTL = 0.1

    # ---------- 13.1 程序化四面体：solid + wireframe 对照 ----------
    print("\n  [13.1] 程序化四面体（顶点+索引注册）：左 solid / 右 wireframe")
    print("    预期：左侧正四面体实心着色，右侧线框（LineList 去重边）")
    print("    验证 RegisterCustomMesh 路径 + DrawCustomMeshBatch + wireframe flag")

    pos, idx = _make_tetrahedron_positions_indices()
    # 四面体边长 sqrt(2)≈1.414，缩放 0.25 使视觉尺寸 ~0.35
    tet_handle = run(env, dd.register_custom_mesh(pos, idx))
    if not tet_handle["valid"]:
        print(f"  [警告] 四面体注册失败：{tet_handle.get('error', '')}，跳过 13.1")
    else:
        print(f"  四面体注册成功：handle index={tet_handle['index']} "
              f"gen={tet_handle['generation']} verts={tet_handle['vertex_count']} "
              f"faces={tet_handle['face_count']}")

        tet_scale = [0.25, 0.25, 0.25]
        # 等待一帧让 GPU 资源就绪（注册后下一帧 Simulate 才创建 buffer）
        hold(env, 0.1)

        def redraw_131():
            run(env, dd.draw_custom_mesh(
                tet_handle, [-0.6, 0, Z_ROW], [0, 0, 0, 1], tet_scale, RED,
                duration=CUSTOM_TTL))
            run(env, dd.draw_custom_mesh(
                tet_handle, [0.6, 0, Z_ROW], [0, 0, 0, 1], tet_scale, CYAN,
                wireframe=True, duration=CUSTOM_TTL))

        render_until_key(env, key_listener, redraw_131,
                         "[13.1] 观察左 solid / 右 wireframe 四面体，按空格键进入 13.2")

    # ---------- 13.2 OBJ 加载：cube + torus + bunny ----------
    print("\n  [13.2] OBJ 文件加载（RegisterCustomMeshFromObjData）")
    print("    预期：cube（小）/ torus（环）/ bunny（斯坦福兔）三个网格并排显示")
    print("    验证 OBJ 解析 + 规范化（y_up→Z-up、recenter、normalize_scale）")
    print("    注：trimesh 导出的 obj 默认 Z-up，故 y_up=False；recenter+normalize 使尺寸一致")

    obj_specs = [
        (_OBJ_CUBE,  RED,     [-1.2, 0, Z_ROW]),
        (_OBJ_TORUS, GREEN,   [0.0, 0, Z_ROW]),
        (_OBJ_BUNNY, MAGENTA, [1.2, 0, Z_ROW]),
    ]
    obj_handles: List[dict] = []
    for path, color, pos3d in obj_specs:
        h = run(env, dd.register_custom_mesh_from_obj_file(
            path, y_up=False, recenter=True, normalize_scale=True))
        if not h["valid"]:
            print(f"  [警告] {path} 加载失败：{h.get('error', '')}")
            continue
        obj_handles.append(h)
        print(f"  {path.split('/')[-1]} 注册成功：index={h['index']} gen={h['generation']}")

    if obj_handles:
        # 等待 GPU 就绪
        hold(env, 0.1)
        # normalize_scale 后最长边=1.0，缩放 0.5 使视觉尺寸 ~0.5
        obj_scale = [0.5, 0.5, 0.5]
        colors = [RED, GREEN, MAGENTA]
        # 线框对照：同位置上方 0.7 画 wireframe 版本（实心/线框上下并排）
        WIRE_Y_OFFSET = 0.7

        def redraw_132():
            # 下排：solid
            for h, color, pos3d in zip(obj_handles, colors, [s[2] for s in obj_specs]):
                run(env, dd.draw_custom_mesh(
                    h, pos3d, [0, 0, 0, 1], obj_scale, color, duration=CUSTOM_TTL))
            # 上排：wireframe 对照（同 handle，wireframe=True）
            for h, color, pos3d in zip(obj_handles, colors, [s[2] for s in obj_specs]):
                wire_pos = [pos3d[0], pos3d[1] + WIRE_Y_OFFSET, pos3d[2]]
                run(env, dd.draw_custom_mesh(
                    h, wire_pos, [0, 0, 0, 1], obj_scale, color,
                    wireframe=True, duration=CUSTOM_TTL))

        render_until_key(env, key_listener, redraw_132,
                         "[13.2] 下排 solid / 上排 wireframe 对照（cube/torus/bunny），按空格键进入 13.3")

    # ---------- 13.3 多实例批量绘制 + 透明度 ----------
    print("\n  [13.3] 多实例批量绘制（DrawCustomMeshBatch）+ 透明度")
    print("    预期：3x3 网格的 torus 实例，部分半透明，验证批量提交 + 混合桶")
    if len(obj_handles) >= 2:
        torus_handle = obj_handles[1]  # torus
        # 等待 GPU 就绪（13.2 已等待，此处冗余保险）
        hold(env, 0.1)

        def redraw_133():
            # 下排：solid（半透明交替）
            instances_solid = []
            for ix in range(3):
                for iy in range(3):
                    x = (ix - 1) * 0.6
                    y = (iy - 1) * 0.6
                    alpha = 0.4 if (ix + iy) % 2 == 0 else 1.0
                    color = [0.2, 0.8, 1.0, alpha]
                    inst = _make_instance(
                        [x, y, Z_ROW], [0, 0, 0, 1], [0.2, 0.2, 0.2], color)
                    instances_solid.append(inst)
            run(env, dd.draw_custom_mesh_batch(
                torus_handle, instances_solid, duration=CUSTOM_TTL))
            # 上排：wireframe 对照（同一批量 API，wireframe flag）
            instances_wire = []
            for ix in range(3):
                for iy in range(3):
                    x = (ix - 1) * 0.6
                    y = (iy - 1) * 0.6 + 0.8
                    color = [0.2, 0.8, 1.0, 1.0]
                    inst = _make_instance(
                        [x, y, Z_ROW], [0, 0, 0, 1], [0.2, 0.2, 0.2], color,
                        flags=InstanceFlags.WIREFRAME)
                    instances_wire.append(inst)
            run(env, dd.draw_custom_mesh_batch(
                torus_handle, instances_wire, duration=CUSTOM_TTL))

        render_until_key(env, key_listener, redraw_133,
                         "[13.3] 下排 solid（半透明交替）/ 上排 wireframe 对照，按空格键进入 13.4")

    # ---------- 13.4 注销 + stale handle 安全验证 ----------
    print("\n  [13.4] 注销 + stale handle 安全验证")
    print("    预期：注销后绘制该 handle 不崩溃（静默丢弃），stale handle 安全")
    all_handles = [tet_handle] if tet_handle.get("valid") else []
    all_handles.extend(obj_handles)

    for h in all_handles:
        run(env, dd.unregister_custom_mesh(h))
    print(f"  已注销 {len(all_handles)} 个 custom mesh")

    # 等待一帧让注销生效（render 线程释放 GPU）
    hold(env, 0.2)

    # 尝试用 stale handle 绘制（应静默丢弃，不崩溃）
    if tet_handle.get("valid"):
        print("  尝试用已注销的四面体 handle 绘制（预期：静默丢弃，不崩溃）...")
        run(env, dd.draw_custom_mesh(
            tet_handle, [0, 0, Z_ROW], [0, 0, 0, 1], [0.3, 0.3, 0.3], YELLOW,
            duration=IMMEDIATE_TTL))
        hold(env, 0.3)
        print("  stale handle 绘制完成（未崩溃）")

    print("  Phase 13 完成")


# ============================================================
# 入口
# ============================================================
PHASES = {
    1: phase1_all_mesh_types,
    2: phase2_edge_highlight,
    3: phase3_transparency,
    4: phase4_clear,
    5: phase5_retained_create,
    6: phase6_retained_animate,
    7: phase7_retained_destroy,
    8: phase8_coordinate_system,
    9: phase9_ttl_duration,
    10: phase10_keepalive,
    11: phase11_flicker_demo,
    12: phase12_wireframe,
    13: phase13_custom_mesh,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="OrcaDebugMesh 路径 A 端到端视觉验收")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--phase", type=int, nargs="*", default=None,
                        help="只运行指定阶段（1-13），默认全部")
    args = parser.parse_args()

    env = make_env(args.addr)
    key_listener = KeyListener()
    key_listener.start()
    try:
        u = env.unwrapped
        dd = u.debug_draw()
        print(f"已连接 OrcaStudio: {args.addr}")
        print(f"frame_skip={u.frame_skip}, time_step={u._time_step}, dt={u.dt}")
        print(f"debug_draw.is_online = {dd.is_online}")
        if not dd.is_online:
            print("ERROR: debug_draw 离线，请确认 OrcaStudio 已启动并进入仿真运行状态")
            return

        print("\n交互说明：每个阶段绘制完成后持续渲染，按【空格键】进入下一步。")
        print("Phase 6 动画、Phase 9 TTL 到期、Phase 10 keepalive 过期/心跳部分基于时间自动推进；")
        print("Phase 11 演示 TTL=0 闪烁（节流渲染下）；其余步骤均按键切换。按 Ctrl+C 可随时中断。")

        selected = sorted(args.phase) if args.phase else sorted(PHASES)
        for p in selected:
            if p not in PHASES:
                print(f"  [跳过] 未知阶段 {p}")
                continue
            try:
                # 传 unwrapped env：phase 函数需访问 .debug_draw() / .loop
                PHASES[p](u, key_listener)
            except Exception as e:  # noqa: BLE001  (单阶段出错不影响后续阶段)
                print(f"  [阶段 {p} 出错] {type(e).__name__}: {e}")
            # 阶段间清空 immediate 队列，避免残留
            run(u, dd.clear())

        print("\n全部阶段完成。")
    except KeyboardInterrupt:
        print("\n已中断")
    finally:
        key_listener.stop()
        env.close()


if __name__ == "__main__":
    main()
