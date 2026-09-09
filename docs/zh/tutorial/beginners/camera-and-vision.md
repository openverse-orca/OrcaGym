# 📷 相机与视觉 — 获取 RGB-D 图像

除了读取关节角度和 body 位姿，你还可以从仿真中获取**图像**——就像给机器人装上了眼睛。

OrcaGym 通过 WebSocket 从 OrcaStudio 获取实时渲染的相机画面，支持按物理步索引对齐帧、区间保存为 MP4、实时预览。

---

## 快速开始：获取第一张图像

```python
"""
first_camera.py — 通过环境层获取相机画面
"""

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class CameraEnv(OrcaGymEulerEnv):
    """最简相机环境"""

    def __init__(self, model_xml_path, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            sync_render=kwargs.pop("sync_render", True),
            render_fps=kwargs.pop("render_fps", 30),
            **kwargs,
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        return self._get_obs(), 0.0, False, False, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.data.qpos.copy()


if __name__ == "__main__":
    env = CameraEnv(
        model_xml_path="tests/orca_gym/environment/euler/fixtures/simple_pendulum.xml",
        orcagym_addr="localhost:50051",
        skip_grpc_load=False,  # 在线模式，需连接 OrcaStudio
        render_mode="human",
    )
    env.reset()

    # 1. 枚举所有已注册相机名称
    camera_names = env.get_camera_names()
    print(f"可用相机: {camera_names}")
    assert camera_names, "未找到相机，请先在 OrcaStudio 场景中添加 Camera Entity"
    camera_name = camera_names[0]

    # 2. 启动串流（一键配置 + 推流）
    env.start_streaming(
        camera_name,
        capture_rgb=True,       # 启用 RGB 流
        color_port=7070,        # RGB 流 WebSocket 端口
        width=640,
        height=480,
    )
    print(f"✅ 相机 '{camera_name}' 串流已启动")

    # 3. 主循环：render(simulate_index=...) 驱动渲染
    #    simulate_index 用于按区间提取帧，必须递增
    for step_idx in range(100):
        action = np.zeros(env.model.nu, dtype=np.float32)
        env.step(action)
        env.render(simulate_index=step_idx)

    # 4. 实时预览（非阻塞）
    env.show_camera(camera_name, camera_type="color")

    # 5. 保存区间视频为 MP4（非阻塞，返回 Future）
    future = env.save_streaming(
        camera_name=camera_name,
        camera_type="color",
        file_path="/tmp/pendulum.mp4",
        start_simulate_index=0,
        end_simulate_index=99,
    )
    result = future.result()  # 等待保存完成
    print(f"✅ 已保存: {result.file_path} ({result.frame_count} 帧)")

    env.close()
```

> ⚠️ **本节示例需在线模式运行**（`skip_grpc_load=False`），需连接 OrcaStudio。
> 离线模式下相机接口返回空列表 / no-op。

---

## 核心 API 详解

### 1. 枚举相机名称

```python
camera_names = env.get_camera_names()
# → ["Camera_Entity_[uuid1]", "Camera_Entity_[uuid2]", ...]
```

返回所有在 OrcaStudio 场景中注册的相机名称。离线模式返回空列表。

### 2. 启动串流（`start_streaming`）⭐ 推荐

```python
env.start_streaming(
    camera_name,
    capture_rgb=True,           # 启用 RGB 流
    capture_depth=True,         # 启用深度流
    color_port=7070,            # RGB 流 WebSocket 端口
    depth_port=7071,            # 深度流 WebSocket 端口
    width=1280,                 # 图像宽度
    height=720,                 # 图像高度
    vertical_fov=60.0,          # 垂直视场角（度）
    near_clip=0.01,             # 近裁剪面
    far_clip=100.0,             # 远裁剪面
    gamma=1.0,                  # 深度相机 gamma 校正
    use_nvenc=True,             # 使用 NvEnc 硬件编码
    nvenc_gpu_index=0,          # NvEnc GPU 索引
)
```

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `capture_rgb` | 是否输出 RGB 彩色图 | `True` |
| `capture_depth` | 是否输出深度图 | `True`（需要时） |
| `capture_normal` | 是否输出法线图 | `False` |
| `capture_object_color` | 是否输出实例分割色标图 | `False` |
| `width` / `height` | 图像分辨率 | 640×480, 1280×720 |
| `vertical_fov` | 垂直视场角（度） | 60.0 |
| `near_clip` / `far_clip` | 裁剪面距离 | 0.01 / 100.0 |
| `color_port` / `depth_port` | WebSocket 端口 | 7070 / 7071 |
| `use_nvenc` | NvEnc 硬件编码 | `True`（有 NVIDIA GPU 时） |

> 💡 该方法内部自动处理相机属性同步和推流状态切换，上层无需关心底层状态机。

### 3. 驱动渲染（`render`）

```python
env.render(simulate_index=step_idx, request_idr=False)
```

- **`simulate_index`**：物理仿真步索引，透传到引擎相机管线用于帧对齐。
  保存区间视频时按此索引提取帧，**必须递增**。
- **`request_idr`**：是否请求引擎输出 IDR 关键帧。
  保存段起点可设 `True`，保证输出视频起点可正常播放。

渲染频率由 `set_render_fps(fps)` 控制：
- `sync_render=True`：按物理步节流（每 N 个物理步渲染一帧）
- `sync_render=False`：按墙钟时间节流（每 `1/fps` 秒渲染一帧）

### 4. 保存区间视频（`save_streaming`）

```python
future = env.save_streaming(
    camera_name="Camera_Entity_[uuid]",
    camera_type="color",          # "color" 或 "depth"
    file_path="/tmp/output.mp4",
    start_simulate_index=50,       # 区间起始（含）
    end_simulate_index=200,       # 区间结束（含）
)
result = future.result()          # 非阻塞，等结果
print(result.file_path, result.frame_count)
```

- **非阻塞**：立即返回 `Future`，保存完成后 `future.result()` 拿结果。
- 需先调用 `start_streaming` 启动推流。

### 5. 实时预览（`show_camera`）

```python
env.show_camera(camera_name, camera_type="color", window_name="Front View")
```

- **非阻塞**：在独立窗口中显示画面，不影响仿真主线程。
- 深度流直接显示原始灰度帧，不做伪彩色转换。
- 需先调用 `start_streaming` 启动推流。

---

## 在环境类中集成相机

```python
"""
vision_env.py — 带相机观测的环境
"""

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class VisionEnv(OrcaGymEulerEnv):
    """在观测中加入相机图像的环境"""

    def __init__(self, model_xml_path, camera_name, **kwargs):
        super().__init__(
            frame_skip=kwargs.pop("frame_skip", 5),
            orcagym_addr=kwargs.pop("orcagym_addr", "localhost:50051"),
            agent_names=kwargs.pop("agent_names", ["agent0"]),
            time_step=kwargs.pop("time_step", 0.002),
            model_xml_path=model_xml_path,
            sync_render=True,
            render_fps=30,
            render_mode="human",
            skip_grpc_load=kwargs.pop("skip_grpc_load", False),
            **kwargs,
        )

        # ── 相机配置 ──
        self._camera_name = camera_name
        self.start_streaming(
            camera_name,
            capture_rgb=True,
            color_port=7070,
            width=640,
            height=480,
        )
        self._step_idx = 0

        # ── 动作空间 ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # ── 观测空间 ──
        obs_sample = self._get_obs()
        self.observation_space = spaces.Dict({
            key: spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype)
            if isinstance(v, np.ndarray) and v.dtype != np.uint8
            else spaces.Box(low=0, high=255, shape=v.shape, dtype=np.uint8)
            for key, v in obs_sample.items()
        })

    def _get_obs(self):
        """观测 = 本体感知 + 视觉"""
        # 获取最新已解码帧（np.ndarray (H, W, 3) uint8 BGR，或 None）
        frame = self.get_recorder_manager().get_last_decoded_frame(
            self._camera_name, "color"
        )
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return {
            "joint_pos": self.data.qpos.copy().astype(np.float32),
            "joint_vel": self.data.qvel.copy().astype(np.float32),
            "image": frame.copy(),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        self.do_simulation(action, self.frame_skip)

        # 驱动渲染 + 帧对齐
        self._step_idx += 1
        self.render(simulate_index=self._step_idx)

        obs = self._get_obs()
        return obs, 0.0, False, False, {}

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos)
        self.set_joint_qvel(self.init_qvel)
        self.mj_forward()
        self._sync_view()
        self._step_idx = 0
        return self._get_obs(), {}
```

---

## 多相机设置

```python
# 1. 枚举所有相机
camera_names = env.get_camera_names()

# 2. 逐个启动串流（用不同端口避免冲突）
env.start_streaming(camera_names[0], capture_rgb=True, color_port=7070)
env.start_streaming(camera_names[1], capture_rgb=True, color_port=7071)
env.start_streaming(camera_names[2], capture_rgb=True, color_port=7072)

# 3. 主循环中统一 render（一次 render 触发所有相机的渲染）
for step_idx in range(100):
    env.step(action)
    env.render(simulate_index=step_idx)

# 4. 分别预览
env.show_camera(camera_names[0], "color", window_name="Front")
env.show_camera(camera_names[2], "color", window_name="Top")
```

> 💡 一次 `render(simulate_index=...)` 触发引擎渲染所有已启用的相机，
> 无需为每个相机单独调用 render。

---

## 深度相机

```python
# 启动深度流
env.start_streaming(
    camera_name,
    capture_rgb=False,
    capture_depth=True,
    depth_port=7071,
    near_clip=0.01,    # 深度相机近裁剪面
    far_clip=100.0,    # 深度相机远裁剪面
    gamma=1.0,         # 深度相机 gamma 校正
)

# 预览（灰度显示，不做伪彩色转换）
env.show_camera(camera_name, camera_type="depth")

# 保存深度视频
future = env.save_streaming(
    camera_name, "depth", "/tmp/depth.mp4",
    start_simulate_index=0, end_simulate_index=99,
)
```

> ⚠️ 深度流与彩色流是独立的码流，需分别用 `camera_type="color"` / `"depth"` 操作。
> 同一相机可同时启用 RGB + Depth（`capture_rgb=True, capture_depth=True`）。

---

## 性能建议

1. **不要在主循环中 `imshow`** — 图像显示很吃 CPU。
   用 `show_camera()` 让它在独立窗口显示，或训练时完全不显示。
2. **降低帧率** — 训练时 15~30 FPS 通常足够，更高的帧率浪费带宽。
3. **NvEnc 硬件编码** — 有 NVIDIA GPU 时设 `use_nvenc=True` 可大幅降低编码 CPU 开销。
4. **缩小图像** — 若 RL 策略不需要全分辨率，在 `_get_obs()` 中 resize：
   ```python
   import cv2
   small = cv2.resize(frame, (128, 128))
   ```
5. **区间录制而非全程录制** — 用 `save_streaming(start, end)` 只保存关键区间，
   内存占用恒定。

---

## 下一步

现在你既能"感觉"（读状态），也能"看"（相机）。接下来学习如何**精确地控制机器人**：[🎮 简单控制器](simple-controller.md)。
