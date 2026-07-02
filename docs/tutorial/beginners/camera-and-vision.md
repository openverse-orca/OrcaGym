# 📷 相机与视觉 — 获取 RGB-D 图像

除了读取关节角度和 body 位姿，你还可以从仿真中获取**图像**——就像给机器人装上了眼睛。

OrcaGym 通过 WebSocket 从 OrcaStudio 获取实时渲染的相机画面。

---

## 快速开始：获取第一张图像

```python
"""
first_camera.py — 获取仿真相机的第一张图像
"""

import time
import numpy as np
from orca_sensor.rgbd_camera import CameraWrapper

# 1. 创建相机包装器
# name: 相机名称（任意）
# port: WebSocket 端口（在 OrcaStudio 中配置）
camera = CameraWrapper(name="front_camera", port=8765)

# 2. 启动相机流（后台线程自动接收和解码）
camera.start()
print("相机已启动，等待第一帧...")

# 3. 等待第一帧到达
while not camera.is_first_frame_received():
 time.sleep(0.1)

print(f"✅ 收到第一帧！")

# 4. 获取图像
image = camera.image # NumPy 数组，形状 (H, W, 3)，格式 BGR
print(f" 分辨率: {image.shape[1]}×{image.shape[0]}")
print(f" 数据类型: {image.dtype}")
print(f" 像素范围: [{image.min()}, {image.max()}]")
print(f" 帧序号: {camera.image_index}")

# 5. 保存图像（用 OpenCV）
import cv2
cv2.imwrite("first_frame.png", image)
print("✅ 图像已保存到 first_frame.png")

# 6. 停止相机
camera.stop()
```

---

## 在环境类中集成相机

将相机作为环境的一部分，每步获取图像：

```python
"""
vision_env.py — 带相机观测的环境
"""

import time
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_sensor.rgbd_camera import CameraWrapper


class VisionEnv(OrcaGymEulerEnv):
 """在观测中加入相机图像的环境"""

 def __init__(self, frame_skip, orcagym_addr, agent_names, time_step,
 camera_port: int = 8765, **kwargs):
 super().__init__(
 frame_skip=frame_skip,
 orcagym_addr=orcagym_addr,
 agent_names=agent_names,
 time_step=time_step,
 **kwargs,
 )

 # ── 设置相机 ──
 self._camera = CameraWrapper(name="agent_view", port=camera_port)
 self._camera.start()

 # 等待第一帧
 print("等待相机就绪...")
 while not self._camera.is_first_frame_received():
 time.sleep(0.1)
 print(f"✅ 相机就绪: {self._camera.image.shape}")

 # 动作空间
 self.action_space = spaces.Box(
 low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
 )
 obs_sample = self._get_obs()
 self.observation_space = spaces.Dict({
 key: spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype)
 if isinstance(v, np.ndarray) and v.dtype != np.uint8
 else spaces.Box(low=0, high=255, shape=v.shape, dtype=np.uint8)
 for key, v in obs_sample.items()
 })

 def _get_obs(self):
 """
 观测 = 本体感知（关节角度）+ 视觉（相机图像）

 本体感知 (proprioception): 机器人"感觉"到的自身状态
 视觉 (vision): 相机看到的画面
 """
 return {
 # 本体感知
 "joint_pos": self.data.qpos.copy().astype(np.float32),
 "joint_vel": self.data.qvel.copy().astype(np.float32),

 # 视觉
 "image": self._camera.image.copy(), # (H, W, 3) uint8
 }

 def step(self, action):
 action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
 self.do_simulation(action, self.frame_skip)

 obs = self._get_obs()
 reward = 0.0
 terminated = False
 truncated = False

 return obs, reward, terminated, truncated, {}

 def reset_model(self):
 self.set_joint_qpos(self.init_qpos)
 self.set_joint_qvel(self.init_qvel)
 self.mj_forward()
 self._sync_view()
 return self._get_obs(), {}

 def close(self):
 self._camera.stop()
 super().close()
```

---

## 显示相机画面

用 Matplotlib 实时显示：

```python
import matplotlib.pyplot as plt

def show_camera_live(camera: CameraWrapper, duration: float = 10.0):
 """
 实时显示相机画面（duration 秒）。

 注意：这只是一个简单的显示示例。
 实际 RL 训练中不需要这样逐帧显示——直接用 image 数组即可。
 """
 plt.ion() # 交互模式
 fig, ax = plt.subplots()
 img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
 ax.set_title("Camera Feed")
 ax.axis('off')

 start = time.time()
 while time.time() - start < duration:
 frame = camera.image.copy()
 # OpenCV 的 BGR → Matplotlib 的 RGB
 frame_rgb = frame[..., ::-1]
 img_display.set_data(frame_rgb)
 fig.canvas.flush_events()
 plt.pause(0.03) # ~30 FPS

 plt.ioff()
 plt.close()
```

---

## 多相机设置

需要多个视角？创建多个 `CameraWrapper`：

```python
def setup_multi_camera():
 """同时启动多个相机"""

 cameras = {
 "front": CameraWrapper("front", port=8765),
 "side": CameraWrapper("side", port=8766),
 "top": CameraWrapper("top", port=8767),
 }

 # 全部启动
 for cam in cameras.values():
 cam.start()

 # 等待所有相机就绪
 for name, cam in cameras.items():
 while not cam.is_first_frame_received():
 time.sleep(0.1)
 print(f"✅ {name}: {cam.image.shape}")

 # 同步获取所有画面
 def get_all_views():
 return {name: cam.image.copy() for name, cam in cameras.items()}

 return cameras, get_all_views


# 用法
cameras, get_views = setup_multi_camera()
views = get_views()
print(f"可用视角: {list(views.keys())}")
```

---

## 相机参数配置

在 OrcaStudio 中可以配置每个相机的参数：

| 参数 | 说明 | 典型值 |
|------|------|--------|
| 分辨率 | 图像宽×高 | 640×480, 1280×720 |
| 帧率 | 每秒帧数 | 15, 30, 60 |
| RGB | 是否输出彩色图 | `True` |
| Depth | 是否输出深度图 | `True`（需要时） |

在 Python 侧通过 `CameraSensorInfo` 配置：

```python
from orca_scene.orca_gym_scene import CameraSensorInfo

# 配置某个相机的传感器参数
camera_config = CameraSensorInfo(
 capture_rgb=True, # 输出 RGB 图像
 capture_depth=True, # 输出深度图
 save_mp4_file=False, # 不保存到文件
 use_dds=False, # 不使用 DDS 压缩
)
scene.set_camera_sensor_info("camera_actor_name", camera_config)
```

---

## 性能建议

1. **不要在主循环中 `imshow`** — 图像显示很吃 CPU。训练时直接处理数组即可
2. **缩小图像** — 如果不需要全分辨率，在 `_get_obs()` 中 resize
3. **降低帧率** — 30 FPS 通常足够，更高的帧率浪费带宽
4. **异步渲染** — 相机在后台线程解码，不阻塞主仿真线程

```python
def _get_obs(self):
 image = self._camera.image.copy()
 # 缩小到 128×128 以减少计算量
 small_image = cv2.resize(image, (128, 128))
 return {..., "image": small_image}
```

---

## 下一步

现在你既能"感觉"（读状态），也能"看"（相机）。接下来学习如何**精确地控制机器人**：[🎮 简单控制器](simple-controller.md)。
