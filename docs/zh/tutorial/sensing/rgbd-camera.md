# 📹 RGB-D 相机

OrcaGym 通过 WebSocket 提供了实时的 RGB-D 相机流。

## CameraWrapper

```python
from orca_sensor.rgbd_camera import CameraWrapper

# 创建相机包装器
camera = CameraWrapper(
 name="front_camera",
 port=8765 # 相机 WebSocket 端口
)

# 启动相机流（在后台线程）
camera.start()

# 等待第一帧
while not camera.is_first_frame_received():
 time.sleep(0.1)

# 获取当前图像
image = camera.image # (H, W, 3) BGR numpy 数组
print(f"图像形状: {image.shape}")
print(f"帧索引: {camera.image_index}")
```

## 多相机设置

```python
# 创建多个相机
cameras = {
 "front": CameraWrapper("front", port=8765),
 "side": CameraWrapper("side", port=8766),
 "top": CameraWrapper("top", port=8767),
}

# 全部启动
for cam in cameras.values():
 cam.start()

# 等待所有相机就绪
for cam in cameras.values():
 while not cam.is_first_frame_received():
 time.sleep(0.1)

# 同步获取所有相机画面
def get_all_frames(cameras):
 return {name: cam.image.copy() for name, cam in cameras.items()}
```

## 相机数据流架构

```
OrcaStudio/OrcaLab
 ├── 渲染引擎
 │ └── 相机帧 → H.264 编码 → WebSocket 流
 │
Python 客户端
 └── CameraWrapper
 ├── WebSocket 连接 (ws://localhost:<port>)
 ├── H.264 解码 (PyAV)
 └── NumPy BGR 数组
```

## 相机位姿

```python
# 获取相机帧和位姿信息
camera_transforms = env.get_frame_png("path/to/save")
# → {"front_camera": {"pos": [x,y,z], "quat": [w,x,y,z]}, ...}

for camera_name, transform in camera_transforms.items():
 print(f"{camera_name}:")
 print(f" 位置: {transform['pos']}")
 print(f" 姿态: {transform['quat']}")
```

## 摄像机监视器

OrcaGym 包含一个相机监视器脚本：

```bash
# 启动相机监视器
python -m orca_scripts.camera_monitor
```

## 相机时间戳

```python
# 获取相机的时间戳信息
last_frame = 0
timestamps = env.get_camera_time_stamp(last_frame)
# → {"camera_name": [ts1, ts2, ts3, ...], ...}

# 获取当前帧索引
current_frame = env.get_current_frame()

# 等待新帧
next_frame = env.get_next_frame()
```

## 在 RL 训练中使用相机

```python
class VisionEnv(OrcaGymEulerEnv):
 def __init__(self, ...):
 super().__init__(...)
 
 # 设置相机
 self.camera = CameraWrapper("agent_view", port=8765)
 self.camera.start()
 
 # 观测空间包含图像
 self.observation_space = spaces.Dict({
 "proprio": spaces.Box(-np.inf, np.inf, shape=(proprio_dim,)),
 "image": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
 })
 
 def _get_obs(self):
 return {
 "proprio": np.concatenate([
 self.data.qpos.copy(),
 self.data.qvel.copy(),
 ]).astype(np.float32),
 "image": self.camera.image.copy(),
 }
```

## 性能建议

1. **异步模式** (CaptureMode.ASYNC) 对视觉 RL 更友好
2. **缩小图像尺寸** 以提高帧率
3. **适当降低帧率** — 30 FPS 通常足够
4. **在多线程中解码** — 避免阻塞主仿真线程
