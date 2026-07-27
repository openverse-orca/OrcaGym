# 🎮 输入设备

OrcaGym 支持多种输入设备用于遥操作。

## 支持的设备

| 设备 | 模块 | 安装 |
|------|------|------|
| Xbox 手柄 | `devices/xbox_joystick.py` | `pip install orca-gym[devices]` |
| Pico VR 手柄 | `devices/pico_joytsick.py` | `pip install orca-gym[devices]` |
| 键盘 | `devices/keyboard.py` | 核心依赖 |
| 手势追踪 | `devices/hand_joytstick.py` | `pip install orca-gym[devices]` |

## 使用手柄

```python
# 需要安装 devices 依赖
# pip install orca-gym[devices]

from orca_devices import xbox_joystick
```

## 遥操作数据流

```
输入设备 → devices/*.py → ctrl 值 → env.step()
 ↓
 override_ctrls (通过 render)
```
