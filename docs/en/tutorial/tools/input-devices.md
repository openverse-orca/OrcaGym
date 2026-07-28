# 🎮 Input Devices

OrcaGym supports a variety of input devices for teleoperation.

## Supported Devices

| Device | Module | Installation |
|--------|--------|--------------|
| Xbox Controller | `devices/xbox_joystick.py` | `pip install orca-gym[devices]` |
| Pico VR Controller | `devices/pico_joytsick.py` | `pip install orca-gym[devices]` |
| Keyboard | `devices/keyboard.py` | Core dependency |
| Hand Tracking | `devices/hand_joytstick.py` | `pip install orca-gym[devices]` |

## Using Controllers

```python
# Requires the devices dependency
# pip install orca-gym[devices]

from orca_devices import xbox_joystick
```

## Teleoperation Data Flow

```
Input Device → devices/*.py → ctrl values → env.step()
 ↓
 override_ctrls (via render)
```
