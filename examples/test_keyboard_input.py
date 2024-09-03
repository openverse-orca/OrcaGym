import os
import sys
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(current_file_path)

if project_root not in sys.path:
    sys.path.append(project_root)

from orca_gym.devices.keyboard import KeyboardInput

def test_keyboard_controller():
    try:
        controller = KeyboardInput()
        while True:
            controller.update()            
            state = controller.get_state()
            print("Keys:", state)
            if state["Esc"]:
                raise KeyboardInterrupt("Esc key pressed")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        controller.close()
        print("Controller closed")

if __name__ == '__main__':
    test_keyboard_controller()