import pygame
import sys

class XboxJoystick:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Initialize Joystick
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise Exception("No joystick detected")
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("Joystick detected:", self.joystick.get_name())

        # Check the operating system
        if sys.platform.startswith('win'):
            self.platform = 'windows'
        elif sys.platform.startswith('linux'):
            self.platform = 'linux'
        else:
            raise Exception("Unsupported platform")

        # Initialize joystick state
        self.joystick_state = {
            "buttons": {
                "A": 0, "B": 0, "X": 0, "Y": 0,
                "LB": 0, "RB": 0, "Back": 0, "Start": 0,
                "LeftStick": 0, "RightStick": 0
            },
            "axes": {
                "LeftStickX": 0.0, "LeftStickY": 0.0,
                "RightStickX": 0.0, "RightStickY": 0.0,
                "LT": -1.0, "RT": -1.0
            },
            "hats": {
                "DPad": (0, 0)
            }
        }

        self.button_map = {
            "windows":{
                0: "A", 1: "B", 2: "X", 3: "Y",
                4: "LB", 5: "RB", 6: "Back", 7: "Start",
                8: "LeftStick", 9: "RightStick"},
            "linux":{            
                0: "A", 1: "B", 2: "X", 3: "Y",
                4: "LB", 5: "RB", 6: "Back", 7: "Start",
                8: "LeftStick", 9: "RightStick"}
        }

        self.axis_map = {
            "windows": {
                0: "LeftStickX", 1: "LeftStickY",
                2: "RightStickX", 3: "RightStickY",
                4: "LT", 5: "RT"},
            "linux": {
                0: "LeftStickX", 1: "LeftStickY",
                3: "RightStickX", 4: "RightStickY",
                2: "LT", 5: "RT"}
        }
        
    def update(self):
        for event in pygame.event.get():
            # print("event: ", event)
            if event.type == pygame.JOYBUTTONDOWN:
                self.joystick_state["buttons"][self.button_map[self.platform][event.button]] = 1
            if event.type == pygame.JOYBUTTONUP:
                self.joystick_state["buttons"][self.button_map[self.platform][event.button]] = 0
            if event.type == pygame.JOYAXISMOTION:
                if event.axis in self.axis_map[self.platform]:
                    self.joystick_state["axes"][self.axis_map[self.platform][event.axis]] = event.value
            if event.type == pygame.JOYHATMOTION:
                self.joystick_state["hats"]["DPad"] = event.value

    def get_state(self):
        return self.joystick_state.copy()

    def close(self):
        pygame.quit()
        print("Joystick closed")
