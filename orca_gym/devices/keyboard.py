import pygame
import sys
import os

class KeyboardInput:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Initialize the display (even if not used, it's required for capturing events)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        pygame.display.set_mode((100, 100))
        

        # Initialize keyboard state
        self.keyboard_state = {
            "W": 0, "A": 0, "S": 0, "D": 0,
            "Space": 0, "Shift": 0, "Ctrl": 0, "Alt": 0,
            "Esc": 0, "Enter": 0, "Up": 0, "Down": 0,
            "Left": 0, "Right": 0
        }

        # Define the key mapping (you can customize this)
        self.key_map = {
            pygame.K_a: "A", pygame.K_b: "B", pygame.K_c: "C", pygame.K_d: "D",
            pygame.K_w: "W", pygame.K_s: "S", pygame.K_x: "X", pygame.K_y: "Y",
            pygame.K_UP: "Up", pygame.K_DOWN: "Down", pygame.K_LEFT: "Left", pygame.K_RIGHT: "Right",
            pygame.K_SPACE: "Space", pygame.K_RETURN: "Enter", pygame.K_ESCAPE: "Escape"
        }


    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in self.key_map:
                    self.keyboard_state[self.key_map[event.key]] = 1
                    # print(f"Key pressed: {self.key_map[event.key]}")
            elif event.type == pygame.KEYUP:
                if event.key in self.key_map:
                    self.keyboard_state[self.key_map[event.key]] = 0
                    # print(f"Key released: {self.key_map[event.key]}")

    def get_state(self):
        return self.keyboard_state.copy()

    def close(self):
        pygame.quit()
        print("Keyboard controller closed")