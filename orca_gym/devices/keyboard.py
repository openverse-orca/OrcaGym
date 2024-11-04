import pygame
import sys
import os
import socket
import threading


class KeyboardInput:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Initialize the display (even if not used, it's required for capturing events)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        pygame.display.set_mode((400, 400))

        # Initialize keyboard state
        self.keyboard_state = {
            "W": 0, "A": 0, "S": 0, "D": 0,
            "Space": 0, "LShift": 0, "RShift": 0, "Ctrl": 0, "Alt": 0,
            "Esc": 0, "Enter": 0, "Up": 0, "Down": 0,
            "Left": 0, "Right": 0, "Q": 0, "E": 0
        }

        # Define the key mapping
        self.key_map = {
            pygame.K_a: "A", pygame.K_b: "B", pygame.K_c: "C", pygame.K_d: "D",
            pygame.K_w: "W", pygame.K_s: "S", pygame.K_x: "X", pygame.K_y: "Y",
            pygame.K_q: "Q", pygame.K_e: "E",
            pygame.K_UP: "Up", pygame.K_DOWN: "Down", pygame.K_LEFT: "Left", pygame.K_RIGHT: "Right",
            pygame.K_SPACE: "Space", pygame.K_RETURN: "Enter", pygame.K_ESCAPE: "Esc",
            pygame.K_LSHIFT: "LShift", pygame.K_RSHIFT: "RShift",
            pygame.K_LCTRL: "Ctrl", pygame.K_RCTRL: "Ctrl",
            pygame.K_LALT: "Alt", pygame.K_RALT: "Alt"
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

    def capture_keyboard_pos_ctrl(self) -> dict:
        # Capture positional control based on keyboard input
        state = self.get_state()
        move_x = state["D"] - state["A"]
        move_y = state["W"] - state["S"]
        move_z = state["Space"] - state["Ctrl"]
        pos_ctrl = {'x': move_x, 'y': move_y, 'z': move_z}
        return pos_ctrl

    def capture_keyboard_rot_ctrl(self) -> dict:
        # Capture rotational control based on keyboard input
        state = self.get_state()
        yaw = state["Right"] - state["Left"]
        pitch = state["Up"] - state["Down"]
        roll = state["E"] - state["Q"]
        rot_ctrl = {'yaw': yaw, 'pitch': pitch, 'roll': roll}
        return rot_ctrl

    def close(self):
        pygame.quit()
        print("Keyboard controller closed")



class KeyboardServer:
    def __init__(self, host='localhost', port=55000):
        self.clients = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.server_socket.bind((host, port))
        except OSError as e:
            print(f"Port {port} is already in use!")
            raise e
        self.server_socket.listen(5)

        pygame.init()
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"        
        self.screen = pygame.display.set_mode((100, 100))
        threading.Thread(target=self.accept_clients).start()

    def accept_clients(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            self.clients.append(client_socket)
            print(f"Client connected: {addr}")

    def broadcast(self, message):
        for client in self.clients:
            try:
                client.sendall(message.encode())
            except:
                self.clients.remove(client)

    def run(self):
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        self.broadcast(f'KEYDOWN:{event.key}')
                    elif event.type == pygame.KEYUP:
                        self.broadcast(f'KEYUP:{event.key}')
                    elif event.type == pygame.QUIT:
                        running = False
        finally:    
            self.close()

    def close(self):
        self.server_socket.close()
        pygame.quit()
        print("Keyboard server closed")




class KeyboardClient:
    def __init__(self, host='localhost', port=55000):

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

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        threading.Thread(target=self.receive).start()

    def receive(self):
        try:
            while True:
                data = self.client_socket.recv(1024).decode()
                if data:
                    event_type, key = data.split(':')
                    if event_type == 'KEYDOWN':
                        key_code = int(key)
                        # print("Key pressed:", key_code)
                        if key_code in self.key_map:
                            self.keyboard_state[self.key_map[key_code]] = 1
                    elif event_type == 'KEYUP':
                        key_code = int(key)
                        # print("Key released:", key_code)
                        if key_code in self.key_map:
                            self.keyboard_state[self.key_map[key_code]] = 0
        finally:
            self.close()

    def update(self):
        pass

    def get_state(self):
        return self.keyboard_state.copy()
    
    def close(self):
        self.client_socket.close()
        print("Keyboard client closed")

