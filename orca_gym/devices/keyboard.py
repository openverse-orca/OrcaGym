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


import socket
import threading
import pygame
import os

class KeyboardServer:
    def __init__(self, host='localhost', port=55000):
        self.clients = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置 SO_REUSEADDR 选项
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
            try:
                client_socket, addr = self.server_socket.accept()
                self.clients.append(client_socket)
                print(f"Client connected: {addr}")
            except Exception as e:
                print(f"Error accepting clients: {e}")
                break

    def broadcast(self, message):
        for client in self.clients[:]:
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
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.close()

    def close(self):
        # 关闭所有客户端套接字
        for client in self.clients:
            client.close()
        self.server_socket.close()
        pygame.quit()
        print("Keyboard server closed")



import socket
import threading
import pygame
import os

class KeyboardClient:
    def __init__(self, host='localhost', port=55000):
        # 初始化键盘状态
        self.keyboard_state = {
            "W": 0, "A": 0, "S": 0, "D": 0,
            "Space": 0, "Shift": 0, "Ctrl": 0, "Alt": 0,
            "Escape": 0, "Enter": 0, "Up": 0, "Down": 0,
            "Left": 0, "Right": 0
        }

        # 定义按键映射
        self.key_map = {
            pygame.K_a: "A", pygame.K_b: "B", pygame.K_c: "C", pygame.K_d: "D",
            pygame.K_w: "W", pygame.K_s: "S", pygame.K_x: "X", pygame.K_y: "Y",
            pygame.K_UP: "Up", pygame.K_DOWN: "Down", pygame.K_LEFT: "Left", pygame.K_RIGHT: "Right",
            pygame.K_SPACE: "Space", pygame.K_RETURN: "Enter", pygame.K_ESCAPE: "Escape"
        }

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((host, port))
        except socket.error as e:
            print(f"无法连接到服务器 {host}:{port}")
            self.client_socket.close()
            raise e

        self.running = True
        self.receive_thread = threading.Thread(target=self.receive)
        self.receive_thread.start()

    def receive(self):
        try:
            while self.running:
                data = self.client_socket.recv(1024).decode()
                if data:
                    event_type, key = data.split(':')
                    key_code = int(key)
                    if key_code in self.key_map:
                        key_name = self.key_map[key_code]
                        if event_type == 'KEYDOWN':
                            self.keyboard_state[key_name] = 1
                        elif event_type == 'KEYUP':
                            self.keyboard_state[key_name] = 0
                else:
                    # 服务器关闭了连接
                    print("服务器关闭了连接")
                    self.running = False
        except Exception as e:
            print(f"接收数据时发生错误: {e}")
        finally:
            self.close()

    def update(self):
        pass

    def get_state(self):
        return self.keyboard_state.copy()
        
    def close(self):
        if self.running:
            self.running = False
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
            except Exception as e:
                print(f"关闭套接字时发生错误: {e}")
            finally:
                self.client_socket.close()
                print("键盘客户端已关闭")

