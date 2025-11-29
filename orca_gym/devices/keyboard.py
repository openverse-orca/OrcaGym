import pygame
import sys
import os
import socket
import threading
import grpc
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub 
import orca_gym.protos.mjc_message_pb2 as mjc_message_pb2
import asyncio
import numpy as np

class KeyboardInputSourceType:
    PYGAME = "pygame"
    ORCASTUDIO = "orcastudio"


class KeyboardInputSourcePygame:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Initialize the display (even if not used, it's required for capturing events)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        pygame.display.set_mode((400, 400))

        # Define the key mapping
        self.key_map = {
            pygame.K_a: "A", pygame.K_b: "B", pygame.K_c: "C", pygame.K_d: "D",
            pygame.K_w: "W", pygame.K_s: "S", pygame.K_x: "X", pygame.K_y: "Y", pygame.K_z: "Z",
            pygame.K_q: "Q", pygame.K_e: "E", pygame.K_r: "R", pygame.K_f: "F",
            pygame.K_UP: "Up", pygame.K_DOWN: "Down", pygame.K_LEFT: "Left", pygame.K_RIGHT: "Right",
            pygame.K_SPACE: "Space", pygame.K_RETURN: "Enter", pygame.K_ESCAPE: "Esc",
            pygame.K_LSHIFT: "LShift", pygame.K_RSHIFT: "RShift",
            pygame.K_LCTRL: "Ctrl", pygame.K_RCTRL: "Ctrl",
            pygame.K_LALT: "Alt", pygame.K_RALT: "Alt"
        }

    def update_keyboard_state(self, keyboard_state):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in self.key_map:
                    keyboard_state[self.key_map[event.key]] = 1
                    # print(f"Key pressed: {self.key_map[event.key]}")
            elif event.type == pygame.KEYUP:
                if event.key in self.key_map:
                    keyboard_state[self.key_map[event.key]] = 0
                    # print(f"Key released: {self.key_map[event.key]}")

class KeyboardInputSourceOrcaStudio:
    def __init__(self,
                 grpc_addr: str,):
        """
        Initialize the OrcaStudio keyboard input source.
        """
        self.grpc_addr = grpc_addr
        self.loop = asyncio.get_event_loop()
        self.initialize_grpc()

        self.keyboard_map = {
            "keyboard_key_alphanumeric_A" : "A",
            "keyboard_key_alphanumeric_B" : "B",
            "keyboard_key_alphanumeric_C" : "C",
            "keyboard_key_alphanumeric_D" : "D",
            "keyboard_key_alphanumeric_E" : "E",
            "keyboard_key_alphanumeric_F" : "F",
            "keyboard_key_alphanumeric_G" : "G",
            "keyboard_key_alphanumeric_H" : "H",
            "keyboard_key_alphanumeric_I" : "I",
            "keyboard_key_alphanumeric_J" : "J",
            "keyboard_key_alphanumeric_K" : "K",
            "keyboard_key_alphanumeric_L" : "L",
            "keyboard_key_alphanumeric_M" : "M",
            "keyboard_key_alphanumeric_N" : "N",
            "keyboard_key_alphanumeric_O" : "O",
            "keyboard_key_alphanumeric_P" : "P",
            "keyboard_key_alphanumeric_Q" : "Q",
            "keyboard_key_alphanumeric_R" : "R",
            "keyboard_key_alphanumeric_S" : "S",
            "keyboard_key_alphanumeric_T" : "T",
            "keyboard_key_alphanumeric_U" : "U",
            "keyboard_key_alphanumeric_V" : "V",
            "keyboard_key_alphanumeric_W" : "W",
            "keyboard_key_alphanumeric_X" : "X",
            "keyboard_key_alphanumeric_Y" : "Y",
            "keyboard_key_alphanumeric_Z" : "Z",
            "keyboard_key_alphanumeric_0" : "0",
            "keyboard_key_alphanumeric_1" : "1",
            "keyboard_key_alphanumeric_2" : "2",
            "keyboard_key_alphanumeric_3" : "3",
            "keyboard_key_alphanumeric_4" : "4",
            "keyboard_key_alphanumeric_5" : "5",
            "keyboard_key_alphanumeric_6" : "6",
            "keyboard_key_alphanumeric_7" : "7",
            "keyboard_key_alphanumeric_8" : "8",
            "keyboard_key_alphanumeric_9" : "9",
            "keyboard_key_edit_backspace" : "Backspace",
            "keyboard_key_edit_capslock" : "CapsLock",
            "keyboard_key_edit_enter" : "Enter",
            "keyboard_key_edit_space" : "Space",
            "keyboard_key_edit_tab" : "Tab",
            "keyboard_key_escape" : "Esc",
            "keyboard_key_function_F01" : "F1",
            "keyboard_key_function_F02" : "F2",
            "keyboard_key_function_F03" : "F3",
            "keyboard_key_function_F04" : "F4",
            "keyboard_key_function_F05" : "F5",
            "keyboard_key_function_F06" : "F6",
            "keyboard_key_function_F07" : "F7",
            "keyboard_key_function_F08" : "F8",
            "keyboard_key_function_F09" : "F9",
            "keyboard_key_function_F10" : "F10",
            "keyboard_key_function_F11" : "F11",
            "keyboard_key_function_F12" : "F12",
            "keyboard_key_modifier_alt_l" : "Alt",
            "keyboard_key_modifier_alt_r" : "Alt",
            "keyboard_key_modifier_ctrl_l" : "Ctrl",
            "keyboard_key_modifier_ctrl_r" : "Ctrl",
            "keyboard_key_modifier_shift_l" : "LShift",
            "keyboard_key_modifier_shift_r" : "RShift",
            "keyboard_key_navigation_arrow_down" : "Down",
            "keyboard_key_navigation_arrow_left" : "Left",
            "keyboard_key_navigation_arrow_right" : "Right",
            "keyboard_key_navigation_arrow_up" : "Up",
            "mouse_button_left" : "MouseLeft",
            "mouse_button_middle" : "MouseMiddle",
            "mouse_button_right" : "MouseRight",
            "mouse_button_x1" : "MouseX1",
            "mouse_button_x2" : "MouseX2",
        }

    def initialize_grpc(self):
        self.channel = grpc.aio.insecure_channel(
            self.grpc_addr,
        )
        self.stub = GrpcServiceStub(self.channel)

    async def _close_grpc(self):
        if self.channel:
            await self.channel.close()

    async def _get_key_pressed_events(self):
        request = mjc_message_pb2.GetKeyPressedEventsRequest()
        response = await self.stub.GetKeyPressedEvents(request)
        if response is None:
            raise RuntimeError("Failed to get key pressed events from OrcaStudio.")
        return response.events

    def get_key_pressed(self) -> list[str]:
        """
        Get the current key pressed state from OrcaStudio.
        """
        return self.loop.run_until_complete(self._get_key_pressed_events())


    def close(self):
        self.loop.run_until_complete(self._close_grpc())

    def update_keyboard_state(self, keyboard_state : dict[str, int]):
        key_pressed_events = self.get_key_pressed()

        # print(f"Key pressed events: {key_pressed_events}")

        [keyboard_state.update({key: 0}) for key in self.keyboard_map.values()]

        for event in key_pressed_events:
            if event in self.keyboard_map:
                key_name = self.keyboard_map[event]
                if key_name in keyboard_state:
                    keyboard_state[key_name] = 1

class KeyboardInput:
    def __init__(self,
                 input_source: str,
                 grpc_address: str = "localhost:50051"):
        
        # Initialize keyboard input source
        if input_source == KeyboardInputSourceType.PYGAME:
            self._source = KeyboardInputSourcePygame()
        elif input_source == KeyboardInputSourceType.ORCASTUDIO:
            self._source = KeyboardInputSourceOrcaStudio(grpc_address)
        else:
            raise ValueError(f"Unknown input source: {input_source}")

        # Initialize keyboard state
        self.keyboard_state = {
            "W": 0, "A": 0, "S": 0, "D": 0,
            "Space": 0, "LShift": 0, "RShift": 0, "Ctrl": 0, "Alt": 0,
            "Esc": 0, "Enter": 0, "Up": 0, "Down": 0,
            "Left": 0, "Right": 0, "Q": 0, "E": 0, "R": 0, "F": 0, "Z": 0, "X": 0, "Y": 0
        }



    def update(self):
        # Update the keyboard state based on the input source
        self._source.update_keyboard_state(self.keyboard_state)

        # Print the current keyboard state for debugging
        # print(self.keyboard_state)

    def get_state(self):
        return self.keyboard_state.copy()

    def close(self):
        pygame.quit()
        _logger.info("Keyboard controller closed")


class KeyboardServer:
    def __init__(self, host='localhost', port=55000):
        self.clients = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置 SO_REUSEADDR 选项
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((host, port))
        except OSError as e:
            _logger.info(f"Port {port} is already in use!")
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
                _logger.info(f"Client connected: {addr}")
            except Exception as e:
                _logger.error(f"Error accepting clients: {e}")
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
            _logger.error(f"An error occurred: {e}")
        finally:
            self.close()

    def close(self):
        # 关闭所有客户端套接字
        for client in self.clients:
            client.close()
        self.server_socket.close()
        pygame.quit()
        _logger.info("Keyboard server closed")



import socket
import threading
import pygame
import os

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


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
            _logger.info(f"无法连接到服务器 {host}:{port}")
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
                    _logger.info("服务器关闭了连接")
                    self.running = False
        except Exception as e:
            _logger.info(f"接收数据时发生错误: {e}")
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
                _logger.info(f"关闭套接字时发生错误: {e}")
            finally:
                self.client_socket.close()
                _logger.info("键盘客户端已关闭")

