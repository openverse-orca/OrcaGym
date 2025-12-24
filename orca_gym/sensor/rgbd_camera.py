import asyncio
import websockets
import av
import cv2
import io
import threading
import time
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class CameraWrapper:
    def __init__(self, name:str, port:int):
        self._name = name
        self.port = port
        self.image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        self.enabled = True
        self.received_first_frame = False
        self.image_index : int = 0

    def __del__(self):
        if not self.enabled:
            return
        self.running = False
        self.thread.join()

    def start(self):
        if not self.enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        asyncio.run(self.do_stuff())
    
    @property
    def name(self):
        return self._name
    
    def is_first_frame_received(self):
        return self.received_first_frame

    async def do_stuff(self):
        uri = f"ws://localhost:{self.port}"
        # print(f"start connecting to {uri}")
        async with websockets.connect(uri) as websocket:
            cur_pos = 0
            rawData = io.BytesIO()
            container = None
            while self.running:
                data = await websocket.recv()
                data = data[8:]
                # print(f'{len(data)}')
                rawData.write(data)
                rawData.seek(cur_pos)
                if cur_pos == 0:
                    container = av.open(rawData, mode='r')
                    # print(container.streams.video[0].codec_context.name)

                for packet in container.demux():
                    if packet.size == 0:
                        continue
                    frames = packet.decode()
                    for frame in frames:
                        self.image = frame.to_ndarray(format='bgr24')
                        self.image_index += 1
                        # print("get new frame for port ", self.port)
                        if self.received_first_frame == False:
                            self.received_first_frame = True
                        # print(img.shape)
                        # cv2.imshow(self.name, self.image)
                cur_pos += len(data)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # cv2.destroyAllWindows()


    def stop(self):
        if not self.enabled:
            return
        self.running = False
        asyncio.get_event_loop().stop()

    def get_frame(self, format='bgr24', size : tuple = None) -> tuple[np.ndarray, int]:
        if format == 'bgr24':
            frame = self.image
        elif format == 'rgb24':
            frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
        if size is not None:
            frame = cv2.resize(frame, size)
            
        return frame, self.image_index
    


class CameraCacher:
    def __init__(self, name:str, port:int):
        self.name = name
        self.port = port
        self.received_first_frame = False

    def __del__(self):
        self.running = False
        self.thread.join()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        asyncio.run(self.do_stuff())

    async def do_stuff(self):
        uri = f"ws://localhost:{self.port}"
        _logger.info("start connecting")
        async with websockets.connect(uri) as websocket:
            _logger.info("connected")
            with open(self.name + "_video.h264", "wb") as video_file:
                with open(self.name + "_ts.bin", "wb") as ts_file:
                    while self.running:
                        if self.received_first_frame == False:
                            self.received_first_frame = True
                        data = await websocket.recv()
                        ts_file.write(data[:8])
                        video_file.write(data[8:])

    
    def is_first_frame_received(self):
        return self.received_first_frame

    def stop(self):
        self.running = False
        self.thread.join()


def find_closest_index(a, target):
    # Use binary search for efficiency on an ordered list
    from bisect import bisect_left
    
    pos = bisect_left(a, target)  # Find the position to insert `target`
    
    # Compare neighbors to find the closest value
    if pos == 0:
        return 0
    if pos == len(a):
        return len(a) - 1
    before = pos - 1
    after = pos
    if abs(a[before] - target) <= abs(a[after] - target):
        return before
    else:
        return after



class CameraDataParser:
    def __init__(self, name:str):
        self.ts_list = []
        with open(name + "_ts.bin", "rb") as f:
            while True:
                ts = f.read(8)
                if not ts:
                    break
                self.ts_list.append(int.from_bytes(ts, "little"))
        self.container = av.open(name + "_video.h264", mode='r')
        self.current_index = -1
        self.last_frame = None


    def get_closed_frame(self, ts):
        index = find_closest_index(self.ts_list, ts)
        return index, self.get_frame(index)

    def get_frame(self, index):
        if index == self.current_index:
            return self.last_frame
        
        for frame in self.container.decode(video=0):
            self.current_index += 1
            if self.current_index == index:
                self.last_frame = frame.to_ndarray(format='bgr24')
                return self.last_frame

class VideoPlayer:
    def __init__(self, name:str):
        self.container = av.open(name + "_video.h264", mode='r')

    def play(self):
        for frame in self.container.decode(video=0):
            self.image = frame.to_ndarray(format='bgr24')
            cv2.imshow('video', self.image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
          
class Monitor:
    def __init__(self, name : str, fps=30, port=7070):
        self.camera = CameraWrapper(name, port)
        self.camera.start()
        
        self.fps = fps
        self.interval = 1000 / self.fps  # 更新间隔，单位为毫秒
        
        # 创建 Matplotlib 图形和轴
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')  # 关闭坐标轴
        
        frame, _ = self.camera.get_frame(format='rgb24')
        
        # 显示初始图像
        self.im = self.ax.imshow(frame)
        self.ax.set_title("Camera Feed")
    
    def update(self, frame_num):
        frame, _ = self.camera.get_frame(format='rgb24')
        
        # 更新图像数据
        self.im.set_data(frame)
        return self.im,
    
    def start(self):
        # 创建动画
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=self.interval,
            blit=True
        )
        plt.show()
    
    def stop(self):
        # 释放摄像头资源
        self.camera.stop()
        plt.close(self.fig)
    
    def __del__(self):
        self.stop()
        

