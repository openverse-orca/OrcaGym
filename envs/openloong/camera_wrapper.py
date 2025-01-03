import asyncio
import websockets
import av
import cv2
import io
import threading
import time
import numpy as np


class CameraWrapper:
    def __init__(self, name:str, port:int):
        self.name = name
        self.port = port
        self.image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        self.enabled = True
        self.received_first_frame = False

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
    
    def get_name(self):
        return self.name
    
    def is_first_frame_received(self):
        return self.received_first_frame

    async def do_stuff(self):
        uri = f"ws://localhost:{self.port}"
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
                        if self.received_first_frame == False:
                            self.received_first_frame = True
                        # print(img.shape)
                        # cv2.imshow(self.name, self.image)
                cur_pos += len(data)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # cv2.destroyAllWindows()


    def stop(self):
        pass

    def get_frame(self):
        return self.image
    


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
        print("start connecting")
        async with websockets.connect(uri) as websocket:
            print("connected")
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


# if __name__ == "__main__":
    # camera = CameraCacher("camera", 7070)
    # camera.start()
    # while not camera.is_first_frame_received():
    #     time.sleep(0.001)
    # time.sleep(10)
    # camera.stop()
    # print("done")

    # parser = CameraDataParser()
    # print(parser.ts_list)
    # print(parser.get_closed_frame(1731662211073)[0])

    # player = VideoPlayer()
    # player.play()
    # print("done")