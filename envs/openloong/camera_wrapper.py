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
        self.enabled = False

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

    async def do_stuff(self):
        uri = f"ws://localhost:{self.port}"
        async with websockets.connect(uri) as websocket:
            cur_pos = 0
            rawData = io.BytesIO()
            container = None
            while self.running:
                data = await websocket.recv()
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