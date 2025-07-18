from flask import Flask, request, jsonify
import numpy as np
import base64
import cv2
import threading
import time
import logging

def display_grayscale(image, windowname:str, waittime: int=10):
    img_bgr = np.repeat(image, 3, 2)
    cv2.imshow(windowname, img_bgr)
    cv2.waitKey(waittime)
    return 

def display_img(image, waittime: int, windowname: str):
    img_bgr = image[..., ::-1]
    # print(img_bgr.size)
    cv2.imshow(windowname, img_bgr)
    return cv2.waitKey(waittime)

class RecFromOrcaMsg:
    def __init__(self,
                 port:int = 15532,
                 ip:str = "192.168.110.227"
                 ):
        logging.getLogger("werkzeug").setLevel(logging.ERROR)  # 仅输出错误日志
        self.pos = None
        self.yaw = None
        self.mat = None
        self.person_pos_xy = None
        self.app = Flask(__name__)
        self.port=port
        self.ip=ip
        self.cnt=0
        self.lock = threading.Lock()            # 线程安全的锁（可选，用于保护共享变量）
        self.thread = None
        self.set_route()                        # 必须初始化
        self.run()

    def set_route(self):
        @self.app.route('/posyaw', methods=['POST'])
        def receive_data():
            data = request.json
            if not data:
                return jsonify({"error": "No data received"}), 400
            
            # 解析数组
            self.yaw = np.array(data["yaw"], dtype=np.float32)
            self.pos = np.array(data["pos"], dtype=np.float32)
            self.mat = np.array(data["mat"], dtype=np.float32)
            self.person_pos_xy = np.array(data["person_pos_xy"], dtype=np.float32)

            # print(f"Received yaw: {self.yaw}")
            # print(f"Received pos: {self.pos}")
            self.cnt+=1
            return jsonify({"status": "success"}), 200

    def run(self):
        # 启动 Flask 服务器到子线程
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()

    def _run_server(self):
        # 启动 Flask 服务器（非阻塞方式）
        self.app.run(host=self.ip, port=self.port, threaded=True)  # 允许多线程处理请求

    def get_pos(self):
        with self.lock:
            return self.pos if self.pos is not None else None
    
    def get_yaw(self):
        with self.lock:
            return self.yaw if self.yaw is not None else None
    def get_mat(self):
        with self.lock:
            return self.mat if self.mat is not None else None
        
    def get_person_pos_xy(self):
        with self.lock:
            return self.person_pos_xy if self.person_pos_xy is not None else None
if __name__ == "__main__":
    server = RecFromOrcaMsg()  # 启动服务器

        # 主线程可以继续执行其他任务
    try:
        while True:
            # 其他操作...
            time.sleep(0.05)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

