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

class RecAction:
    def __init__(self,
                 port:int = 15533,
                 ip:str = "localhost" # 替换为本机IP地址
                 ):
        logging.getLogger("werkzeug").setLevel(logging.ERROR)  # 仅输出错误日志
        self.step = None
        self.mode = None
        self.action = None
        self.app = Flask(__name__)
        self.port=port
        self.ip=ip
        self.cnt=0
        self.trigger = False
        self.lock = threading.Lock()            # 线程安全的锁（可选，用于保护共享变量）
        self.thread = None
        self.set_route()                        # 必须初始化
        self.run()
        # for _ in range(10):
        #     time.sleep(0.2)
        #     print("rec action server start",self.cnt)

    def set_route(self):
        @self.app.route('/rec', methods=['POST'])
        def receive_data():
            data = request.json
            if not data:
                return jsonify({"error": "No data received"}), 400
            # print("--@@__--__--@@--")

            self.step = (data["step"])
            self.mode = (data["mode"])
            self.action = (data["action"])
            self.trigger = True

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

    def get_step(self):
        with self.lock:
            return self.step if self.step is not None else None
    def get_mode(self):
        with self.lock:
            return self.mode if self.mode is not None else None
    def get_action(self):
        with self.lock:
            return self.action if self.action is not None else None


class RecFromRos2Img:
    def __init__(self,
                 port:int = 15631,
                 ip:str = "localhost" # 替换为本机IP地址
                 ):
        logging.getLogger("werkzeug").setLevel(logging.ERROR)  # 仅输出错误日志
        self.img_float_depth = None
        self.img_uint8_color = None
        self.array = None
        self.app = Flask(__name__)
        self.port=port
        self.ip=ip
        self.cnt=0
        self.lock = threading.Lock()            # 线程安全的锁（可选，用于保护共享变量）
        self.thread = None
        self.set_route()                        # 必须初始化
        self.run()

    def set_route(self):
        @self.app.route('/ros2tf', methods=['POST'])
        def receive_data():
            data = request.json
            if not data:
                return jsonify({"error": "No data received"}), 400
            
            # 解析数组
            self.array = (data["array"])
            # print(type(self.array))
            # print(self.array[0],self.array[1])

            # 解析图像
            img_base64 = data["image"]
            img_bytes = base64.b64decode(img_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            self.img_uint8_color = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # cv2.imwrite("received_image.jpg", self.img_uint8_color)

            # 解析深度图像
            depth_base64 = data["depth"]
            shape = tuple(data["shape"])
            depth_bytes = base64.b64decode(depth_base64)
            cv_image_depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(shape)
            # 拓展1唯
            self.img_float_depth = np.expand_dims(cv_image_depth, axis=-1)
            
            # print(self.img_float_depth.shape)
            # img_bgr = np.repeat(cv_image_depth_expa, 3, 2)
            # img_bgr_uint8 = (img_bgr * 255).astype(np.uint8)
            # print(img_bgr.shape)
            # cv2.imwrite("received_depth.jpg", img_bgr_uint8)

            self.cnt+=1
            # print(self.cnt)
            return jsonify({"status": "success"}), 200

    def run(self):
        # 启动 Flask 服务器到子线程
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()

    def _run_server(self):
        # 启动 Flask 服务器（非阻塞方式）
        self.app.run(host=self.ip, port=self.port, threaded=True)  # 允许多线程处理请求


    def get_color(self):
        with self.lock:
            return self.img_uint8_color.copy() if self.img_uint8_color is not None else None

    def get_float_depth(self):
        # 安全地获取 float_image（如果需要）
        with self.lock:
            return self.img_float_depth.copy() if self.img_float_depth is not None else None
    
    def get_transform(self):
        # 安全地获取 float_image（如果需要）
        with self.lock:
            # 返回的是一个4✖4的矩阵
            # 将self.array[6,22]转化为slice
            slice_array = np.array(self.array[6:22])
            return slice_array.reshape(4,4) if slice_array is not None else None
    
if __name__ == "__main__":
    server = RecFromRos2Img(ip="192.168.110.135")

        # 主线程可以继续执行其他任务
    try:
        while True:
            # 示例：定期检查 float_image 的值
            dp,cl=server.get_float_depth(), server.get_color()
            if dp is not None and cl is not None:
                # print(dp.max(), cl.max())
                display_grayscale(dp,"rec depth")
                display_img(cl,10,"rec color")
            else: print("123123")
            if server.array is not None:
                # print(server.array[2],server.array[3],server.array[4],server.array[5])
                T = server.get_transform()
                # 通过T矩阵计算yaw
                yaw = np.arctan2(T[1, 0], T[0, 0])
                # yaw转化为角度
                yaw = np.rad2deg(yaw)
                print(f"yaw:{yaw:.3f} pos:[{T[0, 3]:.3f},{T[1, 3]:.3f},{T[2, 3]:.3f}]")
                
            # print(f"Received Array:[{server.array[0]},{server.array[1]:.3f}]",  server.cnt)
            # print(f"yaw:{server.array[2]:.3f} pos:[{server.array[3]:.3f},{server.array[4]:.3f},{server.array[5]:.3f}]")
            print("----------------------------")
            # 其他操作...
            time.sleep(0.01)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

