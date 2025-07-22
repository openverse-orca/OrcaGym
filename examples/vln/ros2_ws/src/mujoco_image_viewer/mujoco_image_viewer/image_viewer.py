
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Any, Optional,List
import base64
import time
import requests

# 这里加.表示是自定义库
from .net_msg.recOrcaMsg import RecFromOrcaMsg

def display_grayscale(image, waittime: int=100):
    image = np.expand_dims(image, axis=-1)
    img_bgr = np.repeat(image, 3, 2)
    # print(f"Image data type: {img_bgr.dtype}")
    # print(img_bgr.size, img_bgr.shape)
    
    cv2.imshow("Depth Sensor", img_bgr.astype(np.uint8))
    return cv2.waitKey(waittime)

class ImageSender(Node):
    def __init__(self, port:int = 15631, vln_ip:str="localhost", ip:str="localhost"):
        super().__init__('img_send')
        self.subscription = self.create_subscription(
            Image,
            '/sites/camera_image_color',  # 替换为你的图像话题
            self.image_callback,
            20)
        self.subscription = self.create_subscription(
            Image,
            '/sites/camera_image_depth',  # 替换为你的图像话题
            self.depth_callback,
            20)

        self.gray_shape = None
        self.bridge = CvBridge()
        self.current_array = [0.0]*22  # 初始化数组
        self.url = "http://"+vln_ip+f":{port}/ros2tf"  # 目标URL
        self.cnt = 0
        self.img_base64 = None
        self.depth_base64 = None
        self.init_time=time.time()
        self.server_get_pos_yaw = RecFromOrcaMsg(port=port+1, ip="localhost")  # 15532


    def image_callback(self, msg):
        # 将ROS图像转换为OpenCV格式
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('MuJoCo Camera', cv_image)
        cv2.waitKey(1)
        # 将图像编码为Base64
        _, img_encoded = cv2.imencode('.jpg', cv_image)
        self.img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        # 组合数据更新
        if self.cnt==0:
            self.init_time=time.time()
        self.cnt+=1
        self.current_array[0]=time.time()-self.init_time
        self.current_array[1]=self.cnt

        pos = self.server_get_pos_yaw.get_pos()
        yaw = self.server_get_pos_yaw.get_yaw()
        mat = self.server_get_pos_yaw.get_mat()
        person_pos_xy = self.server_get_pos_yaw.get_person_pos_xy()
        # 输出mat的shape
        # print(f"mat shape: {mat}")
        # print(f"pos: {pos}, yaw: {yaw}")
        if yaw is not None:
            self.current_array[2]=yaw.tolist()
        if pos is not None:
            self.current_array[3:6]=pos.tolist()
        if mat is not None:
            self.current_array[6:22]=mat.tolist()
        if person_pos_xy is not None:
            self.current_array[22:24]=person_pos_xy.tolist()
        # print(self.current_array)
        data = {
            "image": self.img_base64,
            "depth": self.depth_base64,
            "array": self.current_array,  # 数组需要转换为列表或JSON兼容格式
            "shape": self.gray_shape
        }
        # 发送HTTP POST请求
        try:
            response = requests.post(self.url, json=data)
            self.get_logger().info(f"Sent data: {response.status_code}")

        except Exception as e:
            # self.get_logger().error(f"Failed to send data: {str(e)}")
            print(e)
            pass
    
    def depth_callback(self, msg):


        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.gray_shape = cv_image.shape
        display_grayscale(cv_image)
        # 将float32数组转为二进制字节流
        depth_bytes = cv_image.tobytes()  # 直接获取原始字节
        self.depth_base64 = base64.b64encode(depth_bytes).decode('utf-8')

        # # 归一化到 0-255 并转为 uint8
        # max_depth = cv_image.max()
        # # 传感器最大支持5m
        # sensor_max = 4.9999995
        # if max_depth > 0:
        #     normalized_depth = (cv_image / sensor_max * 255).astype(np.uint8)
        # else:
        #     normalized_depth = cv_image.astype(np.uint8)  # 全黑
        # print(max_depth,cv_image.min(),"---",normalized_depth.max())
        # cv2.imshow('depth Camera', normalized_depth)
        # cv2.waitKey(1)
        # _, depth_encoded = cv2.imencode('.png', normalized_depth)
        # self.depth_base64 = base64.b64encode(depth_encoded.tobytes()).decode('utf-8')


# --------------------------
class ImageViewer(Node):
    def __init__(self):
        super().__init__('mujoco_image_viewer')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera_image_color',
            self.image_callback,
            10)
        self.get_logger().info("图像订阅已启动")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imshow('MuJoCo Camera', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {str(e)}")

# --------------------------



def main(args=None):
    rclpy.init(args=args)
    # node = ImageViewer()
    node = ImageSender(vln_ip="192.168.1.123", ip="127.0.0.1")
    # node = VideoPublisher()
    
    try:
        # print("spin----")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("关闭节点")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()