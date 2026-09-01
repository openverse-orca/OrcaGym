"""
LiDAR ROS2 Bridge — 将 gRPC LiDAR 数据转换为 ROS2 消息发布

发布 Topic:
  /scan          — sensor_msgs/LaserScan (2D, 最低垂直层)
  /point_cloud   — sensor_msgs/PointCloud2 (3D, 含 intensity)

依赖安装（按 Ubuntu 版本选择对应 ROS2 发行版）:
  Ubuntu 24.04 (noble) — ROS2 Jazzy:
    sudo apt install -y ros-jazzy-ros-base ros-jazzy-sensor-msgs ros-jazzy-sensor-msgs-py ros-jazzy-std-msgs ros-jazzy-rviz2
    conda create -n ros2_bridge python=3.12 -y
  Ubuntu 22.04 (jammy) — ROS2 Humble:
    sudo apt install -y ros-humble-ros-base ros-humble-sensor-msgs ros-humble-sensor-msgs-py ros-humble-std-msgs ros-humble-rviz2
    conda create -n ros2_bridge python=3.10 -y
  通用:
    conda run -n ros2_bridge pip install numpy grpcio grpcio-tools protobuf

用法（脚本启动时自动检测 /opt/ros/<distro> 下已安装的 ROS2 版本，无需手动指定路径）:
  source /opt/ros/<distro>/setup.bash  # humble 或 jazzy
  conda activate ros2_bridge
  python lidar_ros2_bridge.py --entity LiDAR --frame_id base_scan

RViz2 查看:
  rviz2
  → Fixed Frame: base_scan
  → Add → LaserScan → Topic: /scan
  → Add → PointCloud2 → Topic: /point_cloud
"""

import argparse
import sys
import os
import glob
import struct
import time


def _detect_ros2_root():
    """扫描 /opt/ros/ 下已安装的 ROS2 发行版，按字母序取首个有效安装。"""
    for candidate in sorted(glob.glob("/opt/ros/*")):
        if os.path.isfile(os.path.join(candidate, "setup.bash")):
            return candidate
    return None


def _detect_py_version(ros2_root):
    """从 ROS2 安装目录探测 Python 版本目录名（如 python3.12 / python3.10）。"""
    for pattern in ("local/lib/python3.*", "lib/python3.*"):
        matches = sorted(glob.glob(os.path.join(ros2_root, pattern)))
        if matches:
            return os.path.basename(matches[0])
    # 兜底：使用当前解释器版本
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


_ros2_root = _detect_ros2_root()
if _ros2_root:
    _py_ver = _detect_py_version(_ros2_root)
    # Jazzy 布局：lib/python3.12/site-packages（无 local/lib 分支）
    # Humble 布局：local/lib/python3.10/dist-packages + lib/python3.10/site-packages
    _py_candidates = [
        f"{_ros2_root}/local/lib/{_py_ver}/dist-packages",
        f"{_ros2_root}/lib/{_py_ver}/site-packages",
    ]
    _ros2_py_path = ":".join(p for p in _py_candidates if os.path.isdir(p))
    _need_restart = False
    if _ros2_py_path and _ros2_py_path not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = _ros2_py_path + ":" + os.environ.get("PYTHONPATH", "")
        _need_restart = True
    if _ros2_root + "/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = _ros2_root + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
        _need_restart = True
    # LD_LIBRARY_PATH 在进程启动时被 ld.so 缓存，运行时修改 os.environ 不影响当前进程的 dlopen。
    # 通过 os.execv 重启自身，让新进程以正确的环境变量启动。
    if _need_restart and "_ROS2_BRIDGE_RESTARTED" not in os.environ:
        os.environ["_ROS2_BRIDGE_RESTARTED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import grpc

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proto_path = os.path.abspath(os.path.join(proj_dir, "protos"))
sys.path.append(proto_path)
import mjc_message_pb2
import mjc_message_pb2_grpc

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan, PointCloud2, PointField
    from std_msgs.msg import Header
except ImportError:
    print("[ERROR] ROS2 Python packages not found.")
    print("  Ubuntu 22.04 (jammy): sudo apt install -y ros-humble-ros-base ros-humble-sensor-msgs ros-humble-sensor-msgs-py ros-humble-std-msgs")
    print("  Ubuntu 24.04 (noble): sudo apt install -y ros-jazzy-ros-base ros-jazzy-sensor-msgs ros-jazzy-sensor-msgs-py ros-jazzy-std-msgs")
    print("  source /opt/ros/<distro>/setup.bash")
    sys.exit(1)


def query_lidar(stub, entity_name):
    request = mjc_message_pb2.LiDARPointCloudRequest(entity_name=entity_name)
    try:
        response = stub.QueryLiDARPointCloud(request, timeout=2.0)
    except grpc.RpcError as e:
        print(f"[ERROR] gRPC call failed: {e.code()}")
        return None

    if response.status == mjc_message_pb2.LiDARPointCloudResponse.ENTITY_NOT_FOUND:
        print(f"[ERROR] LiDAR entity not found: {entity_name}")
        return None
    if response.status == mjc_message_pb2.LiDARPointCloudResponse.NO_DATA:
        return None

    result = {
        "bin_count": response.bin_count,
        "vertical_layers": response.vertical_layers,
        "angular_resolution": response.angular_resolution,
        "max_h_angle": response.max_h_angle,
        "min_v_angle": response.min_v_angle,
        "v_step": response.v_step,
        "min_range": response.min_range,
        "max_range": response.max_range,
    }

    if response.range_data:
        ranges = np.frombuffer(response.range_data, dtype=np.float32).copy()
        result["ranges"] = ranges.reshape(response.bin_count, response.vertical_layers)
    else:
        result["ranges"] = np.full((response.bin_count, response.vertical_layers), -1.0, dtype=np.float32)

    if response.point_data:
        points = np.frombuffer(response.point_data, dtype=np.float32).copy()
        result["points"] = points.reshape(response.bin_count, response.vertical_layers, 3)
    else:
        result["points"] = np.zeros((response.bin_count, response.vertical_layers, 3), dtype=np.float32)

    if response.intensity_data:
        intensities = np.frombuffer(response.intensity_data, dtype=np.float32).copy()
        result["intensities"] = intensities.reshape(response.bin_count, response.vertical_layers)
    else:
        result["intensities"] = np.zeros((response.bin_count, response.vertical_layers), dtype=np.float32)

    return result


def to_laser_scan(data, frame_id, stamp):
    msg = LaserScan()
    msg.header = Header(frame_id=frame_id, stamp=stamp)

    msg.angle_min = 0.0
    msg.angle_max = data["max_h_angle"]
    msg.angle_increment = data["angular_resolution"]
    msg.time_increment = 0.0
    msg.scan_time = 1.0 / 10.0
    msg.range_min = data["min_range"]
    msg.range_max = data["max_range"]

    ranges_2d = data["ranges"][:, 0]
    intensities_2d = data["intensities"][:, 0]

    valid = ranges_2d > 0
    msg.ranges = [float(r) if v else float("inf") for r, v in zip(ranges_2d, valid)]
    msg.intensities = [float(i) if v else 0.0 for i, v in zip(intensities_2d, valid)]

    return msg


def to_point_cloud2(data, frame_id, stamp):
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)

    ranges = data["ranges"]
    points = data["points"]
    intensities = data["intensities"]

    valid = ranges > 0
    pts = points[valid]
    ints = intensities[valid]

    n = pts.shape[0]

    msg.height = 1
    msg.width = n

    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    msg.is_dense = True

    buf = bytearray(n * 16)
    for i in range(n):
        struct.pack_into("ffff", buf, i * 16,
                         float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]),
                         float(ints[i]))

    msg.data = bytes(buf)
    return msg


class LiDARRos2Bridge(Node):
    def __init__(self, stub, entity_name, frame_id, hz):
        super().__init__("lidar_ros2_bridge")
        self.stub = stub
        self.entity_name = entity_name
        self.frame_id = frame_id

        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/point_cloud", 10)

        interval = 1.0 / hz
        self.timer = self.create_timer(interval, self.timer_callback)

        self.frame_count = 0
        self.get_logger().info(
            f"Bridge started: entity={entity_name}, frame_id={frame_id}, hz={hz}"
        )

    def timer_callback(self):
        data = query_lidar(self.stub, self.entity_name)
        if data is None:
            return

        self.frame_count += 1
        now = self.get_clock().now().to_msg()

        scan_msg = to_laser_scan(data, self.frame_id, now)
        cloud_msg = to_point_cloud2(data, self.frame_id, now)

        self.scan_pub.publish(scan_msg)
        self.cloud_pub.publish(cloud_msg)

        if self.frame_count % 30 == 0:
            n_valid = int(np.sum(data["ranges"] > 0))
            self.get_logger().info(
                f"Published frame {self.frame_count}: {n_valid} valid points"
            )


def main():
    parser = argparse.ArgumentParser(description="LiDAR ROS2 Bridge")
    parser.add_argument("--addr", type=str, default="localhost:50051",
                        help="gRPC server address")
    parser.add_argument("--entity", type=str, default="LiDAR",
                        help="LiDAR entity name")
    parser.add_argument("--frame_id", type=str, default="base_scan",
                        help="TF frame ID for ROS2 messages")
    parser.add_argument("--hz", type=float, default=10,
                        help="Publish rate in Hz")
    args = parser.parse_args()

    channel = grpc.insecure_channel(
        args.addr,
        options=[
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
        ],
    )
    stub = mjc_message_pb2_grpc.GrpcServiceStub(channel)

    rclpy.init()
    node = LiDARRos2Bridge(stub, args.entity, args.frame_id, args.hz)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        channel.close()


if __name__ == "__main__":
    main()
