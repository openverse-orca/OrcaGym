"""
LiDAR 实时可视化工具

实时获取 LiDAR 点云数据，展示四种视图：
  1. 2D LaserScan — 最低层水平扫描的极坐标/笛卡尔投影
  2. Occupancy Grid — XY 平面占据栅格地图
  3. Depth Image — bin × layer 距离图像
  4. 3D Point Cloud — 三维点云散点图，按距离着色

用法:
  python lidar_viewer.py [--addr localhost:50051] [--entity LiDAREntity] [--hz 10]
"""

import argparse
import sys
import os
import time

import numpy as np
import grpc

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proto_path = os.path.abspath(os.path.join(proj_dir, "protos"))
sys.path.append(proto_path)
import mjc_message_pb2
import mjc_message_pb2_grpc

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


def build_2d_laserscan(data):
    ranges_2d = data["ranges"][:, 0]
    bin_count = data["bin_count"]
    angular_res = data["angular_resolution"]
    max_h_angle = data["max_h_angle"]
    max_range = data["max_range"]

    angles = np.linspace(0, max_h_angle, bin_count, endpoint=False)

    valid = ranges_2d > 0
    x = np.where(valid, ranges_2d * np.cos(angles), np.nan)
    y = np.where(valid, ranges_2d * np.sin(angles), np.nan)

    return x, y, angles, ranges_2d, valid, max_range


def build_occupancy_grid(data, resolution=0.05, grid_size=40.0):
    points = data["points"]
    ranges = data["ranges"]

    valid = ranges > 0
    pts = points[valid]

    if pts.shape[0] == 0:
        half = int(grid_size / resolution / 2)
        return np.zeros((half * 2, half * 2), dtype=np.float32), resolution, grid_size

    xy = pts[:, :2]

    half = int(grid_size / resolution / 2)
    grid = np.zeros((half * 2, half * 2), dtype=np.float32)

    gx = ((xy[:, 0] + grid_size / 2) / resolution).astype(np.int32)
    gy = ((xy[:, 1] + grid_size / 2) / resolution).astype(np.int32)

    mask = (gx >= 0) & (gx < half * 2) & (gy >= 0) & (gy < half * 2)
    gx = gx[mask]
    gy = gy[mask]

    grid[gy, gx] = 1.0

    return grid, resolution, grid_size


def build_depth_image(data):
    ranges = data["ranges"].copy()
    max_range = data["max_range"]
    depth = np.where(ranges > 0, ranges / max_range, 0.0)
    return depth


class LiDARViewer:
    def __init__(self, addr, entity_name, hz):
        self.entity_name = entity_name
        self.interval_ms = int(1000 / hz)
        self.last_data = None
        self.frame_count = 0
        self.fps = 0.0
        self._fps_t0 = time.perf_counter()
        self._fps_frames = 0

        self.channel = grpc.insecure_channel(
            addr,
            options=[
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ],
        )
        self.stub = mjc_message_pb2_grpc.GrpcServiceStub(self.channel)
        print(f"Connected to {addr}, entity={entity_name}, hz={hz}")

        self._setup_figure()

    def _setup_figure(self):
        self.fig = plt.figure("LiDAR Viewer", figsize=(18, 9))
        self.fig.patch.set_facecolor("#1a1a2e")

        gs = self.fig.add_gridspec(2, 3, hspace=0.30, wspace=0.28,
                                   left=0.04, right=0.97, top=0.92, bottom=0.06,
                                   width_ratios=[1, 1.3, 0.8])

        self.ax_scan = self.fig.add_subplot(gs[0, 0])
        self.ax_occ = self.fig.add_subplot(gs[1, 0])
        self.ax_3d = self.fig.add_subplot(gs[:, 1], projection="3d")
        self.ax_depth = self.fig.add_subplot(gs[:, 2])

        for ax in [self.ax_scan, self.ax_occ, self.ax_depth]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#444")

        self.ax_3d.set_facecolor("#16213e")
        self.ax_3d.tick_params(colors="white", labelsize=7)
        for pane in [self.ax_3d.xaxis.pane, self.ax_3d.yaxis.pane, self.ax_3d.zaxis.pane]:
            pane.set_facecolor("#16213e")
            pane.set_alpha(1.0)
        self.ax_3d.xaxis.line.set_color("#444")
        self.ax_3d.yaxis.line.set_color("#444")
        self.ax_3d.zaxis.line.set_color("#444")

        self.ax_scan.set_title("2D LaserScan", color="white", fontsize=11, fontweight="bold")
        self.ax_occ.set_title("Occupancy Grid (XY)", color="white", fontsize=11, fontweight="bold")
        self.ax_3d.set_title("3D Point Cloud", color="white", fontsize=11, fontweight="bold")
        self.ax_depth.set_title("Depth Image", color="white", fontsize=11, fontweight="bold")

        self.info_text = self.fig.text(
            0.5, 0.96, "", ha="center", va="center",
            color="#00ccff", fontsize=10, fontfamily="monospace",
        )

        self._3d_elev = 25
        self._3d_azim = -60

    def _update(self, frame):
        data = query_lidar(self.stub, self.entity_name)
        if data is None:
            return []

        self.last_data = data
        self.frame_count += 1
        self._fps_frames += 1

        now = time.perf_counter()
        dt = now - self._fps_t0
        if dt >= 1.0:
            self.fps = self._fps_frames / dt
            self._fps_frames = 0
            self._fps_t0 = now

        x, y, angles, ranges_2d, valid, max_range = build_2d_laserscan(data)

        self.ax_scan.clear()
        self.ax_scan.set_facecolor("#16213e")
        self.ax_scan.set_title("2D LaserScan", color="white", fontsize=11, fontweight="bold")
        self.ax_scan.plot(0, 0, "o", color="#ff4444", markersize=5, zorder=5)

        if np.any(valid):
            self.ax_scan.plot(x[valid], y[valid], ".", color="#00ff88", markersize=2, zorder=3)

            max_r = max_range * 1.1
            self.ax_scan.set_xlim(-max_r, max_r)
            self.ax_scan.set_ylim(-max_r, max_r)
        else:
            self.ax_scan.set_xlim(-5, 5)
            self.ax_scan.set_ylim(-5, 5)

        self.ax_scan.set_aspect("equal")
        self.ax_scan.set_xlabel("X (m)", color="white", fontsize=8)
        self.ax_scan.set_ylabel("Y (m)", color="white", fontsize=8)
        self.ax_scan.tick_params(colors="white", labelsize=7)
        self.ax_scan.grid(True, alpha=0.15, color="white")

        grid, resolution, grid_size = build_occupancy_grid(data, resolution=0.05, grid_size=40.0)

        self.ax_occ.clear()
        self.ax_occ.set_facecolor("#16213e")
        self.ax_occ.set_title("Occupancy Grid (XY)", color="white", fontsize=11, fontweight="bold")

        extent = [-grid_size / 2, grid_size / 2, -grid_size / 2, grid_size / 2]
        cmap = matplotlib.colors.ListedColormap(["#16213e", "#00ff88"])
        self.ax_occ.imshow(grid, origin="lower", extent=extent, cmap=cmap,
                           vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        self.ax_occ.plot(0, 0, "o", color="#ff4444", markersize=4, zorder=5)
        self.ax_occ.set_xlabel("X (m)", color="white", fontsize=8)
        self.ax_occ.set_ylabel("Y (m)", color="white", fontsize=8)
        self.ax_occ.tick_params(colors="white", labelsize=7)

        self._draw_3d(data)

        depth = build_depth_image(data)

        self.ax_depth.clear()
        self.ax_depth.set_facecolor("#16213e")
        self.ax_depth.set_title("Depth Image", color="white", fontsize=11, fontweight="bold")
        self.ax_depth.imshow(depth.T, origin="lower", cmap="inferno",
                             vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        self.ax_depth.set_xlabel("Bin", color="white", fontsize=8)
        self.ax_depth.set_ylabel("Layer", color="white", fontsize=8)
        self.ax_depth.tick_params(colors="white", labelsize=7)

        n_valid = int(np.sum(data["ranges"] > 0))
        total = data["bin_count"] * data["vertical_layers"]
        self.info_text.set_text(
            f"FPS: {self.fps:.1f} | "
            f"Bin×Layer: {data['bin_count']}×{data['vertical_layers']} | "
            f"Valid: {n_valid}/{total} | "
            f"FOV: {np.degrees(data['max_h_angle']):.1f}°×"
            f"{np.degrees(data['v_step'] * data['vertical_layers']):.1f}° | "
            f"Range: [{data['min_range']:.2f}, {data['max_range']:.2f}]m"
        )

        return []

    def _draw_3d(self, data):
        try:
            self._3d_elev = self.ax_3d.elev
            self._3d_azim = self.ax_3d.azim
        except Exception:
            pass

        self.ax_3d.clear()
        self.ax_3d.set_facecolor("#16213e")
        self.ax_3d.set_title("3D Point Cloud", color="white", fontsize=11, fontweight="bold")

        points = data["points"]
        ranges = data["ranges"]
        max_range = data["max_range"]

        valid = ranges > 0
        pts = points[valid]

        self.ax_3d.scatter([0], [0], [0], c="#ff4444", s=30, marker="o", depthshade=False, zorder=5)

        if pts.shape[0] > 0:
            dists = np.linalg.norm(pts, axis=1)
            colors = dists / max_range
            colors = np.clip(colors, 0, 1)

            step = max(1, pts.shape[0] // 8000)
            idx = np.arange(0, pts.shape[0], step)

            self.ax_3d.scatter(
                pts[idx, 0], pts[idx, 1], pts[idx, 2],
                c=colors[idx], cmap="turbo", vmin=0, vmax=1,
                s=1, depthshade=True, alpha=0.8,
            )

        lim = max_range * 0.6
        self.ax_3d.set_xlim(-lim, lim)
        self.ax_3d.set_ylim(-lim, lim)
        self.ax_3d.set_zlim(-lim * 0.5, lim * 0.5)

        self.ax_3d.set_xlabel("X", color="white", fontsize=8, labelpad=1)
        self.ax_3d.set_ylabel("Y", color="white", fontsize=8, labelpad=1)
        self.ax_3d.set_zlabel("Z", color="white", fontsize=8, labelpad=1)
        self.ax_3d.tick_params(colors="white", labelsize=6, pad=0)

        for pane in [self.ax_3d.xaxis.pane, self.ax_3d.yaxis.pane, self.ax_3d.zaxis.pane]:
            pane.set_facecolor("#16213e")
            pane.set_alpha(1.0)
        self.ax_3d.xaxis.line.set_color("#444")
        self.ax_3d.yaxis.line.set_color("#444")
        self.ax_3d.zaxis.line.set_color("#444")

        self.ax_3d.view_init(elev=self._3d_elev, azim=self._3d_azim)

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self._update,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

    def close(self):
        if self.channel:
            self.channel.close()


def main():
    parser = argparse.ArgumentParser(description="LiDAR real-time viewer")
    parser.add_argument("--addr", type=str, default="localhost:50051",
                        help="gRPC server address (default: localhost:50051)")
    parser.add_argument("--entity", type=str, default="LiDAR",
                        help="LiDAR entity name in O3DE (default: LiDAR)")
    parser.add_argument("--hz", type=float, default=10,
                        help="Query rate in Hz (default: 10)")
    args = parser.parse_args()

    viewer = LiDARViewer(args.addr, args.entity, args.hz)
    try:
        viewer.run()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
