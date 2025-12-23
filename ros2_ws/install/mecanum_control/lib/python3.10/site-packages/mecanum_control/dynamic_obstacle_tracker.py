#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Obstacle Tracker (ROS2)

- Nhận LaserScan từ /scan
- Gom điểm LiDAR thành các cluster (vật cản)
- Track mỗi cluster qua thời gian → ước lượng (x, y, vx, vy)
- Phát hiện vật cản ĐỘNG phía trước robot dựa trên:
    * Tốc độ tương đối theo trục x
    * TTC (time-to-collision)
- Xuất:
    * /dyn_front_unsafe    (std_msgs/Bool)
    * /dyn_front_ttc_min   (std_msgs/Float32)

Có thể dùng chung với lidar_processor.py:
    - lidar_processor subscribe 2 topic này
    - Gộp vào logic front_unsafe để né vật cản động.
"""

from typing import List, Optional, Tuple
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32


class Track:
    """Đại diện một vật cản được track qua thời gian."""
    def __init__(self, track_id: int, x: float, y: float, t: float, radius: float):
        self.id = track_id
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.radius = radius
        self.last_t = t
        self.last_update_t = t  # để kiểm tra age

    def predict(self, t_now: float) -> Tuple[float, float]:
        """Dự đoán vị trí hiện tại theo model tuyến tính đơn giản."""
        dt = max(0.0, t_now - self.last_t)
        return self.x + self.vx * dt, self.y + self.vy * dt

    def update(self, cx: float, cy: float, radius: float, t_now: float, alpha: float = 0.6):
        """Cập nhật track với cluster mới."""
        dt = max(1e-3, t_now - self.last_t)
        vx_meas = (cx - self.x) / dt
        vy_meas = (cy - self.y) / dt

        # Exponential smoothing tốc độ
        self.vx = alpha * self.vx + (1.0 - alpha) * vx_meas
        self.vy = alpha * self.vy + (1.0 - alpha) * vy_meas

        self.x = cx
        self.y = cy
        self.radius = radius
        self.last_t = t_now
        self.last_update_t = t_now


class DynamicObstacleTracker(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_tracker')

        # -------- Parameters --------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_arbiter')

        # Clustering
        self.declare_parameter('cluster_dist_thresh', 0.30)  # m, max khoảng cách giữa 2 điểm cùng cluster
        self.declare_parameter('cluster_min_points', 3)      # bỏ các cluster quá nhỏ / noise

        # Tracking / association
        self.declare_parameter('assoc_dist_thresh', 0.50)    # m, khoảng cách cho phép gán cluster → track
        self.declare_parameter('track_max_age_s', 1.5)       # s, xoá track nếu quá lâu không update
        self.declare_parameter('vel_smoothing_alpha', 0.6)   # hệ số lọc tốc độ

        # Dynamic obstacle detection
        self.declare_parameter('dynamic_speed_thresh', 0.25)  # m/s, tốc độ tối thiểu để coi là động
        self.declare_parameter('ttc_thresh_front', 1.5)       # s, TTC nhỏ hơn ngưỡng → nguy hiểm
        self.declare_parameter('lane_half_width', 0.40)       # m, nửa bề rộng "làn" phía trước robot
        self.declare_parameter('min_vrel_x', 0.05)            # m/s, tốc độ tương đối tối thiểu

        # Max range để xét obstacle (giảm nhiễu xa)
        self.declare_parameter('max_range_considered', 8.0)

        # -------- Read parameters --------
        self.scan_topic = self.get_parameter('scan_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.cluster_dist_thresh = float(self.get_parameter('cluster_dist_thresh').value)
        self.cluster_min_points = int(self.get_parameter('cluster_min_points').value)

        self.assoc_dist_thresh = float(self.get_parameter('assoc_dist_thresh').value)
        self.track_max_age_s = float(self.get_parameter('track_max_age_s').value)
        self.vel_smoothing_alpha = float(self.get_parameter('vel_smoothing_alpha').value)

        self.dynamic_speed_thresh = float(self.get_parameter('dynamic_speed_thresh').value)
        self.ttc_thresh_front = float(self.get_parameter('ttc_thresh_front').value)
        self.lane_half_width = float(self.get_parameter('lane_half_width').value)
        self.min_vrel_x = float(self.get_parameter('min_vrel_x').value)

        self.max_range_considered = float(self.get_parameter('max_range_considered').value)

        # -------- State --------
        self.tracks: List[Track] = []
        self.next_track_id: int = 1

        self.robot_vx: float = 0.0  # lấy từ /cmd_vel_arbiter nếu có

        # -------- ROS I/O --------
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic,
                                                 self._cb_scan, qos)
        # không bắt buộc, nhưng nên có để TTC chính xác hơn
        self.sub_cmd = self.create_subscription(Twist, self.cmd_vel_topic,
                                                self._cb_cmd, 10)

        self.pub_dyn_front_unsafe = self.create_publisher(Bool, '/dyn_front_unsafe', 10)
        self.pub_dyn_front_ttc = self.create_publisher(Float32, '/dyn_front_ttc_min', 10)

        self.get_logger().info('DynamicObstacleTracker started.')

    # ---------- Helpers ----------
    def _now(self) -> float:
        sec, nsec = self.get_clock().now().seconds_nanoseconds()
        return sec + nsec * 1e-9

    def _cb_cmd(self, msg: Twist):
        # Giả sử trục x là hướng tiến của robot
        self.robot_vx = float(msg.linear.x)

    # ---------- Core callbacks ----------
    def _cb_scan(self, msg: LaserScan):
        tnow = self._now()

        clusters = self._build_clusters(msg)
        self._update_tracks(clusters, tnow)
        dyn_front_unsafe, ttc_min = self._evaluate_dynamic_front(tnow)

        # Publish kết quả
        self.pub_dyn_front_unsafe.publish(Bool(data=dyn_front_unsafe))
        self.pub_dyn_front_ttc.publish(Float32(data=float(ttc_min if ttc_min is not None else math.inf)))

    # ---------- Clustering ----------
    def _build_clusters(self, scan: LaserScan) -> List[Tuple[float, float, float]]:
        """
        Trả về list các cluster: (cx, cy, radius)
        """
        ranges = np.array(scan.ranges, dtype=np.float32)
        ang_min = scan.angle_min
        ang_inc = scan.angle_increment

        n = ranges.size
        angles = ang_min + np.arange(n, dtype=np.float32) * ang_inc

        # Chỉ lấy điểm hợp lệ + trong khoảng max_range_considered
        valid_mask = np.isfinite(ranges) & (ranges > 0.02) & (ranges < self.max_range_considered)
        if not np.any(valid_mask):
            return []

        idxs = np.nonzero(valid_mask)[0]
        r_valid = ranges[idxs]
        ang_valid = angles[idxs]

        xs = r_valid * np.cos(ang_valid)
        ys = r_valid * np.sin(ang_valid)

        clusters: List[Tuple[float, float, float]] = []
        current_points = []

        prev_x = None
        prev_y = None

        for x, y in zip(xs, ys):
            if prev_x is None:
                current_points = [(x, y)]
            else:
                dist = math.hypot(x - prev_x, y - prev_y)
                if dist <= self.cluster_dist_thresh:
                    current_points.append((x, y))
                else:
                    # Kết thúc cluster cũ
                    if len(current_points) >= self.cluster_min_points:
                        cx, cy, rad = self._cluster_stats(current_points)
                        clusters.append((cx, cy, rad))
                    current_points = [(x, y)]
            prev_x, prev_y = x, y

        # Cluster cuối
        if len(current_points) >= self.cluster_min_points:
            cx, cy, rad = self._cluster_stats(current_points)
            clusters.append((cx, cy, rad))

        return clusters

    @staticmethod
    def _cluster_stats(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        dists = np.hypot(xs - cx, ys - cy)
        radius = float(np.max(dists)) if dists.size > 0 else 0.0
        return cx, cy, radius

    # ---------- Tracking ----------
    def _update_tracks(self, clusters: List[Tuple[float, float, float]], tnow: float):
        """
        Gán cluster vào track (nearest neighbor) và cập nhật trạng thái.
        """
        # Dự đoán vị trí hiện tại của các track
        predicted_positions = []
        for tr in self.tracks:
            px, py = tr.predict(tnow)
            predicted_positions.append((tr, px, py))

        # Data association: greedy nearest neighbor
        used_tracks = set()
        used_clusters = set()

        for ci, (cx, cy, rad) in enumerate(clusters):
            best_tr = None
            best_dist = None

            for tr, px, py in predicted_positions:
                if tr.id in used_tracks:
                    continue
                dist = math.hypot(cx - px, cy - py)
                if dist <= self.assoc_dist_thresh and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_tr = tr

            if best_tr is not None:
                # Gán cluster này cho track
                best_tr.update(cx, cy, rad, tnow, alpha=self.vel_smoothing_alpha)
                used_tracks.add(best_tr.id)
                used_clusters.add(ci)

        # Tạo track mới cho cluster chưa gán
        for ci, (cx, cy, rad) in enumerate(clusters):
            if ci in used_clusters:
                continue
            new_tr = Track(self.next_track_id, cx, cy, tnow, rad)
            self.next_track_id += 1
            self.tracks.append(new_tr)

        # Xoá track quá cũ
        alive_tracks = []
        for tr in self.tracks:
            age = tnow - tr.last_update_t
            if age <= self.track_max_age_s:
                alive_tracks.append(tr)
        self.tracks = alive_tracks

    # ---------- Dynamic front evaluation ----------
    def _evaluate_dynamic_front(self, tnow: float) -> Tuple[bool, Optional[float]]:
        """
        Kiểm tra các track xem có vật cản ĐỘNG phía trước không.
        Trả về:
            (dyn_front_unsafe, ttc_min)
        """
        ttc_min: Optional[float] = None
        dyn_front_unsafe = False

        for tr in self.tracks:
            # Bỏ track phía sau robot
            if tr.x <= 0.0:
                continue

            # Bỏ track ngoài "làn đường" phía trước
            if abs(tr.y) > self.lane_half_width + tr.radius:
                continue

            # Tốc độ tuyệt đối của obstacle
            speed = math.hypot(tr.vx, tr.vy)
            if speed < self.dynamic_speed_thresh:
                continue  # coi như gần tĩnh

            # Vận tốc tương đối theo trục x (obstacle - robot)
            v_rel_x = tr.vx - self.robot_vx
            if v_rel_x <= self.min_vrel_x:
                # không tiến lại đủ nhanh
                continue

            distance_x = tr.x  # khoảng cách phía trước dọc trục x
            ttc = distance_x / max(v_rel_x, 1e-3)

            if ttc <= 0.0:
                continue

            # Cập nhật TTC nhỏ nhất
            if ttc_min is None or ttc < ttc_min:
                ttc_min = ttc

            if ttc < self.ttc_thresh_front:
                dyn_front_unsafe = True

        return dyn_front_unsafe, ttc_min


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
