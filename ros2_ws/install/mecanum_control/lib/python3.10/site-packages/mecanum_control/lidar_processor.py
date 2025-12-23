#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR Processor (ROS2)

- Né ngang (không xoay), có bypass latch + strict release.
- Mask theo khoảng cách người để không coi người là vật cản phía trước.
- DÙNG TRẠNG THÁI BÁM NGƯỜI:
    * Nếu đã LOCKED + person_centered == True + person_distance <= target_distance_m
      → KHÔNG né vật cản (vy=0, safe) TRỪ KHI có vật cản động phía trước.
    * Nếu chưa centered (person_centered == False) hoặc xa hơn target → né bình thường.
- Vẫn giữ quy tắc:
    * Hai bên đều cản, trước trống → đi thẳng (vy=0, safe).
    * Ba phía đều cản → dừng (vy=0, unsafe).

- BỔ SUNG:
    * Nhận thông tin né vật cản động từ node dynamic_obstacle_tracker:
        - /dyn_front_unsafe (Bool)
        - /dyn_front_ttc_min (Float32)
    * Nếu dynamic_front_unsafe == True → coi như front_unsafe.
"""

from typing import Optional, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist


def clamp(x, a, b):
    return a if x < a else b if x > b else x


class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # -------- Parameters: topics --------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('publish_topic', '/cmd_vel_emergency')

        # -------- Safety distances --------
        self.declare_parameter('min_safe_distance', 0.60)   # front
        self.declare_parameter('safety_zone_sides', 0.50)   # sides

        # -------- Angular sectors (deg) --------
        self.declare_parameter('angle_range_front', 90.0)   # +/- 45°
        self.declare_parameter('angle_range_sides', 60.0)   # +/- 30° quanh ±90°
        self.declare_parameter('yaw_offset_deg', 0.0)

        # -------- Lateral avoidance speed --------
        self.declare_parameter('emergency_vy', 0.22)

        # -------- Bypass latch --------
        self.declare_parameter('bypass_min_time_s', 0.40)
        self.declare_parameter('bypass_release_hysteresis_m', 0.05)
        self.declare_parameter('bypass_timeout_s', 3.0)

        # -------- Strict release when clear --------
        self.declare_parameter('release_on_clear_immediate', True)
        self.declare_parameter('clear_debounce_s', 0.02)
        self.declare_parameter('hold_centering_after_release_s', 0.10)
        self.declare_parameter('instant_stop_on_clear', True)

        # -------- SAFE gating mode --------
        self.declare_parameter('gate_mode', 'front_only')

        # -------- Slew & corridor centering --------
        self.declare_parameter('vy_slew_rate', 1.8)
        self.declare_parameter('enable_corridor_centering', True)
        self.declare_parameter('center_k', 0.6)
        self.declare_parameter('center_vy_cap', 0.20)

        # -------- Suppress avoid khi rất gần người --------
        self.declare_parameter('suppress_when_target_close', True)
        self.declare_parameter('suppress_threshold_m', 0.90)

        # -------- Person-distance masking --------
        self.declare_parameter('person_ignore_enable', True)
        self.declare_parameter('person_ignore_margin_m', 0.40)
        self.declare_parameter('person_ignore_only_when_locked', True)
        self.declare_parameter('person_ignore_mode', 'distance_only')
        self.declare_parameter('person_ignore_deg', 60.0)

        # -------- Side-trigger avoid --------
        self.declare_parameter('side_trigger_enable', True)
        self.declare_parameter('side_soft_avoid', True)
        self.declare_parameter('side_soft_gain', 0.18)

        # -------- Target-based gating --------
        # Nếu LOCKED + person_centered == True + person_distance <= target_distance_m
        # → KHÔNG né vật cản nữa (trừ khi có obstacle động phía trước).
        self.declare_parameter('target_distance_m', 1.6)

        # -------- Read parameters --------
        self.scan_topic = self.get_parameter('scan_topic').value
        self.pub_topic  = self.get_parameter('publish_topic').value

        self.min_front = float(self.get_parameter('min_safe_distance').value)
        self.min_side  = float(self.get_parameter('safety_zone_sides').value)
        self.fov_front = float(self.get_parameter('angle_range_front').value)
        self.fov_sides = float(self.get_parameter('angle_range_sides').value)
        self.yaw_off_deg = float(self.get_parameter('yaw_offset_deg').value)

        self.emergency_vy = float(self.get_parameter('emergency_vy').value)

        self.bypass_min_time_s   = float(self.get_parameter('bypass_min_time_s').value)
        self.bypass_release_hyst = float(self.get_parameter('bypass_release_hysteresis_m').value)
        self.bypass_timeout_s    = float(self.get_parameter('bypass_timeout_s').value)

        self.release_on_clear_immediate = bool(self.get_parameter('release_on_clear_immediate').value)
        self.clear_debounce_s           = float(self.get_parameter('clear_debounce_s').value)
        self.hold_center_after_rel_s    = float(self.get_parameter('hold_centering_after_release_s').value)
        self.instant_stop_on_clear      = bool(self.get_parameter('instant_stop_on_clear').value)

        self.gate_mode    = str(self.get_parameter('gate_mode').value)
        self.vy_slew_rate = float(self.get_parameter('vy_slew_rate').value)

        self.enable_center  = bool(self.get_parameter('enable_corridor_centering').value)
        self.center_k       = float(self.get_parameter('center_k').value)
        self.center_vy_cap  = float(self.get_parameter('center_vy_cap').value)

        self.suppress_close = bool(self.get_parameter('suppress_when_target_close').value)
        self.suppress_thr   = float(self.get_parameter('suppress_threshold_m').value)

        self.person_ignore_enable       = bool(self.get_parameter('person_ignore_enable').value)
        self.person_ignore_margin       = float(self.get_parameter('person_ignore_margin_m').value)
        self.person_ignore_only_locked  = bool(self.get_parameter('person_ignore_only_when_locked').value)
        self.person_ignore_mode         = str(self.get_parameter('person_ignore_mode').value)
        self.person_ignore_deg          = float(self.get_parameter('person_ignore_deg').value)

        self.side_trigger_enable = bool(self.get_parameter('side_trigger_enable').value)
        self.side_soft_avoid     = bool(self.get_parameter('side_soft_avoid').value)
        self.side_soft_gain      = float(self.get_parameter('side_soft_gain').value)

        self.target_distance_m = float(self.get_parameter('target_distance_m').value)

        # -------- State --------
        self.is_locked: bool = False          # từ PersonDetector
        self.follow_state: str = 'SEARCHING'  # "LOCKED" / "SEARCHING"
        self.person_dist: Optional[float] = None
        self.person_centered: bool = False    # NEW: từ /person_centered

        self.bypass_active: bool = False
        self.bypass_dir: int = 0
        self.bypass_start_t: float = 0.0

        self.prev_time: Optional[float] = None
        self.prev_vy_cmd: float = 0.0

        self._last_front_clear_t: Optional[float] = None
        self._last_release_t: Optional[float] = None

        # -------- Dynamic obstacle state --------
        self.dynamic_front_unsafe: bool = False
        self.dynamic_front_ttc: Optional[float] = None

        # -------- ROS I/O --------
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.sub_laser = self.create_subscription(LaserScan, self.scan_topic, self._cb_scan, qos)
        self.sub_state = self.create_subscription(String, '/person_detector/follow_state', self._cb_state, 10)
        self.sub_pd    = self.create_subscription(Float32, '/person_distance', self._cb_pd, 10)
        self.sub_centered = self.create_subscription(Bool, '/person_centered', self._cb_centered, 10)

        # Sub thông tin vật cản động
        self.sub_dyn_flag = self.create_subscription(Bool, '/dyn_front_unsafe', self._cb_dyn_flag, 10)
        self.sub_dyn_ttc  = self.create_subscription(Float32, '/dyn_front_ttc_min', self._cb_dyn_ttc, 10)

        self.pub_tw     = self.create_publisher(Twist, self.pub_topic, 10)
        self.pub_unsafe = self.create_publisher(Bool, '/lidar_unsafe', 10)
        self.pub_safe   = self.create_publisher(Bool, '/safe_to_move', 10)

    # ---------- Callbacks ----------
    def _cb_state(self, msg: String):
        self.follow_state = msg.data
        self.is_locked = (msg.data == 'LOCKED')

    def _cb_pd(self, msg: Float32):
        try:
            self.person_dist = float(msg.data)
        except Exception:
            self.person_dist = None

    def _cb_centered(self, msg: Bool):
        self.person_centered = bool(msg.data)

    def _cb_dyn_flag(self, msg: Bool):
        # Cờ vật cản động phía trước (từ dynamic_obstacle_tracker)
        self.dynamic_front_unsafe = bool(msg.data)

    def _cb_dyn_ttc(self, msg: Float32):
        try:
            self.dynamic_front_ttc = float(msg.data)
        except Exception:
            self.dynamic_front_ttc = None

    def _now(self) -> float:
        sec, nsec = self.get_clock().now().seconds_nanoseconds()
        return sec + nsec * 1e-9

    def _cb_scan(self, msg: LaserScan):
        tnow = self._now()

        # Chưa LOCKED: không né, luôn safe
        if not self.is_locked:
            self._publish(0.0, 0.0)
            self.pub_unsafe.publish(Bool(data=False))
            self.pub_safe.publish(Bool(data=True))
            self._last_front_clear_t = None
            self.bypass_active = False
            self.bypass_dir = 0
            self.prev_vy_cmd = 0.0
            return

        emergency, vy, wz, unsafe_any, front_unsafe = self._decide_from_scan(msg, tnow)

        if self.gate_mode == 'front_only':
            safe = not front_unsafe
        else:
            safe = not unsafe_any

        self.pub_unsafe.publish(Bool(data=not safe))
        self.pub_safe.publish(Bool(data=safe))
        self._publish(vy, wz)

    # ---------- Helpers ----------
    def _publish(self, vy: float, wz: float):
        tw = Twist()
        tw.linear.y  = float(vy)
        tw.angular.z = float(wz)
        self.pub_tw.publish(tw)

    def _slew(self, desired_vy: float, tnow: float) -> float:
        if self.prev_time is None:
            self.prev_time = tnow
            self.prev_vy_cmd = desired_vy
            return desired_vy

        dt = max(1e-3, tnow - self.prev_time)
        max_delta = self.vy_slew_rate * dt
        vy = self.prev_vy_cmd

        if desired_vy > vy + max_delta:
            vy += max_delta
        elif desired_vy < vy - max_delta:
            vy -= max_delta
        else:
            vy = desired_vy

        self.prev_time = tnow
        self.prev_vy_cmd = vy
        return vy

    def _is_centered_and_near_target(self) -> bool:
        """
        TRUE nếu:
        - đã LOCKED người
        - /person_centered == True (PersonDetector báo đang giữa màn hình)
        - person_distance <= target_distance_m
        => coi là "đang trong khoảng target & đã ở giữa" → KHÔNG né.

        BỔ SUNG:
        - Nếu có dynamic_front_unsafe == True → KHÔNG được bỏ qua né, trả về False.
        """
        # Nếu phía trước đang có obstacle động nguy hiểm thì KHÔNG cho phép bỏ qua né
        if self.dynamic_front_unsafe:
            return False

        if not self.is_locked:
            return False
        if not self.person_centered:
            return False
        if self.person_dist is None:
            return False
        return (self.person_dist <= self.target_distance_m)

    # ---------- Core decision ----------
    def _decide_from_scan(self, scan: LaserScan, tnow: float) -> Tuple[bool, float, float, bool, bool]:
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        n = ranges.size
        ang_min = scan.angle_min
        ang_inc = scan.angle_increment
        yaw_off = np.deg2rad(self.yaw_off_deg)

        def angle_to_idx(deg):
            rad = np.deg2rad(deg) + yaw_off
            idx = int(np.round((rad - ang_min) / ang_inc))
            return int(clamp(idx, 0, n - 1))

        # FRONT sector
        half_f = self.fov_front / 2.0
        i0 = angle_to_idx(-half_f)
        i1 = angle_to_idx(+half_f)
        lo_f = min(i0, i1)
        hi_f = max(i0, i1)
        front = ranges[lo_f:hi_f + 1].copy()

        # Mask người theo khoảng cách
        if self.person_ignore_enable and (self.person_dist is not None):
            if (not self.person_ignore_only_locked) or self.is_locked:
                d = float(self.person_dist)
                m = float(self.person_ignore_margin)
                if front.size > 0:
                    mask = np.abs(front - d) <= m
                    front[mask] = np.inf

        # SIDE sectors
        half_s = self.fov_sides / 2.0
        iL0 = angle_to_idx(90 - half_s)
        iL1 = angle_to_idx(90 + half_s)
        left = ranges[min(iL0, iL1):max(iL0, iL1) + 1]

        iR0 = angle_to_idx(-90 - half_s)
        iR1 = angle_to_idx(-90 + half_s)
        right = ranges[min(iR0, iR1):max(iR0, iR1) + 1]

        front_min = float(np.min(front)) if front.size else np.inf
        left_min  = float(np.min(left))  if left.size  else np.inf
        right_min = float(np.min(right)) if right.size else np.inf
        left_avg  = float(np.mean(left)) if left.size  else np.inf
        right_avg = float(np.mean(right)) if right.size else np.inf

        # Kết hợp obstacle tĩnh + động
        front_unsafe = (front_min < self.min_front) or self.dynamic_front_unsafe
        left_unsafe  = (left_min  < self.min_side)
        right_unsafe = (right_min < self.min_side)
        side_unsafe  = (left_unsafe or right_unsafe)
        side_both_unsafe = (left_unsafe and right_unsafe)
        unsafe_any   = bool(front_unsafe or side_unsafe)

        # Track front-clear time (chỉ theo khoảng cách tĩnh)
        if not (front_min < self.min_front):
            if self._last_front_clear_t is None:
                self._last_front_clear_t = tnow
        else:
            self._last_front_clear_t = None

        # ===== GATING THEO YÊU CẦU =====
        # Nhỏ hơn target & đã ở giữa màn hình -> KHÔNG NÉ
        # (nhưng đã được chặn dynamic_front_unsafe trong _is_centered_and_near_target)
        if self._is_centered_and_near_target():
            self.bypass_active = False
            self.bypass_dir = 0
            self.prev_vy_cmd = 0.0
            vy_cmd = 0.0
            return False, vy_cmd, 0.0, False, False

        # ===== Suppress avoid khi rất gần người (backup an toàn) =====
        if self.suppress_close and (self.person_dist is not None) and (self.person_dist <= self.suppress_thr):
            self.bypass_active = False
            self.bypass_dir = 0
            vy_cmd = self._slew(0.0, tnow)
            return False, vy_cmd, 0.0, unsafe_any, front_unsafe

        # ===== Quy tắc đặc biệt 2 bên / 3 phía =====
        if side_both_unsafe and not front_unsafe:
            self.bypass_active = False
            self.bypass_dir = 0
            self.prev_vy_cmd = 0.0
            return False, 0.0, 0.0, False, False

        if side_both_unsafe and front_unsafe:
            self.bypass_active = False
            self.bypass_dir = 0
            self.prev_vy_cmd = 0.0
            self.get_logger().debug('Three-side block (including dynamic front) → stop')
            return False, 0.0, 0.0, True, True

        # ===== Front unsafe -> kích bypass =====
        if front_unsafe and not self.bypass_active:
            # né sang bên có không gian rộng hơn
            self.bypass_dir = +1 if left_min > right_min else -1
            self.bypass_active = True
            self.bypass_start_t = tnow

        # ===== Strict release khi trước thoáng (theo khoảng cách) =====
        if self.bypass_active and self.release_on_clear_immediate and (not (front_min < self.min_front)):
            if (self._last_front_clear_t is not None) and \
               ((tnow - self._last_front_clear_t) >= self.clear_debounce_s):
                self.bypass_active = False
                self.bypass_dir = 0
                self.prev_vy_cmd = 0.0
                self._last_release_t = tnow
                if self.instant_stop_on_clear and not side_unsafe:
                    return False, 0.0, 0.0, unsafe_any, front_unsafe

        # ===== Side unsafe & front safe -> né mềm =====
        trigger_side = (self.side_trigger_enable and side_unsafe and not front_unsafe and not side_both_unsafe)
        if trigger_side and self.side_soft_avoid:
            dir_away = +1 if left_min > right_min else -1
            vy_cmd = self._slew(self.side_soft_gain * float(dir_away), tnow)
            return True, vy_cmd, 0.0, unsafe_any, front_unsafe

        # ===== Duy trì / nhả bypass =====
        if self.bypass_active:
            held_enough = (tnow - self.bypass_start_t) >= self.bypass_min_time_s
            clear_enough = (front_min >= (self.min_front + self.bypass_release_hyst))
            timeout = (tnow - self.bypass_start_t) >= self.bypass_timeout_s

            if (held_enough and clear_enough) or timeout:
                self.bypass_active = False
                self.bypass_dir = 0
                self.prev_vy_cmd = 0.0
                self._last_release_t = tnow

        if self.bypass_active:
            desired_vy = self.emergency_vy * float(self.bypass_dir)
            vy_cmd = self._slew(desired_vy, tnow)
            return True, vy_cmd, 0.0, unsafe_any, front_unsafe

        # ===== Vừa nhả: tạm tắt căn giữa =====
        if self._last_release_t is not None and (tnow - self._last_release_t) < self.hold_center_after_rel_s:
            vy_cmd = 0.0
            return False, vy_cmd, 0.0, unsafe_any, front_unsafe

        # ===== Corridor centering =====
        if self.enable_center and np.isfinite(left_avg) and np.isfinite(right_avg):
            err_c = (right_avg - left_avg)
            desired_vy = clamp(self.center_k * err_c, -self.center_vy_cap, self.center_vy_cap)
            vy_cmd = self._slew(desired_vy, tnow)
            return False, vy_cmd, 0.0, unsafe_any, front_unsafe

        # ===== Default: không strafe =====
        vy_cmd = self._slew(0.0, tnow)
        return False, vy_cmd, 0.0, unsafe_any, front_unsafe


def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
