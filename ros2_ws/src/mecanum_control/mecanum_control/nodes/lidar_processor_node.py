#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LidarProcessorNode (ROS2) — Refactored
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist

from mecanum_control.core.motion import ObstacleAvoidance

class LidarProcessorNode(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # -------- Parameters: topics --------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('publish_topic', '/cmd_vel_emergency')

        # -------- Safety distances --------
        self.declare_parameter('min_safe_distance', 0.60)
        self.declare_parameter('safety_zone_sides', 0.50)

        # -------- Angular sectors (deg) --------
        self.declare_parameter('angle_range_front', 90.0)
        self.declare_parameter('angle_range_sides', 60.0)
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
        self.declare_parameter('target_distance_m', 1.6)

        # -------- Read parameters --------
        self.scan_topic = self.get_parameter('scan_topic').value
        self.pub_topic  = self.get_parameter('publish_topic').value

        config = {
            'min_safe_distance': float(self.get_parameter('min_safe_distance').value),
            'safety_zone_sides': float(self.get_parameter('safety_zone_sides').value),
            'angle_range_front': float(self.get_parameter('angle_range_front').value),
            'angle_range_sides': float(self.get_parameter('angle_range_sides').value),
            'yaw_offset_deg': float(self.get_parameter('yaw_offset_deg').value),
            'emergency_vy': float(self.get_parameter('emergency_vy').value),
            'bypass_min_time_s': float(self.get_parameter('bypass_min_time_s').value),
            'bypass_release_hysteresis_m': float(self.get_parameter('bypass_release_hysteresis_m').value),
            'bypass_timeout_s': float(self.get_parameter('bypass_timeout_s').value),
            'release_on_clear_immediate': bool(self.get_parameter('release_on_clear_immediate').value),
            'clear_debounce_s': float(self.get_parameter('clear_debounce_s').value),
            'hold_centering_after_release_s': float(self.get_parameter('hold_centering_after_release_s').value),
            'instant_stop_on_clear': bool(self.get_parameter('instant_stop_on_clear').value),
            'gate_mode': str(self.get_parameter('gate_mode').value),
            'vy_slew_rate': float(self.get_parameter('vy_slew_rate').value),
            'enable_corridor_centering': bool(self.get_parameter('enable_corridor_centering').value),
            'center_k': float(self.get_parameter('center_k').value),
            'center_vy_cap': float(self.get_parameter('center_vy_cap').value),
            'suppress_when_target_close': bool(self.get_parameter('suppress_when_target_close').value),
            'suppress_threshold_m': float(self.get_parameter('suppress_threshold_m').value),
            'person_ignore_enable': bool(self.get_parameter('person_ignore_enable').value),
            'person_ignore_margin_m': float(self.get_parameter('person_ignore_margin_m').value),
            'person_ignore_only_when_locked': bool(self.get_parameter('person_ignore_only_when_locked').value),
            'side_trigger_enable': bool(self.get_parameter('side_trigger_enable').value),
            'side_soft_avoid': bool(self.get_parameter('side_soft_avoid').value),
            'side_soft_gain': float(self.get_parameter('side_soft_gain').value),
            'target_distance_m': float(self.get_parameter('target_distance_m').value),
        }
        
        self.obstacle_avoidance = ObstacleAvoidance(config, logger=self.get_logger())

        # -------- Obstacle warning sound --------
        HERE = Path(__file__).resolve().parent
        self.obstacle_sound_file = str(HERE.parent / "sounds" / "warn_VatCan_viet.wav")

        # -------- State --------
        self.is_locked: bool = False
        self.person_dist: Optional[float] = None
        self.person_centered: bool = False
        self.dynamic_front_unsafe: bool = False

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
        self.is_locked = (msg.data == 'LOCKED')

    def _cb_pd(self, msg: Float32):
        try:
            self.person_dist = float(msg.data)
        except Exception:
            self.person_dist = None

    def _cb_centered(self, msg: Bool):
        self.person_centered = bool(msg.data)

    def _cb_dyn_flag(self, msg: Bool):
        self.dynamic_front_unsafe = bool(msg.data)

    def _cb_dyn_ttc(self, msg: Float32):
        pass # Not used in logic yet, but subscribed

    def _now(self) -> float:
        sec, nsec = self.get_clock().now().seconds_nanoseconds()
        return sec + nsec * 1e-9

    def _cb_scan(self, msg: LaserScan):
        tnow = self._now()

        # Update core state
        self.obstacle_avoidance.update_external_state(
            self.is_locked, self.person_centered, self.person_dist, self.dynamic_front_unsafe
        )

        # Chưa LOCKED: không né, luôn safe
        if not self.is_locked:
            self._publish(0.0, 0.0)
            self.pub_unsafe.publish(Bool(data=False))
            self.pub_safe.publish(Bool(data=True))
            # Reset internal state if needed, but ObstacleAvoidance handles it mostly via logic
            # Actually ObstacleAvoidance logic depends on is_locked.
            # But we should probably call decide even if not locked to reset state?
            # Original code returns early.
            # We should probably respect original logic: return early.
            # But we need to reset bypass state?
            # Original code:
            # self._last_front_clear_t = None
            # self.bypass_active = False
            # ...
            # So I should expose a reset method or just manually reset.
            # Or I can just call decide and let it handle it?
            # No, decide logic is complex.
            # I will add a reset method to ObstacleAvoidance or just access members.
            # Accessing members is fine for now.
            self.obstacle_avoidance.bypass_active = False
            self.obstacle_avoidance.bypass_dir = 0
            self.obstacle_avoidance.prev_vy_cmd = 0.0
            self.obstacle_avoidance._last_front_clear_t = None
            return

        emergency, vy, wz, unsafe_any, front_unsafe, should_play_sound = self.obstacle_avoidance.decide_from_scan(msg, tnow)

        if should_play_sound:
            if os.path.exists(self.obstacle_sound_file):
                os.system(f"aplay {self.obstacle_sound_file} &")
                self.get_logger().info("Playing obstacle warning sound")

        if self.obstacle_avoidance.gate_mode == 'front_only':
            safe = not front_unsafe
        else:
            safe = not unsafe_any

        self.pub_unsafe.publish(Bool(data=not safe))
        self.pub_safe.publish(Bool(data=safe))
        self._publish(vy, wz)

    def _publish(self, vy: float, wz: float):
        tw = Twist()
        tw.linear.y  = float(vy)
        tw.angular.z = float(wz)
        self.pub_tw.publish(tw)

def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
