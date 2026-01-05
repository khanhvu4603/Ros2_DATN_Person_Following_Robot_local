#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PersonDetectorNode (ROS2) — Refactored
"""

import time
import socket
import os
import threading
from typing import Optional, List, Tuple
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# Core Modules
from mecanum_control.core.detector import (
    load_ssd, ssd_detect, iou, center_of, clip_box, enhanced_body_feature, clamp
)
from mecanum_control.core.tracker import PersonTracker
from mecanum_control.core.visualizer import draw_labeled_box, draw_label_top_right
from mecanum_control.core.audio import AudioManager

# ========== Paths ==========
HERE = Path(__file__).resolve().parent
# nodes/ -> mecanum_control/ -> models/
MODELS = HERE.parent / "models"
DATA = HERE.parent / "data"

MB2_ONNX_PATH  = str(MODELS / "mb2_gap.onnx")
MOBILENET_PROTOTXT = str(MODELS / "MobileNetSSD_deploy.prototxt")
MOBILENET_WEIGHTS  = str(MODELS / "MobileNetSSD_deploy.caffemodel")

class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector')

        self.declare_parameters('', [
            ('camera_topic', '/camera/d455/color/image_raw'),
            ('publish_debug_image', True),
            ('image_width', 640), ('image_height', 480),

            # Depth follow
            ('use_depth', True),
            ('depth_topic', '/camera/d455/depth/image_rect_raw'),
            ('depth_encoding', '16UC1'),
            ('target_distance_m', 2),
            ('kd_distance', 0.6),
            ('v_forward_max', 0.3),

            # Heading control
            ('kx_center', 0.00025),
            ('wz_max', 0.25),
            ('center_deadband_px', 40),
            ('center_release_px', 60),
            ('center_first', True),

            # Detector/ReID thresholds
            ('person_conf', 0.35),
            ('accept_threshold', 0.73),
            ('reject_threshold', 0.63),
            ('iou_threshold', 0.4),
            ('margin_delta', 0.07),
            ('confirm_frames', 5),

            # Color weight
            ('body_color_weight', 0.25),
            ('body_color_weight_min', 0.10),
            ('body_color_weight_lowlight_scale', 0.6),
            ('hsv_normalize_brightness', True),
            ('similarity_ema_alpha', 0.8),
            
            # Auto-enroll
            ('auto_timeout_sec', 30.0),
            ('auto_body_min', 30),
            ('auto_body_target', 100),

            # Models
            ('mb2_onnx_path', MB2_ONNX_PATH),

            # Occlusion & Lost handling
            ('occlusion_threshold', 0.45),
            ('grace_period_sec', 3.0),

            # UDP Streaming
            ('enable_udp_stream', True),
            ('udp_host', '127.0.0.1'),
            ('udp_port', 9999),
            
            # Sound
            ('sound_filename', 'lost_target_viet.wav'),
            ('enroll_sound_filename', 'enroll_viet.wav'),
            ('run_sound_filename', 'run_viet.wav'),
            
            # === ANTI-ID-SWITCHING PARAMETERS ===
            ('depth_filter_margin', 0.5),
            ('overlap_iou_thr', 0.20),
            ('overlap_depth_margin', 0.3),
            ('depth_jump_threshold', 0.6),
            ('track_switch_margin', 0.2),
            ('pre_filter_appearance_thr', 0.70),
            
            # DeepSORT params passed to Tracker
            ('max_age', 60),
            ('n_init', 5),
            ('max_cosine_distance', 0.08),
            ('lambda_weight', 0.85),
        ])

        # QoS
        color_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        depth_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Bridge & subs
        self.bridge = CvBridge()
        cam_topic = self.get_parameter('camera_topic').value
        self.create_subscription(Image, cam_topic, self.on_image, color_qos)

        self.depth_img = None
        self.depth_enc = None
        if bool(self.get_parameter('use_depth').value):
            self.create_subscription(Image, self.get_parameter('depth_topic').value, self.on_depth, depth_qos)

        # Publishers
        self.cmd_pub       = self.create_publisher(Twist,  '/cmd_vel_person', 10)
        self.flag_pub      = self.create_publisher(Bool,   '/person_detected', 10)
        self.debug_pub     = self.create_publisher(Image,  '/person_detector/debug_image', 1)
        self.state_pub     = self.create_publisher(String, '/person_detector/follow_state', 10)
        self.dist_depth_pub= self.create_publisher(Float32,'/person_distance', 10)
        self.centered_pub  = self.create_publisher(Bool,   '/person_centered', 10)

        # ONNX Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.mb2_sess = ort.InferenceSession(
            self.get_parameter('mb2_onnx_path').value,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Detector
        self.ssd_net = load_ssd(MOBILENET_PROTOTXT, MOBILENET_WEIGHTS)
        if self.ssd_net is None:
            raise FileNotFoundError(
                "Không tìm thấy MobileNet-SSD Caffe. Hãy đặt file vào:\n"
                f"  - {MOBILENET_PROTOTXT}\n"
                f"  - {MOBILENET_WEIGHTS}"
            )
        self.get_logger().info("Person detector: MobileNet-SSD (Caffe)")
        
        # Tracker Config
        tracker_config = {
            'max_age': self.get_parameter('max_age').value,
            'n_init': self.get_parameter('n_init').value,
            'max_cosine_distance': self.get_parameter('max_cosine_distance').value,
            'lambda_weight': self.get_parameter('lambda_weight').value,
            'accept_threshold': self.get_parameter('accept_threshold').value,
            'track_switch_margin': self.get_parameter('track_switch_margin').value,
        }
        self.tracker = PersonTracker(tracker_config, self.mb2_sess, logger=self.get_logger())
        
        # Audio
        self.audio = AudioManager(logger=self.get_logger())
        self.sound_file = str(HERE.parent / "sounds" / self.get_parameter('sound_filename').value)
        self.enroll_sound_file = str(HERE.parent / "sounds" / self.get_parameter('enroll_sound_filename').value)
        self.run_sound_file = str(HERE.parent / "sounds" / self.get_parameter('run_sound_filename').value)
        self.enroll_audio_played = False
        self.run_audio_played = False

        # --- STATE MACHINE VARIABLES ---
        self.state = 'AUTO-ENROLL'
        self.target_box = None
        self.last_known_depth = None
        self.lost_start_time = None
        self.current_similarity = 0.0
        
        # --- Occlusion State Machine counters ---
        self.miss_count = 0
        self.occl_start_time = None
        self.recover_count = 0
        
        # --- Occlusion/Recover thresholds ---
        self.MISS_TO_SEARCH = 12
        self.OCCL_MAX_SEC = 3.0
        self.RECOVER_CONFIRM = 3
        self.RECOVER_THR = 0.74
        self.RECOVER_DEPTH_THR = 0.6
        self.RECOVER_TIMEOUT = 3.0
        
        # --- OPTIMIZATION VARIABLES ---
        self.frame_count = 0

        # --- ENROLLMENT ---
        self.body_centroid = None
        self.body_samples: List[np.ndarray] = []
        self.auto_start_ts = None
        self.auto_done = False

        # --- CONTROL ---
        self._is_centered = False
        self._dynamic_color_weight = float(self.get_parameter('body_color_weight').value)
        
        # --- DEPTH EMA FILTER ---
        self.depth_ema = None
        self.depth_ema_alpha = 0.3

        # --- ADAPTIVE MODEL UPDATE ---
        self.last_update_time = 0.0
        self.adaptive_update_interval_sec = 1.5
        self.update_streak = 0
        self.UPDATE_STREAK_N = 3

        # UDP Streaming
        self.enable_udp = bool(self.get_parameter('enable_udp_stream').value)
        if self.enable_udp:
            self.udp_host = self.get_parameter('udp_host').value
            self.udp_port = int(self.get_parameter('udp_port').value)
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.get_logger().info(f"UDP Streaming enabled to {self.udp_host}:{self.udp_port}")

    # ---------- Depth ----------
    def on_depth(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.get_parameter('depth_encoding').value)
        self.depth_enc = self.get_parameter('depth_encoding').value

    def get_median_depth_at_box(self, box, depth_img):
        """Lấy giá trị depth trung vị tại một bounding box."""
        if depth_img is None or box is None:
            return None
        
        box = clip_box(box, depth_img.shape)
        if box is None:
            return None
            
        x1, y1, x2, y2 = box
        roi = depth_img[y1:y2, x1:x2]
        if roi.size == 0: return None
        
        valid_pixels = roi[(roi > 100) & (roi < 10000)]
        if valid_pixels.size == 0: return None
            
        depth_m = np.percentile(valid_pixels, 10) / 1000.0
        return float(depth_m)

    def is_target_occluded(self, target_box, depth_img, last_known_depth):
        """Kiểm tra target có bị che khuất hay không."""
        if depth_img is None or target_box is None or last_known_depth is None:
            return False
            
        current_depth = self.get_median_depth_at_box(target_box, depth_img)
        if current_depth is None: return False
            
        threshold = self.get_parameter('occlusion_threshold').value
        if current_depth < (last_known_depth - threshold):
            return True
        return False

    # ---------- Auto-enroll ----------
    def auto_enroll_step(self, frame, pboxes, depth_frame):
        now = time.time()
        if self.auto_start_ts is None:
            self.auto_start_ts = now

        if pboxes:
            j = int(np.argmax([(pb[2]-pb[0])*(pb[3]-pb[1]) for pb in pboxes]))
            pb = pboxes[j]
            feat = enhanced_body_feature(frame, pb, depth_frame, self.mb2_sess,
                                       color_weight=self._dynamic_color_weight)
            if feat is not None:
                self.body_samples.append(feat)
                if self.body_centroid is None:
                    self.body_centroid = feat.copy()
                else:
                    self.body_centroid = 0.9*self.body_centroid + 0.1*feat
                    self.body_centroid /= (np.linalg.norm(self.body_centroid)+1e-8)

        timeout = float(self.get_parameter('auto_timeout_sec').value)
        body_target = int(self.get_parameter('auto_body_target').value)
        if (now - self.auto_start_ts) >= timeout or len(self.body_samples) >= body_target:
            if self.body_centroid is None:
                self.get_logger().warn("AUTO-ENROLL failed: no valid features -> restart enroll")
                self.auto_start_ts = now
                self.body_samples.clear()
                self.body_centroid = None
                return

            # thành công
            self.tracker.target_feature = self.body_centroid.copy()
            self.tracker.original_target_feature = self.body_centroid.copy()
            self.get_logger().info("Target enrolled. ANCHOR feature saved.")
            self.auto_done = True
            self.state = 'SEARCHING'
            
            if not self.run_audio_played:
                self.audio.play_sound_async(self.run_sound_file, repeat=2)
                self.run_audio_played = True

    # ---------- Control ----------
    def compute_cmd(self, frame_w, frame_h, target_box, depth_frame):
        twist = Twist()
        detected = Bool(); detected.data = (target_box is not None)

        if target_box is None:
            self._is_centered = False
            self.depth_ema = None
            return twist, detected, None

        cx, _ = center_of(target_box)
        err_px = (cx - frame_w*0.5)
        dead = float(self.get_parameter('center_deadband_px').value)
        rel  = float(self.get_parameter('center_release_px').value)
        center_first = bool(self.get_parameter('center_first').value)

        if not self._is_centered:
            if abs(err_px) <= dead:
                self._is_centered = True
        else:
            if abs(err_px) > max(rel, dead):
                self._is_centered = False

        err_eff = 0.0 if abs(err_px) <= dead else (np.sign(err_px) * (abs(err_px)-dead))
        kx = float(self.get_parameter('kx_center').value)
        wz = clamp(-kx*err_eff, -float(self.get_parameter('wz_max').value),
                                +float(self.get_parameter('wz_max').value))

        depth_raw = self.get_median_depth_at_box(target_box, depth_frame)
        
        if depth_raw is not None:
            if self.depth_ema is None:
                self.depth_ema = depth_raw
            else:
                self.depth_ema = self.depth_ema_alpha * depth_raw + (1 - self.depth_ema_alpha) * self.depth_ema
        
        depth_m = self.depth_ema
        
        vx = 0.0
        if depth_m is not None:
            kd = float(self.get_parameter('kd_distance').value)
            d_des = float(self.get_parameter('target_distance_m').value)
            err_d = depth_m - d_des
            if (not center_first) or self._is_centered:
                if err_d > 0.0:
                    vx = clamp(kd * err_d, 0.0, float(self.get_parameter('v_forward_max').value))
                else:
                    vx = 0.0

        twist.linear.x = float(vx)
        twist.angular.z = float(wz)
        return twist, detected, depth_m

    # ---------- Proactive Occlusion Check ----------
    def detect_potential_occlusion(self, target_box, detections, depth_frame):
        if target_box is None or depth_frame is None or self.last_known_depth is None:
            return False
        
        target_depth = self.last_known_depth
        tx1, ty1, tx2, ty2 = target_box
        
        APPROACH_DEPTH_MARGIN = 0.6
        APPROACH_HORIZONTAL_OVERLAP = 0.3
        
        for det_box in detections:
            det_depth = self.get_median_depth_at_box(det_box, depth_frame)
            if det_depth is None:
                continue
            
            if (target_depth - det_depth) < APPROACH_DEPTH_MARGIN:
                continue
            
            dx1, dy1, dx2, dy2 = det_box
            overlap_left = max(tx1, dx1)
            overlap_right = min(tx2, dx2)
            
            if overlap_right > overlap_left:
                overlap_width = overlap_right - overlap_left
                target_width = tx2 - tx1
                overlap_ratio = overlap_width / target_width
                
                if overlap_ratio > APPROACH_HORIZONTAL_OVERLAP:
                    self.get_logger().warn(
                        f"⚠️ POTENTIAL OCCLUSION: Intruder at {det_depth:.2f}m "
                        f"(target at {target_depth:.2f}m), overlap={overlap_ratio:.2%}"
                    )
                    return True
        return False

    # ---------- Debug Publisher ----------
    def publish_debug(self, frame, pboxes, target_box, vmean, depth_m):
        publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        if not publish_debug_image and not self.enable_udp:
            return

        dbg = frame.copy()

        if pboxes is not None:
            for box in pboxes:
                if target_box is not None and iou(box, target_box) > 0.5:
                    continue
                draw_labeled_box(dbg, box, color=(0,255,0), label="")

        if self.tracker.current_track_id is not None:
            track = self.tracker.get_track_by_id(self.tracker.current_track_id)
            if track is not None and not track.is_deleted() and track.time_since_update <= 1:
                t_box = tuple(map(int, track.to_tlbr()))
                t_id = track.track_id
                label = f"{'TARGET' if self._is_centered else 'CENTERING'} [ID: {t_id}]"
                draw_labeled_box(dbg, t_box, color=(0,0,255), label=label)

        status = self.state
        cv2.putText(
            dbg, f"State: {status}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (50,220,50) if status == 'LOCKED' else (0,165,255),
            2
        )
        
        if status == 'LOCKED':
            cv2.putText(
                dbg, f"Similarity: {self.current_similarity:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (50, 220, 50),
                2
            )
            if self.tracker.current_track_id is not None:
                cv2.putText(
                    dbg, f"Track ID: {self.tracker.current_track_id}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 200, 0),
                    2
                )

        depth_show = self.last_known_depth if self.last_known_depth is not None else depth_m
        depth_txt = "--" if depth_show is None else f"{float(depth_show):.2f} m"
        mode_txt  = "Centered" if self._is_centered else "Centering"
        hud_right = f"Depth: {depth_txt}   Mode: {mode_txt}"
        draw_label_top_right(dbg, hud_right, margin=10)

        if vmean < 90 or vmean > 200:
            cv2.putText(
                dbg, "LOW-LIGHT / BACKLIT MODE",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,140,255),
                2
            )

        if publish_debug_image:
            try:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
            except Exception:
                pass

        if self.enable_udp:
            try:
                ret, buffer = cv2.imencode('.jpg', dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    self.udp_sock.sendto(buffer.tobytes(), (self.udp_host, self.udp_port))
            except Exception:
                pass

    # ---------- Image callback ----------
    def on_image(self, msg: Image):
        self.frame_count += 1
        if self.frame_count % 1 != 0:
            return

        frame0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        W = int(self.get_parameter('image_width').value)
        H = int(self.get_parameter('image_height').value)
        frame = cv2.resize(frame0, (W, H), interpolation=cv2.INTER_LINEAR)
        
        depth_frame = cv2.resize(self.depth_img, (W, H), interpolation=cv2.INTER_NEAREST) if self.depth_img is not None else None

        vmean = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2])
        base_cw = float(self.get_parameter('body_color_weight').value)
        cw_min = float(self.get_parameter('body_color_weight_min').value)
        scale = float(self.get_parameter('body_color_weight_lowlight_scale').value)

        if vmean < 90 or vmean > 200:
            self._dynamic_color_weight = max(cw_min, min(base_cw, base_cw * scale))
        else:
            self._dynamic_color_weight = base_cw

        conf = float(self.get_parameter('person_conf').value)
        pboxes, _ = ssd_detect(self.ssd_net, frame, conf_thresh=conf)

        # --- Enrollment Phase ---
        if not self.auto_done:
            self.state = 'AUTO-ENROLL'
            
            if not self.enroll_audio_played:
                self.audio.play_sound_async(self.enroll_sound_file, repeat=2)
                self.enroll_audio_played = True

            self.auto_enroll_step(frame, pboxes, depth_frame)

            self.state_pub.publish(String(data=self.state))
            self.flag_pub.publish(Bool(data=False))
            self.centered_pub.publish(Bool(data=False))
            self.cmd_pub.publish(Twist())  # Robot dừng
            self.publish_debug(frame, pboxes, None, vmean, None)
            return

        # ===== ANTI-ID-SWITCHING #1: PRE-UPDATE OCCLUSION FREEZE =====
        occluded_pre = False
        if (self.state == 'LOCKED' and self.target_box is not None and
            self.last_known_depth is not None and depth_frame is not None):
            occluded_pre = self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth)

        if occluded_pre:
            self.get_logger().info("OCCLUDED: predict-only, stop robot (không LOST).")
            
            self.state = 'OCCLUDED'
            if self.occl_start_time is None:
                self.occl_start_time = time.time()
            self.recover_count = 0
            
            self.tracker.update([], [])
            
            target_track = self.tracker.get_track_by_id(self.tracker.current_track_id) if self.tracker.current_track_id is not None else None
            if target_track is not None and (not target_track.is_deleted()):
                self.target_box = tuple(map(int, target_track.to_tlbr()))

            self.cmd_pub.publish(Twist())
            self.state_pub.publish(String(data=self.state))
            self.flag_pub.publish(Bool(data=False))
            self.centered_pub.publish(Bool(data=False))
            self.publish_debug(frame, pboxes, self.target_box, vmean, None)
            return

        # ===== PROACTIVE OCCLUSION CHECK =====
        potential_occlusion = False
        if self.state == 'LOCKED' and self.target_box is not None:
            potential_occlusion = self.detect_potential_occlusion(self.target_box, pboxes, depth_frame)
        if potential_occlusion:
            self.get_logger().warn("Potential occlusion -> OCCLUDED")
            self.state = 'OCCLUDED'
            if self.occl_start_time is None:
                self.occl_start_time = time.time()
            self.recover_count = 0
            
            self.cmd_pub.publish(Twist())
            self.state_pub.publish(String(data=self.state))
            self.flag_pub.publish(Bool(data=False))
            self.centered_pub.publish(Bool(data=False))
            self.publish_debug(frame, pboxes, self.target_box, vmean, None)
            return

        # ===== PREDICT-ONLY INITIALIZATION EARLY =====
        predict_only = False
        
        if self.state == 'LOCKED' and len(pboxes) == 0:
            self.get_logger().warn("LOCKED: No detections (pboxes empty) -> predict-only")
            predict_only = True
        
        if self.state == 'LOCKED' and self.last_known_depth is None and depth_frame is not None and self.target_box is not None:
            self.last_known_depth = self.get_median_depth_at_box(self.target_box, depth_frame)

        # ===== ENHANCED DEPTH PRE-FILTER =====
        filtered_pboxes = pboxes
        det_depths = {}
        
        if self.state == 'LOCKED' and self.last_known_depth is not None and depth_frame is not None and not predict_only:
            depth_filter_margin = 0.5
            overlap_iou_thr = 0.15
            overlap_depth_margin = 0.3
            depth_range_tolerance = 0.4
            cheap_gate_iou = 0.05
            
            cands = []
            for box in pboxes:
                if self.target_box is not None:
                    box_iou = iou(box, self.target_box)
                    if box_iou > cheap_gate_iou:
                        cands.append(box)
                    else:
                        tc = center_of(self.target_box)
                        bc = center_of(box)
                        dist_2d = ((tc[0] - bc[0])**2 + (tc[1] - bc[1])**2) ** 0.5
                        if dist_2d < 150:
                            cands.append(box)
                else:
                    cands.append(box)
            
            filtered_pboxes = []
            for box in cands:
                b = tuple(map(int, box))
                det_depth = self.get_median_depth_at_box(b, depth_frame)
                det_depths[b] = det_depth
                
                if det_depth is None:
                    continue
                
                if self.target_box is not None:
                    box_iou = iou(box, self.target_box)
                    depth_diff = self.last_known_depth - det_depth
                    
                    if box_iou >= overlap_iou_thr and depth_diff > overlap_depth_margin:
                        continue
                    
                    if depth_diff > depth_filter_margin:
                        continue
                    
                    if depth_diff < -depth_filter_margin:
                        continue
                    
                    if box_iou < 0.3:
                        if abs(depth_diff) > depth_range_tolerance:
                            continue
                
                filtered_pboxes.append(box)
            
            if len(filtered_pboxes) == 0:
                self.get_logger().warn("DEPTH FILTER: all rejected in LOCKED -> predict-only this frame")
                predict_only = True

        # === FEATURE EXTRACTION ===
        final_pboxes = []
        detection_features = []
        
        if not predict_only:
            anchor = self.tracker.original_target_feature if self.tracker.original_target_feature is not None else self.tracker.target_feature
            pre_filter_thr = 0.72

            if self.state == 'LOCKED':
                pre_filter_thr = 0.75
            elif self.state == 'SEARCHING':
                pre_filter_thr = 0.70

            for box in filtered_pboxes:
                feat = enhanced_body_feature(frame, box, depth_frame, 
                                              self.mb2_sess, color_weight=self._dynamic_color_weight)
                
                if feat is None:
                    continue
                
                if self.state == 'LOCKED' and anchor is not None:
                    sim = np.dot(feat, anchor)
                    sim_with_current = np.dot(feat, self.tracker.target_feature) if self.tracker.target_feature is not None else 0.0
                    
                    if sim < pre_filter_thr or sim_with_current < (pre_filter_thr - 0.05):
                        continue
                
                final_pboxes.append(box)
                detection_features.append(feat)

        # === OCCLUSION CHECK ===
        if self.state == 'LOCKED' and self.target_box is not None and depth_frame is not None and not predict_only:
            if self.last_known_depth is not None and self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth):
                self.state = 'OCCLUDED'
                if self.occl_start_time is None:
                    self.occl_start_time = time.time()
                self.recover_count = 0
                predict_only = True
        
        if self.state in ('OCCLUDED',):
            predict_only = True
        
        # === Update DeepSORT tracker ===
        if predict_only:
            tracks = self.tracker.update([], [])
        elif self.state == 'LOCKED':
            matched_dets, matched_feats = self.tracker.locked_mode_tracking(
                frame, depth_frame, final_pboxes, detection_features, self.last_known_depth, self.get_median_depth_at_box
            )
            
            if matched_dets is None:
                tracks = self.tracker.update(final_pboxes, detection_features)
            else:
                tracks = self.tracker.update(matched_dets, matched_feats)
        else:
            tracks = self.tracker.update(final_pboxes, detection_features)

        confirmed_tracks = self.tracker.get_confirmed_tracks()
        is_real_update = False
        
        # ===== STATE: SEARCHING =====
        if self.state == 'SEARCHING':
            best_track = self.tracker.find_best_track_by_reid(confirmed_tracks)
            
            if best_track is not None:
                self.state = 'LOCKED'
                self.tracker.current_track_id = best_track.track_id
                self.target_box = tuple(map(int, best_track.to_tlbr()))
                self.last_known_depth = self.get_median_depth_at_box(self.target_box, depth_frame)
                
                self.sim_smooth = self.tracker.current_similarity
                self.current_similarity = self.tracker.current_similarity
                
                self.get_logger().info(f"Target LOCKED track_id={self.tracker.current_track_id}, score={self.current_similarity:.2f}")
                self.audio.stop_lost_sound_loop()

        # ===== STATE: LOCKED =====
        elif self.state == 'LOCKED':
            if self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth):
                self.get_logger().info("Target occluded. -> OCCLUDED")
                self.state = 'OCCLUDED'
                if self.occl_start_time is None:
                    self.occl_start_time = time.time()
                self.recover_count = 0
            else:
                target_track = self.tracker.get_track_by_id(self.tracker.current_track_id)
                reject_thr = self.get_parameter('reject_threshold').value
                
                if target_track is not None and not target_track.is_deleted():
                    is_real_update = (target_track.time_since_update == 0)
                    
                    if is_real_update:
                        self.miss_count = 0
                    else:
                        self.miss_count += 1
                    
                    if self.miss_count >= self.MISS_TO_SEARCH:
                        if not self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth):
                            self.get_logger().warn(f"Miss too long ({self.miss_count} frames, not occluded) -> SEARCHING")
                            self.state = 'SEARCHING'
                            self.tracker.current_track_id = None
                            self.target_box = None
                            self.miss_count = 0

                    new_box = tuple(map(int, target_track.to_tlbr()))
                    new_depth = self.get_median_depth_at_box(new_box, depth_frame)
                    
                    depth_jump_threshold = float(self.get_parameter('depth_jump_threshold').value)
                    if (self.last_known_depth is not None and new_depth is not None and
                        self.last_known_depth - new_depth > depth_jump_threshold):
                        self.get_logger().warn(
                            f"DEPTH JUMP: Intruder detected {self.last_known_depth:.2f}m -> {new_depth:.2f}m. -> LOST"
                        )
                        self.state = 'LOST'
                        self.lost_start_time = time.time()

                        self.state_pub.publish(String(data=self.state))
                        self.flag_pub.publish(Bool(data=False))
                        self.centered_pub.publish(Bool(data=False))
                        self.cmd_pub.publish(Twist())
                        self.publish_debug(frame, pboxes, self.target_box, vmean, None)
                        return
                    else:
                        self.target_box = new_box
                        if is_real_update and (new_depth is not None):
                            self.last_known_depth = new_depth
                    
                    track_feature = target_track.get_feature()
                    anchor = self.tracker.original_target_feature if self.tracker.original_target_feature is not None else self.tracker.target_feature
                    if track_feature is not None and anchor is not None:
                        raw_sim = float(np.dot(track_feature, anchor))
                        
                        alpha = self.get_parameter('similarity_ema_alpha').value
                        if not hasattr(self, 'sim_smooth'): self.sim_smooth = raw_sim
                        self.sim_smooth = alpha * self.sim_smooth + (1 - alpha) * raw_sim
                        
                        self.current_similarity = self.sim_smooth
                        
                        if self.current_similarity < reject_thr:
                            self.get_logger().warn(
                                f"Low Similarity: raw={raw_sim:.3f}, smooth={self.sim_smooth:.3f} < thr={reject_thr} "
                                f"(margin={self.get_parameter('margin_delta').value})"
                            )
                        
                        if is_real_update and (not self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth)):
                            if self.current_similarity >= 0.85 and self.current_similarity <= 0.97:
                                self.update_streak += 1
                            else:
                                self.update_streak = 0
                            
                            now = time.time()
                            if (self.update_streak >= self.UPDATE_STREAK_N and
                                now - self.last_update_time > self.adaptive_update_interval_sec):
                                self.tracker.adaptive_model_update(self.target_box, frame, depth_frame, self._dynamic_color_weight)
                                self.last_update_time = now
                                self.update_streak = 0
                        else:
                            self.update_streak = 0
                        
                        if self.current_similarity < reject_thr:
                            self.get_logger().info(f"Similarity too low ({self.current_similarity:.2f}). -> LOST")
                            self.state = 'LOST'
                            self.lost_start_time = time.time()
                else:
                    self.get_logger().info("Target track lost. -> LOST (no re-matching)")
                    self.state = 'LOST'
                    self.lost_start_time = time.time()

        # ===== STATE: LOST =====
        elif self.state == 'LOST':
            target_track = self.tracker.get_track_by_id(self.tracker.current_track_id)
            
            if target_track is not None and not target_track.is_deleted():
                self.target_box = tuple(map(int, target_track.to_tlbr()))
                
                if target_track.time_since_update == 0:
                    track_feature = target_track.get_feature()
                    anchor = self.tracker.original_target_feature if self.tracker.original_target_feature is not None else self.tracker.target_feature
                    if track_feature is not None and anchor is not None:
                        score = float(np.dot(track_feature, anchor))
                        accept_thr = self.get_parameter('accept_threshold').value
                        if score > accept_thr:
                            self.state = 'LOCKED'
                            self.current_similarity = score
                            self.last_known_depth = self.get_median_depth_at_box(self.target_box, depth_frame)
                            self.get_logger().info(f"Target re-acquired! track_id={self.tracker.current_track_id}, score={score:.2f}")
                            self.audio.stop_lost_sound_loop()
            
            if self.lost_start_time is not None:
                if time.time() - self.lost_start_time > self.get_parameter('grace_period_sec').value:
                    self.get_logger().info("Grace period expired. → SEARCHING")
                    self.state = 'SEARCHING'
                    self.target_box = None
                    self.tracker.current_track_id = None
                    self.audio.start_lost_sound_loop(self.sound_file)

        # ===== STATE: OCCLUDED =====
        elif self.state == 'OCCLUDED':
            target_track = self.tracker.get_track_by_id(self.tracker.current_track_id)

            if target_track is not None and not target_track.is_deleted():
                self.target_box = tuple(map(int, target_track.to_tlbr()))

            if self.occl_start_time is None:
                self.occl_start_time = time.time()

            cleared = False
            if (self.target_box is not None and depth_frame is not None and self.last_known_depth is not None):
                cleared = (not self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth))

            if cleared:
                self.get_logger().info("Occlusion cleared -> RECOVER")
                self.state = 'RECOVER'
                self.recover_count = 0
                self.recover_start_time = time.time()
                self.occl_start_time = None
            else:
                occl_duration = time.time() - self.occl_start_time
                if occl_duration > self.OCCL_MAX_SEC:
                    self.get_logger().warn(
                        f"OCCLUDED timeout ({occl_duration:.1f}s > {self.OCCL_MAX_SEC}s) -> SEARCHING"
                    )
                    self.state = 'SEARCHING'
                    self.occl_start_time = None
                    self.recover_count = 0
                    self.tracker.current_track_id = None
                    self.target_box = None

        # ===== STATE: RECOVER (New) =====
        elif self.state == 'RECOVER':
            if hasattr(self, 'recover_start_time') and (time.time() - self.recover_start_time > self.RECOVER_TIMEOUT):
                self.get_logger().warn(f"RECOVER timeout ({self.RECOVER_TIMEOUT}s) -> SEARCHING")
                self.state = 'SEARCHING'
                self.target_box = None
                self.tracker.current_track_id = None
                self.audio.start_lost_sound_loop(self.sound_file)
                return

            best_box, best_score = self.tracker.find_best_match_by_reid(final_pboxes, frame, depth_frame, self._dynamic_color_weight)
            
            if best_box is not None and best_score >= self.RECOVER_THR:
                det_depth = self.get_median_depth_at_box(best_box, depth_frame)
                
                if (det_depth is not None and self.last_known_depth is not None and
                    abs(det_depth - self.last_known_depth) <= self.RECOVER_DEPTH_THR):
                    
                    try:
                        idx = final_pboxes.index(best_box)
                        self.tracker.update([best_box], [detection_features[idx]])
                    except (ValueError, IndexError):
                        self.tracker.update([], [])
                    
                    self.recover_count += 1
                    
                    if self.recover_count >= self.RECOVER_CONFIRM:
                        self.get_logger().info(f"RECOVER confirmed -> LOCKED (score={best_score:.3f})")
                        self.state = 'LOCKED'
                        self.occl_start_time = None
                        self.miss_count = 0
                        self.recover_count = 0
                        self.target_box = best_box
                        self.last_known_depth = det_depth
                        self.current_similarity = best_score
                else:
                    self.recover_count = 0
            else:
                self.recover_count = 0
            
            if self.target_box is not None:
                if self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth):
                    self.get_logger().info("RECOVER: Occluded again -> OCCLUDED")
                    self.state = 'OCCLUDED'
                    if self.occl_start_time is None:
                        self.occl_start_time = time.time()
                    self.recover_count = 0

        # --- Command & Publishing ---
        twist, detected, depth_m = self.compute_cmd(W, H, self.target_box, depth_frame)
        
        if self.state in ('OCCLUDED', 'RECOVER') or (self.state == 'LOCKED' and not is_real_update):
            twist = Twist()
            
        self.cmd_pub.publish(twist)
        self.flag_pub.publish(Bool(data=(self.state == 'LOCKED')))
        if depth_m is not None:
            self.dist_depth_pub.publish(Float32(data=float(depth_m)))
        self.state_pub.publish(String(data=self.state))

        centered_msg = Bool()
        centered_msg.data = bool((self.state == 'LOCKED') and self._is_centered)
        self.centered_pub.publish(centered_msg)

        self.publish_debug(frame, pboxes, self.target_box, vmean, depth_m)

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.audio.stop_lost_sound_loop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
