#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PersonDetector (ROS2) — Single-Target Tracking with DeepSORT + Anti-ID-Switching
Optimized for Orange Pi 5 Plus (CPU-only) and Intel RealSense D455.

Key Features:
- State Machine (AUTO-ENROLL, SEARCHING, LOCKED, LOST) for robust tracking.
- DeepSORT Tracker with Kalman Filter for robust tracking.
- ANTI-ID-SWITCHING MEASURES:
  1. Pre-update Occlusion Freeze - Không update DeepSORT khi che khuất
  2. Depth Pre-Filter - Loại bỏ detection gần hơn target
  3. Depth Jump Detection - Phát hiện intruder
  4. Track Switching Prevention - Yêu cầu margin để switch track
  5. No Re-match in LOST - Không lấy track khác khi mất target
  6. Anchor Feature Comparison - So sánh với feature gốc
- Depth-aware distance control and occlusion handling.
- Enhanced ReID features with depth information.
- CPU optimizations: lower resolution, frame skipping, ROI-based detection.
- UDP Streaming: Stream debug video to backend server.
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
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# DeepSORT Tracker
from mecanum_control.tracking import DeepSORTTracker

# ========== Paths ==========
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True, parents=True)
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True, parents=True)

MB2_ONNX_PATH  = str(MODELS / "mb2_gap.onnx")
MOBILENET_PROTOTXT = str(MODELS / "MobileNetSSD_deploy.prototxt")
MOBILENET_WEIGHTS  = str(MODELS / "MobileNetSSD_deploy.caffemodel")

# ========== Helpers ==========
def clamp(x,a,b): return a if x<a else b if x>b else x

def iou(a, b):
    if a is None or b is None: return 0.0
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
    return inter/ua if ua>0 else 0.0

def center_of(box):
    x1,y1,x2,y2=box
    return ((x1+x2)//2, (y1+y2)//2)

def expand(box, shape, m=0.20):
    x1,y1,x2,y2 = box
    H,W = shape[:2]; w=x2-x1; h=y2-y1
    x1 = max(0, int(x1 - m*w)); y1 = max(0, int(y1 - m*h))
    x2 = min(W-1, int(x2 + m*w)); y2 = min(H-1, int(y2 + m*h))
    return (x1,y1,x2,y2)

# ===== Overlay helpers =====
def draw_label_top_right(img, text, margin=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x2 = img.shape[1] - margin
    y1 = margin
    x1 = x2 - tw - 16
    y2 = y1 + th + 16
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x1 + 8, y2 - 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_labeled_box(img, box, color=(0,0,255), label="TARGET"):
    x1,y1,x2,y2 = box
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx1, ty1 = x1, max(0, y1 - th - 10)
    tx2, ty2 = x1 + tw + 12, ty1 + th + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.putText(img, label, (tx1 + 6, ty2 - 6), font, scale, (255,255,255), thickness, cv2.LINE_AA)

# =================== ENHANCED BODY FEATURES ===================
def mb2_preprocess_keras_style(x_uint8):
    x = x_uint8.astype(np.float32)
    x = x/127.5 - 1.0
    return x

def body_arr_preserve_aspect_ratio(frame, box, target_size=(224, 224)):
    """Trích xuất ROI và resize về target_size, giữ nguyên tỷ lệ bằng cách thêm padding."""
    x1, y1, x2, y2 = map(int, box)
    
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None, None
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None, None

    h, w = roi.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    if new_w <= 0 or new_h <= 0:
        return None, None
    
    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_roi
    
    return padded, scale

def hsv_histogram(roi_bgr, bins=16, v_weight=0.5, normalize_brightness=True):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    if normalize_brightness:
        v_channel = hsv[:,:,2].astype(np.float32)
        v_mean = v_channel.mean()
        if v_mean > 10:
            v_channel = np.clip(v_channel * (128.0 / v_mean), 0, 255)
            hsv[:,:,2] = v_channel.astype(np.uint8)

    histH = cv2.calcHist([hsv],[0],None,[bins],[0,180]).flatten()
    histS = cv2.calcHist([hsv],[1],None,[bins],[0,256]).flatten()
    histV = cv2.calcHist([hsv],[2],None,[bins],[0,256]).flatten()
    histV *= v_weight

    h = np.concatenate([histH,histS,histV]).astype(np.float32)
    h /= (np.linalg.norm(h)+1e-8)
    return h

def extract_depth_feature(box, depth_img, target_size=(16, 16)):
    """Trích xuất một vector đặc trưng đơn giản từ depth."""
    if depth_img is None or box is None:
        return np.zeros(target_size[0] * target_size[1])
        
    x1, y1, x2, y2 = map(int, box)
    roi = depth_img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return np.zeros(target_size[0] * target_size[1])
    
    roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
    roi_normalized = np.clip((5000 - roi_resized) / 4500.0, 0.0, 1.0)
    
    depth_feat = roi_normalized.flatten().astype(np.float32)
    return depth_feat

def enhanced_body_feature(frame, box, depth_img, ort_sess, color_weight=0.3, normalize_brightness=True):
    """Kết hợp đặc trưng hình thái, màu sắc và depth."""
    roi_padded, _ = body_arr_preserve_aspect_ratio(frame, box)
    if roi_padded is None: return None
    
    roi_rgb = cv2.cvtColor(roi_padded, cv2.COLOR_BGR2RGB)
    arr = mb2_preprocess_keras_style(roi_rgb)[None,...]
    
    inp_name = ort_sess.get_inputs()[0].name
    emb = ort_sess.run(None, {inp_name: arr.astype(np.float32)})[0].reshape(-1).astype(np.float32)
    emb /= (np.linalg.norm(emb)+1e-8)

    col = hsv_histogram(roi_padded, bins=16, v_weight=0.6, normalize_brightness=normalize_brightness)

    depth_feat = extract_depth_feature(box, depth_img)
    depth_feat /= (np.linalg.norm(depth_feat) + 1e-8)

    emb_weighted = emb * (1.0 - color_weight)
    col_weighted = col * color_weight
    depth_weighted = depth_feat * 0.1

    feat = np.concatenate([emb_weighted, col_weighted, depth_weighted], axis=0).astype(np.float32)
    feat /= (np.linalg.norm(feat)+1e-8)
    return feat

# =================== Detector (MobileNet-SSD) ===================
def _load_ssd():
    if Path(MOBILENET_PROTOTXT).exists() and Path(MOBILENET_WEIGHTS).exists():
        return cv2.dnn.readNetFromCaffe(MOBILENET_PROTOTXT, MOBILENET_WEIGHTS)
    return None

def _ssd_detect(net, frame, conf_thresh=0.4):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843, (300,300), 127.5)
    net.setInput(blob)
    det = net.forward()
    boxes, scores = [], []
    for i in range(det.shape[2]):
        conf = det[0,0,i,2]; cls = int(det[0,0,i,1])
        if cls==15 and conf>conf_thresh:
            box = det[0,0,i,3:7]*np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            if x2>x1 and y2>y1:
                boxes.append((x1,y1,x2,y2)); scores.append(float(conf))
    return boxes, scores

# =================== ROS2 Node ===================
class PersonDetector(Node):
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
            ('accept_threshold', 0.78),
            ('reject_threshold', 0.65),
            ('iou_threshold', 0.4),
            ('margin_delta', 0.07),
            ('confirm_frames', 5),

            # Color weight
            ('body_color_weight', 0.25),
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
            ('depth_filter_margin', 0.5),       # Loại bỏ detection gần hơn target X meters
            ('overlap_iou_thr', 0.20),          # IoU threshold để xét intruder
            ('overlap_depth_margin', 0.3),      # Depth margin cho overlapping detection
            ('depth_jump_threshold', 0.6),      # Ngưỡng depth jump để phát hiện intruder
            ('track_switch_margin', 0.2),      # Margin cần để switch track (cao hơn = khó switch hơn)
            ('pre_filter_appearance_thr', 0.70), # Ngưỡng similarity để lọc detection trước khi đưa vào tracker
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
        self.ssd_net = _load_ssd()
        if self.ssd_net is None:
            raise FileNotFoundError(
                "Không tìm thấy MobileNet-SSD Caffe. Hãy đặt file vào:\n"
                f"  - {MOBILENET_PROTOTXT}\n"
                f"  - {MOBILENET_WEIGHTS}"
            )
        self.get_logger().info("Person detector: MobileNet-SSD (Caffe)")
        self.get_logger().info("PersonDetector initialized with DeepSORT + Anti-ID-Switching")

        # --- STATE MACHINE VARIABLES ---
        self.state = 'AUTO-ENROLL'
        self.target_box = None
        self.target_feature = None
        self.original_target_feature = None  # ANCHOR - KHÔNG BAO GIỜ THAY ĐỔI
        self.last_known_depth = None
        self.lost_start_time = None
        self.current_similarity = 0.0
        
        # --- Occlusion State Machine counters ---
        self.miss_count = 0           # số frame liên tiếp track không update thật
        self.occl_start_time = None   # timestamp bắt đầu bị che
        self.recover_count = 0        # số frame liên tiếp match tốt khi recover
        
        # --- Occlusion/Recover thresholds ---
        self.MISS_TO_SEARCH = 12      # mất ~12 frame mà không occlusion → SEARCHING
        self.OCCL_MAX_SEC = 3.0       # che tối đa 3s vẫn cố giữ
        self.RECOVER_CONFIRM = 4      # cần 4 frame match liên tiếp → LOCKED
        self.RECOVER_THR = 0.82       # similarity threshold khi recover (chặt)
        self.RECOVER_DEPTH_THR = 0.35 # depth gate khi recover
        
        # --- DEEPSORT TRACKER với stricter parameters ---
        self.deepsort = DeepSORTTracker(
            max_age=60,
            n_init=5,
            max_cosine_distance=0.08,  # STRICTER: 0.10 thay vì 0.15
            lambda_weight=0.85          # ANTI-HIJACK: Tăng trọng số ReID (0.2 -> 0.8) để ưu tiên đặc điểm nhận dạng
        )
        self.current_track_id = None
        
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
        self.adaptive_update_threshold = 0.75
        self.last_update_time = 0.0
        self.adaptive_update_interval_sec = 1.5

        # UDP Streaming
        self.enable_udp = bool(self.get_parameter('enable_udp_stream').value)
        if self.enable_udp:
            self.udp_host = self.get_parameter('udp_host').value
            self.udp_port = int(self.get_parameter('udp_port').value)
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.get_logger().info(f"UDP Streaming enabled to {self.udp_host}:{self.udp_port}")

        # Sound Path
        self.sound_file = str(HERE / "sounds" / self.get_parameter('sound_filename').value)
        self.enroll_sound_file = str(HERE / "sounds" / self.get_parameter('enroll_sound_filename').value)
        self.run_sound_file = str(HERE / "sounds" / self.get_parameter('run_sound_filename').value)
        self.enroll_audio_played = False
        self.run_audio_played = False
        
        # Threading for lost sound loop
        self.lost_sound_thread = None
        self.stop_lost_sound_event = threading.Event()


    # ---------- Depth ----------
    def on_depth(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.get_parameter('depth_encoding').value)
        self.depth_enc = self.get_parameter('depth_encoding').value

    def get_median_depth_at_box(self, box, depth_img):
        """Lấy giá trị depth trung vị tại một bounding box."""
        if depth_img is None or box is None:
            return None
        x1, y1, x2, y2 = map(int, box)
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
    def auto_enroll_step(self, frame, pboxes):
        now = time.time()
        if self.auto_start_ts is None:
            self.auto_start_ts = now

        if pboxes:
            j = int(np.argmax([(pb[2]-pb[0])*(pb[3]-pb[1]) for pb in pboxes]))
            pb = pboxes[j]
            feat = enhanced_body_feature(frame, pb, self.depth_img, self.mb2_sess,
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
            if self.body_centroid is not None:
                self.target_feature = self.body_centroid.copy()
                self.original_target_feature = self.body_centroid.copy()
                self.get_logger().info("Target enrolled. ANCHOR feature saved.")
            self.auto_done = True
            self.state = 'SEARCHING'
            
            if not self.run_audio_played:
                if os.path.exists(self.run_sound_file):
                    os.system(f"aplay {self.run_sound_file};aplay {self.run_sound_file} &")
                self.run_audio_played = True

    # ---------- Control ----------
    def compute_cmd(self, frame_w, frame_h, target_box):
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

        depth_raw = self.get_median_depth_at_box(target_box, self.depth_img)
        
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

    # ---------- Detector wrap ----------
    def detect_persons(self, frame, conf_thresh: float):
        return _ssd_detect(self.ssd_net, frame, conf_thresh)

    # ---------- DeepSORT Helper với ANTI-ID-SWITCHING ----------
    def _find_best_track_by_reid(self, tracks):
        """
        Tìm track có similarity cao nhất với ANCHOR feature.
        Bao gồm TRACK SWITCHING PREVENTION.
        """
        if self.original_target_feature is None:
            return None
        
        # Dùng ANCHOR feature thay vì target_feature (có thể đã drift)
        anchor = self.original_target_feature
        accept_thr = self.get_parameter('accept_threshold').value
        switch_margin = float(self.get_parameter('track_switch_margin').value)
        
        best_track = None
        best_score = -1.0
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_feature = track.get_feature()
            if track_feature is None:
                continue
            
            # So sánh với ANCHOR (không phải target_feature)
            score = float(np.dot(track_feature, anchor))
            if score > best_score:
                best_score = score
                best_track = track
        
        # === ANTI-ID-SWITCHING: Track Switching Prevention ===
        if self.current_track_id is not None:
            current_track = self.deepsort.get_track_by_id(self.current_track_id)
            if current_track is not None and not current_track.is_deleted():
                current_feature = current_track.get_feature()
                if current_feature is not None:
                    current_score = float(np.dot(current_feature, anchor))
                    
                    # Yêu cầu track mới phải tốt hơn đáng kể mới switch
                    if best_track is not None and best_track.track_id != self.current_track_id:
                        if best_score < current_score + switch_margin:
                            # Keep current track (new track không đủ tốt)
                            if current_score > accept_thr:
                                self.current_similarity = current_score
                                return current_track
        
        if best_score > accept_thr:
            self.current_similarity = best_score
            return best_track
        
        return None

    # ---------- Lost Sound Loop ----------
    def _lost_sound_loop(self):
        while not self.stop_lost_sound_event.is_set():
            if os.path.exists(self.sound_file):
                os.system(f"aplay {self.sound_file}")
            time.sleep(0.5)

    def start_lost_sound_loop(self):
        if self.lost_sound_thread is not None and self.lost_sound_thread.is_alive():
            return
        
        self.stop_lost_sound_event.clear()
        self.lost_sound_thread = threading.Thread(target=self._lost_sound_loop, daemon=True)
        self.lost_sound_thread.start()
        self.get_logger().info("Started lost target sound loop.")

    def stop_lost_sound_loop(self):
        if self.lost_sound_thread is None or not self.lost_sound_thread.is_alive():
            return
        
        self.stop_lost_sound_event.set()
        self.lost_sound_thread.join(timeout=2.0)
        self.lost_sound_thread = None
        self.get_logger().info("Stopped lost target sound loop.")

    # ---------- Matching ----------
    def find_best_match_by_reid(self, boxes, frame, depth_frame):
        best_box, best_score = None, -1.0
        anchor = self.original_target_feature if self.original_target_feature is not None else self.target_feature
        if anchor is None:
            return None, -1.0
            
        for box in boxes:
            feat = enhanced_body_feature(frame, box, depth_frame, self.mb2_sess, color_weight=self._dynamic_color_weight)
            if feat is None: continue
            
            score = np.dot(feat, anchor)
            if score > best_score:
                best_score = score
                best_box = box
        return best_box, best_score

    # ---------- Adaptive Model Update ----------
    def adaptive_model_update(self, box, frame, depth_frame):
        """Cập nhật model thông minh với ANCHOR protection."""
        if box is None or self.target_feature is None:
            return

        candidate_feat = enhanced_body_feature(
            frame, box, depth_frame, self.mb2_sess,
            color_weight=self._dynamic_color_weight
        )
        if candidate_feat is None:
            return

        # So sánh với ANCHOR (không phải target_feature hiện tại)
        anchor = self.original_target_feature if self.original_target_feature is not None else self.target_feature
        similarity_with_anchor = float(np.dot(candidate_feat, anchor))
        
        if similarity_with_anchor < self.get_parameter('reject_threshold').value:
            self.get_logger().warn(f"Update rejected: low anchor similarity {similarity_with_anchor:.2f}")
            return

        if similarity_with_anchor > 0.99:
            return

        self.update_target_model(candidate_feat)
        self.get_logger().info(f"Model updated. Anchor similarity: {similarity_with_anchor:.2f}")

    def update_target_model(self, new_feature):
        """
        Cập nhật target_feature với ANCHOR protection.
        Formula: target = 0.6 × ANCHOR + 0.3 × current + 0.1 × new
        """
        if self.target_feature is None:
            self.target_feature = new_feature.astype(np.float32)
            return
        
        if self.original_target_feature is None:
            alpha = 0.2
            self.target_feature = (1.0 - alpha) * self.target_feature + alpha * new_feature
        else:
            anchor_weight = 0.6
            current_weight = 0.3
            new_weight = 0.1
            
            self.target_feature = (
                anchor_weight * self.original_target_feature +
                current_weight * self.target_feature +
                new_weight * new_feature
            )

        self.target_feature /= (np.linalg.norm(self.target_feature) + 1e-8)

    # ---------- Custom Locked Mode Tracking ----------
    def locked_mode_tracking(self, frame, depth_frame, all_detections, all_features):
        """
        Custom tracking logic khi ở trạng thái LOCKED.
        Ưu tiên tuyệt đối cho target hiện tại, chỉ accept detection
        nếu thực sự match với target.
        """
        if self.current_track_id is None or self.target_box is None:
            return None, None
        
        # Lấy track hiện tại
        target_track = self.deepsort.get_track_by_id(self.current_track_id)
        if target_track is None or target_track.is_deleted():
            return None, None
        
        # Predicted position từ Kalman
        predicted_box = tuple(map(int, target_track.to_tlbr()))
        
        # Tìm detection tốt nhất cho target
        best_detection_idx = None
        best_score = -1.0
        
        anchor = self.original_target_feature if self.original_target_feature is not None else self.target_feature
        
        for idx, (det_box, det_feat) in enumerate(zip(all_detections, all_features)):
            # Score 1: Appearance similarity với Anchor
            appearance_score = np.dot(det_feat, anchor) if anchor is not None else 0.0
            
            # Score 2: IoU với predicted box
            iou_score = iou(det_box, predicted_box)
            
            # Score 3: Depth consistency
            det_depth = self.get_median_depth_at_box(det_box, depth_frame)
            depth_score = 0.0
            if det_depth is not None and self.last_known_depth is not None:
                depth_diff = abs(det_depth - self.last_known_depth)
                depth_score = max(0, 1.0 - depth_diff / 1.0)  # 1m tolerance
            
            # Combined score (weighted)
            combined_score = (
                0.60 * appearance_score +  # Appearance quan trọng nhất
                0.25 * iou_score +          # Position thứ hai
                0.15 * depth_score          # Depth bổ sung
            )
            
            # Thresholds
            MIN_APPEARANCE = 0.70
            MIN_IOU = 0.20
            MIN_COMBINED = 0.65
            
            if (appearance_score >= MIN_APPEARANCE and 
                iou_score >= MIN_IOU and 
                combined_score > best_score and
                combined_score >= MIN_COMBINED):
                best_score = combined_score
                best_detection_idx = idx
                
                self.get_logger().info(
                    f"LOCKED matching: det[{idx}] score={combined_score:.3f} "
                    f"(app={appearance_score:.3f}, iou={iou_score:.3f}, depth={depth_score:.3f})"
                )
        
        if best_detection_idx is not None:
            # Chỉ update DeepSORT với detection này
            matched_det = [all_detections[best_detection_idx]]
            matched_feat = [all_features[best_detection_idx]]
            return matched_det, matched_feat
        else:
            # Không có detection match → predict only
            self.get_logger().warn("LOCKED: No matching detection, predict only")
            return [], []

    # ---------- Proactive Occlusion Check ----------
    def detect_potential_occlusion(self, target_box, detections, depth_frame):
        """
        Phát hiện TRƯỚC khi có occlusion thực sự.
        Trả về True nếu có người đang tiến đến gần và có nguy cơ che khuất.
        """
        if target_box is None or depth_frame is None or self.last_known_depth is None:
            return False
        
        target_depth = self.last_known_depth
        tx1, ty1, tx2, ty2 = target_box
        
        # Ngưỡng để xác định "approaching intruder"
        APPROACH_DEPTH_MARGIN = 0.6  # Gần hơn target 0.6m
        APPROACH_HORIZONTAL_OVERLAP = 0.3  # Overlap ngang 30%
        
        for det_box in detections:
            det_depth = self.get_median_depth_at_box(det_box, depth_frame)
            if det_depth is None:
                continue
            
            # Check 1: Detection gần hơn target đáng kể
            if (target_depth - det_depth) < APPROACH_DEPTH_MARGIN:
                continue
            
            # Check 2: Detection có overlap ngang với target không?
            dx1, dy1, dx2, dy2 = det_box
            
            # Tính horizontal overlap
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

        # --- 1. VẼ TẤT CẢ DETECTION (MÀU XANH) ---
        # Vẽ pboxes (raw detections) để thấy được mọi người, kể cả khi chưa có Track ID
        if pboxes is not None:
            for box in pboxes:
                # Nếu box này trùng với target (đang Locked), bỏ qua để vẽ màu đỏ sau
                if target_box is not None and iou(box, target_box) > 0.5:
                    continue
                # Vẽ người lạ / người đang enroll: Màu XANH LÁ
                draw_labeled_box(dbg, box, color=(0,255,0), label="")

        # --- 2. VẼ TARGET TRACK (MÀU ĐỎ) ---
        # Chỉ vẽ Track của Target đang được theo dõi
        if self.current_track_id is not None:
            track = self.deepsort.get_track_by_id(self.current_track_id)
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
            if self.current_track_id is not None:
                cv2.putText(
                    dbg, f"Track ID: {self.current_track_id}", (10, 90),
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
        if vmean < 90 or vmean > 200:
            self._dynamic_color_weight = min(0.10, base_cw * 0.6)
        else:
            self._dynamic_color_weight = base_cw

        pboxes, _ = self.detect_persons(frame, conf_thresh=0.4)

        # --- Enrollment Phase ---
        if not self.auto_done:
            self.state = 'AUTO-ENROLL'
            
            if not self.enroll_audio_played:
                if os.path.exists(self.enroll_sound_file):
                    os.system(f"(aplay {self.enroll_sound_file}; aplay {self.enroll_sound_file}) &")
                self.enroll_audio_played = True

            self.auto_enroll_step(frame, pboxes)

            self.state_pub.publish(String(data=self.state))
            self.flag_pub.publish(Bool(data=False))
            self.centered_pub.publish(Bool(data=False))
            self.cmd_pub.publish(Twist())  # Robot dừng
            self.publish_debug(frame, pboxes, None, vmean, None)
            return

        # ===== ANTI-ID-SWITCHING #1: PRE-UPDATE OCCLUSION FREEZE =====
        # Nếu target bị che khuất, KHÔNG update DeepSORT với bất kỳ detection nào
        occluded_pre = False
        if (self.state == 'LOCKED' and self.target_box is not None and
            self.last_known_depth is not None and depth_frame is not None):
            occluded_pre = self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth)

        if occluded_pre:
            # OCCLUDED: predict-only, stop robot (không chuyển LOST)
            self.get_logger().info("OCCLUDED: predict-only, stop robot (không LOST).")
            
            self.state = 'OCCLUDED'
            if self.occl_start_time is None:
                self.occl_start_time = time.time()
            self.recover_count = 0
            
            # predict-only
            self.deepsort.update([], [])
            
            # giữ predicted box nếu track còn
            target_track = self.deepsort.get_track_by_id(self.current_track_id) if self.current_track_id is not None else None
            if target_track is not None and (not target_track.is_deleted()):
                self.target_box = tuple(map(int, target_track.to_tlbr()))

            # stop robot
            self.cmd_pub.publish(Twist())
            self.state_pub.publish(String(data=self.state))
            self.flag_pub.publish(Bool(data=False))
            self.centered_pub.publish(Bool(data=False))
            self.publish_debug(frame, pboxes, self.target_box, vmean, None)
            return

        # ===== PROACTIVE OCCLUSION CHECK =====
        potential_occlusion = False
        if self.state == 'LOCKED' and self.target_box is not None:
            potential_occlusion = self.detect_potential_occlusion(
                self.target_box, pboxes, depth_frame
            )

        if potential_occlusion:
            self.get_logger().warn("🛑 FREEZE MODE: Potential occlusion detected")
            
            # FREEZE: Không update DeepSORT với detection mới
            tracks = self.deepsort.update([], [])  # Empty detections = chỉ predict
            
            # Giữ predicted box
            target_track = self.deepsort.get_track_by_id(self.current_track_id) if self.current_track_id is not None else None
            if target_track is not None and not target_track.is_deleted():
                self.target_box = tuple(map(int, target_track.to_tlbr()))
            
            # Dừng robot
            self.cmd_pub.publish(Twist())
            self.state_pub.publish(String(data="FREEZE"))
            self.flag_pub.publish(Bool(data=False))
            self.centered_pub.publish(Bool(data=False))
            self.publish_debug(frame, pboxes, self.target_box, vmean, None)
            return

        # ===== ENHANCED DEPTH PRE-FILTER =====
        filtered_pboxes = pboxes
        if self.state == 'LOCKED' and self.last_known_depth is not None and depth_frame is not None:
            depth_filter_margin = 0.5  # Giảm từ 0.8 → 0.5 (strict hơn)
            overlap_iou_thr = 0.15     # Giảm từ 0.20 → 0.15 (phát hiện overlap sớm hơn)
            overlap_depth_margin = 0.3 # Giảm từ 0.5 → 0.3 (strict hơn)
            
            # THÊM: Depth range tolerance
            depth_range_tolerance = 0.4  # ±0.4m từ target depth
            
            filtered_pboxes = []
            for box in pboxes:
                det_depth = self.get_median_depth_at_box(box, depth_frame)
                
                if det_depth is None:
                    # Nếu không đo được depth → REJECT (an toàn hơn)
                    self.get_logger().warn("DEPTH FILTER: Rejected detection with no depth")
                    continue
                
                if self.target_box is not None:
                    box_iou = iou(box, self.target_box)
                    depth_diff = self.last_known_depth - det_depth
                    
                    # Rule 1: Loại bỏ overlap + gần hơn
                    if box_iou >= overlap_iou_thr and depth_diff > overlap_depth_margin:
                        self.get_logger().warn(
                            f"DEPTH FILTER [OVERLAP]: IoU={box_iou:.2f}, "
                            f"depth_diff={depth_diff:.2f}m → REJECTED"
                        )
                        continue
                    
                    # Rule 2: Loại bỏ detection gần hơn nhiều (intruder từ phía trước)
                    if depth_diff > depth_filter_margin:
                        self.get_logger().warn(
                            f"DEPTH FILTER [TOO CLOSE]: det={det_depth:.2f}m vs "
                            f"target={self.last_known_depth:.2f}m → REJECTED"
                        )
                        continue
                    
                    # Rule 3 (MỚI): Loại bỏ detection xa hơn nhiều (người ở phía sau)
                    if depth_diff < -depth_filter_margin:
                        self.get_logger().warn(
                            f"DEPTH FILTER [TOO FAR]: det={det_depth:.2f}m vs "
                            f"target={self.last_known_depth:.2f}m → REJECTED"
                        )
                        continue
                    
                    # Rule 4 (MỚI): Chỉ chấp nhận detection trong range hợp lý
                    # Nếu detection không overlap với target box, phải trong depth range
                    if box_iou < 0.3:  # Không overlap đáng kể
                        if abs(depth_diff) > depth_range_tolerance:
                            self.get_logger().warn(
                                f"DEPTH FILTER [OUT OF RANGE]: det={det_depth:.2f}m, "
                                f"target={self.last_known_depth:.2f}m, "
                                f"diff={abs(depth_diff):.2f}m → REJECTED"
                            )
                            continue
                
                filtered_pboxes.append(box)
            
            # FIX #4: Fallback chỉ cho phép khi SEARCHING (chưa lock ai)
            # Khi LOCKED/OCCLUDED, fallback sẽ phá anti-switch
            if self.state == 'SEARCHING' and len(filtered_pboxes) == 0 and len(pboxes) > 0:
                self.get_logger().warn("DEPTH FILTER: All detections rejected, keeping closest one (SEARCHING only)")
                closest_box = min(pboxes, 
                    key=lambda b: abs(self.get_median_depth_at_box(b, depth_frame) - self.last_known_depth)
                    if self.get_median_depth_at_box(b, depth_frame) is not None else float('inf')
                )
                filtered_pboxes = [closest_box]

        # === STRICT APPEARANCE PRE-FILTER ===
        final_pboxes = []
        detection_features = []
        
        anchor = self.original_target_feature if self.original_target_feature is not None else self.target_feature
        pre_filter_thr = 0.72  # TĂNG từ 0.6 → 0.72 (strict hơn nhiều)

        # THÊM: Dynamic threshold based on state
        if self.state == 'LOCKED':
            # Khi LOCKED, yêu cầu similarity cao hơn để chống ID switch
            pre_filter_thr = 0.75
        elif self.state == 'SEARCHING':
            # Khi SEARCHING, có thể lỏng hơn một chút
            pre_filter_thr = 0.70

        for box in filtered_pboxes:
            feat = enhanced_body_feature(frame, box, depth_frame, 
                                          self.mb2_sess, color_weight=self._dynamic_color_weight)
            
            # Nếu extract feature lỗi → REJECT (an toàn hơn)
            if feat is None:
                self.get_logger().warn("APPEARANCE FILTER: Rejected detection with invalid feature")
                continue
            
            # Lọc CỨNG: Chỉ giữ detection giống với Anchor
            if self.state == 'LOCKED' and anchor is not None:
                sim = np.dot(feat, anchor)
                
                # THÊM: Kiểm tra cả với target_feature hiện tại
                sim_with_current = np.dot(feat, self.target_feature) if self.target_feature is not None else 0.0
                
                # Phải pass CẢ HAI threshold
                if sim < pre_filter_thr or sim_with_current < (pre_filter_thr - 0.05):
                    self.get_logger().info(
                        f"APPEARANCE FILTER: Rejected detection "
                        f"(anchor_sim={sim:.3f}, current_sim={sim_with_current:.3f})"
                    )
                    continue
            
            final_pboxes.append(box)
            detection_features.append(feat)

        # THÊM: Log để debug
        self.get_logger().info(
            f"Detection pipeline: {len(pboxes)} → "
            f"depth_filter: {len(filtered_pboxes)} → "
            f"appearance_filter: {len(final_pboxes)}"
        )
        
        # === Update DeepSORT tracker ===
        if self.state == 'LOCKED':
            # Custom matching logic cho LOCKED state
            matched_dets, matched_feats = self.locked_mode_tracking(
                frame, depth_frame, final_pboxes, detection_features
            )
            
            if matched_dets is None:  # Track lost or invalid state
                tracks = self.deepsort.update(final_pboxes, detection_features)
            else:
                tracks = self.deepsort.update(matched_dets, matched_feats)
        else:
            # Normal update cho SEARCHING/LOST
            tracks = self.deepsort.update(final_pboxes, detection_features)

        confirmed_tracks = self.deepsort.get_confirmed_tracks()
        
        # Biến cờ để check xem track có được update thực sự không (Chống Ghost)
        is_real_update = False
        
        # ===== STATE: SEARCHING =====
        if self.state == 'SEARCHING':
            best_track = self._find_best_track_by_reid(confirmed_tracks)
            
            if best_track is not None:
                self.state = 'LOCKED'
                self.current_track_id = best_track.track_id
                self.target_box = tuple(map(int, best_track.to_tlbr()))
                self.last_known_depth = self.get_median_depth_at_box(self.target_box, depth_frame)
                self.get_logger().info(f"Target LOCKED track_id={self.current_track_id}, score={self.current_similarity:.2f}")
                self.stop_lost_sound_loop()

        # ===== STATE: LOCKED =====
        elif self.state == 'LOCKED':
            if self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth):
                self.get_logger().info("Target occluded. → OCCLUDED")
                self.state = 'OCCLUDED'
                if self.occl_start_time is None:
                    self.occl_start_time = time.time()
                self.recover_count = 0
                # không return LOST nữa, xử lý OCCLUDED ở phần state bên dưới

            target_track = self.deepsort.get_track_by_id(self.current_track_id)
            reject_thr = self.get_parameter('reject_threshold').value
            
            if target_track is not None and not target_track.is_deleted():
                # GHOST FIX: Kiểm tra xem track có vừa được update bởi detection thật không
                is_real_update = (target_track.time_since_update == 0)
                
                # FIX #6: miss_count để điều hướng SEARCHING khi không phải occlusion
                if is_real_update:
                    self.miss_count = 0
                else:
                    self.miss_count += 1
                
                # Nếu miss quá lâu mà KHÔNG phải occlusion → SEARCHING
                if self.miss_count >= self.MISS_TO_SEARCH:
                    if not self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth):
                        self.get_logger().warn(f"Miss too long ({self.miss_count} frames, not occluded) -> SEARCHING")
                        self.state = 'SEARCHING'
                        self.current_track_id = None
                        self.target_box = None
                        self.miss_count = 0
                        # Không return ở đây, để phần dưới publish state

                new_box = tuple(map(int, target_track.to_tlbr()))
                new_depth = self.get_median_depth_at_box(new_box, depth_frame)
                
                # ===== ANTI-ID-SWITCHING #3: DEPTH JUMP DETECTION =====
                depth_jump_threshold = float(self.get_parameter('depth_jump_threshold').value)
                if (self.last_known_depth is not None and new_depth is not None and
                    self.last_known_depth - new_depth > depth_jump_threshold):
                    self.get_logger().warn(
                        f"DEPTH JUMP: Intruder detected {self.last_known_depth:.2f}m → {new_depth:.2f}m. → LOST"
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
                    # FIX #3: Chỉ update last_known_depth khi is_real_update
                    if is_real_update and (new_depth is not None):
                        self.last_known_depth = new_depth
                
                # Tính similarity với ANCHOR
                track_feature = target_track.get_feature()
                anchor = self.original_target_feature if self.original_target_feature is not None else self.target_feature
                if track_feature is not None and anchor is not None:
                    self.current_similarity = float(np.dot(track_feature, anchor))
                    self.get_logger().info(f"LOCKED: track_id={self.current_track_id}, Similarity={self.current_similarity:.3f}")
                    
                    now = time.time()
                    if (self.current_similarity > reject_thr and 
                        self.current_similarity < self.adaptive_update_threshold and
                        now - self.last_update_time > self.adaptive_update_interval_sec):
                        self.adaptive_model_update(self.target_box, frame, depth_frame)
                        self.last_update_time = now
                    
                    if self.current_similarity < reject_thr:
                        self.get_logger().info(f"Similarity too low ({self.current_similarity:.2f}). → LOST")
                        self.state = 'LOST'
                        self.lost_start_time = time.time()
            else:
                # ===== ANTI-ID-SWITCHING #5: NO RE-MATCH IN LOST =====
                # Track không còn → LOST, KHÔNG tự động match track khác
                self.get_logger().info("Target track lost. → LOST (no re-matching)")
                self.state = 'LOST'
                self.lost_start_time = time.time()

        # ===== STATE: LOST =====
        elif self.state == 'LOST':
            target_track = self.deepsort.get_track_by_id(self.current_track_id)
            
            if target_track is not None and not target_track.is_deleted():
                self.target_box = tuple(map(int, target_track.to_tlbr()))
                
                # Chỉ re-acquire nếu CÙNG track được update
                if target_track.time_since_update == 0:
                    track_feature = target_track.get_feature()
                    anchor = self.original_target_feature if self.original_target_feature is not None else self.target_feature
                    if track_feature is not None and anchor is not None:
                        score = float(np.dot(track_feature, anchor))
                        accept_thr = self.get_parameter('accept_threshold').value
                        if score > accept_thr:
                            self.state = 'LOCKED'
                            self.current_similarity = score
                            self.last_known_depth = self.get_median_depth_at_box(self.target_box, depth_frame)
                            self.get_logger().info(f"Target re-acquired! track_id={self.current_track_id}, score={score:.2f}")
                            self.stop_lost_sound_loop()
            
            # Check grace period
            if self.lost_start_time is not None:
                if time.time() - self.lost_start_time > self.get_parameter('grace_period_sec').value:
                    self.get_logger().info("Grace period expired. → SEARCHING")
                    self.state = 'SEARCHING'
                    self.target_box = None
                    self.current_track_id = None
                    self.start_lost_sound_loop()

        # ===== STATE: OCCLUDED =====
        elif self.state == 'OCCLUDED':
            target_track = self.deepsort.get_track_by_id(self.current_track_id)
            
            # FIX #5.1: Check timeout → SEARCHING (không phải LOST)
            if self.occl_start_time is not None:
                occl_duration = time.time() - self.occl_start_time
                if occl_duration > self.OCCL_MAX_SEC:
                    self.get_logger().warn(f"OCCLUDED timeout ({occl_duration:.1f}s > {self.OCCL_MAX_SEC}s). → SEARCHING")
                    self.state = 'SEARCHING'
                    self.occl_start_time = None
                    self.recover_count = 0
                    self.current_track_id = None
                    self.target_box = None
            else:
                # Check nếu target lộ lại (không còn bị che) → RECOVER
                if target_track is not None and not target_track.is_deleted():
                    self.target_box = tuple(map(int, target_track.to_tlbr()))
                    is_still_occluded = self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth)
                    
                    if not is_still_occluded:
                        self.get_logger().info("Occlusion cleared -> RECOVER")
                        self.state = 'RECOVER'
                        self.recover_count = 0
            
            # stop robot trong OCCLUDED (twist sẽ bị set 0 ở cuối)

        # ===== STATE: RECOVER (New) =====
        elif self.state == 'RECOVER':
            # Trong RECOVER: chọn detection match cực chặt (appearance + depth), cần N frame liên tiếp
            best_box, best_score = self.find_best_match_by_reid(final_pboxes, frame, depth_frame)
            
            if best_box is not None and best_score >= self.RECOVER_THR:
                det_depth = self.get_median_depth_at_box(best_box, depth_frame)
                
                # Check depth gate
                if (det_depth is not None and self.last_known_depth is not None and
                    abs(det_depth - self.last_known_depth) <= self.RECOVER_DEPTH_THR):
                    
                    # Update tracker với box này (cần feature tương ứng)
                    try:
                        idx = final_pboxes.index(best_box)
                        self.deepsort.update([best_box], [detection_features[idx]])
                    except (ValueError, IndexError):
                        # Nếu không tìm được idx, update với empty
                        self.deepsort.update([], [])
                    
                    self.recover_count += 1
                    self.get_logger().info(f"RECOVER: Frame {self.recover_count}/{self.RECOVER_CONFIRM} (score={best_score:.3f})")
                    
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
                    # Depth không match - reset counter
                    self.recover_count = 0
            else:
                # Không có detection match đủ tốt - reset counter
                self.recover_count = 0
            
            # Nếu lại bị che -> quay lại OCCLUDED
            if self.target_box is not None:
                if self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth):
                    self.get_logger().info("RECOVER: Occluded again -> OCCLUDED")
                    self.state = 'OCCLUDED'
                    if self.occl_start_time is None:
                        self.occl_start_time = time.time()
                    self.recover_count = 0

        # --- Command & Publishing ---
        twist, detected, depth_m = self.compute_cmd(W, H, self.target_box)
        
        # GHOST FIX: Nếu đang LOCKED mà chỉ là dự đoán (không có real update) -> Dừng robot
        # OCCLUDED/RECOVER: Luôn dừng robot khi đang OCCLUDED hoặc RECOVER
        if self.state in ('OCCLUDED', 'RECOVER') or (self.state == 'LOCKED' and not is_real_update):
            twist = Twist()  # Vận tốc 0
            
        self.cmd_pub.publish(twist)
        self.flag_pub.publish(Bool(data=(self.state == 'LOCKED')))
        if depth_m is not None:
            self.dist_depth_pub.publish(Float32(data=float(depth_m)))
        self.state_pub.publish(String(data=self.state))

        centered_msg = Bool()
        centered_msg.data = bool((self.state == 'LOCKED') and self._is_centered)
        self.centered_pub.publish(centered_msg)

        self.publish_debug(frame, pboxes, self.target_box, vmean, depth_m)


def main():
    rclpy.init()
    node = PersonDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()