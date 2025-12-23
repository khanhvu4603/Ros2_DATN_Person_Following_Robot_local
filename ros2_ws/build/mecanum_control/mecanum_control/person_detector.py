#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PersonDetector (ROS2) — Single-Target Tracking with Depth Enhancement
Optimized for Orange Pi 5 Plus (CPU-only) and Intel RealSense D455.

Key Features:
- State Machine (AUTO-ENROLL, SEARCHING, LOCKED, LOST) for robust tracking.
- Depth-aware distance control and occlusion handling.
- Enhanced ReID features with depth information.
- CPU optimizations: lower resolution, frame skipping, ROI-based detection.
- Tracker fallback (CSRT) for short-term target loss.
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

def _get_ctor(path):
    cur = cv2
    for name in path.split('.'):
        if not hasattr(cur, name): return None
        cur = getattr(cur, name)
    return cur

def create_tracker():
    for cand in ["legacy.TrackerCSRT_create","TrackerCSRT_create",
                 "legacy.TrackerKCF_create","TrackerKCF_create",
                 "legacy.TrackerMOSSE_create","TrackerMOSSE_create"]:
        c=_get_ctor(cand)
        if callable(c):
            try: return c()
            except Exception: continue
    return None

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
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None, None

    h, w = roi.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Tạo ảnh mục tiêu và thêm padding
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8) # Màu xám padding
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
    """Trích xuất một vector đặc trưng đơn giản từ depth, mô tả hình dạng và khoảng cách."""
    if depth_img is None or box is None:
        return np.zeros(target_size[0] * target_size[1])
        
    x1, y1, x2, y2 = map(int, box)
    roi = depth_img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return np.zeros(target_size[0] * target_size[1])
    
    roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Chuẩn hóa: giá trị gần (số nhỏ) -> 1.0, giá trị xa (số lớn) -> 0.0
    # Giả định target nằm trong khoảng 0.5m đến 5m
    roi_normalized = np.clip((5000 - roi_resized) / 4500.0, 0.0, 1.0)
    
    depth_feat = roi_normalized.flatten().astype(np.float32)
    return depth_feat

def enhanced_body_feature(frame, box, depth_img, ort_sess, color_weight=0.3, normalize_brightness=True):
    """Kết hợp đặc trưng hình thái, màu sắc và depth."""
    # 1. Đặc trưng hình thái từ MobileNetV2
    roi_padded, _ = body_arr_preserve_aspect_ratio(frame, box)
    if roi_padded is None: return None
    
    roi_rgb = cv2.cvtColor(roi_padded, cv2.COLOR_BGR2RGB)
    arr = mb2_preprocess_keras_style(roi_rgb)[None,...]
    
    inp_name = ort_sess.get_inputs()[0].name
    emb = ort_sess.run(None, {inp_name: arr.astype(np.float32)})[0].reshape(-1).astype(np.float32)
    emb /= (np.linalg.norm(emb)+1e-8)

    # 2. Đặc trưng màu sắc
    col = hsv_histogram(roi_padded, bins=16, v_weight=0.6, normalize_brightness=normalize_brightness)

    # 3. Đặc trưng depth
    depth_feat = extract_depth_feature(box, depth_img)
    depth_feat /= (np.linalg.norm(depth_feat) + 1e-8)

    # 4. Kết hợp
    emb_weighted = emb * (1.0 - color_weight)
    col_weighted = col * color_weight
    depth_weighted = depth_feat * 0.1 # Trọng số nhỏ cho depth

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
            ('image_width', 640), ('image_height', 480), # Giảm độ phân giải

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
            ('accept_threshold', 0.75), # Ngưỡng để chấp nhận là target
            ('reject_threshold', 0.6), # Ngưỡng để từ chối khi đã lock
            ('iou_threshold', 0.4),      # Ngưỡng IoU để giữ target
            ('margin_delta', 0.07),
            ('confirm_frames', 5),

            # Chống chói sáng (gốc)
            ('body_color_weight', 0.22),
            ('hsv_normalize_brightness', True),
            ('similarity_ema_alpha', 0.8),

            # Auto-enroll
            ('auto_timeout_sec', 30.0),
            ('auto_body_min', 30),
            ('auto_body_target', 100),

            # Models
            ('mb2_onnx_path', MB2_ONNX_PATH),

            # Occlusion & Lost handling
            ('occlusion_threshold', 0.5), # Ngưỡng depth để phát hiện che khuất
            ('grace_period_sec', 4.0),    # Thời gian chờ khi mất target

            # UDP Streaming
            ('enable_udp_stream', True),
            ('udp_host', '127.0.0.1'),
            ('udp_port', 9999),
            
            # Sound
            ('sound_filename', 'lost_target_viet.wav'),
            ('enroll_sound_filename', 'enroll_viet.wav'),
            ('run_sound_filename', 'run_viet.wav'),
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

        # Models/ReID state
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.SessionOptions().graph_optimization_level.ORT_ENABLE_ALL if hasattr(ort.SessionOptions(), 'graph_optimization_level') else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.mb2_sess   = ort.InferenceSession(
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
        self.get_logger().info("PersonDetector initialized (Debug version with Similarity Log)")

        # --- STATE MACHINE VARIABLES ---
        self.state = 'AUTO-ENROLL'  # AUTO-ENROLL, SEARCHING, LOCKED, LOST
        self.target_box = None
        self.target_feature = None # Đặc trưng của target (sẽ là body_centroid)
        self.last_known_depth = None
        self.tracker = None
        self.lost_start_time = None
        self.current_similarity = 0.0  # Lưu giá trị similarity hiện tại để hiển thị
        
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

        # --- ADAPTIVE MODEL UPDATE ---
        self.adaptive_update_threshold = 0.7
        self.last_update_time = 0.0
        self.adaptive_update_interval_sec = 1.0

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
        """Lấy giá trị depth trung vị (median) tại một bounding box."""
        if depth_img is None or box is None:
            return None
        x1, y1, x2, y2 = map(int, box)
        roi = depth_img[y1:y2, x1:x2]
        if roi.size == 0: return None
        
        valid_pixels = roi[(roi > 100) & (roi < 10000)] # >100mm và <10m
        if valid_pixels.size == 0: return None
            
        # Use 10th percentile to focus on closest object (person)
        depth_m = np.percentile(valid_pixels, 10) / 1000.0 # mm -> m
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
                self.get_logger().info("Target enrolled. Starting search...")
            self.auto_done = True
            self.state = 'SEARCHING'
            
            # Play run sound once after enrollment completes
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

        depth_m = self.get_median_depth_at_box(target_box, self.depth_img)
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

    # ---------- Tracker Management ----------
    def init_tracker(self, frame, box):
        self.tracker = create_tracker()
        if self.tracker:
            x1,y1,x2,y2 = box
            self.tracker.init(frame, (x1, y1, x2-x1, y2-y1))

    #def update_tracker(self, frame):
        # DISABLED TEMPORARILY - tracker đang gây ra false positives
        #return None
        # if self.tracker:
        #     ok, box = self.tracker.update(frame)
        #     if ok:
        #         x, y, w, h = map(int, box)
        #         return (x, y, x+w, y+h)
        # return None

    def update_tracker(self, frame):
    	if self.tracker:
            ok, box = self.tracker.update(frame)
            if ok:
                x, y, w, h = map(int, box)
                return (x, y, x+w, y+h)
        # return None

    # ---------- Lost Sound Loop (Threading) ----------
    def _lost_sound_loop(self):
        """Thread function to play lost_target sound in a loop until stopped."""
        while not self.stop_lost_sound_event.is_set():
            if os.path.exists(self.sound_file):
                os.system(f"aplay {self.sound_file}")
            # Small delay to prevent CPU spinning if file doesn't exist
            time.sleep(0.5)

    def start_lost_sound_loop(self):
        """Start playing lost target sound in a loop."""
        if self.lost_sound_thread is not None and self.lost_sound_thread.is_alive():
            return  # Already playing
        
        self.stop_lost_sound_event.clear()
        self.lost_sound_thread = threading.Thread(target=self._lost_sound_loop, daemon=True)
        self.lost_sound_thread.start()
        self.get_logger().info("Started lost target sound loop.")

    def stop_lost_sound_loop(self):
        """Stop the lost target sound loop."""
        if self.lost_sound_thread is None or not self.lost_sound_thread.is_alive():
            return  # Not playing
        
        self.stop_lost_sound_event.set()
        self.lost_sound_thread.join(timeout=2.0)
        self.lost_sound_thread = None
        self.get_logger().info("Stopped lost target sound loop.")

    # ---------- Matching ----------
    def find_best_match_by_reid(self, boxes, frame, depth_frame):
        best_box, best_score = None, -1.0
        for box in boxes:
            feat = enhanced_body_feature(frame, box, depth_frame, self.mb2_sess, color_weight=self._dynamic_color_weight)
            if feat is None: continue
            
            score = np.dot(feat, self.target_feature)
            if score > best_score:
                best_score = score
                best_box = box
        return best_box, best_score

    def find_best_match_by_iou(self, boxes, target_box, frame, depth_frame):
        best_box, best_score = None, -1.0
        iou_thr = self.get_parameter('iou_threshold').value
        for box in boxes:
            iou_score = iou(box, target_box)
            if iou_score < iou_thr: continue
            
            feat = enhanced_body_feature(frame, box, depth_frame, self.mb2_sess, color_weight=self._dynamic_color_weight)
            if feat is None: continue
            
            score = np.dot(feat, self.target_feature)
            if score > best_score:
                best_score = score
                best_box = box
        return best_box, best_score

    # ---------- Adaptive Model Update ----------
    def adaptive_model_update(self, box, frame, depth_frame):
        """Hàm cập nhật model một cách thông minh khi cần thiết."""
        if box is None or self.target_feature is None:
            return

        # 1. Trích xuất đặc trưng của mẫu ứng viên
        candidate_feat = enhanced_body_feature(
            frame, box, depth_frame, self.mb2_sess,
            color_weight=self._dynamic_color_weight
        )
        if candidate_feat is None:
            return

        # 2. Kiểm tra độ tin cậy cơ bản của ReID
        similarity_with_centroid = float(np.dot(candidate_feat, self.target_feature))
        if similarity_with_centroid < self.get_parameter('reject_threshold').value:
            self.get_logger().warn(f"Update rejected: low similarity {similarity_with_centroid:.2f}")
            return

        # 3. Diversity check: tránh cập nhật với mẫu quá giống
        if similarity_with_centroid > 0.99:
            self.get_logger().info("Update skipped: sample too similar to current model.")
            return

        # 4. Cập nhật model
        self.update_target_model(candidate_feat)
        self.get_logger().info(f"Model updated. New similarity: {similarity_with_centroid:.2f}")

    def update_target_model(self, new_feature):
        """
        Cập nhật target_feature (centroid) bằng EMA.
        Mẫu mới có trọng số cao hơn, giúp model thích ứng nhanh.
        """
        if self.target_feature is None:
            self.target_feature = new_feature.astype(np.float32)
        else:
            alpha = 0.2  # Hệ số học
            self.target_feature = (1.0 - alpha) * self.target_feature + alpha * new_feature

        self.target_feature /= (np.linalg.norm(self.target_feature) + 1e-8)

    # ---------- Debug Publisher (ROS + UDP) ----------
    def publish_debug(self, frame, pboxes, target_box, vmean, depth_m):
        publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        if not publish_debug_image and not self.enable_udp:
            return

        dbg = frame.copy()

        # Draw boxes
        for pb in pboxes:
            if target_box is not None and iou(pb, target_box) >= 0.99:
                label = "TARGET" if self._is_centered else "CENTERING"
                draw_labeled_box(dbg, pb, color=(0,0,255), label=label)
            else:
                cv2.rectangle(dbg, (pb[0], pb[1]), (pb[2], pb[3]), (0,255,0), 2)

        # State text
        status = self.state
        cv2.putText(
            dbg, f"State: {status}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (50,220,50) if status == 'LOCKED' else (0,165,255),
            2
        )
        
        # Similarity score (below State)
        if status == 'LOCKED':
            cv2.putText(
                dbg, f"Similarity: {self.current_similarity:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (50, 220, 50),  # Match LOCKED status color
                2
            )

        # Depth + mode HUD (top-right)
        depth_show = self.last_known_depth if self.last_known_depth is not None else depth_m
        depth_txt = "--" if depth_show is None else f"{float(depth_show):.2f} m"
        mode_txt  = "Centered" if self._is_centered else "Centering"
        hud_right = f"Depth: {depth_txt}   Mode: {mode_txt}"
        draw_label_top_right(dbg, hud_right, margin=10)

        # Low-light / backlit hint
        if vmean < 90 or vmean > 200:
            cv2.putText(
                dbg, "LOW-LIGHT / BACKLIT MODE",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,140,255),
                2
            )

        # Publish ROS debug image
        if publish_debug_image:
            try:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
            except Exception:
                pass

        # Send UDP
        if self.enable_udp:
            try:
                ret, buffer = cv2.imencode('.jpg', dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    self.udp_sock.sendto(buffer.tobytes(), (self.udp_host, self.udp_port))
            except Exception:
                pass

    # ---------- Image callback ----------
    def on_image(self, msg: Image):
        # --- CPU Optimization: Frame Skipping ---
        self.frame_count += 1
        if self.frame_count % 1 != 0: # Process every 2nd frame
            return

        # --- Image Acquisition & Resizing ---
        frame0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        W = int(self.get_parameter('image_width').value)
        H = int(self.get_parameter('image_height').value)
        frame = cv2.resize(frame0, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Resize depth to match color frame
        depth_frame = cv2.resize(self.depth_img, (W, H), interpolation=cv2.INTER_NEAREST) if self.depth_img is not None else None

        # --- Dynamic Color Weight Adjustment ---
        vmean = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2])
        base_cw = float(self.get_parameter('body_color_weight').value)
        if vmean < 90 or vmean > 200:
            self._dynamic_color_weight = min(0.10, base_cw * 0.6)
        else:
            self._dynamic_color_weight = base_cw

        # --- Detect persons (once per frame) ---
        pboxes, _ = self.detect_persons(frame, conf_thresh=0.4)

        # --- Enrollment Phase ---
        if not self.auto_done:
            self.state = 'AUTO-ENROLL'
            
            # Play enroll sound once
            if not self.enroll_audio_played:
                if os.path.exists(self.enroll_sound_file):
                    os.system(f"(aplay {self.enroll_sound_file}; aplay {self.enroll_sound_file}) &")
                self.enroll_audio_played = True

            self.auto_enroll_step(frame, pboxes)

            # Publish state + simple flags (robot không di chuyển)
            self.state_pub.publish(String(data=self.state))
            self.flag_pub.publish(Bool(data=False))
            self.centered_pub.publish(Bool(data=False))

            # Hiển thị hình + UDP trong lúc enroll
            self.publish_debug(frame, pboxes, None, vmean, None)
            return

        # --- Main State Machine ---
        if self.state == 'SEARCHING':
            best_match_box, best_match_score = self.find_best_match_by_reid(pboxes, frame, depth_frame)
            self.current_similarity = best_match_score if best_match_score >= 0 else 0.0  # Cập nhật similarity
            accept_thr = self.get_parameter('accept_threshold').value
            if best_match_box and best_match_score > accept_thr:
                self.state = 'LOCKED'
                self.target_box = best_match_box
                self.last_known_depth = self.get_median_depth_at_box(self.target_box, depth_frame)
                self.get_logger().info(f"Target LOCKED with score: {best_match_score:.2f}")
                self.init_tracker(frame, self.target_box)
                self.stop_lost_sound_loop()  # Stop lost sound when target found

        elif self.state == 'LOCKED':
            # 1. Occlusion Check
            if self.is_target_occluded(self.target_box, depth_frame, self.last_known_depth):
                self.get_logger().info("Target occluded. Switching to LOST state.")
                self.state = 'LOST'
                self.lost_start_time = time.time()

                # Publish state + debug rồi dừng frame này
                self.state_pub.publish(String(data=self.state))
                self.flag_pub.publish(Bool(data=False))
                self.centered_pub.publish(Bool(data=False))
                self.publish_debug(frame, pboxes, self.target_box, vmean, None)
                return

            # 2. Find best match by IoU
            current_box, current_score = self.find_best_match_by_iou(pboxes, self.target_box, frame, depth_frame)
            self.current_similarity = current_score if current_score >= 0 else 0.0  # Cập nhật similarity
            self.get_logger().info(f"LOCKED: Similarity={self.current_similarity:.3f}")
            reject_thr = self.get_parameter('reject_threshold').value

            # 2.1 Adaptive model update - CHỈ khi detector phát hiện được người (pboxes không rỗng)
            # và current_box là từ detector (không phải tracker fallback)
            now = time.time()
            if (len(pboxes) > 0 and  # QUAN TRỌNG: kiểm tra detector có tìm được người không
                current_box is not None and 
                current_score < self.adaptive_update_threshold and
                now - self.last_update_time > self.adaptive_update_interval_sec):
                self.adaptive_model_update(current_box, frame, depth_frame)  #comment sẽ ko online update đc 
                self.last_update_time = now
            
            if current_box and current_score > reject_thr:
                self.target_box = current_box
                self.last_known_depth = self.get_median_depth_at_box(self.target_box, depth_frame)
                self.init_tracker(frame, self.target_box)
            else:
                # Lost, try tracker fallback
                tracker_box = self.update_tracker(frame)
                if tracker_box:
                    # Verify tracker result with ReID to avoid tracking wrong object/background
                    tracker_feat = enhanced_body_feature(frame, tracker_box, depth_frame, 
                                                         self.mb2_sess, color_weight=self._dynamic_color_weight)
                    if tracker_feat is not None:
                        tracker_score = float(np.dot(tracker_feat, self.target_feature))
                        if tracker_score > reject_thr:
                            self.target_box = tracker_box
                        else:
                            # Tracker is tracking wrong object/background
                            self.get_logger().info(f"Tracker verification failed (score={tracker_score:.2f}). Switching to LOST.")
                            self.state = 'LOST'
                            self.lost_start_time = time.time()
                    else:
                        # Cannot extract feature - likely no person there
                        self.get_logger().info("Tracker box has no valid features. Switching to LOST.")
                        self.state = 'LOST'
                        self.lost_start_time = time.time()
                else: # Tracker cũng fail
                    self.get_logger().info("Target lost. Switching to LOST state.")
                    self.state = 'LOST'
                    self.lost_start_time = time.time()

        elif self.state == 'LOST':
            # 1. Try tracker
            tracker_box = self.update_tracker(frame)
            if tracker_box:
                self.target_box = tracker_box
                # 2. Check if a valid person is at the predicted location
                match_box, match_score = self.find_best_match_by_reid([tracker_box], frame, depth_frame)
                accept_thr = self.get_parameter('accept_threshold').value
                if match_box and match_score > accept_thr:
                    self.state = 'LOCKED'
                    self.get_logger().info("Target re-acquired!")
                    self.init_tracker(frame, self.target_box)
                    self.stop_lost_sound_loop()  # Stop lost sound when target re-acquired
            # 3. Check grace period
            if time.time() - self.lost_start_time > self.get_parameter('grace_period_sec').value:
                self.get_logger().info("Grace period expired. Returning to SEARCHING.")
                
                self.state = 'SEARCHING'
                self.target_box = None
                self.tracker = None
                self.start_lost_sound_loop()  # Start playing lost sound only when entering SEARCHING

        # --- Command & Publishing ---
        twist, detected, depth_m = self.compute_cmd(W, H, self.target_box)
        self.cmd_pub.publish(twist)
        self.flag_pub.publish(Bool(data=(self.state == 'LOCKED')))
        if depth_m is not None:
            self.dist_depth_pub.publish(Float32(data=float(depth_m)))
        self.state_pub.publish(String(data=self.state))

        centered_msg = Bool()
        centered_msg.data = bool((self.state == 'LOCKED') and self._is_centered)
        self.centered_pub.publish(centered_msg)

        # --- Debug / UDP ---
        self.publish_debug(frame, pboxes, self.target_box, vmean, depth_m)


def main():
    rclpy.init()
    node = PersonDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
