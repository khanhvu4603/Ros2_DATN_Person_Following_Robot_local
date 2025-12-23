#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PersonDetector (ROS2) — BODY-first ReID + Face assist + Temporal sticky + BodyConf HUD
+ CHỐNG CHÓI SÁNG: OSNet + LBP, dynamic LBP weight, spatial+temporal scoring

Pipeline:
Camera + Depth → Preprocess + vmean.
Detector (dual) → Detections.
Depth → 3D-ish (center+Z) + Tracker → State (Traj + Temporal).
ROI → OSNet emb + Multi-scale Uniform LBP → S_app.
Spatial + Trajectory + Temporal → S_spat, S_traj, S_temp.
Adaptive weights → S_total.
State machine → LOCK/UNLOCK + chọn Target_box.
Control (center-first + depth follow).
Online update centroid (OCL-style).
"""

import time
from typing import Optional, List
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

# ========== (Optional) Face via insightface ==========
try:
    from insightface.app import FaceAnalysis
    _FACE_OK = True
except Exception:
    FaceAnalysis = None
    _FACE_OK = False

# ========== Paths ==========
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True, parents=True)
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True, parents=True)

MB2_ONNX_PATH  = str(MODELS / "osnet_x0_25_msmt17.onnx")  # giờ trỏ tới OSNet-0.25
MOBILENET_PROTOTXT = str(MODELS / "MobileNetSSD_deploy.prototxt")
MOBILENET_WEIGHTS  = str(MODELS / "MobileNetSSD_deploy.caffemodel")


# ========== Helpers ==========
def clamp(x, a, b):
    return a if x < a else b if x > b else x


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
    return inter / ua if ua > 0 else 0.0


def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def expand(box, shape, m=0.20):
    x1, y1, x2, y2 = box
    H, W = shape[:2]
    w = x2 - x1
    h = y2 - y1
    x1 = max(0, int(x1 - m * w))
    y1 = max(0, int(y1 - m * h))
    x2 = min(W - 1, int(x2 + m * w))
    y2 = min(H - 1, int(y2 + m * h))
    return (x1, y1, x2, y2)


def _get_ctor(path):
    cur = cv2
    for name in path.split('.'):
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def create_tracker():
    for cand in [
        "legacy.TrackerCSRT_create", "TrackerCSRT_create",
        "legacy.TrackerKCF_create", "TrackerKCF_create",
        "legacy.TrackerMOSSE_create", "TrackerMOSSE_create"
    ]:
        c = _get_ctor(cand)
        if callable(c):
            try:
                return c()
            except Exception:
                continue
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


def draw_labeled_box(img, box, color=(0, 0, 255), label="TARGET"):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx1, ty1 = x1, max(0, y1 - th - 10)
    tx2, ty2 = x1 + tw + 12, ty1 + th + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.putText(img, label, (tx1 + 6, ty2 - 6),
            font, scale, (255, 255, 255), thickness, cv2.LINE_AA)



# =================== BODY features (OSNet-0.25 + multi-scale LBP) ===================
def mb2_preprocess_keras_style(x_uint8):
    """
    Reuse tên hàm cũ cho đỡ phải sửa – chuẩn hóa [-1,1].
    OSNet cũng xài kiểu này được (tùy bạn train).
    """
    x = x_uint8.astype(np.float32)
    x = x / 127.5 - 1.0
    return x


_LBP_UNIFORM_LUT = None


def _get_uniform_lbp_lut(P=8):
    """
    LUT uniform LBP cho P=8.
    code có <=2 transitions -> bin = số bit 1
    ngược lại -> bin P+1
    """
    global _LBP_UNIFORM_LUT
    if _LBP_UNIFORM_LUT is not None:
        return _LBP_UNIFORM_LUT
    lut = np.zeros(256, dtype=np.uint8)
    for code in range(256):
        bits = [(code >> (P - 1 - i)) & 1 for i in range(P)]
        transitions = 0
        for i in range(P):
            if bits[i] != bits[(i + 1) % P]:
                transitions += 1
        if transitions <= 2:
            idx = sum(bits)
        else:
            idx = P + 1
        lut[code] = idx
    _LBP_UNIFORM_LUT = lut
    return lut


def lbp_uniform_hist(roi_gray, radius_list=(1, 2, 3), num_points=8):
    """
    Multi-scale uniform LBP histogram (vectorized).
    Output: concat hist cho mỗi scale, đã L2-normalize.
    """
    roi_gray = roi_gray.astype(np.uint8)
    H, W = roi_gray.shape
    lut = _get_uniform_lbp_lut(P=num_points)
    feats = []
    for r in radius_list:
        if H <= 2 * r or W <= 2 * r:
            continue

        center = roi_gray[r:H - r, r:W - r]

        # neighbors (8 hướng, integer shift)
        n0 = roi_gray[0:H - 2 * r,      0:W - 2 * r]      # up-left
        n1 = roi_gray[0:H - 2 * r,      r:W - r]          # up
        n2 = roi_gray[0:H - 2 * r,      2 * r:W]          # up-right
        n3 = roi_gray[r:H - r,          2 * r:W]          # right
        n4 = roi_gray[2 * r:H,          2 * r:W]          # down-right
        n5 = roi_gray[2 * r:H,          r:W - r]          # down
        n6 = roi_gray[2 * r:H,          0:W - 2 * r]      # down-left
        n7 = roi_gray[r:H - r,          0:W - 2 * r]      # left

        neighbors = [n0, n1, n2, n3, n4, n5, n6, n7]
        codes = np.zeros_like(center, dtype=np.uint8)

        for idx, neigh in enumerate(neighbors):
            h_c, w_c = center.shape
            nh, nw = neigh.shape
            h_min = min(h_c, nh)
            w_min = min(w_c, nw)
            c = center[:h_min, :w_min]
            n = neigh[:h_min, :w_min]
            bit = (n >= c).astype(np.uint8)
            codes[:h_min, :w_min] |= (bit << (7 - idx))

        mapped = lut[codes]
        hist = np.bincount(mapped.reshape(-1), minlength=num_points + 2).astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        feats.append(hist)

    if not feats:
        hist = np.ones(num_points + 2, dtype=np.float32)
        hist /= hist.sum()
        return hist

    feat = np.concatenate(feats).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-8)
    return feat


def body_arr(frame, box):
    """
    Cắt ROI & resize cho OSNet:
    - Expand box 20%
    - Resize (H,W) = (256,128)
    """
    x1, y1, x2, y2 = expand(box, frame.shape, 0.20)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    roi_resized = cv2.resize(roi, (128, 256), interpolation=cv2.INTER_LINEAR)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    arr = mb2_preprocess_keras_style(roi_rgb)[None, ...]
    return roi_resized, arr


def body_feature_onnx(frame, box, ort_sess,
                      color_weight=0.3,
                      normalize_brightness=True):
    """
    OSNet embedding + multi-scale uniform LBP.
    color_weight: LBP weight (0..1)
    """
    out = body_arr(frame, box)
    if out is None:
        return None
    roi_resized, arr = out

    try:
        inp_name = ort_sess.get_inputs()[0].name
    except Exception:
        return None

    x = arr.astype(np.float32)
    x = np.transpose(x, (0, 3, 1, 2))  # NCHW
    try:
        emb = ort_sess.run(None, {inp_name: x})[0].reshape(-1).astype(np.float32)
    except Exception:
        return None
    emb /= (np.linalg.norm(emb) + 1e-8)

    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    lbp_hist = lbp_uniform_hist(roi_gray, radius_list=(1, 2, 3), num_points=8)

    lbp_w = float(clamp(color_weight, 0.0, 1.0))
    deep_w = 1.0 - lbp_w
    emb_weighted = emb * deep_w
    lbp_weighted = lbp_hist * lbp_w

    feat = np.concatenate([emb_weighted, lbp_weighted], axis=0).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-8)
    return feat


# =================== Detector (MobileNet-SSD) ===================
def _load_ssd():
    if Path(MOBILENET_PROTOTXT).exists() and Path(MOBILENET_WEIGHTS).exists():
        return cv2.dnn.readNetFromCaffe(MOBILENET_PROTOTXT, MOBILENET_WEIGHTS)
    return None


def _ssd_detect(net, frame, conf_thresh=0.4):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    det = net.forward()
    boxes, scores = [], []
    for i in range(det.shape[2]):
        conf = det[0, 0, i, 2]
        cls = int(det[0, 0, i, 1])
        if cls == 15 and conf > conf_thresh:
            box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
                scores.append(float(conf))
    return boxes, scores


# =================== Face (optional) ===================
def build_face_app(det_size=(384, 384), providers=None):
    if not _FACE_OK:
        return None
    app = FaceAnalysis(name='buffalo_sc', providers=providers or ['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def detect_faces_arcface(app, frame_bgr, conf_thresh=0.5, offset_xy=(0, 0)):
    if app is None:
        return []
    ox, oy = offset_xy
    faces = app.get(frame_bgr)
    out = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox[:4])
        if getattr(f, 'det_score', 1.0) < conf_thresh:
            continue
        feat = f.normed_embedding
        if feat is None or feat.size == 0:
            continue
        out.append(dict(
            box=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
            kps=f.kps,
            feat=feat.astype(np.float32)
        ))
    return out


# =================== Backlight enhancement utils ===================
def mean_brightness_v(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 2].mean())


def apply_grayworld(img):
    img = img.astype(np.float32)
    m = img.reshape(-1, 3).mean(axis=0) + 1e-6
    scale = 128.0 / m
    out = np.clip(img * scale, 0, 255).astype(np.uint8)
    return out


def apply_clahe_v(img_bgr, clip=2.0, tile=(8, 8)):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    hsv[..., 2] = clahe.apply(v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def auto_gamma(img_bgr):
    v = mean_brightness_v(img_bgr) / 255.0
    v = max(1e-3, min(0.99, v))
    gamma = np.log(0.5) / np.log(v)
    inv = 1.0 / max(0.1, min(5.0, gamma))
    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((lut ** inv) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)


def enhance_backlit(img_bgr):
    x = apply_grayworld(img_bgr)
    x = apply_clahe_v(x, clip=2.0, tile=(8, 8))
    x = auto_gamma(x)
    return x


def nms_merge_xyxy(boxes, scores, iou_thr=0.5, topk=100):
    if not boxes:
        return [], []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) >= topk:
            break
        xx1 = np.maximum(b[i, 0], b[order[1:], 0])
        yy1 = np.maximum(b[i, 1], b[order[1:], 1])
        xx2 = np.minimum(b[i, 2], b[order[1:], 2])
        yy2 = np.minimum(b[i, 3], b[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        area_o = (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1])
        ovr = inter / (area_i + area_o - inter + 1e-6)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    keep_boxes = [boxes[k] for k in keep]
    keep_scores = [scores[k] for k in keep]
    return keep_boxes, keep_scores


# =================== ROS2 Node ===================
class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')

        self.declare_parameters('', [
            ('camera_topic', '/camera/d455/color/image_raw'),
            ('publish_debug_image', True),
            ('image_width', 960), ('image_height', 540),

            # Depth follow
            ('use_depth', True),
            ('depth_topic', '/camera/d455/depth/image_rect_raw'),
            ('depth_encoding', '16UC1'),
            ('target_distance_m', 1.6),
            ('kd_distance', 0.6),
            ('v_forward_max', 0.3),

            # Heading control
            ('kx_center', 0.00018),
            ('wz_max', 0.25),
            ('center_deadband_px', 30),
            ('center_first', True),
            ('center_release_px', 45),

            # Detector/ReID thresholds
            ('person_conf', 0.3),
            ('accept_face', 0.2),
            ('accept_body', 0.5),
            ('iou_sticky', 0.25),
            ('margin_delta', 0.06),
            ('confirm_frames', 5),

            # LBP / appearance
            ('body_color_weight', 0.2),          # base LBP weight
            ('hsv_normalize_brightness', True),
            ('similarity_ema_alpha', 0.8),

            # Auto-enroll
            ('auto_timeout_sec', 20.0),
            ('auto_body_min', 5),
            ('auto_body_target', 20),

            # Models
            ('mb2_onnx_path', MB2_ONNX_PATH),
            ('face_enable', False),

            # Online enroll (tạm giữ param)
            ('online_enroll_enabled', True),
            ('online_enroll_min_frames', 10),
            ('online_enroll_rate_hz', 0.5),  #2s them 1 sample
            ('online_enroll_body_max', 10000),
            ('online_enroll_face_max', 100),
            ('online_enroll_iou_stable', 0.50),
            ('online_enroll_min_box_ratio', 0.03),
            ('online_enroll_depth_tol_m', 0.20),
            ('online_enroll_diversity_cos', 0.99),

            # MẤT là UNLOCK ngay
            ('unlock_on_miss_frames', 8),

            # Phải thấy lại IOU với bbox hiện tại để coi là LOCKED
            ('lock_requires_visible_iou', 0.25),

            # tracker fallback grace frames
            ('tracker_grace_frames', 2),

            # ROI ReID limits
            ('reid_max_roi_lock', 2),
            ('reid_max_roi_search', 3),
        ])

        # QoS
        color_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST)
        depth_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST)

        # Bridge & subs
        self.bridge = CvBridge()
        cam_topic = self.get_parameter('camera_topic').value
        self.create_subscription(Image, cam_topic, self.on_image, color_qos)

        self.depth_img = None
        self.depth_enc = None
        if bool(self.get_parameter('use_depth').value):
            self.create_subscription(Image, self.get_parameter('depth_topic').value,
                                     self.on_depth, depth_qos)

        # Publishers
        self.cmd_pub        = self.create_publisher(Twist,  '/cmd_vel_person', 10)
        self.flag_pub       = self.create_publisher(Bool,   '/person_detected', 10)
        self.debug_pub      = self.create_publisher(Image,  '/person_detector/debug_image', 1)
        self.state_pub      = self.create_publisher(String, '/person_detector/follow_state', 10)
        self.dist_depth_pub = self.create_publisher(Float32, '/person_distance', 10)

        # Models/ReID state
        self.mb2_sess = ort.InferenceSession(
            self.get_parameter('mb2_onnx_path').value,
            providers=["CPUExecutionProvider"]
        )
        self.face_app = build_face_app() if (_FACE_OK and bool(self.get_parameter('face_enable').value)) else None

        # Detector
        self.ssd_net = _load_ssd()
        if self.ssd_net is None:
            raise FileNotFoundError(
                "Không tìm thấy MobileNet-SSD Caffe. Hãy đặt file vào:\n"
                f"  - {MOBILENET_PROTOTXT}\n  - {MOBILENET_WEIGHTS}"
            )
        self.get_logger().info("Person detector: MobileNet-SSD (Caffe)")

        # ReID / admin
        self.face_db = None
        self.body_centroid = None
        self.auto_start_ts = None
        self.auto_done = False
        self.body_samples: List[np.ndarray] = []
        self.face_samples: List[np.ndarray] = []

        self._body_sim_ema = {}  # {box_key: ema_score}

        self.admin = dict(
            face_box=None,
            person_box=None,
            score_ema=0.0,
            frames_seen=0,
            frames_missing=0
        )
        self.tracker = None
        self._tracker_miss = 0

        # depth overlay
        self.depth_display = None
        self.depth_ema_alpha = 0.6
        self.state_txt = "INIT"

        # online enroll state (giữ chỗ)
        self._online_good_frames = 0
        self._online_last_add_ts = 0.0
        self._last_depth_for_enroll = None
        self._last_person_box_for_enroll = None
        self._last_body_feat_for_enroll = None
        self._last_face_emb_for_enroll = None

        # Center-first hysteresis
        self._is_centered = False

        # HUD body conf
        self._hud_body_conf: Optional[float] = None

        # dynamic LBP weight & lighting mode
        self._dynamic_color_weight = float(self.get_parameter('body_color_weight').value)
         # DEBUG: frame counter + auto-enroll log flag
        self._frame_id = 0
        self._auto_debug_logged = False

        self._lighting_mode = "NORMAL"

        # trajectory cho spatio-temporal scoring
        self.traj_centers_2d: List[tuple] = []
        self.traj_depths: List[float] = []
        self.traj_max_len = 30

    # ---------- Depth ----------
    def on_depth(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding=self.get_parameter('depth_encoding').value
        )
        self.depth_enc = self.get_parameter('depth_encoding').value

    def depth_at_box(self, box_resized, resized_size, depth_img) -> Optional[float]:
        if depth_img is None or box_resized is None:
            return None
        W_r, H_r = resized_size
        depth_h, depth_w = depth_img.shape[:2]
        cx_r, cy_r = center_of(box_resized)
        cx = int(clamp(cx_r * (depth_w / float(W_r)), 0, depth_w - 1))
        cy = int(clamp(cy_r * (depth_h / float(H_r)), 0, depth_h - 1))
        k = 3
        x1 = max(0, cx - k)
        x2 = min(depth_w, cx + k + 1)
        y1 = max(0, cy - k)
        y2 = min(depth_h, cy + k + 1)
        roi = depth_img[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        if self.depth_enc == '16UC1':
            roi_mm = roi.astype(np.float32)
            valid = roi_mm > 0
            if not np.any(valid):
                return None
            depth_m = float(np.median(roi_mm[valid])) / 1000.0
        else:  # 32FC1
            roi_m = roi.astype(np.float32)
            valid = np.isfinite(roi_m) & (roi_m > 0)
            if not np.any(valid):
                return None
            depth_m = float(np.median(roi_m[valid]))

        if np.isnan(depth_m) or depth_m < 0.05 or depth_m > 8.0:
            return None
        return depth_m

    # ---------- Auto-enroll ----------
    # ---------- Auto-enroll ----------
    def auto_enroll_step(self, frame, pboxes, faces):
        now = time.time()
        if self.auto_start_ts is None:
            self.auto_start_ts = now

        # DEBUG: mỗi 30 frame log thử xem có detect được người & số sample hiện tại
        if hasattr(self, "_frame_id") and (self._frame_id % 30 == 0):
            self.get_logger().info(
                f"REID [AUTO] frame={self._frame_id}, num_boxes={len(pboxes)}, "
                f"body_samples={len(self.body_samples)}"
            )

        # [MOD] dynamic color weight
        color_w = getattr(self, '_dynamic_color_weight', float(self.get_parameter('body_color_weight').value))
        norm_bright = bool(self.get_parameter('hsv_normalize_brightness').value)

        # Thu body feature từ box lớn nhất (nếu có người)
        if pboxes:
            j = int(np.argmax([(pb[2] - pb[0]) * (pb[3] - pb[1]) for pb in pboxes]))
            pb = pboxes[j]
            feat = body_feature_onnx(frame, pb, self.mb2_sess,
                                     color_weight=color_w,
                                     normalize_brightness=norm_bright)
            if feat is not None:
                self.body_samples.append(feat)
                if self.body_centroid is None:
                    self.body_centroid = feat.copy()
                else:
                    self.body_centroid = 0.9 * self.body_centroid + 0.1 * feat
                    self.body_centroid /= (np.linalg.norm(self.body_centroid) + 1e-8)

        # Thu face samples (nếu có)
        if faces:
            faces = sorted(
                faces,
                key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]),
                reverse=True
            )
            self.face_samples.append(faces[0]["feat"])

        timeout = float(self.get_parameter('auto_timeout_sec').value)
        body_target = int(self.get_parameter('auto_body_target').value)
        body_min = int(self.get_parameter('auto_body_min').value)

        # Khi hết thời gian hoặc đủ target, kiểm tra xem có đủ mẫu không
        if (now - self.auto_start_ts) >= timeout or len(self.body_samples) >= body_target:
            # Nếu không đủ mẫu body → KHÔNG cho auto_done, tiếp tục AUTO-ENROLL
            if len(self.body_samples) < max(1, body_min):
                if not getattr(self, "_auto_debug_logged", False):
                    self.get_logger().warn(
                        f"REID [AUTO] not enough body samples "
                        f"(have={len(self.body_samples)}, need>={body_min}). "
                        f"Still in AUTO-ENROLL, please stand in front of camera."
                    )
                    self._auto_debug_logged = True
                # reset lại time để cho thêm cơ hội enroll
                self.auto_start_ts = now
                return

            # Nếu đủ mẫu → tính centroid cho face & body
            if len(self.face_samples) >= 10:
                E = np.array(self.face_samples, dtype=np.float32)
                cent = E.mean(axis=0)
                cent /= (np.linalg.norm(cent) + 1e-8)
                self.face_db = (E, cent)

            if self.body_centroid is None and len(self.body_samples) >= 10:
                B = np.array(self.body_samples, dtype=np.float32)
                cent = B.mean(axis=0)
                cent /= (np.linalg.norm(cent) + 1e-8)
                self.body_centroid = cent

            self.auto_done = True
            self.state_txt = "RUN"

            # DEBUG: log auto-enroll result (chạy 1 lần khi DONE thật sự)
            if not getattr(self, "_auto_debug_logged", False):
                body_cnt = len(self.body_samples)
                has_centroid = self.body_centroid is not None
                cent_norm = float(np.linalg.norm(self.body_centroid)) if has_centroid else 0.0
                self.get_logger().info(
                    f"REID [AUTO_DONE] body_samples={body_cnt}, "
                    f"body_centroid_is_none={not has_centroid}, "
                    f"centroid_norm={cent_norm:.4f}"
                )
                self._auto_debug_logged = True



    # ---------- ROI selection for ReID ----------
    def select_reid_rois(self, pboxes, scores):
        """
        Chọn index các box sẽ chạy ReID (OSNet+LBP).
        Khi LOCK: ưu tiên box gần admin + 1 box tiềm năng.
        Khi SEARCHING: lấy top-k theo score*area.
        """
        n = len(pboxes)
        if n == 0:
            return []

        max_lock = int(self.get_parameter('reid_max_roi_lock').value)
        max_search = int(self.get_parameter('reid_max_roi_search').value)
        admin_box = self.admin.get("person_box", None)
        state_locked = (admin_box is not None and self.admin.get("frames_seen", 0) > 0)
        max_roi = max_lock if state_locked else max_search
        max_roi = max(1, max_roi)

        if n <= max_roi:
            return list(range(n))

        areas = [(pb[2] - pb[0]) * (pb[3] - pb[1]) for pb in pboxes]
        indices = []

        if state_locked and admin_box is not None:
            # ROI chính: IOU lớn nhất với admin_box
            best_i = None
            best_iou = 0.0
            for i, pb in enumerate(pboxes):
                iou_val = iou(pb, admin_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_i = i
            if best_i is None:
                best_i = int(np.argmax(areas))
            indices.append(best_i)

            # ROI phụ: box tiềm năng gần đó (score*area cao)
            if max_roi > 1:
                cand = [i for i in range(n) if i not in indices]
                if cand:
                    score_area = [scores[i] * areas[i] for i in cand]
                    j = int(np.argmax(score_area))
                    indices.append(cand[j])
        else:
            # SEARCHING: top-k theo score*area
            score_area = [scores[i] * areas[i] for i in range(n)]
            order = np.argsort(score_area)[::-1]
            indices = [int(i) for i in order[:max_roi]]

        return indices

    # ---------- Select admin ----------
    # ---------- Select admin ----------
    def select_admin(self, frame, pboxes, pscores, faces):
        # [MOD] dynamic color weight
        color_w = getattr(self, '_dynamic_color_weight', float(self.get_parameter('body_color_weight').value))
        norm_bright = bool(self.get_parameter('hsv_normalize_brightness').value)
        ema_alpha = float(self.get_parameter('similarity_ema_alpha').value)

        body_confs = []
        body_feats = []
        for i, pb in enumerate(pboxes):
            feat = body_feature_onnx(frame, pb, self.mb2_sess,
                                     color_weight=color_w,
                                     normalize_brightness=norm_bright)
            body_feats.append(feat)
            if feat is None:
                body_confs.append(0.0)
                continue

            if self.body_centroid is not None:
                cos = float(np.dot(feat, self.body_centroid) /
                            (np.linalg.norm(feat) * np.linalg.norm(self.body_centroid) + 1e-8))
                conf_raw = max(0.0, min(1.0, (cos + 1) / 2))
                box_key = f"{pb[0]}_{pb[1]}_{pb[2]}_{pb[3]}"
                if box_key in self._body_sim_ema:
                    conf = ema_alpha * self._body_sim_ema[box_key] + (1 - ema_alpha) * conf_raw
                else:
                    conf = conf_raw
                self._body_sim_ema[box_key] = conf
            else:
                conf = 0.0
            body_confs.append(conf)

        if len(self._body_sim_ema) > 50:
            self._body_sim_ema.clear()

        if pboxes:
            j_star = int(np.argmax(body_confs))
            pb_star = pboxes[j_star]
            bconf_star = body_confs[j_star]
            bfeat_star = body_feats[j_star]
        else:
            pb_star = None
            bconf_star = 0.0
            bfeat_star = None

        # DEBUG: thỉnh thoảng log tình trạng ReID / admin chọn được
        if hasattr(self, "_frame_id") and (self._frame_id % 15 == 0):
            has_centroid = self.body_centroid is not None
            max_conf = float(bconf_star) if bconf_star is not None else 0.0
            self.get_logger().info(
                f"REID [ADMIN] frame={self._frame_id}, "
                f"num_boxes={len(pboxes)}, "
                f"has_centroid={has_centroid}, "
                f"best_body_conf={max_conf:.3f}, "
                f"admin_frames_seen={self.admin.get('frames_seen', 0)}, "
                f"admin_frames_missing={self.admin.get('frames_missing', 0)}"
            )

        fconf_star = 0.0
        fbox = None
        if self.face_db is not None and faces:
            templates, centroid = self.face_db
            templates_T = templates.T if templates.size else templates
            centroid = centroid.astype(np.float32)

            if pb_star is not None:
                pcx, pcy = center_of(pb_star)
                containing = []
                for i, f in enumerate(faces):
                    (x1, y1, x2, y2) = f["box"]
                    if x1 <= pcx <= x2 and y1 <= pcy <= y2:
                        containing.append(i)
                cand = containing if containing else list(range(len(faces)))
            else:
                cand = list(range(len(faces)))

            best_i = None
            best_s = -1.0
            for i in cand:
                e = faces[i]["feat"]
                cs = float(np.max(e @ templates_T)) if templates.size else 0.0
                cc = float(e @ centroid)
                s = 0.5 * cs + 0.5 * cc
                if s > best_s:
                    best_s = s
                    best_i = i
            if best_i is not None:
                fbox = faces[best_i]["box"]
                fconf_star = float(best_s)

        score_star = max(fconf_star, bconf_star)
        best = dict(face_box=fbox,
                    person_box=pb_star,
                    face_conf=fconf_star,
                    body_conf=bconf_star,
                    score=score_star)

        # lưu lại feat/emb mới nhất cho online_enroll
        try:
            self._last_body_feat_for_enroll = bfeat_star
        except Exception:
            self._last_body_feat_for_enroll = None
        self._last_face_emb_for_enroll = None
        if fbox is not None and faces:
            try:
                for f in faces:
                    if f.get('box') == fbox and f.get('feat') is not None:
                        self._last_face_emb_for_enroll = f.get('feat')
                        break
            except Exception:
                self._last_face_emb_for_enroll = None

        ACPT_FACE = float(self.get_parameter('accept_face').value)
        ACPT_BODY = float(self.get_parameter('accept_body').value)
        IOU_STK = float(self.get_parameter('iou_sticky').value)
        MARGIN = float(self.get_parameter('margin_delta').value)
        CONFIRM = int(self.get_parameter('confirm_frames').value)
        UNLOCK_MISS = int(self.get_parameter('unlock_on_miss_frames').value)

        if best["person_box"] is None:
            self.admin["frames_missing"] = self.admin.get("frames_missing", 0) + 1
            if self.admin["frames_missing"] >= max(1, UNLOCK_MISS):
                self.admin.update(face_box=None,
                                  person_box=None,
                                  score_ema=0.0,
                                  frames_seen=0,
                                  frames_missing=0)
                self.tracker = None
        else:
            still_vis = (self.admin["person_box"] is not None) and \
                        (iou(self.admin["person_box"], best["person_box"]) > IOU_STK)
            if self.admin["frames_seen"] == 0 or self.admin["person_box"] is None:
                self.admin.update(person_box=best["person_box"],
                                  face_box=best["face_box"],
                                  score_ema=best["score"],
                                  frames_seen=1,
                                  frames_missing=0)
                self.tracker = create_tracker()
                if self.tracker is not None:
                    x1, y1, x2, y2 = self.admin["person_box"]
                    self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            else:
                self.admin["score_ema"] = 0.7 * self.admin["score_ema"] + 0.3 * best["score"]
                challenger_better = (best["score"] > self.admin["score_ema"] + MARGIN)
                if (not still_vis) and challenger_better:
                    self.admin.update(person_box=best["person_box"],
                                      face_box=best["face_box"],
                                      score_ema=best["score"],
                                      frames_seen=1,
                                      frames_missing=0)
                    self.tracker = create_tracker()
                    if self.tracker is not None:
                        x1, y1, x2, y2 = self.admin["person_box"]
                        self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                else:
                    self.admin["frames_seen"] += 1
                    self.admin["frames_missing"] = 0
                    if iou(self.admin["person_box"], best["person_box"]) > IOU_STK:
                        self.admin["person_box"] = best["person_box"]
                        if best["face_box"] is not None:
                            self.admin["face_box"] = best["face_box"]

        has_face = (best["face_box"] is not None)
        has_person = (best["person_box"] is not None)
        has_centroid = (self.body_centroid is not None)

        # Nếu chưa có centroid body → KHÔNG cho pass_gate (bắt buộc phải có appearance)
        if not has_centroid:
            pass_gate = False
        else:
            if has_face and has_person:
                pass_gate = (best["face_conf"] >= ACPT_FACE) and (best["body_conf"] >= ACPT_BODY)
            else:
                pass_gate = has_person and (best["body_conf"] >= ACPT_BODY)

        is_admin_by_id = has_person and (self.admin["frames_seen"] >= CONFIRM) and pass_gate
        return is_admin_by_id, best



    # ---------- Control (CENTER-FIRST) ----------
    def compute_cmd(self, frame_w, frame_h, target_box_visible, depth_img):
        twist = Twist()
        detected = Bool()
        detected.data = (target_box_visible is not None)

        if target_box_visible is None:
            self._is_centered = False
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return twist, detected, None

        cx, _ = center_of(target_box_visible)
        err_px = (cx - frame_w * 0.5)
        dead = float(self.get_parameter('center_deadband_px').value)
        rel = float(self.get_parameter('center_release_px').value)
        center_first = bool(self.get_parameter('center_first').value)

        if not self._is_centered:
            if abs(err_px) <= dead:
                self._is_centered = True
        else:
            if abs(err_px) > max(rel, dead):
                self._is_centered = False

        err_eff = 0.0 if abs(err_px) <= dead else (np.sign(err_px) * (abs(err_px) - dead))
        kx = float(self.get_parameter('kx_center').value)
        wz = clamp(-kx * err_eff,
                   -float(self.get_parameter('wz_max').value),
                   +float(self.get_parameter('wz_max').value))

        depth_m = self.depth_at_box(target_box_visible, (frame_w, frame_h), depth_img)
        if depth_m is not None:
            if self.depth_display is None:
                self.depth_display = depth_m
            else:
                a = float(self.depth_ema_alpha)
                self.depth_display = a * depth_m + (1.0 - a) * self.depth_display

        vx = 0.0
        if depth_m is not None:
            kd = float(self.get_parameter('kd_distance').value)
            d_des = float(self.get_parameter('target_distance_m').value)
            err_d = depth_m - d_des
            if (not center_first) or self._is_centered:
                if err_d > 0.0:
                    vx = clamp(kd * err_d, 0.0,
                               float(self.get_parameter('v_forward_max').value))
                else:
                    vx = 0.0
            else:
                vx = 0.0

        twist.linear.x = float(vx)
        twist.angular.z = float(wz)
        return twist, detected, depth_m

    # ---------- Detector wrap ----------
    def detect_persons(self, frame, conf_thresh: float):
        return _ssd_detect(self.ssd_net, frame, conf_thresh)

    # ---------- Image callback ----------
        # ---------- Image callback ----------
    def on_image(self, msg: Image):
        frame0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        W = int(self.get_parameter('image_width').value)
        H = int(self.get_parameter('image_height').value)
        frame = cv2.resize(frame0, (W, H), interpolation=cv2.INTER_LINEAR)

        # DEBUG: tăng frame counter
        if hasattr(self, "_frame_id"):
            self._frame_id += 1


        # Dynamic LBP weight theo ánh sáng
        vmean = mean_brightness_v(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        vchan = hsv[..., 2]
        sat_ratio = float((vchan > 240).sum()) / float(vchan.size + 1e-6)
        base_cw = float(self.get_parameter('body_color_weight').value)

        if vmean < 80:
            self._dynamic_color_weight = min(0.5, base_cw * 1.8)
            lighting_mode = "DARK"
        elif vmean > 200 or sat_ratio > 0.10:
            self._dynamic_color_weight = min(0.5, base_cw * 1.8)
            lighting_mode = "BRIGHT"
        elif 90 <= vmean <= 180 and sat_ratio < 0.05:
            self._dynamic_color_weight = max(0.05, base_cw * 0.5)
            lighting_mode = "NORMAL"
        else:
            self._dynamic_color_weight = base_cw
            lighting_mode = "MID"

        self._lighting_mode = lighting_mode

        # Detector dual (gốc + enhance)
        conf_thr = float(self.get_parameter('person_conf').value)
        p1, s1 = self.detect_persons(frame, conf_thr)

        frame_enh = enhance_backlit(frame)
        p2, s2 = self.detect_persons(frame_enh, max(0.30, conf_thr - 0.05))

        p_all = p1 + p2
        s_all = s1 + s2
        pboxes, pscores = nms_merge_xyxy(p_all, s_all, iou_thr=0.45, topk=100)

        # Face (nếu bật)
        faces = []
        if self.face_app is not None and self.admin["person_box"] is not None:
            rb = expand(self.admin["person_box"], frame.shape, m=0.30)
            rx1, ry1, rx2, ry2 = rb
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                faces = detect_faces_arcface(self.face_app, roi, conf_thresh=0.5,
                                             offset_xy=(rx1, ry1))
        if self.face_app is not None and not faces:
            faces = detect_faces_arcface(self.face_app, frame, conf_thresh=0.5)

        if not self.auto_done:
            self.state_txt = "AUTO-ENROLL"
            self.auto_enroll_step(frame, pboxes, faces)
            is_admin_by_id = False
            best_info = dict(face_box=None, person_box=None,
                             face_conf=0.0, body_conf=0.0, score=0.0)
        else:
            self.state_txt = "RUN"
            is_admin_by_id, best_info = self.select_admin(frame, pboxes, pscores, faces)

        # chọn target_box_visible: cần IOU với admin_box
        target_box_visible = None
        if is_admin_by_id and (self.admin["person_box"] is not None):
            iou_need = float(self.get_parameter('lock_requires_visible_iou').value)
            for pb in pboxes:
                if iou(pb, self.admin["person_box"]) >= iou_need:
                    target_box_visible = pb
                    break

        # tracker fallback
        if target_box_visible is None and self.tracker is not None:
            ok, trk = self.tracker.update(frame)
            if ok:
                x, y, w, h = map(int, trk)
                trk_box = (max(0, x), max(0, y),
                           min(W - 1, x + w), min(H - 1, y + h))
                target_box_visible = trk_box
                self._tracker_miss = 0
            else:
                self._tracker_miss = int(self.get_parameter('tracker_grace_frames').value) + 1

        if target_box_visible is None:
            self._tracker_miss += 1
        else:
            self._tracker_miss = 0

        if self._tracker_miss > int(self.get_parameter('tracker_grace_frames').value):
            target_box_visible = None

        is_locked = (target_box_visible is not None)

        twist, detected, depth_m = self.compute_cmd(W, H, target_box_visible, self.depth_img)
        self._hud_body_conf = float(best_info.get('body_conf', 0.0)) if is_locked else None

        # DEBUG: log LOCK state & body_conf mỗi 15 frame
        if hasattr(self, "_frame_id") and (self._frame_id % 15 == 0):
            admin_box = self.admin.get("person_box")
            admin_score = self.admin.get("score_ema", 0.0)
            self.get_logger().info(
                f"REID [LOCK] frame={self._frame_id}, "
                f"is_admin={is_admin_by_id}, "
                f"locked={is_locked}, "
                f"best_body_conf={float(best_info.get('body_conf', 0.0)):.3f}, "
                f"admin_score_ema={admin_score:.3f}, "
                f"admin_box={admin_box}, "
                f"target_box={target_box_visible}"
            )


        # Update trajectory cho spatio-temporal scoring
        if is_locked and (target_box_visible is not None) and (depth_m is not None):
            cx, cy = center_of(target_box_visible)
            self.traj_centers_2d.append((float(cx), float(cy)))
            self.traj_depths.append(float(depth_m))
            if len(self.traj_centers_2d) > self.traj_max_len:
                self.traj_centers_2d = self.traj_centers_2d[-self.traj_max_len:]
                self.traj_depths = self.traj_depths[-self.traj_max_len:]

        # ===== Online enroll (giữ nguyên logic cũ, dùng feat OSNet+LBP mới) =====
        if bool(self.get_parameter('online_enroll_enabled').value) and is_locked:
            ok_good = True

            iou_min = float(self.get_parameter('online_enroll_iou_stable').value)
            if self._last_person_box_for_enroll is not None and target_box_visible is not None:
                if iou(self._last_person_box_for_enroll, target_box_visible) < iou_min:
                    ok_good = False

            min_ratio = float(self.get_parameter('online_enroll_min_box_ratio').value)
            if target_box_visible is not None:
                area = max(1.0, float((target_box_visible[2] - target_box_visible[0]) *
                                      (target_box_visible[3] - target_box_visible[1])))
                if (area / float(W * H)) < min_ratio:
                    ok_good = False

            depth_tol = float(self.get_parameter('online_enroll_depth_tol_m').value)
            if depth_m is not None and self._last_depth_for_enroll is not None:
                if abs(float(depth_m) - float(self._last_depth_for_enroll)) > depth_tol:
                    ok_good = False

            acpt_body = float(self.get_parameter('accept_body').value)
            if self._last_body_feat_for_enroll is not None and self.body_centroid is not None:
                cs = float(np.dot(self._last_body_feat_for_enroll, self.body_centroid) /
                           (np.linalg.norm(self._last_body_feat_for_enroll) *
                            np.linalg.norm(self.body_centroid) + 1e-8))
                conf_body = max(0.0, min(1.0, (cs + 1.0) / 2.0))
                if conf_body < acpt_body:
                    ok_good = False

            if ok_good:
                self._online_good_frames += 1
            else:
                self._online_good_frames = 0

            now_ts = time.time()
            min_interval = 1.0 / max(1e-6, float(self.get_parameter('online_enroll_rate_hz').value))
            can_add = (now_ts - float(self._online_last_add_ts)) >= min_interval
            need_frames = int(self.get_parameter('online_enroll_min_frames').value)

            if (self._online_good_frames >= max(1, need_frames)) and can_add:
                # ----- update body centroid -----
                if self._last_body_feat_for_enroll is not None:
                    try:
                        self.body_samples.append(self._last_body_feat_for_enroll)
                        cap_b = int(self.get_parameter('online_enroll_body_max').value)
                        if len(self.body_samples) > max(1, cap_b):
                            self.body_samples = self.body_samples[-cap_b:]
                        B = np.array(self.body_samples, dtype=np.float32)
                        cent = B.mean(axis=0)
                        cent /= (np.linalg.norm(cent) + 1e-8)
                        self.body_centroid = cent
                    except Exception:
                        if self.body_centroid is None:
                            self.body_centroid = self._last_body_feat_for_enroll.copy()
                        else:
                            self.body_centroid = 0.9 * self.body_centroid + \
                                                 0.1 * self._last_body_feat_for_enroll
                            self.body_centroid /= (np.linalg.norm(self.body_centroid) + 1e-8)

                # ----- update face DB -----
                if self._last_face_emb_for_enroll is not None:
                    try:
                        if self.face_db is None:
                            E = np.array([self._last_face_emb_for_enroll], dtype=np.float32)
                        else:
                            E, cent = self.face_db
                            if E.size:
                                E = np.vstack([E, self._last_face_emb_for_enroll.astype(np.float32)])
                            else:
                                E = np.array([self._last_face_emb_for_enroll], dtype=np.float32)
                        cap_f = int(self.get_parameter('online_enroll_face_max').value)
                        if len(E) > max(1, cap_f):
                            E = E[-cap_f:]
                        cent = E.mean(axis=0)
                        cent /= (np.linalg.norm(cent) + 1e-8)
                        self.face_db = (E, cent)
                    except Exception:
                        pass

                self._online_last_add_ts = now_ts
                self._online_good_frames = 0

            self._last_person_box_for_enroll = target_box_visible
            if depth_m is not None:
                self._last_depth_for_enroll = float(depth_m)
        else:
            self._online_good_frames = 0

        # Publish
        self.cmd_pub.publish(twist)
        self.flag_pub.publish(Bool(data=is_locked))
        if depth_m is not None:
            self.dist_depth_pub.publish(Float32(data=float(depth_m)))
        self.state_pub.publish(String(data=("LOCKED" if is_locked else "SEARCHING")))

        # Debug overlay
        if bool(self.get_parameter('publish_debug_image').value):
            dbg = frame.copy()
            for pb in pboxes:
                if target_box_visible is not None and iou(pb, target_box_visible) >= 0.99:
                    label = "TARGET" if self._is_centered else "CENTERING"
                    draw_labeled_box(dbg, pb, color=(0, 0, 255), label=label)
                else:
                    cv2.rectangle(dbg, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 0), 2)

            status = "LOCKED" if is_locked else "SEARCHING"
            cv2.putText(dbg, f"{self.state_txt} | {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (50, 220, 50) if is_locked else (0, 165, 255), 2)

            depth_show = self.depth_display if self.depth_display is not None else depth_m
            depth_txt = "--" if depth_show is None else f"{float(depth_show):.2f} m"
            mode_txt = "Centered" if self._is_centered else "Centering"
            body_txt = "--" if (self._hud_body_conf is None) else f"{self._hud_body_conf:.2f}"
            hud_right = f"Depth: {depth_txt}   Mode: {mode_txt}   BodyConf: {body_txt}"
            draw_label_top_right(dbg, hud_right, margin=10)

            if self._lighting_mode in ("DARK", "BRIGHT"):
                cv2.putText(dbg, "LOW-LIGHT / BACKLIT MODE", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            try:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
            except Exception:
                pass



def main():
    rclpy.init()
    node = PersonDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
