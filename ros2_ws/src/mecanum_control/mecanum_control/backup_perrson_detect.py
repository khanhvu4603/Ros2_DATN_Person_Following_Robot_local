#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ffmpeg -video_size 1920x1080 -framerate 25 -f x11grab -i :0.0 -codec:v libx264 -preset ultrafast ~/screen_record.mp4

PersonDetector (ROS2) — BODY-first ReID + Face assist + Temporal sticky + BodyConf HUD
+ CHỐNG CHÓI SÁNG: Giảm trọng số color, chuẩn hóa brightness, EMA smoothing

CENTER-FIRST + HYSTERESIS + HUD BodyConf
+ BACKLIGHT HANDLING (enhancement + dual-detect + NMS)
+ DYNAMIC color_weight (tối/quá sáng → giảm ảnh hưởng màu)
+ TRACKER-FALLBACK (grace frames) khi detector rớt ngắn hạn
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


# =================== BODY features (MB2 + HSV) ===================
def mb2_preprocess_keras_style(x_uint8):
    x = x_uint8.astype(np.float32)
    x = x/127.5 - 1.0
    return x

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

def body_arr(frame, box):
    x1,y1,x2,y2 = expand(box, frame.shape, 0.20)
    roi = frame[y1:y2, x1:x2]
    if roi.size==0: return None
    roi224 = cv2.resize(roi, (224,224))
    roi_rgb = cv2.cvtColor(roi224, cv2.COLOR_BGR2RGB)
    arr = mb2_preprocess_keras_style(roi_rgb)[None,...]
    return roi224, arr

def body_feature_onnx(frame, box, ort_sess, color_weight=0.3, normalize_brightness=True):
    out = body_arr(frame, box)
    if out is None: return None
    roi224, arr = out
    inp_name = ort_sess.get_inputs()[0].name
    emb = ort_sess.run(None, {inp_name: arr.astype(np.float32)})[0].reshape(-1).astype(np.float32)

    emb /= (np.linalg.norm(emb)+1e-8)
    col = hsv_histogram(roi224, bins=16, v_weight=0.6, normalize_brightness=normalize_brightness)

    emb_weighted = emb * (1.0 - color_weight)
    col_weighted = col * color_weight

    feat = np.concatenate([emb_weighted, col_weighted], axis=0).astype(np.float32)
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

# =================== Face (optional) ===================
def build_face_app(det_size=(384,384), providers=None):
    if not _FACE_OK: return None
    app = FaceAnalysis(name='buffalo_sc', providers=providers or ['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=det_size)
    return app

def detect_faces_arcface(app, frame_bgr, conf_thresh=0.5, offset_xy=(0,0)):
    if app is None: return []
    ox, oy = offset_xy
    faces = app.get(frame_bgr)
    out=[]
    for f in faces:
        x1,y1,x2,y2 = map(int, f.bbox[:4])
        if getattr(f, 'det_score', 1.0) < conf_thresh: continue
        feat = f.normed_embedding
        if feat is None or feat.size==0: continue
        out.append(dict(box=(x1+ox,y1+oy,x2+ox,y2+oy), kps=f.kps, feat=feat.astype(np.float32)))
    return out

# =================== [NEW] Backlight enhancement utils ===================
def mean_brightness_v(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[...,2].mean())

def apply_grayworld(img):
    img = img.astype(np.float32)
    m = img.reshape(-1,3).mean(axis=0) + 1e-6
    scale = 128.0 / m
    out = np.clip(img * scale, 0, 255).astype(np.uint8)
    return out

def apply_clahe_v(img_bgr, clip=2.0, tile=(8,8)):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[...,2]
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    hsv[...,2] = clahe.apply(v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def auto_gamma(img_bgr):
    v = mean_brightness_v(img_bgr) / 255.0
    v = max(1e-3, min(0.99, v))
    gamma = np.log(0.5) / np.log(v)            # muốn V_mean ~ 0.5
    inv = 1.0 / max(0.1, min(5.0, gamma))      # chặn miền an toàn
    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((lut ** inv) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)

def enhance_backlit(img_bgr):
    x = apply_grayworld(img_bgr)
    x = apply_clahe_v(x, clip=2.0, tile=(8,8))
    x = auto_gamma(x)
    return x

def nms_merge_xyxy(boxes, scores, iou_thr=0.5, topk=100):
    """NMS cho format (x1,y1,x2,y2)."""
    if not boxes:
        return [], []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) >= topk: break
        xx1 = np.maximum(b[i,0], b[order[1:],0])
        yy1 = np.maximum(b[i,1], b[order[1:],1])
        xx2 = np.minimum(b[i,2], b[order[1:],2])
        yy2 = np.minimum(b[i,3], b[order[1:],3])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        area_i = (b[i,2]-b[i,0])*(b[i,3]-b[i,1])
        area_o = (b[order[1:],2]-b[order[1:],0])*(b[order[1:],3]-b[order[1:],1])
        ovr = inter / (area_i + area_o - inter + 1e-6)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds+1]
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
            ('kx_center', 0.00025),
            ('wz_max', 0.25),
            ('center_deadband_px', 40),
            ('center_first', True),
            ('center_release_px', 60),

            # Detector/ReID thresholds
            ('person_conf', 0.35),
            ('accept_face', 0.2),
            ('accept_body', 0.78),
            ('iou_sticky', 0.35),
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
            ('face_enable', False),

            # Online enroll
            ('online_enroll_enabled', True),
            ('online_enroll_min_frames', 10),
            ('online_enroll_rate_hz', 1),
            ('online_enroll_body_max', 10000),
            ('online_enroll_face_max', 100),
            ('online_enroll_iou_stable', 0.50),
            ('online_enroll_min_box_ratio', 0.03),
            ('online_enroll_depth_tol_m', 0.20),
            ('online_enroll_diversity_cos', 0.99),

            # MẤT là UNLOCK ngay
            ('unlock_on_miss_frames', 3),

            # Phải thấy lại IOU với bbox hiện tại để coi là LOCKED
            ('lock_requires_visible_iou', 0.35),

            # [NEW] tracker fallback grace frames
            ('tracker_grace_frames', 2),
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
        # NEW: publish trạng thái đã căn giữa người
        self.centered_pub  = self.create_publisher(Bool,   '/person_centered', 10)

        # Models/ReID state
        self.mb2_sess   = ort.InferenceSession(self.get_parameter('mb2_onnx_path').value,
                                               providers=["CPUExecutionProvider"])
        self.face_app   = build_face_app() if (_FACE_OK and bool(self.get_parameter('face_enable').value)) else None

        # Detector
        self.ssd_net = _load_ssd()
        if self.ssd_net is None:
            raise FileNotFoundError(
                "Không tìm thấy MobileNet-SSD Caffe. Hãy đặt file vào:\n"
                f"  - {MOBILENET_PROTOTXT}\n"
                f"  - {MOBILENET_WEIGHTS}"
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

        self.admin = dict(face_box=None, person_box=None, score_ema=0.0,
                          frames_seen=0, frames_missing=0)
        self.tracker = None

        # tracker fallback
        self._tracker_miss = 0  # [NEW]

        # depth overlay
        self.depth_display = None
        self.depth_ema_alpha = 0.6
        self.state_txt = "INIT"

        # online enroll state
        self._online_good_frames = 0
        self._online_last_add_ts = 0.0
        self._last_depth_for_enroll = None
        self._last_person_box_for_enroll = None
        self._last_body_feat_for_enroll = None
        self._last_face_emb_for_enroll = None

        # Center-first hysteresis state
        self._is_centered = False

        # HUD body conf
        self._hud_body_conf: Optional[float] = None

        # [NEW] dynamic color weight (init = param)
        self._dynamic_color_weight = float(self.get_parameter('body_color_weight').value)

    # ---------- Depth ----------
    def on_depth(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.get_parameter('depth_encoding').value)
        self.depth_enc = self.get_parameter('depth_encoding').value

    def depth_at_box(self, box_resized, resized_size, depth_img) -> Optional[float]:
        if depth_img is None or box_resized is None: return None
        W_r, H_r = resized_size
        depth_h, depth_w = depth_img.shape[:2]
        cx_r, cy_r = center_of(box_resized)
        cx = int(clamp(cx_r * (depth_w / float(W_r)), 0, depth_w - 1))
        cy = int(clamp(cy_r * (depth_h / float(H_r)), 0, depth_h - 1))
        k = 3
        x1 = max(0, cx - k); x2 = min(depth_w, cx + k + 1)
        y1 = max(0, cy - k); y2 = min(depth_h, cy + k + 1)
        roi = depth_img[y1:y2, x1:x2]
        if roi.size == 0: return None

        if self.depth_enc == '16UC1':
            roi_mm = roi.astype(np.float32)
            valid = roi_mm > 0
            if not np.any(valid): return None
            depth_m = float(np.median(roi_mm[valid])) / 1000.0
        else:  # 32FC1
            roi_m = roi.astype(np.float32)
            valid = np.isfinite(roi_m) & (roi_m > 0)
            if not np.any(valid): return None
            depth_m = float(np.median(roi_m[valid]))

        if np.isnan(depth_m) or depth_m < 0.05 or depth_m > 8.0:
            return None
        return depth_m

    # ---------- Auto-enroll ----------
    def auto_enroll_step(self, frame, pboxes, faces):
        now = time.time()
        if self.auto_start_ts is None:
            self.auto_start_ts = now

        # [MOD] dynamic color weight
        color_w = getattr(self, '_dynamic_color_weight', float(self.get_parameter('body_color_weight').value))
        norm_bright = bool(self.get_parameter('hsv_normalize_brightness').value)

        if pboxes:
            j = int(np.argmax([(pb[2]-pb[0])*(pb[3]-pb[1]) for pb in pboxes]))
            pb = pboxes[j]
            feat = body_feature_onnx(frame, pb, self.mb2_sess,
                                    color_weight=color_w,
                                    normalize_brightness=norm_bright)
            if feat is not None:
                self.body_samples.append(feat)
                if self.body_centroid is None:
                    self.body_centroid = feat.copy()
                else:
                    self.body_centroid = 0.9*self.body_centroid + 0.1*feat
                    self.body_centroid /= (np.linalg.norm(self.body_centroid)+1e-8)

        if faces:
            faces = sorted(faces, key=lambda f:(f["box"][2]-f["box"][0])*(f["box"][3]-f["box"][1]), reverse=True)
            self.face_samples.append(faces[0]["feat"])

        timeout = float(self.get_parameter('auto_timeout_sec').value)
        body_target = int(self.get_parameter('auto_body_target').value)
        if (now - self.auto_start_ts) >= timeout or len(self.body_samples) >= body_target:
            if len(self.face_samples) >= 10:
                E = np.array(self.face_samples, dtype=np.float32)
                cent = E.mean(axis=0); cent /= (np.linalg.norm(cent)+1e-8)
                self.face_db = (E, cent)
            if len(self.body_samples) >= 10 and self.body_centroid is None:
                B = np.array(self.body_samples, dtype=np.float32)
                cent = B.mean(axis=0); cent /= (np.linalg.norm(cent)+1e-8)
                self.body_centroid = cent
            self.auto_done = True
            self.state_txt = "RUN"

    # ---------- Select admin ----------
    def select_admin(self, frame, pboxes, faces):
        # [MOD] dynamic color weight
        color_w = getattr(self, '_dynamic_color_weight', float(self.get_parameter('body_color_weight').value))
        norm_bright = bool(self.get_parameter('hsv_normalize_brightness').value)
        ema_alpha = float(self.get_parameter('similarity_ema_alpha').value)

        body_confs=[]
        body_feats=[]
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
                            (np.linalg.norm(feat)*np.linalg.norm(self.body_centroid)+1e-8))
                conf_raw = max(0.0, min(1.0, (cos+1)/2))
                box_key = f"{pb[0]}_{pb[1]}_{pb[2]}_{pb[3]}"
                if box_key in self._body_sim_ema:
                    conf = ema_alpha * self._body_sim_ema[box_key] + (1-ema_alpha) * conf_raw
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
            pb_star=None; bconf_star=0.0; bfeat_star=None

        fconf_star=0.0; fbox=None
        if self.face_db is not None and faces:
            templates, centroid = self.face_db
            templates_T = templates.T if templates.size else templates
            centroid = centroid.astype(np.float32)

            if pb_star is not None:
                pcx,pcy = center_of(pb_star)
                containing = []
                for i,f in enumerate(faces):
                    (x1,y1,x2,y2)=f["box"]
                    if x1<=pcx<=x2 and y1<=pcy<=y2:
                        containing.append(i)
                cand = containing if containing else list(range(len(faces)))
            else:
                cand = list(range(len(faces)))

            best_i = None; best_s = -1.0
            for i in cand:
                e = faces[i]["feat"]
                cs = float(np.max(e @ templates_T)) if templates.size else 0.0
                cc = float(e @ centroid)
                s = 0.5*cs + 0.5*cc
                if s > best_s:
                    best_s = s; best_i = i
            if best_i is not None:
                fbox = faces[best_i]["box"]; fconf_star=float(best_s)

        score_star = max(fconf_star, bconf_star)
        best = dict(face_box=fbox, person_box=pb_star,
                    face_conf=fconf_star, body_conf=bconf_star, score=score_star)

        try:
            self._last_body_feat_for_enroll = bfeat_star
        except Exception:
            self._last_body_feat_for_enroll = None
        self._last_face_emb_for_enroll = None
        if fbox is not None and faces:
            try:
                for f in faces:
                    if f.get('box') == fbox and f.get('feat') is not None:
                        self._last_face_emb_for_enroll = f.get('feat'); break
            except Exception:
                self._last_face_emb_for_enroll = None

        ACPT_FACE = float(self.get_parameter('accept_face').value)
        ACPT_BODY = float(self.get_parameter('accept_body').value)
        IOU_STK   = float(self.get_parameter('iou_sticky').value)
        MARGIN    = float(self.get_parameter('margin_delta').value)
        CONFIRM   = int(self.get_parameter('confirm_frames').value)
        UNLOCK_MISS = int(self.get_parameter('unlock_on_miss_frames').value)

        if best["person_box"] is None:
            self.admin["frames_missing"] = self.admin.get("frames_missing", 0) + 1
            if self.admin["frames_missing"] >= max(1, UNLOCK_MISS):
                self.admin.update(face_box=None, person_box=None, score_ema=0.0, frames_seen=0, frames_missing=0)
                self.tracker = None
        else:
            still_vis = (self.admin["person_box"] is not None) and (iou(self.admin["person_box"], best["person_box"]) > IOU_STK)
            if self.admin["frames_seen"] == 0 or self.admin["person_box"] is None:
                self.admin.update(person_box=best["person_box"], face_box=best["face_box"],
                                  score_ema=best["score"], frames_seen=1, frames_missing=0)
                self.tracker = create_tracker()
                if self.tracker is not None:
                    x1,y1,x2,y2 = self.admin["person_box"]
                    self.tracker.init(frame, (x1,y1,x2-x1,y2-y1))
            else:
                self.admin["score_ema"] = 0.7*self.admin["score_ema"] + 0.3*best["score"]
                challenger_better = (best["score"] > self.admin["score_ema"] + MARGIN)
                if (not still_vis) and challenger_better:
                    self.admin.update(person_box=best["person_box"], face_box=best["face_box"],
                                      score_ema=best["score"], frames_seen=1, frames_missing=0)
                    self.tracker = create_tracker()
                    if self.tracker is not None:
                        x1,y1,x2,y2 = self.admin["person_box"]
                        self.tracker.init(frame, (x1,y1,x2-x1,y2-y1))
                else:
                    self.admin["frames_seen"] += 1
                    self.admin["frames_missing"] = 0
                    if iou(self.admin["person_box"], best["person_box"]) > IOU_STK:
                        self.admin["person_box"] = best["person_box"]
                        if best["face_box"] is not None:
                            self.admin["face_box"] = best["face_box"]

        has_face   = (best["face_box"] is not None)
        has_person = (best["person_box"] is not None)
        if has_face and has_person:
            pass_gate = (best["face_conf"] >= ACPT_FACE) and (best["body_conf"] >= ACPT_BODY)
        else:
            pass_gate = has_person and (best["body_conf"] >= ACPT_BODY)

        is_admin_by_id = has_person and (self.admin["frames_seen"] >= CONFIRM) and pass_gate
        return is_admin_by_id, best

    # ---------- Control (CENTER-FIRST) ----------
    def compute_cmd(self, frame_w, frame_h, target_box_visible, depth_img):
        twist = Twist()
        detected = Bool(); detected.data = (target_box_visible is not None)

        if target_box_visible is None:
            self._is_centered = False
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return twist, detected, None

        cx, _ = center_of(target_box_visible)
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

        depth_m = self.depth_at_box(target_box_visible, (frame_w, frame_h), depth_img)
        if depth_m is not None:
            if self.depth_display is None:
                self.depth_display = depth_m
            else:
                a = float(self.depth_ema_alpha)
                self.depth_display = a*depth_m + (1.0-a)*self.depth_display

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
            else:
                vx = 0.0

        twist.linear.x = float(vx)
        twist.angular.z = float(wz)
        return twist, detected, depth_m

    # ---------- Detector wrap ----------
    def detect_persons(self, frame, conf_thresh: float):
        return _ssd_detect(self.ssd_net, frame, conf_thresh)

    # ---------- Image callback ----------
    def on_image(self, msg: Image):
        frame0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        W = int(self.get_parameter('image_width').value)
        H = int(self.get_parameter('image_height').value)
        frame = cv2.resize(frame0, (W, H), interpolation=cv2.INTER_LINEAR)

        # [NEW] Dynamic color weight theo độ sáng toàn cảnh
        vmean = mean_brightness_v(frame)  # 0..255
        base_cw = float(self.get_parameter('body_color_weight').value)
        if vmean < 90 or vmean > 200:
            self._dynamic_color_weight = min(0.10, base_cw * 0.6)  # giảm mạnh màu khi tối/quá sáng
        else:
            self._dynamic_color_weight = base_cw

        # [NEW] Dual-detect: gốc + tăng sáng + NMS gộp
        conf_thr = float(self.get_parameter('person_conf').value)

        p1, s1 = self.detect_persons(frame, conf_thr)

        frame_enh = enhance_backlit(frame)
        p2, s2 = self.detect_persons(frame_enh, max(0.30, conf_thr - 0.05))

        p_all = p1 + p2
        s_all = s1 + s2
        pboxes, _ = nms_merge_xyxy(p_all, s_all, iou_thr=0.45, topk=100)

        # Face (nếu bật)
        faces = []
        if self.face_app is not None and self.admin["person_box"] is not None:
            rb = expand(self.admin["person_box"], frame.shape, m=0.30)
            rx1,ry1,rx2,ry2 = rb
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size>0:
                faces = detect_faces_arcface(self.face_app, roi, conf_thresh=0.5, offset_xy=(rx1,ry1))
        if self.face_app is not None and not faces:
            faces = detect_faces_arcface(self.face_app, frame, conf_thresh=0.5)

        if not self.auto_done:
            self.state_txt = "AUTO-ENROLL"
            self.auto_enroll_step(frame, pboxes, faces)
            is_admin_by_id = False
            best_info = dict(face_box=None, person_box=None, face_conf=0.0, body_conf=0.0, score=0.0)
        else:
            self.state_txt = "RUN"
            is_admin_by_id, best_info = self.select_admin(frame, pboxes, faces)

        # ====== CHỌN target_box_visible ======
        target_box_visible = None
        if is_admin_by_id and (self.admin["person_box"] is not None):
            iou_need = float(self.get_parameter('lock_requires_visible_iou').value)
            for pb in pboxes:
                if iou(pb, self.admin["person_box"]) >= iou_need:
                    target_box_visible = pb
                    break

        # [NEW] TRACKER FALLBACK khi detector không ra box hợp lệ
        if target_box_visible is None and self.tracker is not None:
            ok, trk = self.tracker.update(frame)
            if ok:
                x,y,w,h = map(int, trk)
                trk_box = (max(0,x),max(0,y),min(W-1,x+w),min(H-1,y+h))
                target_box_visible = trk_box
                # không tăng miss khi có tracker box
                self._tracker_miss = 0
            else:
                self._tracker_miss = int(self.get_parameter('tracker_grace_frames').value) + 1

        # Nếu vẫn None (không có tracker/không ok) → tăng miss
        if target_box_visible is None:
            self._tracker_miss += 1
        else:
            self._tracker_miss = 0

        # Vượt quá grace frames → coi như mất thật
        if self._tracker_miss > int(self.get_parameter('tracker_grace_frames').value):
            target_box_visible = None

        is_locked = (target_box_visible is not None)

        twist, detected, depth_m = self.compute_cmd(W, H, target_box_visible, self.depth_img)

        self._hud_body_conf = float(best_info.get('body_conf', 0.0)) if is_locked else None

        # Online enroll (giữ nguyên logic cũ)
        if bool(self.get_parameter('online_enroll_enabled').value) and is_locked:
            ok_good = True
            iou_min = float(self.get_parameter('online_enroll_iou_stable').value)
            if self._last_person_box_for_enroll is not None:
                if iou(self._last_person_box_for_enroll, target_box_visible) < iou_min:
                    ok_good = False
            min_ratio = float(self.get_parameter('online_enroll_min_box_ratio').value)
            area = max(1.0, float((target_box_visible[2]-target_box_visible[0]) * (target_box_visible[3]-target_box_visible[1])))
            if (area / float(W * H)) < min_ratio:
                ok_good = False
            depth_tol = float(self.get_parameter('online_enroll_depth_tol_m').value)
            if depth_m is not None and self._last_depth_for_enroll is not None:
                if abs(float(depth_m) - float(self._last_depth_for_enroll)) > depth_tol:
                    ok_good = False

            acpt_body = float(self.get_parameter('accept_body').value)
            if self._last_body_feat_for_enroll is not None and self.body_centroid is not None:
                cs = float(np.dot(self._last_body_feat_for_enroll, self.body_centroid) /
                           (np.linalg.norm(self._last_body_feat_for_enroll)*np.linalg.norm(self.body_centroid)+1e-8))
                conf_body = max(0.0, min(1.0, (cs+1.0)/2.0))
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
                if self._last_body_feat_for_enroll is not None:
                    try:
                        self.body_samples.append(self._last_body_feat_for_enroll)
                        cap_b = int(self.get_parameter('online_enroll_body_max').value)
                        if len(self.body_samples) > max(1, cap_b):
                            self.body_samples = self.body_samples[-cap_b:]
                        B = np.array(self.body_samples, dtype=np.float32)
                        cent = B.mean(axis=0); cent /= (np.linalg.norm(cent)+1e-8)
                        self.body_centroid = cent
                    except Exception:
                        if self.body_centroid is None:
                            self.body_centroid = self._last_body_feat_for_enroll.copy()
                        else:
                            self.body_centroid = 0.9*self.body_centroid + 0.1*self._last_body_feat_for_enroll
                            self.body_centroid /= (np.linalg.norm(self.body_centroid)+1e-8)

                if self._last_face_emb_for_enroll is not None:
                    try:
                        if self.face_db is None:
                            E = np.array([self._last_face_emb_for_enroll], dtype=np.float32)
                        else:
                            E, cent = self.face_db
                            E = np.vstack([E, self._last_face_emb_for_enroll.astype(np.float32)]) if E.size else np.array([self._last_face_emb_for_enroll], dtype=np.float32)
                        cap_f = int(self.get_parameter('online_enroll_face_max').value)
                        if len(E) > max(1, cap_f):
                            E = E[-cap_f:]
                        cent = E.mean(axis=0); cent /= (np.linalg.norm(cent)+1e-8)
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

        # NEW: publish trạng thái căn giữa cho LiDAR / planner
        centered_msg = Bool()
        centered_msg.data = bool(is_locked and self._is_centered)
        self.centered_pub.publish(centered_msg)

        # Debug overlay
        if bool(self.get_parameter('publish_debug_image').value):
            dbg = frame.copy()
            for pb in pboxes:
                if target_box_visible is not None and iou(pb, target_box_visible) >= 0.99:
                    label = "TARGET" if self._is_centered else "CENTERING"
                    draw_labeled_box(dbg, pb, color=(0,0,255), label=label)
                else:
                    cv2.rectangle(dbg, (pb[0], pb[1]), (pb[2], pb[3]), (0,255,0), 2)

            status = "LOCKED" if is_locked else "SEARCHING"
            cv2.putText(dbg, f"{self.state_txt} | {status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (50,220,50) if is_locked else (0,165,255), 2)

            depth_show = self.depth_display if self.depth_display is not None else depth_m
            depth_txt = "--" if depth_show is None else f"{float(depth_show):.2f} m"
            mode_txt  = "Centered" if self._is_centered else "Centering"
            body_txt  = "--" if (self._hud_body_conf is None) else f"{self._hud_body_conf:.2f}"
            hud_right = f"Depth: {depth_txt}   Mode: {mode_txt}   BodyConf: {body_txt}"
            draw_label_top_right(dbg, hud_right, margin=10)

            # [NEW] Hiển thị cảnh báo ánh sáng để debug nhanh
            if vmean < 90 or vmean > 200:
                cv2.putText(dbg, "LOW-LIGHT / BACKLIT MODE", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,140,255), 2)

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
