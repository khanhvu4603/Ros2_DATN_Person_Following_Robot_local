import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# ========== Helpers ==========
def clamp(x, a, b): return a if x < a else b if x > b else x

def iou(a, b):
    if a is None or b is None: return 0.0
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih; ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
    return inter / ua if ua > 0 else 0.0

def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def expand(box, shape, m=0.20):
    x1, y1, x2, y2 = box
    H, W = shape[:2]; w = x2 - x1; h = y2 - y1
    x1 = max(0, int(x1 - m * w)); y1 = max(0, int(y1 - m * h))
    x2 = min(W - 1, int(x2 + m * w)); y2 = min(H - 1, int(y2 + m * h))
    return (x1, y1, x2, y2)

def clip_box(box, shape):
    if box is None:
        return None
    H, W = shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # clamp
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

# =================== ENHANCED BODY FEATURES ===================
def mb2_preprocess_keras_style(x_uint8):
    x = x_uint8.astype(np.float32)
    x = x / 127.5 - 1.0
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
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_roi
    
    return padded, scale

def hsv_histogram(roi_bgr, bins=16, v_weight=0.5, normalize_brightness=True):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    if normalize_brightness:
        v_channel = hsv[:, :, 2].astype(np.float32)
        v_mean = v_channel.mean()
        if v_mean > 10:
            v_channel = np.clip(v_channel * (128.0 / v_mean), 0, 255)
            hsv[:, :, 2] = v_channel.astype(np.uint8)

    histH = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    histS = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    histV = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    histV *= v_weight

    h = np.concatenate([histH, histS, histV]).astype(np.float32)
    h /= (np.linalg.norm(h) + 1e-8)
    return h

def extract_depth_feature(box, depth_img, target_size=(16, 16)):
    """Trích xuất một vector đặc trưng đơn giản từ depth."""
    if depth_img is None or box is None:
        return np.zeros(target_size[0] * target_size[1])
    
    # (FIX L) Clamp box trước khi slice - giống y ảnh
    box = clip_box(box, depth_img.shape)
    if box is None:
        return np.zeros(target_size[0] * target_size[1])
        
    x1, y1, x2, y2 = box
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
    arr = mb2_preprocess_keras_style(roi_rgb)[None, ...]
    
    inp_name = ort_sess.get_inputs()[0].name
    emb = ort_sess.run(None, {inp_name: arr.astype(np.float32)})[0].reshape(-1).astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-8)

    col = hsv_histogram(roi_padded, bins=16, v_weight=0.6, normalize_brightness=normalize_brightness)

    depth_feat = extract_depth_feature(box, depth_img)
    depth_feat /= (np.linalg.norm(depth_feat) + 1e-8)

    emb_weighted = emb * (1.0 - color_weight)
    col_weighted = col * color_weight
    depth_weighted = depth_feat * 0.1

    feat = np.concatenate([emb_weighted, col_weighted, depth_weighted], axis=0).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-8)
    return feat

# =================== Detector (MobileNet-SSD) ===================
def load_ssd(prototxt_path, weights_path):
    if Path(prototxt_path).exists() and Path(weights_path).exists():
        return cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
    return None

def ssd_detect(net, frame, conf_thresh=0.4):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    det = net.forward()
    boxes, scores = [], []
    for i in range(det.shape[2]):
        conf = det[0, 0, i, 2]; cls = int(det[0, 0, i, 1])
        if cls == 15 and conf > conf_thresh:
            box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2)); scores.append(float(conf))
    return boxes, scores
