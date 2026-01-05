#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Tracker Test - V2: Shape + Depth (1536-dim)
Test tracking trên video với feature từ MobileNetV2 + Depth map.

Lưu ý: Video RGB không có depth, nên sẽ dùng dummy depth (all zeros).
Để test thực tế cần video RGB-D hoặc bỏ qua depth.

Usage:
    python3 test_v2_shape_depth.py --video /path/to/video.mp4
"""

import argparse
import time
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ========== Paths ==========
HERE = Path(__file__).resolve().parent.parent
MODELS = HERE / "models"
MB2_ONNX_PATH = str(MODELS / "mb2_gap.onnx")
MOBILENET_PROTOTXT = str(MODELS / "MobileNetSSD_deploy.prototxt")
MOBILENET_WEIGHTS = str(MODELS / "MobileNetSSD_deploy.caffemodel")

# ========== Helpers ==========
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

# ========== Feature Extraction ==========
def mb2_preprocess(x_uint8):
    x = x_uint8.astype(np.float32)
    x = x / 127.5 - 1.0
    return x

def body_arr_preserve_aspect_ratio(frame, box, target_size=(224, 224)):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    h, w = roi.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_roi

    return padded

def extract_depth_feature(box, depth_img, target_size=(16, 16)):
    """Trích xuất depth feature 256-dim từ depth image."""
    if depth_img is None or box is None:
        return np.zeros(target_size[0] * target_size[1], dtype=np.float32)
    
    x1, y1, x2, y2 = map(int, box)
    h, w = depth_img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi = depth_img[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros(target_size[0] * target_size[1], dtype=np.float32)
    
    roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    roi_normalized = np.clip((5000 - roi_resized) / 4500.0, 0.0, 1.0)
    depth_feat = roi_normalized.flatten().astype(np.float32)
    depth_feat /= (np.linalg.norm(depth_feat) + 1e-8)
    
    return depth_feat

def extract_shape_depth_feature(frame, box, depth_img, ort_sess, depth_weight=0.1):
    """V2: MobileNetV2 + Depth - 1536-dim"""
    roi_padded = body_arr_preserve_aspect_ratio(frame, box)
    if roi_padded is None:
        return None

    roi_rgb = cv2.cvtColor(roi_padded, cv2.COLOR_BGR2RGB)
    arr = mb2_preprocess(roi_rgb)[None, ...]

    inp_name = ort_sess.get_inputs()[0].name
    emb = ort_sess.run(None, {inp_name: arr.astype(np.float32)})[0].reshape(-1).astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-8)

    # Depth feature
    depth_feat = extract_depth_feature(box, depth_img)
    
    # Combine: 1280 + 256 = 1536
    shape_weighted = emb * (1.0 - depth_weight)
    depth_weighted = depth_feat * depth_weight
    
    feat = np.concatenate([shape_weighted, depth_weighted])
    feat /= (np.linalg.norm(feat) + 1e-8)
    
    return feat  # 1536-dim

# ========== Detector ==========
def load_ssd():
    if Path(MOBILENET_PROTOTXT).exists() and Path(MOBILENET_WEIGHTS).exists():
        return cv2.dnn.readNetFromCaffe(MOBILENET_PROTOTXT, MOBILENET_WEIGHTS)
    return None

def ssd_detect(net, frame, conf_thresh=0.4):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
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

# ========== Tracker Class ==========
class VideoTracker:
    def __init__(self, mb2_sess, accept_thr=0.75, reject_thr=0.60, iou_thr=0.4):
        self.mb2_sess = mb2_sess
        self.accept_thr = accept_thr
        self.reject_thr = reject_thr
        self.iou_thr = iou_thr
        
        self.state = 'ENROLLING'
        self.target_feature = None
        self.target_box = None
        self.body_samples = []
        self.body_centroid = None
        self.depth_img = None
        
        # Metrics
        self.frame_logs = []
        self.state_changes = []
        self.similarity_history = []
        self.track_durations = []
        self.current_track_start = None
        
    def set_depth(self, depth_img):
        self.depth_img = depth_img
        
    def enroll(self, frame, boxes):
        if not boxes:
            return
            
        j = int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes]))
        box = boxes[j]
        
        feat = extract_shape_depth_feature(frame, box, self.depth_img, self.mb2_sess)
        if feat is not None:
            self.body_samples.append(feat)
            if self.body_centroid is None:
                self.body_centroid = feat.copy()
            else:
                self.body_centroid = 0.9 * self.body_centroid + 0.1 * feat
                self.body_centroid /= (np.linalg.norm(self.body_centroid) + 1e-8)
    
    def finish_enrollment(self):
        if self.body_centroid is not None:
            self.target_feature = self.body_centroid.copy()
            self.state = 'SEARCHING'
            return True
        return False
    
    def find_best_match(self, frame, boxes):
        best_box, best_score = None, -1.0
        for box in boxes:
            feat = extract_shape_depth_feature(frame, box, self.depth_img, self.mb2_sess)
            if feat is None:
                continue
            score = float(np.dot(feat, self.target_feature))
            if score > best_score:
                best_score = score
                best_box = box
        return best_box, best_score
    
    def find_match_by_iou(self, frame, boxes):
        if self.target_box is None:
            return None, -1.0
            
        best_box, best_score = None, -1.0
        for box in boxes:
            if iou(box, self.target_box) < self.iou_thr:
                continue
            feat = extract_shape_depth_feature(frame, box, self.depth_img, self.mb2_sess)
            if feat is None:
                continue
            score = float(np.dot(feat, self.target_feature))
            if score > best_score:
                best_score = score
                best_box = box
        return best_box, best_score
    
    def update(self, frame_id, timestamp, frame, boxes):
        prev_state = self.state
        similarity = 0.0
        
        if self.state == 'SEARCHING':
            best_box, best_score = self.find_best_match(frame, boxes)
            similarity = best_score
            if best_box and best_score > self.accept_thr:
                self.state = 'LOCKED'
                self.target_box = best_box
                self.current_track_start = timestamp
                
        elif self.state == 'LOCKED':
            best_box, best_score = self.find_match_by_iou(frame, boxes)
            similarity = best_score
            
            if best_box and best_score > self.reject_thr:
                self.target_box = best_box
            else:
                best_box, best_score = self.find_best_match(frame, boxes)
                similarity = best_score
                if best_box and best_score > self.accept_thr:
                    self.target_box = best_box
                else:
                    self.state = 'LOST'
                    if self.current_track_start:
                        self.track_durations.append(timestamp - self.current_track_start)
                        
        elif self.state == 'LOST':
            best_box, best_score = self.find_best_match(frame, boxes)
            similarity = best_score
            if best_box and best_score > self.accept_thr:
                self.state = 'LOCKED'
                self.target_box = best_box
                self.current_track_start = timestamp
        
        if self.state != prev_state:
            self.state_changes.append({
                'timestamp': timestamp,
                'from_state': prev_state,
                'to_state': self.state,
                'similarity': similarity
            })
        
        if self.state == 'LOCKED' and similarity > 0:
            self.similarity_history.append(similarity)
        
        bbox_str = f"{self.target_box[0]},{self.target_box[1]},{self.target_box[2]-self.target_box[0]},{self.target_box[3]-self.target_box[1]}" if self.target_box else ""
        self.frame_logs.append({
            'frame_id': frame_id,
            'timestamp': timestamp,
            'state': self.state,
            'similarity': similarity,
            'bbox': bbox_str
        })
        
        return self.state, similarity
    
    def get_summary(self, total_time):
        frames_locked = sum(1 for log in self.frame_logs if log['state'] == 'LOCKED')
        frames_lost = sum(1 for log in self.frame_logs if log['state'] == 'LOST')
        frames_searching = sum(1 for log in self.frame_logs if log['state'] == 'SEARCHING')
        total_frames = len(self.frame_logs)
        
        tf = sum(1 for i, sc in enumerate(self.state_changes) 
                 if sc['to_state'] == 'LOCKED' and i > 0 and self.state_changes[i-1]['from_state'] == 'LOST')
        
        lost_events = sum(1 for sc in self.state_changes if sc['to_state'] == 'LOST')
        reacquired = sum(1 for sc in self.state_changes if sc['to_state'] == 'LOCKED' and sc['from_state'] in ['LOST', 'SEARCHING'])
        
        sim_arr = np.array(self.similarity_history) if self.similarity_history else np.array([0])
        
        return {
            'version': 'V2 (Shape + Depth)',
            'total_frames': total_frames,
            'total_time': total_time,
            'frames_locked': frames_locked,
            'frames_lost': frames_lost,
            'frames_searching': frames_searching,
            'locked_rate': frames_locked / total_frames * 100 if total_frames > 0 else 0,
            'lost_rate': (frames_lost + frames_searching) / total_frames * 100 if total_frames > 0 else 0,
            'track_fragmentations': tf,
            'lost_events': lost_events,
            'reacquired': reacquired,
            'reid_success_rate': reacquired / lost_events * 100 if lost_events > 0 else 100,
            'sim_mean': float(sim_arr.mean()),
            'sim_std': float(sim_arr.std()),
            'sim_min': float(sim_arr.min()),
            'sim_max': float(sim_arr.max()),
            'longest_track': max(self.track_durations) if self.track_durations else total_time,
            'avg_track': np.mean(self.track_durations) if self.track_durations else total_time
        }

# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description='Test V2: Shape + Depth')
    parser.add_argument('--video', type=str, default='/home/khanhvq/backup_16_12_2025/ros2_ws/rgbV2.mp4')
    parser.add_argument('--depth_video', type=str, default=None, help='Optional depth video')
    parser.add_argument('--enroll_frames', type=int, default=100)
    parser.add_argument('--output', type=str, default='results_v2_shape_depth.csv')
    parser.add_argument('--show', action='store_true', help='Show video')
    args = parser.parse_args()
    
    print(f"=== V2: Shape + Depth (1536-dim) ===")
    print(f"Video: {args.video}")
    if args.depth_video:
        print(f"Depth Video: {args.depth_video}")
    else:
        print("Note: No depth video provided, using dummy depth (zeros)")
    
    # Load models
    print("Loading models...")
    ssd_net = load_ssd()
    if ssd_net is None:
        print("ERROR: Cannot load MobileNet-SSD!")
        return
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    mb2_sess = ort.InferenceSession(MB2_ONNX_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"])
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {args.video}")
        return
    
    depth_cap = None
    if args.depth_video:
        depth_cap = cv2.VideoCapture(args.depth_video)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames_video} frames @ {fps:.1f} FPS")
    
    # Tracker
    tracker = VideoTracker(mb2_sess)
    
    frame_id = 0
    start_time = time.time()
    proc_times = []
    
    print(f"\nPhase 1: Enrolling first {args.enroll_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Read depth if available
        depth_img = None
        if depth_cap:
            ret_d, depth_frame = depth_cap.read()
            if ret_d:
                depth_img = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        tracker.set_depth(depth_img)
        
        t0 = time.time()
        
        boxes, _ = ssd_detect(ssd_net, frame, conf_thresh=0.4)
        
        timestamp = frame_id / fps
        
        if frame_id < args.enroll_frames:
            tracker.enroll(frame, boxes)
            state = 'ENROLLING'
            similarity = 0.0
            
            if frame_id == args.enroll_frames - 1:
                if tracker.finish_enrollment():
                    print(f"Enrollment done! {len(tracker.body_samples)} samples collected")
                else:
                    print("Enrollment failed!")
                    break
        else:
            state, similarity = tracker.update(frame_id, timestamp, frame, boxes)
        
        proc_time = (time.time() - t0) * 1000
        proc_times.append(proc_time)
        
        if args.show:
            for box in boxes:
                color = (0, 0, 255) if tracker.target_box and iou(box, tracker.target_box) > 0.9 else (0, 255, 0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
            cv2.putText(frame, f"Sim: {similarity:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
            cv2.putText(frame, f"Frame: {frame_id}/{total_frames_video}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('V2: Shape + Depth', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_id += 1
        
        if frame_id % 100 == 0:
            print(f"  Frame {frame_id}/{total_frames_video} - State: {state} - Sim: {similarity:.3f}")
    
    cap.release()
    if depth_cap:
        depth_cap.release()
    if args.show:
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    
    summary = tracker.get_summary(total_time)
    summary['avg_proc_time'] = np.mean(proc_times)
    summary['fps'] = 1000 / summary['avg_proc_time']
    
    print(f"\n{'='*50}")
    print(f"=== RESULTS: V2 (Shape + Depth) ===")
    print(f"{'='*50}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Total time: {summary['total_time']:.1f}s")
    print(f"\n--- State Distribution ---")
    print(f"LOCKED: {summary['locked_rate']:.1f}% ({summary['frames_locked']} frames)")
    print(f"LOST/SEARCHING: {summary['lost_rate']:.1f}%")
    print(f"\n--- Tracking Stability ---")
    print(f"Track Fragmentations: {summary['track_fragmentations']}")
    print(f"Longest Track: {summary['longest_track']:.1f}s")
    print(f"\n--- Re-ID Performance ---")
    print(f"Lost Events: {summary['lost_events']}")
    print(f"Re-acquired: {summary['reacquired']}")
    print(f"Re-ID Success Rate: {summary['reid_success_rate']:.1f}%")
    print(f"\n--- Similarity Statistics ---")
    print(f"Mean: {summary['sim_mean']:.3f} ± {summary['sim_std']:.3f}")
    print(f"Min: {summary['sim_min']:.3f} | Max: {summary['sim_max']:.3f}")
    print(f"\n--- Performance ---")
    print(f"Avg Processing Time: {summary['avg_proc_time']:.1f} ms")
    print(f"FPS: {summary['fps']:.1f}")
    print(f"{'='*50}")
    
    output_path = Path(__file__).parent / args.output
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_id', 'timestamp', 'state', 'similarity', 'bbox'])
        writer.writeheader()
        writer.writerows(tracker.frame_logs)
    print(f"\nLogs saved to: {output_path}")

if __name__ == '__main__':
    main()
