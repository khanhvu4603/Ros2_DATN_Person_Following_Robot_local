#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Tracker - Abstract base class for all tracker variants.

All tracker variants must inherit from this class and implement process_frame().
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort


class BaseTracker(ABC):
    """
    Abstract base class for tracker variants.
    
    All variants must implement:
        - process_frame(frame_id, rgb_frame, depth_frame) -> (box, state, track_id)
    
    Provides common utilities:
        - Person detection (MobileNet-SSD)
        - Feature extraction (MobileNetV2, HSV, Depth)
        - IoU computation
    """
    
    def __init__(self):
        """Initialize base tracker with common components."""
        self.current_box = None
        self.state = 'SEARCHING'  # SEARCHING, LOCKED, LOST
        
        # Paths
        self.HERE = Path(__file__).resolve().parent.parent.parent
        self.MODELS = self.HERE / "models"
        
        # Load person detector (MobileNet-SSD)
        mobilenet_prototxt = str(self.MODELS / "MobileNetSSD_deploy.prototxt")
        mobilenet_weights = str(self.MODELS / "MobileNetSSD_deploy.caffemodel")
        
        if Path(mobilenet_prototxt).exists() and Path(mobilenet_weights).exists():
            self.ssd_net = cv2.dnn.readNetFromCaffe(mobilenet_prototxt, mobilenet_weights)
        else:
            print(f"⚠️ Warning: MobileNet-SSD model not found at {self.MODELS}")
            self.ssd_net = None
        
        # Load MobileNetV2 for feature extraction
        mb2_onnx_path = str(self.MODELS / "mb2_gap.onnx")
        if Path(mb2_onnx_path).exists():
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.mb2_sess = ort.InferenceSession(
                mb2_onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
        else:
            print(f"⚠️ Warning: MobileNetV2 model not found at {mb2_onnx_path}")
            self.mb2_sess = None
    
    @abstractmethod
    def process_frame(
        self, 
        frame_id: int, 
        rgb_frame: np.ndarray, 
        depth_frame: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Tuple[int, int, int, int]], str, int]:
        """
        Process a frame and return tracking result.
        
        Args:
            frame_id: Frame number
            rgb_frame: RGB image (H, W, 3)
            depth_frame: Depth image (H, W) or None
            
        Returns:
            Tuple of (bounding_box, state, track_id):
                - bounding_box: (x1, y1, x2, y2) or None if not detected
                - state: 'SEARCHING', 'LOCKED', or 'LOST'
                - track_id: Integer ID of the track
        """
        raise NotImplementedError("Subclass must implement process_frame()")
    
    def get_current_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current tracked bounding box."""
        return self.current_box
    
    def get_state(self) -> str:
        """Get current tracker state."""
        return self.state
    
    # ==================== COMMON UTILITIES ====================
    
    def _detect_persons(
        self, 
        frame: np.ndarray, 
        conf_thresh: float = 0.4
    ) -> list:
        """
        Detect persons using MobileNet-SSD.
        
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        if self.ssd_net is None:
            return []
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            0.007843, 
            (300, 300), 
            127.5
        )
        self.ssd_net.setInput(blob)
        detections = self.ssd_net.forward()
        
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            
            if class_id == 15 and confidence > conf_thresh:  # class 15 = person
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        
        return boxes
    
    def _compute_iou(
        self, 
        box_a: Tuple[int, int, int, int], 
        box_b: Tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two boxes."""
        if box_a is None or box_b is None:
            return 0.0
        
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        
        # Intersection
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter_area = iw * ih
        
        # Union
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area + 1e-6
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _body_arr_preserve_aspect_ratio(
        self, 
        frame: np.ndarray, 
        box: Tuple[int, int, int, int], 
        target_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Extract ROI and resize to target size while preserving aspect ratio.
        
        Returns:
            (padded_roi, scale) or (None, None) if extraction failed
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Clamp coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Check validity
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None
        
        roi_h, roi_w = roi.shape[:2]
        target_h, target_w = target_size
        
        scale = min(target_w / roi_w, target_h / roi_h)
        new_w, new_h = int(roi_w * scale), int(roi_h * scale)
        
        if new_w <= 0 or new_h <= 0:
            return None, None
        
        resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_roi
        
        return padded, scale
    
    def _mb2_preprocess(self, img_rgb: np.ndarray) -> np.ndarray:
        """Preprocess image for MobileNetV2 (Keras style)."""
        x = img_rgb.astype(np.float32)
        x = x / 127.5 - 1.0
        return x
    
    def _extract_hsv_histogram(
        self, 
        roi_bgr: np.ndarray, 
        bins: int = 16, 
        v_weight: float = 0.6,
        normalize_brightness: bool = True
    ) -> np.ndarray:
        """
        Extract HSV histogram features (48-D).
        
        Returns:
            Normalized histogram array (48-D)
        """
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
    
    def _extract_depth_feature(
        self, 
        box: Tuple[int, int, int, int], 
        depth_img: Optional[np.ndarray],
        target_size: Tuple[int, int] = (16, 16)
    ) -> np.ndarray:
        """
        Extract depth features (256-D).
        
        Returns:
            Normalized depth feature array (256-D)
        """
        if depth_img is None or box is None:
            return np.zeros(target_size[0] * target_size[1], dtype=np.float32)
        
        x1, y1, x2, y2 = map(int, box)
        
        # Clamp
        h, w = depth_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(target_size[0] * target_size[1], dtype=np.float32)
        
        roi = depth_img[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(target_size[0] * target_size[1], dtype=np.float32)
        
        roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize: closer (smaller value) -> 1.0, farther (larger value) -> 0.0
        # Assume target in 0.5m to 5m range
        roi_normalized = np.clip((5000 - roi_resized) / 4500.0, 0.0, 1.0)
        
        depth_feat = roi_normalized.flatten().astype(np.float32)
        return depth_feat
