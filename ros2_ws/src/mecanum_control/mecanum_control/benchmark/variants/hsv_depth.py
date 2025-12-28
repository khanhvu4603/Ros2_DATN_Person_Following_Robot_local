#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSV+Depth Tracker - Uses color histogram and depth features.

Feature dimension: 48-D (HSV) + 256-D (Depth) = 304-D
No CNN inference (faster than shape-based trackers).
Expected FPS: 25-28 | Expected Accuracy: ~70%
"""

import numpy as np
from typing import Optional, Tuple
from .base_tracker import BaseTracker


class HSVDepthTracker(BaseTracker):
    """
    Tracker using HSV histogram (48-D) and depth features (256-D).
    No CNN - faster but less accurate than shape-based methods.
    """
    
    def __init__(
        self, 
        similarity_threshold: float = 0.55,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 30,
        hsv_weight: float = 0.7,
        depth_weight: float = 0.3
    ):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.hsv_weight = hsv_weight
        self.depth_weight = depth_weight
        self.lost_count = 0
        self.target_feature = None
        
    def process_frame(
        self, 
        frame_id: int, 
        rgb_frame: np.ndarray, 
        depth_frame: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Tuple[int, int, int, int]], str, int]:
        """
        Process frame using HSV + Depth features.
        
        Returns:
            (box, state, track_id) tuple
        """
        # Detect persons
        detections = self._detect_persons(rgb_frame)
        
        if not detections:
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                self.state = 'LOST'
                self.current_box = None
                self.target_feature = None
            return self.current_box, self.state, 1
        
        # If no target yet, enroll largest detection
        if self.target_feature is None:
            largest_idx = np.argmax([self._box_area(box) for box in detections])
            self.current_box = detections[largest_idx]
            self.target_feature = self._extract_hsv_depth_feature(
                rgb_frame, self.current_box, depth_frame
            )
            if self.target_feature is None:
                return None, 'SEARCHING', 1
            self.state = 'LOCKED'
            self.lost_count = 0
            return self.current_box, self.state, 1
        
        # Find best match by HSV+Depth similarity
        best_box, best_score = None, 0.0
        
        for box in detections:
            # Extract HSV+Depth feature
            feat = self._extract_hsv_depth_feature(rgb_frame, box, depth_frame)
            if feat is None:
                continue
            
            # Compute cosine similarity
            score = np.dot(feat, self.target_feature)
            
            if score > best_score:
                best_score = score
                best_box = box
        
        # Update if good match found
        if best_score >= self.similarity_threshold:
            self.current_box = best_box
            self.state = 'LOCKED'
            self.lost_count = 0
            
            # Update target feature (EMA)
            new_feat = self._extract_hsv_depth_feature(rgb_frame, best_box, depth_frame)
            if new_feat is not None:
                self.target_feature = 0.8 * self.target_feature + 0.2 * new_feat
                self.target_feature /= (np.linalg.norm(self.target_feature) + 1e-8)
        else:
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                self.state = 'LOST'
                self.current_box = None
                self.target_feature = None
        
        return self.current_box, self.state, 1
    
    def _extract_hsv_depth_feature(
        self, 
        frame: np.ndarray, 
        box: Tuple[int, int, int, int],
        depth_frame: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Extract HSV histogram (48-D) + Depth (256-D) = 304-D total."""
        roi_padded, _ = self._body_arr_preserve_aspect_ratio(frame, box)
        if roi_padded is None:
            return None
        
        # 1. HSV histogram (48-D)
        hsv_feat = self._extract_hsv_histogram(roi_padded)
        hsv_feat *= self.hsv_weight
        
        # 2. Depth feature (256-D)
        depth_feat = self._extract_depth_feature(box, depth_frame)
        depth_feat *= self.depth_weight
        
        # 3. Concatenate
        feat = np.concatenate([hsv_feat, depth_feat], axis=0).astype(np.float32)
        feat /= (np.linalg.norm(feat) + 1e-8)
        
        return feat
    
    def _box_area(self, box: Tuple[int, int, int, int]) -> float:
        """Compute area of a box."""
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
