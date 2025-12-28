#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IoU-Only Tracker - Baseline tracker using only bounding box overlap.

No feature extraction, fastest variant.
Expected FPS: 25-30 | Expected Accuracy: ~65%
"""

import numpy as np
from typing import Optional, Tuple
from .base_tracker import BaseTracker


class IoUOnlyTracker(BaseTracker):
    """
    Baseline tracker that uses only IoU (Intersection over Union) for matching.
    No feature extraction - purely motion-based tracking.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.lost_count = 0
        
    def process_frame(
        self, 
        frame_id: int, 
        rgb_frame: np.ndarray, 
        depth_frame: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Tuple[int, int, int, int]], str, int]:
        """
        Process frame using IoU matching only.
        
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
            return self.current_box, self.state, 1
        
        # If no target yet, take largest detection
        if self.current_box is None:
            largest_idx = np.argmax([self._box_area(box) for box in detections])
            self.current_box = detections[largest_idx]
            self.state = 'LOCKED'
            self.lost_count = 0
            return self.current_box, self.state, 1
        
        # Find best IoU match
        best_box, best_iou = None, 0.0
        for box in detections:
            iou_score = self._compute_iou(box, self.current_box)
            if iou_score > best_iou:
                best_iou = iou_score
                best_box = box
        
        # Update if good match found
        if best_iou >= self.iou_threshold:
            self.current_box = best_box
            self.state = 'LOCKED'
            self.lost_count = 0
        else:
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                self.state = 'LOST'
                self.current_box = None
        
        return self.current_box, self.state, 1
    
    def _box_area(self, box: Tuple[int, int, int, int]) -> float:
        """Compute area of a box."""
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
