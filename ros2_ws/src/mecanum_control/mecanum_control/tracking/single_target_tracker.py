#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Target Tracker using Kalman Filter with Zero Velocity Model.

This is a simplified tracker that only manages ONE target.
No Hungarian matching, no multi-track management.

Features:
- Kalman Filter for smooth tracking
- Zero Velocity Model: khi không có detection, box ĐỨNG YÊN (không drift)
- Motion detection: tự động phát hiện target đang đi hay đứng

Optimized for single-person following robot applications.
"""

import numpy as np
from typing import Optional, Tuple

from .kalman_filter import KalmanFilter


class SingleTargetTracker:
    """
    Single Target Tracker using Kalman Filter with Zero Velocity switching.
    
    Only manages ONE target at a time. Detection is used to confirm
    the target identity, not to assign via Hungarian matching.
    
    Zero Velocity Model:
    - Khi target DỪNG hoặc bị CHE KHUẤT: velocity = 0, box đứng yên
    - Khi target DI CHUYỂN: sử dụng Kalman predict bình thường
    
    Parameters
    ----------
    max_time_since_update : int
        Maximum frames without detection before marking as lost.
    stop_velocity_threshold : float
        Velocity threshold (pixels/frame) below which target is considered stopped.
    stop_displacement_threshold : float
        Displacement threshold (pixels) for sudden stop detection.
    """
    
    def __init__(self, 
                 max_time_since_update: int = 30,
                 stop_velocity_threshold: float = 3.0,
                 stop_displacement_threshold: float = 15.0):
        self.kf = KalmanFilter()
        self.max_time_since_update = max_time_since_update
        self.stop_velocity_threshold = stop_velocity_threshold
        self.stop_displacement_threshold = stop_displacement_threshold
        
        # Target state
        self.is_tracking = False
        self.mean: Optional[np.ndarray] = None          # Kalman state [x, y, a, h, vx, vy, va, vh]
        self.covariance: Optional[np.ndarray] = None    # Kalman covariance 8x8
        self.target_feature: Optional[np.ndarray] = None # ReID feature
        self.time_since_update: int = 0                 # Frames since last detection match
        
        # Store raw detection bbox for direct access
        self.last_detection_bbox: Optional[Tuple[int, int, int, int]] = None
        
        # Zero Velocity Model state
        self.is_moving = True  # True = target đang đi, False = target dừng
        self.last_position: Optional[Tuple[float, float]] = None  # (cx, cy) vị trí cuối
        self.position_history: list = []  # Lưu vài frame position để detect stop
        self.history_size = 5  # Số frame để tính displacement
    
    def initiate(self, detection_bbox: Tuple, feature: Optional[np.ndarray] = None):
        """
        Initialize tracking with the first detection.
        
        Parameters
        ----------
        detection_bbox : Tuple
            Bounding box in tlbr format (x1, y1, x2, y2).
        feature : np.ndarray, optional
            ReID feature vector for this detection.
        """
        measurement = self._tlbr_to_xyah(detection_bbox)
        self.mean, self.covariance = self.kf.initiate(measurement)
        self.target_feature = feature
        self.is_tracking = True
        self.time_since_update = 0
        self.last_detection_bbox = tuple(map(int, detection_bbox))
        
        # Initialize Zero Velocity state
        self.is_moving = True
        self.last_position = (float(measurement[0]), float(measurement[1]))
        self.position_history = [self.last_position]
    
    def predict(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Run Kalman filter prediction step with Zero Velocity Model.
        
        - Nếu có detection gần đây (time_since_update == 0): predict bình thường
        - Nếu KHÔNG có detection (che khuất): ZERO VELOCITY (box đứng yên)
        
        Returns
        -------
        Tuple or None
            Predicted bounding box in tlbr format (x1, y1, x2, y2), or None if not tracking.
        """
        if not self.is_tracking or self.mean is None or self.covariance is None:
            return None
        
        # === ZERO VELOCITY MODEL - DISABLED ===
        # Khi không có detection (bị che khuất): box sẽ dự đoán theo velocity cũ
        # Đã tắt để robot có thể di chuyển smooth
        # if self.time_since_update > 0:
        #     self.mean[4:8] = 0.0  # [vx, vy, va, vh] = 0
        
        # Kalman predict bình thường
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.time_since_update += 1
        
        return self.to_tlbr()
    
    def update(self, detection_bbox: Tuple, feature: Optional[np.ndarray] = None):
        """
        Run Kalman filter update step with a matched detection.
        Includes motion detection for Zero Velocity switching.
        
        Parameters
        ----------
        detection_bbox : Tuple
            Bounding box in tlbr format (x1, y1, x2, y2).
        feature : np.ndarray, optional
            ReID feature vector. If provided, can be used for adaptive update.
        """
        if not self.is_tracking or self.mean is None or self.covariance is None:
            return
        
        measurement = self._tlbr_to_xyah(detection_bbox)
        current_position = (float(measurement[0]), float(measurement[1]))
        
        # === MOTION DETECTION - DISABLED ===
        # self._update_motion_state(current_position)
        
        # === SUDDEN STOP DETECTION - DISABLED ===
        # if not self.is_moving:
        #     self.mean[4:8] = 0.0  # Reset velocity
        
        # Kalman update bình thường
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)
        
        # === VELOCITY DAMPING (nhẹ) - DISABLED ===
        # self.mean[4:8] *= 0.95  # 5% reduction per frame
        
        # Update tracking state
        self.time_since_update = 0
        self.last_detection_bbox = tuple(map(int, detection_bbox))
        self.last_position = current_position
    
    def _update_motion_state(self, current_position: Tuple[float, float]):
        """
        Detect if target is moving or stopped.
        
        Cập nhật self.is_moving dựa trên displacement trong vài frame gần đây.
        """
        # Add to history
        self.position_history.append(current_position)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        if len(self.position_history) < 2:
            self.is_moving = True
            return
        
        # Calculate displacement over history
        oldest = self.position_history[0]
        displacement = np.sqrt(
            (current_position[0] - oldest[0]) ** 2 + 
            (current_position[1] - oldest[1]) ** 2
        )
        
        # Average displacement per frame
        avg_displacement = displacement / len(self.position_history)
        
        # Determine if moving
        if avg_displacement < self.stop_displacement_threshold / self.history_size:
            self.is_moving = False  # Target dừng
        else:
            self.is_moving = True   # Target đang đi
    
    def to_tlbr(self) -> Tuple[int, int, int, int]:
        """
        Get current bounding box from Kalman state.
        
        Returns
        -------
        Tuple
            Bounding box in tlbr format (x1, y1, x2, y2).
        """
        if self.mean is None:
            return (0, 0, 0, 0)
        
        # mean = [x, y, a, h, vx, vy, va, vh]
        # Extract position components
        cx, cy, a, h = self.mean[:4]
        w = a * h
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        return (x1, y1, x2, y2)
    
    def get_smoothed_box(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get Kalman-smoothed bounding box.
        Alias cho to_tlbr() để rõ nghĩa hơn.
        
        Returns
        -------
        Tuple or None
            Smoothed bounding box in tlbr format.
        """
        if not self.is_tracking:
            return None
        return self.to_tlbr()
    
    def get_raw_detection_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the last raw detection bounding box (not smoothed by Kalman).
        
        Returns
        -------
        Tuple or None
            Last detection bbox in tlbr format, or None if no detection yet.
        """
        return self.last_detection_bbox
    
    def is_target_stopped(self) -> bool:
        """
        Check if target is currently stopped (not moving).
        
        Returns
        -------
        bool
            True if target is stopped.
        """
        return not self.is_moving
    
    def is_lost(self) -> bool:
        """
        Check if the target is considered lost.
        
        Returns
        -------
        bool
            True if time_since_update exceeds max_time_since_update.
        """
        return self.time_since_update > self.max_time_since_update
    
    def reset(self):
        """
        Reset the tracker to initial state.
        """
        self.is_tracking = False
        self.mean = None
        self.covariance = None
        self.target_feature = None
        self.time_since_update = 0
        self.last_detection_bbox = None
        self.is_moving = True
        self.last_position = None
        self.position_history = []
    
    def _tlbr_to_xyah(self, bbox: Tuple) -> np.ndarray:
        """
        Convert bounding box from tlbr (x1, y1, x2, y2) to xyah format.
        
        Parameters
        ----------
        bbox : Tuple
            Bounding box in (x1, y1, x2, y2) format.
            
        Returns
        -------
        np.ndarray
            Bounding box in (center_x, center_y, aspect_ratio, height) format.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        a = w / (h + 1e-8)  # aspect ratio
        
        return np.array([cx, cy, a, h], dtype=np.float32)
