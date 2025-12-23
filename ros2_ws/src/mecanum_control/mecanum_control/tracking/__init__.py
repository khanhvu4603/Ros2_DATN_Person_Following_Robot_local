#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSORT Tracking Sub-package for Person Detection.

This package provides a custom DeepSORT implementation optimized for:
- Single-target person tracking
- OrangePi 5 Plus (CPU-only)
- 2-3 people in frame
- ROS2 integration

Components:
- KalmanFilter: 8-dimensional state space tracker
- Track: Single object track with feature history
- DeepSORTTracker: Main tracker with cascade matching
"""

from .kalman_filter import KalmanFilter
from .track import Track, TrackState
from .tracker import DeepSORTTracker

__all__ = [
    'KalmanFilter',
    'Track',
    'TrackState', 
    'DeepSORTTracker'
]

__version__ = '1.0.0'
