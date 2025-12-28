"""
Tracker variants for benchmark comparison.
Each variant uses different feature combinations for person tracking.
"""

from .base_tracker import BaseTracker
from .iou_only import IoUOnlyTracker
from .shape_only import ShapeOnlyTracker
from .hsv_depth import HSVDepthTracker
from .shape_depth import ShapeDepthTracker
from .full_features import FullFeaturesTracker
from .deepsort_tracker import DeepSORTTracker

__all__ = [
    'BaseTracker',
    'IoUOnlyTracker',
    'ShapeOnlyTracker',
    'HSVDepthTracker',
    'ShapeDepthTracker',
    'FullFeaturesTracker',
    'DeepSORTTracker',
]
