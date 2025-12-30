"""
DeepSORT Tracker variant for benchmark.

Implements full DeepSORT algorithm with Kalman Filter, cascade matching,
and appearance features. Adapted for single-target tracking.
"""

import numpy as np
from typing import Optional, Tuple
import cv2

from .base_tracker import BaseTracker
from .deepsort import (
    KalmanFilter,
    Track,
    Detection,
    NearestNeighborDistanceMetric,
    matching_cascade,
    min_cost_matching,
    gate_cost_matrix,
    chi2inv95
)


def iou(bbox, candidates):
    """
    Computer intersection over union for bboxes.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0.0, br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


class Tracker:
    """
    DeepSORT tracker with Kalman Filter and cascade matching.

    Parameters
    ----------
    metric : NearestNeighborDistanceMetric
        The distance metric for appearance-based matching.
    max_iou_distance : float
        Maximum IoU distance for IOU matching.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before track is confirmed.
    """

    def __init__(
        self,
        metric,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
    ):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """
        Perform measurement update and track management.

        Parameters
        ----------
        detections : List[Detection]
            A list of detections at the current time step.
        """
        # Run matching cascade
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections):
        """Match tracks to detections using cascade and IOU matching."""

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices
            )
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()
        ]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Associate confirmed tracks using appearance features
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            self._iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _iou_cost(self, tracks, detections, track_indices, detection_indices):
        """IOU distance metric for matching."""
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        for row, track_idx in enumerate(track_indices):
            if self.tracks[track_idx].time_since_update > 1:
                cost_matrix[row, :] = 1e5
                continue

            bbox = self.tracks[track_idx].to_tlwh()
            candidates = np.asarray([detections[i].tlwh for i in detection_indices])
            cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
        return cost_matrix

    def _initiate_track(self, detection):
        """Initialize new track from detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.max_age,
                detection.feature,
            )
        )
        self._next_id += 1


class DeepSORTTracker(BaseTracker):
    """
    DeepSORT tracker variant for benchmark with FULL FEATURES.
    
    Wraps full DeepSORT implementation to fit BaseTracker interface.
    Uses MobileNetV2 + HSV + Depth features (1584-D total).
    Adapted for single-target tracking by selecting and following one target.
    """

    def __init__(
        self,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        max_iou_distance: float = 0.7,
        max_age: int = 30,
        n_init: int = 3,
        shape_weight: float = 0.70,
        hsv_weight: float = 0.20,
        depth_weight: float = 0.10,
    ):
        """
        Initialize DeepSORT tracker with full features.

        Args:
            max_cosine_distance: Gating threshold for cosine distance metric
            nn_budget: Maximum size of feature gallery per track
            max_iou_distance: Gating threshold for IOU distance
            max_age: Maximum number of misses before track deletion
            n_init: Number of frames to confirm new track
            shape_weight: Weight for MobileNetV2 features (default 0.7)
            hsv_weight: Weight for HSV histogram features (default 0.2)
            depth_weight: Weight for depth features (default 0.1)
        """
        super().__init__()
        
        # Feature weights (matching person_detector.py)
        self.shape_weight = shape_weight
        self.hsv_weight = hsv_weight
        self.depth_weight = depth_weight
        
        # DeepSORT components
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(metric, max_iou_distance, max_age, n_init)
        
        # Single-target tracking state
        self.target_id = None
        self.state = 'SEARCHING'
        self.current_box = None
        
    def process_frame(
        self,
        frame_id: int,
        rgb_frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Tuple[int, int, int, int]], str, int]:
        """
        Process single frame with DeepSORT tracking using FULL FEATURES.

        Features: MobileNetV2 (1280-D) + HSV (48-D) + Depth (256-D) = 1584-D
        Matches person_detector.py feature extraction.

        Args:
            frame_id: Frame number
            rgb_frame: RGB image
            depth_frame: Depth image (used for depth features)

        Returns:
            (box, state, track_id):
                box: (x1, y1, x2, y2) or None
                state: 'SEARCHING', 'LOCKED', or 'LOST'
                track_id: Target track ID or -1
        """
        # Detect persons
        detections_raw = self._detect_persons(rgb_frame, conf_thresh=0.4)
        
        if len(detections_raw) == 0:
            # No detections
            self.tracker.predict()
            self.tracker.update([])
            self.current_box = None
            self.state = 'LOST' if self.target_id else 'SEARCHING'
            return None, self.state, self.target_id or -1
        
        # Extract FULL features for all detections
        detections = []
        for (x1, y1, x2, y2) in detections_raw:
            # Convert to tlwh format
            tlwh = [x1, y1, x2 - x1, y2 - y1]
            box = (x1, y1, x2, y2)
            
            # Extract full feature (1584-D)
            feature = self._extract_full_feature(rgb_frame, box, depth_frame)
            
            if feature is None:
                # Fallback to zero vector
                feature = np.zeros(1584, dtype=np.float32)
            
            # Use fixed confidence for detections
            confidence = 1.0
            
            detections.append(Detection(tlwh, confidence, feature))
        
        # DeepSORT predict and update
        self.tracker.predict()
        self.tracker.update(detections)
        
        # Get target track (single-target adaptation)
        return self._get_target_box()
    
    def _extract_full_feature(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        depth_frame: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Extract full features: MobileNetV2 (1280-D) + HSV (48-D) + Depth (256-D) = 1584-D.
        Same as enhanced_body_feature() in person_detector.py.
        """
        roi_padded, _ = self._body_arr_preserve_aspect_ratio(frame, box)
        if roi_padded is None:
            return None
        
        # 1. MobileNetV2 features (1280-D)
        roi_rgb = cv2.cvtColor(roi_padded, cv2.COLOR_BGR2RGB)
        arr = self._mb2_preprocess(roi_rgb)[None, ...]
        
        inp_name = self.mb2_sess.get_inputs()[0].name
        shape_feat = self.mb2_sess.run(None, {inp_name: arr.astype(np.float32)})[0]
        shape_feat = shape_feat.reshape(-1).astype(np.float32)
        shape_feat /= (np.linalg.norm(shape_feat) + 1e-8)
        shape_feat *= self.shape_weight
        
        # 2. HSV histogram (48-D)
        hsv_feat = self._extract_hsv_histogram(roi_padded)
        hsv_feat *= self.hsv_weight
        
        # 3. Depth feature (256-D)
        depth_feat = self._extract_depth_feature(box, depth_frame)
        depth_feat /= (np.linalg.norm(depth_feat) + 1e-8)
        depth_feat *= self.depth_weight
        
        # 4. Concatenate all features
        feat = np.concatenate([shape_feat, hsv_feat, depth_feat], axis=0).astype(np.float32)
        feat /= (np.linalg.norm(feat) + 1e-8)
        
        return feat
    
    def _get_target_box(self) -> Tuple[Optional[Tuple[int, int, int, int]], str, int]:
        """
        Extract target box from DeepSORT tracks (single-target adaptation).

        Returns:
            (box, state, track_id)
        """
        confirmed_tracks = [t for t in self.tracker.tracks if t.is_confirmed()]
        
        if self.target_id is None:
            # Frame 0 or no target yet: Select initial target
            if len(confirmed_tracks) == 0:
                self.current_box = None
                self.state = 'SEARCHING'
                return None, 'SEARCHING', -1
            
            # Select target: Largest box (closest person)
            target_track = max(confirmed_tracks, key=lambda t: t.to_tlwh()[2] * t.to_tlwh()[3])
            self.target_id = target_track.track_id
            
            box_tlbr = target_track.to_tlbr().astype(int)
            self.current_box = tuple(box_tlbr)
            self.state = 'LOCKED'
            return self.current_box, 'LOCKED', self.target_id
        
        # Find target track
        target_track = None
        for track in confirmed_tracks:
            if track.track_id == self.target_id:
                target_track = track
                break
        
        if target_track:
            # Target found
            box_tlbr = target_track.to_tlbr().astype(int)
            self.current_box = tuple(box_tlbr)
            self.state = 'LOCKED'
            return self.current_box, 'LOCKED', self.target_id
        else:
            # Target lost
            self.current_box = None
            self.state = 'LOST'
            return None, 'LOST', self.target_id
    
    def get_current_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current tracked box."""
        return self.current_box
    
    def get_state(self) -> str:
        """Get current tracking state."""
        return self.state
