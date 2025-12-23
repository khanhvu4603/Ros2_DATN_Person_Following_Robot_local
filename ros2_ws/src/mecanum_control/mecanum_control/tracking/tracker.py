#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSORT Tracker - Main tracker class.
Manages multiple tracks using Kalman filtering and Hungarian matching.

Optimized for single-target tracking with 2-3 people in frame.
"""

import numpy as np
from typing import List, Optional, Tuple

from .kalman_filter import KalmanFilter
from .track import Track, TrackState
from . import nn_matching


class DeepSORTTracker:
    """
    DeepSORT multi-object tracker with appearance features.
    
    Optimized for single-target person tracking on CPU.
    
    Parameters
    ----------
    max_age : int
        Maximum number of missed frames before a track is deleted.
    n_init : int
        Number of consecutive detections before a track is confirmed.
    max_cosine_distance : float
        Maximum cosine distance threshold for appearance matching.
    lambda_weight : float
        Weight for motion cost (1 - lambda_weight for appearance cost).
    """
    
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        lambda_weight: float = 0.3
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.lambda_weight = lambda_weight
        
        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self._next_id = 1
    
    def predict(self):
        """
        Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)
    
    def update(self, detections: List[Tuple], features: List[np.ndarray]) -> List[Track]:
        """
        Perform measurement update and track management.
        
        Parameters
        ----------
        detections : List[Tuple]
            A list of detections in tlbr format (x1, y1, x2, y2).
        features : List[np.ndarray]
            A list of appearance feature vectors, one for each detection.
            
        Returns
        -------
        List[Track]
            List of currently active tracks.
        """
        # Run Kalman filter prediction
        self.predict()
        
        # Run matching cascade
        matches, unmatched_tracks, unmatched_detections = self._match(
            detections, features
        )
        
        # Update matched tracks
        for track_idx, detection_idx in matches:
            detection = detections[detection_idx]
            feature = features[detection_idx] if detection_idx < len(features) else None
            
            # Convert tlbr to xyah
            measurement = self._tlbr_to_xyah(detection)
            self.tracks[track_idx].update(self.kf, measurement, feature)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Start new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            feature = features[detection_idx] if detection_idx < len(features) else None
            self._initiate_track(detection, feature)
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        return self.tracks
    
    def _match(
        self,
        detections: List[Tuple],
        features: List[np.ndarray]
    ) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Match detections with tracks using cascade matching.
        
        1. Match confirmed tracks using appearance + motion
        2. Match remaining with tentative tracks using IoU only
        
        Returns
        -------
        (List[(track_idx, det_idx)], List[unmatched_track_idx], List[unmatched_det_idx])
        """
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_tracks = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        
        # === STAGE 1: Match confirmed tracks with appearance + motion ===
        matches_a, unmatched_tracks_a, unmatched_detections = self._match_confirmed(
            detections, features, confirmed_tracks
        )
        
        # === STAGE 2: Match tentative tracks with IoU only ===
        remaining_tracks = list(set(tentative_tracks).union(set(unmatched_tracks_a)))
        matches_b, unmatched_tracks_b, unmatched_detections = self._match_iou(
            detections, remaining_tracks, unmatched_detections
        )
        
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_b))
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _match_confirmed(
        self,
        detections: List[Tuple],
        features: List[np.ndarray],
        track_indices: List[int]
    ) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Match confirmed tracks using combined motion and appearance cost.
        """
        if len(track_indices) == 0 or len(detections) == 0:
            return [], track_indices, list(range(len(detections)))
        
        detection_indices = list(range(len(detections)))
        
        # Calculate appearance cost
        appearance_cost_matrix = nn_matching.appearance_cost(
            self.tracks, features, track_indices, detection_indices
        )
        
        # Calculate IoU cost (as proxy for motion when Kalman gating is expensive)
        iou_cost_matrix = nn_matching.iou_cost(
            self.tracks, detections, track_indices, detection_indices
        )
        
        # Combined cost: lambda * motion + (1 - lambda) * appearance
        cost_matrix = (
            self.lambda_weight * iou_cost_matrix +
            (1 - self.lambda_weight) * appearance_cost_matrix
        )
        
        # Apply Kalman gating
        cost_matrix = nn_matching.gate_cost_matrix(
            self.kf, cost_matrix, self.tracks, detections,
            track_indices, detection_indices,
            gated_cost=nn_matching.INFTY_COST
        )
        
        # Apply appearance threshold
        cost_matrix[appearance_cost_matrix > self.max_cosine_distance] = nn_matching.INFTY_COST
        
        # Hungarian matching
        matches, unmatched_tracks, unmatched_detections = nn_matching.min_cost_matching(
            cost_matrix, max_distance=nn_matching.INFTY_COST,
            track_indices=track_indices, detection_indices=detection_indices
        )
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _match_iou(
        self,
        detections: List[Tuple],
        track_indices: List[int],
        detection_indices: List[int]
    ) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Match tracks using IoU only.
        """
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], track_indices, detection_indices
        
        # Calculate IoU cost
        iou_cost_matrix = nn_matching.iou_cost(
            self.tracks, detections, track_indices, detection_indices
        )
        
        # Hungarian matching with IoU threshold
        matches, unmatched_tracks, unmatched_detections = nn_matching.min_cost_matching(
            iou_cost_matrix, max_distance=0.7,  # 1 - 0.3 = IoU > 0.3
            track_indices=track_indices, detection_indices=detection_indices
        )
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _initiate_track(self, detection: Tuple, feature: Optional[np.ndarray]):
        """
        Create a new track from an unmatched detection.
        """
        measurement = self._tlbr_to_xyah(detection)
        mean, covariance = self.kf.initiate(measurement)
        
        self.tracks.append(Track(
            mean=mean,
            covariance=covariance,
            track_id=self._next_id,
            n_init=self.n_init,
            max_age=self.max_age,
            feature=feature
        ))
        self._next_id += 1
    
    def _tlbr_to_xyah(self, bbox: Tuple) -> np.ndarray:
        """
        Convert bounding box from tlbr (x1, y1, x2, y2) to xyah format.
        
        Returns
        -------
        ndarray
            Bounding box in (center_x, center_y, aspect_ratio, height) format.
        """
        if isinstance(bbox, (list, tuple)):
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        a = w / (h + 1e-8)
        
        return np.array([cx, cy, a, h])
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        Get a track by its ID.
        
        Parameters
        ----------
        track_id : int
            The track ID to search for.
            
        Returns
        -------
        Track or None
            The track with the given ID, or None if not found.
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_confirmed_tracks(self) -> List[Track]:
        """
        Get all confirmed tracks.
        
        Returns
        -------
        List[Track]
            List of confirmed tracks.
        """
        return [t for t in self.tracks if t.is_confirmed()]
    
    def reset(self):
        """
        Reset the tracker, clearing all tracks.
        """
        self.tracks = []
        self._next_id = 1
