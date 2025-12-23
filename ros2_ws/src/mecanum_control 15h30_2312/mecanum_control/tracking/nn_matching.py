#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nearest neighbor matching utilities for DeepSORT.
Includes distance metrics and cost matrix generation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


INFTY_COST = 1e5


def _cosine_distance(a, b):
    """
    Compute cosine distance between two vectors.
    
    Parameters
    ----------
    a : ndarray
        A single feature vector.
    b : ndarray
        A single feature vector.
        
    Returns
    -------
    float
        Cosine distance (1 - cosine similarity).
    """
    a = np.asarray(a) / (np.linalg.norm(a) + 1e-8)
    b = np.asarray(b) / (np.linalg.norm(b) + 1e-8)
    return 1. - np.dot(a, b)


def _nn_cosine_distance(track_features, detection_feature):
    """
    Compute the minimum cosine distance between a detection feature 
    and all features in a track's history.
    
    Parameters
    ----------
    track_features : list of ndarray
        List of feature vectors for a track.
    detection_feature : ndarray
        Single feature vector for the detection.
        
    Returns
    -------
    float
        Minimum cosine distance.
    """
    if len(track_features) == 0 or detection_feature is None:
        return INFTY_COST
    
    detection_feature = np.asarray(detection_feature)
    detection_feature = detection_feature / (np.linalg.norm(detection_feature) + 1e-8)
    
    distances = []
    for feat in track_features:
        feat = np.asarray(feat) / (np.linalg.norm(feat) + 1e-8)
        dist = 1. - np.dot(feat, detection_feature)
        distances.append(dist)
    
    return min(distances)


def iou(bbox, candidates):
    """
    Compute intersection over union.
    
    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format.
        
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
    
    tl = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]
    ]
    br = np.c_[
        np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]
    ]
    wh = np.maximum(0., br - tl)
    
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    
    return area_intersection / (area_bbox + area_candidates - area_intersection + 1e-8)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    IoU cost matrix.
    
    Parameters
    ----------
    tracks : List[Track]
        A list of tracks.
    detections : List[ndarray]
        A list of detections in tlbr format (x1, y1, x2, y2).
    track_indices : List[int]
        A list of indices to tracks that should be matched.
    detection_indices : List[int]
        A list of indices to detections that should be matched.
        
    Returns
    -------
    ndarray
        Returns a cost matrix of shape len(track_indices) x len(detection_indices).
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        track_tlwh = track.to_tlwh()
        
        # Convert detections from tlbr to tlwh
        candidates = []
        for det_idx in detection_indices:
            det = detections[det_idx]
            if isinstance(det, (list, tuple)):
                x1, y1, x2, y2 = det
            else:
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            w, h = x2 - x1, y2 - y1
            candidates.append([x1, y1, w, h])
        
        candidates = np.array(candidates)
        if len(candidates) > 0:
            iou_scores = iou(track_tlwh, candidates)
            cost_matrix[row, :] = 1. - iou_scores
    
    return cost_matrix


def appearance_cost(tracks, features, track_indices=None, detection_indices=None):
    """
    Appearance (ReID) cost matrix using cosine distance.
    
    Parameters
    ----------
    tracks : List[Track]
        A list of tracks.
    features : List[ndarray]
        A list of feature vectors for detections.
    track_indices : List[int]
        A list of indices to tracks.
    detection_indices : List[int]
        A list of indices to detections.
        
    Returns
    -------
    ndarray
        Returns a cost matrix.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(features)))
    
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        for col, det_idx in enumerate(detection_indices):
            det_feature = features[det_idx] if det_idx < len(features) else None
            cost_matrix[row, col] = _nn_cosine_distance(track.features, det_feature)
    
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, 
                     track_indices=None, detection_indices=None,
                     gated_cost=INFTY_COST, only_position=False):
    """
    Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.
    
    Parameters
    ----------
    kf : KalmanFilter
        The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix.
    tracks : List[Track]
        A list of predicted tracks.
    detections : List[ndarray]
        A list of detections in tlbr format.
    track_indices : List[int]
        List of track indices to match.
    detection_indices : List[int]
        List of detection indices to match.
    gated_cost : float
        Cost value for infeasible entries.
    only_position : bool
        If True, only x, y position is considered for gating.
        
    Returns
    -------
    ndarray
        The gated cost matrix.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    
    gating_dim = 2 if only_position else 4
    gating_threshold = kf.chi2inv95[gating_dim]
    
    # Convert detections to xyah format
    measurements = []
    for det_idx in detection_indices:
        det = detections[det_idx]
        if isinstance(det, (list, tuple)):
            x1, y1, x2, y2 = det
        else:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        a = w / (h + 1e-8)
        measurements.append([cx, cy, a, h])
    
    measurements = np.array(measurements)
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    
    return cost_matrix


def min_cost_matching(cost_matrix, max_distance, track_indices=None, detection_indices=None):
    """
    Solve linear assignment problem using Hungarian algorithm.
    
    Parameters
    ----------
    cost_matrix : ndarray
        The MxN cost matrix.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    track_indices : List[int]
        List of track indices.
    detection_indices : List[int]
        List of detection indices.
        
    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with:
        - List of matched track-detection index pairs.
        - List of unmatched track indices.
        - List of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = list(range(cost_matrix.shape[0]))
    if detection_indices is None:
        detection_indices = list(range(cost_matrix.shape[1]))
    
    if cost_matrix.size == 0:
        return [], track_indices, detection_indices
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_tracks = []
    unmatched_detections = list(detection_indices)
    
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
        else:
            matches.append((track_idx, detection_idx))
            if detection_idx in unmatched_detections:
                unmatched_detections.remove(detection_idx)
    
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    
    return matches, unmatched_tracks, unmatched_detections
