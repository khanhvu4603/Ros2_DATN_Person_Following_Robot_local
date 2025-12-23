#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track class for DeepSORT.
Represents a single tracked object with Kalman state and feature history.
"""

from enum import IntEnum
import numpy as np


class TrackState(IntEnum):
    """
    Enumeration of possible track states.
    """
    Tentative = 1   # Not yet confirmed
    Confirmed = 2   # Confirmed track
    Deleted = 3     # Marked for deletion


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    
    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed.
    max_age : int
        The maximum number of misses before the track state is set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector for this track.
        
    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurrence.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    """
    
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
        
        self._n_init = n_init
        self._max_age = max_age
    
    def to_tlwh(self):
        """
        Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        
        Returns
        -------
        ndarray
            The bounding box.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # a * h = w
        ret[:2] -= ret[2:] / 2  # center to top-left
        return ret
    
    def to_tlbr(self):
        """
        Get current position in bounding box format `(min x, min y, max x, max y)`.
        
        Returns
        -------
        ndarray
            The bounding box.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    def to_xyah(self):
        """
        Get current position in format `(center x, center y, aspect ratio, height)`.
        
        Returns
        -------
        ndarray
            The bounding box in xyah format.
        """
        return self.mean[:4].copy()
    
    def predict(self, kf):
        """
        Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        
        Parameters
        ----------
        kf : KalmanFilter
            The Kalman filter.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, kf, detection, feature=None):
        """
        Perform Kalman filter measurement update step and update the feature cache.
        
        Parameters
        ----------
        kf : KalmanFilter
            The Kalman filter.
        detection : ndarray
            The associated detection bounding box in format (x, y, a, h).
        feature : Optional[ndarray]
            Feature vector for this detection.
        """
        # === MOTION-ADAPTIVE: Detect sudden stop ===
        # Get predicted position and velocity before update
        predicted_pos = self.mean[:4].copy()  # [x, y, a, h]
        predicted_velocity = self.mean[4:8].copy()  # [vx, vy, va, vh]
        
        # Calculate displacement between prediction and actual detection
        dx = detection[0] - predicted_pos[0]  # center x difference
        dy = detection[1] - predicted_pos[1]  # center y difference
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Calculate velocity magnitude (in pixels/frame)
        velocity_magnitude = np.sqrt(predicted_velocity[0]**2 + predicted_velocity[1]**2)
        
        # Detect sudden stop: high velocity prediction but detection is BEHIND prediction
        # This means target stopped but Kalman predicted it would continue moving
        # Condition: velocity > 5 px/frame AND displacement goes AGAINST velocity direction
        if velocity_magnitude > 5.0:
            # Check if detection is in opposite direction of velocity (target stopped/reversed)
            # Or if the actual movement is much less than predicted
            expected_movement = velocity_magnitude
            if displacement < expected_movement * 0.3:
                # Target moved much less than expected → sudden stop
                # Reset velocity to zero
                self.mean[4:8] = 0.0
        
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection)
        
        if feature is not None:
            self.features.append(feature)
            # Keep only the last 30 features
            if len(self.features) > 30:
                self.features = self.features[-30:]
        
        self.hits += 1
        self.time_since_update = 0
        
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
    
    def mark_missed(self):
        """
        Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
    
    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative
    
    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed
    
    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    
    def get_feature(self):
        """
        Get the mean feature vector from feature history.
        
        Returns
        -------
        ndarray or None
            Mean of features or None if no features.
        """
        if len(self.features) == 0:
            return None
        return np.mean(self.features, axis=0)
