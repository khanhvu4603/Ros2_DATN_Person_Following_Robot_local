#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracking Evaluation Metrics for Single-Target Person Tracking.
Comprehensive metrics for comparing different tracking algorithms.

Usage:
    evaluator = TrackingEvaluator()
    for frame in video:
        evaluator.update(pred_box, gt_box, state, pred_id)
    results = evaluator.compute_metrics()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import time
import json
from pathlib import Path


@dataclass
class FrameResult:
    """Single frame evaluation result."""
    frame_id: int
    pred_box: Optional[Tuple[int, int, int, int]]  # x1, y1, x2, y2
    gt_box: Optional[Tuple[int, int, int, int]]
    pred_id: Optional[int]
    gt_id: int
    state: str  # SEARCHING, LOCKED, LOST
    iou: float
    is_correct_id: bool
    inference_time_ms: float


@dataclass
class TrackingMetrics:
    """All computed tracking metrics."""
    # === Core Accuracy Metrics ===
    target_lock_rate: float = 0.0       # % frames target is locked
    mean_iou: float = 0.0               # Average IoU when locked
    median_iou: float = 0.0
    
    # === ID Consistency ===
    id_switches: int = 0                # Number of wrong ID assignments
    id_precision: float = 0.0           # Correct ID / Total locked frames
    fragmentation: int = 0              # Track interrupted count
    
    # === Recovery Metrics ===
    avg_reacquisition_frames: float = 0.0  # Avg frames to recover after LOST
    max_lost_duration: int = 0
    recovery_success_rate: float = 0.0  # % of LOST episodes recovered
    
    # === Detection Metrics ===
    true_positives: int = 0
    false_positives: int = 0            # Locked on wrong person
    false_negatives: int = 0            # Target visible but not locked
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # === Performance Metrics ===
    mean_fps: float = 0.0
    std_fps: float = 0.0
    min_fps: float = 0.0
    p95_latency_ms: float = 0.0         # 95th percentile latency
    
    # === MOTA/MOTP (MOT Standard) ===
    mota: float = 0.0                   # Multi-Object Tracking Accuracy
    motp: float = 0.0                   # Multi-Object Tracking Precision
    
    # === IDF1 (ID F1 Score) ===
    idf1: float = 0.0
    
    # === Summary ===
    total_frames: int = 0
    total_gt_frames: int = 0            # Frames where GT exists


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    if box_a is None or box_b is None:
        return 0.0
    
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    
    # Intersection
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - inter
    
    return inter / (union + 1e-8)


def compute_center_error(box_a: Tuple, box_b: Tuple) -> float:
    """Compute center point distance between two boxes."""
    if box_a is None or box_b is None:
        return float('inf')
    
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2
    
    return np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)


class TrackingEvaluator:
    """
    Evaluator for single-target person tracking.
    
    Example:
        evaluator = TrackingEvaluator(iou_threshold=0.5)
        
        for frame_id, (pred, gt) in enumerate(results):
            evaluator.update(
                frame_id=frame_id,
                pred_box=pred['box'],
                gt_box=gt['box'],
                pred_id=pred['track_id'],
                gt_id=gt['person_id'],
                state=pred['state'],
                inference_time_ms=pred['time_ms']
            )
        
        metrics = evaluator.compute_metrics()
        evaluator.print_summary()
    """
    
    def __init__(self, iou_threshold: float = 0.5, target_gt_id: int = 1):
        """
        Args:
            iou_threshold: IoU threshold to consider as match (default 0.5)
            target_gt_id: Ground truth ID of the target person to track
        """
        self.iou_threshold = iou_threshold
        self.target_gt_id = target_gt_id
        
        self.frame_results: List[FrameResult] = []
        self.lost_episodes: List[int] = []  # Duration of each LOST episode
        self.current_lost_duration = 0
        
        self.last_correct_id: Optional[int] = None
        self.track_fragments: List[List[int]] = [[]]  # List of continuous track segments
        
    def update(
        self,
        frame_id: int,
        pred_box: Optional[Tuple[int, int, int, int]],
        gt_box: Optional[Tuple[int, int, int, int]],
        pred_id: Optional[int],
        gt_id: int,
        state: str,
        inference_time_ms: float = 0.0
    ):
        """
        Update evaluator with one frame's results.
        
        Args:
            frame_id: Current frame number
            pred_box: Predicted bounding box (x1, y1, x2, y2) or None
            gt_box: Ground truth bounding box or None (target not visible)
            pred_id: Predicted track ID
            gt_id: Ground truth person ID (should match target_gt_id)
            state: Tracker state ('SEARCHING', 'LOCKED', 'LOST', 'AUTO-ENROLL')
            inference_time_ms: Time taken for this frame
        """
        # Compute IoU
        iou = compute_iou(pred_box, gt_box) if pred_box and gt_box else 0.0
        
        # Check if ID is correct (only when locked on target)
        is_correct_id = False
        if state == 'LOCKED' and pred_id is not None:
            if self.last_correct_id is None:
                # First time locking - assume correct if IoU > threshold
                if iou >= self.iou_threshold:
                    self.last_correct_id = pred_id
                    is_correct_id = True
            else:
                is_correct_id = (pred_id == self.last_correct_id)
        
        # Track lost episodes
        if state == 'LOST' or state == 'SEARCHING':
            self.current_lost_duration += 1
        elif state == 'LOCKED':
            if self.current_lost_duration > 0:
                self.lost_episodes.append(self.current_lost_duration)
                self.current_lost_duration = 0
        
        # Track fragmentation
        if state == 'LOCKED' and iou >= self.iou_threshold:
            if len(self.track_fragments[-1]) == 0 or \
               frame_id == self.track_fragments[-1][-1] + 1:
                self.track_fragments[-1].append(frame_id)
            else:
                self.track_fragments.append([frame_id])
        
        # Store result
        result = FrameResult(
            frame_id=frame_id,
            pred_box=pred_box,
            gt_box=gt_box,
            pred_id=pred_id,
            gt_id=gt_id,
            state=state,
            iou=iou,
            is_correct_id=is_correct_id,
            inference_time_ms=inference_time_ms
        )
        self.frame_results.append(result)
    
    def compute_metrics(self) -> TrackingMetrics:
        """Compute all metrics from collected frame results."""
        metrics = TrackingMetrics()
        
        if len(self.frame_results) == 0:
            return metrics
        
        # === Basic counts ===
        metrics.total_frames = len(self.frame_results)
        
        locked_frames = [r for r in self.frame_results if r.state == 'LOCKED']
        gt_visible_frames = [r for r in self.frame_results if r.gt_box is not None]
        metrics.total_gt_frames = len(gt_visible_frames)
        
        # === Target Lock Rate ===
        if metrics.total_gt_frames > 0:
            locked_with_gt = [r for r in locked_frames if r.gt_box is not None]
            metrics.target_lock_rate = len(locked_with_gt) / metrics.total_gt_frames
        
        # === IoU Metrics ===
        ious = [r.iou for r in locked_frames if r.iou > 0]
        if ious:
            metrics.mean_iou = np.mean(ious)
            metrics.median_iou = np.median(ious)
        
        # === ID Consistency ===
        id_correct = [r for r in locked_frames if r.is_correct_id]
        id_wrong = [r for r in locked_frames if not r.is_correct_id and r.pred_id is not None]
        
        metrics.id_switches = len(id_wrong)
        if len(locked_frames) > 0:
            metrics.id_precision = len(id_correct) / len(locked_frames)
        
        # Fragmentation: count track segments - 1
        valid_fragments = [f for f in self.track_fragments if len(f) > 0]
        metrics.fragmentation = max(0, len(valid_fragments) - 1)
        
        # === Recovery Metrics ===
        if self.lost_episodes:
            metrics.avg_reacquisition_frames = np.mean(self.lost_episodes)
            metrics.max_lost_duration = max(self.lost_episodes)
        
        searching_to_locked = sum(1 for i, r in enumerate(self.frame_results[1:], 1)
                                  if r.state == 'LOCKED' and 
                                  self.frame_results[i-1].state in ['LOST', 'SEARCHING'])
        total_lost_episodes = len(self.lost_episodes)
        if total_lost_episodes > 0:
            metrics.recovery_success_rate = searching_to_locked / total_lost_episodes
        
        # === Detection Metrics (TP, FP, FN) ===
        for r in self.frame_results:
            if r.gt_box is not None:  # Target is visible
                if r.state == 'LOCKED' and r.iou >= self.iou_threshold:
                    metrics.true_positives += 1
                elif r.state == 'LOCKED' and r.iou < self.iou_threshold:
                    metrics.false_positives += 1  # Locked on wrong person
                else:
                    metrics.false_negatives += 1  # Target visible but not locked
        
        tp, fp, fn = metrics.true_positives, metrics.false_positives, metrics.false_negatives
        metrics.precision = tp / (tp + fp + 1e-8)
        metrics.recall = tp / (tp + fn + 1e-8)
        metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall + 1e-8)
        
        # === MOTA (Multi-Object Tracking Accuracy) ===
        # MOTA = 1 - (FN + FP + IDSW) / GT
        if metrics.total_gt_frames > 0:
            metrics.mota = 1.0 - (fn + fp + metrics.id_switches) / metrics.total_gt_frames
            metrics.mota = max(-1.0, min(1.0, metrics.mota))  # Clamp to [-1, 1]
        
        # === MOTP (Multi-Object Tracking Precision) ===
        # Average IoU of true positives
        tp_ious = [r.iou for r in self.frame_results 
                   if r.state == 'LOCKED' and r.iou >= self.iou_threshold]
        if tp_ious:
            metrics.motp = np.mean(tp_ious)
        
        # === IDF1 ===
        # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        idtp = len(id_correct)
        idfp = len(id_wrong)
        idfn = metrics.false_negatives
        metrics.idf1 = 2 * idtp / (2 * idtp + idfp + idfn + 1e-8)
        
        # === Performance Metrics ===
        times = [r.inference_time_ms for r in self.frame_results if r.inference_time_ms > 0]
        if times:
            fps_values = [1000.0 / t for t in times]
            metrics.mean_fps = np.mean(fps_values)
            metrics.std_fps = np.std(fps_values)
            metrics.min_fps = np.min(fps_values)
            metrics.p95_latency_ms = np.percentile(times, 95)
        
        return metrics
    
    def print_summary(self, metrics: Optional[TrackingMetrics] = None):
        """Print a formatted summary of metrics."""
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n" + "="*60)
        print("           TRACKING EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 ACCURACY METRICS")
        print(f"   Target Lock Rate:     {metrics.target_lock_rate*100:.1f}%")
        print(f"   Mean IoU:             {metrics.mean_iou:.3f}")
        print(f"   Median IoU:           {metrics.median_iou:.3f}")
        
        print(f"\n🔗 ID CONSISTENCY")
        print(f"   ID Precision:         {metrics.id_precision*100:.1f}%")
        print(f"   ID Switches:          {metrics.id_switches}")
        print(f"   Track Fragmentations: {metrics.fragmentation}")
        
        print(f"\n🔄 RECOVERY METRICS")
        print(f"   Avg Reacquisition:    {metrics.avg_reacquisition_frames:.1f} frames")
        print(f"   Max Lost Duration:    {metrics.max_lost_duration} frames")
        print(f"   Recovery Success:     {metrics.recovery_success_rate*100:.1f}%")
        
        print(f"\n🎯 DETECTION METRICS")
        print(f"   Precision:            {metrics.precision*100:.1f}%")
        print(f"   Recall:               {metrics.recall*100:.1f}%")
        print(f"   F1 Score:             {metrics.f1_score*100:.1f}%")
        
        print(f"\n📈 MOT STANDARD METRICS")
        print(f"   MOTA:                 {metrics.mota*100:.1f}%")
        print(f"   MOTP:                 {metrics.motp:.3f}")
        print(f"   IDF1:                 {metrics.idf1*100:.1f}%")
        
        print(f"\n⚡ PERFORMANCE")
        print(f"   Mean FPS:             {metrics.mean_fps:.1f}")
        print(f"   Min FPS:              {metrics.min_fps:.1f}")
        print(f"   P95 Latency:          {metrics.p95_latency_ms:.1f} ms")
        
        print(f"\n📝 SUMMARY")
        print(f"   Total Frames:         {metrics.total_frames}")
        print(f"   GT Visible Frames:    {metrics.total_gt_frames}")
        print("="*60 + "\n")
    
    def export_to_json(self, filepath: str):
        """Export metrics to JSON file."""
        metrics = self.compute_metrics()
        
        data = {
            'metrics': {
                'target_lock_rate': metrics.target_lock_rate,
                'mean_iou': metrics.mean_iou,
                'median_iou': metrics.median_iou,
                'id_switches': metrics.id_switches,
                'id_precision': metrics.id_precision,
                'fragmentation': metrics.fragmentation,
                'avg_reacquisition_frames': metrics.avg_reacquisition_frames,
                'max_lost_duration': metrics.max_lost_duration,
                'recovery_success_rate': metrics.recovery_success_rate,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'mota': metrics.mota,
                'motp': metrics.motp,
                'idf1': metrics.idf1,
                'mean_fps': metrics.mean_fps,
                'min_fps': metrics.min_fps,
                'p95_latency_ms': metrics.p95_latency_ms,
                'total_frames': metrics.total_frames
            },
            'frame_results': [
                {
                    'frame_id': r.frame_id,
                    'state': r.state,
                    'iou': r.iou,
                    'is_correct_id': r.is_correct_id,
                    'inference_time_ms': r.inference_time_ms
                }
                for r in self.frame_results
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics exported to: {filepath}")
    
    def reset(self):
        """Reset evaluator for new evaluation run."""
        self.frame_results = []
        self.lost_episodes = []
        self.current_lost_duration = 0
        self.last_correct_id = None
        self.track_fragments = [[]]


# ============== QUICK TEST ==============
if __name__ == '__main__':
    # Example usage with dummy data
    evaluator = TrackingEvaluator(iou_threshold=0.5)
    
    # Simulate 100 frames
    np.random.seed(42)
    gt_box = (100, 100, 200, 300)  # Fixed ground truth
    
    states = ['SEARCHING'] * 5 + ['LOCKED'] * 80 + ['LOST'] * 10 + ['LOCKED'] * 5
    
    for i, state in enumerate(states):
        if state == 'LOCKED':
            # Add some noise to prediction
            noise = np.random.randint(-10, 10, 4)
            pred_box = tuple(np.array(gt_box) + noise)
            pred_id = 1
        else:
            pred_box = None
            pred_id = None
        
        evaluator.update(
            frame_id=i,
            pred_box=pred_box,
            gt_box=gt_box,
            pred_id=pred_id,
            gt_id=1,
            state=state,
            inference_time_ms=np.random.uniform(50, 100)
        )
    
    evaluator.print_summary()
