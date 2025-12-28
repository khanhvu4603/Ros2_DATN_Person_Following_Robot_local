#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Runner - Main script to run all tracker variants on test videos.

Usage:
    # With real video (RGB + Depth separate files)
    python benchmark/run_benchmark.py \\
        --rgb-video benchmark/data/videos/test_rgb.mp4 \\
        --depth-video benchmark/data/videos/test_depth.mp4 \\
        --gt benchmark/data/annotations/test_gt.json \\
        --variants all \\
        --output benchmark/results/
    
    # With split-screen video
    python benchmark/run_benchmark.py \\
        --split-video benchmark/data/videos/test_splitscreen.mp4 \\
        --rgb-position right \\
        --gt benchmark/data/annotations/test_gt.json \\
        --variants all
    
    # With synthetic data (testing)
    python benchmark/run_benchmark.py --synthetic --variants all
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import cv2

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from evaluation.tracking_metrics import TrackingEvaluator, TrackingMetrics
from benchmark.variants import (
    IoUOnlyTracker,
    ShapeOnlyTracker,
    HSVDepthTracker,
    ShapeDepthTracker,
    FullFeaturesTracker,
    DeepSORTTracker
)


class BenchmarkRunner:
    """
    Run benchmarks on different tracker variants.
    """
    
    def __init__(
        self,
        rgb_video_path: Optional[str] = None,
        depth_video_path: Optional[str] = None,
        split_video_path: Optional[str] = None,
        rgb_position: str = 'right',
        gt_path: Optional[str] = None,
        output_dir: str = './benchmark/results',
        synthetic: bool = False
    ):
        self.rgb_video_path = rgb_video_path
        self.depth_video_path = depth_video_path
        self.split_video_path = split_video_path
        self.rgb_position = rgb_position
        self.gt_path = gt_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic = synthetic
        
        # Load ground truth if provided
        self.ground_truth = None
        if gt_path and Path(gt_path).exists():
            with open(gt_path, 'r') as f:
                self.ground_truth = json.load(f)
        
        # Results storage
        self.results = {}
    
    def load_video_frames(self):
        """
        Load video frames from either separate RGB/Depth files or split-screen.
        
        Returns:
            Generator of (frame_id, rgb_frame, depth_frame) tuples
        """
        if self.synthetic:
            # Generate synthetic data for testing
            yield from self._generate_synthetic_frames(num_frames=500)
            return
        
        if self.split_video_path:
            # Load from split-screen video
            yield from self._load_split_screen_video()
        elif self.rgb_video_path:
            # Load from separate RGB and Depth videos
            yield from self._load_separate_videos()
        else:
            raise ValueError("Must provide either rgb_video_path or split_video_path")
    
    def _load_separate_videos(self):
        """Load RGB and Depth from separate video files."""
        rgb_cap = cv2.VideoCapture(self.rgb_video_path)
        depth_cap = cv2.VideoCapture(self.depth_video_path) if self.depth_video_path else None
        
        if not rgb_cap.isOpened():
            raise ValueError(f"Cannot open RGB video: {self.rgb_video_path}")
        
        frame_id = 0
        while True:
            ret_rgb, rgb_frame = rgb_cap.read()
            if not ret_rgb:
                break
            
            depth_frame = None
            if depth_cap:
                ret_depth, depth_frame = depth_cap.read()
                if not ret_depth:
                    print(f"⚠️ Warning: Depth video ended at frame {frame_id}")
                    depth_frame = None
            
            yield (frame_id, rgb_frame, depth_frame)
            frame_id += 1
        
        rgb_cap.release()
        if depth_cap:
            depth_cap.release()
    
    def _load_split_screen_video(self):
        """Load from split-screen video (RGB and Depth side-by-side)."""
        cap = cv2.VideoCapture(self.split_video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open split-screen video: {self.split_video_path}")
        
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Split frame
            h, w = frame.shape[:2]
            half_w = w // 2
            
            if self.rgb_position == 'right':
                depth_frame = frame[:, :half_w]
                rgb_frame = frame[:, half_w:]
            else:  # left
                rgb_frame = frame[:, :half_w]
                depth_frame = frame[:, half_w:]
            
            yield (frame_id, rgb_frame, depth_frame)
            frame_id += 1
        
        cap.release()
    
    def _generate_synthetic_frames(self, num_frames: int = 500):
        """Generate synthetic frames for testing (no video needed)."""
        for frame_id in range(num_frames):
            # Generate dummy RGB frame
            rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Generate dummy depth frame
            depth_frame = np.random.randint(500, 5000, (480, 640), dtype=np.uint16)
            
            yield (frame_id, rgb_frame, depth_frame)
    
    def run_variant(
        self,
        variant_name: str,
        tracker_class
    ) -> TrackingMetrics:
        """
        Run a single tracker variant on the video.
        
        Args:
            variant_name: Name of the variant (e.g., 'iou_only')
            tracker_class: Tracker class to instantiate
            
        Returns:
            TrackingMetrics object with results
        """
        print(f"\n{'='*70}")
        print(f"Running: {variant_name}")
        print(f"{'='*70}")
        
        # Initialize tracker
        tracker = tracker_class()
        
        # Initialize evaluator
        evaluator = TrackingEvaluator(iou_threshold=0.5)
        
        # Process all frames
        start_time = time.time()
        frame_count = 0
        
        for frame_id, rgb_frame, depth_frame in self.load_video_frames():
            frame_start = time.time()
            
            # Run tracker
            pred_box, state, track_id = tracker.process_frame(
                frame_id, rgb_frame, depth_frame
            )
            
            frame_time = (time.time() - frame_start) * 1000  # ms
            
            # Get ground truth for this frame
            gt_box = None
            if self.ground_truth and 'frames' in self.ground_truth:
                gt_frames = {f['frame_id']: f for f in self.ground_truth['frames']}
                if frame_id in gt_frames:
                    box_data = gt_frames[frame_id]['box']
                    gt_box = tuple(box_data)  # [x1, y1, x2, y2]
            elif self.synthetic:
                # Synthetic ground truth
                gt_box = (200 + frame_id % 100, 150, 300 + frame_id % 100, 400)
            
            # Update evaluator
            evaluator.update(
                frame_id=frame_id,
                pred_box=pred_box,
                gt_box=gt_box,
                pred_id=track_id,
                gt_id=1,
                state=state,
                inference_time_ms=frame_time
            )
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {frame_count} frames | FPS: {fps:.2f}")
        
        total_time = time.time() - start_time
        
        # Compute metrics
        metrics = evaluator.compute_metrics()
        
        print(f"\n✅ Completed {variant_name}: {frame_count} frames in {total_time:.2f}s")
        print(f"   Average FPS: {metrics.mean_fps:.2f}")
        print(f"   MOTA: {metrics.mota:.3f} | IDF1: {metrics.idf1:.3f}")
        
        # Save individual results
        output_path = self.output_dir / f"{variant_name}_results.json"
        evaluator.export_to_json(str(output_path))
        print(f"   Results saved to: {output_path}")
        
        return metrics
    
    def run_all(self, variants: List[str] = None):
        """
        Run all specified variants.
        
        Args:
            variants: List of variant names, or None for all
        """
        # Define all available variants
        all_variants = {
            'iou_only': IoUOnlyTracker,
            'shape_only': ShapeOnlyTracker,
            'hsv_depth': HSVDepthTracker,
            'shape_depth': ShapeDepthTracker,
            'full_features': FullFeaturesTracker,
            'deepsort': DeepSORTTracker,
        }
        
        # Filter variants
        if variants is None or 'all' in variants:
            variants_to_run = all_variants
        else:
            variants_to_run = {k: v for k, v in all_variants.items() if k in variants}
        
        if not variants_to_run:
            print("❌ No valid variants specified!")
            return
        
        print(f"\n🎯 Running {len(variants_to_run)} variant(s): {list(variants_to_run.keys())}")
        
        # Run each variant
        for variant_name, tracker_class in variants_to_run.items():
            try:
                metrics = self.run_variant(variant_name, tracker_class)
                self.results[variant_name] = metrics
            except Exception as e:
                print(f"❌ Error running {variant_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print comparison
        self.print_comparison()
        
        # Export comparison
        self.export_comparison()
    
    def print_comparison(self):
        """Print a comparison table of all results."""
        if not self.results:
            print("\n⚠️ No results to compare")
            return
        
        print(f"\n{'='*100}")
        print("BENCHMARK COMPARISON")
        print(f"{'='*100}")
        
        # Header
        print(f"{'Variant':<20} | {'Lock Rate':<10} | {'MOTA':<8} | {'IDF1':<8} | "
              f"{'ID Sw':<6} | {'Mean FPS':<10} | {'P95 Lat':<10}")
        print(f"{'-'*100}")
        
        # Rows
        for variant_name, metrics in self.results.items():
            print(f"{variant_name:<20} | "
                  f"{metrics.target_lock_rate:>9.1%} | "
                  f"{metrics.mota:>7.3f} | "
                  f"{metrics.idf1:>7.3f} | "
                  f"{metrics.id_switches:>5d} | "
                  f"{metrics.mean_fps:>9.2f} | "
                  f"{metrics.p95_latency_ms:>9.1f}")
        
        print(f"{'='*100}\n")
    
    def export_comparison(self):
        """Export comparison to JSON."""
        if not self.results:
            return
        
        comparison_data = {}
        for variant_name, metrics in self.results.items():
            comparison_data[variant_name] = {
                'target_lock_rate': metrics.target_lock_rate,
                'mean_iou': metrics.mean_iou,
                'mota': metrics.mota,
                'motp': metrics.motp,
                'idf1': metrics.idf1,
                'id_switches': metrics.id_switches,
                'fragmentation': metrics.fragmentation,
                'mean_fps': metrics.mean_fps,
                'p95_latency_ms': metrics.p95_latency_ms,
                'total_frames': metrics.total_frames,
            }
        
        output_path = self.output_dir / 'comparison.json'
        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"📊 Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark tracker variants on test videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Video input options
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument(
        '--rgb-video',
        help='Path to RGB video file'
    )
    video_group.add_argument(
        '--split-video',
        help='Path to split-screen video (RGB and Depth side-by-side)'
    )
    video_group.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing (no video needed)'
    )
    
    parser.add_argument(
        '--depth-video',
        help='Path to Depth video file (required if using --rgb-video)'
    )
    parser.add_argument(
        '--rgb-position',
        choices=['left', 'right'],
        default='right',
        help='Position of RGB in split-screen video (default: right)'
    )
    parser.add_argument(
        '--gt',
        help='Path to ground truth annotations JSON file'
    )
    parser.add_argument(
        '--variants',
        nargs='+',
        default=['all'],
        choices=['all', 'iou_only', 'shape_only', 'hsv_depth', 'shape_depth', 'full_features', 'deepsort'],
        help='Variants to run (default: all)'
    )
    parser.add_argument(
        '--output',
        default='./benchmark/results',
        help='Output directory for results (default: ./benchmark/results)'
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.rgb_video and not args.depth_video:
        print("⚠️ Warning: RGB video provided without depth video. Depth-based variants may fail.")
    
    # Run benchmark
    runner = BenchmarkRunner(
        rgb_video_path=args.rgb_video,
        depth_video_path=args.depth_video,
        split_video_path=args.split_video,
        rgb_position=args.rgb_position,
        gt_path=args.gt,
        output_dir=args.output,
        synthetic=args.synthetic
    )
    
    runner.run_all(variants=args.variants)


if __name__ == '__main__':
    main()
