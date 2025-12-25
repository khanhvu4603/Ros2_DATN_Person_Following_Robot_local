#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Runner for comparing different tracking methods.
Runs multiple tracking configurations on test videos and compares results.

Usage:
    python benchmark_runner.py --video test.mp4 --gt annotations.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

from tracking_metrics import TrackingEvaluator, TrackingMetrics


@dataclass
class BenchmarkConfig:
    """Configuration for a tracking method to benchmark."""
    name: str
    description: str
    feature_type: str  # 'mobilenetv2', 'deepsort_cnn', 'hsv_only', 'iou_only'
    use_depth: bool = True
    use_hsv: bool = True
    use_shape: bool = True


class BenchmarkRunner:
    """
    Run benchmarks on different tracking configurations.
    
    Example:
        runner = BenchmarkRunner(video_path='test.mp4', gt_path='annotations.json')
        runner.add_config(BenchmarkConfig(name='full', description='MobileNetV2+HSV+Depth', ...))
        results = runner.run_all()
        runner.print_comparison()
    """
    
    def __init__(
        self,
        video_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        output_dir: str = './benchmark_results'
    ):
        self.video_path = video_path
        self.gt_path = gt_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.configs: List[BenchmarkConfig] = []
        self.results: Dict[str, TrackingMetrics] = {}
        
        # Load ground truth if provided
        self.ground_truth = None
        if gt_path and Path(gt_path).exists():
            with open(gt_path, 'r') as f:
                self.ground_truth = json.load(f)
    
    def add_config(self, config: BenchmarkConfig):
        """Add a tracking configuration to benchmark."""
        self.configs.append(config)
    
    def add_default_configs(self):
        """Add default configurations for comparison."""
        self.configs = [
            BenchmarkConfig(
                name='full_features',
                description='MobileNetV2 + HSV + Depth',
                feature_type='mobilenetv2',
                use_depth=True,
                use_hsv=True,
                use_shape=True
            ),
            BenchmarkConfig(
                name='shape_depth',
                description='MobileNetV2 + Depth (no HSV)',
                feature_type='mobilenetv2',
                use_depth=True,
                use_hsv=False,
                use_shape=True
            ),
            BenchmarkConfig(
                name='shape_only',
                description='MobileNetV2 only',
                feature_type='mobilenetv2',
                use_depth=False,
                use_hsv=False,
                use_shape=True
            ),
            BenchmarkConfig(
                name='hsv_depth',
                description='HSV + Depth (no CNN)',
                feature_type='hsv',
                use_depth=True,
                use_hsv=True,
                use_shape=False
            ),
            BenchmarkConfig(
                name='iou_only',
                description='IoU matching only (no ReID)',
                feature_type='iou',
                use_depth=False,
                use_hsv=False,
                use_shape=False
            ),
        ]
    
    def simulate_tracking(
        self,
        config: BenchmarkConfig,
        video_path: str,
        evaluator: TrackingEvaluator
    ) -> TrackingMetrics:
        """
        Simulate tracking with given configuration.
        
        THIS IS A PLACEHOLDER - Replace with actual tracker integration.
        In real usage, this should use your actual tracker.
        """
        # Placeholder: simulate based on config type
        # In real usage, integrate with your actual tracker here
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Processing {frame_count} frames...")
        
        # Simulated performance based on feature type
        base_times = {
            'mobilenetv2': 80,  # ms
            'deepsort_cnn': 30,
            'hsv': 5,
            'iou': 2
        }
        base_time = base_times.get(config.feature_type, 50)
        if config.use_depth:
            base_time += 3
        if config.use_hsv and config.feature_type != 'hsv':
            base_time += 5
        
        # Simulated accuracy based on features
        base_lock_rate = 0.70
        if config.use_shape:
            base_lock_rate += 0.15
        if config.use_hsv:
            base_lock_rate += 0.05
        if config.use_depth:
            base_lock_rate += 0.05
        
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get ground truth for this frame
            gt_box = None
            gt_id = 1
            if self.ground_truth and frame_id < len(self.ground_truth.get('frames', [])):
                gt_data = self.ground_truth['frames'][frame_id]
                gt_box = tuple(gt_data.get('box', []))
                gt_id = gt_data.get('id', 1)
            else:
                # Default GT for testing
                h, w = frame.shape[:2]
                gt_box = (w//4, h//4, w//2, h*3//4)
            
            # Simulate tracking result
            if np.random.random() < base_lock_rate:
                state = 'LOCKED'
                # Add noise based on features
                noise_level = 20 if not config.use_shape else 10
                if config.use_depth:
                    noise_level -= 3
                noise = np.random.randint(-noise_level, noise_level, 4)
                pred_box = tuple(np.clip(np.array(gt_box) + noise, 0, max(frame.shape[:2])))
                pred_id = 1
            else:
                if np.random.random() < 0.3:
                    state = 'LOST'
                else:
                    state = 'SEARCHING'
                pred_box = None
                pred_id = None
            
            # Simulate timing
            inference_time = base_time + np.random.normal(0, base_time * 0.1)
            
            evaluator.update(
                frame_id=frame_id,
                pred_box=pred_box,
                gt_box=gt_box,
                pred_id=pred_id,
                gt_id=gt_id,
                state=state,
                inference_time_ms=max(1, inference_time)
            )
            
            frame_id += 1
        
        cap.release()
        return evaluator.compute_metrics()
    
    def run_config(self, config: BenchmarkConfig) -> TrackingMetrics:
        """Run benchmark for a single configuration."""
        print(f"\n▶ Running: {config.name} - {config.description}")
        
        evaluator = TrackingEvaluator(iou_threshold=0.5)
        
        if self.video_path:
            metrics = self.simulate_tracking(config, self.video_path, evaluator)
        else:
            # Generate synthetic test data
            metrics = self._run_synthetic(config, evaluator)
        
        self.results[config.name] = metrics
        evaluator.export_to_json(str(self.output_dir / f"{config.name}_results.json"))
        
        return metrics
    
    def _run_synthetic(
        self,
        config: BenchmarkConfig,
        evaluator: TrackingEvaluator,
        n_frames: int = 500
    ) -> TrackingMetrics:
        """Run benchmark with synthetic data (no video needed)."""
        print(f"  Using synthetic data ({n_frames} frames)...")
        
        # Base performance by feature type
        perf = {
            'mobilenetv2': {'time': 80, 'accuracy': 0.90, 'id_stability': 0.95},
            'deepsort_cnn': {'time': 30, 'accuracy': 0.88, 'id_stability': 0.93},
            'hsv': {'time': 5, 'accuracy': 0.70, 'id_stability': 0.80},
            'iou': {'time': 2, 'accuracy': 0.60, 'id_stability': 0.60},
        }
        
        base = perf.get(config.feature_type, perf['iou'])
        
        # Adjust based on features
        accuracy = base['accuracy']
        if config.use_depth:
            accuracy += 0.05
        if config.use_hsv and config.feature_type != 'hsv':
            accuracy += 0.03
        accuracy = min(0.99, accuracy)
        
        time_ms = base['time']
        if config.use_depth:
            time_ms += 3
        if config.use_hsv and config.feature_type != 'hsv':
            time_ms += 5
        
        gt_box = (100, 100, 200, 350)
        current_state = 'SEARCHING'
        frames_in_state = 0
        current_id = 1
        
        for i in range(n_frames):
            # State transitions
            if current_state == 'SEARCHING':
                if frames_in_state > 5 and np.random.random() < accuracy:
                    current_state = 'LOCKED'
                    frames_in_state = 0
            elif current_state == 'LOCKED':
                # Occasional lost
                if np.random.random() < 0.02:
                    current_state = 'LOST'
                    frames_in_state = 0
                # Rare ID switch
                elif np.random.random() < (1 - base['id_stability']) * 0.1:
                    current_id += 1  # ID switch
            elif current_state == 'LOST':
                if frames_in_state > 10:
                    if np.random.random() < 0.5:
                        current_state = 'LOCKED'
                    else:
                        current_state = 'SEARCHING'
                    frames_in_state = 0
            
            frames_in_state += 1
            
            # Generate prediction
            if current_state == 'LOCKED':
                noise = np.random.randint(-15, 15, 4)
                pred_box = tuple(np.array(gt_box) + noise)
                pred_id = current_id
            else:
                pred_box = None
                pred_id = None
            
            evaluator.update(
                frame_id=i,
                pred_box=pred_box,
                gt_box=gt_box,
                pred_id=pred_id,
                gt_id=1,
                state=current_state,
                inference_time_ms=time_ms + np.random.normal(0, time_ms * 0.1)
            )
        
        return evaluator.compute_metrics()
    
    def run_all(self) -> Dict[str, TrackingMetrics]:
        """Run all configured benchmarks."""
        print("\n" + "="*60)
        print("       TRACKING BENCHMARK SUITE")
        print("="*60)
        
        if not self.configs:
            self.add_default_configs()
        
        for config in self.configs:
            self.run_config(config)
        
        return self.results
    
    def print_comparison(self):
        """Print a comparison table of all results."""
        if not self.results:
            print("No results to compare. Run benchmarks first.")
            return
        
        print("\n" + "="*100)
        print("                              BENCHMARK COMPARISON")
        print("="*100)
        
        # Header
        print(f"\n{'Method':<20} {'Lock%':>8} {'IoU':>8} {'MOTA':>8} {'IDF1':>8} "
              f"{'IDSw':>6} {'FPS':>8} {'P95ms':>8}")
        print("-"*100)
        
        for name, m in self.results.items():
            print(f"{name:<20} {m.target_lock_rate*100:>7.1f}% {m.mean_iou:>8.3f} "
                  f"{m.mota*100:>7.1f}% {m.idf1*100:>7.1f}% "
                  f"{m.id_switches:>6} {m.mean_fps:>8.1f} {m.p95_latency_ms:>7.1f}ms")
        
        print("-"*100)
        
        # Find best for each metric
        if len(self.results) > 1:
            print("\n🏆 BEST PERFORMERS:")
            
            best_lock = max(self.results.items(), key=lambda x: x[1].target_lock_rate)
            best_mota = max(self.results.items(), key=lambda x: x[1].mota)
            best_fps = max(self.results.items(), key=lambda x: x[1].mean_fps)
            best_idf1 = max(self.results.items(), key=lambda x: x[1].idf1)
            
            print(f"   Best Lock Rate:  {best_lock[0]} ({best_lock[1].target_lock_rate*100:.1f}%)")
            print(f"   Best MOTA:       {best_mota[0]} ({best_mota[1].mota*100:.1f}%)")
            print(f"   Best IDF1:       {best_idf1[0]} ({best_idf1[1].idf1*100:.1f}%)")
            print(f"   Best FPS:        {best_fps[0]} ({best_fps[1].mean_fps:.1f})")
        
        print("="*100 + "\n")
    
    def export_comparison(self, filepath: str = None):
        """Export comparison to JSON."""
        if filepath is None:
            filepath = str(self.output_dir / "comparison.json")
        
        data = {
            'configs': [
                {
                    'name': c.name,
                    'description': c.description,
                    'feature_type': c.feature_type,
                    'use_depth': c.use_depth,
                    'use_hsv': c.use_hsv,
                    'use_shape': c.use_shape
                }
                for c in self.configs
            ],
            'results': {
                name: {
                    'target_lock_rate': m.target_lock_rate,
                    'mean_iou': m.mean_iou,
                    'mota': m.mota,
                    'motp': m.motp,
                    'idf1': m.idf1,
                    'id_switches': m.id_switches,
                    'fragmentation': m.fragmentation,
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1_score': m.f1_score,
                    'mean_fps': m.mean_fps,
                    'p95_latency_ms': m.p95_latency_ms
                }
                for name, m in self.results.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Comparison exported to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Tracking Benchmark Runner')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--gt', type=str, help='Path to ground truth annotations (JSON)')
    parser.add_argument('--output', type=str, default='./benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data instead of video')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        video_path=args.video if not args.synthetic else None,
        gt_path=args.gt,
        output_dir=args.output
    )
    
    # Add default configurations
    runner.add_default_configs()
    
    # Run all benchmarks
    runner.run_all()
    
    # Print and export comparison
    runner.print_comparison()
    runner.export_comparison()


if __name__ == '__main__':
    main()
