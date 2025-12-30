#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-Scenario Benchmark Runner

Runs all 6 tracker variants on 3 test scenarios and generates
CVPR-style comparison tables.

Usage:
    python benchmark/run_3scenario_benchmark.py --output benchmark/results/3scenarios/
    
Scenarios:
    - Scene 1: Có người đi qua (Sence1_RGB.mp4)
    - Scene 2: Đi 1 mình (Sence2_RGB.mp4)
    - Scene 3: Vừa đi 1 mình vừa có người (RGB_sence3.mp4)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys
import numpy as np

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


# ==================== SCENARIO CONFIGURATION ====================

SCENARIOS = {
    'scene1': {
        'name': 'Scene 1: Có người đi qua',
        'rgb_video': 'Sence1_RGB.mp4',
        'gt': 'sence1_CoNguoiDiQua_gt.json',
        'description': 'Single target with distractor walking by'
    },
    'scene2': {
        'name': 'Scene 2: Đi 1 mình',
        'rgb_video': 'Sence2_RGB.mp4',
        'gt': 'sence2_Di1MinhKoCoAi_gt.json',
        'description': 'Single target walking alone'
    },
    'scene3': {
        'name': 'Scene 3: Vừa đi 1 mình vừa có người',
        'rgb_video': 'RGB_sence3.mp4',
        'gt': 'sence3_vuadi1minhVuacoNg_gt.json',
        'description': 'Single target with occasional distractors'
    }
}

ALL_VARIANTS = {
    'iou_only': IoUOnlyTracker,
    'shape_only': ShapeOnlyTracker,
    'hsv_depth': HSVDepthTracker,
    'shape_depth': ShapeDepthTracker,
    'full_features': FullFeaturesTracker,
    'deepsort': DeepSORTTracker,
}


class MultiScenarioBenchmark:
    """Run benchmarks on multiple scenarios."""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        variants: List[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select variants
        if variants is None or 'all' in variants:
            self.variants = ALL_VARIANTS
        else:
            self.variants = {k: v for k, v in ALL_VARIANTS.items() if k in variants}
        
        # Results storage: {scenario: {variant: metrics}}
        self.results = {}
    
    def run_scenario(
        self,
        scenario_id: str,
        scenario_config: dict
    ) -> Dict[str, TrackingMetrics]:
        """Run all variants on a single scenario."""
        
        print(f"\n{'='*80}")
        print(f"  {scenario_config['name']}")
        print(f"  {scenario_config['description']}")
        print(f"{'='*80}")
        
        # Paths
        rgb_path = self.data_dir / 'videos' / scenario_config['rgb_video']
        gt_path = self.data_dir / 'annotations' / scenario_config['gt']
        
        # Check files exist
        if not rgb_path.exists():
            print(f"❌ RGB video not found: {rgb_path}")
            return {}
        
        if not gt_path.exists():
            print(f"❌ Ground truth not found: {gt_path}")
            return {}
        
        # Load ground truth
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
        
        # Create scenario output directory
        scenario_output = self.output_dir / scenario_id
        scenario_output.mkdir(parents=True, exist_ok=True)
        
        scenario_results = {}
        
        for variant_name, tracker_class in self.variants.items():
            print(f"\n  Running: {variant_name}")
            
            try:
                metrics = self._run_single_variant(
                    tracker_class=tracker_class,
                    rgb_video_path=str(rgb_path),
                    ground_truth=ground_truth,
                    output_path=scenario_output / f"{variant_name}_results.json"
                )
                scenario_results[variant_name] = metrics
                
                print(f"    ✅ MOTA: {metrics.mota:.3f} | IDF1: {metrics.idf1:.3f} | "
                      f"FPS: {metrics.mean_fps:.1f} | ID Sw: {metrics.id_switches}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Save scenario comparison
        self._save_comparison(scenario_results, scenario_output / 'comparison.json')
        
        return scenario_results
    
    def _run_single_variant(
        self,
        tracker_class,
        rgb_video_path: str,
        ground_truth: dict,
        output_path: Path
    ) -> TrackingMetrics:
        """Run a single variant on a video."""
        
        import cv2
        
        # Initialize tracker
        tracker = tracker_class()
        
        # Initialize evaluator
        evaluator = TrackingEvaluator(iou_threshold=0.5)
        
        # Prepare ground truth lookup
        gt_frames = {f['frame_id']: f for f in ground_truth.get('frames', [])}
        
        # Open video
        cap = cv2.VideoCapture(rgb_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {rgb_video_path}")
        
        frame_id = 0
        start_time = time.time()
        
        while True:
            ret, rgb_frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Run tracker (no depth for RGB-only videos)
            pred_box, state, track_id = tracker.process_frame(
                frame_id, rgb_frame, None
            )
            
            frame_time = (time.time() - frame_start) * 1000  # ms
            
            # Get ground truth
            gt_box = None
            if frame_id in gt_frames:
                box_data = gt_frames[frame_id].get('box')
                if box_data:
                    gt_box = tuple(box_data)
            
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
            
            frame_id += 1
        
        cap.release()
        
        # Compute metrics
        metrics = evaluator.compute_metrics()
        
        # Save individual results
        evaluator.export_to_json(str(output_path))
        
        return metrics
    
    def _save_comparison(self, results: Dict[str, TrackingMetrics], output_path: Path):
        """Save comparison JSON."""
        comparison = {}
        for variant_name, metrics in results.items():
            comparison[variant_name] = {
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
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    def run_all_scenarios(self):
        """Run all scenarios."""
        
        print("\n" + "="*80)
        print("  3-SCENARIO BENCHMARK")
        print("  Variants:", list(self.variants.keys()))
        print("="*80)
        
        for scenario_id, scenario_config in SCENARIOS.items():
            scenario_results = self.run_scenario(scenario_id, scenario_config)
            self.results[scenario_id] = scenario_results
        
        # Generate combined results
        self._generate_combined_results()
        
        # Print summary
        self._print_summary()
    
    def _generate_combined_results(self):
        """Generate combined comparison across all scenarios."""
        
        combined = {}
        
        for variant_name in self.variants.keys():
            # Collect metrics across scenarios
            mota_list = []
            idf1_list = []
            fps_list = []
            id_switches_list = []
            
            for scenario_id in SCENARIOS.keys():
                if scenario_id in self.results and variant_name in self.results[scenario_id]:
                    m = self.results[scenario_id][variant_name]
                    mota_list.append(m.mota)
                    idf1_list.append(m.idf1)
                    fps_list.append(m.mean_fps)
                    id_switches_list.append(m.id_switches)
            
            if mota_list:
                combined[variant_name] = {
                    'mota_mean': float(np.mean(mota_list)),
                    'mota_std': float(np.std(mota_list)),
                    'idf1_mean': float(np.mean(idf1_list)),
                    'idf1_std': float(np.std(idf1_list)),
                    'fps_mean': float(np.mean(fps_list)),
                    'fps_std': float(np.std(fps_list)),
                    'id_switches_total': int(sum(id_switches_list)),
                    'scenarios_completed': len(mota_list),
                    'per_scenario': {
                        scenario_id: {
                            'mota': self.results[scenario_id][variant_name].mota,
                            'idf1': self.results[scenario_id][variant_name].idf1,
                            'fps': self.results[scenario_id][variant_name].mean_fps,
                            'id_switches': self.results[scenario_id][variant_name].id_switches,
                        }
                        for scenario_id in SCENARIOS.keys()
                        if scenario_id in self.results and variant_name in self.results[scenario_id]
                    }
                }
        
        # Save combined results
        with open(self.output_dir / 'comparison.json', 'w') as f:
            json.dump(combined, f, indent=2)
        
        print(f"\n📊 Combined results saved to: {self.output_dir / 'comparison.json'}")
    
    def _print_summary(self):
        """Print summary table."""
        
        print("\n" + "="*100)
        print("  COMBINED BENCHMARK RESULTS (Average across 3 scenarios)")
        print("="*100)
        
        # Header
        print(f"{'Variant':<20} | {'MOTA':<12} | {'IDF1':<12} | {'FPS':<12} | {'ID Sw':<8}")
        print("-"*100)
        
        # Load combined results
        combined_path = self.output_dir / 'comparison.json'
        if combined_path.exists():
            with open(combined_path, 'r') as f:
                combined = json.load(f)
            
            for variant_name in self.variants.keys():
                if variant_name in combined:
                    c = combined[variant_name]
                    print(f"{variant_name:<20} | "
                          f"{c['mota_mean']:>5.1%}±{c['mota_std']:.1%}  | "
                          f"{c['idf1_mean']:>5.1%}±{c['idf1_std']:.1%}  | "
                          f"{c['fps_mean']:>5.1f}±{c['fps_std']:.1f}   | "
                          f"{c['id_switches_total']:>6d}")
        
        print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmark on 3 scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--data-dir',
        default='./benchmark/data',
        help='Path to data directory containing videos/ and annotations/'
    )
    parser.add_argument(
        '--output',
        default='./benchmark/results/3scenarios',
        help='Output directory for results'
    )
    parser.add_argument(
        '--variants',
        nargs='+',
        default=['all'],
        choices=['all', 'iou_only', 'shape_only', 'hsv_depth', 'shape_depth', 'full_features', 'deepsort'],
        help='Variants to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = MultiScenarioBenchmark(
        data_dir=args.data_dir,
        output_dir=args.output,
        variants=args.variants
    )
    
    benchmark.run_all_scenarios()


if __name__ == '__main__':
    main()
