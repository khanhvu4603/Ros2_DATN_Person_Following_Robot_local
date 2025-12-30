#!/usr/bin/env python3
"""Generate combined CVPR comparison table from all scenario results."""

import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path('/home/khanhvq/backup_16_12_2025/ros2_ws/src/mecanum_control/mecanum_control/benchmark/results/3scenarios')

SCENARIOS = ['scene1', 'scene2', 'scene3']
VARIANTS = ['iou_only', 'shape_only', 'hsv_depth', 'shape_depth', 'full_features']


def main():
    results = {}
    
    for scenario in SCENARIOS:
        scenario_dir = RESULTS_DIR / scenario
        comp_file = scenario_dir / 'comparison.json'
        
        # Start with comparison.json if it exists
        if comp_file.exists():
            print(f"Reading {scenario} from comparison.json")
            with open(comp_file, 'r') as f:
                results[scenario] = json.load(f)
        else:
            results[scenario] = {}
        
        # ALSO check individual result files for missing variants
        for variant in VARIANTS:
            if variant not in results[scenario]:
                result_file = scenario_dir / f'{variant}_results.json'
                if result_file.exists():
                    print(f"  Reading {variant} from individual file for {scenario}")
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        # Summary metrics are under 'metrics' key
                        if 'metrics' in data:
                            results[scenario][variant] = data['metrics']
                        else:
                            # Direct format fallback
                            results[scenario][variant] = {
                                'mota': data.get('mota', 0),
                                'idf1': data.get('idf1', 0),
                                'mean_fps': data.get('mean_fps', 0),
                                'id_switches': data.get('id_switches', 0),
                                'target_lock_rate': data.get('target_lock_rate', 0),
                                'motp': data.get('motp', 0),
                            }
    
    # Generate combined results
    combined = {}
    
    for variant in VARIANTS:
        mota_list = []
        idf1_list = []
        fps_list = []
        id_switches_list = []
        lock_rate_list = []
        
        per_scenario = {}
        
        for scenario in SCENARIOS:
            if scenario in results and variant in results[scenario]:
                m = results[scenario][variant]
                mota_val = m.get('mota', 0)
                idf1_val = m.get('idf1', 0)
                fps_val = m.get('mean_fps', 0)
                id_sw_val = m.get('id_switches', 0)
                lock_val = m.get('target_lock_rate', 0)
                
                mota_list.append(mota_val)
                idf1_list.append(idf1_val)
                fps_list.append(fps_val)
                id_switches_list.append(id_sw_val)
                lock_rate_list.append(lock_val)
                
                per_scenario[scenario] = {
                    'mota': mota_val,
                    'idf1': idf1_val,
                    'fps': fps_val,
                    'id_switches': id_sw_val,
                    'lock_rate': lock_val
                }
        
        if mota_list:
            combined[variant] = {
                'mota_mean': float(np.mean(mota_list)),
                'mota_std': float(np.std(mota_list)),
                'idf1_mean': float(np.mean(idf1_list)),
                'idf1_std': float(np.std(idf1_list)),
                'fps_mean': float(np.mean(fps_list)),
                'fps_std': float(np.std(fps_list)),
                'lock_rate_mean': float(np.mean(lock_rate_list)),
                'id_switches_total': int(sum(id_switches_list)),
                'scenarios_completed': len(mota_list),
                'per_scenario': per_scenario
            }
    
    # Save combined
    with open(RESULTS_DIR / 'comparison_all.json', 'w') as f:
        json.dump(combined, f, indent=2)
    
    # Print table
    print("\n" + "="*110)
    print("  COMBINED BENCHMARK RESULTS (Average across 3 scenarios)")
    print("="*110)
    print(f"{'Variant':<18} | {'Lock Rate':<10} | {'MOTA':<14} | {'IDF1':<14} | {'FPS':<12} | {'ID Sw':<6}")
    print("-"*110)
    
    for variant in VARIANTS:
        if variant in combined:
            c = combined[variant]
            print(f"{variant:<18} | "
                  f"{c['lock_rate_mean']:>8.1%} | "
                  f"{c['mota_mean']:>5.1%}±{c['mota_std']:>4.1%} | "
                  f"{c['idf1_mean']:>5.1%}±{c['idf1_std']:>4.1%} | "
                  f"{c['fps_mean']:>5.1f}±{c['fps_std']:>4.1f} | "
                  f"{c['id_switches_total']:>5d}")
    
    print("="*110)
    
    # Print per-scenario breakdown
    print("\n\n" + "="*110)
    print("  PER-SCENARIO BREAKDOWN")
    print("="*110)
    
    for scenario in SCENARIOS:
        print(f"\n### {scenario.upper()}")
        print(f"{'Variant':<18} | {'Lock Rate':<10} | {'MOTA':<10} | {'IDF1':<10} | {'FPS':<10}")
        print("-"*70)
        
        if scenario in results:
            for variant in VARIANTS:
                if variant in results[scenario]:
                    m = results[scenario][variant]
                    print(f"{variant:<18} | "
                          f"{m.get('target_lock_rate', 0):>8.1%} | "
                          f"{m.get('mota', 0):>8.3f} | "
                          f"{m.get('idf1', 0):>8.3f} | "
                          f"{m.get('mean_fps', 0):>8.1f}")
    
    print(f"\n✅ Combined results saved to: {RESULTS_DIR / 'comparison_all.json'}")


if __name__ == '__main__':
    main()
