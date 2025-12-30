#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR-Style Plots for 3-Scenario Benchmark Results.

Generates publication-ready figures from comparison_all.json format.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Use non-GUI backend
matplotlib.use('Agg')

# CVPR color palette (colorblind-safe)
COLORS = {
    'full_features': '#D55E00',    # Vermillion
    'shape_depth': '#0072B2',      # Blue
    'shape_only': '#009E73',       # Bluish green
    'hsv_depth': '#CC79A7',        # Reddish purple
    'iou_only': '#E69F00',         # Orange
    'deepsort': '#56B4E9',         # Sky blue
}

VARIANT_NAMES = {
    'full_features': 'Full Features',
    'shape_depth': 'Shape+Depth',
    'shape_only': 'Shape Only',
    'hsv_depth': 'HSV+Depth',
    'iou_only': 'IoU Only',
    'deepsort': 'DeepSORT',
}

SCENARIOS = ['scene1', 'scene2', 'scene3']
SCENARIO_NAMES = {
    'scene1': 'Scene 1: Distractor',
    'scene2': 'Scene 2: Solo Target',
    'scene3': 'Scene 3: Mixed'
}


def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_speed_accuracy_tradeoff(data, output_dir):
    """Speed-Accuracy scatter plot (CVPR Fig.1 style)."""
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    for variant, metrics in data.items():
        mota = metrics['mota_mean'] * 100
        fps = metrics['fps_mean']
        mota_std = metrics.get('mota_std', 0) * 100
        fps_std = metrics.get('fps_std', 0)
        
        ax.errorbar(fps, mota, 
                   xerr=fps_std, yerr=mota_std,
                   fmt='o', color=COLORS.get(variant, '#888888'),
                   label=VARIANT_NAMES.get(variant, variant),
                   markersize=12, capsize=4, linewidth=2, markeredgewidth=2)
    
    ax.set_xlabel('FPS ↑', fontsize=14, fontweight='bold')
    ax.set_ylabel('MOTA (%) ↑', fontsize=14, fontweight='bold')
    ax.set_title('Speed-Accuracy Trade-off', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 80)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'speed_accuracy_tradeoff.pdf', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ✅ speed_accuracy_tradeoff.png/pdf")


def plot_per_scenario_bars(data, output_dir):
    """Per-scenario MOTA bar chart."""
    
    variants = list(data.keys())
    x = np.arange(len(SCENARIOS))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    for i, variant in enumerate(variants):
        motas = []
        for scenario in SCENARIOS:
            if scenario in data[variant].get('per_scenario', {}):
                motas.append(data[variant]['per_scenario'][scenario]['mota'] * 100)
            else:
                motas.append(0)
        
        offset = (i - len(variants)/2 + 0.5) * width
        bars = ax.bar(x + offset, motas, width, 
                      label=VARIANT_NAMES.get(variant, variant),
                      color=COLORS.get(variant, '#888888'),
                      edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Scenario', fontsize=14, fontweight='bold')
    ax.set_ylabel('MOTA (%)', fontsize=14, fontweight='bold')
    ax.set_title('MOTA per Scenario', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_NAMES[s] for s in SCENARIOS], fontsize=11)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mota_per_scenario.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'mota_per_scenario.pdf', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ✅ mota_per_scenario.png/pdf")


def plot_combined_metrics(data, output_dir):
    """Combined metrics bar chart (MOTA, IDF1, Lock Rate)."""
    
    variants = list(data.keys())
    metrics_to_plot = ['mota_mean', 'idf1_mean', 'lock_rate_mean']
    metric_labels = ['MOTA', 'IDF1', 'Lock Rate']
    
    x = np.arange(len(variants))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    for i, (metric_key, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [data[v].get(metric_key, 0) * 100 for v in variants]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric_label, alpha=0.9)
    
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Metrics Comparison (Average across 3 Scenarios)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_NAMES.get(v, v) for v in variants], fontsize=11, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'metrics_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ✅ metrics_comparison.png/pdf")


def plot_fps_comparison(data, output_dir):
    """FPS bar chart with error bars."""
    
    variants = list(data.keys())
    fps_means = [data[v]['fps_mean'] for v in variants]
    fps_stds = [data[v].get('fps_std', 0) for v in variants]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    x = np.arange(len(variants))
    colors = [COLORS.get(v, '#888888') for v in variants]
    
    bars = ax.bar(x, fps_means, yerr=fps_stds, 
                  color=colors, edgecolor='white', linewidth=1,
                  capsize=5, error_kw={'linewidth': 2})
    
    ax.set_ylabel('FPS', fontsize=14, fontweight='bold')
    ax.set_title('Processing Speed (FPS)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_NAMES.get(v, v) for v in variants], fontsize=11, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, fps_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + fps_stds[i] + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fps_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fps_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fps_comparison.png/pdf")


def generate_summary_table(data, output_dir):
    """Generate CVPR-style black/white summary table."""
    
    variants = list(data.keys())
    
    # Prepare table data
    headers = ['Method', 'MOTA↑', 'IDF1↑', 'Lock%↑', 'ID Sw↓', 'FPS↑', 'Lat(ms)↓']
    rows = []
    
    # Collect values for finding best
    mota_vals = [data[v]['mota_mean'] for v in variants]
    idf1_vals = [data[v]['idf1_mean'] for v in variants]
    lock_vals = [data[v]['lock_rate_mean'] for v in variants]
    fps_vals = [data[v]['fps_mean'] for v in variants]
    
    best_mota_idx = np.argmax(mota_vals)
    best_idf1_idx = np.argmax(idf1_vals)
    best_lock_idx = np.argmax(lock_vals)
    best_fps_idx = np.argmax(fps_vals)
    
    for v in variants:
        d = data[v]
        # Calculate latency from fps (1000/fps = latency in ms)
        latency = 1000 / d['fps_mean'] if d['fps_mean'] > 0 else 0
        
        row = [
            VARIANT_NAMES.get(v, v),
            f"{d['mota_mean']*100:.1f}",
            f"{d['idf1_mean']*100:.1f}",
            f"{d['lock_rate_mean']*100:.1f}",
            str(d['id_switches_total']),
            f"{d['fps_mean']:.1f}",
            f"{latency:.1f}"
        ]
        rows.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers,
                     loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style: CVPR Black/White with gray header
    # Header row - gray background
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#D9D9D9')  # Light gray
        cell.set_text_props(fontweight='bold', color='black')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
    
    # Data rows - white background, black borders
    for row_idx in range(1, len(variants) + 1):
        for col_idx in range(len(headers)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor('white')
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
            cell.set_text_props(color='black')
    
    # Highlight best values in yellow
    # Best MOTA
    table[(best_mota_idx + 1, 1)].set_facecolor('#FFFF99')  # Light yellow
    table[(best_mota_idx + 1, 1)].set_text_props(fontweight='bold')
    
    # Best IDF1
    table[(best_idf1_idx + 1, 2)].set_facecolor('#FFFF99')
    table[(best_idf1_idx + 1, 2)].set_text_props(fontweight='bold')
    
    # Best Lock Rate
    table[(best_lock_idx + 1, 3)].set_facecolor('#FFFF99')
    table[(best_lock_idx + 1, 3)].set_text_props(fontweight='bold')
    
    # Best FPS
    table[(best_fps_idx + 1, 5)].set_facecolor('#FFFF99')
    table[(best_fps_idx + 1, 5)].set_text_props(fontweight='bold')
    
    ax.set_title('Table 1: Quantitative Comparison of Tracking Variants', 
                 fontsize=14, fontweight='bold', pad=20, color='black')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'summary_table.pdf', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✅ summary_table.png/pdf")


def main():
    results_file = Path('/home/khanhvq/backup_16_12_2025/ros2_ws/src/mecanum_control/mecanum_control/benchmark/results/3scenarios/comparison_all.json')
    output_dir = results_file.parent / 'cvpr'
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("📊 GENERATING CVPR-STYLE FIGURES FOR 3-SCENARIO BENCHMARK")
    print("="*70)
    print(f"\n📂 Input:  {results_file}")
    print(f"📂 Output: {output_dir}\n")
    
    data = load_results(results_file)
    
    print("Generating figures...")
    plot_speed_accuracy_tradeoff(data, output_dir)
    plot_per_scenario_bars(data, output_dir)
    plot_combined_metrics(data, output_dir)
    plot_fps_comparison(data, output_dir)
    generate_summary_table(data, output_dir)
    
    print(f"\n✅ All figures saved to: {output_dir}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
