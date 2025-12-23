#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Script - So sánh 3 phiên bản Feature Extraction
Vẽ biểu đồ từ CSV logs của 3 versions.

Usage:
    python3 plot_comparison.py
    
Yêu cầu các file CSV trong cùng thư mục:
    - results_v1_shape.csv
    - results_v2_shape_depth.csv
    - results_v3_full.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Matplotlib settings
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Colors for versions
COLORS = {
    'V1': '#e74c3c',  # Red
    'V2': '#3498db',  # Blue
    'V3': '#2ecc71',  # Green
}

STATE_COLORS = {
    'ENROLLING': '#95a5a6',
    'SEARCHING': '#f39c12',
    'LOCKED': '#2ecc71',
    'LOST': '#e74c3c',
}

def load_csv(filepath):
    """Load CSV và xử lý data."""
    df = pd.read_csv(filepath)
    df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce').fillna(0)
    return df

def compute_metrics(df, version_name):
    """Tính toán metrics từ dataframe."""
    total_frames = len(df)
    
    frames_locked = (df['state'] == 'LOCKED').sum()
    frames_lost = (df['state'] == 'LOST').sum()
    frames_searching = (df['state'] == 'SEARCHING').sum()
    
    # State changes
    state_changes = df['state'] != df['state'].shift()
    
    # Track fragmentations: LOST -> LOCKED
    tf = 0
    prev_state = None
    for state in df['state']:
        if prev_state == 'LOST' and state == 'LOCKED':
            tf += 1
        prev_state = state
    
    # Similarity stats (LOCKED only)
    locked_sim = df[df['state'] == 'LOCKED']['similarity']
    
    return {
        'version': version_name,
        'total_frames': total_frames,
        'frames_locked': frames_locked,
        'frames_lost': frames_lost,
        'frames_searching': frames_searching,
        'locked_rate': frames_locked / total_frames * 100 if total_frames > 0 else 0,
        'lost_rate': (frames_lost + frames_searching) / total_frames * 100 if total_frames > 0 else 0,
        'track_fragmentations': tf,
        'sim_mean': locked_sim.mean() if len(locked_sim) > 0 else 0,
        'sim_std': locked_sim.std() if len(locked_sim) > 0 else 0,
        'sim_min': locked_sim.min() if len(locked_sim) > 0 else 0,
        'sim_max': locked_sim.max() if len(locked_sim) > 0 else 0,
    }

def plot_similarity_over_time(ax, dfs, labels):
    """Plot similarity theo thời gian."""
    for (df, label) in zip(dfs, labels):
        # Filter only tracking frames (not enrollment)
        track_df = df[df['state'].isin(['SEARCHING', 'LOCKED', 'LOST'])]
        ax.plot(track_df['frame_id'], track_df['similarity'], 
                label=label, color=COLORS[label], alpha=0.7, linewidth=0.8)
    
    ax.axhline(y=0.75, color='green', linestyle='--', linewidth=1, label='Accept Threshold (0.75)')
    ax.axhline(y=0.60, color='red', linestyle='--', linewidth=1, label='Reject Threshold (0.60)')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity Score Over Time')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

def plot_state_timeline(ax, dfs, labels):
    """Plot state timeline."""
    state_to_num = {'ENROLLING': 0, 'SEARCHING': 1, 'LOCKED': 2, 'LOST': 3}
    
    for i, (df, label) in enumerate(zip(dfs, labels)):
        track_df = df[df['state'].isin(['SEARCHING', 'LOCKED', 'LOST'])]
        states_num = track_df['state'].map(state_to_num)
        
        # Offset for each version
        ax.scatter(track_df['frame_id'], states_num + i * 0.15, 
                   c=[STATE_COLORS.get(s, 'gray') for s in track_df['state']],
                   s=2, alpha=0.6, label=label)
    
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['SEARCHING', 'LOCKED', 'LOST'])
    ax.set_xlabel('Frame')
    ax.set_title('State Timeline')
    ax.legend(loc='upper right')

def plot_similarity_distribution(ax, dfs, labels):
    """Plot histogram of similarity distribution."""
    for df, label in zip(dfs, labels):
        locked_sim = df[df['state'] == 'LOCKED']['similarity']
        if len(locked_sim) > 0:
            ax.hist(locked_sim, bins=30, alpha=0.5, label=label, color=COLORS[label])
    
    ax.axvline(x=0.75, color='green', linestyle='--', linewidth=2, label='Accept (0.75)')
    ax.axvline(x=0.60, color='red', linestyle='--', linewidth=2, label='Reject (0.60)')
    
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Similarity Distribution (LOCKED state only)')
    ax.legend()

def plot_metrics_comparison(ax, metrics_list):
    """Plot bar chart comparing metrics."""
    versions = [m['version'] for m in metrics_list]
    x = np.arange(len(versions))
    width = 0.25
    
    # Metrics to compare
    locked_rates = [m['locked_rate'] for m in metrics_list]
    lost_rates = [m['lost_rate'] for m in metrics_list]
    
    bars1 = ax.bar(x - width/2, locked_rates, width, label='Locked Rate %', color='#2ecc71')
    bars2 = ax.bar(x + width/2, lost_rates, width, label='Lost Rate %', color='#e74c3c')
    
    ax.set_ylabel('Percentage')
    ax.set_title('State Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

def plot_similarity_stats(ax, metrics_list):
    """Plot similarity statistics comparison."""
    versions = [m['version'] for m in metrics_list]
    means = [m['sim_mean'] for m in metrics_list]
    stds = [m['sim_std'] for m in metrics_list]
    mins = [m['sim_min'] for m in metrics_list]
    maxs = [m['sim_max'] for m in metrics_list]
    
    x = np.arange(len(versions))
    
    ax.bar(x, means, yerr=stds, capsize=5, color=[COLORS[v] for v in versions], alpha=0.8)
    
    # Add min/max markers
    for i, (mi, ma) in enumerate(zip(mins, maxs)):
        ax.plot([i], [mi], 'v', color='red', markersize=8)
        ax.plot([i], [ma], '^', color='green', markersize=8)
    
    ax.axhline(y=0.60, color='red', linestyle='--', linewidth=1, label='Reject Threshold')
    
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity Statistics (Mean ± Std, Min/Max)')
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.legend(['Min', 'Max', 'Reject Threshold'])
    ax.set_ylim(0, 1.1)

def plot_track_fragmentations(ax, metrics_list):
    """Plot track fragmentations comparison."""
    versions = [m['version'] for m in metrics_list]
    fragmentations = [m['track_fragmentations'] for m in metrics_list]
    
    colors = [COLORS[v] for v in versions]
    bars = ax.bar(versions, fragmentations, color=colors, alpha=0.8)
    
    ax.set_ylabel('Count')
    ax.set_title('Track Fragmentations (Lower is Better)')
    
    for bar, val in zip(bars, fragmentations):
        ax.annotate(str(val), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')

def print_summary_table(metrics_list):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)
    
    headers = ['Metric', 'V1 (Shape)', 'V2 (+Depth)', 'V3 (+HSV)', 'Best']
    
    rows = [
        ('Total Frames', [m['total_frames'] for m in metrics_list]),
        ('Locked Rate (%)', [f"{m['locked_rate']:.1f}" for m in metrics_list]),
        ('Lost Rate (%)', [f"{m['lost_rate']:.1f}" for m in metrics_list]),
        ('Track Fragmentations', [m['track_fragmentations'] for m in metrics_list]),
        ('Similarity Mean', [f"{m['sim_mean']:.3f}" for m in metrics_list]),
        ('Similarity Std', [f"{m['sim_std']:.3f}" for m in metrics_list]),
        ('Similarity Min', [f"{m['sim_min']:.3f}" for m in metrics_list]),
        ('Similarity Max', [f"{m['sim_max']:.3f}" for m in metrics_list]),
    ]
    
    # Determine best
    best_choices = {
        'Locked Rate (%)': lambda vals: ['V1', 'V2', 'V3'][np.argmax([float(v) for v in vals])],
        'Lost Rate (%)': lambda vals: ['V1', 'V2', 'V3'][np.argmin([float(v) for v in vals])],
        'Track Fragmentations': lambda vals: ['V1', 'V2', 'V3'][np.argmin([int(v) for v in vals])],
        'Similarity Mean': lambda vals: ['V1', 'V2', 'V3'][np.argmax([float(v) for v in vals])],
        'Similarity Std': lambda vals: ['V1', 'V2', 'V3'][np.argmin([float(v) for v in vals])],
    }
    
    print(f"\n{'Metric':<25} {'V1 (Shape)':<15} {'V2 (+Depth)':<15} {'V3 (+HSV)':<15} {'Best':<10}")
    print("-" * 80)
    
    for name, vals in rows:
        best = best_choices.get(name, lambda x: '-')(vals) if name in best_choices else '-'
        print(f"{name:<25} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15} {best:<10}")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Plot comparison charts')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing CSV files')
    parser.add_argument('--output', type=str, default='comparison_charts.png', help='Output image filename')
    parser.add_argument('--show', action='store_true', help='Show plot')
    args = parser.parse_args()
    
    base_dir = Path(args.dir)
    
    # Load CSVs
    csv_files = {
        'V1': base_dir / 'results_v1_shape.csv',
        'V2': base_dir / 'results_v2_shape_depth.csv',
        'V3': base_dir / 'results_v3_full.csv',
    }
    
    dfs = {}
    metrics_list = []
    
    for version, filepath in csv_files.items():
        if filepath.exists():
            print(f"Loading {filepath}...")
            dfs[version] = load_csv(filepath)
            metrics_list.append(compute_metrics(dfs[version], version))
        else:
            print(f"WARNING: {filepath} not found!")
    
    if len(dfs) == 0:
        print("ERROR: No CSV files found!")
        return
    
    # Print summary table
    print_summary_table(metrics_list)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 3x2 grid
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)
    
    # Prepare data
    df_list = [dfs.get(v) for v in ['V1', 'V2', 'V3'] if v in dfs]
    labels = [v for v in ['V1', 'V2', 'V3'] if v in dfs]
    
    # Plot 1: Similarity over time
    plot_similarity_over_time(ax1, df_list, labels)
    
    # Plot 2: State timeline
    plot_state_timeline(ax2, df_list, labels)
    
    # Plot 3: Similarity distribution
    plot_similarity_distribution(ax3, df_list, labels)
    
    # Plot 4: Metrics comparison (bar chart)
    plot_metrics_comparison(ax4, metrics_list)
    
    # Plot 5: Similarity stats
    plot_similarity_stats(ax5, metrics_list)
    
    # Plot 6: Track fragmentations
    plot_track_fragmentations(ax6, metrics_list)
    
    plt.suptitle('Feature Extraction Comparison: V1 (Shape) vs V2 (+Depth) vs V3 (+HSV)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    output_path = base_dir / args.output
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {output_path}")
    
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
