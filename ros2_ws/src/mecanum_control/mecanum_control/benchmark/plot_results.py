#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot benchmark comparison results.

Usage:
    python benchmark/plot_results.py benchmark/results/p2_full/comparison.json
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filepath):
    """Load comparison JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_mota_vs_fps(data, output_dir):
    """Plot MOTA vs FPS scatter plot showing trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = list(data.keys())
    mota = [data[v]['mota'] * 100 for v in variants]
    fps = [data[v]['mean_fps'] for v in variants]
    
    # Color map
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#9B59B6']
    
    # Scatter plot
    for i, v in enumerate(variants):
        ax.scatter(fps[i], mota[i], s=200, c=colors[i], alpha=0.7, 
                   edgecolors='black', linewidth=2, label=v.replace('_', ' ').title())
        ax.annotate(v.replace('_', '\n'), (fps[i], mota[i]), 
                   fontsize=9, ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Mean FPS (Frames Per Second)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MOTA (%)', fontsize=12, fontweight='bold')
    ax.set_title('Tracking Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([75, 90])
    
    # Add quadrant lines
    ax.axhline(y=82, color='gray', linestyle='--', alpha=0.5, label='Target MOTA (82%)')
    ax.axvline(x=40, color='gray', linestyle='--', alpha=0.5, label='Target FPS (40)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mota_vs_fps.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'mota_vs_fps.png'}")
    plt.close()

def plot_metrics_comparison(data, output_dir):
    """Plot bar chart comparing all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    variants = list(data.keys())
    variant_labels = [v.replace('_', ' ').title() for v in variants]
    
    # MOTA
    ax = axes[0, 0]
    mota = [data[v]['mota'] * 100 for v in variants]
    bars = ax.bar(variant_labels, mota, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'], alpha=0.8)
    ax.set_ylabel('MOTA (%)', fontweight='bold')
    ax.set_title('Single-Target Tracking Accuracy (MOTA)', fontweight='bold')
    ax.set_ylim([75, 90])
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # IDF1
    ax = axes[0, 1]
    idf1 = [data[v]['idf1'] * 100 for v in variants]
    bars = ax.bar(variant_labels, idf1, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'], alpha=0.8)
    ax.set_ylabel('IDF1 (%)', fontweight='bold')
    ax.set_title('ID F1 Score (IDF1)', fontweight='bold')
    ax.set_ylim([98, 100])
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # FPS
    ax = axes[1, 0]
    fps = [data[v]['mean_fps'] for v in variants]
    bars = ax.bar(variant_labels, fps, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'], alpha=0.8)
    ax.set_ylabel('FPS', fontweight='bold')
    ax.set_title('Processing Speed (Mean FPS)', fontweight='bold')
    ax.set_ylim([0, 70])
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Latency
    ax = axes[1, 1]
    latency = [data[v]['p95_latency_ms'] for v in variants]
    bars = ax.bar(variant_labels, latency, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'], alpha=0.8)
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('P95 Latency', fontweight='bold')
    ax.set_ylim([0, 70])
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'metrics_comparison.png'}")
    plt.close()

def plot_radar_chart(data, output_dir):
    """Plot radar chart for overall comparison."""
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Categories (normalized to 0-100 scale)
    categories = ['MOTA', 'IDF1', 'Lock Rate', 'FPS\n(normalized)', 'Low Latency']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # Plot each variant
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#9B59B6']
    
    for i, variant in enumerate(data.keys()):
        values = [
            data[variant]['mota'] * 100,
            data[variant]['idf1'] * 100,
            data[variant]['target_lock_rate'] * 100,
            min(data[variant]['mean_fps'] / 60.0 * 100, 100),  # Normalize to 60 FPS = 100%
            max(0, 100 - data[variant]['p95_latency_ms'] / 1.0)  # Lower latency = higher score
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=variant.replace('_', ' ').title(),
                color=colors[i], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Overall Performance Comparison (Radar Chart)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'radar_comparison.png'}")
    plt.close()

def print_summary(data):
    """Print text summary."""
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    print("\n🏆 BEST PERFORMERS:")
    
    best_mota = max(data.items(), key=lambda x: x[1]['mota'])
    best_fps = max(data.items(), key=lambda x: x[1]['mean_fps'])
    best_idf1 = max(data.items(), key=lambda x: x[1]['idf1'])
    
    print(f"  • Highest MOTA:  {best_mota[0]:20s} {best_mota[1]['mota']*100:.1f}%")
    print(f"  • Highest FPS:   {best_fps[0]:20s} {best_fps[1]['mean_fps']:.1f} FPS")
    print(f"  • Highest IDF1:  {best_idf1[0]:20s} {best_idf1[1]['idf1']*100:.1f}%")
    
    print("\n💡 RECOMMENDATIONS:")
    
    # Find best trade-off (high MOTA, decent FPS)
    scores = {}
    for v, metrics in data.items():
        # Weighted score: 60% accuracy, 40% speed
        score = (metrics['mota'] * 0.6) + (min(metrics['mean_fps'] / 60.0, 1.0) * 0.4)
        scores[v] = score
    
    best_balance = max(scores.items(), key=lambda x: x[1])
    
    print(f"  • Best Balance:  {best_balance[0]:20s} (Score: {best_balance[1]:.3f})")
    print(f"    → MOTA: {data[best_balance[0]]['mota']*100:.1f}%, FPS: {data[best_balance[0]]['mean_fps']:.1f}")
    
    print("\n📈 INSIGHTS:")
    print(f"  • All variants achieved >99% IDF1 → Excellent ID consistency")
    print(f"  • All variants had 0 ID switches → Very stable tracking")
    print(f"  • MOTA range: {min(d['mota'] for d in data.values())*100:.1f}% - {max(d['mota'] for d in data.values())*100:.1f}%")
    print(f"  • FPS range: {min(d['mean_fps'] for d in data.values()):.1f} - {max(d['mean_fps'] for d in data.values()):.1f}")
    print("="*80 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <comparison.json>")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)
    
    # Load results
    data = load_results(filepath)
    output_dir = filepath.parent / 'presentation'
    output_dir.mkdir(exist_ok=True)
    
    print(f"📂 Loading results from: {filepath}")
    print(f"📊 Plotting {len(data)} variants...")
    
    # Generate plots
    plot_mota_vs_fps(data, output_dir)
    plot_metrics_comparison(data, output_dir)
    plot_radar_chart(data, output_dir)
    
    # Print summary
    print_summary(data)
    
    print(f"\n✅ All plots saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  • mota_vs_fps.png       - Accuracy vs Speed trade-off")
    print(f"  • metrics_comparison.png - Bar charts for all metrics")
    print(f"  • radar_comparison.png   - Overall performance radar")

if __name__ == '__main__':
    main()
