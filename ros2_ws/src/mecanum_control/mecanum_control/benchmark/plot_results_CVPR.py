#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-Ready Plots for CVPR/ICCV Style Papers.

Generates camera-ready figures following CVPR guidelines:
- High DPI (600 for publication)
- LaTeX-compatible fonts
- Colorblind-safe palette
- Minimal design with grid
- Statistical significance markers

Usage:
    python benchmark/plot_results_CVPR.py benchmark/results/p2_full/comparison.json
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# CVPR-style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'text.usetex': False,  # Set to True if LaTeX is installed
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
})

# Colorblind-safe palette (Wong 2011)
COLORS = {
    'full_features': '#D55E00',    # Vermillion
    'shape_depth': '#0072B2',      # Blue
    'shape_only': '#009E73',       # Bluish green
    'hsv_depth': '#CC79A7',        # Reddish purple
    'iou_only': '#E69F00',         # Orange
    'deepsort': '#56B4E9',         # Sky blue
}

MARKERS = {
    'full_features': 'o',
    'shape_depth': 's',
    'shape_only': '^',
    'hsv_depth': 'D',
    'iou_only': 'v',
    'deepsort': 'P',  # Plus marker
}

def load_results(filepath):
    """Load comparison JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_accuracy_speed_tradeoff(data, output_dir):
    """
    Speed-Accuracy trade-off plot (CVPR standard).
    Figure 1 in most tracking papers.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))  # CVPR column width
    
    variants = list(data.keys())
    
    # Plot each variant
    for v in variants:
        mota = data[v]['mota'] * 100
        fps = data[v]['mean_fps']
        
        ax.scatter(fps, mota, s=120, marker=MARKERS[v], 
                  c=COLORS[v], alpha=0.85, 
                  edgecolors='black', linewidth=1.2, 
                  label=v.replace('_', ' ').title(), zorder=3)
    
    # Pareto frontier (optional: connect best performers)
    pareto_variants = ['iou_only', 'hsv_depth', 'shape_depth', 'full_features']
    pareto_fps = [data[v]['mean_fps'] for v in pareto_variants if v in data]
    pareto_mota = [data[v]['mota'] * 100 for v in pareto_variants if v in data]
    
    if len(pareto_fps) > 0:
        sorted_idx = np.argsort(pareto_fps)
        ax.plot(np.array(pareto_fps)[sorted_idx], 
               np.array(pareto_mota)[sorted_idx],
               'k--', alpha=0.3, linewidth=1, zorder=1)
    
    ax.set_xlabel('Speed (FPS)', fontweight='bold')
    ax.set_ylabel('Accuracy (MOTA %)', fontweight='bold')
    ax.set_title('(a) Speed-Accuracy Trade-off', loc='left', fontweight='bold')
    
    # Set limits with margin
    fps_vals = [data[v]['mean_fps'] for v in variants]
    mota_vals = [data[v]['mota'] * 100 for v in variants]
    ax.set_xlim([min(fps_vals) - 5, max(fps_vals) + 5])
    ax.set_ylim([min(mota_vals) - 2, max(mota_vals) + 2])
    
    ax.legend(loc='lower right', frameon=True, fancybox=False, 
             shadow=False, ncol=1, handletextpad=0.5, columnspacing=1.0)
    
    plt.tight_layout(pad=0.3)
    plt.savefig(output_dir / 'cvpr_speed_accuracy.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'cvpr_speed_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'cvpr_speed_accuracy.pdf'}")
    plt.close()

def plot_metrics_table_visual(data, output_dir):
    """
    Visual metrics table (common in CVPR papers).
    Figure 2: Bar chart with statistical comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))  # Two-column layout
    
    variants = list(data.keys())
    variant_labels = [v.replace('_', '\n').title() for v in variants]
    x_pos = np.arange(len(variants))
    width = 0.6
    
    # (a) MOTA Comparison
    ax = axes[0]
    mota = [data[v]['mota'] * 100 for v in variants]
    bars = ax.bar(x_pos, mota, width, color=[COLORS[v] for v in variants],
                  alpha=0.85, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('MOTA (%)', fontweight='bold')
    ax.set_title('(a) Single-Target Tracking Accuracy', loc='left', fontweight='bold', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variant_labels, fontsize=8)
    ax.set_ylim([75, 90])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mota)):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Mark best performer
    best_idx = np.argmax(mota)
    ax.text(best_idx, mota[best_idx] - 2, '★', ha='center', fontsize=14, color='gold')
    
    # (b) FPS Comparison
    ax = axes[1]
    fps = [data[v]['mean_fps'] for v in variants]
    bars = ax.bar(x_pos, fps, width, color=[COLORS[v] for v in variants],
                  alpha=0.85, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Speed (FPS)', fontweight='bold')
    ax.set_title('(b) Processing Speed', loc='left', fontweight='bold', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variant_labels, fontsize=8)
    ax.set_ylim([0, 70])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, fps)):
        ax.text(bar.get_x() + bar.get_width()/2., val + 1.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Mark best performer
    best_idx = np.argmax(fps)
    ax.text(best_idx, fps[best_idx] - 5, '★', ha='center', fontsize=14, color='gold')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(output_dir / 'cvpr_metrics_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'cvpr_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'cvpr_metrics_comparison.pdf'}")
    plt.close()

def plot_detailed_results_table(data, output_dir):
    """
    Detailed results table (Table 1 style in papers).
    Single comprehensive figure with all metrics.
    """
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    variants = list(data.keys())
    
    # Table headers
    headers = ['Method', 'MOTA↑', 'IDF1↑', 'Lock%↑', 'ID Sw↓', 'FPS↑', 'Lat(ms)↓']
    
    # Table data
    table_data = []
    for v in variants:
        row = [
            v.replace('_', ' ').title(),
            f"{data[v]['mota']*100:.1f}",
            f"{data[v]['idf1']*100:.1f}",
            f"{data[v]['target_lock_rate']*100:.1f}",
            f"{data[v]['id_switches']}",
            f"{data[v]['mean_fps']:.1f}",
            f"{data[v]['p95_latency_ms']:.1f}",
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#E0E0E0')
        cell.set_text_props(weight='bold')
    
    # Highlight best values
    metrics_cols = {'MOTA↑': 1, 'IDF1↑': 2, 'Lock%↑': 3, 'FPS↑': 5}
    
    for metric, col_idx in metrics_cols.items():
        values = [float(table_data[i][col_idx]) for i in range(len(variants))]
        best_idx = np.argmax(values)
        cell = table[(best_idx + 1, col_idx)]
        cell.set_facecolor('#FFFFCC')  # Light yellow
        cell.set_text_props(weight='bold')
    
    # Color rows alternately
    for i in range(len(variants)):
        if i % 2 == 0:
            for j in range(len(headers)):
                if j > 0:  # Skip method name column
                    table[(i + 1, j)].set_facecolor('#F5F5F5')
    
    plt.title('Table 1: Quantitative Comparison of Tracking Variants', 
             fontweight='bold', fontsize=11, pad=20)
    
    plt.savefig(output_dir / 'cvpr_results_table.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'cvpr_results_table.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'cvpr_results_table.pdf'}")
    plt.close()

def plot_efficiency_analysis(data, output_dir):
    """
    Efficiency analysis: MOTA per millisecond (similar to FLOPs analysis).
    Shows which method is most efficient.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    variants = list(data.keys())
    
    # Calculate efficiency: MOTA / latency
    efficiency = []
    for v in variants:
        eff = (data[v]['mota'] * 100) / (data[v]['p95_latency_ms'] + 1e-6)
        efficiency.append(eff)
    
    # Sort by efficiency
    sorted_idx = np.argsort(efficiency)[::-1]
    sorted_variants = [variants[i] for i in sorted_idx]
    sorted_eff = [efficiency[i] for i in sorted_idx]
    
    # Horizontal bar chart
    y_pos = np.arange(len(sorted_variants))
    bars = ax.barh(y_pos, sorted_eff, 
                   color=[COLORS[v] for v in sorted_variants],
                   alpha=0.85, edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([v.replace('_', ' ').title() for v in sorted_variants], fontsize=9)
    ax.set_xlabel('Efficiency (MOTA % / ms)', fontweight='bold')
    ax.set_title('(c) Computational Efficiency', loc='left', fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_eff)):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2.,
                f'{val:.2f}', ha='left', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout(pad=0.3)
    plt.savefig(output_dir / 'cvpr_efficiency.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'cvpr_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'cvpr_efficiency.pdf'}")
    plt.close()

def generate_latex_table(data, output_dir):
    """Generate LaTeX table code for direct copy-paste into paper."""
    variants = list(data.keys())
    
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Quantitative comparison of tracking variants on test dataset.}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\begin{tabular}{l|ccc|cc}\n"
    latex += "\\hline\n"
    latex += "Method & MOTA$\\uparrow$ & IDF1$\\uparrow$ & Lock\\%$\\uparrow$ & FPS$\\uparrow$ & Lat(ms)$\\downarrow$ \\\\\n"
    latex += "\\hline\n"
    
    # Find best for each metric
    best_mota = max(data.values(), key=lambda x: x['mota'])['mota']
    best_idf1 = max(data.values(), key=lambda x: x['idf1'])['idf1']
    best_fps = max(data.values(), key=lambda x: x['mean_fps'])['mean_fps']
    
    for v in variants:
        name = v.replace('_', ' ').title()
        mota = data[v]['mota'] * 100
        idf1 = data[v]['idf1'] * 100
        lock = data[v]['target_lock_rate'] * 100
        fps = data[v]['mean_fps']
        lat = data[v]['p95_latency_ms']
        
        # Bold best values
        mota_str = f"\\textbf{{{mota:.1f}}}" if abs(data[v]['mota'] - best_mota) < 0.001 else f"{mota:.1f}"
        idf1_str = f"\\textbf{{{idf1:.1f}}}" if abs(data[v]['idf1'] - best_idf1) < 0.001 else f"{idf1:.1f}"
        fps_str = f"\\textbf{{{fps:.1f}}}" if abs(data[v]['mean_fps'] - best_fps) < 0.1 else f"{fps:.1f}"
        
        latex += f"{name} & {mota_str} & {idf1_str} & {lock:.1f} & {fps_str} & {lat:.1f} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    # Save to file
    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved: {output_dir / 'results_table.tex'}")
    print("\n📄 LaTeX table code:\n")
    print(latex)

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results_CVPR.py <comparison.json>")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)
    
    # Load results
    data = load_results(filepath)
    output_dir = filepath.parent / 'cvpr'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("📊 GENERATING CVPR-STYLE PUBLICATION-READY FIGURES")
    print("=" * 80)
    print(f"\n📂 Input:  {filepath}")
    print(f"📂 Output: {output_dir}/")
    print(f"\n🎨 Standard: CVPR/ICCV Camera-Ready")
    print(f"🎨 DPI: 600 (PDF) / 300 (PNG)")
    print(f"🎨 Colorblind-safe palette (Wong 2011)")
    print("\nGenerating figures...\n")
    
    # Generate all plots
    plot_accuracy_speed_tradeoff(data, output_dir)
    plot_metrics_table_visual(data, output_dir)
    plot_detailed_results_table(data, output_dir)
    plot_efficiency_analysis(data, output_dir)
    generate_latex_table(data, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ALL CVPR-STYLE FIGURES GENERATED")
    print("=" * 80)
    print("\n📁 Generated files:")
    print("  📊 cvpr_speed_accuracy.pdf        - Figure 1: Speed-Accuracy trade-off")
    print("  📊 cvpr_metrics_comparison.pdf    - Figure 2: Metrics comparison")
    print("  📊 cvpr_results_table.pdf         - Table 1: Detailed results")
    print("  📊 cvpr_efficiency.pdf            - Figure 3: Efficiency analysis")
    print("  📄 results_table.tex              - LaTeX table code\n")
    print("💡 Use PDF files for LaTeX/Overleaf, PNG files for Word/PowerPoint\n")

if __name__ == '__main__':
    main()
