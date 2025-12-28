#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Academic-style plots for CVPR/ICCV papers.

Following guidelines:
- Clean, minimal design
- Black/white friendly (grayscale printable)
- Times New Roman or Computer Modern fonts
- No unnecessary decorations
- Proper sizing for two-column format
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# === ACADEMIC STYLE CONFIGURATION ===
# Based on CVPR/ICCV submission guidelines

plt.style.use('seaborn-v0_8-whitegrid')

mpl.rcParams.update({
    # Font settings (Times-like for academic papers)
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    
    # Axes
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'axes.spines.top': False,
    'axes.spines.right': False,
    
    # Ticks
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    
    # Legend
    'legend.fontsize': 8,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    
    # Figure
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    
    # Grid
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.linestyle': ':',
})

# Grayscale-friendly markers and colors
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
HATCHES = ['', '//', '\\\\', 'xx', '..', '++']


def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def fig1_accuracy_speed_tradeoff(data, output_dir):
    """
    Figure 1: Accuracy vs Speed trade-off (scatter plot)
    Standard visualization in tracking papers
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Single column width
    
    variants = list(data.keys())
    
    for i, v in enumerate(variants):
        mota = data[v]['mota'] * 100
        fps = data[v]['mean_fps']
        
        ax.scatter(fps, mota, 
                   marker=MARKERS[i % len(MARKERS)],
                   s=60,
                   c='white',
                   edgecolors='black',
                   linewidths=1.2,
                   zorder=3,
                   label=v.replace('_', ' ').title())
    
    ax.set_xlabel('Speed (FPS)')
    ax.set_ylabel('MOTA (%)')
    
    # Clean legend
    ax.legend(loc='lower right', ncol=1, handletextpad=0.3, 
              columnspacing=0.5, borderpad=0.3)
    
    plt.tight_layout()
    
    # Save both formats
    fig.savefig(output_dir / 'fig1_speed_accuracy.pdf')
    fig.savefig(output_dir / 'fig1_speed_accuracy.png')
    print(f"✅ Figure 1: {output_dir / 'fig1_speed_accuracy.pdf'}")
    plt.close()


def fig2_metrics_bars(data, output_dir):
    """
    Figure 2: Bar chart comparison of key metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2))
    
    variants = list(data.keys())
    x = np.arange(len(variants))
    width = 0.6
    
    # (a) MOTA comparison
    ax = axes[0]
    mota = [data[v]['mota'] * 100 for v in variants]
    
    bars = ax.bar(x, mota, width, color='white', edgecolor='black', linewidth=1)
    for i, bar in enumerate(bars):
        bar.set_hatch(HATCHES[i % len(HATCHES)])
    
    ax.set_ylabel('MOTA (%)')
    ax.set_xlabel('(a) Single-Target Tracking Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=7)
    ax.set_ylim([0, 100])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mota)):
        ax.text(bar.get_x() + bar.get_width()/2., val + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    
    # (b) FPS comparison  
    ax = axes[1]
    fps = [data[v]['mean_fps'] for v in variants]
    
    bars = ax.bar(x, fps, width, color='white', edgecolor='black', linewidth=1)
    for i, bar in enumerate(bars):
        bar.set_hatch(HATCHES[i % len(HATCHES)])
    
    ax.set_ylabel('Speed (FPS)')
    ax.set_xlabel('(b) Processing Speed')
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=7)
    ax.set_ylim([0, max(fps) * 1.15])
    
    for i, (bar, val) in enumerate(zip(bars, fps)):
        ax.text(bar.get_x() + bar.get_width()/2., val + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'fig2_metrics.pdf')
    fig.savefig(output_dir / 'fig2_metrics.png')
    print(f"✅ Figure 2: {output_dir / 'fig2_metrics.pdf'}")
    plt.close()


def table1_latex(data, output_dir):
    """
    Table 1: Quantitative results (LaTeX format)
    """
    latex = r"""\begin{table}[t]
\centering
\caption{Quantitative comparison of tracking variants. Best results in \textbf{bold}.}
\label{tab:results}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Method & MOTA$\uparrow$ & IDF1$\uparrow$ & Lock\%$\uparrow$ & FPS$\uparrow$ & Lat.(ms)$\downarrow$ \\
\midrule
"""
    
    # Find best values
    best_mota = max(data.values(), key=lambda x: x['mota'])['mota']
    best_idf1 = max(data.values(), key=lambda x: x['idf1'])['idf1']
    best_lock = max(data.values(), key=lambda x: x['target_lock_rate'])['target_lock_rate']
    best_fps = max(data.values(), key=lambda x: x['mean_fps'])['mean_fps']
    best_lat = min(data.values(), key=lambda x: x['p95_latency_ms'])['p95_latency_ms']
    
    for v, m in data.items():
        name = v.replace('_', ' ').title()
        
        # Format with bold for best values
        mota = f"\\textbf{{{m['mota']*100:.1f}}}" if abs(m['mota'] - best_mota) < 0.001 else f"{m['mota']*100:.1f}"
        idf1 = f"\\textbf{{{m['idf1']*100:.1f}}}" if abs(m['idf1'] - best_idf1) < 0.001 else f"{m['idf1']*100:.1f}"
        lock = f"\\textbf{{{m['target_lock_rate']*100:.1f}}}" if abs(m['target_lock_rate'] - best_lock) < 0.001 else f"{m['target_lock_rate']*100:.1f}"
        fps = f"\\textbf{{{m['mean_fps']:.1f}}}" if abs(m['mean_fps'] - best_fps) < 0.1 else f"{m['mean_fps']:.1f}"
        lat = f"\\textbf{{{m['p95_latency_ms']:.1f}}}" if abs(m['p95_latency_ms'] - best_lat) < 0.1 else f"{m['p95_latency_ms']:.1f}"
        
        latex += f"{name} & {mota} & {idf1} & {lock} & {fps} & {lat} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'table1.tex', 'w') as f:
        f.write(latex)
    
    print(f"✅ Table 1: {output_dir / 'table1.tex'}")
    print("\n" + "="*60)
    print("LaTeX Code:")
    print("="*60)
    print(latex)


def fig3_ablation(data, output_dir):
    """
    Figure 3: Ablation study - effect of each component
    """
    # Define ablation order (removing components from full)
    ablation_order = ['full_features', 'shape_depth', 'shape_only', 'hsv_depth', 'iou_only']
    ablation_labels = ['Full', '-HSV', '-Depth', '-Shape', 'IoU only']
    
    # Filter to available variants
    available = [(v, l) for v, l in zip(ablation_order, ablation_labels) if v in data]
    if len(available) < 2:
        print("⚠️ Not enough variants for ablation study")
        return
    
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    
    x = np.arange(len(available))
    mota = [data[v]['mota'] * 100 for v, _ in available]
    labels = [l for _, l in available]
    
    bars = ax.bar(x, mota, 0.6, color='white', edgecolor='black', linewidth=1)
    
    ax.set_ylabel('MOTA (%)')
    ax.set_xlabel('Feature Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim([0, 100])
    
    # Reference line at full features
    if 'full_features' in data:
        ax.axhline(y=data['full_features']['mota']*100, 
                   color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    
    for bar, val in zip(bars, mota):
        ax.text(bar.get_x() + bar.get_width()/2., val + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'fig3_ablation.pdf')
    fig.savefig(output_dir / 'fig3_ablation.png')
    print(f"✅ Figure 3: {output_dir / 'fig3_ablation.pdf'}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_academic.py <comparison.json>")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)
    
    data = load_data(filepath)
    output_dir = filepath.parent / 'academic'
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("📊 GENERATING ACADEMIC-STYLE FIGURES")
    print("="*60)
    print(f"Input:  {filepath}")
    print(f"Output: {output_dir}/")
    print()
    
    # Generate all figures
    fig1_accuracy_speed_tradeoff(data, output_dir)
    fig2_metrics_bars(data, output_dir)
    fig3_ablation(data, output_dir)
    table1_latex(data, output_dir)
    
    print()
    print("="*60)
    print("✅ DONE - Academic figures generated")
    print("="*60)
    print()
    print("Generated files:")
    print("  📊 fig1_speed_accuracy.pdf  - Speed vs Accuracy scatter")
    print("  📊 fig2_metrics.pdf          - Bar chart comparison")
    print("  📊 fig3_ablation.pdf         - Ablation study")
    print("  📄 table1.tex                - LaTeX table")
    print()
    print("Style: Grayscale-friendly, Times font, minimal design")


if __name__ == '__main__':
    main()
