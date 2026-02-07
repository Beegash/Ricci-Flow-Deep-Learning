#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1 Plotting Script
=============================

Creates plots from exp1_final_epoch.csv matching reference style:
1. Layer-Ricci coefficient vs Layer (smooth lines, no markers)
2. Test Accuracy vs Depth
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_CSV = "output_experiment_1/exp1_final_epoch.csv"
OUTPUT_DIR = "output_experiment_1/plots"


def create_reference_plot(df: pd.DataFrame, output_dir: str, 
                          architecture: str = 'flat', width: int = 64,
                          depths_to_plot: list = None):
    """
    Create plot matching the reference image style exactly.
    
    Left panel: Layer-Ricci coef vs Layer (lines only, no markers)
    Right panel: Test Accuracy vs Depth
    """
    # Filter by architecture and width
    subset = df[(df['architecture'] == architecture) & (df['width'] == width)]
    
    if subset.empty:
        print(f"No data for {architecture} width={width}")
        return
    
    # Filter to specific depths if provided
    if depths_to_plot:
        subset = subset[subset['depth'].isin(depths_to_plot)]
    
    depths = sorted(subset['depth'].unique())
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # === LEFT PANEL: Layer-Ricci vs Layer ===
    # Use default matplotlib color cycle (matches reference)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for idx, depth in enumerate(depths):
        row = subset[subset['depth'] == depth].iloc[0]
        
        # Get layer coefficients
        layer_vals = []
        for i in range(1, 13):
            val = row[f'L{i}']
            if pd.notna(val):
                layer_vals.append(val)
        
        if layer_vals:
            layers = list(range(len(layer_vals)))
            # Lines only, no markers (like reference)
            ax1.plot(layers, layer_vals, '-', color=colors[idx % len(colors)], 
                     label=f'Depth {depth}', linewidth=1.5)
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Layer-Ricci coef.', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.set_xlim(left=0)
    
    # === RIGHT PANEL: Accuracy vs Depth ===
    subset_sorted = subset.sort_values('depth')
    accuracies = subset_sorted['accuracy'].values * 100
    
    ax2.plot(subset_sorted['depth'], accuracies, '-', 
             color='tab:blue', linewidth=1.5)
    ax2.set_xlabel('Depth', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    filename = f"layer_ricci_{architecture}_width{width}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


def main(input_csv: str, output_dir: str):
    """Generate plots."""
    print("=" * 60)
    print("Experiment 1 Plotting")
    print("=" * 60)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, input_csv)
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded: {csv_path}")
    print(f"Networks: {len(df)}")
    
    output_path = os.path.join(base_path, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\nGenerating plots...")
    
    # Generate plots for each width (flat architecture)
    for width in [16, 32, 64, 128]:
        create_reference_plot(df, output_path, architecture='flat', width=width)
    
    # Bottleneck
    create_reference_plot(df, output_path, architecture='bottleneck', width=128)
    
    print(f"\n" + "=" * 60)
    print("Plots Complete!")
    print("=" * 60)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment 1 Plotting')
    parser.add_argument('--input', type=str, default=INPUT_CSV, help='Input CSV')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output dir')
    args = parser.parse_args()
    
    main(input_csv=args.input, output_dir=args.output)
