#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1: Per-Layer Ricci Coefficients at Final Epoch
==========================================================

Analyzes 45 pre-trained networks using activations from output_k_sweep/
Computes per-layer Ricci coefficients using LOCAL Ricci calculations
for ALL curvature types: Forman-Ricci, Augmented-Forman-Ricci,
Approx-Ollivier-Ricci, Ollivier-Ricci.

Networks:
- Flat: depths 4-12, widths 16/32/64/128 (36 networks)
- Bottleneck: depths 4-12, width 128 (9 networks)

K: 350 (fixed)
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict
import argparse
from tqdm import tqdm

# Import local Ricci computation (uses per-data-point calculations)
from local_knn_fixed import (
    compute_layer_ricci_coefficients,
    format_layer_dict,
    CURVATURE_TYPES,
    CURVATURE_PREFIXES,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
K = 350  # Fixed k for kNN graph
INPUT_DIR = "output_k_sweep"
OUTPUT_DIR = "output_experiment_1"


def parse_folder_name(folder_name: str) -> Dict:
    """
    Parse network folder name to extract architecture info.
    
    Format: {architecture}_{depth}_{width}
    Example: flat_5_64 -> {'architecture': 'flat', 'depth': 5, 'width': 64}
    """
    parts = folder_name.split('_')
    if len(parts) < 3:
        return None
    
    try:
        architecture = parts[0]
        depth = int(parts[1])
        width = int(parts[2])
        return {
            'architecture': architecture,
            'depth': depth,
            'width': width
        }
    except ValueError:
        return None


def format_layer_structure(architecture: str, depth: int, width: int) -> str:
    """Format layer structure as arrow-separated string."""
    if architecture == 'flat':
        return "→".join([str(width)] * depth)
    else:
        # Bottleneck structure
        structure = generate_bottleneck_structure(depth, outer=width, mid=width//2, center=32)
        return "→".join(map(str, structure))


def generate_bottleneck_structure(num_layers: int, outer: int = 128, mid: int = 64, center: int = 32) -> List[int]:
    """Generate bottleneck (hourglass) layer structure."""
    if num_layers < 3:
        return [outer] * num_layers
    
    is_even = (num_layers % 2 == 0)
    center_count = 2 if is_even else 1
    remaining = num_layers - center_count
    half = remaining // 2
    
    if half == 0:
        left_half = []
    elif half == 1:
        left_half = [outer]
    elif half == 2:
        left_half = [outer, mid]
    else:
        outer_count = half - 1
        left_half = [outer] * outer_count + [mid]
    
    structure = left_half + [center] * center_count + left_half[::-1]
    return structure[:num_layers]


def run_experiment1(input_dir: str, output_dir: str, k: int, test_mode: bool = False):
    """
    Run Experiment 1: Per-layer Ricci at final epoch for all curvature types.
    
    Reads activations from input_dir, computes Ricci using local_knn_fixed
    for all 4 curvature types, outputs CSV to output_dir.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Per-Layer Ricci Coefficients (Final Epoch)")
    print("  All Curvature Types: FR, AFR, AOR, OR")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  K = {k}")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Curvature types: {', '.join(CURVATURE_TYPES)}")
    
    # Find all network folders
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, input_dir)
    
    if not os.path.exists(input_path):
        print(f"\nERROR: Input directory not found: {input_path}")
        return
    
    network_folders = sorted([
        f for f in os.listdir(input_path) 
        if os.path.isdir(os.path.join(input_path, f)) and parse_folder_name(f) is not None
    ])
    
    if test_mode:
        network_folders = network_folders[:3]
    
    print(f"  Networks found: {len(network_folders)}")
    
    # Create output directory
    output_path = os.path.join(base_path, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Process each network
    all_rows = []
    
    print(f"\nProcessing networks...")
    for idx, folder in enumerate(tqdm(network_folders, desc="Networks")):
        network_info = parse_folder_name(folder)
        if network_info is None:
            continue
        
        folder_path = os.path.join(input_path, folder)
        activations_path = os.path.join(folder_path, "activations.npy")
        accuracy_path = os.path.join(folder_path, "accuracy.npy")
        
        if not os.path.exists(activations_path):
            tqdm.write(f"  Skipping {folder}: activations.npy not found")
            continue
        
        # Load activations and accuracy
        try:
            activations = np.load(activations_path, allow_pickle=True)
            accuracy = np.load(accuracy_path)[0] if os.path.exists(accuracy_path) else np.nan
        except Exception as e:
            tqdm.write(f"  Error loading {folder}: {e}")
            continue
        
        # Convert to list of arrays
        activations_list = list(activations)
        
        # Build row with base info
        row = {
            'network_id': idx + 1,
            'architecture': network_info['architecture'],
            'depth': network_info['depth'],
            'width': network_info['width'],
            'layer_structure': format_layer_structure(
                network_info['architecture'], 
                network_info['depth'], 
                network_info['width']
            ),
            'accuracy': accuracy,
        }
        
        # Compute layer Ricci coefficients for each curvature type
        for curv_type in CURVATURE_TYPES:
            prefix = CURVATURE_PREFIXES[curv_type]
            try:
                layer_coefs = compute_layer_ricci_coefficients(activations_list, k, curv=curv_type)
                layer_dict = format_layer_dict(layer_coefs, prefix=prefix)
            except Exception as e:
                tqdm.write(f"  Warning: {curv_type} failed for {folder}: {e}")
                # Fill with NaN
                layer_dict = {f'{prefix}_L{i}': np.nan for i in range(1, 13)}
            row.update(layer_dict)
        
        all_rows.append(row)
        tqdm.write(f"  {folder}: acc={accuracy:.3f}")
    
    # Create DataFrame and save
    if not all_rows:
        print("\nNo networks processed successfully!")
        return
    
    # Build column order: base cols + prefixed layer cols for each curvature type
    base_cols = ['network_id', 'architecture', 'depth', 'width', 'layer_structure', 'accuracy']
    layer_cols = []
    for curv_type in CURVATURE_TYPES:
        prefix = CURVATURE_PREFIXES[curv_type]
        layer_cols += [f'{prefix}_L{i}' for i in range(1, 13)]
    
    all_cols = base_cols + layer_cols
    
    # Only include columns that exist in the data
    existing_cols = [c for c in all_cols if c in all_rows[0]]
    
    df = pd.DataFrame(all_rows)[existing_cols]
    csv_path = os.path.join(output_path, "exp1_final_epoch.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n" + "=" * 70)
    print("EXPERIMENT 1 COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: {csv_path}")
    print(f"Networks processed: {len(df)}")
    print(f"Columns: {len(existing_cols)} ({len(base_cols)} base + {len(existing_cols) - len(base_cols)} layer)")
    print(f"\nSample results:")
    print(df[['architecture', 'depth', 'width', 'accuracy']].head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment 1: Per-Layer Ricci Coefficients')
    parser.add_argument('--test', action='store_true', help='Test mode: process only 3 networks')
    parser.add_argument('--input', type=str, default=INPUT_DIR, help='Input directory with activations')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output directory for CSV')
    parser.add_argument('--k', type=int, default=K, help='K value for kNN graph')
    args = parser.parse_args()
    
    run_experiment1(
        input_dir=args.input, 
        output_dir=args.output, 
        k=args.k, 
        test_mode=args.test
    )
