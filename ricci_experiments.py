#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ricci Flow Experiments - Per-Layer and Per-Epoch Analysis
==========================================================

Two experiments:
1. Experiment 1: Per-layer Ricci metrics at final epoch for 45 networks
   - Flat: depths 4-12, widths 16/32/64/128
   - Bottleneck: depths 4-12, width 128
   
2. Experiment 2: Per-epoch per-layer Ricci tracking for 5 networks
   - Flat: depths 5-9, width 64

Uses local_knn_fixed.py for per-layer Ricci coefficient computation.
K: 350 (fixed)
Dataset: Fashion-MNIST (Sandals vs Boots)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm

# Import local Ricci computation
from local_knn_fixed import analyze_network, format_layer_dict

# =============================================================================
# CONFIGURATION
# =============================================================================
K = 350  # Fixed k for kNN graph
EPOCHS = 50

# Experiment 2: Per-epoch tracking (5 networks)
EXP2_NETWORKS = [
    {"network_id": 1, "architecture": "flat", "depth": 5, "width": 64},
    {"network_id": 2, "architecture": "flat", "depth": 6, "width": 64},
    {"network_id": 3, "architecture": "flat", "depth": 7, "width": 64},
    {"network_id": 4, "architecture": "flat", "depth": 8, "width": 64},
    {"network_id": 5, "architecture": "flat", "depth": 9, "width": 64},
]

# Experiment 1: Full network set (45 networks)
def generate_exp1_networks() -> List[Dict]:
    """Generate 45 network configurations for Experiment 1."""
    networks = []
    network_id = 1
    
    # Flat architectures: depths 4-12, widths 16/32/64/128
    for width in [16, 32, 64, 128]:
        for depth in range(4, 13):
            networks.append({
                "network_id": network_id,
                "architecture": "flat",
                "depth": depth,
                "width": width,
            })
            network_id += 1
    
    # Bottleneck architectures: depths 4-12, width 128
    for depth in range(4, 13):
        networks.append({
            "network_id": network_id,
            "architecture": "bottleneck",
            "depth": depth,
            "width": 128,
        })
        network_id += 1
    
    return networks

EXP1_NETWORKS = generate_exp1_networks()
OUTPUT_DIR = "output_ricci_experiments"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fmnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST data (Sandals vs Boots)."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'our_data_fmnist')
    
    csv_test = os.path.join(data_path, "fashion-mnist_test.csv")
    csv_train = os.path.join(data_path, "fashion-mnist_train.csv")
    
    if not os.path.exists(csv_test) or not os.path.exists(csv_train):
        print("  CSV files not found, downloading via Keras...")
        from tensorflow.keras.datasets import fashion_mnist
        (x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()
        
        x_train_full = x_train_full.reshape(-1, 784).astype(np.float32)
        x_test_full = x_test_full.reshape(-1, 784).astype(np.float32)
        
        labels = [5, 9]
        train_idx = np.isin(y_train_full, labels)
        test_idx = np.isin(y_test_full, labels)
        
        x_train = x_train_full[train_idx]
        y_train = (y_train_full[train_idx] == 9).astype(np.int32)
        x_test = x_test_full[test_idx]
        y_test = (y_test_full[test_idx] == 9).astype(np.int32)
        
        return x_train, y_train, x_test, y_test
    
    # Load from CSV files
    x_test = pd.read_csv(csv_test)
    y_test = x_test['label']
    x_test = x_test.iloc[:, 1:]

    x_train = pd.read_csv(csv_train)
    y_train = x_train['label']
    x_train = x_train.iloc[:, 1:]

    labels = [5, 9]
    train_idx = np.concatenate([np.where(y_train == label)[0] for label in labels])
    test_idx = np.concatenate([np.where(y_test == label)[0] for label in labels])

    y_train = y_train.iloc[train_idx].values
    y_test = y_test.iloc[test_idx].values

    y_test[y_test == 5] = 0
    y_test[y_test == 9] = 1
    y_train[y_train == 5] = 0
    y_train[y_train == 9] = 1

    x_train = np.array(x_train.iloc[train_idx, :])
    x_test = np.array(x_test.iloc[test_idx, :])
    
    return x_train, y_train, x_test, y_test


# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_flat_model(depth: int, width: int, input_dim: int) -> Sequential:
    """Build flat architecture: all hidden layers have same width."""
    model = Sequential()
    model.add(Dense(units=width, activation='relu', input_shape=(input_dim,)))
    for _ in range(depth - 1):
        model.add(Dense(units=width, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model


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


def build_bottleneck_model(depth: int, width: int, input_dim: int) -> Sequential:
    """Build bottleneck architecture."""
    layer_structure = generate_bottleneck_structure(depth, outer=width, mid=width//2, center=32)
    
    model = Sequential()
    model.add(Dense(units=layer_structure[0], activation='relu', input_shape=(input_dim,)))
    for neurons in layer_structure[1:]:
        model.add(Dense(units=neurons, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model


def build_model(architecture: str, depth: int, width: int, input_dim: int) -> Sequential:
    """Build model based on architecture type."""
    if architecture == 'flat':
        return build_flat_model(depth, width, input_dim)
    elif architecture == 'bottleneck':
        return build_bottleneck_model(depth, width, input_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def format_layer_structure(architecture: str, depth: int, width: int) -> str:
    """Format layer structure as arrow-separated string."""
    if architecture == 'flat':
        return "→".join([str(width)] * depth)
    else:
        structure = generate_bottleneck_structure(depth, outer=width, mid=width//2, center=32)
        return "→".join(map(str, structure))


def extract_activations(model: Sequential, x_data: np.ndarray) -> List[np.ndarray]:
    """Extract activations from all hidden layers (excluding output)."""
    activations = []
    current_input = x_data
    for layer in model.layers[:-1]:
        current_output = layer(current_input)
        activations.append(current_output.numpy())
        current_input = current_output
    return activations


# =============================================================================
# RICCI CALLBACK FOR PER-EPOCH TRACKING (Experiment 2)
# =============================================================================

class RicciCallback(tf.keras.callbacks.Callback):
    """Callback to compute per-layer Ricci at each epoch."""
    
    def __init__(self, network_id: int, architecture: str, depth: int, width: int, 
                 x_test: np.ndarray, y_test: np.ndarray, k: int):
        super().__init__()
        self.network_id = network_id
        self.architecture = architecture
        self.depth = depth
        self.width = width
        self.x_test = x_test
        self.y_test = y_test
        self.k = k
        self.layer_structure = format_layer_structure(architecture, depth, width)
        self.epoch_results = []
        self.final_accuracy = None
    
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('val_accuracy', logs.get('accuracy', 0))
        
        # Extract activations
        activations = extract_activations(self.model, self.x_test)
        
        # Compute Ricci analysis using local_knn_fixed
        result = analyze_network(activations, self.k)
        layer_dict = format_layer_dict(result['layer_coefficients'])
        
        row = {
            'network_id': self.network_id,
            'epoch': epoch + 1,
            'architecture': self.architecture,
            'depth': self.depth,
            'width': self.width,
            'layer_structure': self.layer_structure,
            'r_all': result['r_all'],
            'p_all': np.nan,  # Not computed in simplified version
            'r_skip': result['r_skip'],
            'p_skip': np.nan,
            'accuracy': acc,
        }
        row.update(layer_dict)
        
        self.epoch_results.append(row)
        self.final_accuracy = acc


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_experiment1(x_train, y_train, x_test, y_test, networks, epochs, output_dir):
    """Run Experiment 1: Per-layer Ricci at final epoch for all networks."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Per-Layer Ricci (Final Epoch)")
    print("=" * 70)
    print(f"  Networks: {len(networks)}")
    print(f"  Epochs: {epochs}")
    
    all_rows = []
    
    for net in tqdm(networks, desc="Exp1 Networks"):
        network_id = net['network_id']
        architecture = net['architecture']
        depth = net['depth']
        width = net['width']
        
        tqdm.write(f"  {network_id}: {architecture}_{depth}_{width}")
        
        model = build_model(architecture, depth, width, x_train.shape[1])
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        
        model.fit(x_train, y_train, epochs=epochs, batch_size=32, 
                  validation_split=0.2, verbose=0)
        
        final_acc = model.evaluate(x_test, y_test, verbose=0)[1]
        activations = extract_activations(model, x_test)
        
        # Use new local_knn_fixed module
        result = analyze_network(activations, K)
        layer_dict = format_layer_dict(result['layer_coefficients'])
        
        row = {
            'network_id': network_id,
            'architecture': architecture,
            'depth': depth,
            'width': width,
            'layer_structure': format_layer_structure(architecture, depth, width),
            'r_all': result['r_all'],
            'p_all': np.nan,
            'r_skip': result['r_skip'],
            'p_skip': np.nan,
            'accuracy': final_acc,
        }
        row.update(layer_dict)
        all_rows.append(row)
        tqdm.write(f"    Accuracy: {final_acc:.4f}, r_all: {result['r_all']:.3f}")
    
    # Save results
    exp1_cols = ['network_id', 'architecture', 'depth', 'width', 'layer_structure',
                 'r_all', 'p_all', 'r_skip', 'p_skip', 'accuracy'] + [f'L{i}' for i in range(1, 13)]
    exp1_df = pd.DataFrame(all_rows)[exp1_cols]
    exp1_path = os.path.join(output_dir, "exp1_final_epoch.csv")
    exp1_df.to_csv(exp1_path, index=False)
    
    print(f"\n  Saved: {exp1_path} ({len(exp1_df)} rows)")
    return exp1_df


def run_experiment2(x_train, y_train, x_test, y_test, networks, epochs, output_dir):
    """Run Experiment 2: Per-epoch per-layer Ricci tracking."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Per-Epoch Per-Layer Ricci")
    print("=" * 70)
    print(f"  Networks: {len(networks)}")
    print(f"  Epochs: {epochs}")
    print(f"  Expected rows: {len(networks) * epochs}")
    
    all_rows = []
    
    for net in tqdm(networks, desc="Exp2 Networks"):
        network_id = net['network_id']
        architecture = net['architecture']
        depth = net['depth']
        width = net['width']
        
        tqdm.write(f"  {network_id}: {architecture}_{depth}_{width}")
        
        model = build_model(architecture, depth, width, x_train.shape[1])
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        
        ricci_cb = RicciCallback(
            network_id=network_id,
            architecture=architecture,
            depth=depth,
            width=width,
            x_test=x_test,
            y_test=y_test,
            k=K
        )
        
        model.fit(x_train, y_train, epochs=epochs, batch_size=32,
                  validation_split=0.2, callbacks=[ricci_cb], verbose=0)
        
        final_acc = model.evaluate(x_test, y_test, verbose=0)[1]
        tqdm.write(f"    Final accuracy: {final_acc:.4f}")
        
        # Update final accuracy
        for row in ricci_cb.epoch_results:
            if row['epoch'] == epochs:
                row['accuracy'] = final_acc
        
        all_rows.extend(ricci_cb.epoch_results)
    
    # Save results
    exp2_cols = ['network_id', 'epoch', 'architecture', 'depth', 'width', 'layer_structure',
                 'r_all', 'p_all', 'r_skip', 'p_skip', 'accuracy'] + [f'L{i}' for i in range(1, 13)]
    exp2_df = pd.DataFrame(all_rows)[exp2_cols]
    exp2_path = os.path.join(output_dir, "exp2_per_epoch.csv")
    exp2_df.to_csv(exp2_path, index=False)
    
    print(f"\n  Saved: {exp2_path} ({len(exp2_df)} rows)")
    return exp2_df


def generate_plots(exp1_df: pd.DataFrame, exp2_df: pd.DataFrame, plots_dir: str):
    """Generate visualization plots."""
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Layer-Ricci vs Layer (from Exp2 final epoch)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    final_epoch = exp2_df['epoch'].max()
    final_epoch_data = exp2_df[exp2_df['epoch'] == final_epoch]
    
    for idx, (_, row) in enumerate(final_epoch_data.iterrows()):
        depth = int(row['depth'])
        ricci_values = [row[f'L{i}'] for i in range(1, depth)]  # L-1 values
        ricci_values = [v for v in ricci_values if not np.isnan(v)]
        layers = list(range(len(ricci_values)))
        ax1.plot(layers, ricci_values, 'o-', label=f"Depth {depth}", 
                 color=colors[idx % 10], markersize=5, linewidth=2)
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Layer-Ricci coef.', fontsize=12)
    ax1.set_title('Per-Layer Ricci Coefficient', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(plots_dir, "layer_ricci_vs_layer.png"), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Accuracy vs Depth
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    depths = final_epoch_data['depth'].values
    accuracies = final_epoch_data['accuracy'].values * 100
    ax2.plot(depths, accuracies, 'o-', color='tab:blue', markersize=8, linewidth=2)
    ax2.set_xlabel('Depth', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Network Depth', fontsize=14)
    ax2.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "accuracy_vs_depth.png"), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"  Plots saved to {plots_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def run_experiments(test_mode: bool = False, exp1_only: bool = False, exp2_only: bool = False):
    """Run experiments."""
    print("=" * 80)
    print("RICCI FLOW EXPERIMENTS")
    print("=" * 80)
    
    epochs = 2 if test_mode else EPOCHS
    exp1_networks = EXP1_NETWORKS[:3] if test_mode else EXP1_NETWORKS
    exp2_networks = EXP2_NETWORKS[:2] if test_mode else EXP2_NETWORKS
    
    print(f"\nConfiguration:")
    print(f"  K = {K}")
    print(f"  Epochs = {epochs}")
    print(f"  Exp1 Networks = {len(exp1_networks)}")
    print(f"  Exp2 Networks = {len(exp2_networks)}")
    
    print("\nLoading Fashion-MNIST data...")
    x_train, y_train, x_test, y_test = load_fmnist_data()
    print(f"  Training: {x_train.shape}, Test: {x_test.shape}")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, OUTPUT_DIR)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    exp1_df = None
    exp2_df = None
    
    if not exp2_only:
        exp1_df = run_experiment1(x_train, y_train, x_test, y_test, 
                                   exp1_networks, epochs, output_dir)
    
    if not exp1_only:
        exp2_df = run_experiment2(x_train, y_train, x_test, y_test, 
                                   exp2_networks, epochs, output_dir)
    
    if exp1_df is not None and exp2_df is not None:
        print("\nGenerating plots...")
        generate_plots(exp1_df, exp2_df, plots_dir)
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ricci Flow Experiments')
    parser.add_argument('--test', action='store_true', help='Test mode: 2 epochs, fewer networks')
    parser.add_argument('--exp1-only', action='store_true', help='Run only Experiment 1')
    parser.add_argument('--exp2-only', action='store_true', help='Run only Experiment 2')
    args = parser.parse_args()
    
    run_experiments(test_mode=args.test, exp1_only=args.exp1_only, exp2_only=args.exp2_only)
